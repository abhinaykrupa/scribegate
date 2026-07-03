"""app/common.py — shared helpers for the ScribeGate v0.2 multi-page Streamlit
app (app/streamlit_app.py + app/views/*.py).

Moved, unchanged in logic, from v0.1's single-file app/streamlit_app.py:
path constants, result/transcript/golden-note loaders, the span-highlighting
core (build_highlighted_transcript_html / extract_span_texts — copied
character-identical, not refactored), the shared note+transcript renderer
used by both the Provenance view and the Live Encounter capture pipeline, and
HIGHLIGHT_CSS.

render_banner() is the one intentional behavior change from v0.1: it now
loads specs/consent_copy.yaml (banner.text) instead of a hardcoded string, so
the demo disclaimer has a single source of truth shared with the consent
gate.

Zero API keys, zero network: everything is read from bundled data/ and
specs/ files. stdlib + streamlit + pandas + pyyaml only.
"""

from __future__ import annotations

import glob
import html
import json
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(APP_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
# RESULTS_DIR is env-overridable (SCRIBEGATE_RESULTS_DIR) so the app can point
# at an alternate results directory (e.g. per-environment or test fixtures)
# without a code change; defaults to the bundled data/results/.
RESULTS_DIR = os.environ.get("SCRIBEGATE_RESULTS_DIR") or os.path.join(DATA_DIR, "results")
TRANSCRIPTS_DIR = os.path.join(DATA_DIR, "transcripts")
GOLDEN_DIR = os.path.join(DATA_DIR, "golden_notes")
BENCHMARK_MD = os.path.join(RESULTS_DIR, "benchmark.md")
DECISION_LOG = os.path.join(RESULTS_DIR, "decision_log.jsonl")
HISTORY_PATH = os.path.join(RESULTS_DIR, "history.jsonl")
HISTORY_DEMO_PATH = os.path.join(RESULTS_DIR, "history_demo.jsonl")
LIVE_RESULTS_DIR = os.path.join(RESULTS_DIR, "live")

SPECS_DIR = os.path.join(REPO_ROOT, "specs")
CONSENT_COPY_PATH = os.path.join(SPECS_DIR, "consent_copy.yaml")

SECTION_ORDER = ["S", "O", "A", "P"]
SECTION_LABELS = {
    "S": "Subjective",
    "O": "Objective",
    "A": "Assessment",
    "P": "Plan",
}


# ---------------------------------------------------------------------------
# Data loading (pure functions — no Streamlit calls, so they're testable and
# cacheable independently of the UI)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=5)
def load_results() -> dict:
    """Load every data/results/{id}.json into {transcript_id: result_dict}.

    Cached for 5s (ttl) so newly-written results/corrections show up in the
    running app without a full restart, while still avoiding a disk re-scan
    on every single widget interaction/rerun."""
    results = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        tid = data.get("transcript_id") or os.path.splitext(os.path.basename(path))[0]
        results[tid] = data
    return results


@st.cache_data(ttl=5)
def load_transcript_text(transcript_id: str) -> str | None:
    path = os.path.join(TRANSCRIPTS_DIR, f"{transcript_id}.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


@st.cache_data(ttl=5)
def load_golden_note(transcript_id: str) -> dict | None:
    path = os.path.join(GOLDEN_DIR, f"{transcript_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(ttl=5)
def load_benchmark_md() -> str | None:
    if not os.path.exists(BENCHMARK_MD):
        return None
    with open(BENCHMARK_MD, "r", encoding="utf-8") as fh:
        return fh.read()


def append_decision(transcript_id: str, decision: str, reviewer: str = "demo-user") -> dict:
    """Append a reviewer decision line to data/results/decision_log.jsonl and
    return the record written. Pure I/O, no Streamlit state touched here."""
    record = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "transcript_id": transcript_id,
        "reviewer": reviewer,
        "decision": decision,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # decision_log.jsonl is multi-producer / multi-schema: the CLI (`python -m
    # scribegate.cli run`) appends route events (ts, transcript_id, aggregate,
    # route, violation_count) and this reviewer flow appends decision events
    # (ts, transcript_id, reviewer, decision) — same file, two record shapes.
    with open(DECISION_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


def append_consent_event(state_code: str, provider_attested: bool, patient_attested: bool) -> dict:
    """Append a consent event to data/results/decision_log.jsonl and return
    the record written. Same file/pattern as append_decision — decision_log
    .jsonl is documented as multi-producer/multi-schema, so this is just
    another record shape sharing the same append-only log."""
    record = {
        "event": "consent",
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "state": state_code,
        "provider_attested": provider_attested,
        "patient_attested": patient_attested,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(DECISION_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


# ---------------------------------------------------------------------------
# Span-highlighting core logic (the load-bearing, testable piece)
# ---------------------------------------------------------------------------

def build_highlighted_transcript_html(transcript_text: str, spans: list[tuple[int, int]]) -> str:
    """Return transcript_text as HTML with the given char-offset spans wrapped
    in <mark>. Escaping happens PER SEGMENT (before any tag insertion) so that
    offsets computed against the raw (unescaped) string stay valid — we never
    escape the whole string first and then index into it.

    spans: list of (start, end) character offsets, end-exclusive, indexing
    into transcript_text directly (Python string / char indexing — correct
    per spec, no byte re-encoding).
    """
    n = len(transcript_text)
    # Clip, drop invalid/empty, then sort + merge overlapping ranges so
    # <mark> tags never nest or double-highlight a character.
    cleaned: list[tuple[int, int]] = []
    for start, end in spans or []:
        start = max(0, min(int(start), n))
        end = max(0, min(int(end), n))
        if end > start:
            cleaned.append((start, end))
    cleaned.sort()

    merged: list[tuple[int, int]] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    pieces: list[str] = []
    cursor = 0
    for start, end in merged:
        pieces.append(html.escape(transcript_text[cursor:start]))
        pieces.append('<mark class="sg-highlight">')
        pieces.append(html.escape(transcript_text[start:end]))
        pieces.append("</mark>")
        cursor = end
    pieces.append(html.escape(transcript_text[cursor:n]))

    body = "".join(pieces)
    # Preserve newlines as <br> since we render inside a <div> via unsafe HTML.
    body = body.replace("\n", "<br>")
    return body


def extract_span_texts(transcript_text: str, spans: list[tuple[int, int]]) -> list[str]:
    """Return the literal transcript substring for each span, in order,
    clipped to valid bounds. Used both for the "supported by: ..." caption
    and for char-exactness verification."""
    n = len(transcript_text)
    out = []
    for start, end in spans or []:
        start = max(0, min(int(start), n))
        end = max(0, min(int(end), n))
        if end > start:
            out.append(transcript_text[start:end])
    return out


# ---------------------------------------------------------------------------
# Shared UI chrome
# ---------------------------------------------------------------------------

@st.cache_data(ttl=5)
def load_consent_copy() -> dict:
    """Load and parse specs/consent_copy.yaml (the single source of truth
    for every string the consent gate + capture UI render)."""
    with open(CONSENT_COPY_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def render_banner() -> None:
    data = load_consent_copy()
    banner_text = ((data or {}).get("banner") or {}).get(
        "text",
        "100% SYNTHETIC data — no PHI. All transcripts, notes, and scores on "
        "this page are generated fixtures for demo purposes only.",
    )
    st.markdown(
        f"""
        <div style="background-color:#1f3d2b;border:1px solid #3c7a52;
                    border-radius:6px;padding:0.5rem 1rem;margin-bottom:1rem;
                    color:#c9f2d6;font-size:0.9rem;">
            <strong>100% SYNTHETIC data — no PHI.</strong>
            {banner_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_wordmark() -> None:
    """Render a simple inline SVG "ScribeGate" wordmark + a small abstract
    gate/gateway glyph (two vertical bars + a horizontal bar) in the sidebar,
    in the deep-teal theme color."""
    svg = """
    <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;">
      <svg width="34" height="34" viewBox="0 0 34 34" xmlns="http://www.w3.org/2000/svg"
           role="img" aria-label="ScribeGate gate glyph">
        <rect x="6" y="6" width="4" height="24" rx="1" fill="#0F5C5C"/>
        <rect x="24" y="6" width="4" height="24" rx="1" fill="#0F5C5C"/>
        <rect x="6" y="14" width="22" height="4" rx="1" fill="#0F5C5C"/>
      </svg>
      <span style="font-size:1.3rem;font-weight:700;color:#0F5C5C;letter-spacing:0.02em;">
        ScribeGate
      </span>
    </div>
    """
    st.sidebar.markdown(svg, unsafe_allow_html=True)


HIGHLIGHT_CSS = """
<style>
mark.sg-highlight {
    background-color: #ffe066;
    color: #111;
    padding: 0 2px;
    border-radius: 2px;
}
mark.sg-highlight-alt {
    background-color: #74c0fc;
    color: #111;
    padding: 0 2px;
    border-radius: 2px;
}
.sg-transcript-box {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 0.85rem;
    line-height: 1.5;
    white-space: normal;
    max-height: 640px;
    overflow-y: auto;
    padding: 0.75rem;
    border: 1px solid #444;
    border-radius: 6px;
}
</style>
"""


# ---------------------------------------------------------------------------
# Shared note + transcript renderer (Provenance view + Live Encounter Step 3)
# ---------------------------------------------------------------------------

def _note_line_key(prefix: str, section: str, idx: int) -> str:
    return f"{prefix}_{section}_{idx}"


def _render_note_and_transcript(
    transcript_id: str,
    transcript_text: str,
    note: dict,
    state_prefix: str,
) -> None:
    """Render the side-by-side SOAP-note / transcript panel for a given note
    (generated or golden). Selected line is tracked in st.session_state under
    a key namespaced by state_prefix so the two toggled views don't clobber
    each other's selection."""
    selection_key = f"{state_prefix}_selected_{transcript_id}"
    if selection_key not in st.session_state:
        st.session_state[selection_key] = None  # (section, idx) tuple

    soap = note.get("soap", {})

    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### SOAP note")
        for section in SECTION_ORDER:
            lines = soap.get(section, [])
            if not lines:
                continue
            st.markdown(f"**{section} — {SECTION_LABELS[section]}**")
            for idx, line in enumerate(lines):
                text = line.get("text", "")
                spans = line.get("spans", [])
                key = _note_line_key(state_prefix, section, idx) + f"_{transcript_id}"
                selected = st.session_state[selection_key] == (section, idx)
                label = ("▶ " if selected else "") + text
                if st.button(label, key=key, use_container_width=True):
                    st.session_state[selection_key] = (section, idx)
                    selected = True
                if selected:
                    span_texts = extract_span_texts(transcript_text, spans)
                    if span_texts:
                        quoted = " / ".join(f"“{t}”" for t in span_texts)
                        st.caption(f"supported by: {quoted}")
                    else:
                        st.caption("supported by: (no transcript span recorded)")

    with right:
        st.markdown("#### Raw transcript")
        st.markdown(HIGHLIGHT_CSS, unsafe_allow_html=True)
        sel = st.session_state[selection_key]
        spans: list[tuple[int, int]] = []
        if sel is not None:
            section, idx = sel
            lines = soap.get(section, [])
            if idx < len(lines):
                spans = [tuple(s) for s in lines[idx].get("spans", [])]
        highlighted = build_highlighted_transcript_html(transcript_text, spans)
        st.markdown(
            f'<div class="sg-transcript-box">{highlighted}</div>',
            unsafe_allow_html=True,
        )
        if not sel:
            st.caption("Click a note line on the left to highlight its supporting span(s).")
