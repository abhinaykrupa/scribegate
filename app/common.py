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
UI_COPY_PATH = os.path.join(SPECS_DIR, "ui_copy.yaml")

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


# ---------------------------------------------------------------------------
# Cold-start self-seeding (Streamlit Cloud: data/results/*.json is gitignored,
# so a fresh deploy starts with an empty results dir and every view renders
# its empty state; this makes the app generate its own demo results in-
# process on first load, rather than requiring a human to run the CLI).
# ---------------------------------------------------------------------------

# Module-level cache of results dirs already confirmed fully-seeded THIS
# PROCESS, keyed by the resolved results_dir string (never a bare no-arg
# flag) so it can never suppress the cheap filesystem check for a different
# results_dir (e.g. two separate tmp_path dirs used by two tests in the same
# pytest process stay fully isolated).
_SEEDED_DIRS: set[str] = set()


def _is_fully_seeded(results_dir: str) -> bool:
    """Cheap correctness check: does results_dir contain a result JSON for
    every transcript id scribegate.cli knows about? glob/exists only — never
    touches mtimes of existing files."""
    if not os.path.isdir(results_dir):
        return False
    from scribegate import cli as scribegate_cli

    transcript_ids = scribegate_cli.discover_transcript_ids()
    if not transcript_ids:
        # No bundled transcripts to seed from at all — nothing to do, treat
        # as "seeded" (no-op) rather than looping forever.
        return True
    return all(
        os.path.exists(os.path.join(results_dir, f"{tid}.json"))
        for tid in transcript_ids
    )


def _seed_results(results_dir: str) -> None:
    """Actually run the pipeline for every bundled transcript and rebuild
    benchmark.md, writing into results_dir. No return value — callers check
    the filesystem afterward via _is_fully_seeded."""
    from pathlib import Path

    from scribegate import benchmark as scribegate_benchmark
    from scribegate import cli as scribegate_cli

    results_dir_path = Path(results_dir)
    scribegate_cli.run_all(results_dir=results_dir_path)
    scribegate_benchmark.main(["--results-dir", str(results_dir_path)])


def ensure_results(results_dir: str | None = None) -> bool:
    """Self-seed data/results on cold start so the app never shows an empty
    dashboard just because Streamlit Cloud's gitignored data/results/ hasn't
    been populated yet (a fresh deploy starts with no result JSONs at all).

    Resolves the effective results dir at CALL TIME (not solely from the
    module-level RESULTS_DIR constant) so callers/tests can override it
    explicitly, and so an env var change between calls is honored:
        results_dir = results_dir or os.environ.get("SCRIBEGATE_RESULTS_DIR") or RESULTS_DIR

    Returns True if this call actually seeded (ran the pipeline), False if
    it was a no-op because results_dir already has a result JSON for every
    bundled transcript id.

    "Run at most once per dir" guard: the cheap filesystem check
    (_is_fully_seeded) always runs first and is the sole source of truth for
    correctness — it's what lets this function be called on every Streamlit
    rerun without re-seeding. A module-level `_SEEDED_DIRS` set, keyed by the
    resolved results_dir path, is consulted before even that cheap check
    purely as a same-process speed-up (skip the glob if we already proved
    this exact dir was fully seeded earlier in this process) — it never
    substitutes for the check across different results_dir values, so two
    tests using different tmp_path dirs in the same pytest process remain
    fully isolated.
    """
    resolved_dir = results_dir or os.environ.get("SCRIBEGATE_RESULTS_DIR") or RESULTS_DIR
    resolved_dir = str(resolved_dir)

    if resolved_dir in _SEEDED_DIRS:
        return False

    if _is_fully_seeded(resolved_dir):
        _SEEDED_DIRS.add(resolved_dir)
        return False

    # Only wrap in st.spinner when there's an active Streamlit script-run
    # context (plain pytest calls / non-Streamlit callers have none) —
    # st.spinner (and most st.* calls) raise/no-op oddly without one.
    ctx = None
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
    except Exception:
        ctx = None

    if ctx is not None:
        with st.spinner("First run — generating demo results..."):
            _seed_results(resolved_dir)
    else:
        _seed_results(resolved_dir)

    _SEEDED_DIRS.add(resolved_dir)
    return True


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


@st.cache_data(ttl=5)
def ui_copy() -> dict:
    """Load and parse specs/ui_copy.yaml — the single source of truth for
    every plain-language string this UI renders: plain_title/one_liner/
    why_it_matters/reading_guide per page key, the glossary, the 5-minute
    tour (tour_5min), jargon_swaps, and the live_mode/economics copy
    blocks. Views must consume this loader rather than hardcoding copy —
    W4's UI-overhaul rule (specs/ui_copy.yaml's own header comment)."""
    with open(UI_COPY_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def jargon_label(technical: str, fallback: str | None = None) -> str:
    """Return the dual "plain (technical)" form for `technical` per
    ui_copy.yaml's jargon_swaps list (e.g. "aggregate" ->
    "quality score (aggregate)"). Falls back to `fallback` (or `technical`
    unchanged) if no swap is defined for that key. Intended as a
    light-touch retrofit on existing metric/column labels inside a view —
    never a rewrite of the view's internals."""
    swaps = {
        s.get("technical"): s.get("combined")
        for s in (ui_copy().get("jargon_swaps") or [])
        if isinstance(s, dict)
    }
    return swaps.get(technical) or (fallback if fallback is not None else technical)


# ---------------------------------------------------------------------------
# Shared page registry — populated once by app/streamlit_app.py with the
# st.Page objects it constructs, so views (start_here.py's "5-minute tour")
# can st.page_link to a sibling page without a circular import on
# streamlit_app.py itself.
# ---------------------------------------------------------------------------

PAGE_REGISTRY: dict[str, object] = {}


def register_pages(pages_by_key: dict) -> None:
    """Populate the shared page registry (page_key -> st.Page object).
    Called once by app/streamlit_app.py right after it constructs its
    st.Page objects, before st.navigation(...) runs."""
    PAGE_REGISTRY.clear()
    PAGE_REGISTRY.update(pages_by_key)


def page_header(page_key: str) -> None:
    """Shared plain-language page header, consumed from ui_copy(): renders
    `plain_title` as the page's st.header, `one_liner` as a caption
    subtitle directly below it, and a single "Why this matters + how to
    read this page" expander built from `why_it_matters` + `reading_guide`
    for `page_key`. Every view calls this instead of its own bespoke
    st.header/caption block; page-specific content renders below,
    untouched."""
    data = ui_copy()
    page = (data.get("pages") or {}).get(page_key) or {}
    plain_title = page.get("plain_title") or page_key.replace("_", " ").title()
    one_liner = page.get("one_liner", "")
    why_it_matters = page.get("why_it_matters", "")
    reading_guide = page.get("reading_guide") or []

    st.header(plain_title)
    if one_liner:
        st.caption(one_liner)

    if why_it_matters or reading_guide:
        with st.expander("Why this matters + how to read this page"):
            if why_it_matters:
                st.markdown("**Why this matters**")
                st.markdown(why_it_matters)
            if reading_guide:
                st.markdown("**How to read this page**")
                for item in reading_guide:
                    st.markdown(f"- {item}")


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
