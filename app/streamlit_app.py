"""ScribeGate demo UI (T7) — Streamlit dashboard over data/results/*.json.

Three views (sidebar radio):
  1. Benchmark dashboard  — aggregate stats, per-visit-type / per-transcript
     tables, route distribution chart, honest-framing callout.
  2. Review queue          — transcripts routed "review"/"regenerate",
     worst-first, with approve/reject buttons appending to
     data/results/decision_log.jsonl.
  3. Provenance note view  — click a note line, see the exact transcript
     span(s) that support it, highlighted char-exact. Also works for the
     golden note.

Zero API keys, zero network: everything is read from bundled data/ files.
stdlib + streamlit + pandas only. Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import glob
import html
import json
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

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

def load_results() -> dict:
    """Load every data/results/{id}.json into {transcript_id: result_dict}."""
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


def load_transcript_text(transcript_id: str) -> str | None:
    path = os.path.join(TRANSCRIPTS_DIR, f"{transcript_id}.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def load_golden_note(transcript_id: str) -> dict | None:
    path = os.path.join(GOLDEN_DIR, f"{transcript_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


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

def render_banner() -> None:
    st.markdown(
        """
        <div style="background-color:#1f3d2b;border:1px solid #3c7a52;
                    border-radius:6px;padding:0.5rem 1rem;margin-bottom:1rem;
                    color:#c9f2d6;font-size:0.9rem;">
            <strong>100% SYNTHETIC data — no PHI.</strong>
            All transcripts, notes, and scores on this page are generated
            fixtures for demo purposes only. Nothing here is a real patient
            encounter.
        </div>
        """,
        unsafe_allow_html=True,
    )


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
# View 1: Benchmark dashboard
# ---------------------------------------------------------------------------

def view_benchmark_dashboard(results: dict) -> None:
    st.header("Benchmark dashboard")

    if not results:
        st.warning(
            "No results found yet. Run the pipeline first:\n\n"
            "```\npython -m scribegate.cli run --all\npython -m scribegate.benchmark\n```"
        )
        return

    # --- Per-transcript table (built directly from results/*.json, always
    # available even if benchmark.md wasn't regenerated) ---
    rows = []
    for tid, r in results.items():
        scores = r.get("judge_result", {}).get("scores", {})
        violations = r.get("violations", [])
        err = sum(1 for v in violations if v.get("severity") == "error")
        warn = sum(1 for v in violations if v.get("severity") == "warn")
        rows.append(
            {
                "Transcript ID": tid,
                "Visit Type": r.get("visit_type", ""),
                "Aggregate": round(r.get("judge_result", {}).get("aggregate", 0.0), 3),
                "Completeness": scores.get("completeness"),
                "Hallucination": scores.get("hallucination"),
                "Coding Plausibility": scores.get("coding_plausibility"),
                "Terminology": scores.get("terminology"),
                "Route": r.get("route", ""),
                "Violations (err/warn)": f"{err}/{warn}",
            }
        )
    df = pd.DataFrame(rows).sort_values("Transcript ID").reset_index(drop=True)

    # --- Per-visit-type summary ---
    agg_df = (
        df.groupby("Visit Type")
        .agg(
            N=("Transcript ID", "count"),
            Completeness=("Completeness", "mean"),
            Hallucination=("Hallucination", "mean"),
            **{"Coding Plausibility": ("Coding Plausibility", "mean")},
            Terminology=("Terminology", "mean"),
            **{"Mean Aggregate": ("Aggregate", "mean")},
        )
        .round(2)
    )
    route_counts = (
        df.pivot_table(index="Visit Type", columns="Route", values="Transcript ID", aggfunc="count")
        .fillna(0)
        .astype(int)
    )
    for col in ["auto_accept", "review", "regenerate"]:
        if col not in route_counts.columns:
            route_counts[col] = 0
    route_counts = route_counts[["auto_accept", "review", "regenerate"]]
    agg_df = agg_df.join(route_counts)

    st.subheader("Per visit type")
    st.dataframe(agg_df, use_container_width=True)

    # Honest-framing callout
    st.info(
        "**Reading this table:** the `contact_lens_fitting` transcripts are "
        "*deliberately messy* (colloquial dictation, more crosstalk, looser "
        "structure) by design of the synthetic fixture set — they exist to "
        "stress the generator and judge, not to represent a typical visit. "
        "A lower mean aggregate and higher `review`/`regenerate` route count "
        "for that visit type is **expected and correct**: it means the "
        "harness is working, not that the pipeline is broken."
    )

    st.subheader("Per transcript")
    sort_col = st.selectbox(
        "Sort by", options=list(df.columns), index=list(df.columns).index("Aggregate")
    )
    ascending = st.checkbox("Ascending", value=True)
    st.dataframe(
        df.sort_values(sort_col, ascending=ascending).reset_index(drop=True),
        use_container_width=True,
    )

    st.subheader("Route distribution")
    route_totals = df["Route"].value_counts().reindex(
        ["auto_accept", "review", "regenerate"], fill_value=0
    )
    st.bar_chart(route_totals)

    with st.expander("Raw benchmark.md"):
        md = load_benchmark_md()
        if md:
            st.markdown(md)
        else:
            st.caption("benchmark.md not found — run `python -m scribegate.benchmark`.")


# ---------------------------------------------------------------------------
# View 2: Review queue
# ---------------------------------------------------------------------------

def view_review_queue(results: dict) -> None:
    st.header("Review queue")

    queue = [
        (tid, r)
        for tid, r in results.items()
        if r.get("route") in ("review", "regenerate")
    ]
    if not queue:
        if not results:
            st.warning(
                "No results found yet. Run the pipeline first:\n\n"
                "```\npython -m scribegate.cli run --all\n```"
            )
        else:
            st.success("Nothing in the queue — every transcript auto-accepted.")
        return

    # Worst-first: regenerate before review, then by ascending aggregate.
    route_rank = {"regenerate": 0, "review": 1}
    queue.sort(
        key=lambda item: (
            route_rank.get(item[1].get("route"), 2),
            item[1].get("judge_result", {}).get("aggregate", 0.0),
        )
    )

    st.caption(f"{len(queue)} transcript(s) awaiting review, worst-first.")

    if "reviewed" not in st.session_state:
        st.session_state["reviewed"] = {}  # transcript_id -> "approved"/"rejected"

    for tid, r in queue:
        jr = r.get("judge_result", {})
        scores = jr.get("scores", {})
        rationales = jr.get("rationales", {})
        aggregate = jr.get("aggregate", 0.0)
        route = r.get("route", "")
        violations = r.get("violations", [])
        decision_reasons = r.get("decision_reasons", [])

        already = st.session_state["reviewed"].get(tid)
        badge = f" — **{already.upper()}**" if already else ""
        with st.expander(f"[{route}] {tid} — aggregate {aggregate:.3f}{badge}"):
            cols = st.columns(4)
            dims = [
                ("Completeness", "completeness"),
                ("Hallucination", "hallucination"),
                ("Coding Plausibility", "coding_plausibility"),
                ("Terminology", "terminology"),
            ]
            for col, (label, key) in zip(cols, dims):
                with col:
                    st.metric(label, scores.get(key, "—"))
                    reason = rationales.get(key)
                    if reason:
                        st.caption(reason)

            st.markdown("**Violations**")
            if violations:
                vdf = pd.DataFrame(
                    [
                        {
                            "code": v.get("code"),
                            "severity": v.get("severity"),
                            "message": v.get("message"),
                        }
                        for v in violations
                    ]
                )
                st.dataframe(vdf, use_container_width=True, hide_index=True)
            else:
                st.caption("None.")

            st.markdown("**Decision reasons (router)**")
            for reason in decision_reasons:
                st.markdown(f"- {reason}")

            # Guard: once a decision is recorded for this transcript in this
            # session, the buttons are disabled instead of re-armed — this is
            # what stops a double-click (or a click-Approve-then-Reject) from
            # writing a second, contradictory line to decision_log.jsonl for
            # the same transcript_id in the same session.
            if already:
                st.caption(f"Decision recorded: **{already}**. Buttons disabled for this session.")
                btn_cols = st.columns(2)
                with btn_cols[0]:
                    st.button("Approve", key=f"approve_{tid}", disabled=True)
                with btn_cols[1]:
                    st.button("Reject", key=f"reject_{tid}", disabled=True)
            else:
                btn_cols = st.columns(2)
                with btn_cols[0]:
                    if st.button("Approve", key=f"approve_{tid}"):
                        append_decision(tid, "approved")
                        st.session_state["reviewed"][tid] = "approved"
                        st.success(f"Recorded: {tid} approved by demo-user.")
                        st.rerun()
                with btn_cols[1]:
                    if st.button("Reject", key=f"reject_{tid}"):
                        append_decision(tid, "rejected")
                        st.session_state["reviewed"][tid] = "rejected"
                        st.error(f"Recorded: {tid} rejected by demo-user.")
                        st.rerun()


# ---------------------------------------------------------------------------
# View 3: Provenance note view (the wow moment)
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


def view_provenance(results: dict) -> None:
    st.header("Provenance note view")
    st.caption(
        "Click any note line to see exactly which part(s) of the transcript "
        "support it, highlighted character-exact."
    )

    if not results:
        st.warning(
            "No results found yet. Run the pipeline first:\n\n"
            "```\npython -m scribegate.cli run --all\n```"
        )
        return

    transcript_ids = sorted(results.keys())
    transcript_id = st.selectbox("Transcript", transcript_ids)

    result = results[transcript_id]
    transcript_text = load_transcript_text(transcript_id)
    if transcript_text is None:
        st.error(f"Transcript file not found for {transcript_id} — expected data/transcripts/{transcript_id}.txt")
        return

    show_golden = st.toggle(
        "Show GOLDEN note instead of generated",
        value=False,
        help="Inspect the reference/golden note's own spans — useful for a "
        "clinician to critique golden content, not just generator output.",
    )

    if show_golden:
        golden = load_golden_note(transcript_id)
        if golden is None:
            st.error(f"Golden note not found for {transcript_id}.")
            return
        st.caption(f"Viewing GOLDEN note for **{transcript_id}** ({result.get('visit_type', '')})")
        _render_note_and_transcript(transcript_id, transcript_text, golden, state_prefix="golden")
    else:
        generated = result.get("generated_note", {})
        jr = result.get("judge_result", {})
        st.caption(
            f"Viewing GENERATED note for **{transcript_id}** "
            f"({result.get('visit_type', '')}) — aggregate "
            f"{jr.get('aggregate', 0.0):.3f}, route `{result.get('route', '')}`"
        )
        _render_note_and_transcript(transcript_id, transcript_text, generated, state_prefix="generated")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="ScribeGate demo", layout="wide")
    render_banner()

    st.sidebar.title("ScribeGate")
    view = st.sidebar.radio(
        "View",
        ["Benchmark dashboard", "Review queue", "Provenance note view"],
    )

    results = load_results()

    if view == "Benchmark dashboard":
        view_benchmark_dashboard(results)
    elif view == "Review queue":
        view_review_queue(results)
    else:
        view_provenance(results)


if __name__ == "__main__":
    main()
