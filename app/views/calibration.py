"""app/views/calibration.py — Judge calibration page (v0.3).

Surfaces scribegate.calibration's probabilistic-judge instrumentation:
mean CI95 width per visit type (the messy contact-lens fixtures are the
widest, by design — see scribegate/calibration.py's module docstring),
how many routes changed under CI-aware (CI95-lower-bound) routing vs.
point-estimate routing, and a per-case table of the full calibration
report.

Loads data/results/calibration_report.json, regenerating it via
scribegate.calibration.calibration_report() if missing — same self-seed
pattern as app.common.ensure_results (cheap existence check first, wrap
the actual (re)generation in st.spinner only when there's an active
Streamlit script-run context, plain function call otherwise so pytest/
non-Streamlit callers work too).

Named `calibration.py` inside app/views/ — this is a different dotted
path (app.views.calibration) than scribegate.calibration, so `from
scribegate import calibration` below always resolves scribegate's
calibration.py, never this module itself (same convention as
app/views/analytics.py's `from scribegate import analytics`).
"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from app.common import RESULTS_DIR, page_header

CALIBRATION_REPORT_PATH = os.path.join(RESULTS_DIR, "calibration_report.json")


def _generate_calibration_report() -> None:
    from scribegate import calibration

    report = calibration.calibration_report()
    os.makedirs(os.path.dirname(CALIBRATION_REPORT_PATH), exist_ok=True)
    with open(CALIBRATION_REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=False)
        fh.write("\n")


@st.cache_data(ttl=5)
def _load_calibration_report_cached(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _ensure_calibration_report() -> dict | None:
    """Load data/results/calibration_report.json, regenerating it if missing.

    Belt-and-suspenders guard: calibration_report.json is now precomputed and
    shipped as a tracked artifact (see .gitignore's
    `!data/results/calibration_report.json` exception), so this should
    always find the file already present on a fresh clone/deploy. On a
    memory-constrained host (e.g. Streamlit Community Cloud's 1GB RAM) where
    the shipped artifact is somehow missing anyway, set
    SCRIBEGATE_DISABLE_HEAVY_SEED=1 to skip the heavy self-seed
    (scribegate.calibration.calibration_report(), which re-judges every
    bundled transcript multiple times) instead of risking an OOM/CPU kill —
    returns None so the caller can fall back to a "run locally" caption.
    """
    if not os.path.exists(CALIBRATION_REPORT_PATH):
        if os.environ.get("SCRIBEGATE_DISABLE_HEAVY_SEED") == "1":
            return None

        ctx = None
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx

            ctx = get_script_run_ctx()
        except Exception:
            ctx = None

        if ctx is not None:
            with st.spinner("Calibration report not found — generating..."):
                _generate_calibration_report()
        else:
            _generate_calibration_report()

    return _load_calibration_report_cached(CALIBRATION_REPORT_PATH)


def _render_headline(report: dict) -> None:
    summary = report.get("summary", {})
    ci_widths = summary.get("mean_ci_width_by_visit_type", {})

    st.subheader("Mean CI95 width per visit type")
    st.caption(
        "Wider bar = a single-draw point estimate is less trustworthy for that visit "
        "type. The messy contact-lens fixtures are deliberately the widest."
    )
    if ci_widths:
        # Column names are visit-type strings with no colons — safe for
        # Altair/st.bar_chart's shorthand encoding (see app/views/drift.py's
        # colon-stripping fix for why a raw "key:value" column name breaks it).
        chart_series = pd.Series(ci_widths)
        st.bar_chart(chart_series)
    else:
        st.caption("No CI width data available.")

    n_cases = summary.get("n_cases", 0)
    n_changed = summary.get("n_routes_changed", 0)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Routes changed under CI-aware routing", f"{n_changed}/{n_cases}")
    with c2:
        widest_vt = max(ci_widths, key=ci_widths.get) if ci_widths else "—"
        st.metric("Widest CI95 (visit type)", widest_vt)


def _render_per_case_table(report: dict) -> None:
    st.subheader("Per-case detail")
    cases = report.get("cases", [])
    if not cases:
        st.caption("No cases in the calibration report.")
        return

    rows = []
    for c in cases:
        ci_lo, ci_hi = c.get("ci95", [None, None])
        rows.append(
            {
                "Transcript ID": c.get("transcript_id"),
                "Visit Type": c.get("visit_type"),
                "Mean Aggregate": round(c.get("aggregate_mean", 0.0), 3),
                "Std": round(c.get("aggregate_std", 0.0), 3),
                "CI95 Low": round(ci_lo, 3) if ci_lo is not None else None,
                "CI95 High": round(ci_hi, 3) if ci_hi is not None else None,
                "Point Route": c.get("point_route"),
                "CI Route": c.get("ci_route"),
                "Changed?": "YES" if c.get("changed") else "no",
                # flags is a list — cast to a comma-joined string so the
                # dataframe never has a mixed-type/list-valued object column
                # (see app/views/analytics.py's assumptions_df str-cast fix
                # for the same Arrow-serialization issue).
                "Flags": ", ".join(c.get("flags", [])) or "—",
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_explainer() -> None:
    with st.expander("How to read this page", expanded=False):
        st.markdown(
            "A probabilistic judge (a real LLM, even at temperature 0 across model/prompt "
            "revisions) gives a **distribution** of scores for the same note, not a single "
            "number — judge it 7 times and you get 7 slightly different aggregates, not 7 "
            "copies of one aggregate.\n\n"
            "Because of that, ScribeGate routes on the **CI95 lower bound** instead of the "
            "mean aggregate: nothing auto-accepts unless even the pessimistic read of "
            "repeated judging still clears the 0.85 threshold. This is strictly more "
            "conservative than point-estimate routing (it can only hold steady or demote a "
            "case to review/regenerate — never promote one point-estimate routing wouldn't "
            "have already accepted).\n\n"
            "The messy contact-lens fixture set genuinely shows the widest score "
            "distributions of the four visit types (see the bar chart above) — noisy audio "
            "(overlapping speech, inaudible markers) is a real signal that a probabilistic "
            "re-judge would disagree with itself more on that transcript. That's exactly "
            "the class of case a point estimate over-trusts, and exactly what this "
            "instrumentation is built to catch before it reaches auto-accept."
        )


def render() -> None:
    page_header("calibration")
    st.caption(
        "**Demo on synthetic data.** CI-aware routing instrumentation over the bundled "
        "20 synthetic transcripts — no PHI, no production customer data."
    )

    report = _ensure_calibration_report()

    if report is None:
        st.caption(
            "Calibration report not available on this host (heavy self-seed disabled "
            "via SCRIBEGATE_DISABLE_HEAVY_SEED and no precomputed report present). Run "
            "`python -m scribegate.calibration` locally to generate "
            "data/results/calibration_report.json."
        )
        return

    _render_headline(report)
    st.divider()
    _render_per_case_table(report)
    st.divider()
    _render_explainer()
