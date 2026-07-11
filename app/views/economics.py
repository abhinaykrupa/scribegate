"""app/views/economics.py — Economics page (v0.4), a CFO-minded operator's page.

Surfaces scribegate.economics's unit-economics engine: headline cards
(cheapest tier clearing the quality floor, its margin, and the uplift vs
the premium tier), a haiku/sonnet/opus tier-comparison table, sliders
bound to NoteEconParams that recompute cost/margin live, the mock-proxy
model x golden-generation matrix (the moat -> margin proof, labeled
honestly as a mock-generator proxy, not a live-API measurement), and an
assumption-honesty note.

Named `economics.py` inside app/views/ — a different dotted path
(app.views.economics) than scribegate.economics, so the import below
always resolves scribegate's economics.py, never this module itself (same
convention as app/views/analytics.py's `from scribegate import
analytics`).

The headline cards and tier-comparison table are rendered into
placeholders created BEFORE the sliders are read, so they appear above the
sliders on the page while still reflecting this run's slider values (an
`st.container()` placeholder can be filled after later code executes,
independent of where it was created) — the live-recompute behavior the W4
spec calls for.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from app.common import RESULTS_DIR, page_header, ui_copy
from scribegate.economics import (
    NoteEconParams,
    econ_summary,
    model_generation_matrix,
    tier_comparison,
)


def _render_headline(summary: dict) -> None:
    st.subheader("Headline")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Cheapest tier clearing the quality floor", summary["cheapest_tier_meeting_floor"])
        st.caption(summary["cheapest_tier_meeting_floor_note"])
    with c2:
        st.metric("Its gross margin", f"{summary['cheapest_tier_margin_pct'] * 100:.1f}%")
    with c3:
        st.metric(
            f"Margin uplift vs {summary['premium_tier']} (premium)",
            f"{summary['margin_uplift_vs_premium_tier_pct'] * 100:+.1f} pts",
        )
    lo, hi = summary["cost_per_note_range_usd"]
    st.caption(
        f"Cost per note ranges ${lo:.4f} - ${hi:.4f} across tiers "
        f"(latest reference-set generation: {summary.get('latest_generation') or '—'})."
    )


def _render_tier_table(comparison: list[dict]) -> None:
    st.subheader("Tier comparison")
    rows = [
        {
            "Tier": r["model_tier"],
            "Cost / note": f"${r['cost_per_note_usd']:.4f}",
            "Notes / month": r["notes_per_month"],
            "Revenue / month": f"${r['revenue_per_month_usd']:.2f}",
            "Gross margin %": f"{r['gross_margin_pct'] * 100:.1f}%",
            "Margin delta vs priciest tier": f"{r['margin_delta_vs_most_expensive_pct'] * 100:+.1f} pts",
        }
        for r in comparison
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_sliders() -> NoteEconParams:
    st.subheader("Assumptions — move a slider, watch margin move")
    defaults = NoteEconParams()
    c1, c2 = st.columns(2)
    with c1:
        price = st.slider(
            "Price / provider / month ($)",
            min_value=20.0, max_value=500.0,
            value=defaults.price_per_provider_per_month, step=1.0,
        )
        providers = st.slider(
            "Providers", min_value=1, max_value=50, value=defaults.providers,
        )
        visits = st.slider(
            "Visits / provider / day", min_value=1, max_value=60,
            value=defaults.visits_per_provider_per_day,
        )
    with c2:
        clinic_days = st.slider(
            "Clinic days / month", min_value=1, max_value=31,
            value=defaults.clinic_days_per_month,
        )
        judge_samples = st.slider(
            "Judge samples / note", min_value=1, max_value=7,
            value=defaults.judge_samples,
        )
    return NoteEconParams(
        price_per_provider_per_month=price,
        providers=providers,
        visits_per_provider_per_day=visits,
        clinic_days_per_month=clinic_days,
        judge_samples=judge_samples,
    )


def _render_matrix(matrix: dict) -> None:
    st.subheader("Model x generation matrix — the moat -> margin proof")
    st.caption(matrix.get("label_note", ""))
    cells = matrix.get("cells", [])
    floor = matrix.get("quality_floor", 0.80)
    rows = [
        {
            "Model quality proxy (mock, labeled honestly)": c["model_quality_proxy"],
            "Reference-set generation": c["generation"],
            "n": c["n"],
            "Mean aggregate": f"{c['mean_aggregate']:.4f}",
            f"Meets {floor} floor?": "YES" if c["meets_floor"] else "no",
        }
        for c in cells
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    story = matrix.get("story", {})
    if story.get("applicable"):
        st.markdown(story.get("narrative", ""))
    else:
        st.caption(story.get("reason", ""))


def render() -> None:
    page_header("economics")

    data = ui_copy()
    econ_copy = data.get("economics") or {}
    framing = econ_copy.get("framing", "")
    if framing:
        st.info(framing)

    headline_placeholder = st.container()
    st.divider()
    tier_table_placeholder = st.container()
    st.divider()

    params = _render_sliders()

    # model_generation_matrix() recomputes multiple full benchmark runs when
    # its cache (data/results/econ_matrix.json) is absent — heavy enough to
    # kill a memory-constrained host mid-render (same class as the moat/
    # calibration self-seeds). The cache ships precomputed in the repo; if
    # it's missing AND heavy recompute is disabled, degrade gracefully.
    matrix_cache = os.path.join(RESULTS_DIR, "econ_matrix.json")
    if os.path.exists(matrix_cache) or os.environ.get("SCRIBEGATE_DISABLE_HEAVY_SEED") != "1":
        matrix = model_generation_matrix()
    else:
        # NOTE: econ_summary(matrix=None) would recompute the matrix itself —
        # the exact heavy path we're avoiding — so skip matrix-dependent
        # sections entirely in this degraded mode.
        matrix = None
        st.info(
            "Model×generation matrix cache not found and heavy recompute is disabled "
            "on this host. Run locally: python -m scribegate.economics"
        )
    comparison = tier_comparison(params)

    if matrix is not None:
        summary = econ_summary(params, matrix=matrix)
        with headline_placeholder:
            _render_headline(summary)
    with tier_table_placeholder:
        _render_tier_table(comparison)

    if matrix is not None:
        st.divider()
        _render_matrix(matrix)

    assumption_honesty = econ_copy.get("assumption_honesty", "")
    if assumption_honesty:
        st.divider()
        st.warning(assumption_honesty)
