"""app/views/analytics.py — Analytics page (v0.2).

Dimension heatmap, failure-mode clustering, and ROI slider panel over
scribegate.analytics. Named `analytics.py` inside app/views/ — this is a
different dotted path (app.views.analytics) than scribegate.analytics, so
`from scribegate import analytics` below always resolves scribegate's
analytics.py, never this module itself.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.common import GOLDEN_DIR, load_results
from scribegate import analytics
from scribegate.analytics import RoiParams, dimension_matrix, failure_modes, roi_model, routing_summary


def _matplotlib_available() -> bool:
    """pandas Styler.background_gradient() imports matplotlib lazily and
    raises ImportError if it isn't installed. matplotlib is an optional
    dependency (heavy, not needed for the rest of the app), so probe for it
    explicitly rather than letting the page crash on a bare/minimal
    environment that lacks it."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def _render_dimension_heatmap(results_list: list[dict]) -> None:
    st.subheader("Dimension matrix (visit type x dimension)")
    matrix = dimension_matrix(results_list)
    grid = matrix.get("grid", [])
    if not grid:
        st.caption("No data available for the dimension matrix.")
        return

    df = pd.DataFrame(grid).set_index("visit_type")
    dims = matrix.get("dimensions", [])
    display_cols = [c for c in dims if c in df.columns]
    if _matplotlib_available():
        styled = df[display_cols].style.background_gradient(cmap="RdYlGn", vmin=1, vmax=5)
        st.dataframe(styled, use_container_width=True)
    else:
        st.caption("(Install matplotlib for color-graded cells; showing plain values.)")
        st.dataframe(df[display_cols], use_container_width=True)
    st.caption("Cell values are mean judge scores (1-5) per visit type; n = transcript count per visit type.")
    st.dataframe(df[["n"]], use_container_width=True)


def _render_failure_modes(results_list: list[dict]) -> None:
    st.subheader("Failure modes")
    modes = failure_modes(results_list, golden_notes_dir=GOLDEN_DIR)

    tabs = st.tabs(["By dimension", "By violation code", "By section", "Worst cases"])

    with tabs[0]:
        by_dim = modes.get("by_dimension", [])
        if by_dim:
            st.dataframe(pd.DataFrame(by_dim), use_container_width=True, hide_index=True)
        else:
            st.caption("No dimension data.")

    with tabs[1]:
        by_code = modes.get("by_violation_code", [])
        if by_code:
            st.dataframe(pd.DataFrame(by_code), use_container_width=True, hide_index=True)
        else:
            st.caption("No violations recorded across the result set.")

    with tabs[2]:
        by_section = modes.get("by_section", [])
        if by_section:
            st.dataframe(pd.DataFrame(by_section), use_container_width=True, hide_index=True)
        else:
            st.caption("No section data.")

    with tabs[3]:
        worst_cases = modes.get("worst_cases", [])
        if not worst_cases:
            st.caption("No worst cases — results set is empty.")
        for case in worst_cases:
            tid = case.get("transcript_id")
            aggregate = case.get("aggregate")
            route = case.get("route")
            with st.expander(f"{tid} — aggregate {aggregate} — route `{route}`"):
                st.write(f"**Visit type:** {case.get('visit_type')}")
                st.write("**Reasons:**")
                for reason in case.get("reasons", []):
                    st.markdown(f"- {reason}")


def _render_roi_section(results_list: list[dict]) -> None:
    st.subheader("ROI model")
    defaults = RoiParams()
    summary = routing_summary(results_list)

    c1, c2 = st.columns(2)
    with c1:
        providers = st.slider("Providers", min_value=1, max_value=50, value=defaults.providers)
        visits_per_provider_per_day = st.slider(
            "Visits per provider per day", min_value=1, max_value=60,
            value=defaults.visits_per_provider_per_day,
        )
        clinic_days_per_month = st.slider(
            "Clinic days per month", min_value=1, max_value=31,
            value=defaults.clinic_days_per_month,
        )
        clinician_hourly_cost = st.slider(
            "Clinician hourly cost ($)", min_value=20.0, max_value=500.0,
            value=defaults.clinician_hourly_cost, step=5.0,
        )
    with c2:
        minutes_full_review = st.slider(
            "Minutes: full review", min_value=0.5, max_value=20.0,
            value=defaults.minutes_full_review, step=0.5,
        )
        minutes_spot_check = st.slider(
            "Minutes: spot check", min_value=0.1, max_value=10.0,
            value=defaults.minutes_spot_check, step=0.1,
        )
        minutes_regenerate_handling = st.slider(
            "Minutes: regenerate handling", min_value=0.5, max_value=20.0,
            value=defaults.minutes_regenerate_handling, step=0.5,
        )

    params = RoiParams(
        providers=providers,
        visits_per_provider_per_day=visits_per_provider_per_day,
        clinic_days_per_month=clinic_days_per_month,
        minutes_full_review=minutes_full_review,
        minutes_spot_check=minutes_spot_check,
        minutes_regenerate_handling=minutes_regenerate_handling,
        clinician_hourly_cost=clinician_hourly_cost,
    )
    roi = roi_model(summary, params)

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Dollars saved / month", f"${roi['dollars_saved_per_month']:,.2f}")
    with m2:
        st.metric("Hours saved / month", f"{roi['hours_saved_per_month']:.1f}")

    st.markdown("**Assumptions**")
    # Values mix numbers and prose (e.g. the methodology note) — cast to str so
    # Streamlit's Arrow serialization never fails on a mixed-type object column.
    assumptions_df = pd.DataFrame(
        [{"key": k, "value": str(v)} for k, v in roi.get("assumptions", {}).items()]
    )
    st.dataframe(assumptions_df, use_container_width=True, hide_index=True)


def render() -> None:
    st.header("Analytics")

    results = load_results()
    if not results:
        st.warning(
            "No results found yet. Run the pipeline first:\n\n"
            "```\npython -m scribegate.cli run --all\n```"
        )
        return

    results_list = list(results.values())

    _render_dimension_heatmap(results_list)
    st.divider()
    _render_failure_modes(results_list)
    st.divider()
    _render_roi_section(results_list)
