"""app/views/overview.py — Overview page (v0.2).

v0.1's benchmark-dashboard view, adapted, plus: a routing bar chart built
from scribegate.analytics.routing_summary, and headline st.metric cards
(overall aggregate, auto-accept rate, tests passing, golden-set size).
"""

from __future__ import annotations

import glob
import os
import re

import pandas as pd
import streamlit as st

from app.common import GOLDEN_DIR, load_benchmark_md, load_results
from scribegate import corrections
from scribegate.analytics import routing_summary

def _count_tests() -> int:
    """Count test functions by scanning tests/*.py — cheap, dynamic, and never
    goes stale the way a hardcoded headline number would."""
    tests_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tests")
    count = 0
    for path in glob.glob(os.path.join(tests_dir, "test_*.py")):
        try:
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
        except OSError:
            continue
        count += len(re.findall(r"^\s*def test_", src, flags=re.MULTILINE))
    return count


def render() -> None:
    st.header("Overview")

    results = load_results()

    if not results:
        st.warning(
            "No results found yet. Run the pipeline first:\n\n"
            "```\npython -m scribegate.cli run --all\npython -m scribegate.benchmark\n```"
        )
        return

    results_list = list(results.values())
    summary = routing_summary(results_list)

    golden_count = len(glob.glob(os.path.join(GOLDEN_DIR, "*.json")))
    correction_count = corrections.correction_stats().get("count", 0)
    golden_set_size = golden_count + correction_count

    overall_aggregate = summary.get("mean_aggregate_overall")
    auto_accept = summary.get("by_route", {}).get("auto_accept", {})
    auto_accept_rate = auto_accept.get("rate", 0.0)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Overall aggregate",
            f"{overall_aggregate:.3f}" if overall_aggregate is not None else "—",
        )
    with m2:
        st.metric("Auto-accept rate", f"{auto_accept_rate * 100:.1f}%")
    with m3:
        st.metric("Test functions", f"{_count_tests()}")
    with m4:
        st.metric("Golden-set size", golden_set_size)

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
    by_route = summary.get("by_route", {})
    route_totals = pd.Series(
        {route: by_route.get(route, {}).get("count", 0) for route in ("auto_accept", "review", "regenerate")}
    )
    st.bar_chart(route_totals)

    with st.expander("Raw benchmark.md"):
        md = load_benchmark_md()
        if md:
            st.markdown(md)
        else:
            st.caption("benchmark.md not found — run `python -m scribegate.benchmark`.")
