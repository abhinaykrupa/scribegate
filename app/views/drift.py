"""app/views/drift.py — Drift page (v0.2).

Time-series drift view over data/results/history.jsonl (falling back to
history_demo.jsonl if history.jsonl is empty/missing), rolling-window
regression alerts, and a "how the CI gate works" expander showing the
committed baseline floors + the actual CI workflow YAML.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from app.common import HISTORY_DEMO_PATH, HISTORY_PATH, REPO_ROOT
from scribegate.drift import check_against_baseline, detect_regression, load_history, summarize_drift

BASELINE_PATH = os.path.join(REPO_ROOT, "specs", "baseline.json")
CI_WORKFLOW_PATH = os.path.join(REPO_ROOT, ".github", "workflows", "ci.yml")


@st.cache_data(ttl=5)
def _load_baseline_json() -> dict:
    with open(BASELINE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(ttl=5)
def _load_ci_yaml_text() -> str:
    with open(CI_WORKFLOW_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


def _load_history_with_fallback() -> tuple[list[dict], str]:
    history = load_history(HISTORY_PATH)
    if history:
        return history, HISTORY_PATH
    history = load_history(HISTORY_DEMO_PATH)
    return history, HISTORY_DEMO_PATH


def render() -> None:
    st.header("Drift")

    history, source_path = _load_history_with_fallback()

    if not history:
        st.warning(
            "No history rows found in either data/results/history.jsonl or "
            "data/results/history_demo.jsonl."
        )
        return

    st.caption(f"Showing history from `{os.path.relpath(source_path, REPO_ROOT)}` ({len(history)} row(s)).")

    # --- Alert banners ---
    alerts = detect_regression(history)
    if alerts:
        for alert in alerts:
            st.error(alert.message)
    else:
        st.success("No regressions detected in the current rolling window.")

    # --- Time-series charts per metric ---
    st.subheader("Score time series")
    summary = summarize_drift(history)
    for metric, series in summary.items():
        if not series:
            continue
        df = pd.DataFrame(series)
        if "ts" not in df.columns or "value" not in df.columns:
            continue
        df = df.set_index("ts")[["value"]].rename(columns={"value": metric})
        st.markdown(f"**{metric}**")
        st.line_chart(df)

    # --- How the CI gate works ---
    with st.expander("How the CI gate works"):
        st.markdown(
            "The `eval-gate` job in CI runs the full pipeline over every "
            "transcript, regenerates the benchmark report, and checks the "
            "latest run against the committed floor baseline below. A run "
            "that falls below any floor fails the build."
        )
        st.markdown("**specs/baseline.json**")
        try:
            baseline = _load_baseline_json()
            st.code(json.dumps(baseline, indent=2), language="json")
        except (OSError, json.JSONDecodeError) as exc:
            st.caption(f"Could not load specs/baseline.json: {exc}")

        st.markdown("**.github/workflows/ci.yml**")
        try:
            ci_yaml_text = _load_ci_yaml_text()
            st.code(ci_yaml_text, language="yaml")
        except OSError as exc:
            st.caption(f"Could not load .github/workflows/ci.yml: {exc}")

        if history:
            latest = history[-1]
            passed, failures = check_against_baseline(latest, _load_baseline_json())
            if passed:
                st.success("Latest history row meets all baseline floors.")
            else:
                st.error("Latest history row falls below baseline floor(s):")
                for msg in failures:
                    st.markdown(f"- {msg}")
