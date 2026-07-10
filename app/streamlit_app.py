"""ScribeGate demo UI (v0.3) — thin multi-page entry point.

Nine pages (via st.navigation/st.Page, dispatching to app/views/*.py's
render() functions):
    1. Overview        — headline metrics, per-visit-type/per-transcript
                          tables, routing chart.
    2. Analytics        — dimension heatmap, failure-mode clustering, ROI
                          slider panel.
    3. Drift            — score time series, regression alerts, CI gate
                          explainer.
    4. Data moat        — moat curve (aggregate + auto-accept rate per
                          golden generation), generations table, promote-
                          pending-candidates button.
    5. Judge calibration — mean CI95 width per visit type, CI-aware vs
                          point-estimate routing deltas, per-case table.
    6. Review queue     — worst-first review, approve/reject, line-level
                          corrections, candidate-golden diffs.
    7. Provenance       — click a note line, see the exact transcript
                          span(s) that support it; audit dossier export.
    8. Live encounter   — consent-gated mic/text capture -> generate ->
                          judge (reference-free) -> route -> provenance.
    9. About            — README excerpts, links, production-path/demo-
                          script expanders.

Zero API keys, zero network by default: everything is read from bundled
data/ and specs/ files. stdlib + streamlit + pandas + pyyaml only. Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys

# Streamlit Cloud runs the script with sys.path[0] = the script's own dir
# (app/), so repo-root imports like `app.common` and `scribegate.*` fail
# unless the repo root is on sys.path. Locally (run from repo root) this
# insert is a harmless no-op duplicate.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st

from app.common import ensure_results, render_banner, render_sidebar_wordmark
from app.views import (
    about,
    analytics,
    calibration,
    drift,
    live_encounter,
    moat,
    overview,
    provenance,
    review_queue,
)

st.set_page_config(page_title="ScribeGate", layout="wide")

render_sidebar_wordmark()
render_banner()

try:
    ensure_results()
except Exception as exc:  # noqa: BLE001 - must never block page render
    st.error(
        f"Demo data seeding failed: {exc}\n\n"
        "Run manually: python -m scribegate.cli run --all"
    )

pages = [
    # Default page always serves at "/" — an explicit url_path would be dead
    # (hitting it triggers Streamlit's "Page not found" toast).
    st.Page(overview.render, title="Overview", default=True),
    st.Page(analytics.render, title="Analytics", url_path="analytics"),
    st.Page(drift.render, title="Drift", url_path="drift"),
    st.Page(moat.render, title="Data moat", url_path="data-moat"),
    st.Page(calibration.render, title="Judge calibration", url_path="judge-calibration"),
    st.Page(review_queue.render, title="Review queue", url_path="review-queue"),
    st.Page(provenance.render, title="Provenance", url_path="provenance"),
    st.Page(live_encounter.render, title="Live encounter", url_path="live-encounter"),
    st.Page(about.render, title="About", url_path="about"),
]

pg = st.navigation(pages)
pg.run()
