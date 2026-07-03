"""ScribeGate demo UI (v0.2) — thin multi-page entry point.

Seven pages (via st.navigation/st.Page, dispatching to app/views/*.py's
render() functions):
    1. Overview        — headline metrics, per-visit-type/per-transcript
                          tables, routing chart.
    2. Analytics        — dimension heatmap, failure-mode clustering, ROI
                          slider panel.
    3. Drift            — score time series, regression alerts, CI gate
                          explainer.
    4. Review queue     — worst-first review, approve/reject, line-level
                          corrections, candidate-golden diffs.
    5. Provenance       — click a note line, see the exact transcript
                          span(s) that support it; audit dossier export.
    6. Live encounter   — consent-gated mic/text capture -> generate ->
                          judge (reference-free) -> route -> provenance.
    7. About            — README excerpts, links, production-path/demo-
                          script expanders.

Zero API keys, zero network by default: everything is read from bundled
data/ and specs/ files. stdlib + streamlit + pandas + pyyaml only. Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from app.common import render_banner, render_sidebar_wordmark
from app.views import about, analytics, drift, live_encounter, overview, provenance, review_queue

st.set_page_config(page_title="ScribeGate", layout="wide")

render_sidebar_wordmark()
render_banner()

pages = [
    st.Page(overview.render, title="Overview", url_path="overview", default=True),
    st.Page(analytics.render, title="Analytics", url_path="analytics"),
    st.Page(drift.render, title="Drift", url_path="drift"),
    st.Page(review_queue.render, title="Review queue", url_path="review-queue"),
    st.Page(provenance.render, title="Provenance", url_path="provenance"),
    st.Page(live_encounter.render, title="Live encounter", url_path="live-encounter"),
    st.Page(about.render, title="About", url_path="about"),
]

pg = st.navigation(pages)
pg.run()
