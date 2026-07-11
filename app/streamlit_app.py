"""ScribeGate demo UI (v0.4) — thin multi-page entry point.

Twelve pages (via st.navigation/st.Page, dispatching to app/views/*.py's
render() functions), grouped into sidebar sections for a non-engineer
audience:
    Start        — Start here (the default landing page: a 5-minute tour
                    + glossary).
    Quality      — Overview, Analytics, Drift, Judge calibration.
    The moat     — Data moat, Review queue.
    Trust        — Provenance, Live encounter.
    Run it live  — Live mode (real API calls, budget-capped), Economics.
    About        — About.

Every page title is plain_title from specs/ui_copy.yaml (loaded once via
app.common.ui_copy()), so the sidebar itself speaks the same
non-engineer-readable language as the pages it links to. Every page's
st.Page object is also registered into app.common.PAGE_REGISTRY so
app/views/start_here.py's tour can st.page_link to a sibling page without
importing this module.

Zero API keys, zero network by default: everything except Live mode is
read from bundled data/ and specs/ files. stdlib + streamlit + pandas +
pyyaml only. Run with:
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

from app.common import ensure_results, register_pages, render_banner, render_sidebar_wordmark, ui_copy
from app.views import (
    about,
    analytics,
    calibration,
    drift,
    economics,
    live_encounter,
    live_mode,
    moat,
    overview,
    provenance,
    review_queue,
    start_here,
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

_pages_copy = (ui_copy().get("pages") or {})


def _title(page_key: str, fallback: str) -> str:
    return (_pages_copy.get(page_key) or {}).get("plain_title") or fallback


# Default page always serves at "/" — an explicit url_path on the default
# page would be dead (hitting it triggers Streamlit's "Page not found"
# toast), so start_here is the only st.Page below with no url_path.
page_start_here = st.Page(start_here.render, title=_title("start_here", "Start here"), default=True)
page_overview = st.Page(overview.render, title=_title("overview", "Overview"), url_path="overview")
page_analytics = st.Page(analytics.render, title=_title("analytics", "Analytics"), url_path="analytics")
page_drift = st.Page(drift.render, title=_title("drift", "Drift"), url_path="drift")
page_calibration = st.Page(
    calibration.render, title=_title("calibration", "Judge calibration"), url_path="judge-calibration"
)
page_moat = st.Page(moat.render, title=_title("moat", "Data moat"), url_path="data-moat")
page_review_queue = st.Page(
    review_queue.render, title=_title("review_queue", "Review queue"), url_path="review-queue"
)
page_provenance = st.Page(
    provenance.render, title=_title("provenance", "Receipts (provenance)"), url_path="provenance"
)
page_live_encounter = st.Page(
    live_encounter.render, title=_title("live_encounter", "Live encounter"), url_path="live-encounter"
)
page_live_mode = st.Page(live_mode.render, title=_title("live_mode", "Live mode"), url_path="live-mode")
page_economics = st.Page(economics.render, title=_title("economics", "Economics"), url_path="economics")
page_about = st.Page(about.render, title=_title("about", "About"), url_path="about")

register_pages(
    {
        "start_here": page_start_here,
        "overview": page_overview,
        "analytics": page_analytics,
        "drift": page_drift,
        "calibration": page_calibration,
        "moat": page_moat,
        "review_queue": page_review_queue,
        "provenance": page_provenance,
        "live_encounter": page_live_encounter,
        "live_mode": page_live_mode,
        "economics": page_economics,
        "about": page_about,
    }
)

SECTIONS = {
    "Start": [page_start_here],
    "Quality": [page_overview, page_analytics, page_drift, page_calibration],
    "The moat": [page_moat, page_review_queue],
    "Trust": [page_provenance, page_live_encounter],
    "Run it live": [page_live_mode, page_economics],
    "About": [page_about],
}

pg = st.navigation(SECTIONS)
pg.run()
