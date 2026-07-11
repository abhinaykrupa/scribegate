"""app/views/start_here.py — Start here page (v0.4), the app's DEFAULT page.

A five-minute tour for a non-engineer opening ScribeGate for the first
time: a compact "what is this" intro, the six-stop tour from
specs/ui_copy.yaml's `tour_5min` (each stop links straight to the page it
describes via st.page_link, using the shared app.common.PAGE_REGISTRY so
this module never imports app/streamlit_app.py directly), and the full
glossary in a two-column expander.

All copy comes from app.common.ui_copy() (specs/ui_copy.yaml) — nothing
here is hardcoded prose beyond structural labels ("Stop N", "Look at",
"Takeaway").
"""

from __future__ import annotations

import streamlit as st

from app.common import PAGE_REGISTRY, page_header, ui_copy


def _render_tour(data: dict) -> None:
    st.subheader("The 5-minute tour")
    st.caption("Six stops, in order — each one is a single page. Follow them top to bottom.")

    pages_copy = data.get("pages") or {}
    tour = data.get("tour_5min") or []

    for i, stop in enumerate(tour, start=1):
        page_key = stop.get("page")
        look_at = stop.get("look_at", "")
        takeaway = stop.get("takeaway", "")
        plain_title = (pages_copy.get(page_key) or {}).get("plain_title", page_key or "")

        with st.container(border=True):
            st.markdown(f"**Stop {i}**")
            target_page = PAGE_REGISTRY.get(page_key)
            if target_page is not None:
                st.page_link(target_page, label=plain_title or page_key, icon="➡️")
            elif plain_title:
                st.markdown(f"**{plain_title}**")
            if look_at:
                st.markdown(f"Look at: {look_at}")
            if takeaway:
                st.markdown(f"Takeaway: {takeaway}")


def _render_glossary(data: dict) -> None:
    glossary = data.get("glossary") or {}
    if not glossary:
        return
    with st.expander("Glossary — every term used on this site, in one place"):
        terms = list(glossary.items())
        half = (len(terms) + 1) // 2
        col1, col2 = st.columns(2)
        for i, (term, definition) in enumerate(terms):
            target = col1 if i < half else col2
            with target:
                st.markdown(f"**{term}** — {definition}")


def render() -> None:
    page_header("start_here")

    data = ui_copy()
    intro = (data.get("pages") or {}).get("start_here", {}).get("why_it_matters", "").strip()
    if intro:
        st.markdown(f"**What is this?** {intro}")

    st.divider()
    _render_tour(data)

    st.divider()
    _render_glossary(data)
