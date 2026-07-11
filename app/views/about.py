"""app/views/about.py — About page (v0.2).

Renders the "Why" paragraph and "Architecture" ASCII diagram straight from
README.md, a GitHub repo link, and PRODUCTION_PATH.md / DEMO_SCRIPT.md in
expanders — all read fresh from disk (cached) rather than duplicated as
copy in this module.
"""

from __future__ import annotations

import os
import re

import streamlit as st

from app.common import REPO_ROOT, page_header

README_PATH = os.path.join(REPO_ROOT, "README.md")
PRODUCTION_PATH_MD = os.path.join(REPO_ROOT, "PRODUCTION_PATH.md")
DEMO_SCRIPT_MD = os.path.join(REPO_ROOT, "DEMO_SCRIPT.md")


@st.cache_data(ttl=5)
def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _extract_section(markdown_text: str, heading: str) -> str:
    """Return the body text of the first `## {heading}` section (up to the
    next `## ` heading or end of file), heading line excluded."""
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*$\n(.*?)(?=^##\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(markdown_text)
    return m.group(1).strip() if m else ""


def _extract_first_fenced_code_block(section_text: str) -> str:
    m = re.search(r"```(?:\w*)\n(.*?)```", section_text, re.DOTALL)
    return m.group(1).rstrip("\n") if m else ""


def render() -> None:
    page_header("about")

    readme_text = _read_file(README_PATH)

    why_text = _extract_section(readme_text, "Why")
    if why_text:
        st.subheader("Why")
        st.markdown(why_text)

    architecture_text = _extract_section(readme_text, "Architecture")
    diagram_text = _extract_first_fenced_code_block(architecture_text)
    if diagram_text:
        st.subheader("Architecture")
        st.code(diagram_text)

    st.subheader("Links")
    st.link_button("GitHub repo", "https://github.com/abhinaykrupa/scribegate")

    with st.expander("Production path (PRODUCTION_PATH.md)"):
        try:
            st.markdown(_read_file(PRODUCTION_PATH_MD))
        except OSError:
            st.caption("PRODUCTION_PATH.md not found.")

    with st.expander("Demo script (DEMO_SCRIPT.md)"):
        try:
            st.markdown(_read_file(DEMO_SCRIPT_MD))
        except OSError:
            st.caption("DEMO_SCRIPT.md not found.")
