"""app/views/provenance.py — Provenance note view page (v0.2).

v0.1's provenance view, unchanged, plus an "Export audit dossier" button per
transcript that builds a self-contained JSON + Markdown dossier
(scribegate.audit.export_dossier) into a tempfile.mkdtemp() directory (never
under repo data/ or app/) and offers both as st.download_button downloads.
"""

from __future__ import annotations

import tempfile

import streamlit as st

from app.common import (
    _render_note_and_transcript,
    load_golden_note,
    load_results,
    load_transcript_text,
    page_header,
)
from scribegate import audit


def _render_export_dossier_button(transcript_id: str) -> None:
    if st.button("Export audit dossier", key=f"export_dossier_{transcript_id}"):
        try:
            json_path, md_path = audit.export_dossier(transcript_id, tempfile.mkdtemp())
            with open(json_path, "rb") as fh:
                json_bytes = fh.read()
            with open(md_path, "rb") as fh:
                md_bytes = fh.read()
        except Exception as exc:  # defensive — never crash the page on export failure
            st.error(f"Could not export dossier: {exc}")
            return

        st.success("Dossier built. Download below.")
        dl_cols = st.columns(2)
        with dl_cols[0]:
            st.download_button(
                "Download dossier (JSON)",
                data=json_bytes,
                file_name=f"{transcript_id}_dossier.json",
                mime="application/json",
                key=f"dl_json_{transcript_id}",
            )
        with dl_cols[1]:
            st.download_button(
                "Download dossier (Markdown)",
                data=md_bytes,
                file_name=f"{transcript_id}_dossier.md",
                mime="text/markdown",
                key=f"dl_md_{transcript_id}",
            )


def render() -> None:
    page_header("provenance")

    results = load_results()
    if not results:
        st.warning(
            "No results found yet. Run the pipeline first:\n\n"
            "```\npython -m scribegate.cli run --all\n```"
        )
        return

    transcript_ids = sorted(results.keys())
    transcript_id = st.selectbox("Transcript", transcript_ids)

    result = results[transcript_id]
    transcript_text = load_transcript_text(transcript_id)
    if transcript_text is None:
        st.error(f"Transcript file not found for {transcript_id} — expected data/transcripts/{transcript_id}.txt")
        return

    show_golden = st.toggle(
        "Show GOLDEN note instead of generated",
        value=False,
        help="Inspect the reference/golden note's own spans — useful for a "
        "clinician to critique golden content, not just generator output.",
    )

    if show_golden:
        golden = load_golden_note(transcript_id)
        if golden is None:
            st.error(f"Golden note not found for {transcript_id}.")
            return
        st.caption(f"Viewing GOLDEN note for **{transcript_id}** ({result.get('visit_type', '')})")
        _render_note_and_transcript(transcript_id, transcript_text, golden, state_prefix="golden")
    else:
        generated = result.get("generated_note", {})
        jr = result.get("judge_result", {})
        st.caption(
            f"Viewing GENERATED note for **{transcript_id}** "
            f"({result.get('visit_type', '')}) — aggregate "
            f"{jr.get('aggregate', 0.0):.3f}, route `{result.get('route', '')}`"
        )
        _render_note_and_transcript(transcript_id, transcript_text, generated, state_prefix="generated")

    st.divider()
    _render_export_dossier_button(transcript_id)
