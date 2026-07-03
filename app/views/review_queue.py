"""app/views/review_queue.py — Review queue page (v0.2).

v0.1's worst-first review queue (approve/reject with a session-state guard)
plus: per-line inline editing that records corrections via
scribegate.corrections, and a "Candidate golden" tab per transcript built
from those corrections.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.common import SECTION_LABELS, SECTION_ORDER, append_decision, load_results
from scribegate import corrections


def _render_review_tab(tid: str, r: dict) -> None:
    jr = r.get("judge_result", {})
    scores = jr.get("scores", {})
    rationales = jr.get("rationales", {})
    violations = r.get("violations", [])
    decision_reasons = r.get("decision_reasons", [])

    cols = st.columns(4)
    dims = [
        ("Completeness", "completeness"),
        ("Hallucination", "hallucination"),
        ("Coding Plausibility", "coding_plausibility"),
        ("Terminology", "terminology"),
    ]
    for col, (label, key) in zip(cols, dims):
        with col:
            st.metric(label, scores.get(key, "—"))
            reason = rationales.get(key)
            if reason:
                st.caption(reason)

    st.markdown("**Violations**")
    if violations:
        vdf = pd.DataFrame(
            [
                {
                    "code": v.get("code"),
                    "severity": v.get("severity"),
                    "message": v.get("message"),
                }
                for v in violations
            ]
        )
        st.dataframe(vdf, use_container_width=True, hide_index=True)
    else:
        st.caption("None.")

    st.markdown("**Decision reasons (router)**")
    for reason in decision_reasons:
        st.markdown(f"- {reason}")

    # Guard: once a decision is recorded for this transcript in this
    # session, the buttons are disabled instead of re-armed — this is
    # what stops a double-click (or a click-Approve-then-Reject) from
    # writing a second, contradictory line to decision_log.jsonl for
    # the same transcript_id in the same session.
    already = st.session_state["reviewed"].get(tid)
    if already:
        st.caption(f"Decision recorded: **{already}**. Buttons disabled for this session.")
        btn_cols = st.columns(2)
        with btn_cols[0]:
            st.button("Approve", key=f"approve_{tid}", disabled=True)
        with btn_cols[1]:
            st.button("Reject", key=f"reject_{tid}", disabled=True)
    else:
        btn_cols = st.columns(2)
        with btn_cols[0]:
            if st.button("Approve", key=f"approve_{tid}"):
                append_decision(tid, "approved")
                st.session_state["reviewed"][tid] = "approved"
                st.success(f"Recorded: {tid} approved by demo-user.")
                st.rerun()
        with btn_cols[1]:
            if st.button("Reject", key=f"reject_{tid}"):
                append_decision(tid, "rejected")
                st.session_state["reviewed"][tid] = "rejected"
                st.error(f"Recorded: {tid} rejected by demo-user.")
                st.rerun()

    st.divider()
    st.markdown("**Edit SOAP lines**")

    generated_note = r.get("generated_note", {})
    soap = generated_note.get("soap", {})

    for section in SECTION_ORDER:
        lines = soap.get(section, [])
        if not lines:
            continue
        st.markdown(f"_{section} — {SECTION_LABELS[section]}_")
        for idx, line in enumerate(lines):
            current_text = line.get("text", "")
            edit_key = f"editing_{tid}_{section}_{idx}"
            saved_key = f"correction_saved_{tid}_{section}_{idx}"
            original_key = f"orig_text_{tid}_{section}_{idx}"

            if saved_key not in st.session_state.get("correction_saved", {}):
                st.session_state.setdefault("correction_saved", {})

            row_cols = st.columns([4, 1])
            with row_cols[0]:
                st.write(current_text)
            with row_cols[1]:
                if st.button("Edit", key=f"edit_btn_{tid}_{section}_{idx}"):
                    st.session_state[edit_key] = True
                    st.session_state[original_key] = current_text

            already_saved = st.session_state["correction_saved"].get(f"{tid}_{section}_{idx}")
            if already_saved:
                st.caption(f"Correction saved: {already_saved}")

            if st.session_state.get(edit_key) and not already_saved:
                textarea_key = f"textarea_{tid}_{section}_{idx}"
                corrected_text = st.text_area(
                    "Corrected text",
                    value=st.session_state.get(original_key, current_text),
                    key=textarea_key,
                )
                if st.button("Save", key=f"save_{tid}_{section}_{idx}"):
                    original_text = st.session_state.get(original_key, current_text)
                    try:
                        corrections.record_correction(
                            transcript_id=tid,
                            section=section,
                            line_index=idx,
                            original_text=original_text,
                            corrected_text=corrected_text,
                            reviewer="demo-user",
                        )
                    except ValueError as exc:
                        st.error(
                            f"Could not save correction: {exc} — someone else may have "
                            "changed this line — refresh and try again."
                        )
                    else:
                        diff = corrections.diff_lines(original_text, corrected_text)
                        st.session_state["correction_saved"][f"{tid}_{section}_{idx}"] = diff
                        st.session_state[edit_key] = False
                        st.code(diff)
                        st.rerun()


def _render_candidate_golden_tab(tid: str) -> None:
    candidate = corrections.build_candidate_golden(tid)
    if candidate is None:
        st.caption("No corrections yet.")
        return

    corr_records = corrections.list_corrections(tid)
    corr_by_id = {c["correction_id"]: c for c in corr_records}

    st.markdown("**Candidate golden note — diffs from original generated note**")
    for correction_id in candidate.get("source_corrections", []):
        c = corr_by_id.get(correction_id)
        if not c:
            continue
        section = c.get("section")
        line_index = c.get("line_index")
        original_text = c.get("original_text", "")
        corrected_text = c.get("corrected_text", "")
        diff = corrections.diff_lines(original_text, corrected_text)
        st.markdown(f"- **{section}[{line_index}]** (reviewer {c.get('reviewer')}, {c.get('ts')})")
        st.code(diff)


def render() -> None:
    st.header("Review queue")

    results = load_results()

    queue = [
        (tid, r)
        for tid, r in results.items()
        if r.get("route") in ("review", "regenerate")
    ]
    if not queue:
        if not results:
            st.warning(
                "No results found yet. Run the pipeline first:\n\n"
                "```\npython -m scribegate.cli run --all\n```"
            )
        else:
            st.success("Nothing in the queue — every transcript auto-accepted.")
        return

    # Worst-first: regenerate before review, then by ascending aggregate.
    route_rank = {"regenerate": 0, "review": 1}
    queue.sort(
        key=lambda item: (
            route_rank.get(item[1].get("route"), 2),
            item[1].get("judge_result", {}).get("aggregate", 0.0),
        )
    )

    st.caption(f"{len(queue)} transcript(s) awaiting review, worst-first.")

    if "reviewed" not in st.session_state:
        st.session_state["reviewed"] = {}  # transcript_id -> "approved"/"rejected"
    if "correction_saved" not in st.session_state:
        st.session_state["correction_saved"] = {}

    for tid, r in queue:
        jr = r.get("judge_result", {})
        aggregate = jr.get("aggregate", 0.0)
        route = r.get("route", "")
        already = st.session_state["reviewed"].get(tid)
        badge = f" — **{already.upper()}**" if already else ""
        with st.expander(f"[{route}] {tid} — aggregate {aggregate:.3f}{badge}"):
            tabs = st.tabs(["Review", "Candidate golden"])
            with tabs[0]:
                _render_review_tab(tid, r)
            with tabs[1]:
                _render_candidate_golden_tab(tid)
