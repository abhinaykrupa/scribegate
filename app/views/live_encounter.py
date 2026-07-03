"""app/views/live_encounter.py — Live encounter capture page (v0.2).

Three-step flow:
  1. Consent gate — mic cannot arm until both provider + patient attestations
     are checked; a consent event is logged exactly once per session.
  2. Capture — speaker-toggled utterance capture (mic if available, else a
     text-area fallback) building a running transcript.
  3. Pipeline — generate -> normalize -> judge (reference-free) -> route,
     rendered with the same provenance span-highlight view as the Provenance
     page, and saved to data/results/live/{transcript_id}.json.
"""

from __future__ import annotations

import dataclasses
import json
import os
import uuid

import streamlit as st

from app.common import LIVE_RESULTS_DIR, _render_note_and_transcript, append_consent_event, load_consent_copy
from scribegate import normalizer
from scribegate.generator import generate_note
from scribegate.judge import judge_note_reference_free
from scribegate.router import decide as router_decide

try:
    from streamlit_mic_recorder import speech_to_text

    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False

VISIT_TYPES = [
    "comprehensive_exam",
    "glaucoma_followup",
    "cataract_postop",
    "contact_lens_fitting",
]


def _violation_to_dict(v):
    if dataclasses.is_dataclass(v):
        return dataclasses.asdict(v)
    if isinstance(v, dict):
        return v
    return {"value": str(v)}


def _init_session_state() -> None:
    if "live_step" not in st.session_state:
        st.session_state["live_step"] = 1
    if "consent_recorded" not in st.session_state:
        st.session_state["consent_recorded"] = False
    if "live_transcript_lines" not in st.session_state:
        st.session_state["live_transcript_lines"] = []


def _render_consent_gate() -> None:
    data = load_consent_copy()
    gate = data.get("consent_gate", {})

    st.subheader(gate.get("title", "Recording consent"))
    st.markdown(gate.get("explanation", ""))

    selector = gate.get("state_selector", {})
    two_party_states = selector.get("two_party_states", []) or []
    options = [f"{s.get('code')} - {s.get('name')}" for s in two_party_states]
    options = options + ["Other / not listed"]
    choice = st.selectbox(selector.get("prompt", "Select state"), options=options)
    state_code = choice.split(" - ")[0] if " - " in choice else "OTHER"

    is_two_party = any(choice.startswith(s.get("code", "")) for s in two_party_states)
    if is_two_party:
        st.warning(selector.get("disclaimer", ""))

    attestations = gate.get("attestations", {})
    provider_label = (attestations.get("provider", {}) or {}).get("label", "Provider consents.")
    patient_label = (attestations.get("patient", {}) or {}).get("label", "Patient consents.")

    provider_checked = st.checkbox(provider_label, key="consent_provider_checkbox")
    patient_checked = st.checkbox(patient_label, key="consent_patient_checkbox")

    both_checked = provider_checked and patient_checked

    st.button("Start recording", disabled=not both_checked, key="start_recording_placeholder")

    if not both_checked:
        st.warning(gate.get("blocked_message", "Both attestations are required before recording can start."))

    if both_checked and not st.session_state.get("consent_recorded"):
        append_consent_event(state_code, provider_checked, patient_checked)
        st.session_state["consent_recorded"] = True
        st.session_state["live_step"] = 2
        st.success(gate.get("logged_notice", "Consent recorded."))
        st.rerun()
    elif both_checked and st.session_state.get("consent_recorded"):
        st.success(gate.get("logged_notice", "Consent recorded."))


def _render_capture_step() -> None:
    data = load_consent_copy()
    capture_ui = data.get("capture_ui", {})
    speaker_toggle = capture_ui.get("speaker_toggle", {})

    st.subheader(speaker_toggle.get("header", "Current speaker"))
    speaker = st.radio(
        speaker_toggle.get("header", "Current speaker"),
        options=[
            speaker_toggle.get("provider_label", "Provider"),
            speaker_toggle.get("patient_label", "Patient"),
        ],
        label_visibility="collapsed",
        key="live_speaker_toggle",
    )
    is_provider = speaker == speaker_toggle.get("provider_label", "Provider")

    captured_text = None
    if MIC_AVAILABLE:
        try:
            captured_text = speech_to_text(
                language="en", start_prompt="Start", stop_prompt="Stop", key="mic_recorder"
            )
        except Exception:
            captured_text = None

    if captured_text:
        prefix = "DOCTOR: " if is_provider else "PATIENT: "
        st.session_state["live_transcript_lines"].append(f"{prefix}{captured_text}")

    if not MIC_AVAILABLE:
        with st.form(key="live_text_capture_form", clear_on_submit=True):
            text_input = st.text_area("Utterance text", key="live_text_input")
            submitted = st.form_submit_button("Add to transcript")
        if submitted and text_input and text_input.strip():
            prefix = "DOCTOR: " if is_provider else "PATIENT: "
            st.session_state["live_transcript_lines"].append(f"{prefix}{text_input.strip()}")

    st.caption(capture_ui.get("quality_note", ""))

    btn_cols = st.columns(2)
    with btn_cols[0]:
        if st.button("Undo last", key="live_undo_last"):
            if st.session_state["live_transcript_lines"]:
                st.session_state["live_transcript_lines"].pop()
                st.rerun()
    with btn_cols[1]:
        if st.button("Clear", key="live_clear"):
            st.session_state["live_transcript_lines"] = []
            st.rerun()

    st.markdown(f"**{capture_ui.get('live_transcript_header', 'Live transcript')}**")
    running_text = "\n".join(st.session_state["live_transcript_lines"])
    st.text(running_text if running_text else "(no utterances captured yet)")

    st.divider()
    if st.button("Proceed to pipeline", key="live_proceed_to_pipeline"):
        st.session_state["live_step"] = 3
        st.rerun()


def _render_pipeline_step() -> None:
    st.subheader("Generate + evaluate")

    visit_type = st.selectbox("Visit type", options=VISIT_TYPES, key="live_visit_type")

    if st.button("Generate note", key="live_generate_note"):
        lines = st.session_state.get("live_transcript_lines", [])
        header = "# SYNTHETIC TRANSCRIPT — no PHI. Live encounter capture.\n"
        transcript_text = header + "\n".join(lines)
        transcript_id = f"live_{uuid.uuid4().hex[:8]}"

        generated_note = generate_note(transcript_text, transcript_id, visit_type)

        try:
            violations = normalizer.check_note(generated_note, transcript_text)
        except Exception:
            violations = []

        judge_result = judge_note_reference_free(generated_note, transcript_text)
        decision = router_decide(judge_result, violations)

        st.session_state["live_last_run"] = {
            "transcript_id": transcript_id,
            "visit_type": visit_type,
            "generated_note": generated_note,
            "judge_result": judge_result,
            "violations": violations,
            "route": decision.route,
            "decision_reasons": decision.reasons,
            "transcript_text": transcript_text,
        }

        os.makedirs(LIVE_RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(LIVE_RESULTS_DIR, f"{transcript_id}.json")
        serializable = {
            "transcript_id": transcript_id,
            "visit_type": visit_type,
            "generated_note": generated_note,
            "judge_result": judge_result,
            "violations": [_violation_to_dict(v) for v in violations],
            "route": decision.route,
            "decision_reasons": decision.reasons,
            "transcript_text": transcript_text,
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(serializable, fh, indent=2)

    last_run = st.session_state.get("live_last_run")
    if not last_run:
        st.caption("No note generated yet in this session.")
        return

    jr = last_run["judge_result"]
    scores = jr.get("scores", {})
    rationales = jr.get("rationales", {})

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

    st.markdown(f"**Route:** `{last_run['route']}`")
    st.markdown("**Decision reasons**")
    for reason in last_run.get("decision_reasons", []):
        st.markdown(f"- {reason}")

    violations = last_run.get("violations", [])
    if violations:
        st.markdown("**Violations**")
        for v in violations:
            v_dict = _violation_to_dict(v)
            st.markdown(f"- `{v_dict.get('code')}` ({v_dict.get('severity')}): {v_dict.get('message')}")

    st.divider()
    _render_note_and_transcript(
        last_run["transcript_id"],
        last_run["transcript_text"],
        last_run["generated_note"],
        state_prefix="live",
    )


def render() -> None:
    st.header("Live encounter capture")

    _init_session_state()

    step = st.session_state["live_step"]

    st.caption(f"Step {step} of 3")

    if step == 1 or not st.session_state.get("consent_recorded"):
        _render_consent_gate()
        return

    if step == 2:
        _render_capture_step()
        return

    _render_pipeline_step()
