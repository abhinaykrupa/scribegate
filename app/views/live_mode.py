"""app/views/live_mode.py — Live mode page (v0.4).

The one page in ScribeGate that calls a real, paid Claude model
(scribegate.live.run_live_note) instead of the free offline mock every
other page runs on. Three states:

  1. Unavailable (no ANTHROPIC_API_KEY configured, or today's budget is
     already exhausted) — an honest banner explains why, and the rest of
     the page renders a bundled SAMPLE saved run (data/live_demo_sample.json,
     a mock-generated example shaped exactly like run_live_note()'s real
     return value) in a disabled/preview state, so the page is never empty.
  2. Available but locked — a passcode gate (constant-time compare via
     live.check_passcode; a fixed 1s sleep on a wrong guess, no lockout).
  3. Unlocked — transcript + drafter-model selectors, a "Run live" button
     (client-side session guard: max 5 runs/session, one run at a time, on
     top of live.py's own server-side daily budget enforcement), and the
     rendered result: generated note with per-line span-confidence badges,
     sampled judge scores + CI95 + route, and the per-note cost breakdown.

All copy comes from app.common.ui_copy() (specs/ui_copy.yaml's `live_mode`
block) — nothing here is hardcoded prose.
"""

from __future__ import annotations

import json
import os
import time

import streamlit as st

from app.common import DATA_DIR, SECTION_LABELS, SECTION_ORDER, page_header, ui_copy
from scribegate import costs, economics, live

SAMPLE_PATH = os.path.join(DATA_DIR, "live_demo_sample.json")

UNLOCKED_KEY = "live_mode_unlocked"
RUN_COUNT_KEY = "live_mode_run_count"
RUNNING_KEY = "live_mode_running"
LAST_RESULT_KEY = "live_mode_last_result"
RUN_CAP = 5
FAILED_PASSCODE_SLEEP_SECONDS = 1.0

DRAFTER_MODEL_OPTIONS = ["claude-haiku-4-5", "claude-sonnet-4-5"]


@st.cache_data(ttl=5)
def _load_sample() -> dict:
    with open(SAMPLE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _render_passcode_gate(config) -> None:
    st.subheader("Unlock live mode")
    st.caption("A passcode is required because every run below spends real money on a real model call.")

    entered = st.text_input("Passcode", type="password", key="live_mode_passcode_input")
    if st.button("Unlock", key="live_mode_unlock_button"):
        if live.check_passcode(entered, config):
            st.session_state[UNLOCKED_KEY] = True
            st.rerun()
        else:
            # Fixed delay on a wrong guess — no lockout tracking needed for
            # a demo, but a bare-instant reject invites naive brute-forcing.
            time.sleep(FAILED_PASSCODE_SLEEP_SECONDS)
            st.error("Incorrect passcode.")


def _render_budget_banner(config) -> None:
    data = ui_copy()
    live_copy = data.get("live_mode") or {}
    remaining = costs.budget_remaining(config)

    if remaining <= 0:
        st.warning(
            live_copy.get(
                "budget_banner_exhausted",
                "Live budget spent. Further runs fall back to the free offline mock automatically.",
            )
        )
        return

    # A representative single-run cost estimate for the banner — reuses
    # economics.py's own documented per-note assumptions (haiku tier) rather
    # than duplicating a token/pricing estimate here.
    per_run = economics.cost_per_note("haiku").get("total_usd", 0.0)
    template = live_copy.get(
        "budget_banner_remaining",
        "Live budget: ${remaining} of ${total} left. Each run spends about ${per_run}.",
    )
    message = template.format(
        remaining=f"{remaining:.2f}", total=f"{config.daily_budget_usd:.2f}", per_run=f"{per_run:.4f}"
    )
    st.info(message)


def _render_note_with_confidence(soap: dict) -> None:
    data = ui_copy()
    honesty_note = (data.get("live_mode") or {}).get("span_confidence_note", "")
    if honesty_note:
        st.caption(honesty_note)

    for section in SECTION_ORDER:
        lines = soap.get(section, [])
        if not lines:
            continue
        st.markdown(f"**{section} — {SECTION_LABELS[section]}**")
        for line in lines:
            text = line.get("text", "")
            confidence = line.get("span_confidence", "low")
            cols = st.columns([5, 1])
            with cols[0]:
                st.write(text)
            with cols[1]:
                if confidence == "high":
                    st.badge("high confidence", color="green")
                else:
                    st.badge("low confidence", color="orange")


def _render_judge_and_route(result: dict) -> None:
    sampled = result.get("judge_sampled") or {}
    route_result = result.get("route_result") or {}
    mean_scores = sampled.get("mean_scores") or {}

    st.markdown("#### Sampled judge scores")
    dims = [
        ("Completeness", "completeness"),
        ("Hallucination (made-up detail)", "hallucination"),
        ("Billing plausibility (coding)", "coding_plausibility"),
        ("Terminology", "terminology"),
    ]
    cols = st.columns(4)
    for col, (label, key) in zip(cols, dims):
        with col:
            val = mean_scores.get(key)
            st.metric(label, f"{val:.2f}" if val is not None else "—")

    ci95 = sampled.get("ci95") or [None, None]
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Aggregate mean", f"{sampled.get('aggregate_mean', 0.0):.3f}")
    with m2:
        ci_text = f"[{ci95[0]:.3f}, {ci95[1]:.3f}]" if ci95[0] is not None else "—"
        st.metric("The cautious read (CI95 lower bound)", ci_text)
    with m3:
        st.metric("Route", result.get("route") or route_result.get("route") or "—")

    for reason in route_result.get("reasons", []) or []:
        st.caption(f"- {reason}")


def _render_cost_breakdown(result: dict) -> None:
    st.markdown("#### Cost per note")
    breakdown = result.get("cost_breakdown") or {}
    drafting = breakdown.get("drafting") or {}
    judging = breakdown.get("judging") or {}

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Drafting", f"${drafting.get('usd', 0.0):.4f}")
    with c2:
        st.metric("Judging", f"${judging.get('usd', 0.0):.4f}")
    with c3:
        st.metric("Total per note", f"${breakdown.get('total_usd', 0.0):.4f}")


def _render_run_result(result: dict, *, sample: bool) -> None:
    if sample:
        st.info(result.get("sample_note") or "Showing a bundled SAMPLE run (mock-generated).")

    if result.get("error"):
        st.error(result["error"])
        return

    note = result.get("generated_note") or {}
    soap = note.get("soap") or {}

    st.markdown("#### Generated note")
    _render_note_with_confidence(soap)
    st.divider()
    _render_judge_and_route(result)
    st.divider()
    _render_cost_breakdown(result)


def _render_unlocked(config) -> None:
    st.success("Live mode unlocked for this session.")

    from scribegate import cli

    transcript_ids = cli.discover_transcript_ids()
    transcript_id = st.selectbox("Transcript", transcript_ids, key="live_mode_transcript_select")
    drafter_model = st.selectbox(
        "Drafter model",
        options=DRAFTER_MODEL_OPTIONS,
        index=0,
        key="live_mode_drafter_model_select",
        help="haiku is the cost-efficient default; sonnet trades cost for a stronger drafter.",
    )

    run_count = st.session_state.get(RUN_COUNT_KEY, 0)
    running = st.session_state.get(RUNNING_KEY, False)
    at_cap = run_count >= RUN_CAP

    if at_cap:
        st.warning(
            f"Session run cap reached ({RUN_CAP} runs). Refresh the page to reset the client-side "
            "cap — the server-side daily budget below is tracked separately and is not affected."
        )

    if st.button("Run live", key="live_mode_run_button", disabled=at_cap or running):
        st.session_state[RUNNING_KEY] = True
        try:
            with st.spinner("Calling the real model — drafting, then sampled judging..."):
                result = live.run_live_note(transcript_id, drafter_model=drafter_model, config=config)
            st.session_state[LAST_RESULT_KEY] = result
            st.session_state[RUN_COUNT_KEY] = run_count + 1
        finally:
            st.session_state[RUNNING_KEY] = False

    last_result = st.session_state.get(LAST_RESULT_KEY)
    if last_result:
        _render_run_result(last_result, sample=False)
    else:
        st.caption("No live run yet this session — pick a transcript and click Run live.")

    st.divider()
    _render_budget_banner(config)


def render() -> None:
    page_header("live_mode")

    data = ui_copy()
    live_copy = data.get("live_mode") or {}
    passcode_note = live_copy.get("passcode_gate", "")
    if passcode_note:
        st.info(passcode_note)

    config = live.LiveConfig.from_env()
    available, reason = live.live_available(config)

    if not available:
        st.warning(f"Live mode is not available right now: {reason}")
        st.caption(
            "Everything below is a bundled SAMPLE saved run (mock-generated, not a real API call) "
            "so this page is never empty while live mode is unavailable."
        )
        try:
            sample = _load_sample()
        except OSError as exc:
            st.error(f"Could not load the bundled sample run: {exc}")
            return
        _render_run_result(sample, sample=True)
        return

    if not st.session_state.get(UNLOCKED_KEY):
        _render_passcode_gate(config)
        return

    _render_unlocked(config)
