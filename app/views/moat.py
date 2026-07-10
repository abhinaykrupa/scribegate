"""app/views/moat.py — Data moat page (v0.3).

Surfaces scribegate.moat's generation model in the UI: the moat curve
(overall aggregate + auto-accept rate per golden generation), headline
metrics, a plain-text "how the loop works" strip, a per-generation table,
and a "Promote pending candidates" button that closes the loop live —
correction recorded (Review queue) -> clinician-signed promotion (this
button) -> new golden generation -> benchmark re-run.

Named `moat.py` inside app/views/ — this is a different dotted path
(app.views.moat) than scribegate.moat, so `from scribegate import moat`
below always resolves scribegate's moat.py, never this module itself
(same convention as app/views/analytics.py's `from scribegate import
analytics`).

100% synthetic bundled data (data/golden_notes/, data/results/): every
number on this page is a demo-on-synthetic-data illustration of the loop,
not a claim about production customer data.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from app.common import RESULTS_DIR
from scribegate import corrections
from scribegate import moat as moat_module

_PENDING_GUARD_KEY = "moat_promote_clicked"


def _manifest_path(generation: int) -> str:
    return os.path.join(
        RESULTS_DIR,
        corrections.GOLDEN_GENERATIONS_DIRNAME,
        f"gen_{generation}",
        corrections.GENERATION_MANIFEST_NAME,
    )


def _pending_correction_ids() -> set[str]:
    """Correction ids recorded (candidate_golden.jsonl) that have never
    appeared in any promoted generation's manifest source_corrections —
    i.e. corrections a reviewer made that are not yet reflected in the
    golden set. Empty for the bundled demo data (its 3 seeded corrections
    are already folded into gen_1), so the button starts in its graceful
    empty state out of the box; adding a new correction via the Review
    queue page is what makes this non-empty."""
    all_ids = {r["correction_id"] for r in corrections.list_corrections()}
    if not all_ids:
        return set()

    promoted_ids: set[str] = set()
    for gen in corrections.list_generations():
        path = _manifest_path(gen)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        source = manifest.get("source_corrections", {})
        if isinstance(source, dict):
            for ids in source.values():
                promoted_ids.update(ids)

    return all_ids - promoted_ids


def _generation_rows(metrics: dict) -> list[dict]:
    """Build the per-generation table rows, gen-0 first, using
    moat_metrics()'s cached gen0/gen-N benchmark summaries so this never
    re-derives anything moat.py already computed."""
    gen0_summary = moat_module.rebenchmark_generation(0)
    rows = [
        {
            "generation": 0,
            "ts": None,
            "reviewer": None,
            "promoted_note_ids": "",
            "aggregate": gen0_summary.get("overall_aggregate"),
            "auto_accept_rate": gen0_summary.get("auto_accept_rate"),
            "delta": None,
        }
    ]
    prev_aggregate = gen0_summary.get("overall_aggregate")
    for gen in metrics.get("generations", []):
        summary = gen.get("benchmark_summary", {})
        aggregate = summary.get("overall_aggregate")
        delta = None
        if aggregate is not None and prev_aggregate is not None:
            delta = aggregate - prev_aggregate
        rows.append(
            {
                "generation": gen.get("generation"),
                "ts": gen.get("ts"),
                "reviewer": gen.get("reviewer"),
                "promoted_note_ids": ", ".join(gen.get("promoted_this_generation", [])),
                "aggregate": aggregate,
                "auto_accept_rate": summary.get("auto_accept_rate"),
                "delta": delta,
            }
        )
        prev_aggregate = aggregate if aggregate is not None else prev_aggregate
    return rows


def _render_moat_curve(rows: list[dict]) -> None:
    st.subheader("Moat curve — aggregate + auto-accept rate per golden generation")
    chart_df = pd.DataFrame(
        [
            {
                "generation": r["generation"],
                "aggregate": r["aggregate"],
                "auto_accept_rate": r["auto_accept_rate"],
            }
            for r in rows
            if r["aggregate"] is not None
        ]
    ).set_index("generation")
    if chart_df.empty:
        st.caption("No generations to chart yet.")
        return
    st.line_chart(chart_df)


def _render_headline_cards(metrics: dict, rows: list[dict]) -> None:
    golden_set = metrics.get("golden_set", {})
    golden_set_size = golden_set.get("base_count", 0) + golden_set.get("cumulative_promoted_notes", 0)
    corrections_promoted = metrics.get("corrections_recorded_total", 0)

    lift = None
    if len(rows) >= 2 and rows[0]["aggregate"] is not None and rows[-1]["aggregate"] is not None:
        lift = rows[-1]["aggregate"] - rows[0]["aggregate"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Golden-set size", golden_set_size)
    with c2:
        st.metric("Corrections promoted", corrections_promoted)
    with c3:
        st.metric("Benchmark lift since gen-0", f"{lift:+.3f}" if lift is not None else "—")


def _render_loop_strip() -> None:
    st.subheader("How the loop works")
    steps = [
        "1. Correction recorded — a reviewer edits a SOAP line on the Review queue page; the correction is appended to candidate_golden.jsonl.",
        "2. Clinician-signed promotion — a reviewer promotes all pending corrections into a new golden generation overlay (the button below).",
        "3. New golden generation — data/results/golden_generations/gen_{N}/ is written, overlaying only the notes that changed on top of gen-0.",
        "4. Benchmark re-run — every stored result is re-judged against the new golden set, producing a fresh aggregate + auto-accept rate for the moat curve above.",
    ]
    for step in steps:
        st.markdown(f"- {step}")


def _render_generations_table(rows: list[dict]) -> None:
    st.subheader("Generations")
    df = pd.DataFrame(rows)
    # Arrow-safe: ts/reviewer/delta can be None for gen-0 — cast to str so the
    # dataframe never mixes None with numeric/str types in the same column
    # (see app/views/analytics.py's assumptions_df fix for the same issue).
    df["ts"] = df["ts"].astype(str)
    df["reviewer"] = df["reviewer"].astype(str)
    df["delta"] = df["delta"].apply(lambda v: f"{v:+.3f}" if v is not None else "—")
    df["aggregate"] = df["aggregate"].apply(lambda v: f"{v:.3f}" if v is not None else "—")
    df["auto_accept_rate"] = df["auto_accept_rate"].apply(lambda v: f"{v * 100:.1f}%" if v is not None else "—")
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_promote_button() -> None:
    st.subheader("Promote pending candidates")

    pending_ids = _pending_correction_ids()

    if not pending_ids:
        st.session_state[_PENDING_GUARD_KEY] = False
        st.success(
            "No corrections pending promotion — the golden set already reflects every "
            "recorded correction. Make a new correction on the Review queue page to see "
            "this button do something."
        )
        return

    st.info(f"{len(pending_ids)} correction(s) recorded but not yet promoted into a golden generation.")

    if _PENDING_GUARD_KEY not in st.session_state:
        st.session_state[_PENDING_GUARD_KEY] = False

    disabled = st.session_state[_PENDING_GUARD_KEY]
    if st.button("Promote pending candidates", disabled=disabled, key="moat_promote_button"):
        st.session_state[_PENDING_GUARD_KEY] = True
        try:
            manifest = corrections.promote_all_candidates(reviewer="demo-user")
            gen_n = manifest["gen"]
            moat_module.rebenchmark_generation(gen_n)
        except ValueError as exc:
            st.session_state[_PENDING_GUARD_KEY] = False
            st.error(f"Could not promote pending candidates: {exc}")
        else:
            st.success(f"Promoted generation {gen_n} and re-benchmarked.")
            st.rerun()


def _ensure_moat_seed() -> None:
    """Self-seed the demo moat curve on cold start (hosted deploys have an
    empty gitignored results dir) — same pattern as ensure_results /
    calibration's report self-seed. simulate_moat_demo() is idempotent, so
    this is a no-op once gen_1 exists."""
    if corrections.list_generations():
        return
    with st.spinner("First run — seeding the correction-loop demo..."):
        moat_module.simulate_moat_demo()


def render() -> None:
    st.header("Data moat")
    st.caption(
        "**Demo on synthetic data.** Every generation/correction below comes from the "
        "bundled synthetic fixtures — no PHI, no production customer data."
    )

    try:
        _ensure_moat_seed()
    except Exception as exc:  # noqa: BLE001 — never block page render
        st.error(f"Moat demo seeding failed: {exc}\n\nRun manually: python -m scribegate.moat --seed-demo")

    metrics = moat_module.moat_metrics()
    rows = _generation_rows(metrics)

    _render_headline_cards(metrics, rows)
    st.divider()
    _render_moat_curve(rows)
    st.divider()
    _render_loop_strip()
    st.divider()
    _render_generations_table(rows)
    st.divider()
    _render_promote_button()
    st.divider()

    st.info(
        "**Why this is the moat:** the correction loop above is the entire strategic "
        "argument — every promoted correction makes the golden set (and therefore the "
        "benchmark and the auto-accept rate) a little better, and that improvement "
        "compounds generation over generation. A competitor starting today starts at "
        "gen-0; every month ScribeGate is in market with real reviewer corrections "
        "flowing through this loop widens the gap. (Demo on synthetic data — the "
        "mechanism is real, the numbers above are illustrative.)"
    )
