"""moat.py (V1) — closes the correction loop: turns reviewer corrections into
a MEASURABLE, compounding improvement in the golden set ("the data moat,
demonstrated").

Built entirely on top of scribegate.corrections' generation model
(active_golden_dir / load_golden_note / promote_candidate /
promote_all_candidates): this module never writes golden overlays itself, it
only reads generation manifests, re-judges frozen generated notes against
each generation's resolved golden, and reports the trend.

Key functions:
    rebenchmark_generation(N) -> dict
        Re-run judge_note for every currently-stored data/results/{id}.json
        result against the golden resolved at generation N (holding the
        generated_note + violations fixed), producing a fresh
        benchmark-style summary (via scribegate.benchmark.compute_summary)
        and caching it at
        <results_dir>/golden_generations/gen_{N}/benchmark_summary.json.

    moat_metrics() -> dict
        Golden-set size, corrections recorded, per-generation benchmark
        summaries, and a moat_curve list of
        (generation, overall_aggregate, golden_count, corrections_count)
        tuples for UI charting (see app/ in a later milestone).

    simulate_moat_demo() -> dict
        Deterministic, script-callable demo: derives up to 3 REAL corrections
        from actual diffs between an already-generated note and its golden
        reference (judge.align_notes — never fabricated text), records them,
        promotes them as one generation, re-benchmarks before/after, and
        returns the comparison. Idempotent: if generations already exist
        (real or previously seeded), it reports the current curve instead of
        seeding again.

Usage: `python -m scribegate.moat` (prints moat_metrics as JSON) or
`python -m scribegate.moat --seed-demo` (seeds the demo, prints the
before/after curve).

stdlib + pyyaml only (via scribegate.router). No network.
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

from scribegate import corrections
from scribegate.benchmark import compute_summary
from scribegate.judge import align_notes, judge_note
from scribegate.router import decide

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_GOLDEN_DIR = _DATA_DIR / "golden_notes"  # gen-0, pristine
_TRANSCRIPT_DIR = _DATA_DIR / "transcripts"

BENCHMARK_SUMMARY_NAME = "benchmark_summary.json"
DEMO_REVIEWER = "moat_demo_seed"

# Matched (aligned) generated<->golden line pairs with an alignment ratio
# below this are "drifted enough" to be a plausible paraphrase-fix candidate
# for the demo. Pairs at/above this are close enough that "correcting" them
# would be cosmetic, not a meaningful review action.
_PARAPHRASE_RATIO_CEILING = 0.85


# ---------------------------------------------------------------------------
# Small local loaders (mirrors cli.py's, kept independent so moat.py never
# needs to import cli.py — avoids any risk of a cli<->corrections<->moat
# import cycle and keeps this module a pure read+report layer).
# ---------------------------------------------------------------------------

def _load_transcript_text(transcript_id: str) -> str:
    path = _TRANSCRIPT_DIR / f"{transcript_id}.txt"
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _load_pristine_golden(transcript_id: str) -> dict | None:
    path = _GOLDEN_DIR / f"{transcript_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _iter_current_results() -> list[dict]:
    """Every currently-stored data/results/{id}.json-shaped payload (under
    the active, possibly-sandboxed results dir), sorted by transcript_id."""
    results_dir = corrections._results_dir()
    out = []
    if not results_dir.exists():
        return out
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "transcript_id" in data and "generated_note" in data:
            out.append(data)
    return out


def _read_manifest(gen: int) -> dict:
    path = corrections._generation_dir(gen) / corrections.GENERATION_MANIFEST_NAME
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# rebenchmark_generation
# ---------------------------------------------------------------------------

def rebenchmark_generation(generation: int) -> dict:
    """Recompute judge_result for every currently-stored result against the
    golden resolved at `generation`, holding generated_note + violations
    fixed (frozen at whatever pipeline run produced them) — this isolates
    the one variable that promotion changes (golden content), so the
    resulting compute_summary() reflects exactly the effect of the
    generation's overlay, not a new pipeline run.

    Deterministic: calling this twice for the same generation against
    unchanged results/goldens yields byte-identical output (no timestamps or
    randomness involved).

    Stores the summary at
    <results_dir>/golden_generations/gen_{generation}/benchmark_summary.json
    (creating that directory if needed — this is the only case where a
    gen_0/ directory may exist without a generation.json manifest, since it
    is a benchmark-summary cache, not a promotion) and returns it.
    """
    rebuilt = []
    for data in _iter_current_results():
        transcript_id = data["transcript_id"]
        generated_note = data["generated_note"]
        violations = data.get("violations", [])
        transcript_text = _load_transcript_text(transcript_id)
        golden = corrections.load_golden_note(transcript_id, generation=generation)

        if golden is None:
            judge_result = data.get(
                "judge_result",
                {
                    "scores": {"completeness": 1, "hallucination": 1, "coding_plausibility": 1, "terminology": 1},
                    "aggregate": 0.0,
                    "rationales": {},
                },
            )
        else:
            judge_result = judge_note(generated_note, golden, transcript_text)

        route_decision = decide(judge_result, violations)
        rebuilt.append(
            {
                "transcript_id": transcript_id,
                "visit_type": data.get("visit_type"),
                "judge_result": judge_result,
                "route": route_decision.route,
                "violations": violations,
            }
        )

    summary = compute_summary(rebuilt)
    summary = dict(summary)
    summary["generation"] = generation
    summary["n_rebenchmarked"] = len(rebuilt)

    gen_dir = corrections._generation_dir(generation)
    gen_dir.mkdir(parents=True, exist_ok=True)
    out_path = gen_dir / BENCHMARK_SUMMARY_NAME
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=False)
        fh.write("\n")

    return summary


# ---------------------------------------------------------------------------
# moat_metrics
# ---------------------------------------------------------------------------

def moat_metrics() -> dict:
    """Golden-set size per generation (base gen-0 count + cumulative
    promoted-note overlays), total corrections recorded, per-generation
    benchmark summaries, and a moat_curve list of
    (generation, overall_aggregate, golden_count, corrections_count) tuples
    suitable for UI charting."""
    base_count = len(list(_GOLDEN_DIR.glob("*.json")))
    gens = corrections.list_generations()
    corr_stats = corrections.correction_stats()

    gen0_summary = rebenchmark_generation(0)

    cumulative_ids: set[str] = set()
    cumulative_corrections = 0
    per_generation = []
    moat_curve = [(0, gen0_summary.get("overall_aggregate"), base_count, 0)]

    for gen in gens:
        manifest = _read_manifest(gen)
        promoted = manifest.get("promoted", [])
        cumulative_ids.update(promoted)

        source = manifest.get("source_corrections", {})
        n_source = sum(len(v) for v in source.values()) if isinstance(source, dict) else len(source)
        cumulative_corrections += n_source

        summary = rebenchmark_generation(gen)
        per_generation.append(
            {
                "generation": gen,
                "ts": manifest.get("ts"),
                "reviewer": manifest.get("reviewer"),
                "promoted_this_generation": promoted,
                "cumulative_promoted_notes": len(cumulative_ids),
                "benchmark_summary": summary,
            }
        )
        moat_curve.append((gen, summary.get("overall_aggregate"), len(cumulative_ids), cumulative_corrections))

    return {
        "golden_set": {
            "base_count": base_count,
            "cumulative_promoted_notes": len(cumulative_ids),
        },
        "corrections_recorded_total": corr_stats["count"],
        "generations": per_generation,
        "moat_curve": moat_curve,
    }


# ---------------------------------------------------------------------------
# simulate_moat_demo
# ---------------------------------------------------------------------------

def _select_demo_corrections(max_n: int = 3) -> list[dict]:
    """Deterministically derive up to `max_n` REAL corrections from actual
    diffs between currently-generated notes (data/results/*.json) and their
    pristine golden references (data/golden_notes/*.json), using
    judge.align_notes for real semantic alignment — nothing here is
    fabricated text.

    Two categories of real diff, matching the spirit of "restore a dropped
    golden line" / "fix a paraphrase-lossy line":
      - "paraphrase_fix": a generated line IS aligned to a golden line
        (align_notes matched them) but the match is drifted (ratio below
        _PARAPHRASE_RATIO_CEILING) — corrected_text = the golden line's exact
        text.
      - "restore_dropped": a golden line has NO aligned generated
        counterpart at all (content entirely absent from the generated
        note's alignment) — the best raw-text-overlap generated line in the
        same section is used as the correction's carrier line, corrected to
        the golden line's exact text.

    Selection is deterministic: candidates are ranked by how drifted/lost
    they are (lowest ratio first), tie-broken by (transcript_id, section,
    line_index), and at least one of each category is included when
    available so the demo shows both flavors of correction."""
    restore_candidates: list[dict] = []
    paraphrase_candidates: list[dict] = []

    for data in _iter_current_results():
        transcript_id = data["transcript_id"]
        generated_note = data["generated_note"]
        golden_note = _load_pristine_golden(transcript_id)
        if golden_note is None:
            continue

        align = align_notes(generated_note, golden_note)

        for (g_sec, g_idx), (gold_sec, gold_idx, ratio, _same_section) in align["gen_to_gold"].items():
            if ratio >= _PARAPHRASE_RATIO_CEILING:
                continue
            gen_text = generated_note["soap"][g_sec][g_idx]["text"]
            gold_text = golden_note["soap"][gold_sec][gold_idx]["text"]
            if gen_text == gold_text:
                continue
            paraphrase_candidates.append(
                {
                    "transcript_id": transcript_id,
                    "section": g_sec,
                    "line_index": g_idx,
                    "original_text": gen_text,
                    "corrected_text": gold_text,
                    "kind": "paraphrase_fix",
                    "ratio": ratio,
                }
            )

        matched_gold = align["matched_gold_keys"]
        for gold_sec, gold_idx, gold_line in align["gold_lines"]:
            if (gold_sec, gold_idx) in matched_gold:
                continue
            gold_text = gold_line.get("text", "")
            section_gen_lines = generated_note.get("soap", {}).get(gold_sec, [])
            if not section_gen_lines:
                continue

            best_idx, best_ratio = None, -1.0
            for idx, line in enumerate(section_gen_lines):
                r = difflib.SequenceMatcher(None, line.get("text", ""), gold_text).ratio()
                if r > best_ratio:
                    best_idx, best_ratio = idx, r
            if best_idx is None:
                continue

            carrier_text = section_gen_lines[best_idx]["text"]
            if carrier_text == gold_text:
                continue
            restore_candidates.append(
                {
                    "transcript_id": transcript_id,
                    "section": gold_sec,
                    "line_index": best_idx,
                    "original_text": carrier_text,
                    "corrected_text": gold_text,
                    "kind": "restore_dropped",
                    "ratio": best_ratio,
                }
            )

    sort_key = lambda c: (c["ratio"], c["transcript_id"], c["section"], c["line_index"])  # noqa: E731
    restore_candidates.sort(key=sort_key)
    paraphrase_candidates.sort(key=sort_key)

    selected: list[dict] = []
    seen: set[tuple] = set()

    def _take(pool: list[dict], n: int) -> None:
        taken = 0
        for c in pool:
            key = (c["transcript_id"], c["section"], c["line_index"])
            if key in seen:
                continue
            seen.add(key)
            selected.append(c)
            taken += 1
            if taken >= n:
                return

    _take(restore_candidates, 1)
    _take(paraphrase_candidates, 1)
    remaining = max_n - len(selected)
    if remaining > 0:
        rest_pool = sorted(restore_candidates + paraphrase_candidates, key=sort_key)
        _take(rest_pool, remaining)

    return selected[:max_n]


def simulate_moat_demo(max_corrections: int = 3) -> dict:
    """Deterministic, script-callable demo of the correction loop compounding
    into the golden set: derives up to `max_corrections` real corrections
    (see _select_demo_corrections), records them, promotes them as ONE new
    generation, re-benchmarks before (gen-0) and after (the new generation),
    and returns the comparison.

    Idempotent: if any promoted generation already exists (a real one, or a
    previous demo seed), this does NOT seed again — it just returns the
    current moat_metrics() so re-running `--seed-demo` is always safe and
    never duplicates generations. Operates against whatever results dir is
    currently active (SCRIBEGATE_RESULTS_DIR if set, else the real
    data/results/) — sandboxed automatically by corrections._results_dir()."""
    existing_gens = corrections.list_generations()
    if existing_gens:
        return {
            "seeded": False,
            "reason": (
                f"generation(s) {existing_gens} already exist under "
                f"{corrections._golden_generations_root()}; not re-seeding"
            ),
            **moat_metrics(),
        }

    before = rebenchmark_generation(0)

    selected = _select_demo_corrections(max_corrections)
    if not selected:
        raise RuntimeError(
            "no candidate corrections could be derived from real generated/golden "
            "diffs in the currently active results dir — run `python -m scribegate.cli "
            "run --all` first so data/results/*.json exists"
        )

    for c in selected:
        corrections.record_correction(
            transcript_id=c["transcript_id"],
            section=c["section"],
            line_index=c["line_index"],
            original_text=c["original_text"],
            corrected_text=c["corrected_text"],
            reviewer=DEMO_REVIEWER,
            note=f"moat demo seed ({c['kind']}, alignment ratio {c['ratio']:.3f})",
        )

    manifest = corrections.promote_all_candidates(reviewer=DEMO_REVIEWER, note="moat demo seed promotion")
    gen_n = manifest["gen"]
    after = rebenchmark_generation(gen_n)

    return {
        "seeded": True,
        "generation": gen_n,
        "corrections": selected,
        "before": before,
        "after": after,
        "moat_metrics": moat_metrics(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scribegate.moat")
    parser.add_argument(
        "--seed-demo",
        action="store_true",
        help=(
            "Derive real corrections from actual generated-vs-golden diffs, promote "
            "them as one generation, re-benchmark, and print the before/after curve."
        ),
    )
    args = parser.parse_args(argv)

    if args.seed_demo:
        result = simulate_moat_demo()
    else:
        result = moat_metrics()

    print(json.dumps(result, indent=2, sort_keys=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
