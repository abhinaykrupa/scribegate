"""benchmark.py (T6) — aggregates data/results/*.json into a markdown report.

Usage:
    python -m scribegate.benchmark [--results-dir DIR] [--out PATH]

Reads every data/results/{id}.json (as written by scribegate.cli), and
writes data/results/benchmark.md containing:
  - Table 1: per visit_type x dimension mean scores, mean aggregate, and
    route counts.
  - Table 2: per-transcript row (id, aggregate, 4 dimension scores, route,
    violation count).

Deterministic ordering throughout: visit types in a fixed canonical order,
transcripts sorted alphabetically by id.

stdlib only. No network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_DEFAULT_RESULTS_DIR = _DATA_DIR / "results"
_DEFAULT_OUT_PATH = _DEFAULT_RESULTS_DIR / "benchmark.md"

DIMENSIONS = ("completeness", "hallucination", "coding_plausibility", "terminology")
ROUTES = ("auto_accept", "review", "regenerate")

# Canonical visit-type ordering for deterministic table output.
VISIT_TYPE_ORDER = (
    "comprehensive_exam",
    "glaucoma_followup",
    "cataract_postop",
    "contact_lens_fitting",
)


def load_results(results_dir: Path = _DEFAULT_RESULTS_DIR) -> list[dict]:
    """Load every data/results/{id}.json file, sorted by transcript_id for
    deterministic ordering. Skips non-result files (e.g. decision_log.jsonl,
    benchmark.md itself)."""
    results = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError:
                continue
        if isinstance(data, dict) and "transcript_id" in data and "judge_result" in data:
            results.append(data)
    results.sort(key=lambda r: r["transcript_id"])
    return results


def _visit_type_of(result: dict) -> str:
    return result.get("visit_type") or result.get("generated_note", {}).get("visit_type") or "unknown"


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def _group_by_visit_type(results: list[dict]) -> dict[str, list[dict]]:
    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(_visit_type_of(r), []).append(r)
    return by_type


def _ordered_visit_types(by_type: dict[str, list[dict]]) -> list[str]:
    ordered_types = [vt for vt in VISIT_TYPE_ORDER if vt in by_type]
    ordered_types += sorted(vt for vt in by_type if vt not in VISIT_TYPE_ORDER)
    return ordered_types


def _dim_means(group: list[dict]) -> dict[str, float]:
    dim_means = {}
    for dim in DIMENSIONS:
        scores = [r["judge_result"]["scores"][dim] for r in group]
        dim_means[dim] = mean(scores) if scores else 0.0
    return dim_means


def _agg_mean(group: list[dict]) -> float:
    return mean(r["judge_result"]["aggregate"] for r in group) if group else 0.0


def compute_summary(results: list[dict]) -> dict:
    """Single source of truth for aggregation math, reused by build_markdown /
    _build_visit_type_table and by cli.append_history_row (via the caller,
    which merges in ts/tag/quality — compute_summary itself stays agnostic
    of those run-metadata fields, taking only the result payloads).

    Returns exactly:
        {"overall_aggregate": float,
         "per_visit_type": {vt: float, ...},   # one of the 4 canonical types
         "per_dimension": {dim: float, ...},   # mean *aggregate-normalized*
                                                # score (0..1) per dimension
         "n_notes": int,
         "auto_accept_rate": float}
    """
    n_notes = len(results)
    overall_aggregate = _agg_mean(results)

    by_type = _group_by_visit_type(results)
    per_visit_type = {vt: _agg_mean(by_type.get(vt, [])) for vt in VISIT_TYPE_ORDER}

    per_dimension = {}
    for dim in DIMENSIONS:
        # Normalize each 1-5 score to 0..1 the same way aggregate is derived
        # (per specs/INTERFACES.md: aggregate = (mean(scores) - 1) / 4), so
        # per_dimension values sit on the same 0..1 scale as overall_aggregate
        # and per_visit_type — directly comparable for drift detection.
        scores = [r["judge_result"]["scores"][dim] for r in results]
        per_dimension[dim] = (mean(scores) - 1) / 4 if scores else 0.0

    n_auto_accept = sum(1 for r in results if r.get("route") == "auto_accept")
    auto_accept_rate = (n_auto_accept / n_notes) if n_notes else 0.0

    return {
        "overall_aggregate": overall_aggregate,
        "per_visit_type": per_visit_type,
        "per_dimension": per_dimension,
        "n_notes": n_notes,
        "auto_accept_rate": auto_accept_rate,
    }


def _build_visit_type_table(results: list[dict]) -> str:
    by_type = _group_by_visit_type(results)
    ordered_types = _ordered_visit_types(by_type)

    header = (
        "| Visit Type | N | Completeness | Hallucination | Coding Plausibility | "
        "Terminology | Mean Aggregate | auto_accept | review | regenerate |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]

    for vt in ordered_types:
        group = by_type[vt]
        dim_means = _dim_means(group)
        agg_mean = _agg_mean(group)
        route_counts = {rt: sum(1 for r in group if r.get("route") == rt) for rt in ROUTES}
        rows.append(
            f"| {vt} | {len(group)} | {_fmt(dim_means['completeness'])} | "
            f"{_fmt(dim_means['hallucination'])} | {_fmt(dim_means['coding_plausibility'])} | "
            f"{_fmt(dim_means['terminology'])} | {_fmt(agg_mean)} | "
            f"{route_counts['auto_accept']} | {route_counts['review']} | {route_counts['regenerate']} |"
        )

    return "\n".join(rows)


def _build_transcript_table(results: list[dict]) -> str:
    header = (
        "| Transcript ID | Visit Type | Aggregate | Completeness | Hallucination | "
        "Coding Plausibility | Terminology | Route | Violations (err/warn) |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]

    for r in sorted(results, key=lambda x: x["transcript_id"]):
        scores = r["judge_result"]["scores"]
        aggregate = r["judge_result"]["aggregate"]
        violations = r.get("violations", [])
        n_err = sum(1 for v in violations if v.get("severity") == "error")
        n_warn = sum(1 for v in violations if v.get("severity") == "warn")
        rows.append(
            f"| {r['transcript_id']} | {_visit_type_of(r)} | {_fmt(aggregate)} | "
            f"{scores['completeness']} | {scores['hallucination']} | "
            f"{scores['coding_plausibility']} | {scores['terminology']} | "
            f"{r.get('route', 'n/a')} | {n_err}/{n_warn} |"
        )

    return "\n".join(rows)


def build_markdown(results: list[dict]) -> str:
    lines = []
    lines.append("# ScribeGate Benchmark Report")
    lines.append("")
    lines.append(
        "Synthetic/educational data only — not clinical guidance. Generated by "
        "`python -m scribegate.benchmark` from `data/results/*.json`."
    )
    lines.append("")
    lines.append("## Reading this table")
    lines.append("")
    lines.append(
        "Scores are on a 1-5 scale per dimension (completeness, hallucination, "
        "coding_plausibility, terminology); aggregate is the normalized 0-1 mean "
        "per `specs/rubric.yaml`. The `contact_lens_fitting` transcripts are "
        "**deliberately messy** (colloquial dictation, more crosstalk, looser "
        "structure) by design of the synthetic fixture set — they exist to stress "
        "the generator and judge, not to represent a typical visit. A lower mean "
        "aggregate and higher `regenerate`/`review` route counts for that visit "
        "type is **expected and correct**: it means the harness is doing its job "
        "and catching genuinely harder-to-document encounters, not that the "
        "pipeline is broken. Read a low contact-lens score as evidence the eval "
        "gate works, not as a regression to chase."
    )
    lines.append("")
    lines.append("## Table 1: Per visit type")
    lines.append("")
    lines.append(_build_visit_type_table(results))
    lines.append("")
    lines.append("## Table 2: Per transcript")
    lines.append("")
    lines.append(_build_transcript_table(results))
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scribegate.benchmark")
    parser.add_argument("--results-dir", default=str(_DEFAULT_RESULTS_DIR))
    parser.add_argument("--out", default=None, help="Output path (default: <results-dir>/benchmark.md)")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    out_path = Path(args.out) if args.out else results_dir / "benchmark.md"

    results = load_results(results_dir)
    markdown = build_markdown(results)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(markdown)

    print(f"Wrote {out_path} ({len(results)} transcript(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
