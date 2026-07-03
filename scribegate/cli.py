"""cli.py (T6) — command-line entry point tying generator -> normalizer ->
judge -> router together for one or all transcripts.

Contract (specs/INTERFACES.md):
    python -m scribegate.cli run [--transcript ID] [--all]
    -> writes data/results/{id}.json (generated note + judge result + route
       + violations), appends one line to data/results/decision_log.jsonl,
       and prints a human-readable one-line summary per case to stdout.

stdlib + pyyaml only. Deterministic. No network by default (generator/judge
API backends are env-gated, see their modules).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import sys
from pathlib import Path

from scribegate.generator import generate_note, visit_type_for
from scribegate.normalizer import check_note
from scribegate.judge import judge_note
from scribegate.router import decide
from scribegate.benchmark import compute_summary

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_TRANSCRIPT_DIR = _DATA_DIR / "transcripts"
_GOLDEN_DIR = _DATA_DIR / "golden_notes"
_DEFAULT_RESULTS_DIR = _DATA_DIR / "results"
_DEFAULT_HISTORY_PATH = _DEFAULT_RESULTS_DIR / "history.jsonl"

DECISION_LOG_NAME = "decision_log.jsonl"
HISTORY_NAME = "history.jsonl"


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def discover_transcript_ids(transcript_dir: Path = _TRANSCRIPT_DIR) -> list[str]:
    """All transcript ids available (from data/transcripts/*.txt), sorted
    for deterministic ordering."""
    return sorted(p.stem for p in transcript_dir.glob("*.txt"))


def _load_transcript_text(transcript_id: str, transcript_dir: Path = _TRANSCRIPT_DIR) -> str:
    path = transcript_dir / f"{transcript_id}.txt"
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _load_golden(transcript_id: str, golden_dir: Path = _GOLDEN_DIR) -> dict | None:
    path = golden_dir / f"{transcript_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _violation_to_dict(v) -> dict:
    if dataclasses.is_dataclass(v):
        return dataclasses.asdict(v)
    if isinstance(v, dict):
        return v
    return {
        "code": getattr(v, "code", "UNKNOWN"),
        "severity": getattr(v, "severity", "warn"),
        "message": getattr(v, "message", ""),
        "line_text": getattr(v, "line_text", ""),
    }


def process_transcript(
    transcript_id: str,
    transcript_dir: Path = _TRANSCRIPT_DIR,
    golden_dir: Path = _GOLDEN_DIR,
    quality: str = "baseline",
) -> dict:
    """Run the full generate -> normalize -> judge -> route pipeline for one
    transcript id. Returns the result payload written to
    data/results/{id}.json (without timestamps merged yet — caller adds
    those so the same payload is reusable/testable).

    `quality` ("baseline" | "degraded") is threaded straight through to
    `generate_note` — see generator.py for what each tier changes."""
    transcript_text = _load_transcript_text(transcript_id, transcript_dir)
    visit_type = visit_type_for(transcript_id)

    generated_note = generate_note(transcript_text, transcript_id, visit_type, quality=quality)
    violations = check_note(generated_note, transcript=transcript_text)

    golden = _load_golden(transcript_id, golden_dir)
    if golden is not None:
        judge_result = judge_note(generated_note, golden, transcript_text)
    else:
        # No golden reference available: still produce a well-formed result
        # rather than crashing, with a neutral/zero judge result so routing
        # falls through to "regenerate" (safest default for missing ground truth).
        judge_result = {
            "scores": {"completeness": 1, "hallucination": 1, "coding_plausibility": 1, "terminology": 1},
            "aggregate": 0.0,
            "rationales": {dim: "no golden note available for comparison" for dim in
                           ("completeness", "hallucination", "coding_plausibility", "terminology")},
        }

    decision = decide(judge_result, violations)

    return {
        "transcript_id": transcript_id,
        "visit_type": visit_type,
        "generated_note": generated_note,
        "judge_result": judge_result,
        "violations": [_violation_to_dict(v) for v in violations],
        "route": decision.route,
        "decision_reasons": decision.reasons,
    }


def _write_result_json(result: dict, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{result['transcript_id']}.json"
    payload = dict(result)
    payload["timestamps"] = {"generated_at": _utc_now_iso()}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return out_path


def _append_decision_log(result: dict, results_dir: Path) -> Path:
    """Append-only provenance/audit log — never overwritten. One JSON
    object per line: {ts, transcript_id, aggregate, route, violation_count}."""
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / DECISION_LOG_NAME
    entry = {
        "ts": _utc_now_iso(),
        "transcript_id": result["transcript_id"],
        "aggregate": result["judge_result"].get("aggregate"),
        "route": result["route"],
        "violation_count": len(result["violations"]),
    }
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=False))
        fh.write("\n")
    return log_path


def _summary_line(result: dict) -> str:
    aggregate = result["judge_result"].get("aggregate")
    agg_str = f"{aggregate:.3f}" if isinstance(aggregate, (int, float)) else "n/a"
    n_errors = sum(1 for v in result["violations"] if v.get("severity") == "error")
    n_warns = sum(1 for v in result["violations"] if v.get("severity") == "warn")
    return (
        f"{result['transcript_id']:<20} visit={result['visit_type']:<22} "
        f"aggregate={agg_str} route={result['route']:<11} "
        f"violations(error={n_errors},warn={n_warns})"
    )


def run(
    transcript_ids: list[str],
    results_dir: Path = _DEFAULT_RESULTS_DIR,
    transcript_dir: Path = _TRANSCRIPT_DIR,
    golden_dir: Path = _GOLDEN_DIR,
    stream=None,
    quality: str = "baseline",
) -> list[dict]:
    """Run the pipeline for each transcript id (deterministic order as
    given), writing results + appending to the decision log. Returns the
    list of result payloads (as written, incl. timestamps).

    `quality` is threaded through to `process_transcript` -> `generate_note`
    for every transcript id in this run."""
    stream = stream or sys.stdout
    results = []
    for transcript_id in transcript_ids:
        result = process_transcript(transcript_id, transcript_dir, golden_dir, quality=quality)
        _write_result_json(result, results_dir)
        _append_decision_log(result, results_dir)
        print(_summary_line(result), file=stream)
        results.append(result)
    return results


def append_history_row(
    results: list[dict],
    tag: str,
    quality: str,
    history_path: Path = _DEFAULT_HISTORY_PATH,
) -> Path:
    """Compute a summary for `results` (via `benchmark.compute_summary`) and
    append one JSON line to `history_path`, merging in run metadata
    (`ts`, `tag`, `quality`) that `compute_summary` intentionally does not
    know about — it operates purely on result payloads so it stays reusable
    by benchmark.py's markdown builder without any run-provenance coupling.

    Approach: history.jsonl is an unbounded append-only log (same append-only
    convention as decision_log.jsonl) rather than a single mutable "latest
    summary" file — each `cli.py run` invocation contributes exactly one row,
    so a sequence of runs (e.g. baseline then degraded) naturally accumulates
    a time series that scribegate.drift can read with a rolling window,
    without needing to reconstruct history from individual per-transcript
    result files (which get overwritten on every run and don't retain
    superseded runs' aggregate scores).
    """
    summary = compute_summary(results)
    row = dict(summary)
    row["ts"] = _utc_now_iso()
    row["tag"] = tag
    row["quality"] = quality

    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=False))
        fh.write("\n")
    return history_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scribegate.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Generate, check, judge, and route transcripts.")
    group = run_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--transcript", metavar="ID", help="Run a single transcript id.")
    group.add_argument("--all", action="store_true", help="Run all transcripts in data/transcripts/.")
    run_parser.add_argument(
        "--results-dir",
        metavar="DIR",
        default=str(_DEFAULT_RESULTS_DIR),
        help="Directory to write results into (default: data/results/).",
    )
    run_parser.add_argument(
        "--transcript-dir",
        metavar="DIR",
        default=str(_TRANSCRIPT_DIR),
        help=argparse.SUPPRESS,
    )
    run_parser.add_argument(
        "--golden-dir",
        metavar="DIR",
        default=str(_GOLDEN_DIR),
        help=argparse.SUPPRESS,
    )
    run_parser.add_argument(
        "--quality",
        choices=("baseline", "degraded"),
        default="baseline",
        help="Generator quality tier to use (default: baseline).",
    )
    run_parser.add_argument(
        "--tag",
        metavar="TEXT",
        default=None,
        help="Label for this run's history.jsonl row (default: falls back to --quality).",
    )
    run_parser.add_argument(
        "--history-path",
        metavar="PATH",
        default=None,
        help=argparse.SUPPRESS,
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        transcript_dir = Path(args.transcript_dir)
        golden_dir = Path(args.golden_dir)
        results_dir = Path(args.results_dir)
        quality = args.quality
        tag = args.tag if args.tag else quality
        # history-path defaults to <results-dir>/history.jsonl (not a fixed
        # repo-root path) so that overriding --results-dir (as tests do, to
        # isolate runs into tmp_path) also isolates the history file — a
        # fixed absolute default would otherwise let any test that exercises
        # cli.main(["run", ...]) silently append rows to the real repo's
        # data/results/history.jsonl.
        history_path = Path(args.history_path) if args.history_path else results_dir / HISTORY_NAME

        if args.all:
            transcript_ids = discover_transcript_ids(transcript_dir)
        else:
            transcript_ids = [args.transcript]

        results = run(
            transcript_ids,
            results_dir=results_dir,
            transcript_dir=transcript_dir,
            golden_dir=golden_dir,
            quality=quality,
        )
        append_history_row(results, tag=tag, quality=quality, history_path=history_path)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
