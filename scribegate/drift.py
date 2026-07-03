"""drift.py (U2) — drift detection over data/results/history.jsonl and the
CI eval gate against a committed floor baseline (specs/baseline.json).

Two independent concerns live here, both pure-stdlib and deterministic:

1. Rolling-window regression detection (`detect_regression`) over the full
   history.jsonl time series — informational, used for the drift-summary
   view (`summarize_drift`) and for local/manual inspection. Not itself the
   CI gate.
2. A hard CI gate (`check_against_baseline`, wired up via the `__main__`
   entry point) that compares only the LATEST history row against fixed
   floors committed in specs/baseline.json. This is what actually fails CI
   — it is deliberately simpler and stricter than the rolling-window check
   so a single bad run cannot slip through averaged out by a rolling mean.

Usage:
    python -m scribegate.drift --check-baseline specs/baseline.json \
        [--history data/results/history.jsonl]

stdlib only (argparse, json, sys, dataclasses, pathlib). No network.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_HISTORY_PATH = _REPO_ROOT / "data" / "results" / "history.jsonl"

_DIMENSION_KEYS = ("completeness", "hallucination", "coding_plausibility", "terminology")


def load_history(path) -> list[dict]:
    """Load a history.jsonl file into a list of dicts, one per line.

    Matches repo convention (see benchmark.load_results): never crashes on
    a malformed/missing file. Blank lines and lines that fail to parse as a
    JSON object are silently skipped rather than raising."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                rows.append(data)
    return rows


@dataclass
class Alert:
    metric: str            # "overall", "visit_type:glaucoma_followup", "dimension:completeness"
    baseline_value: float
    current_value: float
    drop: float             # baseline_value - current_value (positive = regression)
    message: str = field(default="")

    def __post_init__(self) -> None:
        if not self.message:
            self.message = (
                f"{self.metric}: baseline={self.baseline_value:.3f} "
                f"current(rolling)={self.current_value:.3f} drop={self.drop:.3f}"
            )


def _metric_names(history: list[dict]) -> list[str]:
    """All metric keys observed across history: 'overall', 'visit_type:X',
    'dimension:Y'. Deterministic order: overall first, then visit types in
    first-seen order, then dimensions in first-seen order."""
    names: list[str] = ["overall"]
    seen = set(names)
    for row in history:
        for vt in (row.get("per_visit_type") or {}):
            key = f"visit_type:{vt}"
            if key not in seen:
                seen.add(key)
                names.append(key)
        for dim in (row.get("per_dimension") or {}):
            key = f"dimension:{dim}"
            if key not in seen:
                seen.add(key)
                names.append(key)
    return names


def _metric_value(row: dict, metric: str) -> float | None:
    if metric == "overall":
        val = row.get("overall_aggregate")
    elif metric.startswith("visit_type:"):
        vt = metric.split(":", 1)[1]
        val = (row.get("per_visit_type") or {}).get(vt)
    elif metric.startswith("dimension:"):
        dim = metric.split(":", 1)[1]
        val = (row.get("per_dimension") or {}).get(dim)
    else:
        val = None
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _is_baseline_row(row: dict) -> bool:
    if row.get("quality") == "baseline":
        return True
    tag = row.get("tag")
    if isinstance(tag, str) and "baseline" in tag:
        return True
    return False


def _find_baseline_value(history: list[dict], metric: str) -> float | None:
    """First history row whose quality=='baseline' (or tag contains/equals
    'baseline') that has a value for `metric`; falls back to the first
    chronological row with a value for `metric` if no baseline-tagged row
    exists."""
    for row in history:
        if _is_baseline_row(row):
            val = _metric_value(row, metric)
            if val is not None:
                return val
    for row in history:
        val = _metric_value(row, metric)
        if val is not None:
            return val
    return None


def detect_regression(
    history: list[dict], window: int = 3, threshold: float = 0.05
) -> list[Alert]:
    """Per metric (overall aggregate, each visit type, each dimension),
    compute the rolling mean of the last `window` runs' values and compare
    against a baseline value (first baseline-tagged row for that metric, or
    the first chronological row if none is tagged baseline). Fires an Alert
    when `baseline_value - rolling_mean >= threshold`.

    Pure and deterministic: no I/O, no randomness — operates purely on the
    already-loaded `history` list.
    """
    alerts: list[Alert] = []
    if not history:
        return alerts

    for metric in _metric_names(history):
        baseline_value = _find_baseline_value(history, metric)
        if baseline_value is None:
            continue

        recent_values = [
            v for v in (_metric_value(row, metric) for row in history[-window:]) if v is not None
        ]
        if not recent_values:
            continue
        rolling_mean = sum(recent_values) / len(recent_values)

        drop = baseline_value - rolling_mean
        if drop >= threshold:
            alerts.append(
                Alert(
                    metric=metric,
                    baseline_value=baseline_value,
                    current_value=rolling_mean,
                    drop=drop,
                )
            )

    return alerts


def summarize_drift(history: list[dict]) -> dict:
    """Pivot history rows into per-metric ordered time series:
        {"overall": [{"ts":..., "tag":..., "value":...}, ...],
         "visit_type:glaucoma_followup": [...],
         "dimension:completeness": [...],
         ...}
    Pure stdlib, deterministic — preserves history's original row order."""
    summary: dict[str, list[dict]] = {}
    for metric in _metric_names(history):
        series = []
        for row in history:
            val = _metric_value(row, metric)
            if val is None:
                continue
            series.append({"ts": row.get("ts"), "tag": row.get("tag"), "value": val})
        summary[metric] = series
    return summary


# ---------------------------------------------------------------------------
# CI eval gate: latest history row vs. committed specs/baseline.json floors
# ---------------------------------------------------------------------------

def check_against_baseline(latest_summary: dict, baseline: dict) -> tuple[bool, list[str]]:
    """Compare `latest_summary` (one history row / compute_summary-shaped
    dict) against `baseline` (specs/baseline.json shape: {"overall_aggregate":
    float, "per_dimension": {dim: float, ...}}).

    Returns (passed, failure_messages) — passed is True iff overall_aggregate
    and every per_dimension value in `baseline` are met or exceeded by
    `latest_summary`. `failure_messages` is empty when passed is True, and
    otherwise contains one human-readable message per metric that fell
    below its floor. Pure function — no I/O — so tests don't need to shell
    out to exercise this logic.
    """
    failures: list[str] = []

    floor_overall = baseline.get("overall_aggregate")
    current_overall = latest_summary.get("overall_aggregate")
    if isinstance(floor_overall, (int, float)):
        if not isinstance(current_overall, (int, float)) or current_overall < floor_overall:
            failures.append(
                f"overall_aggregate {current_overall} < floor {floor_overall}"
            )

    floor_dims = baseline.get("per_dimension") or {}
    current_dims = latest_summary.get("per_dimension") or {}
    for dim, floor_val in floor_dims.items():
        if not isinstance(floor_val, (int, float)):
            continue
        current_val = current_dims.get(dim)
        if not isinstance(current_val, (int, float)) or current_val < floor_val:
            failures.append(
                f"per_dimension[{dim}] {current_val} < floor {floor_val}"
            )

    return (len(failures) == 0, failures)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scribegate.drift")
    parser.add_argument(
        "--check-baseline",
        metavar="PATH",
        required=True,
        help="Path to specs/baseline.json (floors for overall_aggregate + per_dimension).",
    )
    parser.add_argument(
        "--history",
        metavar="PATH",
        default=str(_DEFAULT_HISTORY_PATH),
        help="Path to data/results/history.jsonl (default: data/results/history.jsonl).",
    )
    args = parser.parse_args(argv)

    history = load_history(args.history)
    if not history:
        print(f"FAIL: no history rows found in {args.history} — run the pipeline first.")
        return 1

    latest = history[-1]
    baseline_path = Path(args.check_baseline)
    try:
        baseline = _load_json(baseline_path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: could not load baseline file {baseline_path}: {exc}")
        return 1

    passed, failures = check_against_baseline(latest, baseline)
    if passed:
        print(
            f"PASS: latest run (tag={latest.get('tag')!r}, quality={latest.get('quality')!r}) "
            f"meets all floors in {baseline_path} "
            f"(overall_aggregate={latest.get('overall_aggregate')})."
        )
        return 0

    print(
        f"FAIL: latest run (tag={latest.get('tag')!r}, quality={latest.get('quality')!r}) "
        f"fell below floor(s) in {baseline_path}:"
    )
    for msg in failures:
        print(f"  - {msg}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
