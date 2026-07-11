"""costs.py (W1) — pricing table + append-only cost ledger for live,
API-backed drafting/judging (scribegate/live.py).

Pricing is loaded from specs/pricing.yaml (USD per 1,000,000 tokens) so rates
can be corrected without a code change; a hardcoded fallback table (same
values, "verify before demo") is used if that file is missing/malformed/
unreadable so this module never crashes at import or call time.

The ledger is a plain append-only JSONL file
(data/results/live_runs/cost_ledger.jsonl by default) — one line per API call,
each carrying a UTC timestamp, a same-timezone `day` bucket (for daily-budget
rollover), the stage ("draft" / "judge_sample_0" / ...), model, token counts,
and the computed USD cost. Never carries the API key or any transcript
content — this file is safe to inspect/ship as a cost audit trail.

stdlib + pyyaml only. Deterministic given inputs (cost math has no
randomness). No network.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml is a declared dependency
    yaml = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SPECS_DIR = _REPO_ROOT / "specs"
_PRICING_PATH = _SPECS_DIR / "pricing.yaml"
_DEFAULT_LIVE_RUNS_DIR = _REPO_ROOT / "data" / "results" / "live_runs"
_DEFAULT_LEDGER_PATH = _DEFAULT_LIVE_RUNS_DIR / "cost_ledger.jsonl"

# Fallback pricing table (USD per 1M tokens) — kept in sync with
# specs/pricing.yaml by convention, used only if that file can't be loaded.
# VERIFY BEFORE DEMO — see specs/pricing.yaml header.
_FALLBACK_PRICING = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-opus-4-8": {"input": 15.00, "output": 75.00},
}


def load_pricing(path: str | Path | None = None) -> dict:
    """Load the per-model {"input": usd_per_mtok, "output": usd_per_mtok}
    pricing table from specs/pricing.yaml. Falls back to _FALLBACK_PRICING on
    ANY failure (missing file, unreadable, malformed YAML, wrong shape,
    pyyaml not installed) — never raises, so a broken pricing.yaml can never
    take down live mode."""
    resolved = Path(path) if path else _PRICING_PATH
    if yaml is not None:
        try:
            with open(resolved, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if isinstance(data, dict) and isinstance(data.get("models"), dict):
                table = {}
                for model, rates in data["models"].items():
                    if isinstance(rates, dict) and "input" in rates and "output" in rates:
                        table[model] = {
                            "input": float(rates["input"]),
                            "output": float(rates["output"]),
                        }
                if table:
                    return table
        except Exception:
            pass
    return {k: dict(v) for k, v in _FALLBACK_PRICING.items()}


def cost_of(usage: dict, pricing: dict | None = None) -> float:
    """usage: {"model": str, "input_tokens": int, "output_tokens": int} -> USD.

    Unknown models fall back to the most expensive known rate (a
    conservative overestimate) rather than zero-cost — undercounting spend
    for budget enforcement would be the unsafe failure mode here, not
    overcounting it.
    """
    pricing = pricing if pricing is not None else load_pricing()
    model = usage.get("model", "")
    rates = pricing.get(model)
    if rates is None:
        if pricing:
            rates = max(pricing.values(), key=lambda r: r.get("input", 0) + r.get("output", 0))
        else:
            rates = {"input": 15.0, "output": 75.0}
    input_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0
    return (input_tokens / 1_000_000.0) * rates["input"] + (output_tokens / 1_000_000.0) * rates["output"]


def _resolve_ledger_path(ledger_path: str | Path | None = None) -> Path:
    return Path(ledger_path) if ledger_path else _DEFAULT_LEDGER_PATH


def record_usage(
    stage: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    ledger_path: str | Path | None = None,
    pricing: dict | None = None,
) -> dict:
    """Append one usage record to the cost ledger (append-only JSONL) and
    return the record written. Never accepts or persists API key material —
    callers only ever pass stage/model/token counts here."""
    now = datetime.now(timezone.utc)
    usage = {
        "model": model,
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
    }
    usd = cost_of(usage, pricing=pricing)
    record = {
        "ts": now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "day": now.strftime("%Y-%m-%d"),
        "stage": stage,
        "model": model,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "usd": usd,
    }
    path = _resolve_ledger_path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


_LEDGER_REQUIRED_KEYS = ("day", "usd")


def _iter_ledger(ledger_path: str | Path | None = None):
    """Yield each valid ledger record (a dict carrying at least the
    `_LEDGER_REQUIRED_KEYS`) from the JSONL ledger file, one per line.

    Every line is treated as untrusted input: a line that isn't valid JSON,
    that parses to a JSON-valid but non-dict value (a bare int/float/bool,
    `null`, a list, or a string), or that parses to a dict missing any of
    the keys this module's readers depend on, is corrupt and is skipped
    silently rather than raised — a single bad line must never crash
    Live mode's spend/budget math.
    """
    path = _resolve_ledger_path(ledger_path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            if any(key not in rec for key in _LEDGER_REQUIRED_KEYS):
                continue
            yield rec


def today_spend(ledger_path: str | Path | None = None, today: str | None = None) -> float:
    """Sum of `usd` across all ledger records whose `day` bucket matches
    `today` (default: today's UTC date) — this is the daily-budget rollover:
    a new UTC day always starts fresh regardless of ledger history."""
    day = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    total = 0.0
    for rec in _iter_ledger(ledger_path):
        if rec.get("day") == day:
            total += float(rec.get("usd", 0.0))
    return total


def _config_budget(config) -> float:
    if config is None:
        return 5.0
    budget = getattr(config, "daily_budget_usd", None)
    if budget is None and isinstance(config, dict):
        budget = config.get("daily_budget_usd")
    try:
        return float(budget) if budget is not None else 5.0
    except (TypeError, ValueError):
        return 5.0


def budget_remaining(config, ledger_path: str | Path | None = None) -> float:
    """Daily budget minus today's spend (can go negative once exceeded)."""
    budget = _config_budget(config)
    spend = today_spend(ledger_path=ledger_path)
    return budget - spend


def per_note_breakdown(run: dict) -> dict:
    """Summarize a run's `cost_records` (list of records shaped like
    `record_usage`'s return value) into drafting/judging/total USD + token
    buckets. Stage names are bucketed by prefix: anything starting with
    "draft" -> drafting, anything starting with "judge" -> judging; any
    other stage name is still counted in `total_usd` but not double-bucketed."""
    breakdown = {
        "drafting": {"usd": 0.0, "input_tokens": 0, "output_tokens": 0},
        "judging": {"usd": 0.0, "input_tokens": 0, "output_tokens": 0},
        "total_usd": 0.0,
    }
    for rec in (run or {}).get("cost_records", []) or []:
        stage = str(rec.get("stage", ""))
        usd = float(rec.get("usd", 0.0))
        input_tokens = int(rec.get("input_tokens", 0) or 0)
        output_tokens = int(rec.get("output_tokens", 0) or 0)
        breakdown["total_usd"] += usd
        if stage.startswith("draft"):
            bucket = breakdown["drafting"]
        elif stage.startswith("judge"):
            bucket = breakdown["judging"]
        else:
            bucket = None
        if bucket is not None:
            bucket["usd"] += usd
            bucket["input_tokens"] += input_tokens
            bucket["output_tokens"] += output_tokens
    return breakdown
