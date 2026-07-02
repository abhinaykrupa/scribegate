"""router.py (T6) — decision routing for judged notes.

Contract (specs/INTERFACES.md):
    route(judge_result, violations) -> str
    # "auto_accept" if aggregate >= 0.85 and no error-severity violation
    # "review" if 0.60 <= aggregate < 0.85 and no error-severity violation
    # "regenerate" otherwise (aggregate < 0.60 OR any error-severity violation)

Thresholds are read from specs/rubric.yaml `router_thresholds` at module
import time, with a hard-coded fallback (matching the values documented in
INTERFACES.md) if the file is missing, unreadable, or malformed — this
module must never crash on import, matching repo convention (see
normalizer.py's `_load_terminology`).

stdlib + pyyaml only. Deterministic. No network. Python >= 3.10.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml is a declared dependency
    yaml = None  # type: ignore[assignment]

from scribegate.normalizer import Violation

_RUBRIC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "specs",
    "rubric.yaml",
)

# ---------------------------------------------------------------------------
# Built-in fallback thresholds (used if specs/rubric.yaml can't be loaded)
# ---------------------------------------------------------------------------

_DEFAULT_AUTO_ACCEPT = 0.85
_DEFAULT_REVIEW = 0.60


def _load_router_thresholds(path: str = _RUBRIC_PATH) -> dict:
    """Load specs/rubric.yaml `router_thresholds` once. Returns {} on any
    failure so the module can fall back to built-in defaults without
    crashing at import."""
    if yaml is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, dict):
            thresholds = data.get("router_thresholds")
            if isinstance(thresholds, dict):
                return thresholds
    except (OSError, IOError, ValueError):
        return {}
    except Exception:  # pragma: no cover - defensive catch-all, never crash
        return {}
    return {}


def _build_thresholds(raw: dict) -> tuple[float, float]:
    try:
        auto_accept = float(raw.get("auto_accept", _DEFAULT_AUTO_ACCEPT))
    except (TypeError, ValueError):
        auto_accept = _DEFAULT_AUTO_ACCEPT
    try:
        review = float(raw.get("review", _DEFAULT_REVIEW))
    except (TypeError, ValueError):
        review = _DEFAULT_REVIEW
    return auto_accept, review


_RAW_THRESHOLDS = _load_router_thresholds()
AUTO_ACCEPT_THRESHOLD, REVIEW_THRESHOLD = _build_thresholds(_RAW_THRESHOLDS)


# ---------------------------------------------------------------------------
# RouteDecision
# ---------------------------------------------------------------------------

@dataclass
class RouteDecision:
    route: str
    aggregate: float
    reasons: list[str] = field(default_factory=list)


def _has_error_violation(violations: list) -> bool:
    for v in violations or []:
        severity = getattr(v, "severity", None)
        if severity is None and isinstance(v, dict):
            severity = v.get("severity")
        if severity == "error":
            return True
    return False


def _error_codes(violations: list) -> list[str]:
    codes = []
    for v in violations or []:
        severity = getattr(v, "severity", None)
        code = getattr(v, "code", None)
        if severity is None and isinstance(v, dict):
            severity = v.get("severity")
            code = v.get("code")
        if severity == "error":
            codes.append(code or "UNKNOWN")
    return codes


def decide(judge_result: dict, violations: list[Violation] | None = None) -> RouteDecision:
    """Pure function producing a full RouteDecision (route + aggregate + reasons).

    `violations` may be a list of `Violation` dataclass instances or plain
    dicts with "severity"/"code" keys (append-friendly for callers that have
    already serialized violations, e.g. the CLI re-loading a results file).
    """
    violations = violations or []
    aggregate = float(judge_result.get("aggregate", 0.0))
    has_error = _has_error_violation(violations)
    reasons: list[str] = []

    if has_error:
        codes = _error_codes(violations)
        reasons.append(
            f"error-severity violation(s) present: {', '.join(codes)} — forces regenerate"
        )
        route = "regenerate"
    elif aggregate >= AUTO_ACCEPT_THRESHOLD:
        reasons.append(
            f"aggregate {aggregate:.3f} >= auto_accept threshold {AUTO_ACCEPT_THRESHOLD:.2f}"
        )
        route = "auto_accept"
    elif aggregate >= REVIEW_THRESHOLD:
        reasons.append(
            f"aggregate {aggregate:.3f} in [{REVIEW_THRESHOLD:.2f}, {AUTO_ACCEPT_THRESHOLD:.2f}) — routed to human review"
        )
        route = "review"
    else:
        reasons.append(
            f"aggregate {aggregate:.3f} < review threshold {REVIEW_THRESHOLD:.2f} — regenerate"
        )
        route = "regenerate"

    warn_codes = [
        getattr(v, "code", None) or (v.get("code") if isinstance(v, dict) else None)
        for v in violations
        if (getattr(v, "severity", None) or (v.get("severity") if isinstance(v, dict) else None)) == "warn"
    ]
    if warn_codes:
        reasons.append(f"warn-severity violation(s) noted (non-blocking): {', '.join(warn_codes)}")

    return RouteDecision(route=route, aggregate=aggregate, reasons=reasons)


def route(judge_result: dict, violations: list[Violation]) -> str:
    """Return the route string ("auto_accept" / "review" / "regenerate")
    per specs/INTERFACES.md. Thin wrapper over `decide` for callers that
    only need the route label."""
    return decide(judge_result, violations).route
