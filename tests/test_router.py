"""Tests for scribegate.router: threshold boundaries + error/warn violation
handling per specs/rubric.yaml router_thresholds and specs/INTERFACES.md."""

from scribegate.normalizer import Violation
from scribegate.router import route, decide, RouteDecision, AUTO_ACCEPT_THRESHOLD, REVIEW_THRESHOLD


def _judge_result(aggregate: float) -> dict:
    return {
        "scores": {"completeness": 5, "hallucination": 5, "coding_plausibility": 5, "terminology": 5},
        "aggregate": aggregate,
        "rationales": {},
    }


def _warn_violation() -> Violation:
    return Violation(code="VA_FORMAT", severity="warn", message="minor formatting", line_text="20/20")


def _error_violation() -> Violation:
    return Violation(code="LATERALITY_CONFLICT", severity="error", message="OD/OS mismatch", line_text="IOP 30 OS")


def test_thresholds_match_rubric_defaults():
    # specs/rubric.yaml router_thresholds: auto_accept 0.85, review 0.60
    assert AUTO_ACCEPT_THRESHOLD == 0.85
    assert REVIEW_THRESHOLD == 0.60


def test_boundary_0_85_is_auto_accept():
    assert route(_judge_result(0.85), []) == "auto_accept"


def test_boundary_0_849_is_review():
    assert route(_judge_result(0.849), []) == "review"


def test_boundary_0_60_is_review():
    assert route(_judge_result(0.60), []) == "review"


def test_boundary_0_599_is_regenerate():
    assert route(_judge_result(0.599), []) == "regenerate"


def test_high_aggregate_no_violations_is_auto_accept():
    assert route(_judge_result(0.95), []) == "auto_accept"


def test_mid_aggregate_no_violations_is_review():
    assert route(_judge_result(0.72), []) == "review"


def test_low_aggregate_is_regenerate():
    assert route(_judge_result(0.10), []) == "regenerate"


def test_error_violation_forces_regenerate_even_at_high_aggregate():
    result = route(_judge_result(0.95), [_error_violation()])
    assert result == "regenerate"


def test_warn_violation_does_not_force_regenerate():
    result = route(_judge_result(0.95), [_warn_violation()])
    assert result == "auto_accept"


def test_warn_violation_at_review_band_stays_review():
    result = route(_judge_result(0.70), [_warn_violation()])
    assert result == "review"


def test_error_violation_at_low_aggregate_still_regenerate():
    result = route(_judge_result(0.10), [_error_violation()])
    assert result == "regenerate"


def test_decide_returns_route_decision_dataclass():
    decision = decide(_judge_result(0.90), [])
    assert isinstance(decision, RouteDecision)
    assert decision.route == "auto_accept"
    assert decision.aggregate == 0.90
    assert isinstance(decision.reasons, list)
    assert len(decision.reasons) >= 1


def test_decide_reasons_mention_error_violation_code():
    decision = decide(_judge_result(0.95), [_error_violation()])
    assert decision.route == "regenerate"
    assert any("LATERALITY_CONFLICT" in reason for reason in decision.reasons)


def test_decide_reasons_mention_warn_violation_non_blocking():
    decision = decide(_judge_result(0.95), [_warn_violation()])
    assert decision.route == "auto_accept"
    assert any("warn" in reason.lower() for reason in decision.reasons)


def test_route_accepts_violations_as_dicts():
    # append-friendly: callers re-loading serialized results pass dicts
    dict_violation = {"code": "IOP_RANGE", "severity": "error", "message": "out of range", "line_text": "x"}
    assert route(_judge_result(0.95), [dict_violation]) == "regenerate"


def test_multiple_error_violations_all_listed_in_reasons():
    decision = decide(_judge_result(0.95), [_error_violation(), _error_violation()])
    assert decision.route == "regenerate"
    reason_text = " ".join(decision.reasons)
    assert reason_text.count("LATERALITY_CONFLICT") >= 1


def test_empty_violations_list_default():
    decision = decide(_judge_result(0.90))
    assert decision.route == "auto_accept"
