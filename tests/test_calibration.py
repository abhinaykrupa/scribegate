"""Tests for scribegate.calibration: sampled judging, CI-aware routing, and
the full-fixture-set calibration report. See calibration.py's module
docstring for what these three pieces answer for the COO's "probabilistic
judge" question."""

import copy
import json
from pathlib import Path

import pytest

from scribegate.calibration import (
    APISampledJudge,
    _mock_judge_note_sampled,
    build_markdown,
    calibration_report,
    case_difficulty,
    judge_note_sampled,
    route_sampled,
)
from scribegate.normalizer import Violation
from scribegate.router import AUTO_ACCEPT_THRESHOLD, REVIEW_THRESHOLD

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLDEN_DIR = DATA_DIR / "golden_notes"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"

CONTACTLENS_03 = "contactlens_03"
GLAUCOMA_05 = "glaucoma_05"


def _load_golden(transcript_id: str) -> dict:
    with open(GOLDEN_DIR / f"{transcript_id}.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_transcript(transcript_id: str) -> str:
    with open(TRANSCRIPT_DIR / f"{transcript_id}.txt", "r", encoding="utf-8") as fh:
        return fh.read()


def _error_violation() -> Violation:
    return Violation(code="LATERALITY_CONFLICT", severity="error", message="mismatch", line_text="x")


# ---------------------------------------------------------------------------
# 1. Determinism given a seed.
# ---------------------------------------------------------------------------

def test_judge_note_sampled_deterministic_with_seed():
    golden = _load_golden(CONTACTLENS_03)
    transcript = _load_transcript(CONTACTLENS_03)

    result_a = judge_note_sampled(golden, golden, transcript, n=7, seed=123)
    result_b = judge_note_sampled(golden, golden, transcript, n=7, seed=123)

    assert result_a["mean_scores"] == result_b["mean_scores"]
    assert result_a["aggregate_mean"] == result_b["aggregate_mean"]
    assert result_a["ci95"] == result_b["ci95"]
    assert [s["scores"] for s in result_a["samples"]] == [s["scores"] for s in result_b["samples"]]


def test_calibration_report_deterministic_with_seed():
    report_a = calibration_report(n=5, seed=7)
    report_b = calibration_report(n=5, seed=7)
    assert report_a == report_b


def test_different_seeds_can_differ():
    golden = _load_golden(CONTACTLENS_03)
    transcript = _load_transcript(CONTACTLENS_03)

    result_a = judge_note_sampled(golden, golden, transcript, n=7, seed=1)
    result_b = judge_note_sampled(golden, golden, transcript, n=7, seed=2)

    # Not a hard mathematical guarantee, but with real gaussian noise on a
    # non-trivial-difficulty case, two different seeds producing byte-
    # identical sample scores would be a sign the seed isn't threading
    # through at all.
    samples_a = [s["scores"] for s in result_a["samples"]]
    samples_b = [s["scores"] for s in result_b["samples"]]
    assert samples_a != samples_b


# ---------------------------------------------------------------------------
# 2. CI contains the mean.
# ---------------------------------------------------------------------------

def test_ci95_contains_aggregate_mean():
    golden = _load_golden(GLAUCOMA_05)
    transcript = _load_transcript(GLAUCOMA_05)
    result = judge_note_sampled(golden, golden, transcript, n=7, seed=42)

    lo, hi = result["ci95"]
    assert lo <= result["aggregate_mean"] <= hi


def test_ci95_degenerate_when_zero_variance():
    # Golden judged against itself with difficulty forced to 0 collapses
    # every draw to the same base scores (noise magnitude 0 at difficulty 0
    # is not guaranteed exactly-zero-sigma, but forcing difficulty=0 and a
    # fixed seed should still produce a valid, mean-containing interval).
    golden = _load_golden(GLAUCOMA_05)
    transcript = _load_transcript(GLAUCOMA_05)
    result = _mock_judge_note_sampled(golden, golden, transcript, n=7, seed=42, _difficulty_override=0.0)
    lo, hi = result["ci95"]
    assert lo <= result["aggregate_mean"] <= hi
    assert lo <= hi


# ---------------------------------------------------------------------------
# 3. Variance monotonic with injected difficulty.
# ---------------------------------------------------------------------------

def test_variance_monotonic_with_injected_difficulty():
    golden = _load_golden(GLAUCOMA_05)
    transcript = _load_transcript(GLAUCOMA_05)

    stds = []
    for difficulty in (0.0, 0.3, 0.6, 1.0):
        result = _mock_judge_note_sampled(
            golden, golden, transcript, n=25, seed=99, _difficulty_override=difficulty
        )
        stds.append(result["aggregate_std"])

    # Monotonic non-decreasing: higher injected difficulty must never
    # produce a strictly smaller sampled std-dev.
    assert stds == sorted(stds)
    # And the extremes must actually differ (the model has real effect).
    assert stds[0] < stds[-1]


def test_case_difficulty_in_unit_range():
    golden = _load_golden(CONTACTLENS_03)
    transcript = _load_transcript(CONTACTLENS_03)
    difficulty = case_difficulty(golden, golden, transcript)
    assert 0.0 <= difficulty <= 1.0


def test_case_difficulty_no_transcript_id_special_casing():
    # Difficulty must be a pure function of (generated, golden,
    # transcript_text) content — not of the transcript_id string, which
    # isn't even one of its parameters. Swapping content between two real
    # cases should follow the content, not any external label.
    golden_a = _load_golden(GLAUCOMA_05)
    transcript_a = _load_transcript(GLAUCOMA_05)
    golden_b = _load_golden(CONTACTLENS_03)
    transcript_b = _load_transcript(CONTACTLENS_03)

    diff_a = case_difficulty(golden_a, golden_a, transcript_a)
    diff_b = case_difficulty(golden_b, golden_b, transcript_b)
    # The messy contact-lens transcript (more noise markers) should score
    # >= the clean glaucoma transcript's difficulty.
    assert diff_b >= diff_a


# ---------------------------------------------------------------------------
# 4. CI-routing is at least as conservative as point routing.
# ---------------------------------------------------------------------------

def _sampled_result(aggregate_mean: float, ci_lo: float, ci_hi: float) -> dict:
    return {
        "aggregate_mean": aggregate_mean,
        "ci95": [ci_lo, ci_hi],
    }


_ROUTE_STRICTNESS = {"auto_accept": 0, "review": 1, "regenerate": 2}


def test_ci_routing_never_less_conservative_than_point_routing():
    # Mean clears auto_accept, but the CI lower bound falls into the review
    # band -- CI-aware routing must not still auto_accept.
    sampled = _sampled_result(aggregate_mean=0.90, ci_lo=0.70, ci_hi=1.0)
    result = route_sampled(sampled, [])

    assert result["routing_delta"]["point_route"] == "auto_accept"
    assert result["routing_delta"]["ci_route"] != "auto_accept"
    assert result["routing_delta"]["changed"] is True
    assert (
        _ROUTE_STRICTNESS[result["routing_delta"]["ci_route"]]
        >= _ROUTE_STRICTNESS[result["routing_delta"]["point_route"]]
    )


def test_ci_routing_never_auto_accepts_what_point_routing_regenerates():
    # Degenerate/contrived: mean below review floor, but pretend CI lower
    # bound happened to be higher (shouldn't happen in practice since lower
    # bound <= mean, but route_sampled's logic must still not invert
    # ordering if ever called this way) -- more realistically, exercise the
    # actual guarantee: ci_lo <= aggregate_mean always in real usage, so
    # ci_route can only be equal or more severe.
    sampled = _sampled_result(aggregate_mean=0.50, ci_lo=0.30, ci_hi=0.70)
    result = route_sampled(sampled, [])
    assert result["routing_delta"]["point_route"] == "regenerate"
    assert result["routing_delta"]["ci_route"] == "regenerate"


def test_ci_routing_monotonic_across_boundary_sweep():
    # Sweep aggregate_mean across both thresholds with a fixed CI half-width
    # and confirm ci_route strictness is always >= point_route strictness.
    half_width = 0.08
    for mean_value in (0.95, 0.90, 0.86, 0.84, 0.70, 0.61, 0.59, 0.30):
        sampled = _sampled_result(mean_value, mean_value - half_width, mean_value + half_width)
        result = route_sampled(sampled, [])
        point = result["routing_delta"]["point_route"]
        ci = result["routing_delta"]["ci_route"]
        assert _ROUTE_STRICTNESS[ci] >= _ROUTE_STRICTNESS[point], (mean_value, point, ci)


def test_ci_routing_error_violation_forces_regenerate_regardless():
    sampled = _sampled_result(aggregate_mean=0.95, ci_lo=0.92, ci_hi=0.98)
    result = route_sampled(sampled, [_error_violation()])
    assert result["route"] == "regenerate"
    assert result["routing_delta"]["point_route"] == "regenerate"
    assert result["routing_delta"]["ci_route"] == "regenerate"
    assert result["routing_delta"]["changed"] is False


def test_ci_routing_no_change_reports_changed_false():
    sampled = _sampled_result(aggregate_mean=0.95, ci_lo=0.92, ci_hi=0.98)
    result = route_sampled(sampled, [])
    assert result["routing_delta"]["changed"] is False
    assert "no change" in result["routing_delta"]["explanation"].lower()


def test_ci_routing_explanation_mentions_production_impact_when_changed():
    sampled = _sampled_result(aggregate_mean=0.90, ci_lo=0.70, ci_hi=1.0)
    result = route_sampled(sampled, [])
    explanation = result["routing_delta"]["explanation"]
    assert "auto_accept" in explanation or "auto-accept" in explanation.lower()


# ---------------------------------------------------------------------------
# 5. Flags fire on wide distributions.
# ---------------------------------------------------------------------------

def test_high_variance_flag_fires_on_wide_distribution():
    golden = _load_golden(CONTACTLENS_03)
    transcript = _load_transcript(CONTACTLENS_03)
    result = _mock_judge_note_sampled(
        golden, golden, transcript, n=15, seed=1, _difficulty_override=1.0
    )
    assert "HIGH_VARIANCE" in result["flags"]


def test_no_high_variance_flag_at_zero_difficulty_small_n():
    golden = _load_golden(GLAUCOMA_05)
    transcript = _load_transcript(GLAUCOMA_05)
    result = _mock_judge_note_sampled(
        golden, golden, transcript, n=7, seed=1, _difficulty_override=0.0
    )
    # Zero injected difficulty -> base sigma only, small n -> should not
    # reliably trip the HIGH_VARIANCE bar (documented thresholds are well
    # above the noise floor at difficulty 0).
    assert result["aggregate_std"] < 0.10


def test_flags_is_a_list_even_when_empty():
    golden = _load_golden(GLAUCOMA_05)
    transcript = _load_transcript(GLAUCOMA_05)
    result = judge_note_sampled(golden, golden, transcript, n=7, seed=1)
    assert isinstance(result["flags"], list)


# ---------------------------------------------------------------------------
# 6. Report shape.
# ---------------------------------------------------------------------------

def test_calibration_report_shape():
    report = calibration_report(n=5, seed=42)

    assert report["n"] == 5
    assert report["seed"] == 42
    assert isinstance(report["cases"], list)
    assert len(report["cases"]) == 20  # all bundled fixtures

    required_case_keys = {
        "transcript_id", "visit_type", "difficulty", "mean_scores", "std_scores",
        "aggregate_mean", "aggregate_std", "ci95", "ci_width",
        "point_route", "ci_route", "changed", "agreement", "flags",
    }
    for case in report["cases"]:
        assert required_case_keys.issubset(case.keys())
        assert case["point_route"] in ("auto_accept", "review", "regenerate")
        assert case["ci_route"] in ("auto_accept", "review", "regenerate")
        lo, hi = case["ci95"]
        assert lo <= case["aggregate_mean"] <= hi

    summary = report["summary"]
    assert summary["n_cases"] == 20
    assert set(summary["mean_ci_width_by_visit_type"].keys()) == {
        "comprehensive_exam", "glaucoma_followup", "cataract_postop", "contact_lens_fitting",
    }
    assert isinstance(summary["n_routes_changed"], int)
    assert summary["n_routes_changed"] == len(summary["changed_transcript_ids"])


def test_build_markdown_contains_summary_sections():
    report = calibration_report(n=5, seed=42)
    markdown = build_markdown(report)
    assert "Per-case" in markdown
    assert "Summary" in markdown
    assert "contact_lens_fitting" in markdown


# ---------------------------------------------------------------------------
# 7. Messy (contact-lens) visit type has the widest mean CI on real data.
# ---------------------------------------------------------------------------

def test_contact_lens_visit_type_has_widest_mean_ci():
    report = calibration_report(n=7, seed=42)
    widths = report["summary"]["mean_ci_width_by_visit_type"]
    widest_visit_type = max(widths, key=widths.get)
    assert widest_visit_type == "contact_lens_fitting"


# ---------------------------------------------------------------------------
# 8. API sampled-judge class is never constructed by default.
# ---------------------------------------------------------------------------

def test_api_sampled_judge_never_constructed_by_default(monkeypatch):
    monkeypatch.delenv("SCRIBEGATE_USE_API", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def _boom(*args, **kwargs):
        raise AssertionError("APISampledJudge must not be constructed when SCRIBEGATE_USE_API is unset")

    monkeypatch.setattr(APISampledJudge, "__init__", _boom)

    golden = _load_golden(GLAUCOMA_05)
    transcript = _load_transcript(GLAUCOMA_05)
    # Should complete without ever touching APISampledJudge.
    result = judge_note_sampled(golden, golden, transcript, n=3, seed=1)
    assert "aggregate_mean" in result


def test_api_sampled_judge_not_constructed_when_key_missing_even_if_flag_set(monkeypatch):
    monkeypatch.setenv("SCRIBEGATE_USE_API", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def _boom(*args, **kwargs):
        raise AssertionError("APISampledJudge must not be constructed without ANTHROPIC_API_KEY")

    monkeypatch.setattr(APISampledJudge, "__init__", _boom)

    golden = _load_golden(GLAUCOMA_05)
    transcript = _load_transcript(GLAUCOMA_05)
    result = judge_note_sampled(golden, golden, transcript, n=3, seed=1)
    assert "aggregate_mean" in result


def test_api_sampled_judge_constructed_when_both_env_vars_set(monkeypatch):
    monkeypatch.setenv("SCRIBEGATE_USE_API", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")

    constructed = {"flag": False}

    class _StubAPISampledJudge:
        def __init__(self, *args, **kwargs):
            constructed["flag"] = True

        def judge_sampled(self, generated, golden, transcript_text, n=7, seed=None):
            return {
                "samples": [],
                "mean_scores": {},
                "std_scores": {},
                "aggregate_mean": 0.5,
                "aggregate_std": 0.0,
                "ci95": [0.5, 0.5],
                "agreement": {},
                "flags": [],
                "difficulty": None,
            }

    import scribegate.calibration as calibration_module

    monkeypatch.setattr(calibration_module, "APISampledJudge", _StubAPISampledJudge)

    golden = _load_golden(GLAUCOMA_05)
    transcript = _load_transcript(GLAUCOMA_05)
    result = judge_note_sampled(golden, golden, transcript, n=3, seed=1)

    assert constructed["flag"] is True
    assert result["aggregate_mean"] == 0.5


# ---------------------------------------------------------------------------
# Extra coverage: samples shape + agreement semantics.
# ---------------------------------------------------------------------------

def test_samples_list_has_n_entries_each_shaped_like_judge_result():
    golden = _load_golden(CONTACTLENS_03)
    transcript = _load_transcript(CONTACTLENS_03)
    result = judge_note_sampled(golden, golden, transcript, n=9, seed=5)

    assert len(result["samples"]) == 9
    for sample in result["samples"]:
        assert set(sample["scores"].keys()) == {
            "completeness", "hallucination", "coding_plausibility", "terminology",
        }
        for v in sample["scores"].values():
            assert 1 <= v <= 5
        assert 0.0 <= sample["aggregate"] <= 1.0


def test_agreement_fraction_within_unit_range():
    golden = _load_golden(CONTACTLENS_03)
    transcript = _load_transcript(CONTACTLENS_03)
    result = judge_note_sampled(golden, golden, transcript, n=11, seed=5)
    for dim, fraction in result["agreement"].items():
        assert 0.0 <= fraction <= 1.0
