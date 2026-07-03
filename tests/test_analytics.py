"""Tests for scribegate.analytics: failure_modes, routing_summary, roi_model,
dimension_matrix, and the load_results convenience helper.

Tested against the U3 task spec (analytics + ROI backend for a future
Streamlit UI worker) and cross-checked against the real fixtures in
data/results/*.json (20 per-transcript result records written by
scribegate.cli / scribegate.benchmark) and data/golden_notes/*.json, per
specs/INTERFACES.md's Note dict shape and judge_note output shape.

Path resolution matches the existing convention in tests/test_judge.py and
tests/test_cli.py: `Path(__file__).resolve().parent.parent / "data"`.
"""

import json
from pathlib import Path

import pytest

from scribegate.analytics import (
    RoiParams,
    dimension_matrix,
    failure_modes,
    roi_model,
    routing_summary,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
GOLDEN_DIR = DATA_DIR / "golden_notes"


def _load_real_results() -> list[dict]:
    """Load the 20 real result fixtures from data/results/*.json, excluding
    non-per-transcript files (benchmark.md, decision_log.jsonl) — mirrors
    the exclusion logic in scribegate.benchmark.load_results."""
    results = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError:
                continue
        if isinstance(data, dict) and "transcript_id" in data and "judge_result" in data:
            results.append(data)
    return results


REAL_RESULTS = _load_real_results()


# ---------------------------------------------------------------------------
# Sanity: fixture loading itself
# ---------------------------------------------------------------------------

def test_real_results_fixture_count_is_20():
    assert len(REAL_RESULTS) == 20


# ---------------------------------------------------------------------------
# failure_modes — structural invariants
# ---------------------------------------------------------------------------

def test_failure_modes_by_dimension_always_has_4_entries_in_fixed_order():
    result = failure_modes(REAL_RESULTS, golden_notes_dir=str(GOLDEN_DIR))
    dims = [entry["dimension"] for entry in result["by_dimension"]]
    assert dims == ["completeness", "hallucination", "coding_plausibility", "terminology"]
    for entry in result["by_dimension"]:
        assert "count_le_3" in entry
        assert "visit_types" in entry
        assert "transcript_ids" in entry
        assert entry["visit_types"] == sorted(entry["visit_types"])
        assert entry["transcript_ids"] == sorted(entry["transcript_ids"])


def test_failure_modes_by_section_always_has_4_entries_in_fixed_order():
    result = failure_modes(REAL_RESULTS, golden_notes_dir=str(GOLDEN_DIR))
    sections = [entry["section"] for entry in result["by_section"]]
    assert sections == ["S", "O", "A", "P"]
    for entry in result["by_section"]:
        assert isinstance(entry["mean_line_count_delta"], float)
        assert isinstance(entry["transcripts_below_golden"], list)
        assert entry["transcripts_below_golden"] == sorted(entry["transcripts_below_golden"])


def test_failure_modes_worst_cases_at_most_5_and_ascending_by_aggregate():
    result = failure_modes(REAL_RESULTS, golden_notes_dir=str(GOLDEN_DIR))
    worst_cases = result["worst_cases"]
    assert len(worst_cases) <= 5
    aggregates = [wc["aggregate"] for wc in worst_cases]
    assert aggregates == sorted(aggregates)
    for wc in worst_cases:
        assert set(wc.keys()) == {"transcript_id", "visit_type", "aggregate", "route", "reasons"}
        assert isinstance(wc["reasons"], list)


# ---------------------------------------------------------------------------
# failure_modes — real-data violation check (cataract_03 / cataract_05)
# ---------------------------------------------------------------------------

def test_cataract_03_has_iop_range_warn_and_laterality_conflict_error_in_source_file():
    # Verify directly from the raw fixture file before trusting the
    # analytics output built on top of it.
    with open(RESULTS_DIR / "cataract_03.json", "r", encoding="utf-8") as fh:
        data = json.load(fh)
    codes_severities = {(v["code"], v["severity"]) for v in data["violations"]}
    assert ("IOP_RANGE", "warn") in codes_severities
    assert ("LATERALITY_CONFLICT", "error") in codes_severities


def test_failure_modes_by_violation_code_reflects_real_data():
    result = failure_modes(REAL_RESULTS, golden_notes_dir=str(GOLDEN_DIR))
    by_code = {entry["code"]: entry for entry in result["by_violation_code"]}

    assert "IOP_RANGE" in by_code
    assert by_code["IOP_RANGE"]["max_severity"] == "warn"
    assert "cataract_03" in by_code["IOP_RANGE"]["transcript_ids"]

    assert "LATERALITY_CONFLICT" in by_code
    assert by_code["LATERALITY_CONFLICT"]["max_severity"] == "error"
    assert "cataract_03" in by_code["LATERALITY_CONFLICT"]["transcript_ids"]
    # cataract_05 also has a LATERALITY_CONFLICT error per manual inspection.
    assert "cataract_05" in by_code["LATERALITY_CONFLICT"]["transcript_ids"]

    # by_violation_code sorted by count desc then code asc.
    counts_and_codes = [(entry["count"], entry["code"]) for entry in result["by_violation_code"]]
    assert counts_and_codes == sorted(counts_and_codes, key=lambda cc: (-cc[0], cc[1]))


# ---------------------------------------------------------------------------
# routing_summary — real data
# ---------------------------------------------------------------------------

def test_routing_summary_real_results_total_and_counts():
    result = routing_summary(REAL_RESULTS)
    assert result["total"] == 20

    counted = sum(v["count"] for v in result["by_route"].values())
    assert counted == 20

    rates_sum = sum(v["rate"] for v in result["by_route"].values())
    assert rates_sum == pytest.approx(1.0, abs=1e-6)

    assert result["review_queue_depth"] == (
        result["by_route"]["review"]["count"] + result["by_route"]["regenerate"]["count"]
    )


def test_routing_summary_known_routes_always_present():
    result = routing_summary([])
    assert set(result["by_route"].keys()) >= {"auto_accept", "review", "regenerate"}
    assert result["total"] == 0
    assert result["mean_aggregate_overall"] is None
    for route_stats in result["by_route"].values():
        assert route_stats["count"] == 0
        assert route_stats["rate"] == 0.0
        assert route_stats["mean_aggregate"] is None


# ---------------------------------------------------------------------------
# roi_model — hand-computed exact synthetic scenario
# ---------------------------------------------------------------------------

def test_roi_model_hand_computed_exact_scenario():
    routing = {
        "total": 100,
        "by_route": {
            "auto_accept": {"count": 80, "rate": 0.8, "mean_aggregate": 0.9},
            "review": {"count": 15, "rate": 0.15, "mean_aggregate": 0.7},
            "regenerate": {"count": 5, "rate": 0.05, "mean_aggregate": 0.3},
        },
        "review_queue_depth": 20,
        "mean_aggregate_overall": 0.83,
    }

    result = roi_model(routing, RoiParams())

    # Independent hand arithmetic (not re-deriving the same code path):
    # notes_per_month = 4 providers * 22 visits/day/provider * 21 days = 1848
    # auto_accept_notes = round(1848 * 0.8) = round(1478.4) = 1478
    # review_notes      = round(1848 * 0.15) = round(277.2) = 277
    # regenerate_notes  = round(1848 * 0.05) = round(92.4)  = 92
    # hours_without_gate = 1848 * 4.0 / 60 = 7392 / 60 = 123.2
    # hours_with_gate = (1478*0.5 + 277*4.0 + 92*(2.0+4.0)) / 60
    #                 = (739 + 1108 + 552) / 60 = 2399 / 60 = 39.9833...
    # hours_saved = 123.2 - 39.9833... = 83.21666...
    # dollars_saved = 83.21666... * 140 = 11650.3333...
    assert result["notes_per_month"] == 1848
    assert result["notes_by_route"] == {"auto_accept": 1478, "review": 277, "regenerate": 92}
    assert result["hours_without_gate"] == pytest.approx(123.2, abs=1e-6)
    assert result["hours_with_gate"] == pytest.approx(40.0, abs=1e-6)
    assert result["hours_saved_per_month"] == pytest.approx(83.2, abs=1e-6)
    assert result["dollars_saved_per_month"] == pytest.approx(11650.33, abs=1e-2)

    assumptions = result["assumptions"]
    assert assumptions["auto_accept_rate"] == 0.8
    assert assumptions["review_rate"] == 0.15
    assert assumptions["regenerate_rate"] == 0.05
    assert "regenerate_handling" in assumptions["notes"]
    assert "full_review" in assumptions["notes"]


def test_roi_model_with_overridden_params_changes_output():
    routing = {
        "total": 100,
        "by_route": {
            "auto_accept": {"count": 80, "rate": 0.8, "mean_aggregate": 0.9},
            "review": {"count": 15, "rate": 0.15, "mean_aggregate": 0.7},
            "regenerate": {"count": 5, "rate": 0.05, "mean_aggregate": 0.3},
        },
        "review_queue_depth": 20,
        "mean_aggregate_overall": 0.83,
    }
    default_result = roi_model(routing, RoiParams())

    custom_params = RoiParams(
        clinician_hourly_cost=200.0,
        minutes_full_review=5.0,
        minutes_spot_check=1.0,
        minutes_regenerate_handling=3.0,
    )
    custom_result = roi_model(routing, custom_params)

    assert custom_result["dollars_saved_per_month"] != default_result["dollars_saved_per_month"]
    assert custom_result["hours_with_gate"] != default_result["hours_with_gate"]
    assert custom_result["hours_without_gate"] != default_result["hours_without_gate"]

    assumptions = custom_result["assumptions"]
    assert assumptions["clinician_hourly_cost"] == 200.0
    assert assumptions["minutes_full_review"] == 5.0
    assert assumptions["minutes_spot_check"] == 1.0
    assert assumptions["minutes_regenerate_handling"] == 3.0


def test_roi_model_empty_routing_never_crashes_and_zeroes_out():
    for empty_routing in ({}, {"total": 0, "by_route": {}}, {"total": 0}):
        result = roi_model(empty_routing, RoiParams())
        assert result["notes_by_route"] == {"auto_accept": 0, "review": 0, "regenerate": 0}
        assert result["hours_with_gate"] == 0.0
        assert result["hours_saved_per_month"] == result["hours_without_gate"]
        assert result["dollars_saved_per_month"] == pytest.approx(
            result["hours_without_gate"] * RoiParams().clinician_hourly_cost, abs=1e-2
        )
        assert isinstance(result["assumptions"]["notes"], str)
        assert len(result["assumptions"]["notes"]) > 0


# ---------------------------------------------------------------------------
# Empty-results handling across all 4 functions
# ---------------------------------------------------------------------------

def test_failure_modes_empty_results_well_formed():
    result = failure_modes([], golden_notes_dir=str(GOLDEN_DIR))
    assert len(result["by_dimension"]) == 4
    assert all(entry["count_le_3"] == 0 for entry in result["by_dimension"])
    assert result["by_violation_code"] == []
    assert len(result["by_section"]) == 4
    assert all(entry["mean_line_count_delta"] == 0.0 for entry in result["by_section"])
    assert result["worst_cases"] == []


def test_routing_summary_empty_results_well_formed():
    result = routing_summary([])
    assert result["total"] == 0
    assert result["review_queue_depth"] == 0
    assert result["mean_aggregate_overall"] is None


def test_dimension_matrix_empty_results_well_formed():
    result = dimension_matrix([])
    assert result["visit_types"] == []
    assert result["dimensions"] == [
        "completeness", "hallucination", "coding_plausibility", "terminology"
    ]
    assert result["grid"] == []
    assert result["rows"] == []


def test_failure_modes_missing_golden_files_skipped_not_crashed():
    fake_result = {
        "transcript_id": "nonexistent_transcript_xyz",
        "visit_type": "comprehensive_exam",
        "generated_note": {"soap": {"S": [{"text": "x", "spans": [[0, 1]]}], "O": [], "A": [], "P": []}},
        "judge_result": {
            "scores": {"completeness": 5, "hallucination": 5, "coding_plausibility": 5, "terminology": 5},
            "aggregate": 1.0,
            "rationales": {},
        },
        "violations": [],
        "route": "auto_accept",
    }
    # Should not raise even though no golden file exists for this transcript_id.
    result = failure_modes([fake_result], golden_notes_dir=str(GOLDEN_DIR))
    assert len(result["by_section"]) == 4
    for entry in result["by_section"]:
        assert entry["mean_line_count_delta"] == 0.0
        assert entry["transcripts_below_golden"] == []


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_failure_modes_deterministic():
    r1 = failure_modes(REAL_RESULTS, golden_notes_dir=str(GOLDEN_DIR))
    r2 = failure_modes(REAL_RESULTS, golden_notes_dir=str(GOLDEN_DIR))
    assert r1 == r2


def test_routing_summary_deterministic():
    r1 = routing_summary(REAL_RESULTS)
    r2 = routing_summary(REAL_RESULTS)
    assert r1 == r2


def test_roi_model_deterministic():
    routing = routing_summary(REAL_RESULTS)
    r1 = roi_model(routing, RoiParams())
    r2 = roi_model(routing, RoiParams())
    assert r1 == r2


def test_dimension_matrix_deterministic():
    r1 = dimension_matrix(REAL_RESULTS)
    r2 = dimension_matrix(REAL_RESULTS)
    assert r1 == r2


# ---------------------------------------------------------------------------
# dimension_matrix — real data
# ---------------------------------------------------------------------------

def test_dimension_matrix_real_results_grid_has_4_visit_types():
    result = dimension_matrix(REAL_RESULTS)
    assert result["visit_types"] == sorted(
        ["comprehensive_exam", "glaucoma_followup", "cataract_postop", "contact_lens_fitting"]
    )
    assert len(result["grid"]) == 4
    assert sum(row["n"] for row in result["grid"]) == 20


def test_dimension_matrix_real_results_rows_length_20_and_sorted():
    result = dimension_matrix(REAL_RESULTS)
    assert len(result["rows"]) == 20
    ids = [row["transcript_id"] for row in result["rows"]]
    assert ids == sorted(ids)


def test_dimension_matrix_rows_match_source_judge_result_scores():
    result = dimension_matrix(REAL_RESULTS)
    rows_by_id = {row["transcript_id"]: row for row in result["rows"]}
    for r in REAL_RESULTS:
        tid = r["transcript_id"]
        row = rows_by_id[tid]
        scores = r["judge_result"]["scores"]
        assert row["completeness"] == scores["completeness"]
        assert row["hallucination"] == scores["hallucination"]
        assert row["coding_plausibility"] == scores["coding_plausibility"]
        assert row["terminology"] == scores["terminology"]
        assert row["aggregate"] == pytest.approx(r["judge_result"]["aggregate"], abs=1e-6)
