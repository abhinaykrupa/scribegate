"""Tests for scribegate.drift (U2 — drift detection + CI eval gate) and the
generator's quality knob / cli history-append wiring that feed it.

Covers: quality-knob determinism, backward-compatible no-kwarg call,
degraded-vs-baseline divergence, history row schema, load_history parsing
(including malformed-line tolerance), detect_regression firing/not-firing on
synthetic time series, and check_against_baseline pass/fail behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scribegate import cli
from scribegate.generator import generate_note, visit_type_for
from scribegate.drift import (
    Alert,
    load_history,
    detect_regression,
    summarize_drift,
    check_against_baseline,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = REPO_ROOT / "data" / "transcripts"

GLAUCOMA_01 = "glaucoma_01"


def _transcript_text(transcript_id: str) -> str:
    return (TRANSCRIPT_DIR / f"{transcript_id}.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Quality knob determinism (baseline and degraded).
# ---------------------------------------------------------------------------

def test_quality_knob_deterministic_baseline():
    text = _transcript_text(GLAUCOMA_01)
    vt = visit_type_for(GLAUCOMA_01)
    a = generate_note(text, GLAUCOMA_01, vt, quality="baseline")
    b = generate_note(text, GLAUCOMA_01, vt, quality="baseline")
    assert a == b


def test_quality_knob_deterministic_degraded():
    text = _transcript_text(GLAUCOMA_01)
    vt = visit_type_for(GLAUCOMA_01)
    a = generate_note(text, GLAUCOMA_01, vt, quality="degraded")
    b = generate_note(text, GLAUCOMA_01, vt, quality="degraded")
    assert a == b


# ---------------------------------------------------------------------------
# 2. No-kwarg call is byte-identical to explicit quality="baseline".
# ---------------------------------------------------------------------------

def test_no_kwarg_call_matches_explicit_baseline():
    text = _transcript_text(GLAUCOMA_01)
    vt = visit_type_for(GLAUCOMA_01)
    no_kwarg = generate_note(text, GLAUCOMA_01, vt)
    explicit = generate_note(text, GLAUCOMA_01, vt, quality="baseline")
    assert no_kwarg == explicit
    assert json.dumps(no_kwarg, sort_keys=True) == json.dumps(explicit, sort_keys=True)


# ---------------------------------------------------------------------------
# 3. Degraded measurably differs from baseline (concrete assertion).
# ---------------------------------------------------------------------------

def test_degraded_differs_measurably_from_baseline():
    text = _transcript_text(GLAUCOMA_01)
    vt = visit_type_for(GLAUCOMA_01)
    baseline = generate_note(text, GLAUCOMA_01, vt, quality="baseline")
    degraded = generate_note(text, GLAUCOMA_01, vt, quality="degraded")

    assert baseline != degraded

    def _n_lines(note):
        return sum(len(v) for v in note["soap"].values())

    # glaucoma_01 is known (empirically, see drift.py seeding run) to drop at
    # least one more line under degraded than baseline.
    assert _n_lines(degraded) < _n_lines(baseline)


# ---------------------------------------------------------------------------
# 4. History row schema after cli.run + append_history_row.
# ---------------------------------------------------------------------------

def test_history_row_schema(tmp_path):
    results_dir = tmp_path / "results"
    history_path = tmp_path / "history.jsonl"
    results = cli.run(["glaucoma_01", "cataract_01"], results_dir=results_dir, quality="baseline")
    cli.append_history_row(results, tag="baseline", quality="baseline", history_path=history_path)

    rows = load_history(history_path)
    assert len(rows) == 1
    row = rows[0]

    assert isinstance(row["overall_aggregate"], float)
    assert isinstance(row["per_visit_type"], dict)
    assert set(row["per_visit_type"]) == {
        "comprehensive_exam",
        "glaucoma_followup",
        "cataract_postop",
        "contact_lens_fitting",
    }
    assert all(isinstance(v, float) for v in row["per_visit_type"].values())
    assert isinstance(row["per_dimension"], dict)
    assert set(row["per_dimension"]) == {
        "completeness",
        "hallucination",
        "coding_plausibility",
        "terminology",
    }
    assert row["n_notes"] == 2
    assert isinstance(row["auto_accept_rate"], float)
    assert isinstance(row["ts"], str) and row["ts"]
    assert row["tag"] == "baseline"
    assert row["quality"] == "baseline"


# ---------------------------------------------------------------------------
# 5. load_history parses well-formed jsonl.
# ---------------------------------------------------------------------------

def test_load_history_parses_well_formed_jsonl(tmp_path):
    path = tmp_path / "history.jsonl"
    rows = [
        {"overall_aggregate": 0.9, "tag": "baseline", "quality": "baseline"},
        {"overall_aggregate": 0.8, "tag": "degraded", "quality": "degraded"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    loaded = load_history(path)
    assert loaded == rows


# ---------------------------------------------------------------------------
# 6. load_history skips malformed lines without crashing.
# ---------------------------------------------------------------------------

def test_load_history_skips_malformed_lines(tmp_path):
    path = tmp_path / "history.jsonl"
    content = (
        json.dumps({"overall_aggregate": 0.9, "tag": "baseline"}) + "\n"
        "{not valid json,\n"
        "\n"
        "42\n"
        + json.dumps({"overall_aggregate": 0.8, "tag": "degraded"}) + "\n"
    )
    path.write_text(content, encoding="utf-8")

    loaded = load_history(path)
    assert len(loaded) == 2
    assert loaded[0]["tag"] == "baseline"
    assert loaded[1]["tag"] == "degraded"


def test_load_history_missing_file_returns_empty(tmp_path):
    loaded = load_history(tmp_path / "does_not_exist.jsonl")
    assert loaded == []


# ---------------------------------------------------------------------------
# 7. detect_regression fires an Alert on a synthetic clear-drop history.
# ---------------------------------------------------------------------------

def _synthetic_row(agg: float, tag: str, quality: str, ts: str) -> dict:
    return {
        "overall_aggregate": agg,
        "per_visit_type": {
            "comprehensive_exam": agg,
            "glaucoma_followup": agg,
            "cataract_postop": agg,
            "contact_lens_fitting": agg,
        },
        "per_dimension": {
            "completeness": agg,
            "hallucination": agg,
            "coding_plausibility": agg,
            "terminology": agg,
        },
        "n_notes": 20,
        "auto_accept_rate": 0.5,
        "ts": ts,
        "tag": tag,
        "quality": quality,
    }


def test_detect_regression_fires_on_clear_drop():
    history = [
        _synthetic_row(0.9, "baseline", "baseline", "2026-01-01T00:00:00Z"),
        _synthetic_row(0.7, "degraded", "degraded", "2026-01-02T00:00:00Z"),
        _synthetic_row(0.7, "degraded", "degraded", "2026-01-03T00:00:00Z"),
        _synthetic_row(0.7, "degraded", "degraded", "2026-01-04T00:00:00Z"),
    ]
    alerts = detect_regression(history, window=3, threshold=0.05)

    overall_alerts = [a for a in alerts if a.metric == "overall"]
    assert len(overall_alerts) == 1
    alert = overall_alerts[0]
    assert isinstance(alert, Alert)
    assert alert.baseline_value == pytest.approx(0.9)
    assert alert.current_value == pytest.approx(0.7)
    assert alert.drop == pytest.approx(0.2)
    assert alert.drop >= 0.05
    assert alert.message


# ---------------------------------------------------------------------------
# 8. detect_regression returns empty list on stable/improving history.
# ---------------------------------------------------------------------------

def test_detect_regression_empty_on_stable_history():
    history = [
        _synthetic_row(0.85, "baseline", "baseline", "2026-01-01T00:00:00Z"),
        _synthetic_row(0.86, "run2", "baseline", "2026-01-02T00:00:00Z"),
        _synthetic_row(0.90, "run3", "baseline", "2026-01-03T00:00:00Z"),
        _synthetic_row(0.88, "run4", "baseline", "2026-01-04T00:00:00Z"),
    ]
    alerts = detect_regression(history, window=3, threshold=0.05)
    assert alerts == []


# ---------------------------------------------------------------------------
# 9/10. check_against_baseline pass / fail.
# ---------------------------------------------------------------------------

def test_check_against_baseline_passes_when_meeting_floor():
    baseline = {
        "overall_aggregate": 0.78,
        "per_dimension": {
            "completeness": 0.45,
            "hallucination": 0.85,
            "coding_plausibility": 0.89,
            "terminology": 0.93,
        },
    }
    current = {
        "overall_aggregate": 0.815,
        "per_dimension": {
            "completeness": 0.49,
            "hallucination": 0.89,
            "coding_plausibility": 0.93,
            "terminology": 0.96,
        },
    }
    passed, failures = check_against_baseline(current, baseline)
    assert passed is True
    assert failures == []


def test_check_against_baseline_fails_when_below_floor():
    baseline = {
        "overall_aggregate": 0.78,
        "per_dimension": {
            "completeness": 0.45,
            "hallucination": 0.85,
            "coding_plausibility": 0.89,
            "terminology": 0.93,
        },
    }
    current = {
        "overall_aggregate": 0.70,
        "per_dimension": {
            "completeness": 0.40,
            "hallucination": 0.49,
            "coding_plausibility": 0.93,
            "terminology": 0.96,
        },
    }
    passed, failures = check_against_baseline(current, baseline)
    assert passed is False
    assert len(failures) >= 1
    assert any("overall_aggregate" in msg for msg in failures)
    assert any("completeness" in msg for msg in failures)
    assert any("hallucination" in msg for msg in failures)


# ---------------------------------------------------------------------------
# Bonus: summarize_drift shape.
# ---------------------------------------------------------------------------

def test_summarize_drift_shape():
    history = [
        _synthetic_row(0.9, "baseline", "baseline", "2026-01-01T00:00:00Z"),
        _synthetic_row(0.8, "degraded", "degraded", "2026-01-02T00:00:00Z"),
    ]
    summary = summarize_drift(history)

    assert "overall" in summary
    assert "dimension:completeness" in summary
    assert "visit_type:glaucoma_followup" in summary

    overall_series = summary["overall"]
    assert len(overall_series) == 2
    assert overall_series[0]["ts"] == "2026-01-01T00:00:00Z"
    assert overall_series[0]["tag"] == "baseline"
    assert overall_series[0]["value"] == pytest.approx(0.9)
    assert overall_series[1]["value"] == pytest.approx(0.8)
