"""Tests for scribegate.corrections (U4): recording line-level corrections,
appending to candidate_golden.jsonl / decision_log.jsonl, merging into a
candidate golden note, and computing correction stats.

Every test sets SCRIBEGATE_RESULTS_DIR to a tmp_path-derived directory so
none of them ever touch the real data/results/ directory.
"""

import json

import pytest

from scribegate import corrections


def _write_result(results_dir, transcript_id="glaucoma_05", visit_type="glaucoma_followup", lines=None):
    results_dir.mkdir(parents=True, exist_ok=True)
    soap = lines or {
        "S": [{"text": "Patient reports mild irritation.", "spans": [[0, 10]]}],
        "O": [{"text": "IOP 18 mmHg OD.", "spans": [[20, 30]]}],
        "A": [{"text": "Stable glaucoma.", "spans": [[40, 50]]}],
        "P": [{"text": "Continue current drops.", "spans": [[60, 70]]}],
    }
    payload = {
        "transcript_id": transcript_id,
        "visit_type": visit_type,
        "generated_note": {
            "transcript_id": transcript_id, "visit_type": visit_type, "synthetic": True,
            "soap": soap, "generated": True, "generator": "mock",
        },
        "judge_result": {"scores": {"completeness": 4, "hallucination": 5, "coding_plausibility": 4, "terminology": 5},
                          "aggregate": 0.8, "rationales": {"completeness": "ok"}},
        "violations": [],
        "route": "review",
        "decision_reasons": ["aggregate 0.800 in [0.60, 0.85) — routed to human review"],
        "timestamps": {"generated_at": "2026-07-02T15:00:00Z"},
    }
    (results_dir / f"{transcript_id}.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


# ---------------------------------------------------------------------------
# record_correction
# ---------------------------------------------------------------------------

def test_record_correction_happy_path_writes_both_files(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _write_result(results_dir)

    record = corrections.record_correction(
        transcript_id="glaucoma_05",
        section="O",
        line_index=0,
        original_text="IOP 18 mmHg OD.",
        corrected_text="IOP 20 mmHg OD.",
        reviewer="dr_smith",
        note="corrected typo in IOP reading",
    )

    assert record["transcript_id"] == "glaucoma_05"
    assert record["section"] == "O"
    assert record["line_index"] == 0
    assert record["original_text"] == "IOP 18 mmHg OD."
    assert record["corrected_text"] == "IOP 20 mmHg OD."
    assert record["reviewer"] == "dr_smith"
    assert record["visit_type"] == "glaucoma_followup"
    assert "correction_id" in record and len(record["correction_id"]) == 16
    assert record["spans"] == [[20, 30]]

    candidate_path = results_dir / "candidate_golden.jsonl"
    assert candidate_path.exists()
    lines = candidate_path.read_text().strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["correction_id"] == record["correction_id"]
    assert parsed["note"] == "corrected typo in IOP reading"

    decision_log_path = results_dir / "decision_log.jsonl"
    assert decision_log_path.exists()
    log_lines = decision_log_path.read_text().strip().splitlines()
    assert len(log_lines) == 1
    log_entry = json.loads(log_lines[0])
    assert log_entry["transcript_id"] == "glaucoma_05"
    assert log_entry["event"] == "correction"
    assert log_entry["correction_id"] == record["correction_id"]
    assert log_entry["reviewer"] == "dr_smith"
    assert log_entry["section"] == "O"
    assert log_entry["line_index"] == 0
    assert "ts" in log_entry


def test_record_correction_rejects_mismatched_original_text(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _write_result(results_dir)

    with pytest.raises(ValueError):
        corrections.record_correction(
            transcript_id="glaucoma_05",
            section="O",
            line_index=0,
            original_text="this does not match the actual line text",
            corrected_text="IOP 20 mmHg OD.",
            reviewer="dr_smith",
        )


def test_record_correction_rejects_out_of_range_line_index(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _write_result(results_dir)

    with pytest.raises(ValueError):
        corrections.record_correction(
            transcript_id="glaucoma_05",
            section="O",
            line_index=99,
            original_text="IOP 18 mmHg OD.",
            corrected_text="IOP 20 mmHg OD.",
            reviewer="dr_smith",
        )


def test_record_correction_rejects_bad_section(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _write_result(results_dir)

    with pytest.raises(ValueError):
        corrections.record_correction(
            transcript_id="glaucoma_05",
            section="X",
            line_index=0,
            original_text="IOP 18 mmHg OD.",
            corrected_text="IOP 20 mmHg OD.",
            reviewer="dr_smith",
        )


def test_record_correction_missing_result_file_raises(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    results_dir.mkdir(parents=True)

    with pytest.raises(ValueError):
        corrections.record_correction(
            transcript_id="does_not_exist",
            section="O",
            line_index=0,
            original_text="anything",
            corrected_text="anything else",
            reviewer="dr_smith",
        )


# ---------------------------------------------------------------------------
# Append-only behavior
# ---------------------------------------------------------------------------

def test_candidate_golden_is_append_only(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _write_result(results_dir)

    corrections.record_correction(
        transcript_id="glaucoma_05", section="O", line_index=0,
        original_text="IOP 18 mmHg OD.", corrected_text="IOP 20 mmHg OD.",
        reviewer="dr_smith",
    )

    candidate_path = results_dir / "candidate_golden.jsonl"
    first_read = candidate_path.read_text().strip().splitlines()
    assert len(first_read) == 1

    corrections.record_correction(
        transcript_id="glaucoma_05", section="A", line_index=0,
        original_text="Stable glaucoma.", corrected_text="Improving glaucoma.",
        reviewer="dr_jones",
    )

    second_read = candidate_path.read_text().strip().splitlines()
    assert len(second_read) == 2
    for line in second_read:
        json.loads(line)  # both parseable
    assert second_read[0] == first_read[0]  # byte-identical, untouched


# ---------------------------------------------------------------------------
# build_candidate_golden
# ---------------------------------------------------------------------------

def test_build_candidate_golden_merge_correctness(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _write_result(results_dir)

    r1 = corrections.record_correction(
        transcript_id="glaucoma_05", section="O", line_index=0,
        original_text="IOP 18 mmHg OD.", corrected_text="IOP 20 mmHg OD.",
        reviewer="dr_smith",
    )
    r2 = corrections.record_correction(
        transcript_id="glaucoma_05", section="P", line_index=0,
        original_text="Continue current drops.", corrected_text="Increase drop frequency.",
        reviewer="dr_smith",
    )

    note = corrections.build_candidate_golden("glaucoma_05")
    assert note is not None
    assert note["candidate"] is True
    assert note["source_corrections"] == [r1["correction_id"], r2["correction_id"]]

    assert note["soap"]["O"][0]["text"] == "IOP 20 mmHg OD."
    assert note["soap"]["P"][0]["text"] == "Increase drop frequency."
    # Untouched lines/sections remain as generated.
    assert note["soap"]["S"][0]["text"] == "Patient reports mild irritation."
    assert note["soap"]["A"][0]["text"] == "Stable glaucoma."


def test_build_candidate_golden_returns_none_without_corrections(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _write_result(results_dir)

    assert corrections.build_candidate_golden("glaucoma_05") is None


def test_build_candidate_golden_last_correction_wins_for_same_line(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _write_result(results_dir)

    corrections.record_correction(
        transcript_id="glaucoma_05", section="O", line_index=0,
        original_text="IOP 18 mmHg OD.", corrected_text="IOP 19 mmHg OD.",
        reviewer="dr_smith",
    )
    # Second correction must use the CURRENT generated line text as original_text,
    # since record_correction validates against the generated note, not prior corrections.
    corrections.record_correction(
        transcript_id="glaucoma_05", section="O", line_index=0,
        original_text="IOP 18 mmHg OD.", corrected_text="IOP 21 mmHg OD.",
        reviewer="dr_jones",
    )

    note = corrections.build_candidate_golden("glaucoma_05")
    assert note["soap"]["O"][0]["text"] == "IOP 21 mmHg OD."
    assert len(note["source_corrections"]) == 2


# ---------------------------------------------------------------------------
# correction_stats / list_corrections
# ---------------------------------------------------------------------------

def test_correction_stats_across_transcripts(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _write_result(results_dir, transcript_id="glaucoma_05", visit_type="glaucoma_followup")
    _write_result(results_dir, transcript_id="cataract_09", visit_type="cataract_postop")

    corrections.record_correction(
        transcript_id="glaucoma_05", section="O", line_index=0,
        original_text="IOP 18 mmHg OD.", corrected_text="IOP 20 mmHg OD.",
        reviewer="dr_smith",
    )
    corrections.record_correction(
        transcript_id="glaucoma_05", section="A", line_index=0,
        original_text="Stable glaucoma.", corrected_text="Improving glaucoma.",
        reviewer="dr_smith",
    )
    corrections.record_correction(
        transcript_id="cataract_09", section="S", line_index=0,
        original_text="Patient reports mild irritation.", corrected_text="Patient reports no irritation.",
        reviewer="dr_jones",
    )

    stats = corrections.correction_stats()
    assert stats["count"] == 3
    assert stats["by_visit_type"] == {"glaucoma_followup": 2, "cataract_postop": 1}
    assert stats["by_section"] == {"S": 1, "O": 1, "A": 1, "P": 0}


def test_list_corrections_filtering(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _write_result(results_dir, transcript_id="glaucoma_05", visit_type="glaucoma_followup")
    _write_result(results_dir, transcript_id="cataract_09", visit_type="cataract_postop")

    corrections.record_correction(
        transcript_id="glaucoma_05", section="O", line_index=0,
        original_text="IOP 18 mmHg OD.", corrected_text="IOP 20 mmHg OD.",
        reviewer="dr_smith",
    )
    corrections.record_correction(
        transcript_id="cataract_09", section="S", line_index=0,
        original_text="Patient reports mild irritation.", corrected_text="Patient reports no irritation.",
        reviewer="dr_jones",
    )

    only_glaucoma = corrections.list_corrections("glaucoma_05")
    assert len(only_glaucoma) == 1
    assert only_glaucoma[0]["transcript_id"] == "glaucoma_05"

    all_records = corrections.list_corrections()
    assert len(all_records) == 2


def test_list_corrections_returns_empty_list_when_file_missing(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    results_dir.mkdir(parents=True)

    assert corrections.list_corrections() == []
    assert corrections.list_corrections("anything") == []


# ---------------------------------------------------------------------------
# diff_lines
# ---------------------------------------------------------------------------

def test_diff_lines_reflects_change(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(tmp_path / "results"))
    result = corrections.diff_lines("IOP 18 mmHg OD.", "IOP 20 mmHg OD.")
    assert result != ""
    assert ("-" in result) or ("+" in result)


def test_diff_lines_identical_strings_report_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(tmp_path / "results"))
    result = corrections.diff_lines("same text here", "same text here")
    assert "unchanged" in result.lower()
