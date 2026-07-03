"""Tests for scribegate.audit (U5): dossier assembly, integrity hashing,
markdown rendering, and export.

Every test sets SCRIBEGATE_RESULTS_DIR to a tmp_path-derived directory so
none of them ever touch the real data/results/ directory.
"""

import json

from scribegate import audit
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
# build_dossier
# ---------------------------------------------------------------------------

def test_build_dossier_includes_only_matching_transcript_events(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _write_result(results_dir, transcript_id="X", visit_type="glaucoma_followup")
    _write_result(results_dir, transcript_id="Y", visit_type="cataract_postop")

    # Route-shaped (cli.py) line for X.
    route_event = {"ts": "2026-07-02T15:00:01Z", "transcript_id": "X", "aggregate": 0.8,
                   "route": "review", "violation_count": 0}
    # Reviewer/decision-shaped (streamlit) line for X.
    reviewer_event = {"ts": "2026-07-02T15:05:00Z", "transcript_id": "X", "reviewer": "dr_smith",
                      "decision": "approved"}
    # Line for a DIFFERENT transcript_id "Y" — must be excluded.
    other_event = {"ts": "2026-07-02T15:06:00Z", "transcript_id": "Y", "reviewer": "dr_jones",
                   "decision": "rejected"}

    log_path = results_dir / "decision_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(route_event) + "\n")
        fh.write(json.dumps(reviewer_event) + "\n")
        fh.write(json.dumps(other_event) + "\n")

    # Correction-shaped line for X (via the real recording path).
    corrections.record_correction(
        transcript_id="X", section="O", line_index=0,
        original_text="IOP 18 mmHg OD.", corrected_text="IOP 20 mmHg OD.",
        reviewer="dr_smith",
    )

    dossier = audit.build_dossier("X")
    events = dossier["decision_log_events"]

    assert len(events) == 3
    assert all(e.get("transcript_id") == "X" for e in events)
    kinds = [e.get("event") or ("decision" if "decision" in e else "route") for e in events]
    assert "route" in kinds
    assert "decision" in kinds
    assert "correction" in kinds

    assert dossier["transcript_id"] == "X"
    assert dossier["visit_type"] == "glaucoma_followup"
    assert dossier["synthetic"] is True
    assert len(dossier["corrections"]) == 1


def test_build_dossier_handles_missing_decision_log(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _write_result(results_dir, transcript_id="solo_01")

    dossier = audit.build_dossier("solo_01")
    assert dossier["decision_log_events"] == []
    assert dossier["corrections"] == []


# ---------------------------------------------------------------------------
# Integrity hash stability
# ---------------------------------------------------------------------------

def test_dossier_sha256_stable_across_calls(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _write_result(results_dir, transcript_id="stable_01")

    dossier1 = audit.build_dossier("stable_01")
    dossier2 = audit.build_dossier("stable_01")

    assert dossier1["dossier_sha256"] == dossier2["dossier_sha256"]
    # dossier_generated_at is not asserted unequal here (it could
    # coincidentally match at second granularity) — the stable, guaranteed
    # invariant is the sha256 match above.


def test_dossier_sha256_changes_when_underlying_data_changes(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _write_result(results_dir, transcript_id="changing_01")

    dossier_before = audit.build_dossier("changing_01")

    corrections.record_correction(
        transcript_id="changing_01", section="O", line_index=0,
        original_text="IOP 18 mmHg OD.", corrected_text="IOP 20 mmHg OD.",
        reviewer="dr_smith",
    )

    dossier_after = audit.build_dossier("changing_01")
    assert dossier_before["dossier_sha256"] != dossier_after["dossier_sha256"]


# ---------------------------------------------------------------------------
# render_dossier_md
# ---------------------------------------------------------------------------

def test_render_dossier_md_contains_all_section_headers(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _write_result(results_dir, transcript_id="headers_01")

    dossier = audit.build_dossier("headers_01")
    md = audit.render_dossier_md(dossier)

    for header in (
        "## Encounter",
        "## Generation",
        "## Evaluation",
        "## Routing",
        "## Human Review Trail",
        "## Corrections",
        "## Integrity",
    ):
        assert header in md


def test_render_dossier_md_flags_synthetic_prominently(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _write_result(results_dir, transcript_id="synthetic_01")

    dossier = audit.build_dossier("synthetic_01")
    md = audit.render_dossier_md(dossier)
    assert "SYNTHETIC DATA" in md


def test_render_dossier_md_with_corrections_includes_diff(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _write_result(results_dir, transcript_id="diffcheck_01")

    corrections.record_correction(
        transcript_id="diffcheck_01", section="O", line_index=0,
        original_text="IOP 18 mmHg OD.", corrected_text="IOP 20 mmHg OD.",
        reviewer="dr_smith", note="fixed reading",
    )

    dossier = audit.build_dossier("diffcheck_01")
    md = audit.render_dossier_md(dossier)
    assert "dr_smith" in md
    assert "fixed reading" in md


# ---------------------------------------------------------------------------
# export_dossier
# ---------------------------------------------------------------------------

def test_export_dossier_round_trip(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _write_result(results_dir, transcript_id="export_01")

    out_dir = tmp_path / "out"
    json_path, md_path = audit.export_dossier("export_01", out_dir)

    assert json_path.exists()
    assert md_path.exists()
    assert json_path == out_dir / "export_01_dossier.json"
    assert md_path == out_dir / "export_01_dossier.md"

    with open(json_path, "r", encoding="utf-8") as fh:
        exported = json.load(fh)

    fresh_dossier = audit.build_dossier("export_01")
    assert exported["dossier_sha256"] == fresh_dossier["dossier_sha256"]

    md_text = md_path.read_text()
    assert "## Encounter" in md_text
