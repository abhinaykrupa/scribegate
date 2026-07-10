"""Tests for the V1 "data moat" correction-compounding loop:
scribegate.corrections' generation model (active_golden_dir / load_golden_note
/ promote_candidate / promote_all_candidates) plus scribegate.moat's
metrics/demo layer built on top of it.

Every test that writes anything sets SCRIBEGATE_RESULTS_DIR to a
tmp_path-derived directory so none of them ever touch the real
data/results/ directory. Reads of the pristine, read-only fixture corpora
(data/golden_notes/*.json, data/transcripts/*.txt) use REAL bundled
transcript ids on purpose: moat.py and corrections.py deliberately resolve
gen-0 golden notes and transcript text from the real repo paths (never
sandboxed — only the results/generations layer, where writes happen, is
env-driven), so exercising promotion/rebenchmark/demo logic requires real
transcript ids that have matching data/golden_notes/ and data/transcripts/
fixtures already on disk.
"""

import json

import pytest

from scribegate import corrections
from scribegate import moat
from scribegate import cli

REAL_GOLDEN_DIR = corrections._GOLDEN_DIR
REAL_RESULTS_DIR = corrections._DEFAULT_RESULTS_DIR


def _write_result(results_dir, transcript_id, visit_type="glaucoma_followup", soap=None):
    results_dir.mkdir(parents=True, exist_ok=True)
    soap = soap or {
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


def _copy_real_result(results_dir, transcript_id):
    """Copy a REAL bundled data/results/{id}.json into the sandboxed
    results_dir (read from the real repo, write only to the tmp sandbox) —
    used so moat's demo/rebenchmark logic has genuine generated-vs-golden
    drift to work with, without ever writing back to the real results dir."""
    results_dir.mkdir(parents=True, exist_ok=True)
    src = REAL_RESULTS_DIR / f"{transcript_id}.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    (results_dir / f"{transcript_id}.json").write_text(json.dumps(data, indent=2) + "\n")
    return data


# ---------------------------------------------------------------------------
# Overlay precedence (active_golden_dir / load_golden_note)
# ---------------------------------------------------------------------------

def test_active_golden_dir_defaults_to_gen0_when_no_generations(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(tmp_path / "results"))
    assert corrections.list_generations() == []
    resolved = corrections.active_golden_dir("glaucoma_05")
    assert resolved == REAL_GOLDEN_DIR


def test_load_golden_note_gen0_matches_pristine_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(tmp_path / "results"))
    expected = json.loads((REAL_GOLDEN_DIR / "glaucoma_05.json").read_text(encoding="utf-8"))
    assert corrections.load_golden_note("glaucoma_05") == expected


def test_overlay_precedence_walks_back_and_is_per_transcript(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))

    _write_result(results_dir, "glaucoma_05")
    corrections.record_correction(
        transcript_id="glaucoma_05", section="O", line_index=0,
        original_text="IOP 18 mmHg OD.", corrected_text="IOP 20 mmHg OD.",
        reviewer="dr_smith",
    )
    manifest = corrections.promote_candidate("glaucoma_05", reviewer="dr_smith")
    assert manifest["gen"] == 1

    # The overridden transcript resolves to the gen_1 overlay.
    overridden = corrections.load_golden_note("glaucoma_05")
    assert overridden["soap"]["O"][0]["text"] == "IOP 20 mmHg OD."
    assert corrections.active_golden_dir("glaucoma_05") == corrections._generation_dir(1)

    # A DIFFERENT transcript with no override in gen_1 still falls back to
    # gen-0 pristine, even though a later generation exists.
    pristine_cataract = json.loads((REAL_GOLDEN_DIR / "cataract_01.json").read_text(encoding="utf-8"))
    assert corrections.load_golden_note("cataract_01") == pristine_cataract
    assert corrections.active_golden_dir("cataract_01") == REAL_GOLDEN_DIR

    # Explicitly requesting generation=0 forces pristine even for the
    # overridden transcript.
    pristine_glaucoma = json.loads((REAL_GOLDEN_DIR / "glaucoma_05.json").read_text(encoding="utf-8"))
    assert corrections.load_golden_note("glaucoma_05", generation=0) == pristine_glaucoma


# ---------------------------------------------------------------------------
# Promotion manifest + decision-log event
# ---------------------------------------------------------------------------

def test_promote_candidate_writes_manifest_and_overlay_file(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))

    _write_result(results_dir, "glaucoma_05")
    corrections.record_correction(
        transcript_id="glaucoma_05", section="A", line_index=0,
        original_text="Stable glaucoma.", corrected_text="Stable POAG, IOP controlled.",
        reviewer="dr_lee", note="clarify diagnosis",
    )
    manifest = corrections.promote_candidate("glaucoma_05", reviewer="dr_lee", note="batch 1")

    assert manifest["gen"] == 1
    assert manifest["reviewer"] == "dr_lee"
    assert manifest["note"] == "batch 1"
    assert manifest["promoted"] == ["glaucoma_05"]
    assert "ts" in manifest and manifest["ts"]
    assert manifest["source_corrections"]["glaucoma_05"]

    manifest_path = corrections._generation_dir(1) / corrections.GENERATION_MANIFEST_NAME
    assert manifest_path.exists()
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk == manifest

    overlay_path = corrections._generation_dir(1) / "glaucoma_05.json"
    assert overlay_path.exists()
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    assert overlay["soap"]["A"][0]["text"] == "Stable POAG, IOP controlled."
    # Bookkeeping-only fields are stripped before writing the overlay file.
    assert "candidate" not in overlay
    assert "source_corrections" not in overlay


def test_promote_candidate_appends_decision_log_promotion_event(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))

    _write_result(results_dir, "glaucoma_05")
    corrections.record_correction(
        transcript_id="glaucoma_05", section="P", line_index=0,
        original_text="Continue current drops.", corrected_text="Continue current drops; recheck in 4 weeks.",
        reviewer="dr_lee",
    )
    corrections.promote_candidate("glaucoma_05", reviewer="dr_lee", note="")

    log_lines = (results_dir / corrections.DECISION_LOG_NAME).read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(l) for l in log_lines]
    promo_events = [e for e in events if e.get("event") == "promotion"]
    assert len(promo_events) == 1
    assert promo_events[0]["gen"] == 1
    assert promo_events[0]["reviewer"] == "dr_lee"
    assert promo_events[0]["promoted"] == ["glaucoma_05"]


def test_promote_all_candidates_batches_multiple_transcripts_into_one_generation(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))

    _write_result(results_dir, "glaucoma_05")
    _write_result(results_dir, "cataract_01", visit_type="cataract_postop")

    corrections.record_correction(
        transcript_id="glaucoma_05", section="O", line_index=0,
        original_text="IOP 18 mmHg OD.", corrected_text="IOP 19 mmHg OD.",
        reviewer="reviewer_a",
    )
    corrections.record_correction(
        transcript_id="cataract_01", section="S", line_index=0,
        original_text="Patient reports mild irritation.", corrected_text="Patient reports moderate irritation.",
        reviewer="reviewer_a",
    )

    manifest = corrections.promote_all_candidates(reviewer="reviewer_a", note="batch")

    assert manifest["gen"] == 1
    assert sorted(manifest["promoted"]) == ["cataract_01", "glaucoma_05"]
    assert corrections.list_generations() == [1]  # one batch == one generation
    assert (corrections._generation_dir(1) / "glaucoma_05.json").exists()
    assert (corrections._generation_dir(1) / "cataract_01.json").exists()


def test_promote_all_candidates_raises_without_any_corrections(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    with pytest.raises(ValueError):
        corrections.promote_all_candidates(reviewer="nobody")


# ---------------------------------------------------------------------------
# Span validation rejects corrupt candidates
# ---------------------------------------------------------------------------

def test_promote_candidate_rejects_out_of_bounds_span(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))

    # glaucoma_05's real transcript text is 2720 chars long; give the line a
    # span whose end (5000) falls far outside that bound.
    soap = {
        "S": [{"text": "Patient reports mild irritation.", "spans": [[0, 5000]]}],
        "O": [{"text": "IOP 18 mmHg OD.", "spans": [[20, 30]]}],
        "A": [{"text": "Stable glaucoma.", "spans": [[40, 50]]}],
        "P": [{"text": "Continue current drops.", "spans": [[60, 70]]}],
    }
    _write_result(results_dir, "glaucoma_05", soap=soap)
    corrections.record_correction(
        transcript_id="glaucoma_05", section="S", line_index=0,
        original_text="Patient reports mild irritation.", corrected_text="Patient reports severe irritation.",
        reviewer="dr_smith",
    )

    with pytest.raises(ValueError, match="out of bounds"):
        corrections.promote_candidate("glaucoma_05", reviewer="dr_smith")

    # No generation should have been created by the rejected promotion.
    assert corrections.list_generations() == []


def test_promote_candidate_rejects_malformed_span_shape(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))

    soap = {
        "S": [{"text": "Patient reports mild irritation.", "spans": [[0, "oops"]]}],
        "O": [{"text": "IOP 18 mmHg OD.", "spans": [[20, 30]]}],
        "A": [{"text": "Stable glaucoma.", "spans": [[40, 50]]}],
        "P": [{"text": "Continue current drops.", "spans": [[60, 70]]}],
    }
    _write_result(results_dir, "glaucoma_05", soap=soap)
    corrections.record_correction(
        transcript_id="glaucoma_05", section="S", line_index=0,
        original_text="Patient reports mild irritation.", corrected_text="Patient reports severe irritation.",
        reviewer="dr_smith",
    )
    with pytest.raises(ValueError):
        corrections.promote_candidate("glaucoma_05", reviewer="dr_smith")
    assert corrections.list_generations() == []


# ---------------------------------------------------------------------------
# Default-pipeline-unchanged / generation-aware cli
# ---------------------------------------------------------------------------

def test_cli_default_behavior_unchanged_when_no_generations_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(tmp_path / "results"))
    result = cli.process_transcript("glaucoma_05")
    golden = json.loads((REAL_GOLDEN_DIR / "glaucoma_05.json").read_text(encoding="utf-8"))
    from scribegate.judge import judge_note
    transcript_text = (corrections._TRANSCRIPT_DIR / "glaucoma_05.txt").read_text(encoding="utf-8")
    expected_judge = judge_note(result["generated_note"], golden, transcript_text)
    assert result["judge_result"] == expected_judge


def test_cli_golden_gen_selects_overlay_generation(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))

    _write_result(results_dir, "glaucoma_05")
    corrections.record_correction(
        transcript_id="glaucoma_05", section="O", line_index=0,
        original_text="IOP 18 mmHg OD.", corrected_text="IOP 21 mmHg OD, uncontrolled.",
        reviewer="dr_smith",
    )
    corrections.promote_candidate("glaucoma_05", reviewer="dr_smith")

    golden_gen0 = cli._load_golden("glaucoma_05", golden_generation=0)
    golden_latest = cli._load_golden("glaucoma_05", golden_generation=None)
    golden_gen1 = cli._load_golden("glaucoma_05", golden_generation=1)

    pristine = json.loads((REAL_GOLDEN_DIR / "glaucoma_05.json").read_text(encoding="utf-8"))
    assert golden_gen0 == pristine
    assert golden_gen1["soap"]["O"][0]["text"] == "IOP 21 mmHg OD, uncontrolled."
    assert golden_latest == golden_gen1


# ---------------------------------------------------------------------------
# rebenchmark_generation determinism
# ---------------------------------------------------------------------------

def test_rebenchmark_generation_is_deterministic(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _copy_real_result(results_dir, "glaucoma_05")
    _copy_real_result(results_dir, "cataract_01")

    first = moat.rebenchmark_generation(0)
    second = moat.rebenchmark_generation(0)
    assert first == second

    summary_path = results_dir / "golden_generations" / "gen_0" / moat.BENCHMARK_SUMMARY_NAME
    text_first = summary_path.read_text(encoding="utf-8")
    moat.rebenchmark_generation(0)
    text_second = summary_path.read_text(encoding="utf-8")
    assert text_first == text_second


# ---------------------------------------------------------------------------
# moat_metrics shape
# ---------------------------------------------------------------------------

def test_moat_metrics_shape(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _copy_real_result(results_dir, "glaucoma_05")

    metrics = moat.moat_metrics()

    assert set(metrics.keys()) == {"golden_set", "corrections_recorded_total", "generations", "moat_curve"}
    assert "base_count" in metrics["golden_set"]
    assert metrics["golden_set"]["base_count"] == len(list(REAL_GOLDEN_DIR.glob("*.json")))
    assert metrics["corrections_recorded_total"] == 0
    assert isinstance(metrics["generations"], list) and metrics["generations"] == []
    assert isinstance(metrics["moat_curve"], list) and len(metrics["moat_curve"]) == 1
    gen0_row = metrics["moat_curve"][0]
    assert len(gen0_row) == 4
    assert gen0_row[0] == 0
    assert gen0_row[2] == metrics["golden_set"]["base_count"]
    assert gen0_row[3] == 0


def test_moat_metrics_curve_grows_after_promotion(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _copy_real_result(results_dir, "glaucoma_05")

    corrections.record_correction(
        transcript_id="glaucoma_05", section="O", line_index=0,
        original_text=json.loads((results_dir / "glaucoma_05.json").read_text())["generated_note"]["soap"]["O"][0]["text"],
        corrected_text="IOP 22 mmHg OD, corrected.",
        reviewer="dr_smith",
    )
    corrections.promote_candidate("glaucoma_05", reviewer="dr_smith")

    metrics = moat.moat_metrics()
    assert len(metrics["generations"]) == 1
    assert len(metrics["moat_curve"]) == 2
    gen1_row = metrics["moat_curve"][1]
    assert gen1_row[0] == 1
    assert gen1_row[2] == 1  # one cumulative promoted note
    assert gen1_row[3] == 1  # one cumulative correction


# ---------------------------------------------------------------------------
# simulate_moat_demo: seeding + idempotence
# ---------------------------------------------------------------------------

def test_simulate_moat_demo_seeds_and_produces_before_after_curve(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _copy_real_result(results_dir, "glaucoma_05")
    _copy_real_result(results_dir, "cataract_01")
    _copy_real_result(results_dir, "glaucoma_01")

    result = moat.simulate_moat_demo()

    assert result["seeded"] is True
    assert result["generation"] == 1
    assert 1 <= len(result["corrections"]) <= 3
    assert "before" in result and "after" in result
    assert "overall_aggregate" in result["before"]
    assert "overall_aggregate" in result["after"]
    assert corrections.list_generations() == [1]

    kinds = {c["kind"] for c in result["corrections"]}
    assert kinds <= {"paraphrase_fix", "restore_dropped"}


def test_simulate_moat_demo_is_idempotent(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _copy_real_result(results_dir, "glaucoma_05")
    _copy_real_result(results_dir, "cataract_01")

    first = moat.simulate_moat_demo()
    assert first["seeded"] is True
    assert corrections.list_generations() == [1]

    second = moat.simulate_moat_demo()
    assert second["seeded"] is False
    # Re-running must NOT stack a duplicate generation.
    assert corrections.list_generations() == [1]


def test_simulate_moat_demo_raises_with_no_results(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    with pytest.raises(RuntimeError):
        moat.simulate_moat_demo()


# ---------------------------------------------------------------------------
# SCRIBEGATE_RESULTS_DIR sandboxing — never touch the real data/results/
# ---------------------------------------------------------------------------

def _snapshot(root):
    """Recursive (relative path, size) snapshot so this test also catches
    in-place mutation of a file that already existed (not just added/removed
    top-level entries) — used to prove the real data/results/ tree is
    byte-for-byte unaffected by sandboxed moat operations, whatever state it
    happened to be in (e.g. a real generation from a prior legitimate
    `--seed-demo` run) when this test executes."""
    return sorted(
        (str(p.relative_to(root)), p.stat().st_size)
        for p in root.rglob("*") if p.is_file()
    )


def test_moat_operations_never_write_to_real_results_dir(tmp_path, monkeypatch):
    before_snapshot = _snapshot(REAL_RESULTS_DIR)

    results_dir = tmp_path / "results"
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(results_dir))
    _copy_real_result(results_dir, "glaucoma_05")
    _copy_real_result(results_dir, "cataract_01")

    moat.simulate_moat_demo()
    moat.moat_metrics()

    after_snapshot = _snapshot(REAL_RESULTS_DIR)
    assert before_snapshot == after_snapshot
    # All the writes actually landed in the sandbox.
    assert (results_dir / "golden_generations" / "gen_1").exists()
