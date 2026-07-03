import copy
import json
from pathlib import Path
from statistics import mean

import pytest

from scribegate.judge import judge_note, judge_note_reference_free

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLDEN_DIR = DATA_DIR / "golden_notes"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"


def _load_golden(transcript_id: str) -> dict:
    with open(GOLDEN_DIR / f"{transcript_id}.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_transcript(transcript_id: str) -> str:
    with open(TRANSCRIPT_DIR / f"{transcript_id}.txt", "r", encoding="utf-8") as fh:
        return fh.read()


GLAUCOMA_05 = "glaucoma_05"
CATARACT_02 = "cataract_02"


# ---------------------------------------------------------------------------
# 1. Golden judged against itself -> perfect score.
# ---------------------------------------------------------------------------

def test_golden_judged_against_itself_is_perfect():
    golden = _load_golden(GLAUCOMA_05)
    transcript = _load_transcript(GLAUCOMA_05)

    result = judge_note(golden, golden, transcript)

    assert result["scores"] == {
        "completeness": 5,
        "hallucination": 5,
        "coding_plausibility": 5,
        "terminology": 5,
    }
    assert result["aggregate"] == 1.0


# ---------------------------------------------------------------------------
# 2. A fabricated IOP value (unsupported span) -> hallucination <= 2.
# ---------------------------------------------------------------------------

def test_fabricated_iop_value_scores_low_hallucination():
    golden = _load_golden(GLAUCOMA_05)
    transcript = _load_transcript(GLAUCOMA_05)

    generated = copy.deepcopy(golden)
    # Replace the O line's IOP text with a fabricated value and point the
    # span at unrelated transcript text (the header line) so there is no
    # legitimate support for the new number.
    generated["soap"]["O"][0] = {
        "text": "IOP 47 mmHg OD / 45 mmHg OS (GAT); fabricated reading.",
        "spans": [[0, 40]],  # header line, unrelated to IOP numbers
    }

    result = judge_note(generated, golden, transcript)

    assert result["scores"]["hallucination"] <= 2


# ---------------------------------------------------------------------------
# 3. Missing about half the golden lines -> completeness <= 3.
# ---------------------------------------------------------------------------

def test_missing_half_the_lines_scores_low_completeness():
    golden = _load_golden(GLAUCOMA_05)
    transcript = _load_transcript(GLAUCOMA_05)

    generated = copy.deepcopy(golden)
    for section in ("S", "O", "A", "P"):
        lines = generated["soap"][section]
        generated["soap"][section] = [ln for i, ln in enumerate(lines) if i % 2 == 0]

    result = judge_note(generated, golden, transcript)

    assert result["scores"]["completeness"] <= 3


# ---------------------------------------------------------------------------
# 4. Empty section -> coding_plausibility <= 3 or lower than self-judge.
# ---------------------------------------------------------------------------

def test_empty_section_scores_lower_coding_plausibility():
    golden = _load_golden(GLAUCOMA_05)
    transcript = _load_transcript(GLAUCOMA_05)

    perfect_result = judge_note(golden, golden, transcript)

    generated = copy.deepcopy(golden)
    generated["soap"]["O"] = []

    result = judge_note(generated, golden, transcript)

    assert (
        result["scores"]["coding_plausibility"] <= 3
        or result["scores"]["coding_plausibility"] < perfect_result["scores"]["coding_plausibility"]
    )


# ---------------------------------------------------------------------------
# 5. Determinism.
# ---------------------------------------------------------------------------

def test_determinism_identical_inputs_identical_output():
    golden = _load_golden(GLAUCOMA_05)
    transcript = _load_transcript(GLAUCOMA_05)

    generated = copy.deepcopy(golden)
    generated["soap"]["A"][0]["text"] += " Extra unsupported note."

    result1 = judge_note(generated, golden, transcript)
    result2 = judge_note(generated, golden, transcript)

    assert result1 == result2


# ---------------------------------------------------------------------------
# 6. Return-shape validation.
# ---------------------------------------------------------------------------

def test_return_shape_is_valid():
    golden = _load_golden(GLAUCOMA_05)
    transcript = _load_transcript(GLAUCOMA_05)

    result = judge_note(golden, golden, transcript)

    assert set(result.keys()) == {"scores", "aggregate", "rationales"}

    assert set(result["scores"].keys()) == {
        "completeness", "hallucination", "coding_plausibility", "terminology"
    }
    for dim, score in result["scores"].items():
        assert isinstance(score, int)
        assert 1 <= score <= 5

    assert set(result["rationales"].keys()) == {
        "completeness", "hallucination", "coding_plausibility", "terminology"
    }
    for dim, rationale in result["rationales"].items():
        assert isinstance(rationale, str)
        assert len(rationale) > 0

    assert isinstance(result["aggregate"], float)
    assert 0.0 <= result["aggregate"] <= 1.0


# ---------------------------------------------------------------------------
# 7. Aggregate formula exactness across several notes.
# ---------------------------------------------------------------------------

def _assert_aggregate_exact(result: dict):
    scores = result["scores"]
    expected = (mean(scores.values()) - 1) / 4
    assert abs(result["aggregate"] - expected) < 1e-9


def test_aggregate_formula_exactness_multiple_notes():
    golden_glaucoma = _load_golden(GLAUCOMA_05)
    transcript_glaucoma = _load_transcript(GLAUCOMA_05)
    golden_cataract = _load_golden(CATARACT_02)
    transcript_cataract = _load_transcript(CATARACT_02)

    # perfect self-judge
    _assert_aggregate_exact(judge_note(golden_glaucoma, golden_glaucoma, transcript_glaucoma))
    _assert_aggregate_exact(judge_note(golden_cataract, golden_cataract, transcript_cataract))

    # degraded notes with various mutations
    degraded1 = copy.deepcopy(golden_glaucoma)
    degraded1["soap"]["O"] = degraded1["soap"]["O"][:1]
    _assert_aggregate_exact(judge_note(degraded1, golden_glaucoma, transcript_glaucoma))

    degraded2 = copy.deepcopy(golden_cataract)
    degraded2["soap"]["P"] = []
    _assert_aggregate_exact(judge_note(degraded2, golden_cataract, transcript_cataract))

    degraded3 = copy.deepcopy(golden_glaucoma)
    degraded3["soap"]["O"][0]["text"] = "IOP 99 mmHg OD / 98 mmHg OS fabricated."
    degraded3["soap"]["O"][0]["spans"] = [[0, 10]]
    _assert_aggregate_exact(judge_note(degraded3, golden_glaucoma, transcript_glaucoma))


# ---------------------------------------------------------------------------
# 8. Cross-section misplacement.
# ---------------------------------------------------------------------------

def test_cross_section_misplacement_lowers_completeness():
    golden = _load_golden(GLAUCOMA_05)
    transcript = _load_transcript(GLAUCOMA_05)

    perfect_result = judge_note(golden, golden, transcript)
    assert perfect_result["scores"]["completeness"] == 5

    generated = copy.deepcopy(golden)
    # Move the golden O[0] line (IOP finding) into S, same text/spans.
    moved_line = generated["soap"]["O"].pop(0)
    generated["soap"]["S"].append(moved_line)

    result = judge_note(generated, golden, transcript)

    assert result["scores"]["completeness"] <= 4
    assert isinstance(result["rationales"]["completeness"], str)
    assert len(result["rationales"]["completeness"]) > 0


# ---------------------------------------------------------------------------
# 9. Terminology fallback path works even though normalizer is a stub.
# ---------------------------------------------------------------------------

def test_terminology_fallback_path_does_not_crash(monkeypatch):
    # judge.py imports scribegate.normalizer.check_note defensively (try/
    # except ImportError/AttributeError/Exception) because per the T5 spec
    # normalizer.py may be nothing but the literal TODO stub docstring (no
    # functions/classes) at the time this module is exercised. Simulate that
    # exact stub condition here by making the import path fail, regardless
    # of whether normalizer.py in this checkout happens to be implemented,
    # and assert judge_note still produces a valid, non-crashing result via
    # the internal regex-based fallback checker.
    import scribegate.judge as judge_module

    original_fallback = judge_module._fallback_terminology_violations
    calls = {"used_fallback": False}

    def _tracking_fallback(*args, **kwargs):
        calls["used_fallback"] = True
        return original_fallback(*args, **kwargs)

    monkeypatch.setattr(judge_module, "_fallback_terminology_violations", _tracking_fallback)

    # Force the normalizer import inside score_terminology to fail by
    # removing check_note from the already-imported normalizer module (this
    # reproduces "normalizer is a stub with no check_note function").
    import scribegate.normalizer as normalizer_module

    had_check_note = hasattr(normalizer_module, "check_note")
    if had_check_note:
        monkeypatch.delattr(normalizer_module, "check_note")

    golden = _load_golden(GLAUCOMA_05)
    transcript = _load_transcript(GLAUCOMA_05)

    result = judge_note(golden, golden, transcript)

    term_score = result["scores"]["terminology"]
    assert isinstance(term_score, int)
    assert 1 <= term_score <= 5
    assert calls["used_fallback"] is True


# ---------------------------------------------------------------------------
# 10. At least 2 fixture pairs judged against themselves -> all 5s / 1.0.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("transcript_id", [GLAUCOMA_05, "contactlens_03", "comprehensive_05", "glaucoma_02"])
def test_multiple_fixtures_self_judged_perfect(transcript_id):
    golden = _load_golden(transcript_id)
    transcript = _load_transcript(transcript_id)

    result = judge_note(golden, golden, transcript)

    assert result["scores"] == {
        "completeness": 5,
        "hallucination": 5,
        "coding_plausibility": 5,
        "terminology": 5,
    }
    assert result["aggregate"] == 1.0


# ---------------------------------------------------------------------------
# Extra: rationales differ meaningfully between a good note and a bad note.
# ---------------------------------------------------------------------------

def test_rationales_differ_between_good_and_bad_note():
    golden = _load_golden(GLAUCOMA_05)
    transcript = _load_transcript(GLAUCOMA_05)

    good_result = judge_note(golden, golden, transcript)

    bad = copy.deepcopy(golden)
    bad["soap"]["O"][0]["text"] = "IOP 91 mmHg OD / 90 mmHg OS fabricated finding."
    bad["soap"]["O"][0]["spans"] = [[0, 10]]
    bad["soap"]["A"] = []
    bad_result = judge_note(bad, golden, transcript)

    assert good_result["rationales"]["hallucination"] != bad_result["rationales"]["hallucination"]
    assert good_result["scores"]["hallucination"] > bad_result["scores"]["hallucination"]


# ---------------------------------------------------------------------------
# Extra: empty-golden edge case (avoid div-by-zero, still valid shape).
# ---------------------------------------------------------------------------

def test_empty_golden_edge_case():
    transcript = _load_transcript(GLAUCOMA_05)
    empty_golden = {
        "transcript_id": GLAUCOMA_05,
        "visit_type": "glaucoma_followup",
        "synthetic": True,
        "soap": {"S": [], "O": [], "A": [], "P": []},
    }
    generated = {
        "transcript_id": GLAUCOMA_05,
        "visit_type": "glaucoma_followup",
        "synthetic": True,
        "soap": {"S": [], "O": [], "A": [], "P": []},
    }

    result = judge_note(generated, empty_golden, transcript)

    assert result["scores"]["completeness"] == 5
    assert 1 <= result["scores"]["coding_plausibility"] <= 5
    assert isinstance(result["aggregate"], float)


# ---------------------------------------------------------------------------
# Extra: API judge path is never triggered by default (no env vars set).
# ---------------------------------------------------------------------------

def test_default_path_never_requires_api(monkeypatch):
    monkeypatch.delenv("SCRIBEGATE_USE_API", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    golden = _load_golden(GLAUCOMA_05)
    transcript = _load_transcript(GLAUCOMA_05)

    # Should not raise even though `anthropic` package is not installed /
    # no network access is available in this environment.
    result = judge_note(golden, golden, transcript)
    assert result["scores"]["completeness"] == 5


# ---------------------------------------------------------------------------
# judge_note_reference_free — reference-free judging (no golden note).
# ---------------------------------------------------------------------------

def test_reference_free_return_shape_is_valid():
    golden = _load_golden(GLAUCOMA_05)
    transcript = _load_transcript(GLAUCOMA_05)

    result = judge_note_reference_free(golden, transcript)

    assert set(result.keys()) == {"scores", "aggregate", "rationales"}

    assert set(result["scores"].keys()) == {
        "completeness", "hallucination", "coding_plausibility", "terminology"
    }
    for dim, score in result["scores"].items():
        assert isinstance(score, int)
        assert 1 <= score <= 5

    assert set(result["rationales"].keys()) == {
        "completeness", "hallucination", "coding_plausibility", "terminology"
    }
    for dim, rationale in result["rationales"].items():
        assert isinstance(rationale, str)
        assert len(rationale) > 0

    assert "reference-free mode" in result["rationales"]["completeness"]

    assert isinstance(result["aggregate"], float)
    assert 0.0 <= result["aggregate"] <= 1.0


def test_reference_free_well_covered_transcript_scores_completeness_reasonably_high():
    golden = _load_golden(GLAUCOMA_05)
    transcript = _load_transcript(GLAUCOMA_05)

    # The golden note, by construction, covers its own transcript's
    # clinically-salient content well — using it as the "generated" note
    # exercises the well-covered case for reference-free completeness.
    result = judge_note_reference_free(golden, transcript)

    assert result["scores"]["completeness"] >= 3


def test_reference_free_sparse_transcript_scores_completeness_lower():
    transcript = _load_transcript(GLAUCOMA_05)

    well_covered_golden = _load_golden(GLAUCOMA_05)
    well_covered_result = judge_note_reference_free(well_covered_golden, transcript)

    sparse_generated = {
        "soap": {
            "S": [{"text": "Patient seen.", "spans": []}],
            "O": [],
            "A": [],
            "P": [],
        }
    }
    sparse_result = judge_note_reference_free(sparse_generated, transcript)

    assert sparse_result["scores"]["completeness"] < well_covered_result["scores"]["completeness"]
    assert sparse_result["scores"]["completeness"] <= 2


def test_reference_free_fabricated_iop_value_scores_low_hallucination_no_golden():
    golden = _load_golden(GLAUCOMA_05)
    transcript = _load_transcript(GLAUCOMA_05)

    generated = copy.deepcopy(golden)
    # Same mutation pattern as test_fabricated_iop_value_scores_low_hallucination:
    # replace the O line's IOP text with a fabricated value and point the span
    # at unrelated transcript text (the header line) so there is no
    # legitimate support for the new number.
    generated["soap"]["O"][0] = {
        "text": "IOP 47 mmHg OD / 45 mmHg OS (GAT); fabricated reading.",
        "spans": [[0, 40]],  # header line, unrelated to IOP numbers
    }

    result = judge_note_reference_free(generated, transcript)

    assert result["scores"]["hallucination"] <= 2
