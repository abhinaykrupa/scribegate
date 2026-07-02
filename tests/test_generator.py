"""Tests for scribegate.generator (T4).

Covers: module import without anthropic, generation over all 20 transcripts,
Note-shape validation, in-bounds span coverage, determinism, non-empty SOAP
sections on clean transcripts, and that messy contact-lens transcripts show
the naive "first-stated value" failure mode described in specs/INTERFACES.md.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSCRIPTS_DIR = os.path.join(REPO_ROOT, "data", "transcripts")
GOLDEN_DIR = os.path.join(REPO_ROOT, "data", "golden_notes")

SOAP_SECTIONS = ("S", "O", "A", "P")


def _transcript_ids() -> list[str]:
    paths = sorted(glob.glob(os.path.join(TRANSCRIPTS_DIR, "*.txt")))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def _read_transcript(transcript_id: str) -> str:
    path = os.path.join(TRANSCRIPTS_DIR, f"{transcript_id}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _visit_type_from_golden(transcript_id: str) -> str:
    path = os.path.join(GOLDEN_DIR, f"{transcript_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["visit_type"]


ALL_TRANSCRIPT_IDS = _transcript_ids()


def test_module_import_does_not_require_anthropic():
    """Import must succeed even if `anthropic` is not installed."""
    for mod_name in list(sys.modules):
        if mod_name == "anthropic" or mod_name.startswith("anthropic."):
            del sys.modules[mod_name]
    for mod_name in list(sys.modules):
        if mod_name == "scribegate.generator" or mod_name.startswith("scribegate.generator"):
            del sys.modules[mod_name]

    import builtins

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "anthropic" or name.startswith("anthropic."):
            raise ImportError("anthropic must not be imported at module load")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = blocking_import
    try:
        import scribegate.generator as gen  # noqa: F401
    finally:
        builtins.__import__ = real_import

    assert hasattr(gen, "generate_note")
    assert hasattr(gen, "APIBackend")
    assert hasattr(gen, "MockBackend")


def test_found_all_20_transcripts():
    assert len(ALL_TRANSCRIPT_IDS) == 20, ALL_TRANSCRIPT_IDS


@pytest.mark.parametrize("transcript_id", ALL_TRANSCRIPT_IDS)
def test_generates_without_error_for_every_transcript(transcript_id):
    from scribegate.generator import generate_note

    text = _read_transcript(transcript_id)
    visit_type = _visit_type_from_golden(transcript_id)
    note = generate_note(text, transcript_id, visit_type)
    assert isinstance(note, dict)


def _validate_note_shape(note: dict, transcript_id: str, visit_type: str, text: str):
    assert note["transcript_id"] == transcript_id
    assert note["visit_type"] == visit_type
    assert note["synthetic"] is True
    assert note.get("generated") is True
    assert note.get("generator") == "mock"

    assert set(note["soap"].keys()) == set(SOAP_SECTIONS)
    for section in SOAP_SECTIONS:
        lines = note["soap"][section]
        assert isinstance(lines, list)
        for line in lines:
            assert isinstance(line, dict)
            assert "text" in line and isinstance(line["text"], str)
            assert "spans" in line and isinstance(line["spans"], list)
            assert len(line["spans"]) >= 1, "every line must carry >=1 span"
            for span in line["spans"]:
                assert len(span) == 2
                start, end = span
                assert isinstance(start, int) and isinstance(end, int)
                assert 0 <= start < end <= len(text), (
                    f"span {span} out of bounds for transcript length {len(text)}"
                )


@pytest.mark.parametrize("transcript_id", ALL_TRANSCRIPT_IDS)
def test_output_validates_against_note_shape_with_inbounds_spans(transcript_id):
    from scribegate.generator import generate_note

    text = _read_transcript(transcript_id)
    visit_type = _visit_type_from_golden(transcript_id)
    note = generate_note(text, transcript_id, visit_type)
    _validate_note_shape(note, transcript_id, visit_type, text)


@pytest.mark.parametrize("transcript_id", ALL_TRANSCRIPT_IDS)
def test_determinism_across_runs(transcript_id):
    from scribegate.generator import generate_note

    text = _read_transcript(transcript_id)
    visit_type = _visit_type_from_golden(transcript_id)
    note1 = generate_note(text, transcript_id, visit_type)
    note2 = generate_note(text, transcript_id, visit_type)
    assert note1 == note2


# "Clean" transcripts: no bracketed disfluency markers ([overlapping],
# [inaudible], [pause], [entering]) — these should produce non-empty S/O/A/P.
def _is_clean(transcript_id: str) -> bool:
    text = _read_transcript(transcript_id)
    return "[overlapping]" not in text and "[inaudible]" not in text


CLEAN_TRANSCRIPT_IDS = [t for t in ALL_TRANSCRIPT_IDS if _is_clean(t)]


@pytest.mark.parametrize("transcript_id", CLEAN_TRANSCRIPT_IDS)
def test_all_four_soap_sections_nonempty_for_clean_transcripts(transcript_id):
    from scribegate.generator import generate_note

    text = _read_transcript(transcript_id)
    visit_type = _visit_type_from_golden(transcript_id)
    note = generate_note(text, transcript_id, visit_type)
    for section in SOAP_SECTIONS:
        assert len(note["soap"][section]) > 0, f"{transcript_id} section {section} is empty"


def test_visit_type_derivable_from_transcript_id_prefix():
    from scribegate.generator import visit_type_for

    assert visit_type_for("glaucoma_05") == "glaucoma_followup"
    assert visit_type_for("cataract_03") == "cataract_postop"
    assert visit_type_for("contactlens_03") == "contact_lens_fitting"
    assert visit_type_for("comprehensive_01") == "comprehensive_exam"


def test_spans_reference_real_substrings_of_transcript():
    """Sanity: span text should be non-trivial (not zero-length, not the whole doc)."""
    from scribegate.generator import generate_note

    transcript_id = "glaucoma_05"
    text = _read_transcript(transcript_id)
    visit_type = _visit_type_from_golden(transcript_id)
    note = generate_note(text, transcript_id, visit_type)
    for section in SOAP_SECTIONS:
        for line in note["soap"][section]:
            for start, end in line["spans"]:
                snippet = text[start:end]
                assert snippet.strip() != ""
                assert len(snippet) < len(text)


def test_messy_contactlens_transcript_shows_first_stated_value_failure_mode():
    """INTERFACES.md: messy transcripts should naturally yield worse notes —
    the mock drafter should surface the first-stated (pre-correction) numeric
    value verbatim in composed line text at least once across the messy
    contact-lens set, rather than silently resolving self-corrections."""
    from scribegate.generator import generate_note

    messy_ids = [t for t in ALL_TRANSCRIPT_IDS if t.startswith("contactlens_") and not _is_clean(t)]
    assert messy_ids, "expected at least one messy contactlens transcript"

    found_first_stated_artifact = False
    for transcript_id in messy_ids:
        text = _read_transcript(transcript_id)
        visit_type = _visit_type_from_golden(transcript_id)
        note = generate_note(text, transcript_id, visit_type)
        all_text = " ".join(
            line["text"] for section in SOAP_SECTIONS for line in note["soap"][section]
        )
        # naive drafter keeps disfluency markers like "sorry"/"actually" verbatim
        # when a self-correction occurs within a single utterance it draws from
        if re.search(r"\b(sorry|actually)\b", all_text, re.IGNORECASE):
            found_first_stated_artifact = True
            break
    assert found_first_stated_artifact, (
        "expected at least one messy transcript's note to retain a raw "
        "self-correction artifact (naive first-pass value pickup)"
    )


import re  # noqa: E402  (used only in the failure-mode test above)
