"""corrections.py (U4) — human-in-the-loop line-level corrections capture.

Reviewers can correct individual SOAP lines in a generated note. Each
correction is recorded as an immutable, append-only record in
data/results/candidate_golden.jsonl (never rewritten/truncated), and a
matching event is appended to data/results/decision_log.jsonl for the
shared audit trail (see audit.py, which reads both).

Corrections can later be merged into a "candidate golden" note
(build_candidate_golden) for potential promotion into data/golden_notes/
by a human maintainer — that promotion step is out of scope here.

stdlib only: json, os, hashlib, difflib, pathlib, datetime, copy.

Import-time side effects: NONE. No directory is created and no file is
touched merely by importing this module — _results_dir() is resolved
fresh inside every function call (never cached at import time or in a
module-level constant computed once), because tests monkeypatch
SCRIBEGATE_RESULTS_DIR per-test and expect the change to take effect
immediately on the next call.
"""

from __future__ import annotations

import copy
import datetime
import difflib
import hashlib
import json
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RESULTS_DIR = _REPO_ROOT / "data" / "results"

_VALID_SECTIONS = ("S", "O", "A", "P")

CANDIDATE_GOLDEN_NAME = "candidate_golden.jsonl"
DECISION_LOG_NAME = "decision_log.jsonl"


def _results_dir() -> Path:
    """Resolve the results directory fresh on every call (env-driven,
    never cached) so tests that monkeypatch SCRIBEGATE_RESULTS_DIR per-test
    see the effect immediately."""
    return Path(os.environ.get("SCRIBEGATE_RESULTS_DIR") or str(_DEFAULT_RESULTS_DIR))


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_result(transcript_id: str) -> dict:
    path = _results_dir() / f"{transcript_id}.json"
    if not path.exists():
        raise ValueError(f"no result file found for transcript_id={transcript_id!r} at {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def record_correction(
    transcript_id: str,
    section: str,
    line_index: int,
    original_text: str,
    corrected_text: str,
    reviewer: str,
    note: str = "",
) -> dict:
    """Validate and record a single line-level correction against the
    generated note for `transcript_id`.

    Validation order (raises ValueError with a clear message on first
    failure encountered, never swallowed):
      1. `section` must be one of "S", "O", "A", "P".
      2. `section` must exist in generated_note["soap"] as a list.
      3. `line_index` must be a valid index into that section's line list.
      4. `original_text` must exactly string-equal the current generated
         line's "text" at that section/index (stale-correction guard).

    On success, builds a correction record (see fields below), appends it
    as one JSON line to data/results/candidate_golden.jsonl (append-only,
    never rewritten/truncated), appends a matching event to
    data/results/decision_log.jsonl, and returns the correction record.

    correction_id is `sha256(f"{transcript_id}|{section}|{line_index}|"
    f"{original_text}|{corrected_text}|{reviewer}|{ts}").hexdigest()[:16]`.
    Including `ts` to-the-second in the hash input means two corrections
    made in the same second with byte-identical content (same transcript,
    section, line_index, original/corrected text, and reviewer) would
    collide. This is acceptable for this scope: correction records are not
    required to be globally unique beyond distinguishing normal review
    activity, and a true same-second byte-identical duplicate submission
    carries no additional information anyway.
    """
    if section not in _VALID_SECTIONS:
        raise ValueError(f"invalid section {section!r}; must be one of {_VALID_SECTIONS}")

    result = _load_result(transcript_id)
    generated_note = result.get("generated_note", {})
    soap = generated_note.get("soap", {})

    lines = soap.get(section)
    if not isinstance(lines, list):
        raise ValueError(
            f"section {section!r} not present as a list in generated_note['soap'] "
            f"for transcript_id={transcript_id!r}"
        )

    if not (0 <= line_index < len(lines)):
        raise ValueError(
            f"line_index {line_index} out of range for section {section} "
            f"(has {len(lines)} lines)"
        )

    line = lines[line_index]
    actual_text = line.get("text")
    if actual_text != original_text:
        raise ValueError(
            f"original_text mismatch for {transcript_id}/{section}[{line_index}]: "
            f"expected (current) {actual_text!r}, got {original_text!r}"
        )

    ts = _utc_now_iso()
    hash_input = f"{transcript_id}|{section}|{line_index}|{original_text}|{corrected_text}|{reviewer}|{ts}"
    correction_id = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]

    record = {
        "correction_id": correction_id,
        "ts": ts,
        "transcript_id": transcript_id,
        "visit_type": result.get("visit_type"),
        "section": section,
        "line_index": line_index,
        "original_text": original_text,
        "corrected_text": corrected_text,
        "reviewer": reviewer,
        "note": note,
        "spans": list(copy.deepcopy(line.get("spans", []))),
    }

    results_dir = _results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    candidate_golden_path = results_dir / CANDIDATE_GOLDEN_NAME
    with open(candidate_golden_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=False))
        fh.write("\n")

    decision_log_path = results_dir / DECISION_LOG_NAME
    log_entry = {
        "ts": ts,
        "transcript_id": transcript_id,
        "event": "correction",
        "correction_id": correction_id,
        "reviewer": reviewer,
        "section": section,
        "line_index": line_index,
    }
    with open(decision_log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(log_entry, sort_keys=False))
        fh.write("\n")

    return record


def list_corrections(transcript_id: str | None = None) -> list[dict]:
    """Read all correction records from candidate_golden.jsonl, in file
    order. Returns [] if the file does not exist. If `transcript_id` is
    given, filters to only that transcript's records."""
    path = _results_dir() / CANDIDATE_GOLDEN_NAME
    if not path.exists():
        return []

    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if transcript_id is not None:
        records = [r for r in records if r.get("transcript_id") == transcript_id]

    return records


def correction_stats() -> dict:
    """Aggregate stats across ALL correction records (no filter).

    Returns {"count": int, "by_visit_type": {visit_type: count, ...},
    "by_section": {"S": n, "O": n, "A": n, "P": n}} — by_section always has
    all 4 keys present even if some counts are 0. Uses the visit_type
    stored on each correction record (does not re-read result files)."""
    records = list_corrections()

    by_visit_type: dict = {}
    by_section = {sec: 0 for sec in _VALID_SECTIONS}

    for r in records:
        vt = r.get("visit_type")
        by_visit_type[vt] = by_visit_type.get(vt, 0) + 1
        sec = r.get("section")
        if sec in by_section:
            by_section[sec] += 1

    return {
        "count": len(records),
        "by_visit_type": by_visit_type,
        "by_section": by_section,
    }


def build_candidate_golden(transcript_id: str) -> dict | None:
    """Merge all recorded corrections for `transcript_id` (applied in file
    order — if multiple corrections target the same section/line_index,
    the LAST one in file order wins, a natural consequence of sequential
    application) onto a deep copy of the generated note, producing a
    "candidate golden" note.

    Returns None if there are no corrections recorded for this transcript.

    Note: "spans" on each corrected line is left UNCHANGED (not
    recomputed) — a deliberate simplification. Spans may become stale
    relative to the corrected text after this merge; recomputing spans
    against corrected text is out of scope here.
    """
    corrections = list_corrections(transcript_id)
    if not corrections:
        return None

    result = _load_result(transcript_id)
    note = copy.deepcopy(result["generated_note"])

    for correction in corrections:
        section = correction["section"]
        line_index = correction["line_index"]
        note["soap"][section][line_index]["text"] = correction["corrected_text"]

    note["candidate"] = True
    note["source_corrections"] = [c["correction_id"] for c in corrections]

    return note


def diff_lines(original: str, corrected: str) -> str:
    """Compact inline diff between two line strings, using stdlib
    difflib only (word-level SequenceMatcher-based rendering).

    Example:
        >>> diff_lines("IOP 18 mmHg OD", "IOP 20 mmHg OD")
        '[IOP] [-18-]{+20+} [mmHg] [OD]'  (illustrative — exact tokens/markup
        may vary, but '-'/'+' change markers are always present when the
        inputs differ)

    If `original == corrected`, returns a string clearly indicating no
    change (contains the substring "unchanged", case-insensitive), e.g.:
        >>> diff_lines("same text", "same text")
        '(unchanged)'
    """
    if original == corrected:
        return "(unchanged)"

    orig_words = original.split()
    corr_words = corrected.split()
    matcher = difflib.SequenceMatcher(a=orig_words, b=corr_words)

    parts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(" ".join(orig_words[i1:i2]))
        elif tag == "delete":
            parts.append(f"[-{' '.join(orig_words[i1:i2])}-]")
        elif tag == "insert":
            parts.append(f"{{+{' '.join(corr_words[j1:j2])}+}}")
        elif tag == "replace":
            parts.append(f"[-{' '.join(orig_words[i1:i2])}-]")
            parts.append(f"{{+{' '.join(corr_words[j1:j2])}+}}")

    return " ".join(p for p in parts if p)
