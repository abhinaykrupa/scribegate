"""corrections.py (U4) — human-in-the-loop line-level corrections capture.

Reviewers can correct individual SOAP lines in a generated note. Each
correction is recorded as an immutable, append-only record in
data/results/candidate_golden.jsonl (never rewritten/truncated), and a
matching event is appended to data/results/decision_log.jsonl for the
shared audit trail (see audit.py, which reads both).

Corrections can later be merged into a "candidate golden" note
(build_candidate_golden) for promotion into a generation overlay under
data/results/golden_generations/gen_{N}/ (see promote_candidate /
promote_all_candidates below) — data/golden_notes/ itself (gen-0) is never
modified; it stays the pristine base that every generation overlays on top
of. See active_golden_dir / load_golden_note for the overlay resolver, and
scribegate/moat.py for the metrics/demo layer built on top of this.

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
_TRANSCRIPT_DIR = _REPO_ROOT / "data" / "transcripts"
_GOLDEN_DIR = _REPO_ROOT / "data" / "golden_notes"  # gen-0, pristine, never written to

_VALID_SECTIONS = ("S", "O", "A", "P")

CANDIDATE_GOLDEN_NAME = "candidate_golden.jsonl"
DECISION_LOG_NAME = "decision_log.jsonl"
GOLDEN_GENERATIONS_DIRNAME = "golden_generations"
GENERATION_MANIFEST_NAME = "generation.json"


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


def _load_transcript_text(transcript_id: str) -> str:
    path = _TRANSCRIPT_DIR / f"{transcript_id}.txt"
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Generation model (V1 "data moat"): gen-0 is data/golden_notes/ (pristine,
# never modified). Promotions write ONLY the changed golden notes into a new
# data/results/golden_generations/gen_{N}/ overlay directory alongside a
# generation.json manifest. active_golden_dir()/load_golden_note() resolve a
# transcript's CURRENT golden by walking gen_N -> gen_{N-1} -> ... -> gen_0,
# returning the first generation that actually has an override file for that
# transcript_id (or gen-0 itself if none do) — classic overlay/copy-on-write
# semantics, so a generation only needs to contain the notes it changed.
# ---------------------------------------------------------------------------

def _golden_generations_root() -> Path:
    return _results_dir() / GOLDEN_GENERATIONS_DIRNAME


def _generation_dir(n: int) -> Path:
    return _golden_generations_root() / f"gen_{n}"


def list_generations() -> list[int]:
    """Sorted list of generation numbers (>=1) that have an actual
    generation.json manifest under <results_dir>/golden_generations/gen_{N}/.
    Empty list if no promotions have happened yet. A gen_0/ directory may
    exist purely as a benchmark-summary cache (see moat.rebenchmark_generation)
    but is deliberately NOT counted here — it never represents a real
    promotion and has no manifest."""
    root = _golden_generations_root()
    if not root.exists():
        return []

    gens = []
    for p in root.iterdir():
        if not p.is_dir() or not p.name.startswith("gen_"):
            continue
        suffix = p.name[len("gen_"):]
        if suffix.isdigit() and int(suffix) >= 1 and (p / GENERATION_MANIFEST_NAME).exists():
            gens.append(int(suffix))
    return sorted(gens)


def latest_generation() -> int | None:
    """Highest promoted generation number, or None if no promotions exist yet."""
    gens = list_generations()
    return gens[-1] if gens else None


def active_golden_dir(
    transcript_id: str,
    generation: int | None = None,
    base_golden_dir: Path | None = None,
) -> Path:
    """Resolve the directory that holds the CURRENTLY ACTIVE golden file for
    `transcript_id` at `generation` (default: latest promoted generation, or
    gen-0 if none exist). Walks gen_N -> gen_{N-1} -> ... -> gen_1, returning
    the first generation directory that actually contains a
    {transcript_id}.json override; falls back to `base_golden_dir` (default:
    the pristine data/golden_notes/, i.e. gen-0) if no generation up to and
    including `generation` overrides this transcript.

    This is intentionally per-transcript (not "the one active dir for
    everything") because a generation overlay only contains the notes it
    changed — different transcripts can have their most-recent override in
    different generations.
    """
    base = Path(base_golden_dir) if base_golden_dir is not None else _GOLDEN_DIR
    gen = generation if generation is not None else latest_generation()

    n = gen
    while n is not None and n >= 1:
        candidate_dir = _generation_dir(n)
        if (candidate_dir / f"{transcript_id}.json").exists():
            return candidate_dir
        n -= 1

    return base


def load_golden_note(
    transcript_id: str,
    generation: int | None = None,
    base_golden_dir: Path | None = None,
) -> dict | None:
    """Load the currently-active golden note dict for `transcript_id` at
    `generation` (see active_golden_dir for resolution order). Returns None
    if no golden file exists anywhere in the resolved chain (mirrors
    cli._load_golden's None-safety for transcripts with no golden reference)."""
    d = active_golden_dir(transcript_id, generation=generation, base_golden_dir=base_golden_dir)
    path = d / f"{transcript_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _validate_note_spans_in_bounds(note: dict, transcript_text: str, transcript_id: str) -> None:
    """Raise ValueError (first failure found, never swallowed) if any SOAP
    line's span in `note` is corrupt or falls outside [0, len(transcript_text)]
    — the guard that stops a corrupted candidate golden (e.g. stale spans left
    over from a bad merge) from ever being promoted into a generation overlay."""
    text_len = len(transcript_text)
    soap = note.get("soap", {}) if note else {}
    for section in _VALID_SECTIONS:
        lines = soap.get(section) or []
        for idx, line in enumerate(lines):
            if not isinstance(line, dict):
                continue
            for span in line.get("spans", []) or []:
                if not (isinstance(span, (list, tuple)) and len(span) == 2):
                    raise ValueError(
                        f"corrupt span {span!r} in {transcript_id}/{section}[{idx}]: "
                        f"expected a 2-element [start, end] pair"
                    )
                start, end = span
                if not (isinstance(start, int) and isinstance(end, int)):
                    raise ValueError(
                        f"corrupt span {span!r} in {transcript_id}/{section}[{idx}]: "
                        f"start/end must be integers, got {type(start).__name__}/{type(end).__name__}"
                    )
                if not (0 <= start <= end <= text_len):
                    raise ValueError(
                        f"span {span!r} in {transcript_id}/{section}[{idx}] out of bounds "
                        f"for transcript_id={transcript_id!r} (transcript length {text_len})"
                    )


def _clean_golden_payload(candidate: dict) -> dict:
    """Strip generation-bookkeeping-only fields (candidate flag, source
    correction ids, generator provenance) off a build_candidate_golden()
    result so the file written into a generation overlay has the same shape
    as a hand-authored data/golden_notes/*.json fixture."""
    cleaned = copy.deepcopy(candidate)
    for key in ("candidate", "source_corrections", "generated", "generator"):
        cleaned.pop(key, None)
    return cleaned


def _promote_batch(transcript_ids: list[str], reviewer: str, note: str = "") -> dict:
    """Shared implementation for promote_candidate/promote_all_candidates:
    one promotion batch = one NEW generation directory containing every
    transcript_id's candidate golden (validated), plus a manifest and a
    single decision-log "promotion" event covering the whole batch."""
    if not transcript_ids:
        raise ValueError("no transcript ids given to promote")

    to_write: dict[str, dict] = {}
    source_corrections: dict[str, list[str]] = {}
    for transcript_id in transcript_ids:
        candidate = build_candidate_golden(transcript_id)
        if candidate is None:
            raise ValueError(
                f"no candidate golden available for transcript_id={transcript_id!r} "
                "(no corrections recorded for it)"
            )
        transcript_text = _load_transcript_text(transcript_id)
        _validate_note_spans_in_bounds(candidate, transcript_text, transcript_id)

        source_corrections[transcript_id] = list(candidate.get("source_corrections", []))
        to_write[transcript_id] = _clean_golden_payload(candidate)

    next_gen = (latest_generation() or 0) + 1
    gen_dir = _generation_dir(next_gen)
    gen_dir.mkdir(parents=True, exist_ok=True)

    for transcript_id, payload in to_write.items():
        out_path = gen_dir / f"{transcript_id}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=False)
            fh.write("\n")

    ts = _utc_now_iso()
    promoted_ids = sorted(to_write.keys())
    manifest = {
        "gen": next_gen,
        "ts": ts,
        "reviewer": reviewer,
        "note": note,
        "promoted": promoted_ids,
        "source_corrections": source_corrections,
    }
    manifest_path = gen_dir / GENERATION_MANIFEST_NAME
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")

    log_entry = {
        "ts": ts,
        "event": "promotion",
        "gen": next_gen,
        "reviewer": reviewer,
        "promoted": promoted_ids,
        "note": note,
    }
    decision_log_path = _results_dir() / DECISION_LOG_NAME
    with open(decision_log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(log_entry, sort_keys=False))
        fh.write("\n")

    return manifest


def promote_candidate(transcript_id: str, reviewer: str, note: str = "") -> dict:
    """Promote the single candidate golden for `transcript_id` (built fresh
    from all corrections recorded for it) into a new generation. One call =
    one new generation containing exactly this transcript's overlay. Returns
    the generation manifest dict ({gen, ts, reviewer, promoted, source_corrections}).
    Raises ValueError if there are no corrections recorded for this transcript
    (build_candidate_golden would return None) or if any span in the merged
    candidate is out of bounds for the transcript text."""
    return _promote_batch([transcript_id], reviewer, note)


def promote_all_candidates(reviewer: str, note: str = "") -> dict:
    """Promote candidate goldens for EVERY transcript_id that currently has
    at least one recorded correction, all as ONE new generation (one
    promotion batch = one generation, per the module contract). Raises
    ValueError if no corrections have been recorded for any transcript."""
    ids = sorted({r["transcript_id"] for r in list_corrections()})
    if not ids:
        raise ValueError("no corrections recorded for any transcript; nothing to promote")
    return _promote_batch(ids, reviewer, note)


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
