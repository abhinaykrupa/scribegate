"""audit.py (U5) — assembles a self-contained "dossier" per transcript for
human/legal review: the generated note, judge scores, violations, routing
decision, the full human review trail (route + reviewer decisions +
corrections, all sourced from the shared multi-producer decision_log.jsonl),
recorded corrections, and integrity hashes of the rubric/terminology specs
that governed evaluation — plus a self-hash of the dossier's own content.

Reuses scribegate.corrections for all correction-reading logic (list_corrections)
and diff rendering (diff_lines) rather than reimplementing it, and imports
_results_dir directly from there so both modules resolve
SCRIBEGATE_RESULTS_DIR identically with zero duplication risk.

stdlib only: json, os, hashlib, difflib (via corrections.diff_lines),
pathlib, datetime. No YAML parsing — rubric.yaml/terminology.yaml are only
hashed as raw bytes, never parsed.

Import-time side effects: NONE.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

from scribegate.corrections import _results_dir, diff_lines, list_corrections

_REPO_ROOT = Path(__file__).resolve().parent.parent  # repo-root, for specs/ (NOT results-dir-relative)
_SPECS_DIR = _REPO_ROOT / "specs"

DECISION_LOG_NAME = "decision_log.jsonl"

_SECTION_LABELS = {
    "S": "Subjective",
    "O": "Objective",
    "A": "Assessment",
    "P": "Plan",
}


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_result(transcript_id: str) -> dict:
    path = _results_dir() / f"{transcript_id}.json"
    if not path.exists():
        raise ValueError(f"no result file found for transcript_id={transcript_id!r} at {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _spec_version(path: Path) -> dict:
    """{"path": <absolute str path>, "sha256": <first 16 hex chars of the
    file's raw-bytes sha256>}. Reads in binary mode; never parses YAML."""
    with open(path, "rb") as fh:
        data = fh.read()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest()[:16],
    }


def _read_decision_log_events(transcript_id: str) -> list[dict]:
    """Read every line of decision_log.jsonl (if present), parsing each as
    JSON and skipping (continue) any line that fails to parse. Filters to
    events whose "transcript_id" key (via .get, so malformed/missing-key
    lines simply don't match rather than crashing) equals `transcript_id`.
    Preserves file order. This naturally picks up route-shaped (cli.py),
    reviewer/decision-shaped (streamlit_app.py), and correction-shaped
    (corrections.py) events with no special-casing, since all producers
    share the same "transcript_id" key."""
    path = _results_dir() / DECISION_LOG_NAME
    if not path.exists():
        return []

    events = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if entry.get("transcript_id") == transcript_id:
                events.append(entry)

    return events


def build_dossier(transcript_id: str) -> dict:
    """Assemble the full audit dossier dict for `transcript_id`.

    Sourced from: the result file ({_results_dir()}/{transcript_id}.json),
    decision_log.jsonl (filtered to this transcript's events, across all
    producers), candidate_golden.jsonl (via list_corrections), and the
    repo-root-relative specs/rubric.yaml + specs/terminology.yaml (hashed,
    never parsed).

    Naming note: "rubric_version" / "terminology_version" are named after
    what they represent (which version of the governing spec files were in
    effect), not "rubric_hash" / "terminology_hash", even though their
    content is a {path, sha256} dict — the sha256 is simply how "version"
    is identified for these static, hand-edited YAML files (no separate
    version number exists in the repo).

    "dossier_sha256" is computed over the entire dossier dict built so far
    MINUS "dossier_sha256" and "dossier_generated_at" themselves —
    "dossier_generated_at" is deliberately excluded from the hash so that
    calling build_dossier() multiple times against identical underlying
    data yields an IDENTICAL dossier_sha256 regardless of wall-clock time
    at which the dossier happened to be generated. This makes the hash a
    reliable "has anything actually changed" integrity signal rather than
    changing spuriously on every call.
    """
    result = _load_result(transcript_id)
    generated_note = result.get("generated_note", {})

    dossier: dict = {
        "transcript_id": transcript_id,
        "visit_type": result.get("visit_type"),
        "synthetic": bool(generated_note.get("synthetic")),
        "generated_note": generated_note,
        "judge_result": result.get("judge_result"),
        "violations": result.get("violations"),
        "route": result.get("route"),
        "decision_reasons": result.get("decision_reasons"),
        "decision_log_events": _read_decision_log_events(transcript_id),
        "corrections": list_corrections(transcript_id),
        "rubric_version": _spec_version(_SPECS_DIR / "rubric.yaml"),
        "terminology_version": _spec_version(_SPECS_DIR / "terminology.yaml"),
        "generated_at": (result.get("timestamps") or {}).get("generated_at"),
    }

    payload = {k: v for k, v in dossier.items() if k not in ("dossier_sha256", "dossier_generated_at")}
    dossier_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    dossier["dossier_generated_at"] = _utc_now_iso()
    dossier["dossier_sha256"] = dossier_sha256

    return dossier


def _describe_decision_log_event(event: dict) -> str:
    """Plain-English one-line description of a single decision_log event,
    disambiguated purely by which keys are present (the file is a shared,
    multi-producer, multi-schema log — see corrections.py / cli.py /
    streamlit_app.py for the different shapes)."""
    ts = event.get("ts", "unknown time")

    if "reviewer" in event and "decision" in event and "event" not in event:
        return f"On {ts}, reviewer {event.get('reviewer')} recorded decision: {event.get('decision')}."

    if event.get("event") == "correction":
        return (
            f"On {ts}, reviewer {event.get('reviewer')} submitted a correction to "
            f"section {event.get('section')}, line {event.get('line_index')} "
            f"(correction id {event.get('correction_id')})."
        )

    if "aggregate" in event and "route" in event and "violation_count" in event:
        return (
            f"On {ts}, the pipeline generated this note with aggregate score "
            f"{event.get('aggregate')}, routed to '{event.get('route')}', with "
            f"{event.get('violation_count')} violation(s) flagged."
        )

    # Generic fallback: summarize present keys without raw-dumping JSON.
    other_keys = [k for k in event.keys() if k not in ("ts", "transcript_id")]
    if other_keys:
        summary = ", ".join(f"{k}={event.get(k)}" for k in other_keys)
        return f"On {ts}, an event was recorded with: {summary}."
    return f"On {ts}, an event was recorded (no further details available)."


def render_dossier_md(dossier: dict) -> str:
    """Render `dossier` as a markdown document for a reviewer/lawyer with
    zero system knowledge — spelling things out, avoiding unexplained
    jargon. Section headers appear verbatim, each on its own line, in this
    exact order: Encounter, Generation, Evaluation, Routing, Human Review
    Trail, Corrections, Integrity."""
    lines: list[str] = []

    transcript_id = dossier.get("transcript_id")
    visit_type = dossier.get("visit_type")
    synthetic = dossier.get("synthetic")
    generated_note = dossier.get("generated_note") or {}
    judge_result = dossier.get("judge_result") or {}

    # --- Encounter -----------------------------------------------------
    lines.append("## Encounter")
    lines.append("")
    lines.append(f"- Transcript ID: `{transcript_id}`")
    lines.append(f"- Visit type: {visit_type}")
    if synthetic:
        lines.append(
            "- **SYNTHETIC DATA — no real patient information.** This encounter was "
            "generated for testing/demonstration purposes and does not represent an "
            "actual patient visit."
        )
    else:
        lines.append(
            "- This encounter is **NOT marked synthetic** — it represents real "
            "encounter data and should be handled accordingly."
        )
    lines.append(f"- Note generated at: {dossier.get('generated_at')}")
    lines.append("")

    # --- Generation ------------------------------------------------------
    lines.append("## Generation")
    lines.append("")
    generator = generated_note.get("generator")
    if generator:
        lines.append(f"Generated using backend: {generator}")
        lines.append("")

    soap = generated_note.get("soap") or {}
    for section in ("S", "O", "A", "P"):
        section_lines = soap.get(section) or []
        if not section_lines:
            continue
        label = _SECTION_LABELS.get(section, section)
        lines.append(f"**{section} — {label}**")
        lines.append("")
        for entry in section_lines:
            text = entry.get("text", "") if isinstance(entry, dict) else str(entry)
            lines.append(f"- {text}")
        lines.append("")

    # --- Evaluation --------------------------------------------------------
    lines.append("## Evaluation")
    lines.append("")
    scores = judge_result.get("scores") or {}
    rationales = judge_result.get("rationales") or {}
    lines.append("| Dimension | Score |")
    lines.append("|---|---|")
    for dim, score in scores.items():
        lines.append(f"| {dim} | {score} |")
    aggregate = judge_result.get("aggregate")
    if aggregate is not None:
        lines.append(f"| **Aggregate** | **{aggregate}** |")
    lines.append("")
    if rationales:
        lines.append("Rationales:")
        lines.append("")
        for dim, rationale in rationales.items():
            lines.append(f"- **{dim}**: {rationale}")
        lines.append("")

    # --- Routing -------------------------------------------------------
    lines.append("## Routing")
    lines.append("")
    lines.append(f"Route: **{dossier.get('route')}**")
    lines.append("")
    decision_reasons = dossier.get("decision_reasons") or []
    if decision_reasons:
        lines.append("Reasons:")
        lines.append("")
        for reason in decision_reasons:
            lines.append(f"- {reason}")
        lines.append("")

    # --- Human Review Trail ----------------------------------------------
    lines.append("## Human Review Trail")
    lines.append("")
    events = dossier.get("decision_log_events") or []
    if not events:
        lines.append("No human review or pipeline events were recorded for this transcript.")
        lines.append("")
    else:
        for event in events:
            lines.append(f"- {_describe_decision_log_event(event)}")
        lines.append("")

    # --- Corrections -----------------------------------------------------
    lines.append("## Corrections")
    lines.append("")
    corrections = dossier.get("corrections") or []
    if not corrections:
        lines.append("No corrections have been recorded for this transcript.")
        lines.append("")
    else:
        for c in corrections:
            lines.append(
                f"- Reviewer **{c.get('reviewer')}** corrected section {c.get('section')}, "
                f"line {c.get('line_index')} (correction id `{c.get('correction_id')}`, "
                f"recorded {c.get('ts')}):"
            )
            diff = diff_lines(c.get("original_text", ""), c.get("corrected_text", ""))
            lines.append(f"  - Diff: `{diff}`")
            note = c.get("note")
            if note:
                lines.append(f"  - Note/reason: {note}")
        lines.append("")

    # --- Integrity -----------------------------------------------------
    lines.append("## Integrity")
    lines.append("")
    rubric_version = dossier.get("rubric_version") or {}
    terminology_version = dossier.get("terminology_version") or {}
    lines.append(
        f"- Rubric spec: `{rubric_version.get('path')}` "
        f"(sha256 prefix `{rubric_version.get('sha256')}`)"
    )
    lines.append(
        f"- Terminology spec: `{terminology_version.get('path')}` "
        f"(sha256 prefix `{terminology_version.get('sha256')}`)"
    )
    lines.append(f"- Dossier generated at: {dossier.get('dossier_generated_at')}")
    lines.append(f"- Dossier integrity hash (sha256): `{dossier.get('dossier_sha256')}`")
    lines.append("")

    return "\n".join(lines)


def export_dossier(transcript_id: str, out_dir) -> tuple[Path, Path]:
    """Build the dossier for `transcript_id` and write both a JSON and a
    Markdown rendering to `out_dir` (created if needed). Returns
    (json_path, md_path) as Path objects."""
    dossier = build_dossier(transcript_id)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{transcript_id}_dossier.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(dossier, fh, indent=2, sort_keys=False)
        fh.write("\n")

    md_path = out_dir / f"{transcript_id}_dossier.md"
    md_text = render_dossier_md(dossier)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md_text)

    return json_path, md_path
