"""analytics.py (U3) — analytics + ROI backend for a future Streamlit UI.

Contract (mirrors the specs/INTERFACES.md contract-block style; this module
is NOT part of the locked G1 interfaces but follows the same conventions:
pure functions, deterministic, defensive against missing/malformed data,
never crashes on import or on empty input).

stdlib only (dataclasses, statistics, collections, json, os, difflib) — no
third-party dependencies, no pyyaml. Python >= 3.10.

Consumes the result-record shape written by scribegate.cli / scribegate.
benchmark to data/results/{transcript_id}.json:
    {
      "transcript_id": str, "visit_type": str,
      "generated_note": {..., "soap": {"S": [...], "O": [...], "A": [...], "P": [...]}},
      "judge_result": {"scores": {dim: int}, "aggregate": float, "rationales": {dim: str}},
      "violations": [{"code": str, "severity": "error"|"warn", "message": str, "line_text": str}, ...],
      "route": "auto_accept" | "review" | "regenerate",
      "decision_reasons": [...],
      "timestamps": {"generated_at": str},
    }

Golden notes (used by failure_modes' by_section comparison) live at
data/golden_notes/{transcript_id}.json and share the Note dict shape
documented in specs/INTERFACES.md (top-level "soap" key with "S"/"O"/"A"/"P"
lists of {"text": str, "spans": [[start, end], ...]}).

Functions in this module take already-loaded `results: list[dict]` — none
of the 4 analytics functions read from disk themselves except for the
golden-note lookups inside `failure_modes` (which are defensive: a missing
or unreadable golden file for a given transcript_id is silently skipped for
that computation only, never raises).
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean

DIMENSIONS = ("completeness", "hallucination", "coding_plausibility", "terminology")
SECTIONS = ("S", "O", "A", "P")
KNOWN_ROUTES = ("auto_accept", "review", "regenerate")


# ---------------------------------------------------------------------------
# load_results — convenience helper (NOT used internally by the 4 functions)
# ---------------------------------------------------------------------------

def load_results(results_dir: str = "data/results") -> list[dict]:
    """Load every per-transcript result JSON file in `results_dir`.

    Globs for files matching `*.json` inside `results_dir`, parses each as
    JSON, and returns the list of loaded dicts sorted by filename (basename)
    for deterministic ordering. Files that fail to parse as JSON, or that do
    not look like a per-transcript result record (must be a dict containing
    both "transcript_id" and "judge_result" keys), are silently excluded —
    this is how non-per-transcript files such as "benchmark.md" (not even
    JSON) and "decision_log.jsonl" (wrong extension, glob won't match it
    anyway) are excluded. Never raises for a missing/empty directory (glob
    on a nonexistent dir simply yields no matches).

    This is a convenience helper for callers (e.g. a Streamlit app). The 4
    main analytics functions below take an already-loaded `results` list as
    a parameter and do NOT call this helper internally.
    """
    paths = sorted(glob.glob(os.path.join(results_dir, "*.json")), key=os.path.basename)
    results: list[dict] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "transcript_id" in data and "judge_result" in data:
            results.append(data)
    return results


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _violations_of(result: dict) -> list[dict]:
    v = result.get("violations")
    return v if isinstance(v, list) else []


def _scores_of(result: dict) -> dict:
    jr = result.get("judge_result") or {}
    scores = jr.get("scores") or {}
    return scores if isinstance(scores, dict) else {}


def _aggregate_of(result: dict) -> float:
    jr = result.get("judge_result") or {}
    try:
        return float(jr.get("aggregate", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _rationales_of(result: dict) -> dict:
    jr = result.get("judge_result") or {}
    r = jr.get("rationales") or {}
    return r if isinstance(r, dict) else {}


def _load_golden(golden_notes_dir: str, transcript_id: str) -> dict | None:
    path = os.path.join(golden_notes_dir, f"{transcript_id}.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


# ---------------------------------------------------------------------------
# 1. failure_modes
# ---------------------------------------------------------------------------

def failure_modes(results: list[dict], golden_notes_dir: str = "data/golden_notes") -> dict:
    """Cluster weak points across the result set.

    Returns:
        {
          "by_dimension": [
              {"dimension": str, "count_le_3": int,
               "visit_types": [sorted str, ...], "transcript_ids": [sorted str, ...]},
              # exactly 4 entries, one per dimension, ALWAYS in this fixed
              # order regardless of data: completeness, hallucination,
              # coding_plausibility, terminology. Present (count 0 if none)
              # even if no result scored <=3 on that dimension.
          ],
          "by_violation_code": [
              {"code": str, "count": int, "max_severity": "error"|"warn",
               "transcript_ids": [sorted str, ...]},
              # one entry per violation code that appears anywhere across
              # all results' "violations" lists. Sorted by count desc, then
              # code asc (ties broken alphabetically). Empty list if no
              # violations anywhere.
          ],
          "by_section": [
              {"section": "S"|"O"|"A"|"P",
               "mean_line_count_delta": float,   # generated_count - golden_count, mean across
                                                  # transcripts with a loadable golden note; negative
                                                  # means under-documented relative to golden. 0.0 if
                                                  # no transcript had a loadable golden note for this
                                                  # section (never NaN, never crashes).
               "transcripts_below_golden": [sorted str, ...]},  # transcript_ids where
                                                                 # generated has fewer lines than golden
              # exactly 4 entries, fixed order S, O, A, P, always present.
          ],
          "worst_cases": [
              {"transcript_id": str, "visit_type": str, "aggregate": float,
               "route": str, "reasons": [str, ...]},
              # top 5 by lowest aggregate ascending (worst first); if fewer
              # than 5 results total, returns all of them (still sorted
              # ascending by aggregate). Empty list if results == [].
          ],
        }

    "scoring <=3 anywhere" means judge_result["scores"][dim] <= 3 for that
    result. by_violation_code aggregates violations across all results'
    "violations" lists, grouping by "code"; max_severity is "error" if any
    occurrence of that code across the whole result set has severity
    "error", else "warn". by_section loads
    `{golden_notes_dir}/{transcript_id}.json` per transcript (path joined
    with os.path.join, resolved relative to the current working directory);
    if that golden file is missing/unreadable/malformed, that transcript_id
    is skipped for the by_section computation only (never crashes; other
    transcripts/sections are unaffected). worst_cases "reasons" are derived
    from judge_result["rationales"] text for the lowest-scoring dimension(s)
    of that transcript, plus any violation codes present for that
    transcript_id (e.g. "IOP_RANGE (warn)").
    """
    # --- by_dimension ---
    by_dimension = []
    for dim in DIMENSIONS:
        visit_types = set()
        transcript_ids = set()
        count = 0
        for r in results:
            scores = _scores_of(r)
            score = scores.get(dim)
            if isinstance(score, (int, float)) and score <= 3:
                count += 1
                tid = r.get("transcript_id")
                vt = r.get("visit_type")
                if tid is not None:
                    transcript_ids.add(tid)
                if vt is not None:
                    visit_types.add(vt)
        by_dimension.append(
            {
                "dimension": dim,
                "count_le_3": count,
                "visit_types": sorted(visit_types),
                "transcript_ids": sorted(transcript_ids),
            }
        )

    # --- by_violation_code ---
    code_counts: Counter = Counter()
    code_max_severity: dict[str, str] = {}
    code_transcripts: dict[str, set] = defaultdict(set)
    for r in results:
        tid = r.get("transcript_id")
        for v in _violations_of(r):
            if not isinstance(v, dict):
                continue
            code = v.get("code")
            if code is None:
                continue
            severity = v.get("severity")
            code_counts[code] += 1
            if severity == "error":
                code_max_severity[code] = "error"
            elif code not in code_max_severity:
                code_max_severity[code] = "warn"
            if tid is not None:
                code_transcripts[code].add(tid)

    by_violation_code = [
        {
            "code": code,
            "count": count,
            "max_severity": code_max_severity.get(code, "warn"),
            "transcript_ids": sorted(code_transcripts.get(code, set())),
        }
        for code, count in sorted(code_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    # --- by_section ---
    by_section = []
    for section in SECTIONS:
        deltas = []
        below_golden = set()
        for r in results:
            tid = r.get("transcript_id")
            if tid is None:
                continue
            golden = _load_golden(golden_notes_dir, tid)
            if golden is None:
                continue
            golden_soap = (golden.get("soap") or {})
            golden_lines = golden_soap.get(section)
            if not isinstance(golden_lines, list):
                continue
            generated_note = r.get("generated_note") or {}
            gen_soap = (generated_note.get("soap") or {})
            gen_lines = gen_soap.get(section)
            if not isinstance(gen_lines, list):
                continue
            golden_count = len(golden_lines)
            gen_count = len(gen_lines)
            deltas.append(gen_count - golden_count)
            if gen_count < golden_count:
                below_golden.add(tid)
        mean_delta = mean(deltas) if deltas else 0.0
        by_section.append(
            {
                "section": section,
                "mean_line_count_delta": round(mean_delta, 4),
                "transcripts_below_golden": sorted(below_golden),
            }
        )

    # --- worst_cases ---
    def _reasons_for(r: dict) -> list[str]:
        scores = _scores_of(r)
        rationales = _rationales_of(r)
        reasons: list[str] = []
        valid_scores = {d: s for d, s in scores.items() if isinstance(s, (int, float))}
        if valid_scores:
            min_score = min(valid_scores.values())
            worst_dims = sorted(d for d, s in valid_scores.items() if s == min_score)
            for d in worst_dims:
                rationale = rationales.get(d)
                if isinstance(rationale, str) and rationale:
                    reasons.append(f"{d} ({min_score}): {rationale}")
                else:
                    reasons.append(f"{d} ({min_score})")
        for v in _violations_of(r):
            if not isinstance(v, dict):
                continue
            code = v.get("code")
            severity = v.get("severity")
            if code:
                reasons.append(f"{code} ({severity})" if severity else str(code))
        return reasons

    sortable = sorted(results, key=lambda r: (_aggregate_of(r), r.get("transcript_id") or ""))
    worst_cases = [
        {
            "transcript_id": r.get("transcript_id"),
            "visit_type": r.get("visit_type"),
            "aggregate": round(_aggregate_of(r), 4),
            "route": r.get("route"),
            "reasons": _reasons_for(r),
        }
        for r in sortable[:5]
    ]

    return {
        "by_dimension": by_dimension,
        "by_violation_code": by_violation_code,
        "by_section": by_section,
        "worst_cases": worst_cases,
    }


# ---------------------------------------------------------------------------
# 2. routing_summary
# ---------------------------------------------------------------------------

def routing_summary(results: list[dict]) -> dict:
    """Summarize route distribution and aggregate scores across results.

    Returns:
        {
          "total": int,
          "by_route": {
              "auto_accept": {"count": int, "rate": float, "mean_aggregate": float | None},
              "review":      {"count": int, "rate": float, "mean_aggregate": float | None},
              "regenerate":  {"count": int, "rate": float, "mean_aggregate": float | None},
              # any additional unknown route string seen in the data also
              # gets its own key here with the same shape (tallied, not
              # dropped), but the 3 known keys above are ALWAYS present
              # (with count 0 / rate 0.0 / mean_aggregate None if unseen).
          },
          "review_queue_depth": int,  # count("review") + count("regenerate")
          "mean_aggregate_overall": float | None,  # None if results == []
        }

    "rate" = count / total, rounded to 4 decimals; 0.0 if total == 0 (never
    raises ZeroDivisionError). "mean_aggregate" is the mean of
    judge_result["aggregate"] across results routed to that bucket, rounded
    to 4 decimals; None if that bucket has 0 results. "mean_aggregate_overall"
    is the mean aggregate across ALL results, rounded to 4 decimals, or None
    if results is empty.
    """
    total = len(results)
    counts: Counter = Counter()
    aggregates_by_route: dict[str, list[float]] = defaultdict(list)

    for r in results:
        route = r.get("route")
        if route is None:
            route = "unknown"
        counts[route] += 1
        aggregates_by_route[route].append(_aggregate_of(r))

    # Ensure the 3 known routes are always present.
    for known in KNOWN_ROUTES:
        counts.setdefault(known, 0)

    by_route = {}
    for route, count in counts.items():
        rate = round(count / total, 4) if total else 0.0
        agg_list = aggregates_by_route.get(route, [])
        mean_agg = round(mean(agg_list), 4) if agg_list else None
        by_route[route] = {"count": count, "rate": rate, "mean_aggregate": mean_agg}

    review_queue_depth = counts.get("review", 0) + counts.get("regenerate", 0)

    all_aggregates = [_aggregate_of(r) for r in results]
    mean_aggregate_overall = round(mean(all_aggregates), 4) if all_aggregates else None

    return {
        "total": total,
        "by_route": by_route,
        "review_queue_depth": review_queue_depth,
        "mean_aggregate_overall": mean_aggregate_overall,
    }


# ---------------------------------------------------------------------------
# 3. roi_model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoiParams:
    providers: int = 4
    visits_per_provider_per_day: int = 22
    clinic_days_per_month: int = 21
    minutes_full_review: float = 4.0
    minutes_spot_check: float = 0.5
    minutes_regenerate_handling: float = 2.0
    clinician_hourly_cost: float = 140.0


def roi_model(routing: dict, params: RoiParams = RoiParams()) -> dict:
    """Project monthly clinician-time and dollar savings from ScribeGate's
    routing gate, given the output of `routing_summary` and RoiParams.

    Returns:
        {
          "notes_per_month": int,
          "notes_by_route": {"auto_accept": int, "review": int, "regenerate": int},
          "hours_with_gate": float,     # round 1
          "hours_without_gate": float,  # round 1
          "hours_saved_per_month": float,  # round 1; NOT clamped to 0 — can be
                                            # negative if regenerate rate is high
                                            # enough that gate overhead exceeds savings
          "dollars_saved_per_month": float,  # round 2
          "assumptions": {
              "providers": int, "visits_per_provider_per_day": int,
              "clinic_days_per_month": int, "minutes_full_review": float,
              "minutes_spot_check": float, "minutes_regenerate_handling": float,
              "clinician_hourly_cost": float,
              "auto_accept_rate": float, "review_rate": float, "regenerate_rate": float,
              "notes": str,
          },
        }

    notes_per_month = providers * visits_per_provider_per_day * clinic_days_per_month.
    Notes are split across routes using the `rate` fractions from
    `routing["by_route"]` (auto_accept_rate, review_rate, regenerate_rate).
    If `routing` is empty/missing keys, or routing.get("total", 0) == 0,
    all rates are treated as 0 and a zeroed-out result is returned (no
    crash, no ZeroDivisionError) — assumptions["notes"] explains this case.

    Baseline (no gate): every note gets minutes_full_review:
        hours_without_gate = notes_per_month * minutes_full_review / 60
    With gate: auto_accept notes get minutes_spot_check each; review notes
    get minutes_full_review each; regenerate notes get
    (minutes_regenerate_handling + minutes_full_review) each (a regenerate
    retry still needs a full human review pass):
        hours_with_gate = (
            auto_accept_notes * minutes_spot_check
            + review_notes * minutes_full_review
            + regenerate_notes * (minutes_regenerate_handling + minutes_full_review)
        ) / 60
    hours_saved_per_month = hours_without_gate - hours_with_gate (computed
    honestly, not clamped — if the regenerate rate is very high, savings can
    shrink or even go negative, since regenerate costs more per-note than a
    plain review).
    dollars_saved_per_month = hours_saved_per_month * clinician_hourly_cost.
    """
    notes_per_month = int(
        params.providers * params.visits_per_provider_per_day * params.clinic_days_per_month
    )

    by_route = (routing or {}).get("by_route") or {}
    total = (routing or {}).get("total", 0) if routing else 0

    def _rate(route: str) -> float:
        entry = by_route.get(route) or {}
        try:
            return float(entry.get("rate", 0.0))
        except (TypeError, ValueError):
            return 0.0

    if not routing or not total:
        auto_accept_rate = review_rate = regenerate_rate = 0.0
        notes = (
            "routing is empty or total==0; all rates treated as 0, so "
            "notes_by_route/hours/dollars are all zeroed out — no crash, "
            "no division by zero."
        )
    else:
        auto_accept_rate = _rate("auto_accept")
        review_rate = _rate("review")
        regenerate_rate = _rate("regenerate")
        notes = (
            "notes_per_month = providers * visits_per_provider_per_day * "
            "clinic_days_per_month, split across routes using routing_summary's "
            "rate fractions; with-gate hours use minutes_spot_check for "
            "auto_accept, minutes_full_review for review, and "
            "minutes_regenerate_handling + minutes_full_review for regenerate "
            "(a regenerate retry still needs a full human review pass); "
            "hours_saved_per_month is the honest difference vs. the no-gate "
            "baseline (every note gets minutes_full_review) and can shrink "
            "or go negative if the regenerate rate is high enough."
        )

    auto_accept_notes = round(notes_per_month * auto_accept_rate)
    review_notes = round(notes_per_month * review_rate)
    regenerate_notes = round(notes_per_month * regenerate_rate)

    hours_without_gate = notes_per_month * params.minutes_full_review / 60.0

    hours_with_gate = (
        auto_accept_notes * params.minutes_spot_check
        + review_notes * params.minutes_full_review
        + regenerate_notes * (params.minutes_regenerate_handling + params.minutes_full_review)
    ) / 60.0

    hours_saved_per_month = hours_without_gate - hours_with_gate
    dollars_saved_per_month = hours_saved_per_month * params.clinician_hourly_cost

    return {
        "notes_per_month": notes_per_month,
        "notes_by_route": {
            "auto_accept": auto_accept_notes,
            "review": review_notes,
            "regenerate": regenerate_notes,
        },
        "hours_with_gate": round(hours_with_gate, 1),
        "hours_without_gate": round(hours_without_gate, 1),
        "hours_saved_per_month": round(hours_saved_per_month, 1),
        "dollars_saved_per_month": round(dollars_saved_per_month, 2),
        "assumptions": {
            "providers": params.providers,
            "visits_per_provider_per_day": params.visits_per_provider_per_day,
            "clinic_days_per_month": params.clinic_days_per_month,
            "minutes_full_review": params.minutes_full_review,
            "minutes_spot_check": params.minutes_spot_check,
            "minutes_regenerate_handling": params.minutes_regenerate_handling,
            "clinician_hourly_cost": params.clinician_hourly_cost,
            "auto_accept_rate": auto_accept_rate,
            "review_rate": review_rate,
            "regenerate_rate": regenerate_rate,
            "notes": notes,
        },
    }


# ---------------------------------------------------------------------------
# 4. dimension_matrix
# ---------------------------------------------------------------------------

def dimension_matrix(results: list[dict]) -> dict:
    """Build a visit-type x dimension score matrix plus a per-transcript
    detail table.

    Returns:
        {
          "visit_types": [sorted str, ...],  # distinct visit_type values present
          "dimensions": ["completeness", "hallucination", "coding_plausibility", "terminology"],
          "grid": [
              {"visit_type": str, "completeness": float, "hallucination": float,
               "coding_plausibility": float, "terminology": float, "n": int},
              # one row per distinct visit_type, sorted by visit_type ascending;
              # each dimension value is the mean score (round 2) across
              # transcripts of that visit_type; "n" = count of transcripts of
              # that visit_type.
          ],
          "rows": [
              {"transcript_id": str, "visit_type": str, "completeness": int,
               "hallucination": int, "coding_plausibility": int,
               "terminology": int, "aggregate": float},
              # one row per transcript, sorted by transcript_id ascending;
              # dimension values are the raw ints from judge_result["scores"];
              # aggregate is judge_result["aggregate"] (round 4).
          ],
        }

    Empty `results` yields {"visit_types": [], "dimensions": [...4 dims...],
    "grid": [], "rows": []} — never raises.
    """
    by_visit_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        vt = r.get("visit_type")
        if vt is not None:
            by_visit_type[vt].append(r)

    grid = []
    for vt in sorted(by_visit_type.keys()):
        members = by_visit_type[vt]
        row = {"visit_type": vt}
        for dim in DIMENSIONS:
            vals = [
                s.get(dim)
                for s in (_scores_of(m) for m in members)
                if isinstance(s.get(dim), (int, float))
            ]
            row[dim] = round(mean(vals), 2) if vals else 0.0
        row["n"] = len(members)
        grid.append(row)

    rows = []
    for r in sorted(results, key=lambda r: r.get("transcript_id") or ""):
        scores = _scores_of(r)
        row = {"transcript_id": r.get("transcript_id"), "visit_type": r.get("visit_type")}
        for dim in DIMENSIONS:
            row[dim] = scores.get(dim)
        row["aggregate"] = round(_aggregate_of(r), 4)
        rows.append(row)

    return {
        "visit_types": sorted(by_visit_type.keys()),
        "dimensions": list(DIMENSIONS),
        "grid": grid,
        "rows": rows,
    }
