"""calibration.py (V2) — what breaks when the judge is a probabilistic LLM
instead of deterministic mock rules?

`judge.judge_note` / `judge._mock_judge_note` are pure functions: the same
(generated, golden, transcript_text) triple always produces the same 1-5
scores. A real LLM judge (even at temperature 0, and certainly at any
temperature > 0, or across model/prompt revisions) does not have that
property — repeated judging of the *same* note produces a *distribution* of
scores, not a point value. That has a concrete, safety-relevant consequence
for `router.decide`: a note whose single-draw aggregate clears the 0.85
auto_accept bar can easily have a 95% confidence interval whose lower bound
sits below 0.85 — meaning a materially different fraction of "repeat judge
runs" on that same note would NOT have auto-accepted it. Point-estimate
routing (what today's harness does) is blind to that risk entirely.

This module answers the COO's question with working instrumentation rather
than a slide:

  1. `judge_note_sampled` — a stochastic mock judge: n independent noisy
     draws around the deterministic mock judge's scores, where the PER-CASE
     noise variance scales with a real, non-transcript-ID-special-cased
     difficulty signal (see `case_difficulty`): how close the point aggregate
     sits to a router threshold, how many transcript noise/crosstalk markers
     are present, and how many normalizer warn/error violations the
     generated note triggers. Messy contact-lens cases score higher
     difficulty on these real signals (more `[overlapping]`/`[inaudible]`
     markers empirically), so they show up with genuinely wider sampled
     distributions — not because we hard-coded "contactlens = wide".

  2. `route_sampled` — routes on the CI95 LOWER bound instead of the point
     aggregate (same thresholds as `router.decide`, reused unmodified), and
     reports `routing_delta`: what point-estimate routing would have done
     vs. what CI-aware routing does. This is the concrete "what changes in
     production" answer for the COO: any case whose point aggregate clears
     0.85 but whose CI95 lower bound does not gets demoted from
     auto_accept to review (or lower) under CI-aware routing. Nothing ever
     moves the other way — see `route_sampled` docstring for why that's
     mathematically guaranteed here, not just empirically observed.

  3. `calibration_report` — runs (1)+(2) over all 20 bundled synthetic
     cases and summarizes: per-case mean/std/CI/point-route/CI-route, and a
     visit-type breakdown of mean CI width (the messy contact-lens set
     should be visibly widest) plus how many routes changed under CI-aware
     routing vs point-estimate routing.

Nothing in this module touches judge.py or router.py — `judge_note_sampled`
reuses `scribegate.judge._mock_judge_note` (the same deterministic scorer
already exercised by the 244-test suite) as the center of its noise model,
and `route_sampled` reuses `scribegate.router.decide` unmodified, called
twice (once at the mean, once at the CI95 lower bound) — the "more
conservative" property falls straight out of `decide` already being
monotonic in its aggregate argument.

Default path (no SCRIBEGATE_USE_API): stdlib only, deterministic given a
seed, offline. When SCRIBEGATE_USE_API=1 and ANTHROPIC_API_KEY are both set,
the same interface would fan out `n` real API judge calls per note via
`APISampledJudge` (stub class, `anthropic` imported lazily inside it, never
at module load) — that path is illustrative/inert by default and is never
constructed unless both env vars are set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from statistics import median

from scribegate.generator import generate_note, visit_type_for
from scribegate.judge import DIMENSIONS, _aggregate, _mock_judge_note
from scribegate.normalizer import check_note
from scribegate.router import AUTO_ACCEPT_THRESHOLD, REVIEW_THRESHOLD, decide

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_TRANSCRIPT_DIR = _DATA_DIR / "transcripts"
_GOLDEN_DIR = _DATA_DIR / "golden_notes"
_DEFAULT_RESULTS_DIR = _DATA_DIR / "results"
_DEFAULT_OUT_PATH = _DEFAULT_RESULTS_DIR / "calibration_report.json"

# Canonical visit-type ordering (matches scribegate.benchmark).
VISIT_TYPE_ORDER = (
    "comprehensive_exam",
    "glaucoma_followup",
    "cataract_postop",
    "contact_lens_fitting",
)


# ---------------------------------------------------------------------------
# Difficulty model — the "variance scales with case difficulty" signal.
#
# All three components are derived from real signals already computed
# elsewhere in the pipeline (router thresholds, transcript text, normalizer
# violations) — there is no per-transcript-ID special-casing anywhere here.
# ---------------------------------------------------------------------------

# Noise/crosstalk markers that indicate messy audio (as opposed to e.g.
# "[pause]", which just means a silence, not degraded transcription
# confidence). Case-insensitive, matched inside square brackets.
_NOISE_MARKER_RE = re.compile(
    r"\[(overlapping|inaudible|crosstalk|cross-talk|garbled)[^\]]*\]",
    re.IGNORECASE,
)

# Margin-to-threshold component: an aggregate sitting exactly ON a router
# threshold is the most ambiguous possible case (a single flipped point
# could change the route), so margin_component -> 1.0 as margin -> 0, and
# decays to 0.0 once the aggregate is MARGIN_BAND or further from both
# thresholds.
_MARGIN_BAND = 0.15

# Noise-marker component: saturates once a case has >= 2 messy-audio
# markers (empirically the max seen in the bundled contact-lens fixtures).
_NOISE_MARKER_SATURATION = 2.0

# Normalizer warn/error component: errors weighted 2x warns (matching the
# severity weighting judge.score_terminology already uses), saturating at 4
# weighted points.
_WARN_ERROR_SATURATION = 4.0

# Blend weights for the three difficulty components (sum to 1.0). Noise
# markers get the largest weight deliberately: audio quality is the most
# direct proxy for "how much would a probabilistic re-judge actually
# disagree with itself", ahead of threshold proximity or notation slips
# (empirically, on the bundled fixtures, threshold-proximity and
# normalizer-warn signals fire close to uniformly across all four visit
# types, while noise markers are the one signal that is systematically
# concentrated in the messy contact-lens set — see case_difficulty
# docstring / module docstring).
_W_MARGIN = 0.15
_W_NOISE = 0.65
_W_WARN = 0.20


def _noise_marker_count(transcript_text: str) -> int:
    return len(_NOISE_MARKER_RE.findall(transcript_text or ""))


def _warn_error_weighted(generated: dict, transcript_text: str) -> float:
    violations = check_note(generated, transcript=transcript_text)
    warns = sum(1 for v in violations if getattr(v, "severity", None) == "warn")
    errors = sum(1 for v in violations if getattr(v, "severity", None) == "error")
    return warns + 2.0 * errors


def case_difficulty(generated: dict, golden: dict, transcript_text: str) -> float:
    """Derive a 0..1 difficulty score for a case from three real signals:

      1. How close the deterministic mock judge's point aggregate sits to a
         router threshold (0.60 / 0.85) — a near-boundary case is the one
         where repeated probabilistic judging is most likely to disagree
         with itself about the route.
      2. How many transcript noise/crosstalk markers are present (messy
         audio -> a real LLM judge would plausibly read the same
         transcript differently draw to draw).
      3. How many normalizer warn/error violations the generated note
         triggers (notation problems correlate with judge disagreement:
         a borderline terminology call is exactly the kind of thing a
         probabilistic judge wobbles on).

    Deterministic, stdlib-only, no transcript-ID special-casing — purely a
    function of the note/transcript content.
    """
    base = _mock_judge_note(generated, golden, transcript_text)
    aggregate = base["aggregate"]

    margin = min(abs(aggregate - AUTO_ACCEPT_THRESHOLD), abs(aggregate - REVIEW_THRESHOLD))
    margin_component = max(0.0, 1.0 - margin / _MARGIN_BAND)

    noise_component = min(_noise_marker_count(transcript_text) / _NOISE_MARKER_SATURATION, 1.0)

    warn_component = min(_warn_error_weighted(generated, transcript_text) / _WARN_ERROR_SATURATION, 1.0)

    difficulty = _W_MARGIN * margin_component + _W_NOISE * noise_component + _W_WARN * warn_component
    return max(0.0, min(1.0, difficulty))


# Base per-dimension noise std-dev (score points, pre-rounding) at
# difficulty == 0.0, and how many multiples of that base the hardest
# (difficulty == 1.0) cases scale up to. Chosen so an easy case (difficulty
# ~0.2, the comprehensive/glaucoma/cataract norm) rarely flips a score by
# more than 1 point, while a hard case (difficulty ~0.4+, the contact-lens
# norm) visibly produces wider score spreads and CI widths.
_BASE_SIGMA = 0.35
_DIFFICULTY_SCALE = 10.0


def _sigma_for_difficulty(difficulty: float) -> float:
    return _BASE_SIGMA * (1.0 + _DIFFICULTY_SCALE * difficulty)


# ---------------------------------------------------------------------------
# t-distribution critical values (two-tailed, 95%) by degrees of freedom.
# stdlib has no t-distribution quantile function, so this small lookup table
# (standard textbook values) stands in; falls back to the normal-approx
# 1.96 for df beyond the table (n > 31 samples).
# ---------------------------------------------------------------------------

_T_TABLE_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
_NORMAL_APPROX_95 = 1.960


def _t_critical(df: int) -> float:
    if df < 1:
        return _NORMAL_APPROX_95
    return _T_TABLE_95.get(df, _NORMAL_APPROX_95)


def _confidence_interval(mean_value: float, std_value: float, n: int) -> list[float]:
    if n < 2 or std_value <= 0.0:
        return [mean_value, mean_value]
    se = std_value / math.sqrt(n)
    t_crit = _t_critical(n - 1)
    half_width = t_crit * se
    return [mean_value - half_width, mean_value + half_width]


def _stdev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    variance = sum((v - m) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# Flag thresholds (documented, not magic).
# ---------------------------------------------------------------------------

# A case is flagged HIGH_VARIANCE if repeated-draw aggregate std-dev or CI95
# width crosses these bars — either is enough on its own since a fat middle
# (high std) and a fat tail (wide CI, e.g. from small n) are both signs a
# single-draw point estimate is not trustworthy for routing.
_HIGH_VARIANCE_STD_THRESHOLD = 0.10
_HIGH_VARIANCE_CI_WIDTH_THRESHOLD = 0.20

# A dimension is flagged LOW_AGREEMENT if fewer than this fraction of draws
# land within +/-1 of the median draw for that dimension.
_LOW_AGREEMENT_THRESHOLD = 0.70


def _agreement_for_dimension(dim_scores: list[int]) -> float:
    med = median(dim_scores)
    within = sum(1 for s in dim_scores if abs(s - med) <= 1)
    return within / len(dim_scores)


def _aggregate_sampled_result(samples: list[dict], n: int) -> dict:
    """Shared stats layer used by both the mock noise model and the (inert
    by default) real-API fan-out path — computes mean/std/CI/agreement/flags
    purely from a list of n already-produced judge-result dicts, agnostic of
    how those n results were generated."""
    mean_scores: dict[str, float] = {}
    std_scores: dict[str, float] = {}
    agreement: dict[str, float] = {}

    for dim in DIMENSIONS:
        dim_scores = [s["scores"][dim] for s in samples]
        mean_scores[dim] = sum(dim_scores) / len(dim_scores)
        std_scores[dim] = _stdev([float(x) for x in dim_scores])
        agreement[dim] = _agreement_for_dimension(dim_scores)

    aggregates = [s["aggregate"] for s in samples]
    aggregate_mean = sum(aggregates) / len(aggregates)
    aggregate_std = _stdev(aggregates)
    ci95 = _confidence_interval(aggregate_mean, aggregate_std, n)

    flags: list[str] = []
    ci_width = ci95[1] - ci95[0]
    if aggregate_std >= _HIGH_VARIANCE_STD_THRESHOLD or ci_width >= _HIGH_VARIANCE_CI_WIDTH_THRESHOLD:
        flags.append("HIGH_VARIANCE")
    for dim in DIMENSIONS:
        if agreement[dim] < _LOW_AGREEMENT_THRESHOLD:
            flags.append(f"LOW_AGREEMENT_{dim}")

    return {
        "samples": samples,
        "mean_scores": mean_scores,
        "std_scores": std_scores,
        "aggregate_mean": aggregate_mean,
        "aggregate_std": aggregate_std,
        "ci95": ci95,
        "agreement": agreement,
        "flags": flags,
    }


def _mock_judge_note_sampled(
    generated: dict,
    golden: dict,
    transcript_text: str,
    n: int = 7,
    seed: int | None = None,
    _difficulty_override: float | None = None,
) -> dict:
    """Deterministic-given-seed stochastic mock judge. `_difficulty_override`
    is a private testing hook (not part of the public interface) that lets
    tests inject a specific difficulty value directly, to verify the
    variance-scales-with-difficulty relationship in isolation from the
    heuristics in `case_difficulty`."""
    import random

    base = _mock_judge_note(generated, golden, transcript_text)
    difficulty = (
        _difficulty_override
        if _difficulty_override is not None
        else case_difficulty(generated, golden, transcript_text)
    )
    sigma = _sigma_for_difficulty(difficulty)

    rng = random.Random(seed)

    samples: list[dict] = []
    for i in range(n):
        scores = {}
        for dim in DIMENSIONS:
            noise = rng.gauss(0.0, sigma)
            noisy = round(base["scores"][dim] + noise)
            scores[dim] = max(1, min(5, noisy))
        rationales = {
            dim: (
                f"{base['rationales'].get(dim, '')} "
                f"[sampled draw {i + 1}/{n}: base={base['scores'][dim]}, "
                f"difficulty={difficulty:.2f}, sigma={sigma:.2f}]"
            ).strip()
            for dim in DIMENSIONS
        }
        samples.append(
            {
                "scores": scores,
                "aggregate": _aggregate(scores),
                "rationales": rationales,
            }
        )

    result = _aggregate_sampled_result(samples, n)
    result["difficulty"] = difficulty
    return result


class APISampledJudge:
    """Stub for a real-API sampled judge. Only ever constructed by
    `judge_note_sampled` when SCRIBEGATE_USE_API=1 and ANTHROPIC_API_KEY are
    both set — `anthropic` is imported lazily inside `judge_sampled` (never
    at module load), matching `judge.APIJudge`'s convention. Fans out n real
    API judge calls (via `judge.APIJudge`) and reduces them through the same
    `_aggregate_sampled_result` stats layer the mock path uses, so the
    return shape is identical regardless of which path produced it. Never
    active by default; never constructed in the default test/CI path."""

    def __init__(self, model: str = "claude-haiku-4-5"):
        self.model = model

    def judge_sampled(self, generated: dict, golden: dict, transcript_text: str, n: int = 7, seed: int | None = None) -> dict:
        from scribegate.judge import APIJudge  # lazy import, anthropic pulled in inside APIJudge itself

        judge = APIJudge(model=self.model)
        samples = [judge.judge(generated, golden, transcript_text) for _ in range(n)]
        result = _aggregate_sampled_result(samples, n)
        result["difficulty"] = None  # not applicable: variance here is the model's own, not injected
        return result


def judge_note_sampled(
    generated: dict,
    golden: dict,
    transcript_text: str,
    n: int = 7,
    seed: int | None = None,
) -> dict:
    """Judge a note `n` times and return the resulting score distribution.

    Returns:
        {
          "samples": [n judge-result dicts, each shaped like judge_note()'s
                      return value],
          "mean_scores": {dim: float}, "std_scores": {dim: float},
          "aggregate_mean": float, "aggregate_std": float,
          "ci95": [lo, hi],
          "agreement": {dim: fraction of samples within +/-1 of the median},
          "flags": ["HIGH_VARIANCE", "LOW_AGREEMENT_<dim>", ...] (may be empty),
          "difficulty": float in [0, 1] (None for the real-API path),
        }

    Default path (no SCRIBEGATE_USE_API): deterministic given `seed`, offline,
    stdlib only — draws n noisy samples around `judge._mock_judge_note`'s
    deterministic scores, with per-case noise variance set by
    `case_difficulty`. When SCRIBEGATE_USE_API=1 and ANTHROPIC_API_KEY are
    both set, delegates to `APISampledJudge` (n real API calls) instead —
    never the default, never exercised by the test suite.
    """
    use_api = os.environ.get("SCRIBEGATE_USE_API") == "1"
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if use_api and has_key:
        return APISampledJudge().judge_sampled(generated, golden, transcript_text, n=n, seed=seed)
    return _mock_judge_note_sampled(generated, golden, transcript_text, n=n, seed=seed)


# ---------------------------------------------------------------------------
# 2. CI-aware routing
# ---------------------------------------------------------------------------

def route_sampled(sampled_result: dict, violations: list | None = None) -> dict:
    """CI-aware routing: route on the CI95 LOWER bound instead of the point
    aggregate — a note only auto-accepts if even the pessimistic read of
    repeated probabilistic judging still clears 0.85.

    Reuses `router.decide` unmodified, called twice: once at
    `aggregate_mean` (what point-estimate routing would have done) and once
    at `ci95[0]` (what CI-aware routing actually does). Because `decide` is
    monotonic non-decreasing in strictness as its aggregate argument
    decreases (same thresholds, same error-violation override, for both
    calls), and `ci95[0] <= aggregate_mean` always holds (a std-dev can't be
    negative), CI-aware routing can never be LESS conservative than
    point-estimate routing for the same case — it can only hold steady or
    demote (auto_accept -> review, review -> regenerate, etc.), never
    promote. That's a structural guarantee, not an empirical one.

    Returns:
        {
          "route": str,               # the CI-aware route (production route)
          "ci_lower": float, "ci_upper": float, "aggregate_mean": float,
          "reasons": [str, ...],       # from the CI-based decide() call
          "routing_delta": {
              "point_route": str, "ci_route": str, "changed": bool,
              "explanation": str,      # the concrete "what changes in prod" answer
          },
        }
    """
    violations = violations or []
    ci_lo, ci_hi = sampled_result["ci95"]
    aggregate_mean = sampled_result["aggregate_mean"]

    point_decision = decide({"aggregate": aggregate_mean}, violations)
    ci_decision = decide({"aggregate": ci_lo}, violations)
    changed = point_decision.route != ci_decision.route

    if changed:
        explanation = (
            f"Point-estimate routing on the mean aggregate ({aggregate_mean:.3f}) would have "
            f"routed '{point_decision.route}', but the CI95 lower bound ({ci_lo:.3f}) — the "
            "pessimistic read of what repeated probabilistic judging of this same note would "
            f"produce — only supports '{ci_decision.route}'. Production impact: under "
            f"point-estimate routing this case would have been {point_decision.route}d "
            f"{'without human review' if point_decision.route == 'auto_accept' else ''}; "
            f"CI-aware routing is strictly more conservative and sends it to "
            f"'{ci_decision.route}' instead."
        ).replace("  ", " ")
    else:
        explanation = (
            f"CI95 lower bound ({ci_lo:.3f}) agrees with the mean aggregate ({aggregate_mean:.3f}) "
            f"on route '{ci_decision.route}' — no change from point-estimate routing for this case."
        )

    return {
        "route": ci_decision.route,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "aggregate_mean": aggregate_mean,
        "reasons": ci_decision.reasons,
        "routing_delta": {
            "point_route": point_decision.route,
            "ci_route": ci_decision.route,
            "changed": changed,
            "explanation": explanation,
        },
    }


# ---------------------------------------------------------------------------
# 3. Calibration report over all 20 bundled cases
# ---------------------------------------------------------------------------

def _stable_case_seed(base_seed: int, transcript_id: str) -> int:
    """Deterministic per-case seed derived from `base_seed` + `transcript_id`,
    stable across processes/PYTHONHASHSEED (uses hashlib, not Python's salted
    `hash()`) — so `calibration_report(seed=42)` is byte-for-byte
    reproducible regardless of hash randomization."""
    digest = hashlib.sha256(f"{base_seed}:{transcript_id}".encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16)


def _discover_transcript_ids(transcript_dir: Path) -> list[str]:
    return sorted(p.stem for p in transcript_dir.glob("*.txt"))


def _load_transcript_text(transcript_id: str, transcript_dir: Path) -> str:
    with open(transcript_dir / f"{transcript_id}.txt", "r", encoding="utf-8") as fh:
        return fh.read()


def _load_golden(transcript_id: str, golden_dir: Path) -> dict | None:
    path = golden_dir / f"{transcript_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def calibration_report(
    n: int = 7,
    seed: int = 42,
    transcript_dir: Path = _TRANSCRIPT_DIR,
    golden_dir: Path = _GOLDEN_DIR,
) -> dict:
    """Run judge_note_sampled + route_sampled over all bundled cases (fresh
    `generate_note` output for each transcript, same deterministic MockBackend
    path `cli.process_transcript` uses) and summarize the result.

    Deterministic given `seed`: same seed always produces the same per-case
    seeds (via `_stable_case_seed`) and therefore the same report.

    Returns:
        {
          "n": int, "seed": int,
          "cases": [
            {transcript_id, visit_type, difficulty, aggregate_mean,
             aggregate_std, ci95, ci_width, point_route, ci_route, changed,
             agreement, flags},
            ...
          ],
          "summary": {
            "n_cases": int,
            "mean_ci_width_by_visit_type": {vt: float},
            "n_routes_changed": int,
            "changed_transcript_ids": [str, ...],
            "mean_agreement_by_dimension": {dim: float},
          },
        }
    """
    transcript_dir = Path(transcript_dir)
    golden_dir = Path(golden_dir)
    transcript_ids = _discover_transcript_ids(transcript_dir)

    cases: list[dict] = []
    for transcript_id in transcript_ids:
        transcript_text = _load_transcript_text(transcript_id, transcript_dir)
        visit_type = visit_type_for(transcript_id)
        golden = _load_golden(transcript_id, golden_dir) or {}
        generated = generate_note(transcript_text, transcript_id, visit_type, quality="baseline")
        violations = check_note(generated, transcript=transcript_text)

        case_seed = _stable_case_seed(seed, transcript_id)
        sampled = judge_note_sampled(generated, golden, transcript_text, n=n, seed=case_seed)
        routed = route_sampled(sampled, violations)

        ci_lo, ci_hi = sampled["ci95"]
        cases.append(
            {
                "transcript_id": transcript_id,
                "visit_type": visit_type,
                "difficulty": sampled.get("difficulty"),
                "mean_scores": sampled["mean_scores"],
                "std_scores": sampled["std_scores"],
                "aggregate_mean": sampled["aggregate_mean"],
                "aggregate_std": sampled["aggregate_std"],
                "ci95": sampled["ci95"],
                "ci_width": ci_hi - ci_lo,
                "point_route": routed["routing_delta"]["point_route"],
                "ci_route": routed["routing_delta"]["ci_route"],
                "changed": routed["routing_delta"]["changed"],
                "explanation": routed["routing_delta"]["explanation"],
                "agreement": sampled["agreement"],
                "flags": sampled["flags"],
            }
        )

    by_visit_type: dict[str, list[dict]] = {}
    for c in cases:
        by_visit_type.setdefault(c["visit_type"], []).append(c)

    mean_ci_width_by_visit_type = {}
    ordered_types = [vt for vt in VISIT_TYPE_ORDER if vt in by_visit_type]
    ordered_types += sorted(vt for vt in by_visit_type if vt not in VISIT_TYPE_ORDER)
    for vt in ordered_types:
        group = by_visit_type[vt]
        mean_ci_width_by_visit_type[vt] = sum(c["ci_width"] for c in group) / len(group)

    changed_cases = [c for c in cases if c["changed"]]

    mean_agreement_by_dimension = {}
    for dim in DIMENSIONS:
        vals = [c["agreement"][dim] for c in cases]
        mean_agreement_by_dimension[dim] = sum(vals) / len(vals) if vals else 0.0

    summary = {
        "n_cases": len(cases),
        "mean_ci_width_by_visit_type": mean_ci_width_by_visit_type,
        "n_routes_changed": len(changed_cases),
        "changed_transcript_ids": [c["transcript_id"] for c in changed_cases],
        "mean_agreement_by_dimension": mean_agreement_by_dimension,
    }

    return {"n": n, "seed": seed, "cases": cases, "summary": summary}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt(x: float) -> str:
    return f"{x:.3f}"


def build_markdown(report: dict) -> str:
    lines = []
    lines.append("# ScribeGate Calibration Report — probabilistic-judge instrumentation")
    lines.append("")
    lines.append(
        f"n={report['n']} samples/case, seed={report['seed']}. Synthetic/educational data only. "
        "Generated by `python -m scribegate.calibration`."
    )
    lines.append("")
    lines.append(
        "Answers: what breaks when the judge is a probabilistic LLM instead of deterministic "
        "mock rules? Point-estimate routing (today's harness) is blind to score variance across "
        "repeated judging of the SAME note. CI-aware routing (`route_sampled`) routes on the "
        "CI95 lower bound instead, and is structurally guaranteed to never be less conservative "
        "than point-estimate routing (see `route_sampled` docstring)."
    )
    lines.append("")
    lines.append("## Per-case")
    lines.append("")
    lines.append(
        "| Transcript ID | Visit Type | Difficulty | Mean Agg | Std | CI95 | Point Route | CI Route | Changed? |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c in report["cases"]:
        ci_lo, ci_hi = c["ci95"]
        lines.append(
            f"| {c['transcript_id']} | {c['visit_type']} | {_fmt(c['difficulty'])} | "
            f"{_fmt(c['aggregate_mean'])} | {_fmt(c['aggregate_std'])} | "
            f"[{_fmt(ci_lo)}, {_fmt(ci_hi)}] | {c['point_route']} | {c['ci_route']} | "
            f"{'YES' if c['changed'] else 'no'} |"
        )
    lines.append("")

    summary = report["summary"]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- n_cases: {summary['n_cases']}")
    lines.append(f"- routes changed (CI-aware vs point-estimate): {summary['n_routes_changed']}")
    if summary["changed_transcript_ids"]:
        lines.append(f"  - changed: {', '.join(summary['changed_transcript_ids'])}")
    lines.append("- mean CI95 width by visit type (wider = less trustworthy point estimate):")
    for vt, width in summary["mean_ci_width_by_visit_type"].items():
        lines.append(f"  - {vt}: {_fmt(width)}")
    lines.append("- mean agreement by dimension (fraction of draws within +/-1 of the median):")
    for dim, agreement in summary["mean_agreement_by_dimension"].items():
        lines.append(f"  - {dim}: {_fmt(agreement)}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scribegate.calibration")
    parser.add_argument("--n", type=int, default=7, help="Samples per case (default: 7).")
    parser.add_argument("--seed", type=int, default=42, help="Base seed (default: 42).")
    parser.add_argument("--transcript-dir", default=str(_TRANSCRIPT_DIR), help=argparse.SUPPRESS)
    parser.add_argument("--golden-dir", default=str(_GOLDEN_DIR), help=argparse.SUPPRESS)
    parser.add_argument(
        "--out",
        default=str(_DEFAULT_OUT_PATH),
        help="Output path for the JSON report (default: data/results/calibration_report.json).",
    )
    args = parser.parse_args(argv)

    report = calibration_report(
        n=args.n,
        seed=args.seed,
        transcript_dir=Path(args.transcript_dir),
        golden_dir=Path(args.golden_dir),
    )
    markdown = build_markdown(report)
    print(markdown)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
