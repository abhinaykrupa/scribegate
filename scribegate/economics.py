"""economics.py (W2) — the unit-economics engine: a CFO-minded operator's page backend.

Pure functions, deterministic, stdlib + pyyaml (pyyaml only inside
`load_pricing`, imported lazily so this module never fails to import if
pyyaml somehow isn't installed — mirrors judge.py/router.py/normalizer.py's
existing pyyaml usage elsewhere in this repo). No network.

Contract style follows scribegate.analytics (RoiParams pattern: a frozen
dataclass of documented defaults feeding pure calculator functions) and
scribegate.moat (re-benchmarking frozen pipeline output against different
golden generations, caching a per-generation summary to disk).

Pricing note (coordination with a concurrently-active teammate): another
worker owns scribegate/live.py, scribegate/costs.py, and specs/pricing.yaml
in this same milestone. This module does NOT create or require any of
those — `load_pricing()` tries to read specs/pricing.yaml lazily (only when
called, never at import time) using the schema documented on
`_FALLBACK_PRICING` below, and silently falls back to the documented inline
constants for any tier/direction that file doesn't provide (file absent,
unreadable, malformed, or partial). This module's tests never depend on
specs/pricing.yaml existing.

Five building blocks:
  1. NoteEconParams        — documented cost/practice assumptions.
  2. cost_per_note / margin_model / tier_comparison
                           — per-model-tier unit economics.
  3. model_generation_matrix
                           — the moat->margin proof: real benchmark numbers
                             (mock-generator quality knob as a model-quality
                             proxy) x golden generation, cached to disk.
  4. econ_summary          — headline cards for the CFO-facing UI.

Usage: `python -m scribegate.economics` prints tier_comparison and the
moat->margin matrix as markdown.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from scribegate import corrections
from scribegate.generator import generate_note, visit_type_for
from scribegate.judge import judge_note

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_TRANSCRIPT_DIR = _DATA_DIR / "transcripts"
_SPECS_DIR = _REPO_ROOT / "specs"
_PRICING_PATH = _SPECS_DIR / "pricing.yaml"

MATRIX_CACHE_NAME = "econ_matrix.json"

MODEL_TIERS = ("haiku", "sonnet", "opus")

# Inline fallback pricing (USD per 1,000,000 tokens = "per MTok"), used
# whenever specs/pricing.yaml is missing, unreadable, malformed, or simply
# doesn't cover a given model tier / direction. These mirror the shape of
# public Anthropic per-model-tier list pricing (cheap/fast Haiku tier, mid
# Sonnet tier, premium Opus tier) closely enough to be a realistic
# benchmark for this unit-economics model; they are NOT fetched at runtime
# (no network) and are deliberately documented here so the numbers in this
# module's report are traceable to a fixed, reviewable source.
#
# Expected specs/pricing.yaml shape (if/when the other worker's file
# exists): {"models": {tier: {"input_per_mtok": float, "output_per_mtok":
# float}, ...}}. `load_pricing` merges that shape on top of these fallback
# constants per tier per direction.
_FALLBACK_PRICING = {
    "haiku": {"input_per_mtok": 1.00, "output_per_mtok": 5.00},
    "sonnet": {"input_per_mtok": 3.00, "output_per_mtok": 15.00},
    "opus": {"input_per_mtok": 15.00, "output_per_mtok": 75.00},
}

# Which bundled mock-generator quality knob (generator.generate_note's
# `quality` argument: "baseline" | "degraded") stands in for each pricing
# tier's judged output quality in econ_summary's floor check. This is a
# deliberate, clearly-labeled PROXY, not a live measurement — live API runs
# are gated in this repo (see calibration.py / moat.py module docstrings
# for the same SCRIBEGATE_USE_API gating pattern), and the bundled mock
# generator only exposes two quality knobs, not three. haiku (cheapest) is
# proxied by the lower-fidelity "degraded" pass; sonnet and opus (the two
# pricier tiers) both share the "baseline" pass, since there is no third,
# even-higher-fidelity mock knob to distinguish them on quality. Wherever
# this mapping drives a decision (econ_summary), that fact is restated in
# the returned dict, never silently assumed.
QUALITY_PROXY_BY_TIER = {"haiku": "degraded", "sonnet": "baseline", "opus": "baseline"}

# Quality floor used by econ_summary / model_generation_matrix cells:
# normalized aggregate >= 0.80 vs the golden generation being compared
# against (matches specs/rubric.yaml's `router_thresholds.auto_accept` of
# 0.85 minus a small margin — this floor is intentionally a notch below
# auto-accept, since "viable to run in production at all" is a slightly
# lower bar than "clears auto-accept on every note").
QUALITY_FLOOR = 0.80


# ---------------------------------------------------------------------------
# Pricing loader
# ---------------------------------------------------------------------------

def load_pricing(path: Path | str | None = None) -> dict:
    """Load the per-MTok input/output pricing table for each model tier.

    Tries to read `path` (default: specs/pricing.yaml at the repo root) via
    pyyaml, lazily — pyyaml is imported inside this function body, not at
    module load, and the read is wrapped so a missing file, unreadable
    file, malformed YAML, or wrong top-level shape all fall back silently
    rather than raising. Merges whatever the file DOES provide on top of
    `_FALLBACK_PRICING` per tier per direction: e.g. a file that only
    specifies `models.haiku.input_per_mtok` still gets the fallback value
    for `haiku.output_per_mtok` and for every other tier untouched.

    Schema tolerance: this module was written concurrently with a
    teammate's scribegate/costs.py + specs/pricing.yaml (see the module
    docstring), so the exact on-disk shape wasn't known up front. Two
    conventions are both accepted per model entry: value keys
    `input_per_mtok`/`output_per_mtok` (this module's own convention) OR
    plain `input`/`output` (costs.py's convention, also USD per MTok per
    its own header comment); and the model key itself may be a bare tier
    name (`haiku`) OR a concrete model id containing the tier name as a
    substring (`claude-haiku-4-5`) — matched case-insensitively against
    each of MODEL_TIERS. Anything that doesn't match either convention for
    a given tier/direction is simply left at the fallback value; nothing
    here ever raises on an unrecognized shape.

    Returns: {tier: {"input_per_mtok": float, "output_per_mtok": float},
    ...} covering at least every tier in MODEL_TIERS (a pricing file may
    define additional tiers beyond those three; those pass through too,
    keyed by whatever bare tier name they matched or their raw model id if
    no MODEL_TIERS substring matched).
    """
    pricing = {tier: dict(vals) for tier, vals in _FALLBACK_PRICING.items()}

    load_path = Path(path) if path is not None else _PRICING_PATH
    try:
        import yaml  # lazy import — this module works even if pyyaml were absent, until this call
        with open(load_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except (OSError, ValueError, ImportError):
        return pricing
    except Exception:
        # Any YAML-library-specific parse error (yaml.YAMLError and its
        # subclasses) also falls back rather than crashing this module.
        return pricing

    if not isinstance(raw, dict):
        return pricing
    models = raw.get("models")
    if not isinstance(models, dict):
        return pricing

    def _resolve_tier_key(raw_key: str) -> str:
        lowered = raw_key.lower()
        for tier in MODEL_TIERS:
            if tier in lowered:
                return tier
        return raw_key

    for raw_key, vals in models.items():
        if not isinstance(vals, dict):
            continue
        tier = _resolve_tier_key(str(raw_key))
        merged = dict(pricing.get(tier, {}))
        for out_key, aliases in (
            ("input_per_mtok", ("input_per_mtok", "input")),
            ("output_per_mtok", ("output_per_mtok", "output")),
        ):
            for alias in aliases:
                v = vals.get(alias)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    merged[out_key] = float(v)
                    break
        pricing[tier] = merged

    return pricing


# ---------------------------------------------------------------------------
# 1. NoteEconParams
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoteEconParams:
    """Documented cost/practice assumptions feeding the calculators below.
    Every default is derived and explained inline; override any subset via
    `NoteEconParams(field=value, ...)` (frozen dataclass — construct a new
    instance rather than mutating)."""

    # --- Practice / pricing assumptions ---
    # A freed.ai-style per-seat SaaS benchmark: $99/provider/month is in the
    # range publicly-known AI-scribe competitors charge per clinician seat.
    providers: int = 4
    visits_per_provider_per_day: int = 22
    clinic_days_per_month: int = 21
    price_per_provider_per_month: float = 99.0

    # --- Token footprint per note, draft pass ---
    # ~1200 input tokens: the bundled synthetic transcripts (data/transcripts/
    # *.txt) run roughly 600-1500 words for a single ophthalmology encounter
    # (comprehensive exam / glaucoma follow-up / cataract post-op / contact
    # lens fitting); at ~0.75 words/token that's roughly 800-2000 tokens,
    # plus drafting-prompt/system overhead, so 1200 is a realistic midpoint.
    tokens_per_note_draft_in: int = 1200
    # ~700 output tokens: a 4-section SOAP note (S/O/A/P) at the density the
    # bundled golden notes (data/golden_notes/*.json) exhibit runs roughly
    # 350-550 words, i.e. ~500-750 tokens; 700 leaves headroom for a
    # verbose comprehensive-exam note.
    tokens_per_note_draft_out: int = 700

    # --- Token footprint per judge sample ---
    # judge.judge_note / calibration.judge_note_sampled both send the
    # generated note + golden reference note + source transcript to the
    # judge. ~1500 input tokens: generated note (~700) + golden reference
    # note (~500) + a trimmed/summarized transcript excerpt for grounding
    # (~300) is the realistic per-sample judge-prompt footprint.
    tokens_per_note_judge_in: int = 1500
    # ~300 output tokens: a per-dimension 1-5 score plus a one-line
    # rationale for each of the 4 rubric dimensions (specs/rubric.yaml:
    # completeness, hallucination, coding_plausibility, terminology) is a
    # compact structured response, not free-form prose.
    tokens_per_note_judge_out: int = 300
    # calibration.py's judge_note_sampled defaults to n=7 draws (chosen
    # there for CI-width estimation quality); production auto-accept/review
    # routing doesn't need that many independent judge calls per note to be
    # useful, so judge_samples defaults to 3 here — enough to catch a
    # single-draw fluke without tripling judge cost for every note the way
    # n=7 would.
    judge_samples: int = 3

    # --- Fixed monthly infra floor (hosting, storage, monitoring) added to
    # COGS regardless of note volume — a nominal placeholder representing
    # baseline infra spend, not a detailed line-itemized infra bill.
    fixed_infra_usd: float = 50.0


# ---------------------------------------------------------------------------
# 2. cost_per_note / margin_model / tier_comparison
# ---------------------------------------------------------------------------

def cost_per_note(
    model_tier: str,
    params: NoteEconParams = NoteEconParams(),
    pricing: dict | None = None,
) -> dict:
    """Per-note draft + judge token cost for `model_tier`.

    Returns:
        {"draft_usd": float, "judge_usd": float, "total_usd": float,
         "tokens": {"draft_in": int, "draft_out": int,
                    "judge_in_per_sample": int, "judge_out_per_sample": int,
                    "judge_samples": int,
                    "total_in": int, "total_out": int, "total": int}}

    draft_usd = (draft_in/1e6)*input_rate + (draft_out/1e6)*output_rate.
    judge_usd = judge_samples * ((judge_in/1e6)*input_rate +
    (judge_out/1e6)*output_rate) — i.e. every one of `judge_samples`
    independent judge calls costs the same per-sample token footprint.
    total_usd = draft_usd + judge_usd. Raises ValueError for an unknown
    `model_tier` not present in `pricing` (and not in the fallback table),
    since a silently-wrong cost number would be worse than a loud error
    here.
    """
    pricing = pricing if pricing is not None else load_pricing()
    rates = pricing.get(model_tier) or _FALLBACK_PRICING.get(model_tier)
    if rates is None:
        raise ValueError(
            f"unknown model_tier {model_tier!r}; known tiers: {sorted(pricing) or sorted(_FALLBACK_PRICING)}"
        )
    input_rate = float(rates["input_per_mtok"])
    output_rate = float(rates["output_per_mtok"])

    draft_in = params.tokens_per_note_draft_in
    draft_out = params.tokens_per_note_draft_out
    judge_in_per_sample = params.tokens_per_note_judge_in
    judge_out_per_sample = params.tokens_per_note_judge_out
    n_judge = params.judge_samples

    draft_usd = (draft_in / 1_000_000) * input_rate + (draft_out / 1_000_000) * output_rate
    judge_usd = n_judge * (
        (judge_in_per_sample / 1_000_000) * input_rate
        + (judge_out_per_sample / 1_000_000) * output_rate
    )
    total_usd = draft_usd + judge_usd

    total_in = draft_in + n_judge * judge_in_per_sample
    total_out = draft_out + n_judge * judge_out_per_sample

    return {
        "draft_usd": round(draft_usd, 6),
        "judge_usd": round(judge_usd, 6),
        "total_usd": round(total_usd, 6),
        "tokens": {
            "draft_in": draft_in,
            "draft_out": draft_out,
            "judge_in_per_sample": judge_in_per_sample,
            "judge_out_per_sample": judge_out_per_sample,
            "judge_samples": n_judge,
            "total_in": total_in,
            "total_out": total_out,
            "total": total_in + total_out,
        },
    }


def margin_model(
    model_tier: str,
    params: NoteEconParams = NoteEconParams(),
    pricing: dict | None = None,
) -> dict:
    """Monthly margin projection for running `model_tier` on every note.

    Returns:
        {"model_tier": str, "notes_per_month": int,
         "revenue_per_month_usd": float, "cogs_usd": float,
         "cost_per_note_usd": float, "gross_margin_usd": float,
         "gross_margin_pct": float,   # fraction 0..1 (or negative), NOT *100
         "cost_per_note_as_pct_of_revenue": float}  # this one IS already *100

    notes/mo = providers * visits_per_provider_per_day * clinic_days_per_month
    (same formula as analytics.roi_model's notes_per_month). revenue/mo =
    providers * price_per_provider_per_month. COGS = notes/mo *
    cost_per_note(total_usd) + fixed_infra_usd. gross_margin_usd =
    revenue/mo - COGS (never clamped — can be negative if COGS exceeds
    revenue, e.g. a premium model tier at low seat pricing). gross_margin_pct
    = gross_margin_usd / revenue/mo (0.0 if revenue/mo == 0, never raises).
    cost_per_note_as_pct_of_revenue expresses cost_per_note against the
    PER-NOTE revenue basis (price_per_provider_per_month divided by that
    provider's notes/month, so it's independent of the number of providers),
    as a percentage (already multiplied by 100).
    """
    pricing = pricing if pricing is not None else load_pricing()
    cost = cost_per_note(model_tier, params, pricing)
    cost_per_note_usd = cost["total_usd"]

    notes_per_provider_per_month = params.visits_per_provider_per_day * params.clinic_days_per_month
    notes_per_month = params.providers * notes_per_provider_per_month

    revenue_per_month = params.providers * params.price_per_provider_per_month
    cogs_usd = notes_per_month * cost_per_note_usd + params.fixed_infra_usd
    gross_margin_usd = revenue_per_month - cogs_usd
    gross_margin_pct = (gross_margin_usd / revenue_per_month) if revenue_per_month else 0.0

    revenue_per_note = (
        params.price_per_provider_per_month / notes_per_provider_per_month
        if notes_per_provider_per_month
        else 0.0
    )
    cost_per_note_as_pct_of_revenue = (
        round(cost_per_note_usd / revenue_per_note * 100, 4) if revenue_per_note else 0.0
    )

    return {
        "model_tier": model_tier,
        "notes_per_month": notes_per_month,
        "revenue_per_month_usd": round(revenue_per_month, 2),
        "cogs_usd": round(cogs_usd, 2),
        "cost_per_note_usd": cost_per_note_usd,
        "gross_margin_usd": round(gross_margin_usd, 2),
        "gross_margin_pct": round(gross_margin_pct, 6),
        "cost_per_note_as_pct_of_revenue": cost_per_note_as_pct_of_revenue,
    }


def tier_comparison(
    params: NoteEconParams = NoteEconParams(),
    pricing: dict | None = None,
    tiers: tuple[str, ...] = MODEL_TIERS,
) -> list[dict]:
    """One margin_model() row per tier in `tiers`, sorted ascending by
    cost_per_note_usd (cheapest tier first) — with the fallback pricing
    table this is the natural haiku -> sonnet -> opus order, but the sort
    is by actual computed cost, not a hard-coded tier list, so a pricing
    override that re-orders tiers is still reported cheapest-first.

    Each row additionally carries "margin_delta_vs_most_expensive_pct":
    this tier's gross_margin_pct minus the most expensive tier's
    gross_margin_pct (0.0 for the most expensive tier itself; positive for
    every cheaper tier whenever a cheaper tier's margin beats the priciest
    tier's, which is the common case since COGS grows with per-note cost
    but revenue does not)."""
    pricing = pricing if pricing is not None else load_pricing()
    rows = [margin_model(tier, params, pricing) for tier in tiers]
    rows.sort(key=lambda r: r["cost_per_note_usd"])

    most_expensive_row = max(rows, key=lambda r: r["cost_per_note_usd"])
    for r in rows:
        r["margin_delta_vs_most_expensive_pct"] = round(
            r["gross_margin_pct"] - most_expensive_row["gross_margin_pct"], 6
        )
    return rows


# ---------------------------------------------------------------------------
# 3. model_generation_matrix — the moat->margin proof
# ---------------------------------------------------------------------------

def _discover_transcript_ids() -> list[str]:
    return sorted(p.stem for p in _TRANSCRIPT_DIR.glob("*.txt"))


def _normalize_generation(g) -> int:
    if isinstance(g, int):
        return g
    s = str(g)
    if s.startswith("gen_"):
        s = s[4:]
    return int(s)


def _generation_label(n: int) -> str:
    return f"gen_{n}"


def _matrix_cache_path() -> Path:
    return corrections._results_dir() / MATRIX_CACHE_NAME


def _matrix_signature(models: list[str], generations: list, subset: list[str] | None) -> dict:
    """JSON-serializable signature of everything that would change the
    matrix's real computed numbers: the requested (models, generations,
    subset) key, plus per-generation per-transcript resolved golden file
    path + mtime (so promoting a new correction generation, or editing an
    existing overlay/pristine golden file, invalidates the cache). This is
    the "stale" check behind model_generation_matrix's "recompute only if
    missing/stale" caching rule."""
    transcript_ids = list(subset) if subset else _discover_transcript_ids()
    gen_sig = []
    for g in generations:
        n = _normalize_generation(g)
        per_tid = []
        for tid in transcript_ids:
            d = corrections.active_golden_dir(tid, generation=n)
            p = d / f"{tid}.json"
            mtime = p.stat().st_mtime if p.exists() else None
            per_tid.append([tid, str(p), mtime])
        gen_sig.append([n, per_tid])
    return {
        "models": list(models),
        "generations": [_normalize_generation(g) for g in generations],
        "subset": list(subset) if subset else None,
        "transcript_ids": transcript_ids,
        "golden_signature": gen_sig,
    }


def _build_story(cells: list[dict], generations: list) -> dict:
    """The moat->margin narrative: does the degraded (cheap-proxy) model
    clear a higher bar against the LATEST requested golden generation than
    against the EARLIEST one? Built purely from the real `cells` computed
    by model_generation_matrix — never tuned to force either answer; if the
    bundled fixtures don't show the effect, `effect_present` is False and
    `narrative` says so plainly."""
    normalized = sorted({_normalize_generation(g) for g in generations})
    if len(normalized) < 2:
        return {"applicable": False, "reason": "fewer than 2 distinct generations requested; nothing to compare"}

    gen0, gen_latest = normalized[0], normalized[-1]

    def _cell(quality, gen_n):
        label = _generation_label(gen_n)
        for c in cells:
            if c["model_quality_proxy"] == quality and c["generation"] == label:
                return c
        return None

    degraded_gen0 = _cell("degraded", gen0)
    degraded_latest = _cell("degraded", gen_latest)

    if degraded_gen0 is None or degraded_latest is None:
        return {
            "applicable": False,
            "reason": (
                "no 'degraded' quality-proxy cell for both the earliest and latest "
                "requested generations (pass models=['baseline','degraded'] to compute this)"
            ),
        }

    delta = round(degraded_latest["mean_aggregate"] - degraded_gen0["mean_aggregate"], 6)
    effect_present = degraded_latest["mean_aggregate"] > degraded_gen0["mean_aggregate"]
    clears_floor_only_at_latest = (not degraded_gen0["meets_floor"]) and degraded_latest["meets_floor"]

    if effect_present:
        narrative_tail = (
            "This IS the moat->margin effect on these real numbers: the golden-set corrections "
            "captured in the newer generation let the cheap-proxy model clear a higher bar, which "
            "is exactly the mechanism that would let a cheaper model tier become viable over time "
            "as the golden set compounds."
        )
    else:
        narrative_tail = (
            "This does NOT show the moat->margin effect on the bundled fixtures: the cheap-proxy "
            "model does not score higher against the newer generation's goldens than against the "
            "pristine gen_0 goldens, so no upgrade-golden-set -> cheaper-model-viable story is "
            "supported by these real numbers as computed."
        )

    return {
        "applicable": True,
        "earliest_generation": _generation_label(gen0),
        "latest_generation": _generation_label(gen_latest),
        "degraded_earliest_mean_aggregate": degraded_gen0["mean_aggregate"],
        "degraded_latest_mean_aggregate": degraded_latest["mean_aggregate"],
        "delta": delta,
        "effect_present": effect_present,
        "clears_quality_floor_only_at_latest_generation": clears_floor_only_at_latest,
        "narrative": (
            f"Cheap-proxy (degraded) mean aggregate moved from {degraded_gen0['mean_aggregate']:.4f} "
            f"(vs {_generation_label(gen0)} goldens) to {degraded_latest['mean_aggregate']:.4f} "
            f"(vs {_generation_label(gen_latest)} goldens), a delta of {delta:+.4f}. " + narrative_tail
        ),
    }


def model_generation_matrix(
    models: list[str] | None = None,
    generations: list | None = None,
    subset: list[str] | None = None,
    force_recompute: bool = False,
) -> dict:
    """The moat->margin proof: for each (model-quality-proxy, golden
    generation) cell, generate a fresh note for every bundled transcript
    (or every transcript_id in `subset`, if given) via the EXISTING mock
    generator's `quality` knob (`quality="baseline"` proxies a strong
    model, `quality="degraded"` proxies a cheap model — see
    QUALITY_PROXY_BY_TIER's docstring for why this is a labeled proxy, not
    a live measurement), judge it against the golden resolved at that
    generation (`corrections.load_golden_note`), and aggregate the mean
    normalized score across included transcripts. These are REAL numbers
    computed by the bundled deterministic pipeline (generator.generate_note
    + judge.judge_note), not fabricated or tuned to fit a narrative.

    Returns:
        {"signature": {...}, "cells": [
            {"model_quality_proxy": "baseline"|"degraded",
             "generation": "gen_0"|"gen_1"|...,
             "n": int, "mean_aggregate": float, "meets_floor": bool,
             "transcript_ids": [str, ...]},
            ...  # one per (model, generation) pair requested
         ],
         "story": {...},  # see _build_story
         "quality_floor": float,
         "label_note": str,  # the "this is a mock-proxy, not live-API" caveat
         "cache_hit": bool}

    Caching: results are cached at
    `<active results dir>/econ_matrix.json` (respects
    SCRIBEGATE_RESULTS_DIR via corrections._results_dir(), so tests can
    sandbox this fully). Recomputes only if the cache file is missing,
    unreadable, or its signature (see `_matrix_signature`) no longer
    matches the current request + on-disk golden files — i.e. only if
    missing/stale, per the W2 spec. Pass `force_recompute=True` to bypass
    the cache unconditionally (used by tests exercising the stale path
    without needing to touch mtimes).
    """
    models = list(models) if models is not None else ["baseline", "degraded"]
    generations = list(generations) if generations is not None else [0, 1]

    signature = _matrix_signature(models, generations, subset)
    cache_path = _matrix_cache_path()

    if not force_recompute and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                cached = json.load(fh)
        except (OSError, json.JSONDecodeError):
            cached = None
        if isinstance(cached, dict) and cached.get("signature") == signature:
            result = dict(cached)
            result["cache_hit"] = True
            return result

    transcript_ids = list(subset) if subset else _discover_transcript_ids()

    cells = []
    for quality in models:
        for g in generations:
            n = _normalize_generation(g)
            aggregates: list[float] = []
            included_ids: list[str] = []
            for tid in transcript_ids:
                transcript_path = _TRANSCRIPT_DIR / f"{tid}.txt"
                if not transcript_path.exists():
                    continue
                transcript_text = transcript_path.read_text(encoding="utf-8")
                golden = corrections.load_golden_note(tid, generation=n)
                if golden is None:
                    continue
                visit_type = visit_type_for(tid)
                generated = generate_note(transcript_text, tid, visit_type, quality=quality)
                judge_result = judge_note(generated, golden, transcript_text)
                aggregates.append(judge_result["aggregate"])
                included_ids.append(tid)

            mean_aggregate = round(sum(aggregates) / len(aggregates), 6) if aggregates else 0.0
            cells.append(
                {
                    "model_quality_proxy": quality,
                    "generation": _generation_label(n),
                    "n": len(included_ids),
                    "mean_aggregate": mean_aggregate,
                    "meets_floor": mean_aggregate >= QUALITY_FLOOR,
                    "transcript_ids": included_ids,
                }
            )

    story = _build_story(cells, generations)

    result = {
        "signature": signature,
        "cells": cells,
        "story": story,
        "quality_floor": QUALITY_FLOOR,
        "label_note": (
            "model_quality_proxy values ('baseline'/'degraded') are the bundled MOCK generator's "
            "quality knob, used as a stand-in for strong-model/cheap-model output quality because "
            "live API runs are gated in this repo — these are REAL computed benchmark numbers from "
            "the deterministic mock pipeline, not live-API measurements. See QUALITY_PROXY_BY_TIER "
            "for how this maps onto the haiku/sonnet/opus pricing tiers used elsewhere in this module."
        ),
        "cache_hit": False,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=False)
        fh.write("\n")

    return result


# ---------------------------------------------------------------------------
# 4. econ_summary — headline cards for the CFO-facing UI
# ---------------------------------------------------------------------------

def econ_summary(
    params: NoteEconParams = NoteEconParams(),
    pricing: dict | None = None,
    matrix: dict | None = None,
) -> dict:
    """Headline cards for a Streamlit "unit economics" page.

    Returns:
        {"latest_generation": "gen_N" | None,
         "quality_floor": float,
         "tier_floor_status": {tier: {"quality_proxy": str,
             "mean_aggregate_vs_latest_generation": float | None,
             "meets_floor": bool}, ...},  # one entry per MODEL_TIERS
         "cheapest_tier_meeting_floor": str,   # falls back to cheapest tier
                                                # overall (flagged in the
                                                # accompanying note) if NO
                                                # tier's proxy clears the floor
         "cheapest_tier_meeting_floor_note": str,
         "cheapest_tier_margin_pct": float,
         "premium_tier": str,                  # most expensive tier by cost/note
         "margin_uplift_vs_premium_tier_pct": float,
         "cost_per_note_range_usd": [min, max],
         "tier_comparison": [... tier_comparison() rows ...]}

    "cheapest tier meeting quality floor" = the tier (haiku/sonnet/opus)
    whose QUALITY_PROXY_BY_TIER-mapped mock-generator quality knob's
    mean_aggregate against the LATEST generation in `matrix` is >=
    QUALITY_FLOOR (0.80), with the lowest cost_per_note_usd among those
    that qualify. If none qualify, falls back to the cheapest tier overall
    and says so explicitly rather than silently picking one.
    """
    pricing = pricing if pricing is not None else load_pricing()
    matrix = matrix if matrix is not None else model_generation_matrix()

    generations_present = sorted({_normalize_generation(c["generation"]) for c in matrix["cells"]})
    latest_gen = generations_present[-1] if generations_present else None
    latest_label = _generation_label(latest_gen) if latest_gen is not None else None

    def _quality_cell(quality: str | None):
        if quality is None or latest_label is None:
            return None
        for c in matrix["cells"]:
            if c["model_quality_proxy"] == quality and c["generation"] == latest_label:
                return c
        return None

    comparison = tier_comparison(params, pricing)
    by_tier = {r["model_tier"]: r for r in comparison}

    tier_floor_status = {}
    for tier in MODEL_TIERS:
        quality = QUALITY_PROXY_BY_TIER.get(tier)
        cell = _quality_cell(quality)
        tier_floor_status[tier] = {
            "quality_proxy": quality,
            "mean_aggregate_vs_latest_generation": cell["mean_aggregate"] if cell else None,
            "meets_floor": bool(cell["meets_floor"]) if cell else False,
        }

    eligible = [t for t in MODEL_TIERS if t in by_tier and tier_floor_status[t]["meets_floor"]]
    if eligible:
        cheapest_tier = min(eligible, key=lambda t: by_tier[t]["cost_per_note_usd"])
        floor_note = (
            f"cheapest tier whose quality proxy clears the {QUALITY_FLOOR} floor vs {latest_label}"
        )
    else:
        cheapest_tier = min(MODEL_TIERS, key=lambda t: by_tier[t]["cost_per_note_usd"])
        floor_note = (
            f"no tier's quality proxy cleared the {QUALITY_FLOOR} floor vs {latest_label} on these "
            "real computed numbers; falling back to the cheapest tier overall (flagged, not silently "
            "assumed safe)"
        )

    premium_tier = max(MODEL_TIERS, key=lambda t: by_tier[t]["cost_per_note_usd"])
    cost_values = [by_tier[t]["cost_per_note_usd"] for t in MODEL_TIERS]

    return {
        "latest_generation": latest_label,
        "quality_floor": QUALITY_FLOOR,
        "tier_floor_status": tier_floor_status,
        "cheapest_tier_meeting_floor": cheapest_tier,
        "cheapest_tier_meeting_floor_note": floor_note,
        "cheapest_tier_margin_pct": by_tier[cheapest_tier]["gross_margin_pct"],
        "premium_tier": premium_tier,
        "margin_uplift_vs_premium_tier_pct": round(
            by_tier[cheapest_tier]["gross_margin_pct"] - by_tier[premium_tier]["gross_margin_pct"], 6
        ),
        "cost_per_note_range_usd": [round(min(cost_values), 6), round(max(cost_values), 6)],
        "tier_comparison": comparison,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def build_markdown(
    params: NoteEconParams | None = None,
    pricing: dict | None = None,
    matrix: dict | None = None,
) -> str:
    params = params or NoteEconParams()
    pricing = pricing if pricing is not None else load_pricing()
    comparison = tier_comparison(params, pricing)
    matrix = matrix if matrix is not None else model_generation_matrix()
    summary = econ_summary(params, pricing, matrix)

    lines: list[str] = []
    lines.append("# ScribeGate Unit Economics (W2)")
    lines.append("")
    lines.append(
        "Synthetic/educational data only. Generated by `python -m scribegate.economics`."
    )
    lines.append("")
    lines.append("## Tier comparison")
    lines.append("")
    lines.append(
        "| Tier | Cost/note | Notes/mo | Revenue/mo | Gross margin % | Margin delta vs priciest tier |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in comparison:
        lines.append(
            f"| {r['model_tier']} | ${r['cost_per_note_usd']:.4f} | {r['notes_per_month']} | "
            f"${r['revenue_per_month_usd']:.2f} | {_fmt_pct(r['gross_margin_pct'])} | "
            f"{_fmt_pct(r['margin_delta_vs_most_expensive_pct'])} |"
        )
    lines.append("")
    lines.append("## Moat -> margin proxy matrix")
    lines.append("")
    lines.append(matrix["label_note"])
    lines.append("")
    lines.append("| Quality proxy | Generation | n | Mean aggregate | Meets 0.80 floor? |")
    lines.append("|---|---|---|---|---|")
    for c in matrix["cells"]:
        lines.append(
            f"| {c['model_quality_proxy']} | {c['generation']} | {c['n']} | "
            f"{c['mean_aggregate']:.4f} | {'YES' if c['meets_floor'] else 'no'} |"
        )
    lines.append("")
    story = matrix["story"]
    lines.append(f"Story: {story.get('narrative', story.get('reason'))}")
    lines.append("")
    lines.append("## Headline (econ_summary)")
    lines.append("")
    lines.append(
        f"- Cheapest tier meeting quality floor: **{summary['cheapest_tier_meeting_floor']}** "
        f"({summary['cheapest_tier_meeting_floor_note']})"
    )
    lines.append(f"- Its gross margin: {_fmt_pct(summary['cheapest_tier_margin_pct'])}")
    lines.append(
        f"- Margin uplift vs premium tier ({summary['premium_tier']}): "
        f"{_fmt_pct(summary['margin_uplift_vs_premium_tier_pct'])}"
    )
    lines.append(
        f"- Cost/note range: ${summary['cost_per_note_range_usd'][0]:.4f} - "
        f"${summary['cost_per_note_range_usd'][1]:.4f}"
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scribegate.economics")
    parser.parse_args(argv)
    print(build_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
