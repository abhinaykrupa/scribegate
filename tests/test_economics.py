"""Tests for scribegate.economics (W2) — the unit-economics engine.

Every test that touches the on-disk matrix cache sets SCRIBEGATE_RESULTS_DIR
to a tmp_path-derived directory (mirrors tests/test_moat.py's convention) so
none of them ever write to the real data/results/econ_matrix.json. Reads of
the pristine bundled fixtures (data/transcripts/*.txt, data/golden_notes/
*.json, and the real data/results/golden_generations/gen_1 overlay) use real
transcript ids on purpose, matching test_moat.py's rationale: the matrix
computation deliberately resolves golden notes from the real repo paths.
"""

from __future__ import annotations

import json

import pytest

from scribegate import costs, economics


# ---------------------------------------------------------------------------
# 0. Fallback pricing agreement across modules
# ---------------------------------------------------------------------------

def test_fallback_pricing_agrees_with_costs_module():
    """scribegate.economics._FALLBACK_PRICING and scribegate.costs._FALLBACK_PRICING
    are two independent fallback tables (different key/id conventions: bare
    tier names with *_per_mtok keys here vs. concrete model ids with
    input/output keys in costs.py) that must never silently drift apart —
    both exist so pricing math never crashes if specs/pricing.yaml is
    missing/malformed, and both should reflect the same underlying rates as
    specs/pricing.yaml."""
    tier_to_model_id = {
        "haiku": "claude-haiku-4-5",
        "sonnet": "claude-sonnet-4-5",
        "opus": "claude-opus-4-8",
    }
    for tier, model_id in tier_to_model_id.items():
        econ_rates = economics._FALLBACK_PRICING[tier]
        cost_rates = costs._FALLBACK_PRICING[model_id]
        assert econ_rates["input_per_mtok"] == pytest.approx(cost_rates["input"])
        assert econ_rates["output_per_mtok"] == pytest.approx(cost_rates["output"])


# ---------------------------------------------------------------------------
# 1. cost_per_note — hand-computed
# ---------------------------------------------------------------------------

def test_cost_per_note_hand_computed_sonnet():
    params = economics.NoteEconParams()
    pricing = {"sonnet": {"input_per_mtok": 3.0, "output_per_mtok": 15.0}}
    result = economics.cost_per_note("sonnet", params, pricing)

    # draft: 1200 in @ $3/MTok + 700 out @ $15/MTok
    expected_draft = (1200 / 1_000_000) * 3.0 + (700 / 1_000_000) * 15.0
    # judge: 3 samples * (1500 in @ $3/MTok + 300 out @ $15/MTok)
    expected_judge = 3 * ((1500 / 1_000_000) * 3.0 + (300 / 1_000_000) * 15.0)
    expected_total = expected_draft + expected_judge

    assert result["draft_usd"] == pytest.approx(expected_draft, abs=1e-9)
    assert result["judge_usd"] == pytest.approx(expected_judge, abs=1e-9)
    assert result["total_usd"] == pytest.approx(expected_total, abs=1e-9)
    assert result["tokens"]["total_in"] == 1200 + 3 * 1500
    assert result["tokens"]["total_out"] == 700 + 3 * 300
    assert result["tokens"]["judge_samples"] == 3


def test_cost_per_note_unknown_tier_raises():
    with pytest.raises(ValueError):
        economics.cost_per_note("gpt5", pricing={"haiku": {"input_per_mtok": 1, "output_per_mtok": 1}})


# ---------------------------------------------------------------------------
# 2. margin_model — math exactness
# ---------------------------------------------------------------------------

def test_margin_model_math_exactness():
    params = economics.NoteEconParams(
        providers=2,
        visits_per_provider_per_day=10,
        clinic_days_per_month=20,
        price_per_provider_per_month=100.0,
        fixed_infra_usd=10.0,
    )
    pricing = {"sonnet": {"input_per_mtok": 3.0, "output_per_mtok": 15.0}}
    m = economics.margin_model("sonnet", params, pricing)

    notes_per_month = 2 * 10 * 20
    cost = economics.cost_per_note("sonnet", params, pricing)["total_usd"]
    revenue = 2 * 100.0
    cogs = notes_per_month * cost + 10.0
    gross_margin = revenue - cogs

    assert m["notes_per_month"] == notes_per_month
    assert m["revenue_per_month_usd"] == pytest.approx(revenue, abs=1e-6)
    assert m["cogs_usd"] == pytest.approx(cogs, abs=1e-6)
    assert m["gross_margin_usd"] == pytest.approx(gross_margin, abs=1e-6)
    assert m["gross_margin_pct"] == pytest.approx(gross_margin / revenue, abs=1e-6)


def test_margin_model_can_go_negative_for_expensive_tier():
    # A deliberately tiny revenue vs. an expensive tier should yield a loss.
    params = economics.NoteEconParams(price_per_provider_per_month=1.0)
    pricing = {"opus": {"input_per_mtok": 15.0, "output_per_mtok": 75.0}}
    m = economics.margin_model("opus", params, pricing)
    assert m["gross_margin_usd"] < 0
    assert m["gross_margin_pct"] < 0


# ---------------------------------------------------------------------------
# 3. NoteEconParams overrides
# ---------------------------------------------------------------------------

def test_params_overrides_change_notes_per_month():
    default_params = economics.NoteEconParams()
    custom_params = economics.NoteEconParams(providers=1, visits_per_provider_per_day=1, clinic_days_per_month=1)
    pricing = {"haiku": {"input_per_mtok": 0.8, "output_per_mtok": 4.0}}

    default_margin = economics.margin_model("haiku", default_params, pricing)
    custom_margin = economics.margin_model("haiku", custom_params, pricing)

    assert custom_margin["notes_per_month"] == 1
    assert default_margin["notes_per_month"] == 4 * 22 * 21
    assert custom_margin["notes_per_month"] != default_margin["notes_per_month"]


def test_params_override_is_frozen_dataclass():
    params = economics.NoteEconParams()
    with pytest.raises(Exception):
        params.providers = 99  # frozen dataclass -> raises FrozenInstanceError


# ---------------------------------------------------------------------------
# 4. Pricing fallback / lazy-load path
# ---------------------------------------------------------------------------

def test_load_pricing_missing_file_falls_back(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    pricing = economics.load_pricing(missing)
    assert pricing["haiku"]["input_per_mtok"] == economics._FALLBACK_PRICING["haiku"]["input_per_mtok"]
    assert pricing["sonnet"]["output_per_mtok"] == economics._FALLBACK_PRICING["sonnet"]["output_per_mtok"]
    assert pricing["opus"]["input_per_mtok"] == economics._FALLBACK_PRICING["opus"]["input_per_mtok"]


def test_load_pricing_malformed_file_falls_back(tmp_path):
    bad = tmp_path / "pricing.yaml"
    bad.write_text("not: valid: yaml: [structure\n")
    pricing = economics.load_pricing(bad)
    assert pricing == {t: dict(v) for t, v in economics._FALLBACK_PRICING.items()}


def test_load_pricing_partial_file_merges_with_fallback(tmp_path):
    partial = tmp_path / "pricing.yaml"
    partial.write_text(
        "models:\n"
        "  haiku:\n"
        "    input_per_mtok: 0.50\n"
    )
    pricing = economics.load_pricing(partial)
    # overridden field
    assert pricing["haiku"]["input_per_mtok"] == 0.50
    # non-overridden field falls back
    assert pricing["haiku"]["output_per_mtok"] == economics._FALLBACK_PRICING["haiku"]["output_per_mtok"]
    # untouched tier falls back entirely
    assert pricing["sonnet"] == economics._FALLBACK_PRICING["sonnet"]


def test_cost_per_note_uses_default_load_pricing_when_none_given(monkeypatch, tmp_path):
    monkeypatch.setattr(economics, "_PRICING_PATH", tmp_path / "nope.yaml")
    result = economics.cost_per_note("haiku")
    assert result["total_usd"] > 0


# ---------------------------------------------------------------------------
# 5. tier_comparison ordering
# ---------------------------------------------------------------------------

def test_tier_comparison_ordering_cheapest_first():
    rows = economics.tier_comparison(pricing={t: dict(v) for t, v in economics._FALLBACK_PRICING.items()})
    costs = [r["cost_per_note_usd"] for r in rows]
    assert costs == sorted(costs)
    assert rows[0]["model_tier"] == "haiku"
    assert rows[-1]["model_tier"] == "opus"


def test_tier_comparison_margin_delta_zero_for_priciest_tier():
    rows = economics.tier_comparison(pricing={t: dict(v) for t, v in economics._FALLBACK_PRICING.items()})
    priciest = rows[-1]
    assert priciest["margin_delta_vs_most_expensive_pct"] == 0.0
    for r in rows[:-1]:
        assert r["margin_delta_vs_most_expensive_pct"] == pytest.approx(
            r["gross_margin_pct"] - priciest["gross_margin_pct"], abs=1e-9
        )


# ---------------------------------------------------------------------------
# 6. model_generation_matrix — cache behavior (sandboxed)
# ---------------------------------------------------------------------------

_SUBSET = ["glaucoma_01", "cataract_01"]


def test_matrix_computes_real_cells_and_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(tmp_path / "results"))

    result = economics.model_generation_matrix(subset=_SUBSET)
    assert result["cache_hit"] is False
    assert len(result["cells"]) == 4  # 2 qualities x 2 generations
    for cell in result["cells"]:
        assert cell["n"] == len(_SUBSET)
        assert 0.0 <= cell["mean_aggregate"] <= 1.0

    cache_path = tmp_path / "results" / economics.MATRIX_CACHE_NAME
    assert cache_path.exists()
    on_disk = json.loads(cache_path.read_text())
    assert on_disk["cells"] == result["cells"]


def test_matrix_cache_reused_when_not_stale(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(tmp_path / "results"))

    first = economics.model_generation_matrix(subset=_SUBSET)
    assert first["cache_hit"] is False

    second = economics.model_generation_matrix(subset=_SUBSET)
    assert second["cache_hit"] is True
    assert second["cells"] == first["cells"]


def test_matrix_cache_recompute_when_subset_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(tmp_path / "results"))

    economics.model_generation_matrix(subset=_SUBSET)
    different = economics.model_generation_matrix(subset=["contactlens_01"])
    assert different["cache_hit"] is False
    assert all(c["n"] == 1 for c in different["cells"])


def test_matrix_force_recompute_bypasses_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(tmp_path / "results"))

    economics.model_generation_matrix(subset=_SUBSET)
    forced = economics.model_generation_matrix(subset=_SUBSET, force_recompute=True)
    assert forced["cache_hit"] is False


# ---------------------------------------------------------------------------
# 7. Determinism
# ---------------------------------------------------------------------------

def test_cost_per_note_deterministic():
    a = economics.cost_per_note("sonnet")
    b = economics.cost_per_note("sonnet")
    assert a == b


def test_matrix_deterministic_across_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(tmp_path / "results"))
    first = economics.model_generation_matrix(subset=_SUBSET, force_recompute=True)
    second = economics.model_generation_matrix(subset=_SUBSET, force_recompute=True)
    assert first["cells"] == second["cells"]


# ---------------------------------------------------------------------------
# 8. econ_summary shape + floor logic
# ---------------------------------------------------------------------------

def test_econ_summary_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(tmp_path / "results"))
    matrix = economics.model_generation_matrix(subset=_SUBSET)
    summary = economics.econ_summary(matrix=matrix)

    for key in (
        "latest_generation", "quality_floor", "tier_floor_status",
        "cheapest_tier_meeting_floor", "cheapest_tier_meeting_floor_note",
        "cheapest_tier_margin_pct", "premium_tier",
        "margin_uplift_vs_premium_tier_pct", "cost_per_note_range_usd",
        "tier_comparison",
    ):
        assert key in summary

    assert set(summary["tier_floor_status"].keys()) == set(economics.MODEL_TIERS)
    assert summary["premium_tier"] == "opus"
    assert len(summary["cost_per_note_range_usd"]) == 2
    assert summary["cost_per_note_range_usd"][0] <= summary["cost_per_note_range_usd"][1]


def test_econ_summary_floor_logic_picks_cheapest_qualifying_tier(monkeypatch):
    # Force a matrix where only "opus" (proxied by "baseline") clears the
    # floor, and "haiku" (proxied by "degraded") does not, and verify
    # econ_summary picks the cheapest tier that qualifies, not the
    # cheapest tier overall.
    fake_matrix = {
        "cells": [
            {"model_quality_proxy": "baseline", "generation": "gen_1", "n": 5, "mean_aggregate": 0.9, "meets_floor": True, "transcript_ids": []},
            {"model_quality_proxy": "degraded", "generation": "gen_1", "n": 5, "mean_aggregate": 0.5, "meets_floor": False, "transcript_ids": []},
        ],
        "story": {"applicable": False, "reason": "n/a"},
        "quality_floor": 0.80,
        "label_note": "test",
        "cache_hit": True,
    }
    summary = economics.econ_summary(matrix=fake_matrix)
    # haiku's proxy ("degraded") does not meet floor; sonnet/opus's proxy
    # ("baseline") does -> cheapest qualifying tier is "sonnet".
    assert summary["tier_floor_status"]["haiku"]["meets_floor"] is False
    assert summary["tier_floor_status"]["sonnet"]["meets_floor"] is True
    assert summary["cheapest_tier_meeting_floor"] == "sonnet"


def test_econ_summary_falls_back_when_no_tier_meets_floor():
    fake_matrix = {
        "cells": [
            {"model_quality_proxy": "baseline", "generation": "gen_1", "n": 5, "mean_aggregate": 0.3, "meets_floor": False, "transcript_ids": []},
            {"model_quality_proxy": "degraded", "generation": "gen_1", "n": 5, "mean_aggregate": 0.2, "meets_floor": False, "transcript_ids": []},
        ],
        "story": {"applicable": False, "reason": "n/a"},
        "quality_floor": 0.80,
        "label_note": "test",
        "cache_hit": True,
    }
    summary = economics.econ_summary(matrix=fake_matrix)
    assert summary["cheapest_tier_meeting_floor"] == "haiku"  # cheapest overall
    assert "no tier's quality proxy cleared" in summary["cheapest_tier_meeting_floor_note"]


# ---------------------------------------------------------------------------
# 9. Story (moat -> margin) sanity, using the REAL bundled gen_0/gen_1 goldens
# ---------------------------------------------------------------------------

def test_story_reports_real_direction_honestly(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(tmp_path / "results"))
    result = economics.model_generation_matrix(subset=_SUBSET)
    story = result["story"]
    assert story["applicable"] is True
    assert isinstance(story["effect_present"], bool)
    assert "narrative" in story
    # delta should be the actual arithmetic difference of the two reported means
    assert story["delta"] == pytest.approx(
        story["degraded_latest_mean_aggregate"] - story["degraded_earliest_mean_aggregate"], abs=1e-6
    )


def test_build_markdown_runs_and_contains_headline(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(tmp_path / "results"))
    matrix = economics.model_generation_matrix(subset=_SUBSET)
    md = economics.build_markdown(matrix=matrix)
    assert "ScribeGate Unit Economics" in md
    assert "Moat -> margin" in md
    assert "Headline" in md
