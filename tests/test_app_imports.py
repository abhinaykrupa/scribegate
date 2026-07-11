"""Tests for app/ import health and DEMO_SCRIPT.md/data fidelity.

Two concerns:
  1. app/common.py and every app/views/*.py module must import cleanly
     (guards against the class of bug this suite is fixing today — a
     module-level bug or stale reference that only surfaces at import
     time, e.g. `streamlit run` failing to even load a page).
  2. DEMO_SCRIPT.md's quoted glaucoma_05 Plan line must appear verbatim in
     data/results/glaucoma_05.json — locks the demo script to the actual
     generated fixture data so a future regeneration of that fixture (or a
     script edit) can't silently drift the two apart again, which is
     exactly the regression a fresh-eyes QA pass caught (P1-1).
"""

from __future__ import annotations

import glob
import importlib
import json
import os
import sys
import time

import pytest
import yaml

pytest.importorskip("streamlit")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_SCRIPT_PATH = os.path.join(REPO_ROOT, "DEMO_SCRIPT.md")
GLAUCOMA_05_RESULT_PATH = os.path.join(REPO_ROOT, "data", "results", "glaucoma_05.json")
UI_COPY_PATH = os.path.join(REPO_ROOT, "specs", "ui_copy.yaml")

APP_MODULES = [
    "app.common",
    "app.views.about",
    "app.views.analytics",
    "app.views.calibration",
    "app.views.drift",
    "app.views.economics",
    "app.views.live_encounter",
    "app.views.live_mode",
    "app.views.moat",
    "app.views.overview",
    "app.views.provenance",
    "app.views.review_queue",
    "app.views.start_here",
]


@pytest.mark.parametrize("module_name", APP_MODULES)
def test_app_module_imports_cleanly(module_name):
    module = importlib.import_module(module_name)
    assert module is not None


def test_demo_script_glaucoma_05_line_matches_fixture():
    """The Plan line DEMO_SCRIPT.md tells the presenter to click for
    glaucoma_05 must be the literal text of a real generated Plan line —
    not a paraphrase or a line that doesn't exist in the fixture."""
    with open(DEMO_SCRIPT_PATH, "r", encoding="utf-8") as fh:
        demo_script = fh.read()
    # DEMO_SCRIPT.md hard-wraps quoted lines across multiple markdown lines
    # (each continuation prefixed with "> "), so normalize all whitespace
    # before comparing rather than requiring an exact substring match.
    normalized_demo_script = " ".join(demo_script.replace("> ", " ").split())

    with open(GLAUCOMA_05_RESULT_PATH, "r", encoding="utf-8") as fh:
        result = json.load(fh)

    plan_lines = result["generated_note"]["soap"]["P"]
    plan_texts = [line["text"] for line in plan_lines]

    quoted_line = (
        "I know it's a big step, but the right eye needs more than drops can "
        "give. In the meantime, keep taking all 3 medications as best you can "
        "— every bit of pressure lowering helps until surgery. I'll continue "
        "the latanoprost, timolol, and brimonidine unchanged for now. I'll "
        "call the surgeon's office today and want you seen within two weeks. "
        "The staff will arrange the surgical referral and get you the "
        "soonest appointment."
    )

    assert quoted_line in plan_texts, "quoted line must be a real glaucoma_05 Plan line"
    normalized_quoted_line = " ".join(quoted_line.split())
    assert normalized_quoted_line in normalized_demo_script, (
        "DEMO_SCRIPT.md must quote the line verbatim (modulo markdown line-wrap whitespace)"
    )

    matching_line = next(line for line in plan_lines if line["text"] == quoted_line)
    assert len(matching_line["spans"]) == 5


def test_app_script_executes_without_exception():
    """Regression: st.Page url_path collisions (all views expose `render`)
    only surface when the script actually runs — health checks don't catch it."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(REPO_ROOT, "app", "streamlit_app.py"), default_timeout=30).run()
    assert not at.exception, f"app script raised: {at.exception}"


VIEW_MODULES_FOR_RENDER_SWEEP = [
    "overview",
    "analytics",
    "drift",
    "moat",
    "calibration",
    "review_queue",
    "provenance",
    "live_encounter",
    "about",
    "start_here",
    "live_mode",
    "economics",
]


@pytest.mark.parametrize("view_name", VIEW_MODULES_FOR_RENDER_SWEEP)
def test_view_render_survives_without_matplotlib(view_name, monkeypatch):
    """Regression for the matplotlib-ImportError crash class (analytics.py's
    df.style.background_gradient(cmap=...) raises ImportError when
    matplotlib isn't installed, and pandas' .style accessor itself imports
    matplotlib-adjacent machinery lazily). Simulate a minimal environment
    that lacks matplotlib entirely by blocking the import at the
    sys.modules level, then require every view's render() to still
    complete without raising — pages must degrade gracefully, not crash."""
    pytest.importorskip("streamlit")

    # Block matplotlib (and any submodule import, e.g. matplotlib.pyplot)
    # by making sys.modules lookups for it resolve to None, which forces
    # Python's import machinery to raise ImportError, and clear any already
    # -imported matplotlib modules so the blocked state actually takes
    # effect for code that imports it lazily inside render().
    for mod_name in list(sys.modules):
        if mod_name == "matplotlib" or mod_name.startswith("matplotlib."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    monkeypatch.setitem(sys.modules, "matplotlib.colors", None)

    module = importlib.import_module(f"app.views.{view_name}")
    importlib.reload(module)
    try:
        module.render()
    except Exception as exc:  # noqa: BLE001 - intentional broad catch to report which view broke
        pytest.fail(f"app.views.{view_name}.render() raised with matplotlib blocked: {exc!r}")


# ---------------------------------------------------------------------------
# Cold-start self-seeding (app.common.ensure_results)
# ---------------------------------------------------------------------------

def test_ensure_results_seeds_empty_dir(tmp_path, monkeypatch):
    """First call against an empty tmp_path must seed a result JSON for
    every bundled transcript id, plus benchmark.md."""
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(tmp_path))
    import app.common as app_common
    import scribegate.cli as cli

    start = time.monotonic()
    seeded = app_common.ensure_results(results_dir=str(tmp_path))
    elapsed = time.monotonic() - start
    print(f"ensure_results cold-start seed took {elapsed:.2f}s")

    assert seeded is True

    expected_ids = cli.discover_transcript_ids()
    for tid in expected_ids:
        assert os.path.exists(os.path.join(str(tmp_path), f"{tid}.json")), (
            f"missing seeded result for {tid}"
        )

    benchmark_path = os.path.join(str(tmp_path), "benchmark.md")
    assert os.path.exists(benchmark_path), "benchmark.md must be written by ensure_results"


def test_ensure_results_is_noop_when_already_seeded(tmp_path, monkeypatch):
    """Calling ensure_results again against an already-fully-seeded dir must
    not touch any file (mtimes unchanged) and must return False."""
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(tmp_path))
    import app.common as app_common

    first = app_common.ensure_results(results_dir=str(tmp_path))
    assert first is True

    files_before = sorted(glob.glob(os.path.join(str(tmp_path), "*")))
    mtimes_before = {p: os.stat(p).st_mtime_ns for p in files_before}

    second = app_common.ensure_results(results_dir=str(tmp_path))
    assert second is False

    files_after = sorted(glob.glob(os.path.join(str(tmp_path), "*")))
    mtimes_after = {p: os.stat(p).st_mtime_ns for p in files_after}

    assert files_before == files_after
    assert mtimes_before == mtimes_after


def test_ensure_results_repeat_call_files_still_valid_json(tmp_path, monkeypatch):
    """Repeat call against an already-populated dir doesn't error, doesn't
    change file count, and every per-transcript result JSON still parses."""
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(tmp_path))
    import app.common as app_common
    import scribegate.cli as cli

    app_common.ensure_results(results_dir=str(tmp_path))
    count_before = len(glob.glob(os.path.join(str(tmp_path), "*.json")))

    app_common.ensure_results(results_dir=str(tmp_path))
    count_after = len(glob.glob(os.path.join(str(tmp_path), "*.json")))

    assert count_before == count_after

    expected_ids = cli.discover_transcript_ids()
    for tid in expected_ids:
        with open(os.path.join(str(tmp_path), f"{tid}.json"), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["transcript_id"] == tid


def test_app_cold_start_self_seeds_via_apptest(tmp_path, monkeypatch):
    """End-to-end regression: on a fresh/empty results dir (simulating a
    Streamlit Cloud deploy where data/results/*.json is gitignored), just
    running the app script must self-seed results in-process, without
    raising, via AppTest — the same harness used by
    test_app_script_executes_without_exception."""
    monkeypatch.setenv("SCRIBEGATE_RESULTS_DIR", str(tmp_path))
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(REPO_ROOT, "app", "streamlit_app.py"), default_timeout=30).run()
    assert not at.exception, f"app script raised: {at.exception}"

    import scribegate.cli as cli

    expected_ids = cli.discover_transcript_ids()
    seeded_ids = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(str(tmp_path), "*.json"))
    )
    for tid in expected_ids:
        assert tid in seeded_ids, f"AppTest cold start did not seed {tid}"


# ---------------------------------------------------------------------------
# W4 UI overhaul — copy-contract lock + live-mode no-key preview path
# ---------------------------------------------------------------------------

def test_ui_copy_every_page_has_required_fields():
    """Copy-contract lock: every page key in specs/ui_copy.yaml's `pages`
    block must carry plain_title, one_liner, and why_it_matters — the three
    fields app.common.page_header() renders for every view. A page missing
    one of these would silently fall back to a blank/placeholder header."""
    with open(UI_COPY_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    pages = data.get("pages") or {}
    assert pages, "specs/ui_copy.yaml must define at least one page"

    required_fields = ("plain_title", "one_liner", "why_it_matters")
    for page_key, page in pages.items():
        for field in required_fields:
            value = (page or {}).get(field)
            assert isinstance(value, str) and value.strip(), (
                f"specs/ui_copy.yaml pages.{page_key}.{field} must be a non-empty string"
            )


def test_live_mode_fallback_banner_generalizes_both_directions(monkeypatch):
    """The fallback banner text must read correctly regardless of which
    provider was primary — it must never hardcode "fell back to Anthropic"
    (or DeepSeek) as the only direction. Exercises app.views.live_mode's
    real `_render_fallback_events`, capturing what it passes to `st.info`
    (Streamlit widgets are safely callable outside a live script run, as
    the existing render-sweep test above already relies on)."""
    pytest.importorskip("streamlit")
    import app.views.live_mode as live_mode

    captured = []
    monkeypatch.setattr(live_mode.st, "info", lambda msg: captured.append(msg))

    # DeepSeek (primary, default) fails; Anthropic (fallback) serves.
    result_deepseek_to_anthropic = {
        "fallback_events": [
            {"stage": "judge_sample_0", "from_provider": "deepseek", "to_provider": "anthropic", "reason_class": "rate_limit"}
        ]
    }
    live_mode._render_fallback_events(result_deepseek_to_anthropic)
    assert captured == ["Judging fell back to Anthropic (deepseek: rate_limit)."]

    # Mirror direction: Anthropic (primary) fails; DeepSeek (fallback) serves.
    captured.clear()
    result_anthropic_to_deepseek = {
        "fallback_events": [
            {"stage": "draft", "from_provider": "anthropic", "to_provider": "deepseek", "reason_class": "auth_error"}
        ]
    }
    live_mode._render_fallback_events(result_anthropic_to_deepseek)
    assert captured == ["Drafting fell back to DeepSeek (anthropic: auth_error)."]


def test_live_mode_provider_chain_badges_reflect_configured_order(monkeypatch):
    """`_render_provider_chain_status` must label badges from the actual
    configured chain order (`live.provider_status(config)["order"]`), not a
    hardcoded Anthropic-first assumption. With DeepSeek primary configured
    (has a key) and Anthropic as fallback with no key, the primary badge
    must read "DeepSeek" and the fallback badge must say Anthropic is
    skipped for lacking a key."""
    pytest.importorskip("streamlit")
    import app.views.live_mode as live_mode
    from scribegate import live

    captured = []
    monkeypatch.setattr(live_mode.st, "badge", lambda label, color=None: captured.append((label, color)))
    monkeypatch.setattr(live_mode.st, "columns", lambda n: [_NullColumn() for _ in range(n)])

    config = live.LiveConfig(api_key=None, deepseek_api_key="dk", primary_provider="deepseek")
    monkeypatch.setattr(live, "_openai_importable", lambda: True)

    live_mode._render_provider_chain_status(config)

    labels = [label for label, _color in captured]
    assert labels[0] == "Primary: DeepSeek"
    assert labels[1] == "Fallback: Anthropic (no key — skipped)"
    assert labels[2] == "Final: Mock preview"


class _NullColumn:
    """Minimal `with st.columns(n)[i]:` stand-in for tests that don't need
    real Streamlit layout, just the ability to enter/exit a `with` block."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_live_mode_renders_in_no_key_preview_mode(monkeypatch):
    """CI path: with no ANTHROPIC_API_KEY configured (the default in CI),
    app.views.live_mode.render() must fall into the "unavailable" branch
    and render the bundled SAMPLE saved run without raising — never require
    a real API key just to import/render the page."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("SCRIBEGATE_DEMO_PASSCODE", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    import app.views.live_mode as live_mode

    importlib.reload(live_mode)
    live_mode.render()  # must not raise
