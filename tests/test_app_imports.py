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

pytest.importorskip("streamlit")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_SCRIPT_PATH = os.path.join(REPO_ROOT, "DEMO_SCRIPT.md")
GLAUCOMA_05_RESULT_PATH = os.path.join(REPO_ROOT, "data", "results", "glaucoma_05.json")

APP_MODULES = [
    "app.common",
    "app.views.about",
    "app.views.analytics",
    "app.views.drift",
    "app.views.live_encounter",
    "app.views.overview",
    "app.views.provenance",
    "app.views.review_queue",
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
    "review_queue",
    "provenance",
    "live_encounter",
    "about",
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
