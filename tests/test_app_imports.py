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

import importlib
import json
import os
import sys

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
