"""Tests for scribegate.live (W1) — live-mode orchestrator, and
scribegate.costs — pricing + cost ledger.

Everything here stays keyless/deterministic/offline: no test ever imports or
requires the real `anthropic` package. Every "API call" is a monkeypatched
`scribegate.live._client_factory` returning a fake client with canned
responses, per the module's documented test hook.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scribegate import costs, live

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = REPO_ROOT / "data" / "transcripts"

GLAUCOMA_05 = "glaucoma_05"


# ---------------------------------------------------------------------------
# Fake Anthropic client
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, text: str, input_tokens: int = 100, output_tokens: int = 50):
        self.content = [SimpleNamespace(type="text", text=text)]
        self.usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)


class _FakeMessages:
    def __init__(self, responses: list[_FakeResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, model, max_tokens, messages):
        self.calls.append({"model": model, "max_tokens": max_tokens, "messages": messages})
        if not self._responses:
            raise AssertionError("FakeMessages: ran out of canned responses")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse]):
        self.messages = _FakeMessages(responses)


def _draft_response_json() -> str:
    return json.dumps(
        {
            "S": [
                {
                    "text": "Patient reports adherence is imperfect, especially the midday brimonidine dose.",
                    "quotes": ["with three different drops on different schedules I know I miss some"],
                }
            ],
            "O": [
                {
                    "text": "IOP 26 mmHg OD (up from 21), 24 mmHg OS. Goldmann.",
                    "quotes": ["Goldmann pressures today, right eye twenty-six, left eye twenty-four"],
                }
            ],
            "A": [
                {
                    "text": "Progressing POAG OD despite maximal medical therapy.",
                    "quotes": ["the right eye is progressing structurally and functionally"],
                }
            ],
            "P": [
                {
                    "text": "Refer to glaucoma surgeon for trabeculectomy vs. drainage device evaluation.",
                    "quotes": ["refer you to a glaucoma surgeon to discuss surgical options"],
                }
            ],
        }
    )


def _judge_response_json(agg_scores=(4, 5, 4, 5)) -> str:
    dims = ["completeness", "hallucination", "coding_plausibility", "terminology"]
    return json.dumps(
        {
            "scores": dict(zip(dims, agg_scores)),
            "rationales": {d: f"canned rationale for {d}" for d in dims},
        }
    )


def _install_fake_client(monkeypatch, responses):
    fake_client = _FakeClient(responses)
    monkeypatch.setattr(live, "_client_factory", lambda api_key: fake_client)
    return fake_client


def _make_config(tmp_path, **overrides) -> live.LiveConfig:
    defaults = dict(
        api_key="sk-ant-TESTKEY-should-never-be-saved",
        demo_passcode="letmein",
        daily_budget_usd=5.0,
        drafter_model="claude-haiku-4-5",
        judge_model="claude-haiku-4-5",
        judge_samples=3,
    )
    defaults.update(overrides)
    return live.LiveConfig(**defaults)


# ---------------------------------------------------------------------------
# check_passcode
# ---------------------------------------------------------------------------

def test_check_passcode_matches():
    config = live.LiveConfig(demo_passcode="s3cret")
    assert live.check_passcode("s3cret", config) is True


def test_check_passcode_mismatch():
    config = live.LiveConfig(demo_passcode="s3cret")
    assert live.check_passcode("wrong", config) is False


def test_check_passcode_no_configured_passcode_returns_false():
    config = live.LiveConfig(demo_passcode=None)
    assert live.check_passcode("anything", config) is False
    assert live.check_passcode(None, config) is False


# ---------------------------------------------------------------------------
# LiveConfig.from_env
# ---------------------------------------------------------------------------

def test_live_config_from_env_reads_env_vars(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-value")
    monkeypatch.setenv("SCRIBEGATE_DEMO_PASSCODE", "env-passcode")
    monkeypatch.setenv("SCRIBEGATE_DAILY_BUDGET_USD", "2.50")
    monkeypatch.setenv("SCRIBEGATE_JUDGE_SAMPLES", "5")

    config = live.LiveConfig.from_env()

    assert config.api_key == "env-key-value"
    assert config.demo_passcode == "env-passcode"
    assert config.daily_budget_usd == 2.50
    assert config.judge_samples == 5
    assert config.drafter_model == "claude-haiku-4-5"
    assert config.judge_model == "claude-haiku-4-5"


def test_live_config_from_env_malformed_numeric_overrides_fall_back(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("SCRIBEGATE_DAILY_BUDGET_USD", "not-a-number")
    monkeypatch.setenv("SCRIBEGATE_JUDGE_SAMPLES", "not-an-int")

    config = live.LiveConfig.from_env()

    assert config.daily_budget_usd == 5.0
    assert config.judge_samples == 3


def test_live_config_repr_never_leaks_key():
    config = live.LiveConfig(api_key="sk-ant-super-secret-value", demo_passcode="p4ss")
    rendered = repr(config)
    assert "sk-ant-super-secret-value" not in rendered
    assert "p4ss" not in rendered
    assert "<set>" in rendered


# ---------------------------------------------------------------------------
# costs.py: pricing
# ---------------------------------------------------------------------------

def test_pricing_yaml_loads_expected_models():
    pricing = costs.load_pricing()
    assert pricing["claude-haiku-4-5"] == {"input": 1.00, "output": 5.00}
    assert pricing["claude-sonnet-4-5"] == {"input": 3.00, "output": 15.00}
    assert pricing["claude-opus-4-8"] == {"input": 15.00, "output": 75.00}


def test_pricing_fallback_on_missing_file(tmp_path):
    missing = tmp_path / "nope.yaml"
    pricing = costs.load_pricing(path=missing)
    assert pricing["claude-haiku-4-5"] == {"input": 1.00, "output": 5.00}


def test_pricing_fallback_on_malformed_file(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: [valid, mapping, shape\n")
    pricing = costs.load_pricing(path=bad)
    assert "claude-haiku-4-5" in pricing


def test_cost_of_computes_expected_usd():
    usage = {"model": "claude-haiku-4-5", "input_tokens": 1_000_000, "output_tokens": 1_000_000}
    usd = costs.cost_of(usage)
    assert usd == pytest.approx(1.00 + 5.00)


def test_cost_of_unknown_model_uses_conservative_fallback():
    usage = {"model": "some-future-model", "input_tokens": 1_000_000, "output_tokens": 1_000_000}
    usd = costs.cost_of(usage)
    # Falls back to the most expensive known rate rather than zero.
    assert usd == pytest.approx(15.00 + 75.00)


# ---------------------------------------------------------------------------
# costs.py: ledger + budget math
# ---------------------------------------------------------------------------

def test_record_usage_appends_ledger_and_today_spend_sums_same_day(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    costs.record_usage("draft", "claude-haiku-4-5", 1000, 500, ledger_path=ledger)
    costs.record_usage("judge_sample_0", "claude-haiku-4-5", 2000, 1000, ledger_path=ledger)

    lines = ledger.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        rec = json.loads(line)
        assert "sk-ant" not in json.dumps(rec)

    spend = costs.today_spend(ledger_path=ledger)
    expected = costs.cost_of({"model": "claude-haiku-4-5", "input_tokens": 1000, "output_tokens": 500})
    expected += costs.cost_of({"model": "claude-haiku-4-5", "input_tokens": 2000, "output_tokens": 1000})
    assert spend == pytest.approx(expected)


def test_today_spend_ledger_rollover_by_day(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    yesterday_rec = {
        "ts": "2020-01-01T00:00:00.000Z", "day": "2020-01-01", "stage": "draft",
        "model": "claude-haiku-4-5", "input_tokens": 1_000_000, "output_tokens": 1_000_000,
        "usd": 6.0,
    }
    today_rec = {
        "ts": "2020-01-02T00:00:00.000Z", "day": "2020-01-02", "stage": "draft",
        "model": "claude-haiku-4-5", "input_tokens": 0, "output_tokens": 0, "usd": 1.23,
    }
    with open(ledger, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(yesterday_rec) + "\n")
        fh.write(json.dumps(today_rec) + "\n")

    spend_today = costs.today_spend(ledger_path=ledger, today="2020-01-02")
    assert spend_today == pytest.approx(1.23)

    spend_yesterday = costs.today_spend(ledger_path=ledger, today="2020-01-01")
    assert spend_yesterday == pytest.approx(6.0)


def test_today_spend_skips_non_dict_ledger_lines(tmp_path):
    """A ledger line that's valid JSON but not a dict (bare int, null,
    list) must be skipped silently, not crash today_spend with an
    AttributeError from calling .get() on a non-dict."""
    ledger = tmp_path / "ledger.jsonl"
    today = "2020-06-01"
    good_rec = {
        "ts": "2020-06-01T00:00:00.000Z", "day": today, "stage": "draft",
        "model": "claude-haiku-4-5", "input_tokens": 1_000_000, "output_tokens": 0,
        "usd": 1.0,
    }
    with open(ledger, "w", encoding="utf-8") as fh:
        fh.write("12345\n")            # bare int
        fh.write("null\n")             # JSON null
        fh.write("[]\n")               # empty list
        fh.write('"just a string"\n')  # bare string
        fh.write(json.dumps(good_rec) + "\n")

    spend = costs.today_spend(ledger_path=ledger, today=today)
    assert spend == pytest.approx(1.0)


def test_today_spend_skips_dicts_missing_required_keys(tmp_path):
    """A dict-shaped ledger line missing required keys (e.g. no 'day' or
    no 'usd') is corrupt/non-schema and must be skipped silently rather
    than counted or crashing."""
    ledger = tmp_path / "ledger.jsonl"
    today = "2020-06-01"
    missing_day = {"stage": "draft", "model": "claude-haiku-4-5", "usd": 99.0}
    missing_usd = {"day": today, "stage": "draft", "model": "claude-haiku-4-5"}
    good_rec = {
        "ts": "2020-06-01T00:00:00.000Z", "day": today, "stage": "draft",
        "model": "claude-haiku-4-5", "input_tokens": 0, "output_tokens": 0,
        "usd": 2.5,
    }
    with open(ledger, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(missing_day) + "\n")
        fh.write(json.dumps(missing_usd) + "\n")
        fh.write(json.dumps(good_rec) + "\n")

    spend = costs.today_spend(ledger_path=ledger, today=today)
    assert spend == pytest.approx(2.5)


def test_budget_remaining_correct_with_corrupt_ledger_lines(tmp_path):
    """budget_remaining must also survive a ledger with corrupt lines mixed
    in with valid ones, deriving the correct remaining budget from only the
    valid lines."""
    ledger = tmp_path / "ledger.jsonl"
    config = live.LiveConfig(daily_budget_usd=5.0)

    with open(ledger, "w", encoding="utf-8") as fh:
        fh.write("null\n")
        fh.write("[1, 2, 3]\n")
        fh.write(json.dumps({"stage": "draft", "usd": 1.0}) + "\n")  # missing "day"

    costs.record_usage("draft", "claude-haiku-4-5", 1_000_000, 0, ledger_path=ledger)  # $1.00, valid

    assert costs.budget_remaining(config, ledger_path=ledger) == pytest.approx(4.0)


def test_budget_remaining_math(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    config = live.LiveConfig(daily_budget_usd=5.0)
    assert costs.budget_remaining(config, ledger_path=ledger) == pytest.approx(5.0)

    costs.record_usage("draft", "claude-haiku-4-5", 1_000_000, 0, ledger_path=ledger)  # $1.00
    assert costs.budget_remaining(config, ledger_path=ledger) == pytest.approx(4.0)


def test_per_note_breakdown_shapes():
    run = {
        "cost_records": [
            {"stage": "draft", "model": "m", "input_tokens": 100, "output_tokens": 50, "usd": 1.0},
            {"stage": "judge_sample_0", "model": "m", "input_tokens": 10, "output_tokens": 5, "usd": 0.5},
            {"stage": "judge_sample_1", "model": "m", "input_tokens": 10, "output_tokens": 5, "usd": 0.5},
        ]
    }
    breakdown = costs.per_note_breakdown(run)
    assert breakdown["drafting"]["usd"] == pytest.approx(1.0)
    assert breakdown["judging"]["usd"] == pytest.approx(1.0)
    assert breakdown["total_usd"] == pytest.approx(2.0)
    assert breakdown["judging"]["input_tokens"] == 20


# ---------------------------------------------------------------------------
# live_available
# ---------------------------------------------------------------------------

def test_live_available_false_no_key(tmp_path):
    config = live.LiveConfig(api_key=None, daily_budget_usd=5.0)
    available, reason = live.live_available(config, ledger_path=tmp_path / "ledger.jsonl")
    assert available is False
    assert "ANTHROPIC_API_KEY" in reason


def test_live_available_false_budget_exhausted(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    config = live.LiveConfig(api_key="k", daily_budget_usd=1.0)
    costs.record_usage("draft", "claude-opus-4-8", 1_000_000, 1_000_000, ledger_path=ledger)  # $90
    available, reason = live.live_available(config, ledger_path=ledger)
    assert available is False
    assert "budget" in reason.lower()


def test_live_available_true_when_key_and_budget_ok(tmp_path):
    config = live.LiveConfig(api_key="k", daily_budget_usd=5.0)
    available, reason = live.live_available(config, ledger_path=tmp_path / "ledger.jsonl")
    assert available is True
    assert reason


# ---------------------------------------------------------------------------
# GuardedClient
# ---------------------------------------------------------------------------

def test_guarded_client_blocks_when_over_budget(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    config = live.LiveConfig(api_key="k", daily_budget_usd=1.0)
    costs.record_usage("draft", "claude-opus-4-8", 1_000_000, 1_000_000, ledger_path=ledger)  # $90 > $1 budget

    def _explode(api_key):
        raise AssertionError("client factory must not be called when budget is already exceeded")

    monkeypatch.setattr(live, "_client_factory", _explode)

    guarded = live.GuardedClient(config, ledger_path=ledger)
    with pytest.raises(live.BudgetExceededError):
        guarded.create(stage="draft", model="claude-haiku-4-5", max_tokens=10, messages=[])


def test_guarded_client_records_usage_after_call(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    config = live.LiveConfig(api_key="k", daily_budget_usd=5.0)
    _install_fake_client(monkeypatch, [_FakeResponse("hello", input_tokens=123, output_tokens=45)])

    guarded = live.GuardedClient(config, ledger_path=ledger)
    result = guarded.create(stage="draft", model="claude-haiku-4-5", max_tokens=10, messages=[])

    assert result["cost_record"]["input_tokens"] == 123
    assert result["cost_record"]["output_tokens"] == 45
    assert len(guarded.cost_records) == 1
    assert ledger.exists()
    spend = costs.today_spend(ledger_path=ledger)
    assert spend > 0


# ---------------------------------------------------------------------------
# run_live_note — happy path
# ---------------------------------------------------------------------------

def test_run_live_note_happy_path_shape_and_spans(monkeypatch, tmp_path):
    responses = [
        _FakeResponse(_draft_response_json(), input_tokens=500, output_tokens=200),
        _FakeResponse(_judge_response_json(), input_tokens=300, output_tokens=100),
        _FakeResponse(_judge_response_json(), input_tokens=300, output_tokens=100),
        _FakeResponse(_judge_response_json(), input_tokens=300, output_tokens=100),
    ]
    fake_client = _install_fake_client(monkeypatch, responses)
    config = _make_config(tmp_path, judge_samples=3)
    ledger = tmp_path / "ledger.jsonl"
    live_runs_dir = tmp_path / "live_runs"

    result = live.run_live_note(
        GLAUCOMA_05, config=config, ledger_path=ledger, live_runs_dir=live_runs_dir
    )

    assert result["budget_exhausted"] is False
    assert result["partial"] is False
    note = result["generated_note"]
    assert note["transcript_id"] == GLAUCOMA_05
    assert note["generator"] == "api"
    for section in ("S", "O", "A", "P"):
        assert section in note["soap"]
        for line in note["soap"][section]:
            assert "text" in line and "spans" in line and "span_confidence" in line
            assert line["span_confidence"] in ("high", "low")

    # At least one line should have a real, high-confidence span attached
    # (the canned quotes are lifted near-verbatim from the transcript).
    all_lines = [ln for sec in note["soap"].values() for ln in sec]
    assert any(ln["spans"] for ln in all_lines)
    assert any(ln["span_confidence"] == "high" for ln in all_lines)

    # 1 draft call + 3 judge calls
    assert len(fake_client.messages.calls) == 4
    assert result["judge_samples_collected"] == 3
    assert len(result["judge_sampled"]["samples"]) == 3

    assert result["route"] in ("auto_accept", "review", "regenerate")
    assert "ci95" in result["judge_sampled"]

    # Cost record written for every call.
    assert len(result["cost_records"]) == 4
    assert result["cost_breakdown"]["total_usd"] > 0
    assert ledger.exists()
    ledger_lines = ledger.read_text().strip().splitlines()
    assert len(ledger_lines) == 4

    # Saved artifact exists under live_runs_dir.
    saved_files = list(live_runs_dir.glob(f"*_{GLAUCOMA_05}.json"))
    assert len(saved_files) == 1
    with open(saved_files[0], "r", encoding="utf-8") as fh:
        saved = json.load(fh)
    assert saved["transcript_id"] == GLAUCOMA_05


def test_run_live_note_unknown_transcript_id_returns_error(monkeypatch, tmp_path):
    _install_fake_client(monkeypatch, [])
    config = _make_config(tmp_path)
    result = live.run_live_note(
        "not_a_real_transcript_id",
        config=config,
        ledger_path=tmp_path / "ledger.jsonl",
        live_runs_dir=tmp_path / "live_runs",
    )
    assert result["generated_note"] is None
    assert "error" in result
    assert result["partial"] is True


# ---------------------------------------------------------------------------
# run_live_note — budget exhausted mid-run
# ---------------------------------------------------------------------------

def test_run_live_note_budget_exhausted_before_draft(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    config = _make_config(tmp_path, daily_budget_usd=1.0)
    # Pre-exhaust the budget before the run even starts.
    costs.record_usage("draft", "claude-opus-4-8", 1_000_000, 1_000_000, ledger_path=ledger)

    def _explode(api_key):
        raise AssertionError("must not construct a client once budget is exhausted")

    monkeypatch.setattr(live, "_client_factory", _explode)

    result = live.run_live_note(
        GLAUCOMA_05, config=config, ledger_path=ledger, live_runs_dir=tmp_path / "live_runs"
    )

    assert result["budget_exhausted"] is True
    assert result["partial"] is True
    assert result["generated_note"] is None


def test_run_live_note_budget_exhausted_mid_run_partial(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    live_runs_dir = tmp_path / "live_runs"

    # Draft call is cheap; each judge call is deliberately made to look
    # expensive (huge token counts) so the budget runs out partway through
    # the judge-sampling loop, not before the first call.
    responses = [
        _FakeResponse(_draft_response_json(), input_tokens=10, output_tokens=10),
        _FakeResponse(_judge_response_json(), input_tokens=2_000_000, output_tokens=2_000_000),
        _FakeResponse(_judge_response_json(), input_tokens=2_000_000, output_tokens=2_000_000),
        _FakeResponse(_judge_response_json(), input_tokens=2_000_000, output_tokens=2_000_000),
    ]
    _install_fake_client(monkeypatch, responses)
    # claude-haiku-4-5: $1/$5 per MTok -> one judge call of 2M in + 2M out =
    # $2 + $10 = $12. The guard checks remaining budget BEFORE each call
    # (not "would this call afford itself"), so with a $12 budget: the tiny
    # draft call leaves ~$12 remaining (still > 0) -> judge sample #0 is
    # allowed and consumes all $12, driving remaining to ~$0 (<= 0) -> judge
    # sample #1 is blocked pre-flight. Net: exactly 1 judge sample collected.
    config = _make_config(tmp_path, daily_budget_usd=12.0, judge_samples=3)

    result = live.run_live_note(
        GLAUCOMA_05, config=config, ledger_path=ledger, live_runs_dir=live_runs_dir
    )

    assert result["budget_exhausted"] is True
    assert result["partial"] is True
    assert result["judge_samples_collected"] == 1
    assert result["judge_samples_collected"] < result["judge_samples_requested"]
    # The note itself was still drafted (draft call succeeded before the cap hit).
    assert result["generated_note"] is not None


def test_run_live_note_no_judge_samples_routes_conservatively(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    # Draft call itself is already over-budget-inducing: budget runs out
    # exactly after the draft call, before any judge call can proceed.
    responses = [
        _FakeResponse(_draft_response_json(), input_tokens=2_000_000, output_tokens=2_000_000),
    ]
    _install_fake_client(monkeypatch, responses)
    config = _make_config(tmp_path, daily_budget_usd=12.0, judge_samples=3)

    result = live.run_live_note(
        GLAUCOMA_05, config=config, ledger_path=ledger, live_runs_dir=tmp_path / "live_runs"
    )

    assert result["judge_samples_collected"] == 0
    assert result["budget_exhausted"] is True
    assert result["route"] in ("auto_accept", "review", "regenerate")
    assert "NO_JUDGE_SAMPLES" in result["judge_sampled"]["flags"]


# ---------------------------------------------------------------------------
# Safety: the API key must never appear in any saved artifact or ledger.
# ---------------------------------------------------------------------------

def test_api_key_never_in_saved_artifacts_or_ledger(monkeypatch, tmp_path):
    secret_key = "sk-ant-THIS-MUST-NEVER-BE-PERSISTED-abc123"
    responses = [
        _FakeResponse(_draft_response_json(), input_tokens=100, output_tokens=50),
        _FakeResponse(_judge_response_json(), input_tokens=50, output_tokens=20),
    ]
    _install_fake_client(monkeypatch, responses)
    config = _make_config(tmp_path, api_key=secret_key, judge_samples=1)
    ledger = tmp_path / "ledger.jsonl"
    live_runs_dir = tmp_path / "live_runs"

    result = live.run_live_note(
        GLAUCOMA_05, config=config, ledger_path=ledger, live_runs_dir=live_runs_dir
    )

    result_str = json.dumps(result)
    assert secret_key not in result_str

    ledger_text = ledger.read_text()
    assert secret_key not in ledger_text

    for saved_path in live_runs_dir.glob("*.json"):
        assert secret_key not in saved_path.read_text()


def test_live_config_from_env_never_leaks_key_via_str(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-super-secret-env-value")
    config = live.LiveConfig.from_env()
    assert "sk-ant-super-secret-env-value" not in str(config)
    assert "sk-ant-super-secret-env-value" not in repr(config)


# ---------------------------------------------------------------------------
# DeepSeek fallback provider — fake OpenAI-compatible client + test-double
# provider SDK exceptions.
#
# These exception classes are plain `Exception` subclasses named to match
# real anthropic/openai SDK exception class names (AuthenticationError,
# RateLimitError, ...) — `scribegate.live._classify_exception` matches on
# TYPE NAME only (never message/body), so these test doubles exercise the
# exact same classification path a real SDK exception would, without this
# suite ever needing the real `anthropic`/`openai` exception hierarchies.
# ---------------------------------------------------------------------------

class AuthenticationError(Exception):
    status_code = 401


class RateLimitError(Exception):
    status_code = 429


class InternalServerError(Exception):
    status_code = 500


class APITimeoutError(Exception):
    pass


class APIConnectionError(Exception):
    pass


class _FakeDeepSeekMessage:
    def __init__(self, content):
        self.content = content


class _FakeDeepSeekChoice:
    def __init__(self, content):
        self.message = _FakeDeepSeekMessage(content)


class _FakeDeepSeekUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeDeepSeekResponse:
    def __init__(self, text: str, prompt_tokens: int = 80, completion_tokens: int = 40):
        self.choices = [_FakeDeepSeekChoice(text)]
        self.usage = _FakeDeepSeekUsage(prompt_tokens, completion_tokens)


class _FakeDeepSeekCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, model, max_tokens, messages):
        self.calls.append({"model": model, "max_tokens": max_tokens, "messages": messages})
        if not self._responses:
            raise AssertionError("FakeDeepSeekCompletions: ran out of canned responses")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeDeepSeekChat:
    def __init__(self, responses):
        self.completions = _FakeDeepSeekCompletions(responses)


class _FakeDeepSeekClient:
    def __init__(self, responses):
        self.chat = _FakeDeepSeekChat(responses)


def _install_fake_deepseek_client(monkeypatch, responses):
    """Installs a fake DeepSeek client AND pretends the optional `openai`
    package is importable (it genuinely isn't in this environment — see
    `test_deepseek_provider_unavailable_without_openai_package` below, which
    exercises the real absence — but these failover tests need
    `DeepSeekProvider.available()` to be True so the chain actually reaches
    it)."""
    fake_client = _FakeDeepSeekClient(responses)
    monkeypatch.setattr(live, "_deepseek_client_factory", lambda api_key: fake_client)
    monkeypatch.setattr(live, "_openai_importable", lambda: True)
    return fake_client


def _both_keys_config(tmp_path, **overrides) -> live.LiveConfig:
    defaults = dict(deepseek_api_key="sk-deepseek-TESTKEY-should-never-be-saved", deepseek_model="deepseek-chat")
    defaults.update(overrides)
    return _make_config(tmp_path, **defaults)


# ---------------------------------------------------------------------------
# FailoverClient — failover triggers on transient provider errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "exc_factory,expected_reason_class",
    [
        (lambda: AuthenticationError("unauthorized"), "auth_error"),
        (lambda: RateLimitError("too many requests"), "rate_limit"),
        (lambda: InternalServerError("server exploded"), "server_error"),
        (lambda: APITimeoutError("timed out"), "timeout"),
        (lambda: APIConnectionError("connection reset"), "connection_error"),
    ],
)
def test_failover_client_falls_over_to_deepseek_on_transient_anthropic_errors(
    monkeypatch, tmp_path, exc_factory, expected_reason_class
):
    ledger = tmp_path / "ledger.jsonl"
    config = _both_keys_config(tmp_path)

    _install_fake_client(monkeypatch, [exc_factory()])
    _install_fake_deepseek_client(monkeypatch, [_FakeDeepSeekResponse("deepseek drafted text")])

    client = live.FailoverClient(
        config,
        [live.AnthropicProvider(config.api_key), live.DeepSeekProvider(config.deepseek_api_key, config.deepseek_model)],
        ledger_path=ledger,
    )
    result = client.create(
        stage="draft", model="claude-haiku-4-5", max_tokens=100, messages=[{"role": "user", "content": "hi"}]
    )

    assert result["text"] == "deepseek drafted text"
    assert result["provider"] == "deepseek"
    assert len(client.fallback_events) == 1
    event = client.fallback_events[0]
    assert event == {
        "stage": "draft",
        "from_provider": "anthropic",
        "to_provider": "deepseek",
        "reason_class": expected_reason_class,
    }
    # The cost record for the call that actually succeeded is tagged with
    # the provider that served it — used for correct per-provider pricing.
    assert result["cost_record"]["provider"] == "deepseek"
    assert result["cost_record"]["model"] == "deepseek-chat"


def test_failover_client_no_failover_when_budget_exhausted_before_call(monkeypatch, tmp_path):
    """Budget guard sits OUTSIDE the provider chain: when the budget is
    already exhausted, BudgetExceededError is raised before any provider is
    even touched — no fallback event, no client construction at all, for
    either provider."""
    ledger = tmp_path / "ledger.jsonl"
    config = _both_keys_config(tmp_path, daily_budget_usd=1.0)
    costs.record_usage("draft", "claude-opus-4-8", 1_000_000, 1_000_000, ledger_path=ledger)  # $90 > $1 budget

    def _explode_anthropic(api_key):
        raise AssertionError("anthropic client factory must not be called when budget is already exceeded")

    def _explode_deepseek(api_key):
        raise AssertionError("deepseek client factory must not be called when budget is already exceeded")

    monkeypatch.setattr(live, "_client_factory", _explode_anthropic)
    monkeypatch.setattr(live, "_deepseek_client_factory", _explode_deepseek)
    monkeypatch.setattr(live, "_openai_importable", lambda: True)

    client = live.FailoverClient(
        config,
        [live.AnthropicProvider(config.api_key), live.DeepSeekProvider(config.deepseek_api_key, config.deepseek_model)],
        ledger_path=ledger,
    )
    with pytest.raises(live.BudgetExceededError):
        client.create(stage="draft", model="claude-haiku-4-5", max_tokens=10, messages=[{"role": "user", "content": "hi"}])

    assert client.fallback_events == []


def test_failover_client_fallback_event_contains_no_key_material(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    anthropic_secret = "sk-ant-THIS-MUST-NEVER-LEAK-abc123"
    deepseek_secret = "sk-deepseek-THIS-MUST-NEVER-LEAK-xyz789"
    config = _both_keys_config(tmp_path, api_key=anthropic_secret, deepseek_api_key=deepseek_secret)

    _install_fake_client(monkeypatch, [AuthenticationError("unauthorized: key " + anthropic_secret)])
    _install_fake_deepseek_client(monkeypatch, [_FakeDeepSeekResponse("ok")])

    client = live.FailoverClient(
        config,
        [live.AnthropicProvider(config.api_key), live.DeepSeekProvider(config.deepseek_api_key, config.deepseek_model)],
        ledger_path=ledger,
    )
    client.create(stage="draft", model="claude-haiku-4-5", max_tokens=10, messages=[{"role": "user", "content": "hi"}])

    events_str = json.dumps(client.fallback_events)
    assert anthropic_secret not in events_str
    assert deepseek_secret not in events_str
    assert "unauthorized" not in events_str  # no raw error body/message either
    assert set(client.fallback_events[0].keys()) == {"stage", "from_provider", "to_provider", "reason_class"}

    ledger_text = ledger.read_text()
    assert anthropic_secret not in ledger_text
    assert deepseek_secret not in ledger_text


def test_failover_client_no_provider_available_when_neither_key_configured(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    config = live.LiveConfig(api_key=None, deepseek_api_key=None, daily_budget_usd=5.0)
    client = live.FailoverClient(
        config,
        [live.AnthropicProvider(config.api_key), live.DeepSeekProvider(config.deepseek_api_key, config.deepseek_model)],
        ledger_path=ledger,
    )
    with pytest.raises(live.NoProviderAvailableError):
        client.create(stage="draft", model="claude-haiku-4-5", max_tokens=10, messages=[{"role": "user", "content": "hi"}])


def test_failover_client_all_providers_fail_raises_all_providers_failed(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    config = _both_keys_config(tmp_path)
    _install_fake_client(monkeypatch, [AuthenticationError("nope")])
    _install_fake_deepseek_client(monkeypatch, [RateLimitError("also nope")])

    client = live.FailoverClient(
        config,
        [live.AnthropicProvider(config.api_key), live.DeepSeekProvider(config.deepseek_api_key, config.deepseek_model)],
        ledger_path=ledger,
    )
    with pytest.raises(live.AllProvidersFailedError):
        client.create(stage="draft", model="claude-haiku-4-5", max_tokens=10, messages=[{"role": "user", "content": "hi"}])
    # A fallback event was still logged for the anthropic -> deepseek hop,
    # even though deepseek itself then failed too.
    assert len(client.fallback_events) == 1
    assert client.fallback_events[0]["to_provider"] == "deepseek"


# ---------------------------------------------------------------------------
# costs.py: DeepSeek pricing + provider field
# ---------------------------------------------------------------------------

def test_pricing_yaml_loads_deepseek_chat():
    pricing = costs.load_pricing()
    assert pricing["deepseek-chat"] == {"input": 0.27, "output": 1.10}


def test_cost_of_deepseek_chat_pricing_math():
    usage = {"model": "deepseek-chat", "input_tokens": 1_000_000, "output_tokens": 1_000_000}
    usd = costs.cost_of(usage)
    assert usd == pytest.approx(0.27 + 1.10)


def test_record_usage_provider_field_defaults_to_anthropic_when_omitted(tmp_path):
    """Backward compatibility: every call site that predates the DeepSeek
    fallback (and any external code calling record_usage the old way,
    without `provider=`) must still get a record with `"provider":
    "anthropic"` — never a missing/None value."""
    ledger = tmp_path / "ledger.jsonl"
    record = costs.record_usage("draft", "claude-haiku-4-5", 100, 50, ledger_path=ledger)
    assert record["provider"] == "anthropic"

    line = json.loads(ledger.read_text().strip())
    assert line["provider"] == "anthropic"


def test_record_usage_provider_field_recorded_when_given(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    record = costs.record_usage("draft", "deepseek-chat", 100, 50, ledger_path=ledger, provider="deepseek")
    assert record["provider"] == "deepseek"

    line = json.loads(ledger.read_text().strip())
    assert line["provider"] == "deepseek"


def test_per_note_breakdown_provider_per_stage():
    run = {
        "cost_records": [
            {"stage": "draft", "model": "claude-haiku-4-5", "input_tokens": 100, "output_tokens": 50, "usd": 1.0, "provider": "anthropic"},
            {"stage": "judge_sample_0", "model": "deepseek-chat", "input_tokens": 10, "output_tokens": 5, "usd": 0.5, "provider": "deepseek"},
            {"stage": "judge_sample_1", "model": "claude-haiku-4-5", "input_tokens": 10, "output_tokens": 5, "usd": 0.5, "provider": "anthropic"},
        ]
    }
    breakdown = costs.per_note_breakdown(run)
    assert breakdown["drafting"]["providers"] == ["anthropic"]
    assert sorted(breakdown["judging"]["providers"]) == ["anthropic", "deepseek"]
    assert breakdown["stage_providers"] == {
        "draft": "anthropic",
        "judge_sample_0": "deepseek",
        "judge_sample_1": "anthropic",
    }
    # Old top-level totals are unaffected by the new provider bookkeeping.
    assert breakdown["total_usd"] == pytest.approx(2.0)


def test_per_note_breakdown_old_records_without_provider_field_still_work():
    """Run record shape stability for old consumers: cost_records written
    before this feature existed (no "provider" key at all — e.g. a ledger
    line from before this change, or a hand-built dict in an older test)
    must not crash per_note_breakdown, and must be treated as anthropic
    (the only provider that existed at the time)."""
    run = {
        "cost_records": [
            {"stage": "draft", "model": "claude-haiku-4-5", "input_tokens": 100, "output_tokens": 50, "usd": 1.0},
            {"stage": "judge_sample_0", "model": "claude-haiku-4-5", "input_tokens": 10, "output_tokens": 5, "usd": 0.5},
        ]
    }
    breakdown = costs.per_note_breakdown(run)
    assert breakdown["drafting"]["usd"] == pytest.approx(1.0)
    assert breakdown["judging"]["usd"] == pytest.approx(0.5)
    assert breakdown["drafting"]["providers"] == ["anthropic"]
    assert breakdown["stage_providers"]["draft"] == "anthropic"


# ---------------------------------------------------------------------------
# live_available / provider_status — either-key availability
# ---------------------------------------------------------------------------

def test_live_available_true_with_only_deepseek_key(tmp_path):
    config = live.LiveConfig(api_key=None, deepseek_api_key="sk-deepseek-k", daily_budget_usd=5.0)
    available, reason = live.live_available(config, ledger_path=tmp_path / "ledger.jsonl")
    assert available is True
    assert reason


def test_live_available_false_when_both_keys_explicitly_absent(tmp_path):
    config = live.LiveConfig(api_key=None, deepseek_api_key=None, daily_budget_usd=5.0)
    available, reason = live.live_available(config, ledger_path=tmp_path / "ledger.jsonl")
    assert available is False
    assert "ANTHROPIC_API_KEY" in reason
    assert "DEEPSEEK_API_KEY" in reason


def test_provider_status_shape(monkeypatch):
    monkeypatch.setattr(live, "_openai_importable", lambda: True)
    config = live.LiveConfig(api_key="k", deepseek_api_key="dk")
    status = live.provider_status(config)
    assert status == {
        "anthropic": {"key": True},
        "deepseek": {"key": True, "sdk": True},
        "order": ["deepseek", "anthropic"],
        "primary_provider": "deepseek",
    }

    config_no_keys = live.LiveConfig(api_key=None, deepseek_api_key=None)
    status_no_keys = live.provider_status(config_no_keys)
    assert status_no_keys["anthropic"]["key"] is False
    assert status_no_keys["deepseek"]["key"] is False


def test_provider_status_order_reflects_anthropic_primary_override(monkeypatch):
    monkeypatch.setattr(live, "_openai_importable", lambda: True)
    config = live.LiveConfig(api_key="k", deepseek_api_key="dk", primary_provider="anthropic")
    status = live.provider_status(config)
    assert status["order"] == ["anthropic", "deepseek"]
    assert status["primary_provider"] == "anthropic"


# ---------------------------------------------------------------------------
# openai package missing -> DeepSeek unavailable, Anthropic unaffected
# ---------------------------------------------------------------------------

def test_deepseek_provider_unavailable_without_openai_package():
    """The `openai` package is NOT installed in this test environment (it's
    an optional dependency purely for the DeepSeek fallback) — so this
    exercises the real absence, not a monkeypatched simulation of it."""
    provider = live.DeepSeekProvider(api_key="sk-deepseek-k", model="deepseek-chat")
    assert provider.available() is False


def test_anthropic_provider_unaffected_by_missing_openai_package(monkeypatch, tmp_path):
    """A missing `openai` package must never affect the Anthropic path —
    DeepSeekProvider simply reports unavailable and is skipped by
    FailoverClient, Anthropic-only calls proceed exactly as before."""
    ledger = tmp_path / "ledger.jsonl"
    config = _make_config(tmp_path)  # deepseek_api_key defaults to None anyway
    _install_fake_client(monkeypatch, [_FakeResponse("hello", input_tokens=10, output_tokens=5)])

    client = live.FailoverClient(
        config,
        [live.AnthropicProvider(config.api_key), live.DeepSeekProvider(config.deepseek_api_key, config.deepseek_model)],
        ledger_path=ledger,
    )
    result = client.create(stage="draft", model="claude-haiku-4-5", max_tokens=10, messages=[{"role": "user", "content": "hi"}])
    assert result["provider"] == "anthropic"
    assert result["text"] == "hello"
    assert client.fallback_events == []


def test_provider_status_reports_no_sdk_when_openai_missing():
    status = live.provider_status(live.LiveConfig(api_key="k", deepseek_api_key="dk"))
    assert status["deepseek"]["sdk"] is False


# ---------------------------------------------------------------------------
# run_live_note — full-run failover integration
# ---------------------------------------------------------------------------

def test_run_live_note_fails_over_full_run_populates_fallback_events(monkeypatch, tmp_path):
    """End-to-end with primary_provider explicitly set to "anthropic": every
    call Anthropic receives fails with an auth error; DeepSeek is configured
    and healthy and serves every stage instead as the fallback. The run
    completes normally (not partial, not budget_exhausted) with
    fallback_events recorded for each of the 4 stages, and no key material
    anywhere in the saved artifact."""
    anthropic_secret = "sk-ant-MUST-NOT-LEAK"
    deepseek_secret = "sk-deepseek-MUST-NOT-LEAK"
    config = _both_keys_config(
        tmp_path,
        api_key=anthropic_secret,
        deepseek_api_key=deepseek_secret,
        judge_samples=3,
        primary_provider="anthropic",
    )
    ledger = tmp_path / "ledger.jsonl"
    live_runs_dir = tmp_path / "live_runs"

    _install_fake_client(
        monkeypatch,
        [AuthenticationError("nope")] * 4,  # 1 draft + 3 judge calls, all fail
    )
    _install_fake_deepseek_client(
        monkeypatch,
        [
            _FakeDeepSeekResponse(_draft_response_json(), prompt_tokens=500, completion_tokens=200),
            _FakeDeepSeekResponse(_judge_response_json(), prompt_tokens=300, completion_tokens=100),
            _FakeDeepSeekResponse(_judge_response_json(), prompt_tokens=300, completion_tokens=100),
            _FakeDeepSeekResponse(_judge_response_json(), prompt_tokens=300, completion_tokens=100),
        ],
    )

    result = live.run_live_note(
        GLAUCOMA_05, config=config, ledger_path=ledger, live_runs_dir=live_runs_dir
    )

    assert result["budget_exhausted"] is False
    assert result["provider_unavailable"] is False
    assert result["partial"] is False
    assert result["generated_note"] is not None
    assert result["judge_samples_collected"] == 3

    fallback_events = result["fallback_events"]
    assert len(fallback_events) == 4
    assert {e["stage"] for e in fallback_events} == {"draft", "judge_sample_0", "judge_sample_1", "judge_sample_2"}
    assert all(e["from_provider"] == "anthropic" and e["to_provider"] == "deepseek" for e in fallback_events)
    assert all(e["reason_class"] == "auth_error" for e in fallback_events)

    breakdown = result["cost_breakdown"]
    assert breakdown["drafting"]["providers"] == ["deepseek"]
    assert breakdown["judging"]["providers"] == ["deepseek"]

    result_str = json.dumps(result)
    assert anthropic_secret not in result_str
    assert deepseek_secret not in result_str

    saved_files = list(live_runs_dir.glob(f"*_{GLAUCOMA_05}.json"))
    assert len(saved_files) == 1
    saved_text = saved_files[0].read_text()
    assert anthropic_secret not in saved_text
    assert deepseek_secret not in saved_text
    ledger_text = ledger.read_text()
    assert anthropic_secret not in ledger_text
    assert deepseek_secret not in ledger_text


# ---------------------------------------------------------------------------
# provider_chain — configurable primary/fallback order (default DeepSeek)
# ---------------------------------------------------------------------------

def test_live_config_default_primary_provider_is_deepseek():
    """A bare `LiveConfig()` (as tests construct directly) and
    `LiveConfig.from_env()` (as production code always uses) must both
    default to "deepseek" as the primary provider — this demo ships with a
    DeepSeek key in Streamlit secrets; Anthropic is optional/added later."""
    assert live.LiveConfig().primary_provider == "deepseek"


def test_live_config_from_env_default_primary_provider_deepseek(monkeypatch):
    monkeypatch.delenv("SCRIBEGATE_PRIMARY_PROVIDER", raising=False)
    config = live.LiveConfig.from_env()
    assert config.primary_provider == "deepseek"


def test_live_config_from_env_primary_provider_override_anthropic(monkeypatch):
    monkeypatch.setenv("SCRIBEGATE_PRIMARY_PROVIDER", "anthropic")
    config = live.LiveConfig.from_env()
    assert config.primary_provider == "anthropic"


def test_live_config_from_env_primary_provider_invalid_value_falls_back_to_deepseek(monkeypatch):
    monkeypatch.setenv("SCRIBEGATE_PRIMARY_PROVIDER", "not-a-real-provider")
    config = live.LiveConfig.from_env()
    assert config.primary_provider == "deepseek"


def test_provider_chain_default_order_deepseek_first(tmp_path):
    config = _both_keys_config(tmp_path)
    chain = live.provider_chain(config)
    assert [p.name for p in chain] == ["deepseek", "anthropic"]


def test_provider_chain_anthropic_primary_override_order(tmp_path):
    config = _both_keys_config(tmp_path, primary_provider="anthropic")
    chain = live.provider_chain(config)
    assert [p.name for p in chain] == ["anthropic", "deepseek"]


def test_failover_client_missing_primary_key_skips_to_secondary_cleanly(monkeypatch, tmp_path):
    """Default order is deepseek-first, but with no DEEPSEEK_API_KEY
    configured (Anthropic-only deployment), the chain must filter down to
    just AnthropicProvider and succeed cleanly via Anthropic — no fallback
    event logged (there was nothing to fail over FROM; DeepSeek was never
    even attempted because it reported unavailable)."""
    ledger = tmp_path / "ledger.jsonl"
    config = live.LiveConfig(
        api_key="k", deepseek_api_key=None, daily_budget_usd=5.0, primary_provider="deepseek"
    )
    _install_fake_client(monkeypatch, [_FakeResponse("served by anthropic", input_tokens=10, output_tokens=5)])

    chain = live.provider_chain(config)
    assert [p.name for p in chain] == ["deepseek", "anthropic"]

    client = live.FailoverClient(config, chain, ledger_path=ledger)
    result = client.create(
        stage="draft", model="claude-haiku-4-5", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
    )

    assert result["provider"] == "anthropic"
    assert result["text"] == "served by anthropic"
    assert client.fallback_events == []


def test_run_live_note_default_chain_serves_via_deepseek_no_failover(monkeypatch, tmp_path):
    """Integration: both keys configured, default primary_provider
    ("deepseek"). DeepSeek serves every stage on the first attempt — no
    fallback events, and Anthropic's fake client is never even invoked,
    proving DeepSeek (not Anthropic) is tried first by default."""
    ledger = tmp_path / "ledger.jsonl"
    live_runs_dir = tmp_path / "live_runs"
    config = _both_keys_config(tmp_path, judge_samples=1)

    fake_anthropic = _install_fake_client(monkeypatch, [])  # must never be called
    _install_fake_deepseek_client(
        monkeypatch,
        [
            _FakeDeepSeekResponse(_draft_response_json(), prompt_tokens=500, completion_tokens=200),
            _FakeDeepSeekResponse(_judge_response_json(), prompt_tokens=300, completion_tokens=100),
        ],
    )

    result = live.run_live_note(
        GLAUCOMA_05, config=config, ledger_path=ledger, live_runs_dir=live_runs_dir
    )

    assert result["fallback_events"] == []
    assert result["cost_breakdown"]["drafting"]["providers"] == ["deepseek"]
    assert result["cost_breakdown"]["judging"]["providers"] == ["deepseek"]
    assert fake_anthropic.messages.calls == []


def test_run_live_note_default_chain_falls_back_to_anthropic_when_deepseek_fails(monkeypatch, tmp_path):
    """Integration: default order (deepseek primary), DeepSeek fails with a
    transient error on every call, Anthropic (fallback) serves instead. This
    is the mirror image of the legacy anthropic-primary failover test —
    proof the failover direction generalizes both ways, not just the old
    Anthropic-primary one."""
    ledger = tmp_path / "ledger.jsonl"
    live_runs_dir = tmp_path / "live_runs"
    config = _both_keys_config(tmp_path, judge_samples=1)

    _install_fake_deepseek_client(monkeypatch, [RateLimitError("rate limited")] * 2)
    _install_fake_client(
        monkeypatch,
        [
            _FakeResponse(_draft_response_json(), input_tokens=500, output_tokens=200),
            _FakeResponse(_judge_response_json(), input_tokens=300, output_tokens=100),
        ],
    )

    result = live.run_live_note(
        GLAUCOMA_05, config=config, ledger_path=ledger, live_runs_dir=live_runs_dir
    )

    fallback_events = result["fallback_events"]
    assert len(fallback_events) == 2
    assert all(e["from_provider"] == "deepseek" and e["to_provider"] == "anthropic" for e in fallback_events)
    assert all(e["reason_class"] == "rate_limit" for e in fallback_events)
    assert result["cost_breakdown"]["drafting"]["providers"] == ["anthropic"]
    assert result["cost_breakdown"]["judging"]["providers"] == ["anthropic"]


# ---------------------------------------------------------------------------
# Per-provider model routing — drafter/judge model must follow whichever
# provider actually serves a given call, never the other provider's name.
# ---------------------------------------------------------------------------

def test_per_provider_model_routing_deepseek_call_gets_deepseek_chat_model(monkeypatch, tmp_path):
    """When DeepSeek serves a call, the underlying SDK must receive
    `config.deepseek_model` ("deepseek-chat" by default) — never the
    Anthropic model name that was requested for the logical call."""
    ledger = tmp_path / "ledger.jsonl"
    config = _both_keys_config(tmp_path)
    fake_deepseek = _install_fake_deepseek_client(monkeypatch, [_FakeDeepSeekResponse("hi from deepseek")])

    client = live.FailoverClient(config, live.provider_chain(config), ledger_path=ledger)
    result = client.create(
        stage="draft", model="claude-haiku-4-5", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
    )

    assert result["provider"] == "deepseek"
    assert fake_deepseek.chat.completions.calls[0]["model"] == "deepseek-chat"
    assert result["cost_record"]["model"] == "deepseek-chat"


def test_per_provider_model_routing_anthropic_call_gets_requested_claude_model(monkeypatch, tmp_path):
    """When DeepSeek is unavailable (no key) and Anthropic serves the call
    instead, the underlying SDK must receive the ACTUAL requested Anthropic
    model name unchanged — never DeepSeek's model string."""
    ledger = tmp_path / "ledger.jsonl"
    config = live.LiveConfig(api_key="k", deepseek_api_key=None, daily_budget_usd=5.0)
    fake_anthropic = _install_fake_client(monkeypatch, [_FakeResponse("hi from anthropic")])

    client = live.FailoverClient(config, live.provider_chain(config), ledger_path=ledger)
    result = client.create(
        stage="draft", model="claude-sonnet-4-5", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
    )

    assert result["provider"] == "anthropic"
    assert fake_anthropic.messages.calls[0]["model"] == "claude-sonnet-4-5"
    assert result["cost_record"]["model"] == "claude-sonnet-4-5"


def test_run_live_note_all_providers_fail_returns_clean_error(monkeypatch, tmp_path):
    """When the only configured provider (Anthropic; no DEEPSEEK_API_KEY set)
    fails outright, run_live_note must fail closed exactly like the existing
    "no bundled transcript" / budget-exhausted paths: a clean result dict
    with an `error`, `generated_note: None`, `partial: True` — never a
    raised exception bubbling into the caller (the UI)."""
    config = _make_config(tmp_path)  # anthropic-only; no deepseek key
    ledger = tmp_path / "ledger.jsonl"
    _install_fake_client(monkeypatch, [AuthenticationError("nope")])

    result = live.run_live_note(
        GLAUCOMA_05, config=config, ledger_path=ledger, live_runs_dir=tmp_path / "live_runs"
    )

    assert result["generated_note"] is None
    assert result["provider_unavailable"] is True
    assert result["budget_exhausted"] is False
    assert result["partial"] is True
    assert "error" in result
    assert "nope" not in result["error"]  # no raw error body/message leaked
