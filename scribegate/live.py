"""live.py (W1) — LIVE mode orchestrator: real Claude-backed drafting +
sampled judging, with cost telemetry and hard budget enforcement, safe for a
PUBLIC hosted app.

This module is inert by default: nothing here runs unless a caller (the
Streamlit UI, W4) explicitly calls `run_live_note` with a `LiveConfig` that
has an API key configured. Importing this module never requires the
`anthropic` package or network access — `anthropic` is imported lazily inside
`_default_client_factory`, exactly like `generator.APIBackend` /
`judge.APIJudge`. The test suite monkeypatches the module-level
`_client_factory` hook so it stays keyless/deterministic/offline; see
tests/test_live.py.

Safety properties this module is responsible for (a public hosted demo runs
on real spend and a shared API key, so these are load-bearing, not cosmetic):

  1. Never construct a real API client unless SCRIBEGATE_DEMO_PASSCODE-style
     gating (`check_passcode`) and budget checks happen in the caller/UI;
     this module additionally self-defends via `GuardedClient`, which checks
     `budget_remaining` BEFORE every single API call and records usage AFTER
     — so even a caller that forgets to pre-check still can't blow past the
     daily budget by more than one in-flight call.
  2. Never let the API key reach a log line, an exception message, or a
     saved run artifact. `LiveConfig.__repr__` redacts it; every dict this
     module writes to disk or returns is built field-by-field from safe
     values (model names, token counts, scores, transcript text) — never a
     blind dump of `LiveConfig.__dict__`.
  3. Never crash the whole run on a malformed/unparseable model response —
     a public demo is exposed to a live model's occasional non-JSON reply,
     truncated output, etc. Drafting and judging both fail closed (empty
     SOAP / conservative low scores) rather than raising, so a single bad
     API response degrades gracefully instead of 500ing the app.
"""

from __future__ import annotations

import dataclasses
import hmac
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from scribegate import calibration, corrections, costs, normalizer, router
from scribegate.generator import SOAP_SECTIONS, parse_utterances, split_into_clauses, visit_type_for
from scribegate.judge import DIMENSIONS, APIJudge, _aggregate, _content_words, _jaccard

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TRANSCRIPT_DIR = _REPO_ROOT / "data" / "transcripts"
_DEFAULT_LIVE_RUNS_DIR = _REPO_ROOT / "data" / "results" / "live_runs"

# Same alignment bar judge.py uses for "is this a real semantic match" — used
# here to decide whether a drafted line's best-matching transcript utterance
# is a confident span or a low-confidence best-effort guess.
_SPAN_CONFIDENCE_THRESHOLD = 0.22  # matches judge._JACCARD_ALIGN_THRESHOLD


# ---------------------------------------------------------------------------
# Secrets / config
# ---------------------------------------------------------------------------

def _get_secret(name: str, default: str | None = None) -> str | None:
    """Look up a config value from st.secrets first (Streamlit Cloud's
    secrets.toml), then os.environ, then `default`. Never raises: st.secrets
    access throws if no secrets.toml is configured at all (a normal state for
    local/dev/test runs), so that's caught defensively — this function must
    be safe to call with zero Streamlit configuration present, including in
    the offline test suite (streamlit is imported lazily, only here, and
    only for a best-effort secrets lookup)."""
    try:
        import streamlit as st  # lazy import — never required for tests

        secrets = getattr(st, "secrets", None)
        if secrets is not None:
            try:
                if name in secrets:
                    return secrets[name]
            except Exception:
                pass
    except Exception:
        pass
    return os.environ.get(name, default)


@dataclass
class LiveConfig:
    """Live-mode configuration. ALWAYS construct via `LiveConfig.from_env()`
    in real usage — the bare dataclass constructor is what tests use to
    inject explicit values, but production code must never hardcode an API
    key or passcode here."""

    api_key: str | None = None
    demo_passcode: str | None = None
    daily_budget_usd: float = 5.0
    drafter_model: str = "claude-haiku-4-5"
    judge_model: str = "claude-haiku-4-5"
    judge_samples: int = 3
    max_tokens_draft: int = 1024
    max_tokens_judge: int = 512

    @classmethod
    def from_env(cls) -> "LiveConfig":
        """Build config from st.secrets/env, never hardcoded. Malformed
        numeric overrides (bad SCRIBEGATE_DAILY_BUDGET_USD /
        SCRIBEGATE_JUDGE_SAMPLES) fall back to the documented defaults
        rather than raising, since this runs on every app cold-start."""
        api_key = _get_secret("ANTHROPIC_API_KEY")
        demo_passcode = _get_secret("SCRIBEGATE_DEMO_PASSCODE")

        budget_raw = _get_secret("SCRIBEGATE_DAILY_BUDGET_USD")
        try:
            daily_budget_usd = float(budget_raw) if budget_raw is not None else 5.0
        except (TypeError, ValueError):
            daily_budget_usd = 5.0

        drafter_model = _get_secret("SCRIBEGATE_DRAFTER_MODEL") or "claude-haiku-4-5"
        judge_model = _get_secret("SCRIBEGATE_JUDGE_MODEL") or "claude-haiku-4-5"

        samples_raw = _get_secret("SCRIBEGATE_JUDGE_SAMPLES")
        try:
            judge_samples = int(samples_raw) if samples_raw is not None else 3
        except (TypeError, ValueError):
            judge_samples = 3

        return cls(
            api_key=api_key,
            demo_passcode=demo_passcode,
            daily_budget_usd=daily_budget_usd,
            drafter_model=drafter_model,
            judge_model=judge_model,
            judge_samples=judge_samples,
        )

    def __repr__(self) -> str:  # never let the key leak into logs/tracebacks
        return (
            "LiveConfig("
            f"api_key={'<set>' if self.api_key else None}, "
            f"demo_passcode={'<set>' if self.demo_passcode else None}, "
            f"daily_budget_usd={self.daily_budget_usd}, "
            f"drafter_model={self.drafter_model!r}, judge_model={self.judge_model!r}, "
            f"judge_samples={self.judge_samples})"
        )

    __str__ = __repr__


def live_available(config: LiveConfig | None = None, ledger_path=None) -> tuple[bool, str]:
    """(available, reason). Checks: is an API key configured? Is there any
    daily budget remaining? This is the "global kill switch" the UI (W4)
    consults before ever offering live mode — if this returns False, the UI
    falls back to mock."""
    config = config or LiveConfig.from_env()
    if not config.api_key:
        return False, "no ANTHROPIC_API_KEY configured"
    remaining = costs.budget_remaining(config, ledger_path=ledger_path)
    if remaining <= 0:
        spent = costs.today_spend(ledger_path=ledger_path)
        return False, (
            f"daily budget exhausted (${spent:.4f} spent of ${config.daily_budget_usd:.2f} today)"
        )
    return True, "live mode available"


def check_passcode(entered: str | None, config: LiveConfig | None = None) -> bool:
    """Constant-time compare of `entered` against the configured demo
    passcode. Returns False (never raises) if no passcode is configured at
    all — an unconfigured passcode must never be treated as "anything
    passes"."""
    config = config or LiveConfig.from_env()
    expected = config.demo_passcode
    if not expected or entered is None:
        return False
    return hmac.compare_digest(str(entered), str(expected))


# ---------------------------------------------------------------------------
# Budget-guarded client wrapper
# ---------------------------------------------------------------------------

class BudgetExceededError(Exception):
    """Raised by GuardedClient.create() when the daily budget is already
    exhausted BEFORE the call would be made (checked pre-flight, every call,
    not just at run start)."""


def _default_client_factory(api_key: str | None):
    """Real Anthropic client construction — `anthropic` is imported lazily
    here, never at module load, so importing scribegate.live never requires
    the package or network. Tests replace the module-level `_client_factory`
    name with a fake factory so this function is never actually invoked in
    the test suite."""
    import anthropic  # lazy import — never at module load

    return anthropic.Anthropic(api_key=api_key)


# Module-level hook: tests monkeypatch `scribegate.live._client_factory` to
# inject a fake client, keeping the whole suite offline/deterministic/keyless.
_client_factory = _default_client_factory


def _response_text(response) -> str:
    """Extract concatenated text blocks from an Anthropic Messages response.
    Defensive against a response with no text blocks at all (empty string,
    not a crash)."""
    content = getattr(response, "content", None) or []
    return "".join(
        getattr(block, "text", "") for block in content if getattr(block, "type", None) == "text"
    )


class GuardedClient:
    """Wraps `client.messages.create` with budget-before / usage-after
    accounting. Every single call — drafting AND every judge sample — goes
    through `create()`. Never logs or persists the API key: only
    stage/model/token-count/usd ever appear in `cost_records` or the ledger.
    """

    def __init__(self, config: LiveConfig, ledger_path=None):
        self.config = config
        self.ledger_path = ledger_path
        self.cost_records: list[dict] = []
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            self._client = _client_factory(self.config.api_key)
        return self._client

    def create(self, *, stage: str, model: str, max_tokens: int, messages: list) -> dict:
        remaining = costs.budget_remaining(self.config, ledger_path=self.ledger_path)
        if remaining <= 0:
            raise BudgetExceededError(
                f"daily budget exhausted before stage '{stage}' (remaining ${remaining:.4f})"
            )
        client = self._ensure_client()
        response = client.messages.create(model=model, max_tokens=max_tokens, messages=messages)
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        record = costs.record_usage(
            stage, model, input_tokens, output_tokens, ledger_path=self.ledger_path
        )
        self.cost_records.append(record)
        return {"response": response, "cost_record": record}


# ---------------------------------------------------------------------------
# Drafting: single real API call, S/O/A/P + span-tagging instructions
# ---------------------------------------------------------------------------

_DRAFTER_INSTRUCTIONS = (
    "You are a clinical scribe drafting a structured SOAP note from a "
    "SYNTHETIC, educational eye-care visit transcript (no real PHI — this is "
    "a demo). Visit type: {visit_type}.\n\n"
    "Read the transcript below and produce concise Subjective/Objective/"
    "Assessment/Plan note lines. For every line, also return one or more "
    "short VERBATIM quotes copied exactly from the transcript text that "
    "support it (do not paraphrase the quotes — copy the exact substring), "
    "so the supporting span can be located programmatically afterward.\n\n"
    'Return STRICT JSON only (no markdown code fences, no commentary), '
    'exactly in this shape:\n'
    '{{"S": [{{"text": "...", "quotes": ["..."]}}], '
    '"O": [{{"text": "...", "quotes": ["..."]}}], '
    '"A": [{{"text": "...", "quotes": ["..."]}}], '
    '"P": [{{"text": "...", "quotes": ["..."]}}]}}\n\n'
    "TRANSCRIPT:\n{transcript}\n"
)


def _build_drafter_prompt(transcript_text: str, visit_type: str) -> str:
    return _DRAFTER_INSTRUCTIONS.format(visit_type=visit_type, transcript=transcript_text)


def _parse_drafter_response(text: str) -> dict[str, list[dict]]:
    """Parse the drafter's JSON response into {section: [{"text","quotes"}]}.
    Fails closed to an empty note (never raises) on any malformed/non-JSON
    response — a public demo must survive an occasional bad model reply."""
    empty = {section: [] for section in SOAP_SECTIONS}
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return empty
    if not isinstance(data, dict):
        return empty

    out: dict[str, list[dict]] = {}
    for section in SOAP_SECTIONS:
        raw_lines = data.get(section)
        lines: list[dict] = []
        if isinstance(raw_lines, list):
            for item in raw_lines:
                if isinstance(item, dict) and item.get("text"):
                    quotes = item.get("quotes")
                    quotes = [str(q) for q in quotes if q] if isinstance(quotes, list) else []
                    lines.append({"text": str(item["text"]), "quotes": quotes})
                elif isinstance(item, str) and item.strip():
                    lines.append({"text": item.strip(), "quotes": []})
        out[section] = lines
    return out


def _attach_spans(draft_lines: dict[str, list[dict]], transcript_text: str) -> dict[str, list[dict]]:
    """For each drafted line, find the transcript utterance with the highest
    content-word Jaccard overlap (reusing judge.py's alignment signal —
    `_content_words` / `_jaccard`) against the line's quoted excerpts (or its
    own text if no quotes were returned), and attach that utterance's REAL
    char span.

    Real LLM-drafted spans are only ever approximate — this is itself an
    honest demo point, not a bug to hide: every attached line gets a
    "span_confidence" of "high" (Jaccard cleared the same bar judge.py uses
    for a real semantic match) or "low" (best-effort guess; either a weak
    match or no matching utterance at all)."""
    utterances = split_into_clauses(parse_utterances(transcript_text))
    utt_word_sets = [(_content_words(u.text), u) for u in utterances]

    out: dict[str, list[dict]] = {}
    for section, lines in draft_lines.items():
        new_lines = []
        for line in lines:
            text = line.get("text", "")
            quotes = line.get("quotes") or []
            search_text = " ".join(quotes) if quotes else text
            line_words = _content_words(search_text)

            best_score = 0.0
            best_utt = None
            for words, u in utt_word_sets:
                score = _jaccard(line_words, words)
                if score > best_score:
                    best_score = score
                    best_utt = u

            if best_utt is not None and best_score >= _SPAN_CONFIDENCE_THRESHOLD:
                spans = [[best_utt.start, best_utt.end]]
                confidence = "high"
            elif best_utt is not None and best_score > 0:
                spans = [[best_utt.start, best_utt.end]]
                confidence = "low"
            else:
                spans = []
                confidence = "low"

            new_lines.append({"text": text, "spans": spans, "span_confidence": confidence})
        out[section] = new_lines
    return out


# ---------------------------------------------------------------------------
# Judging: n real API calls vs the active golden generation
# ---------------------------------------------------------------------------

def _parse_judge_response(text: str) -> dict:
    """Parse one judge sample's JSON response into the standard judge-result
    shape ({"scores", "aggregate", "rationales"}). Fails closed to the lowest
    (most conservative) scores on any parse failure — an unparseable judge
    response must never be silently treated as a passing note."""
    try:
        parsed = json.loads(text)
        scores = {dim: int(parsed["scores"][dim]) for dim in DIMENSIONS}
        scores = {dim: max(1, min(5, s)) for dim, s in scores.items()}
        rationales = {dim: str(parsed.get("rationales", {}).get(dim, "")) for dim in DIMENSIONS}
    except Exception:
        scores = {dim: 1 for dim in DIMENSIONS}
        rationales = {
            dim: "judge response unparseable; scored conservatively (fail-closed)."
            for dim in DIMENSIONS
        }
    return {"scores": scores, "aggregate": _aggregate(scores), "rationales": rationales}


_NO_SAMPLES_RESULT = {
    "samples": [],
    "mean_scores": {dim: 0.0 for dim in DIMENSIONS},
    "std_scores": {dim: 0.0 for dim in DIMENSIONS},
    "aggregate_mean": 0.0,
    "aggregate_std": 0.0,
    "ci95": [0.0, 0.0],
    "agreement": {dim: 0.0 for dim in DIMENSIONS},
    "flags": ["NO_JUDGE_SAMPLES"],
    "difficulty": None,
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _iso_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _violation_to_dict(v) -> dict:
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return dataclasses.asdict(v)
    if isinstance(v, dict):
        return v
    return {"value": str(v)}


def run_live_note(
    transcript_id: str,
    drafter_model: str | None = None,
    config: LiveConfig | None = None,
    *,
    transcript_dir: str | Path | None = None,
    live_runs_dir: str | Path | None = None,
    ledger_path: str | Path | None = None,
) -> dict:
    """Run one live (real-API) drafting + sampled-judging pass for a bundled
    transcript, save the result under data/results/live_runs/{ts}_{id}.json,
    and return it.

    Pipeline: real single-call API drafter (S/O/A/P + span-tagging
    instructions) -> span post-processing/confidence tagging -> normalizer
    violations -> REAL sampled judging (n=config.judge_samples API calls)
    against the transcript's active golden generation (via
    `corrections.load_golden_note`, may be None/empty if this transcript_id
    has no golden reference) -> CI-aware routing (`calibration.route_sampled`,
    reusing the same stats layer `calibration.py` uses for the mock sampled
    judge) -> cost record.

    Every single API call (the draft call and each judge sample) goes through
    `GuardedClient`, which checks the remaining daily budget BEFORE the call
    and records usage AFTER. If the budget runs out mid-run, this aborts
    cleanly at the next call boundary and returns a partial result with
    `"budget_exhausted": True` — never a raised exception, and never a
    partially-billed call (the guard is pre-flight, not a rollback).

    Never raises on malformed model output (drafting/judging both fail
    closed — see `_parse_drafter_response` / `_parse_judge_response`), and
    never includes the API key in the returned dict or the saved JSON file.
    """
    config = config or LiveConfig.from_env()
    model = drafter_model or config.drafter_model
    transcript_dir = Path(transcript_dir) if transcript_dir else _DEFAULT_TRANSCRIPT_DIR
    live_runs_dir = Path(live_runs_dir) if live_runs_dir else _DEFAULT_LIVE_RUNS_DIR

    ts = _iso_ts()
    guarded = GuardedClient(config, ledger_path=ledger_path)

    def _finalize(result: dict) -> dict:
        result.setdefault("cost_records", guarded.cost_records)
        result.setdefault("cost_breakdown", costs.per_note_breakdown({"cost_records": guarded.cost_records}))
        result.setdefault("ts", ts)
        result.setdefault("transcript_id", transcript_id)
        live_runs_dir.mkdir(parents=True, exist_ok=True)
        out_path = live_runs_dir / f"{ts}_{transcript_id}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=False)
        result["_saved_path"] = str(out_path)
        return result

    transcript_path = transcript_dir / f"{transcript_id}.txt"
    if not transcript_path.exists():
        return _finalize(
            {
                "error": f"no bundled transcript found for transcript_id={transcript_id!r}",
                "generated_note": None,
                "budget_exhausted": False,
                "partial": True,
            }
        )
    with open(transcript_path, "r", encoding="utf-8") as fh:
        transcript_text = fh.read()

    visit_type = visit_type_for(transcript_id)
    golden = corrections.load_golden_note(transcript_id) or {}

    # --- Drafting: one real API call -------------------------------------
    drafter_prompt = _build_drafter_prompt(transcript_text, visit_type)
    try:
        draft_call = guarded.create(
            stage="draft",
            model=model,
            max_tokens=config.max_tokens_draft,
            messages=[{"role": "user", "content": drafter_prompt}],
        )
    except BudgetExceededError as exc:
        return _finalize(
            {
                "error": str(exc),
                "generated_note": None,
                "budget_exhausted": True,
                "partial": True,
                "drafter_model": model,
                "judge_model": config.judge_model,
                "visit_type": visit_type,
                "transcript_text": transcript_text,
            }
        )

    draft_text = _response_text(draft_call["response"])
    draft_lines = _parse_drafter_response(draft_text)
    soap = _attach_spans(draft_lines, transcript_text)

    generated_note = {
        "transcript_id": transcript_id,
        "visit_type": visit_type,
        "synthetic": True,
        "soap": soap,
        "generated": True,
        "generator": "api",
        "drafter_model": model,
    }

    try:
        violations = normalizer.check_note(generated_note, transcript=transcript_text)
    except Exception:
        violations = []

    # --- Sampled judging: n real API calls vs the active golden ----------
    judge_prompt = APIJudge(model=config.judge_model)._build_prompt(
        generated_note, golden, transcript_text
    )

    samples: list[dict] = []
    budget_exhausted = False
    for i in range(config.judge_samples):
        try:
            judge_call = guarded.create(
                stage=f"judge_sample_{i}",
                model=config.judge_model,
                max_tokens=config.max_tokens_judge,
                messages=[{"role": "user", "content": judge_prompt}],
            )
        except BudgetExceededError:
            budget_exhausted = True
            break
        text = _response_text(judge_call["response"])
        samples.append(_parse_judge_response(text))

    if samples:
        sampled_result = calibration._aggregate_sampled_result(samples, len(samples))
        sampled_result["difficulty"] = None  # real API variance, not injected
        route_result = calibration.route_sampled(sampled_result, violations)
    else:
        sampled_result = dict(_NO_SAMPLES_RESULT)
        decision = router.decide({"aggregate": 0.0}, violations)
        route_result = {
            "route": decision.route,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "aggregate_mean": 0.0,
            "reasons": decision.reasons + ["no judge samples collected (budget exhausted before first judge call)"],
            "routing_delta": {
                "point_route": decision.route,
                "ci_route": decision.route,
                "changed": False,
                "explanation": "no judge samples available; routed conservatively.",
            },
        }

    result = {
        "transcript_id": transcript_id,
        "visit_type": visit_type,
        "drafter_model": model,
        "judge_model": config.judge_model,
        "judge_samples_requested": config.judge_samples,
        "judge_samples_collected": len(samples),
        "generated_note": generated_note,
        "golden_used": bool(golden),
        "violations": [_violation_to_dict(v) for v in violations],
        "judge_sampled": sampled_result,
        "route_result": route_result,
        "route": route_result["route"],
        "budget_exhausted": budget_exhausted,
        "partial": budget_exhausted,
        "transcript_text": transcript_text,
    }
    return _finalize(result)
