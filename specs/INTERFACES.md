# ScribeGate module contracts (LOCKED at G1 — do not drift)

All modules are pure-Python, stdlib + pyyaml only (no API calls unless env-gated as noted).
Demo is MOCK-ONLY and deterministic: `SCRIBEGATE_USE_API=1` + `ANTHROPIC_API_KEY` enables API
backends; default path must never require a key or network.

## Data shapes

**Note dict** (same shape as golden notes in `data/golden_notes/*.json`):
```json
{
  "transcript_id": "glaucoma_05",
  "visit_type": "glaucoma_followup",
  "synthetic": true,
  "soap": {
    "S": [{"text": "...", "spans": [[start, end]]}],
    "O": [...], "A": [...], "P": [...]
  }
}
```
Spans are **character** offsets [start, end) into the raw transcript text (incl. header line).
Generated notes may add top-level `"generated": true` and `"generator": "mock"|"api"`.

Visit types: `comprehensive_exam`, `glaucoma_followup`, `cataract_postop`, `contact_lens_fitting`
(derivable from transcript id prefix: comprehensive_/glaucoma_/cataract_/contactlens_).

## normalizer.py (T3)
```python
@dataclass
class Violation:
    code: str          # e.g. "VA_FORMAT", "IOP_RANGE", "LATERALITY_CONFLICT", "CYL_SIGN", "AXIS_RANGE"
    severity: str      # "error" | "warn"
    message: str
    line_text: str

def check_line(text: str, transcript: str | None = None) -> list[Violation]
def check_note(note: dict, transcript: str | None = None) -> list[Violation]
def normalize_line(text: str) -> str   # canonicalize VA (20/x), IOP units, OD/OS/OU casing, Rx formatting
```
Rules sourced from `specs/terminology.yaml` (load once, module-level).

## generator.py (T4)
```python
def generate_note(transcript_text: str, transcript_id: str, visit_type: str) -> dict  # Note dict
```
- MockBackend (default): deterministic (seed = transcript_id) rule-based drafter + S/O/A/P
  section critics pass. Extracts dialogue content into SOAP lines with real char-span refs.
  MUST be imperfect on purpose: realistic misses/paraphrase drift so the judge has signal;
  messy transcripts (contactlens_*) should naturally yield worse notes.
- APIBackend: Claude drafter + per-section critics; only active when env-gated. Code present,
  never imported at module load without the env flag.

## judge.py (T5)
```python
def judge_note(generated: dict, golden: dict, transcript_text: str) -> dict
# returns:
# {
#   "scores": {"completeness": int, "hallucination": int, "coding_plausibility": int, "terminology": int},  # 1-5
#   "aggregate": float,   # (mean(scores) - 1) / 4  → 0..1
#   "rationales": {dim: "one-line reason"}
# }
```
- Deterministic mock judge (default): golden-comparison scorer — line alignment by similarity
  (difflib), unmatched-golden→completeness penalty, generated-line-without-span-support or
  content absent from transcript→hallucination penalty, normalizer violations→terminology
  penalty, section balance / plan-documentation checks→coding_plausibility. Anchors per
  `specs/rubric.yaml`.
- API judge (Haiku model) env-gated, same return shape.

## router.py + cli.py + benchmark.py (T6)
```python
def route(judge_result: dict, violations: list[Violation]) -> str
# "auto_accept" if aggregate >= 0.85 and no error-severity violation
# "review" if 0.60 <= aggregate < 0.85 and no error-severity violation
# "regenerate" otherwise (aggregate < 0.60 OR any error-severity violation)
```
Thresholds read from `specs/rubric.yaml` router_thresholds.
CLI: `python -m scribegate.cli run [--transcript ID] [--all]` → writes `data/results/{id}.json`
(generated note + judge result + route + violations). `benchmark.py` aggregates results into
a per-visit-type / per-dimension markdown table → `data/results/benchmark.md`.
