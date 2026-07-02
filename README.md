# ScribeGate

**Eval-gated quality harness for AI-generated eye-care SOAP notes, with line-level provenance.**

**⚠️ 100% SYNTHETIC data — no PHI. Educational/demo artifact, not clinical software.**

## Why

AI-generated clinical notes need a QA spine: ScribeGate uses deterministic evals as regression tests to catch hallucinations, missing details, and terminology errors before a note leaves the system. Every line in a generated note is traceable to its source transcript—click a note line in the UI and see the exact character span it came from. This provenance chain is the foundation for human review.

## Architecture

```
Transcript (text file)
    |
    v
Generator (mock-only by default; Claude API env-gated via SCRIBEGATE_USE_API=1)
    |---> Generates SOAP note with char-span references to source
    v
Normalizer
    |---> Checks VA format, IOP units, laterality, Rx syntax
    v
Judge (mock scoring engine; Haiku API available env-gated)
    |---> Scores completeness, hallucination, coding_plausibility, terminology (1-5 scale)
    |---> Computes aggregate (0-1 normalized)
    v
Router (confidence thresholds)
    |---> aggregate >= 0.85 + no error violations → auto_accept
    |---> 0.60 <= aggregate < 0.85 + no error violations → review
    |---> otherwise → regenerate
    v
Provenance view + append-only decision log
```

Mock mode is deterministic by default. Claude API backends are never required and only activate when SCRIBEGATE_USE_API=1 is set; default path works offline.

## Quickstart

Cold-start in under 5 minutes:

```bash
git clone <repo-url>
cd scribegate
pip install -r requirements.txt

# Generate notes, run judges, route decisions
python -m scribegate.cli run --all

# Aggregate per-visit-type benchmark report
python -m scribegate.benchmark

# Launch interactive dashboard (3 views: benchmark, review queue, provenance)
streamlit run app/streamlit_app.py

# Run full test suite (162 tests)
python -m pytest -q
```

## Benchmark Results

Current results on 20 synthetic transcripts (4 visit types × 5 cases each), generated and judged by mock backends:

| Visit Type | N | Completeness | Hallucination | Coding Plausibility | Terminology | Mean Aggregate | auto_accept | review | regenerate |
|---|---|---|---|---|---|---|---|---|---|
| comprehensive_exam | 5 | 3.00 | 5.00 | 4.80 | 5.00 | 0.86 | 4 | 1 | 0 |
| glaucoma_followup | 5 | 3.20 | 5.00 | 4.20 | 5.00 | 0.84 | 2 | 3 | 0 |
| cataract_postop | 5 | 2.40 | 5.00 | 4.80 | 4.40 | 0.79 | 2 | 1 | 2 |
| contact_lens_fitting | 5 | 3.20 | 3.20 | 5.00 | 5.00 | 0.78 | 1 | 4 | 0 |

**Reading the table:** Completeness tops out at 3 because the mock generator is deliberately extractive, not synthesizing—it pulls content directly from the transcript. Contact-lens scores lowest by design; the fixture set is intentionally messy (colloquial dictation, crosstalk, loose structure) to stress-test the harness. Low scores on noisy input mean the eval gate is working correctly, not that the pipeline is broken.

## Repo Layout

```
scribegate/
  __init__.py                   # Package marker
  cli.py                        # Entry point: generate → normalize → judge → route
  generator.py                  # Mock drafter + critics (Claude API optional, env-gated)
  normalizer.py                 # VA format, IOP, laterality, Rx rules per specs/terminology.yaml
  judge.py                      # Mock scorer or Haiku API (line alignment, similarity scoring)
  router.py                     # Route decision logic: thresholds from specs/rubric.yaml
  benchmark.py                  # Aggregate results into per-visit-type markdown table

specs/
  INTERFACES.md                 # Data shapes and module contracts (frozen)
  terminology.yaml              # Validation rules for VA, IOP, laterality, Rx
  rubric.yaml                   # Judge scoring anchors and router thresholds

data/
  transcripts/                  # Synthetic clinic visit recordings (*.txt)
  golden_notes/                 # Reference SOAP notes (*.json) for comparison
  results/
    *.json                      # Generated note + judge result + route per transcript
    decision_log.jsonl          # Append-only log; multi-producer (CLI route events +
                                 # reviewer approve/reject events, different schemas)
    benchmark.md                # Aggregated per-visit-type results

app/
  streamlit_app.py              # Dashboard: Benchmark view (⚠️ synthetic-data banner
                                 # rendered live in-app), Review queue, Provenance viewer

tests/
  test_*.py                     # 162 unit & integration tests (mock-only, no API calls)
```

## What This Is Not

- **Not a scribe product.** ScribeGate is a QA harness for evals and provenance, not a note-generation system ready for clinical use.
- **Not clinical advice.** Synthetic data only; no real patient records, no EHR integration, no regulatory clearance.
- **Not production-ready.** This is a reference implementation demonstrating eval-gated quality gates. Hardening for production (HIPAA, audit trails, live API backends, monitoring) is documented separately.

**⚠️ Every view in the Streamlit app (`streamlit run app/streamlit_app.py`) renders its own "100% SYNTHETIC data — no PHI" banner** so the warning travels with the UI, not just this file.

---

*For the production story (hardening roadmap, provenance-in-production, EHR adapter pattern), see [PRODUCTION_PATH.md](PRODUCTION_PATH.md) in this repo.*
