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
Drift detection + CI eval gate | Analytics/ROI | Correction loop (reviewer edits → golden examples) | Audit dossier + integrity hashes | Consent-gated live encounter capture (reference-free judging)
    v
Provenance view + append-only decision log
```

Mock mode is deterministic by default. Claude API backends are never required and only activate when SCRIBEGATE_USE_API=1 is set; default path works offline.

## Feature Tour

- **Overview:** Dashboard landing page showing synthetic-data banner and key metrics.
- **Analytics:** Per-visit-type benchmark summary, mean scores, and routing breakdown (auto_accept / review / regenerate).
- **Drift:** Rolling-window quality regression detection and alert status.
- **Review queue:** Generated notes routed to `review` (0.60–0.85 aggregate); click to view provenance and make line-level corrections.
- **Provenance:** Click any SOAP line to highlight the exact transcript character spans it traces to (one or many spans stitched into a single line).
- **Live encounter:** Browser mic input → speech-to-text → eval pipeline; **microphone disabled until provider and patient consent is both recorded to the audit log.**
- **About:** Product story, synthetic-data notice, link to PRODUCTION_PATH.md.

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

# Launch interactive dashboard (7-page demo app — see Feature Tour)
streamlit run app/streamlit_app.py

# optional: mic capture with browser speech engine (streamlit-mic-recorder in requirements)
# demo works fully via text entry without a mic; Chrome recommended if using mic

# Run full test suite (223 tests)
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

## CI Eval Gate

PRs run the full pipeline (`python -m scribegate.cli run --all`) and fail if quality falls below committed baseline floors in `specs/baseline.json`. The drift detector tracks the rolling-window history in `data/results/history.jsonl` for operational insight; the hard CI gate compares only the latest run against static baselines so a single regression cannot slip through. A degraded-model demo row in `history_demo.jsonl` shows the gate catching a silent model downgrade in CI.

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
  drift.py                      # Rolling-window regression detection + CI baseline gate
  analytics.py                  # Per-visit-type ROI and failure-mode analytics
  corrections.py                # Reviewer line-level corrections → candidate golden examples
  audit.py                      # Dossier assembly: note + scores + corrections + hashes

specs/
  INTERFACES.md                 # Data shapes and module contracts (frozen)
  terminology.yaml              # Validation rules for VA, IOP, laterality, Rx
  rubric.yaml                   # Judge scoring anchors and router thresholds
  baseline.json                 # CI eval gate floors per metric
  consent_copy.yaml             # UI strings for live-capture consent gate

data/
  transcripts/                  # Synthetic clinic visit recordings (*.txt)
  golden_notes/                 # Reference SOAP notes (*.json) for comparison
  results/
    *.json                      # Generated note + judge result + route per transcript
    decision_log.jsonl          # Append-only log; CLI route events + reviewer decisions
    candidate_golden.jsonl      # Reviewer corrections (immutable append-only)
    history.jsonl               # Rolling-window history for drift detection
    benchmark.md                # Aggregated per-visit-type results

app/
  streamlit_app.py              # Dashboard: Benchmark, Drift, Review queue, Provenance, Live capture
  views/                        # Streamlit page modules
  common.py                     # Shared UI/data helpers

.streamlit/                     # Streamlit config (secrets, theme)

tests/
  test_*.py                     # 223 unit & integration tests (mock-only, no API calls)
```

## What This Is Not

- **Not a scribe product.** ScribeGate is a QA harness for evals and provenance, not a note-generation system ready for clinical use.
- **Not clinical advice.** Synthetic data only; no real patient records, no EHR integration, no regulatory clearance.
- **Not production-ready.** This is a reference implementation demonstrating eval-gated quality gates. Hardening for production (HIPAA, audit trails, live API backends, monitoring) is documented separately.

**⚠️ Every view in the Streamlit app (`streamlit run app/streamlit_app.py`) renders its own "100% SYNTHETIC data — no PHI" banner** so the warning travels with the UI, not just this file.

---

*For the production story (hardening roadmap, provenance-in-production, EHR adapter pattern), see [PRODUCTION_PATH.md](PRODUCTION_PATH.md) in this repo.*
