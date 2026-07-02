# PRODUCTION_PATH.md — from demo to QA spine

This document is for evaluating ScribeGate as a *pattern*, not a product. ScribeGate
itself (this repo) is a synthetic-data reference implementation: mock generator, mock
judge, 20 fixture transcripts, no PHI, no live APIs by default. What follows is how the
same architecture — generate → normalize → judge → route → provenance → decision log —
hardens into the QA layer that sits in front of a real AI-scribe pipeline in production.

Nothing below is built. It is the roadmap this repo was designed to extend into.

## 1. From synthetic to real: golden-set governance

The judge (`judge.py`) scores a generated note against a **golden note**. In this repo
golden notes are hand-authored fixtures. In production the golden set has to be a living,
governed asset, not a one-time fixture drop:

- **Clinician-reviewed golden notes.** Every golden note is signed off by a credentialed
  reviewer (ophthalmologist or optometrist) before it enters the set — same standing as a
  chart audit, not a QA nicety.
- **The review queue feeds the golden set.** Today `decision_log.jsonl` records
  approve/reject only. In production, a reviewer's *correction* (edit a line, not just
  reject the note) becomes a candidate golden example. This is the compounding asset:
  every real correction a clinician makes tightens the eval, which tightens the model,
  which reduces future corrections. The moat isn't the model — it's this loop.
- **Drift detection.** Track the score time-series (aggregate + per-dimension) per visit
  type, plotted over time, not just as a point-in-time table like today's
  `benchmark.md`. Alert when a rolling window regresses past a threshold — this catches
  silent model updates, prompt drift, or upstream transcription quality decay before a
  clinician notices in the wild.
- **Versioned rubrics.** `specs/rubric.yaml` is currently a single frozen file. In
  production it's versioned (`rubric_v3.yaml`), every judge run records which version
  scored it, and rubric changes go through the same review as a golden-note change —
  a rubric edit silently changes what "good" means for every note scored after it.

## 2. From mock to live

| Component | Today (this repo) | Production |
|---|---|---|
| Transcription | Pre-recorded `.txt` fixtures | Live ASR/diarization upstream; **diarization quality becomes a first-class eval input** — speaker-attribution errors corrupt S/O sectioning before the drafter even runs |
| Generator | `MockBackend`, deterministic, extractive | `APIBackend` (already stubbed, env-gated) promoted to default; same drafter+critic contract |
| Judge | Mock similarity/rule scorer | API judge (already stubbed) under the same rubric contract — the eval gate does not change shape when the backend does |
| Rollout | N/A | **Shadow mode**: new backend runs alongside the incumbent, judge scores both, results are logged but the gate does not act on the shadow output until its score distribution matches or beats the incumbent over a defined window |

The core discipline: the router contract (`aggregate >= 0.85 → auto_accept`, `0.60–0.85
→ review`, `< 0.60 → regenerate`) does not change when the backend changes. Shadow mode
means the eval gate *observes* a new model/prompt before it *enforces* against it —
never flip a backend live without weeks of shadow data first.

## 3. Provenance in production

The provenance view (span-exact highlighting from note line to transcript offset) is the
most defensible part of this demo, and the part most worth carrying forward unchanged:

- **Append-only decision log → tamper-evident store.** `decision_log.jsonl` today is a
  local file anyone with disk access can edit. In production this becomes a
  write-once store (hash-chained log or an append-only table with a cryptographic
  checksum per row) so a reviewer decision cannot be silently altered after the fact.
- **Per-line span provenance retained through EHR write-back.** The char-offset spans
  that back each SOAP line today must survive the trip into the EHR — not just as a
  UI feature, but as metadata attached to the written note, so a chart audit can
  reconstruct "this sentence came from this transcript segment" after the fact.
- **Audit export.** A single command/report that reconstructs, for any note: generated
  text → judge scores + rationale → route decision → reviewer action → final EHR
  write, with timestamps and actor at each step. This is the artifact a compliance
  review or malpractice inquiry actually asks for.

## 4. EHR adapter pattern — roadmap only, not built

```
                    +------------------------+
                    |   ScribeGate QA spine   |
                    |  (generate/judge/route) |
                    +-----------+------------+
                                |
                       one write-back interface
                                |
              +-----------------+------------------+
              |                 |                   |
      +-------v------+  +-------v------+   +--------v-----+
      | ModMed        |  | Nextech      |   | RevolutionEHR|
      | adapter       |  | adapter      |   | adapter      |
      +---------------+  +--------------+   +--------------+
```

- One internal write-back interface; a thin adapter per EHR translates the SOAP-note
  shape (`specs/INTERFACES.md`'s Note dict) into that EHR's note/encounter API.
- **Auto-detection**: practice-management config identifies which EHR is active: the
  correct adapter loads without a manual switch.
- **Per-adapter contract tests**: each adapter is tested against a frozen contract
  (same discipline as `specs/INTERFACES.md` locking module shapes in this repo) so an
  EHR API change breaks a test, not a production write.
- This is the F-option in the broader roadmap. It is explicitly not scoped or built in
  this repo — the point of ScribeGate is that the QA spine is EHR-agnostic by
  construction, so this layer is additive, not a rearchitecture.

## 5. Consent workflow hook — roadmap only (E-option)

Two-party-consent capture (required in a number of states for recorded clinical
encounters) slots in at the recording step, before any transcript reaches the
generator: a consent prompt/confirmation gates whether the recording pipeline is
armed for that encounter at all. This repo's pipeline starts from an already-consented
transcript file and does not model consent capture — that's upstream of everything
here, and is scoped as future work, not a gap in the QA spine itself.

## 6. Scale / infra sketch

- **Queue-based pipeline.** Transcript-ready events push into a queue; generate →
  normalize → judge → route run as workers, not inline with the recording session —
  same stage boundaries as `cli.py` today, decoupled and horizontally scalable.
- **pgvector for retrieval.** Transcript and note embeddings stored for similarity
  retrieval — surfacing prior visits for the same patient, finding near-duplicate
  golden examples during rubric review, and clustering low-score notes by root cause.
- **Per-visit-type model routing.** Not every visit type needs the same model tier.
  Route cheap/fast models to visit types where evals prove score parity with the
  expensive model (this mirrors the tiered-delegation principle this repo itself was
  built under: don't spend frontier-model budget on work a cheaper tier handles
  equally well, verified, not assumed).
- **Cost ceilings per note.** A hard per-note spend cap (drafter + critics + judge
  combined) with graceful degradation (fewer critic passes, cheaper judge model) rather
  than silent overrun — cost is a gate input, not an afterthought.

## 7. Week 1 in a real deployment

Priority order, and why:

1. **Stand up the eval harness first, before any model changes.** Point judge + rubric
   at the incumbent system's *current* output (whatever generates notes today) to get a
   baseline score distribution. You cannot claim a new model is better without a
   baseline measured the same way.
2. **Wire the decision log and provenance view second.** Reviewers need to see spans and
   record decisions from day one, even against the baseline system — this is also how
   the golden-set correction pipeline (Section 1) starts accumulating real data
   immediately instead of waiting for a cutover.
3. **Shadow the first real model change third**, only after 1 and 2 are producing
   trustworthy numbers. The gate observes before it enforces.

Everything else in this document — EHR adapters, consent hooks, infra scaling — comes
after the harness is trusted, not before.
