# DEMO_SCRIPT.md — ScribeGate, 5-minute walkthrough

**Audience:** the COO and the clinician founder (ophthalmologist CEO).
**Data:** 100% synthetic throughout. No PHI, no real patients, no live clinical APIs.

## Cold open (30s)

> "This is the QA spine your scribe needs, not a competing scribe. Whatever
> generates your notes today — or next year — this is the eval gate, the
> provenance trail, and the review loop in front of it that decides which notes
> are safe to keep. I'll show you it catching its own mistakes on purpose."

Open the dashboard: `streamlit run app/streamlit_app.py`. The synthetic-data
banner rides on every view — the warning travels with the UI, not just the README.

## Beat 1 — Overview dashboard: honest low scores are the feature (45s)

**Click:** Overview. Four visit types, 20 synthetic transcripts, mock backends.

- Completeness tops out at **3.0** — the mock generator is deliberately extractive,
  so it *pulls* from the transcript rather than *synthesizing*.
- `contact_lens_fitting` sits lowest (0.78) **by design** — that fixture set is
  intentionally messy: colloquial dictation, crosstalk, loose structure.

> "If I showed you all fives, I'd be showing you a broken eval. Low scores on
> noisy input mean the gate is discriminating — that's the product."

## Beat 2 — Provenance on glaucoma_05: the wow (60s)

**Click:** Provenance → `glaucoma_05` (POAG both eyes, progressing on maximal
medical therapy, referred to surgery). Click the **Plan** line: *"I know it's a
big step, but the right eye needs more than drops can give. In the meantime,
keep taking all 3 medications as best you can — every bit of pressure lowering
helps until surgery. I'll continue the latanoprost, timolol, and brimonidine
unchanged for now. I'll call the surgeon's office today and want you seen
within two weeks. The staff will arrange the surgical referral and get you the
soonest appointment."* Watch **five separate transcript segments** light up —
the "big step" framing, the interim-medication instruction, the drug list, the
two-week timeline, and the referral-arrangement line — all stitched into one
Plan line.

> "Every clause traces to an exact character range in the source. Not 'the model
> probably said this' — *this sentence came from these words, here and here.*
> That's what a chart audit or a malpractice inquiry actually asks for."

## Beat 3 — Review queue + one correction: the data moat (45s)

**Click:** Review queue. Open a `review`-routed note (0.60–0.85 aggregate). Edit
one line — tighten a finding rather than reject the whole note.

> "Today a reviewer's correction logs as approve/reject. In production, that edited
> line becomes a candidate golden example. Every correction a clinician makes
> tightens the eval, which tightens the model, which reduces future corrections.
> The moat isn't the model — it's this loop."

## Beat 4 — Drift dashboard: CI blocks the bad PR (40s)

**Click:** Drift. A simulated model-v2 regresses the rolling score window past
threshold. An **alert fires**; the gate **blocks the PR** in CI.

> "This catches silent model updates, prompt drift, or transcription decay *before*
> a clinician notices in the wild. The gate observes a new model before it enforces
> against live notes — never flip a backend without shadow data."

## Beat 5 — Analytics: the ROI model, the COO minute (30s, for the COO)

**Click:** Analytics → scroll to the **ROI model** section.

> "The value isn't 'AI writes notes.' It's that a bounded fraction reaches a
> human at all: `auto_accept` above 0.85, `review` in the middle, `regenerate`
> below 0.60. You're buying down reviewer minutes per note and audit risk per note,
> both measured, not asserted."

## Beat 6 — Live encounter capture: consent gate FIRST (60s)

**Click:** Live encounter. **Stop on the consent gate. Do not rush it.**

> "The microphone will not arm without recorded consent. Recording a clinical
> conversation implicates all-party-consent law in about a dozen states — so both
> the provider and the patient attest, the state is logged, and a consent event is
> written to the same append-only log as the note itself, timestamped, before a
> single word is captured."

Check both attestation boxes. The record button enables. **Click Start.** Set the
speaker toggle and read this synthetic exchange aloud (~20s):

> **PROVIDER:** "Your right eye pressure is up to twenty-six today, above target,
> and the nerve looks like it's thinned a bit more since last time."
>
> **PATIENT:** "So the drops aren't holding it anymore?"
>
> **PROVIDER:** "Right — you're already on three. I'd rather refer you to a glaucoma
> surgeon now than add a fourth drop that won't do much."

**Click:** *Proceed to pipeline*, then *Generate note*.

> "One caveat, stated plainly: the demo speech engine has no diarization — the
> speaker labels come from that toggle, not the audio. Production treats diarization
> quality as a first-class eval input, because a wrong speaker label corrupts S/O
> sectioning before the drafter runs."

## Beat 7 — Generated note with provenance (20s)

The live transcript flows through generate → normalize → judge → route and lands as
a scored SOAP note with the same clickable span provenance from Beat 2.

> "Same spine, whether the transcript came from a fixture file or that microphone
> thirty seconds ago. The gate doesn't change shape when the input does."

## The clinician founder hook

> "The golden notes are the thing I most want wrong. Every rubric anchor
> and reference note is a clinical judgment call, and I'd rather have your red pen
> on my rubric than your compliments. Show me where the eval is too lenient and
> I'll show you a better gate by Friday."

## Objection handling

**Q: Isn't this just our product?** A: No — it's the QA layer in front of your
product. It doesn't compete with the scribe; it decides which of the scribe's notes
are safe to keep and proves why.

**Q: Why synthetic data?** A: So this can be a public, inspectable reference
implementation with zero PHI risk. The architecture is identical against real
transcripts — only the data governance and backends change, not the pipeline shape.

**Q: Why are the completeness scores only 3?** A: The mock generator is deliberately
extractive, not synthesizing, so 3 is its ceiling by design. A real drafter scores
higher; a flat 5 would mean the eval wasn't discriminating.

**Q: What happens with real audio and PHI?** A: The browser speech engine is
demo-only; production swaps in on-prem or BAA-covered ASR with diarization, and the
decision log becomes a tamper-evident, hash-chained store. None of that is built
here — it's the documented next step.

**Q: How long to production-harden?** A: Week 1 stands up the harness against your
*current* note output for a baseline, then wires the decision log and provenance
view — value before any model change. Shadowing the first real model comes only
after those numbers are trusted.

## Close — the week-1 de-risk

> "The low-risk way in: week one, we point this eval and rubric at whatever
> generates your notes today and get a baseline — no model change, no cutover. Two,
> we wire the decision log and provenance so reviewers see spans and the golden-set
> loop starts accumulating real corrections immediately. Three, and only then, we
> shadow the first real model change — the gate observes before it enforces.
> Everything else — EHR adapters, live consent capture, scaling — comes after the
> harness is trusted. You get the safety spine before you take any model risk."
