"""judge.py (T5) — deterministic mock judge + env-gated API judge.

Contract (specs/INTERFACES.md):
    judge_note(generated: dict, golden: dict, transcript_text: str) -> dict
    # returns:
    # {
    #   "scores": {"completeness": int, "hallucination": int,
    #              "coding_plausibility": int, "terminology": int},  # 1-5
    #   "aggregate": float,   # (mean(scores) - 1) / 4  -> 0..1
    #   "rationales": {dim: "one-line reason"}
    # }

Default path is stdlib-only (+ pyyaml for the rubric prompt used by APIJudge),
deterministic, offline. The API judge (Haiku) is only ever used when
SCRIBEGATE_USE_API=1 and ANTHROPIC_API_KEY are both set; `anthropic` is
imported lazily inside APIJudge so this module never requires the package (or
network) at import time or in the default path.

scribegate.normalizer is currently a stub (just a TODO docstring — no
functions/classes). We import it defensively and always have a working
regex-based fallback terminology checker so this module functions correctly
today, independent of when T3 lands.
"""

from __future__ import annotations

import difflib
import os
import re

SOAP_SECTIONS = ("S", "O", "A", "P")

DIMENSIONS = ("completeness", "hallucination", "coding_plausibility", "terminology")

# Alignment match threshold for difflib.SequenceMatcher ratio on line text.
# Kept as a secondary/tie-break signal alongside the primary semantic
# alignment (content-word Jaccard + numeric-fact agreement) below — a very
# high literal ratio (near-verbatim/self-match) is still treated as aligned
# even if Jaccard is low for some degenerate short-line case.
_ALIGN_THRESHOLD = 0.55

# Primary semantic alignment threshold: a golden line and a generated line
# in the SAME section are considered "aligned" (same clinical content) if
# their stopword-stripped, lemma-ish content-word Jaccard similarity is at
# or above this bar, OR they share key numeric facts (see
# _numeric_facts_agree). This replaces raw difflib.SequenceMatcher-on-text
# as the primary alignment signal — the mock generator composes lines with
# extractive/paraphrased phrasing that legitimately differs in wording from
# golden lines while covering the same clinical content, so a literal
# character-similarity bar was never crossed and the completeness dimension
# was dead (pinned near its floor for every case). Jaccard over normalized
# content words is far more robust to that kind of surface rewording.
_JACCARD_ALIGN_THRESHOLD = 0.22

# Alongside the Jaccard ratio bar, also require at least this many shared
# content words in absolute terms. This guards against short-line noise
# (e.g. two 3-4-word lines sharing one common word can clear a 0.22 RATIO
# purely by having tiny denominators, producing a spurious match) without
# raising the ratio bar itself, which would exclude legitimate longer-line
# paraphrase matches that only share a modest fraction of their (larger)
# vocabularies.
_JACCARD_MIN_SHARED_WORDS = 2

# Lenient span-support threshold: generator/golden lines both heavily
# compress and restructure the raw transcript utterance text into clinical
# shorthand (e.g. "IOP 16 mmHg OS, 17 mmHg OD (iCare)." vs the transcript's
# "iCare tonometry: IOP 16 mmHg OS, 17 mmHg OD"), so a strict character-level
# SequenceMatcher ratio against the *literal* span text is often well below
# 0.5 even for perfectly faithful golden lines. Empirically, golden lines
# against their own cited spans in the fixture data range as low as ~0.25.
# We keep this threshold low and treat span-support as a coarse signal (did
# the generator cite *some* real transcript text at all, however compressed)
# while leaning on the numeric-token check below as the primary, precise
# fabrication detector.
_SPAN_SUPPORT_THRESHOLD = 0.2

# Word-set (Jaccard-style) overlap fallback for span support — see
# score_hallucination for why this is needed alongside the char-ratio check.
_SPAN_WORD_OVERLAP_THRESHOLD = 0.3

_STOPWORDS = {
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "of", "to",
    "in", "on", "for", "with", "at", "by", "from", "no", "not", "this",
    "that", "it", "as", "be", "today", "vs", "prior", "unchanged", "continue",
}


def _significant_words(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z/'\-]*", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


# Simple suffix-stripping "lemma-ish" normalizer — not a real lemmatizer,
# just enough to fold common plural/verb-form variants together so e.g.
# "pressures"/"pressure", "readings"/"reading", "stabilized"/"stable" share
# a token for Jaccard purposes. Deterministic, stdlib-only.
_SUFFIXES = ("ing", "edly", "ed", "ies", "es", "s")


def _lemma_ish(word: str) -> str:
    if word.endswith("ies") and len(word) > 5:
        return word[:-3] + "y"
    for suf in ("ing", "edly", "ed", "es", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def _content_words(text: str) -> set[str]:
    """Stopword-stripped, lowercase, lemma-ish content-word set used for
    semantic line alignment (Jaccard overlap) — see _JACCARD_ALIGN_THRESHOLD."""
    return {_lemma_ish(w) for w in _significant_words(text)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# Bare eye-by-eye paired-number phrasing ("right eye 18, left eye 17") —
# alignment-only IOP fallback. The mock generator's clause-splitter
# sometimes separates the measurement-method clause ("Today's pressures,
# Goldmann.") from the paired-reading clause ("Right eye 18, left eye 17.")
# into consecutive lines, so the literal "IOP"/"mmHg" cue _extract_numeric_
# tokens requires can end up in a different line than the numbers
# themselves. For ALIGNMENT purposes (not fabrication-checking, which stays
# on the stricter cue-gated extractor), also recognize this phrasing as
# carrying IOP-range numeric facts.
_EYE_PAIR_RE = re.compile(r"\bright eye\b.{0,10}?(\d{1,3}).{0,20}\bleft eye\b.{0,10}?(\d{1,3})", re.IGNORECASE)


def _numeric_facts(text: str) -> set[str]:
    """Extract key numeric clinical facts (VA fractions, IOP-context
    integers, diopter/sphere/cylinder decimals, axis values, base-curve/
    diameter values) from a line as a comparable token set. Reuses the same
    extraction as the hallucination numeric-token checker (see
    _extract_numeric_tokens) plus base-curve/diameter tokens, since those are
    the "key numeric facts" call out for alignment per the fitting spec."""
    tokens = set(_extract_numeric_tokens(text))
    for m in _BC_DIA_RE.finditer(text):
        tokens.add(m.group(1))
    for m in _EYE_PAIR_RE.finditer(text):
        for g in (m.group(1), m.group(2)):
            val = int(g)
            if 5 <= val <= 60:
                tokens.add(g)
    return tokens


def _numeric_facts_agree(gen_text: str, gold_text: str) -> bool:
    """True if the two lines share at least one key numeric clinical fact."""
    gen_facts = _numeric_facts(gen_text)
    if not gen_facts:
        return False
    gold_facts = _numeric_facts(gold_text)
    return bool(gen_facts & gold_facts)


# Copy-forward duplicate detection threshold (coding_plausibility check 4).
_DUPLICATE_THRESHOLD = 0.92


# ---------------------------------------------------------------------------
# Word-to-digit fallback mapping for numeric-token hallucination checks.
#
# WHY: synthetic transcripts spell numbers as words ("twenty-six") while
# golden/generated note lines write digits ("26"). A naive "is this digit
# string present in transcript_text" check would fail even for perfectly
# legitimate, transcript-supported content (see glaucoma_05: the golden O
# line says "IOP 26 mmHg OD" but the transcript literally says "right eye
# twenty-six"). So before checking numeric-token support we build a parallel
# spelled-out-number representation of the transcript and check both forms.
# ---------------------------------------------------------------------------

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}

_NUMBER_WORD_RE = re.compile(
    r"\b(" + "|".join(sorted(list(_TENS) + list(_ONES), key=len, reverse=True))
    + r")(?:[\s-](" + "|".join(sorted(_ONES, key=len, reverse=True)) + r"))?\b",
    re.IGNORECASE,
)

# Diopter/Rx-style spoken decimals: clinicians dictate values like "minus
# three seventy-five" (= -3.75), "five twenty-five" (= 5.25), "five fifty"
# (= 5.50), or "five point zero zero" (= 5.00). These transcripts spell such
# compound decimals as WORD WORD (whole-number word, then a 2-digit
# fractional-part word or word-pair), or WORD "point" DIGIT-WORDS. We
# capture both patterns and emit the plausible decimal-string forms so the
# numeric-token support check can match e.g. "-3.75" against the transcript
# phrase "three seventy-five" without requiring literal digits.
_DECIMAL_PHRASE_RE = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+"
    r"(point\s+(?:zero|one|two|three|four|five|six|seven|eight|nine)(?:\s+(?:zero|one|two|three|four|five|six|seven|eight|nine))*"
    r"|(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[\s-](?:one|two|three|four|five|six|seven|eight|nine))?"
    r"|zero|five)\b",
    re.IGNORECASE,
)


def _spelled_numbers_to_digits(text: str) -> set[str]:
    """Scan `text` for spelled-out number words/compounds (e.g. 'twenty-six',
    'thirty two') and return the set of digit-string equivalents found
    ('26', '32', ...), PLUS diopter-style spoken decimals ('three
    seventy-five' -> '3.75', 'five fifty' -> '5.50'/'5.5', 'five point zero
    zero' -> '5.00'/'5.0'). Used as a fallback so numeric-token support
    checks don't false-positive-fail just because the transcript spells
    numbers out instead of using digits (see module docstring)."""
    found: set[str] = set()
    for m in _NUMBER_WORD_RE.finditer(text):
        first = m.group(1).lower()
        second = m.group(2).lower() if m.group(2) else None
        if first in _TENS:
            value = _TENS[first]
            if second and second in _ONES:
                value += _ONES[second]
            found.add(str(value))
        elif first in _ONES:
            found.add(str(_ONES[first]))

    for m in _DECIMAL_PHRASE_RE.finditer(text):
        whole_word = m.group(1).lower()
        frac_phrase = m.group(2).lower()
        whole = _ONES.get(whole_word, _TENS.get(whole_word))
        if whole is None:
            continue
        if frac_phrase.startswith("point"):
            digit_words = frac_phrase.split()[1:]
            digits = "".join(str(_ONES.get(w, "")) for w in digit_words)
            if digits:
                found.add(f"{whole}.{digits}")
                found.add(f"{whole}.{digits.rstrip('0') or '0'}")
        else:
            # compound like "seventy-five", "fifty", "twenty"
            parts = re.split(r"[\s-]", frac_phrase)
            frac_first = parts[0]
            frac_val = _TENS.get(frac_first)
            if frac_val is None:
                continue
            if len(parts) > 1 and parts[1] in _ONES:
                frac_val += _ONES[parts[1]]
            found.add(f"{whole}.{frac_val:02d}")
            # also the reduced form, e.g. 5.50 -> 5.5
            reduced = f"{frac_val:02d}".rstrip("0") or "0"
            found.add(f"{whole}.{reduced}")
            # also the concatenated (hundreds-style) integer reading, since
            # spoken axis values like "axis one seventy" (= 170 degrees) use
            # the identical WORD WORD surface pattern as spoken diopter
            # decimals like "minus four twenty-five" (= -4.25). We can't
            # disambiguate from the words alone, so emit both candidate
            # readings and let the numeric-token check accept either.
            found.add(f"{whole}{frac_val:02d}")

    return found


# ---------------------------------------------------------------------------
# Correction-awareness: superseded-value detection
#
# WHY: the mock generator is deliberately naive on self-correcting
# transcripts — its first-pass drafter can pick up the FIRST-stated value in
# a dictation correction ("minus three seventy-five, uh, sorry, minus three
# fifty") rather than the corrected final value. That's the intended failure
# mode the harness exists to catch (see specs/INTERFACES.md generator.py
# docstring). For the judge to actually punish it, we need to know, from the
# transcript alone, which numeric values were explicitly superseded by a
# later correction — then flag any generated line that asserts a superseded
# value as a hallucination ("asserted superseded value"), even though the
# literal digits *do* appear somewhere in the transcript (so the plain
# numeric-token support check alone would wrongly treat it as supported).
# ---------------------------------------------------------------------------

# A short numeric-ish phrase: an optional "minus", then either literal
# digits or a run of 1-3 number words (covering compounds like "one
# hundred", "one twenty-five", "eight point six", "nine point"), used as the
# building block for both correction patterns below.
_NUM_WORD_ALT = (
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|point"
)
_NUM_PHRASE = (
    r"(?:minus\s+)?(?:[+-]?\d[\d.]*"
    r"|(?:" + _NUM_WORD_ALT + r")(?:[\s-](?:" + _NUM_WORD_ALT + r"))"
    r"{0,2})"
)

# Inline self-correction: "<value> — sorry/actually/no, <value>" or
# "<value>, sorry, <value>" etc. Captures a numeric-ish phrase before and
# after a correction cue. Deliberately generic (content-blind, no
# per-transcript special-casing) — it just looks for the standard
# dictation-correction surface pattern, and requires both sides to actually
# look like numbers (so it can't latch onto unrelated prose around "sorry"/
# "actually" that doesn't involve a value correction at all).
_INLINE_CORRECTION_RE = re.compile(
    r"(?P<first>" + _NUM_PHRASE + r")"
    r"\s*(?:—|-{1,2}|,)?\s*(?:uh,?\s*)?"
    r"(?:sorry|actually|let me correct that|no)\b[,]?\s*"
    r"(?:for this brand it's\s*)?(?:let me read it properly,?\s*)?"
    r"(?P<second>" + _NUM_PHRASE + r")",
    re.IGNORECASE,
)

# Explicit correction dialogue: "moved/nudged/bumped ... from X to Y",
# "changed from X to Y", "went from X to Y". Both sides must look numeric.
_FROM_TO_CORRECTION_RE = re.compile(
    r"\bfrom\s+(?P<first>" + _NUM_PHRASE + r")"
    r"\s+to\s+(?P<second>" + _NUM_PHRASE + r")",
    re.IGNORECASE,
)


_HUNDRED_PHRASE_RE = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)\s+hundred\b",
    re.IGNORECASE,
)


def _numeric_tokens_from_phrase(phrase: str) -> set[str]:
    """Extract digit-string tokens from a short phrase, covering both
    literal digits and spelled-out number words (via the same word-to-digit
    fallback used elsewhere in this module), plus "<word> hundred" spoken
    diopter-style readings (e.g. "one hundred" -> both '100' and '1.00',
    since dictated diopter powers like "minus one hundred" mean -1.00, while
    plain counts elsewhere in the module use the '100' integer reading;
    correction-matching only needs the tokens to be internally consistent
    between the two sides of a correction, so emitting both is safe)."""
    tokens: set[str] = set()
    for m in re.finditer(r"[+-]?\d+(?:\.\d+)?", phrase):
        tokens.add(m.group(0).lstrip("+-"))
    tokens |= _spelled_numbers_to_digits(phrase)
    for m in _HUNDRED_PHRASE_RE.finditer(phrase):
        whole = _ONES.get(m.group(1).lower())
        if whole is not None:
            tokens.add(f"{whole}00")
            tokens.add(f"{whole}.00")
    return tokens


def build_corrected_value_map(transcript_text: str) -> dict[str, set[str]]:
    """Scan `transcript_text` for self-correction dialogue and return a map
    of {superseded_token: {corrected_token, ...}} — pre-correction numeric
    values paired with the final value(s) that replaced them. Purely
    pattern-based on the transcript text itself; no transcript-ID
    special-casing.

    A pair is only recorded if the "before" and "after" sides of the
    correction actually disagree (a correction that restates the same
    number, e.g. clarifying phrasing without changing the value, supersedes
    nothing). Returning the corrected value alongside each stale token (not
    just a flat superseded set) lets callers distinguish "note asserts the
    stale value as if final" (punish) from "note documents the correction
    itself, e.g. '-5.25 -> -5.50'" (legitimate audit trail, matches golden
    note style — see contactlens_03 golden P line) — a line that carries
    BOTH the stale and the corrected token together is not hallucinating,
    it's citing the correction.
    """
    transcript_text = transcript_text or ""
    superseded: dict[str, set[str]] = {}

    for pattern in (_INLINE_CORRECTION_RE, _FROM_TO_CORRECTION_RE):
        for m in pattern.finditer(transcript_text):
            first_tokens = _numeric_tokens_from_phrase(m.group("first"))
            second_tokens = _numeric_tokens_from_phrase(m.group("second"))
            if not first_tokens or not second_tokens:
                continue
            stale = first_tokens - second_tokens
            for tok in stale:
                superseded.setdefault(tok, set()).update(second_tokens)

    return superseded


# ---------------------------------------------------------------------------
# Alignment (shared helper used by completeness + hallucination)
# ---------------------------------------------------------------------------

def _all_lines(soap: dict) -> list[tuple[str, int, dict]]:
    """Flatten a soap dict into (section, index, line_dict) tuples in
    S,O,A,P / list order."""
    out = []
    for section in SOAP_SECTIONS:
        lines = soap.get(section) or []
        for idx, line in enumerate(lines):
            if isinstance(line, dict):
                out.append((section, idx, line))
    return out


def _line_text(line: dict) -> str:
    return line.get("text", "") if isinstance(line, dict) else ""


def _ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _alignment_score(g_text: str, gold_text: str) -> tuple[bool, float]:
    """Semantic alignment test between one generated line and one golden
    line. Returns (is_aligned, score) where score is used only to rank
    candidates for the greedy assignment below (higher = better match).

    Primary signal: content-word Jaccard (stopword-stripped, lowercase,
    lemma-ish) >= _JACCARD_ALIGN_THRESHOLD, OR shared key numeric clinical
    facts (VA/IOP/diopter/axis/BC-DIA) between the two lines — either is
    sufficient on its own, since a line can be a legitimate paraphrase with
    low lexical overlap but identical numbers (or vice versa: same
    descriptive vocabulary summarizing a finding with the number dropped).
    A high literal difflib ratio (near-verbatim reuse) also counts, as a
    fallback for short/degenerate lines where word-set overlap is too coarse
    to be meaningful.
    """
    g_words = _content_words(g_text)
    gold_words = _content_words(gold_text)
    content_jaccard = _jaccard(g_words, gold_words)
    shared_word_count = len(g_words & gold_words)
    numeric_agree = _numeric_facts_agree(g_text, gold_text)
    char_ratio = _ratio(g_text, gold_text)

    jaccard_aligned = (
        content_jaccard >= _JACCARD_ALIGN_THRESHOLD
        and shared_word_count >= _JACCARD_MIN_SHARED_WORDS
    )
    aligned = jaccard_aligned or numeric_agree or char_ratio >= _ALIGN_THRESHOLD
    # Composite score for greedy ranking: numeric agreement is the strongest
    # signal (exact clinical fact match), then Jaccard, then char ratio as a
    # tie-break floor.
    score = max(content_jaccard, char_ratio * 0.9)
    if numeric_agree:
        score = max(score, 0.9)
    return aligned, score


def align_notes(generated: dict, golden: dict) -> dict:
    """Greedily align generated SOAP lines to golden SOAP lines by semantic
    content match (content-word Jaccard combined with numeric-fact
    agreement; see _alignment_score), across ALL sections (not just same-
    section), one-to-one.

    Returns a dict:
        {
          "gen_to_gold": {(section, idx): (gold_section, gold_idx, ratio, same_section)},
          "matched_gold_keys": set of (gold_section, gold_idx) that got matched,
          "gen_lines": list of (section, idx, line) for generated note,
          "gold_lines": list of (section, idx, line) for golden note,
        }
    """
    gen_lines = _all_lines(generated.get("soap", {}) if generated else {})
    gold_lines = _all_lines(golden.get("soap", {}) if golden else {})

    candidates = []
    for g_section, g_idx, g_line in gen_lines:
        g_text = _line_text(g_line)
        for gold_section, gold_idx, gold_line in gold_lines:
            gold_text = _line_text(gold_line)
            aligned, score = _alignment_score(g_text, gold_text)
            if aligned:
                candidates.append(
                    ((g_section, g_idx), (gold_section, gold_idx), score)
                )

    # Greedy assignment: highest score first, skip already-used keys.
    candidates.sort(key=lambda c: c[2], reverse=True)
    used_gen: set = set()
    used_gold: set = set()
    gen_to_gold: dict = {}
    for gen_key, gold_key, ratio in candidates:
        if gen_key in used_gen or gold_key in used_gold:
            continue
        used_gen.add(gen_key)
        used_gold.add(gold_key)
        same_section = gen_key[0] == gold_key[0]
        gen_to_gold[gen_key] = (gold_key[0], gold_key[1], ratio, same_section)

    return {
        "gen_to_gold": gen_to_gold,
        "matched_gold_keys": used_gold,
        "gen_lines": gen_lines,
        "gold_lines": gold_lines,
    }


# ---------------------------------------------------------------------------
# 1. completeness
# ---------------------------------------------------------------------------

def _anchor_from_fraction(fraction: float) -> int:
    """Map matched/total-golden-lines weighted fraction to a 1-5 anchor.

    Calibration note: the alignment fraction is a LINE-level coverage proxy.
    A generated note built by deleting half of a golden note's own lines
    verbatim (fraction ~0.5-0.6, exact-text matches on the surviving half)
    must still land <=3 per rubric.yaml ("several material findings
    missing... clinical picture is distorted"), so the breakpoints can't be
    pushed arbitrarily low just to flatter a naive extractive drafter — the
    fidelity gap for real (non-verbatim) generated notes needs to be closed
    on the generator side (numeric-fact canonicalization, section density)
    rather than papered over here. These breakpoints are modestly softened
    from a stricter near-verbatim-only curve to account for legitimate
    paraphrase/fragmentation drift, not to hide missing content.
    """
    if fraction >= 0.75:
        return 5
    if fraction > 0.60:
        return 4
    if fraction >= 0.40:
        return 3
    if fraction >= 0.20:
        return 2
    return 1


def score_completeness(alignment: dict) -> tuple[int, str]:
    gold_lines = alignment["gold_lines"]
    total_gold = len(gold_lines)
    if total_gold == 0:
        return 5, "no golden lines to compare against (empty golden note); treated as complete."

    gen_to_gold = alignment["gen_to_gold"]

    # weight: same-section match = 1.0, cross-section match = 0.5
    weighted = 0.0
    cross_section_count = 0
    matched_gold_keys = set()
    for gen_key, (gold_section, gold_idx, ratio, same_section) in gen_to_gold.items():
        gold_key = (gold_section, gold_idx)
        if gold_key in matched_gold_keys:
            continue
        matched_gold_keys.add(gold_key)
        if same_section:
            weighted += 1.0
        else:
            weighted += 0.5
            cross_section_count += 1

    fraction = weighted / total_gold
    score = _anchor_from_fraction(fraction)
    # Any cross-section (wrong-section) placement means a clinically material
    # item was misfiled — rubric.yaml explicitly calls this out as something
    # to penalize ("e.g., an objective IOP reading written into the
    # Subjective section"), distinct from a same-section paraphrase match.
    # A single cross-section hit can round up to the same fraction-anchor as
    # a perfect note (e.g. 9.5/10 == 0.95, the same >=0.95 bucket as a fully
    # correct 10/10), so we explicitly cap the score below the top anchor
    # whenever at least one cross-section match occurred.
    if cross_section_count > 0 and score >= 5:
        score = 4

    matched_count = len(matched_gold_keys)
    missed_count = total_gold - matched_count

    if missed_count == 0 and cross_section_count == 0:
        rationale = (
            f"matched {matched_count}/{total_gold} golden lines "
            "(all same-section); nothing material missing."
        )
    else:
        parts = [f"matched {matched_count}/{total_gold} golden lines"]
        if cross_section_count:
            parts.append(f"{cross_section_count} cross-section")
        rationale = ", ".join(parts)
        if missed_count:
            # identify a representative missed line for a concrete rationale
            missed_keys = [
                (s, i) for (s, i, _l) in gold_lines if (s, i) not in matched_gold_keys
            ]
            if missed_keys:
                m_section, m_idx = missed_keys[0]
                missed_line = gold_lines[
                    [k[:2] for k in [(s, i) for (s, i, _l) in gold_lines]].index(
                        (m_section, m_idx)
                    )
                ][2]
                snippet = _line_text(missed_line)[:60]
                rationale += f"; missed e.g. \"{snippet}\" ({m_section})."
            else:
                rationale += "."
        else:
            rationale += "."

    return score, rationale


# ---------------------------------------------------------------------------
# 2. hallucination (inverse-scored)
# ---------------------------------------------------------------------------

# VA fractions: 20/25, 20/400, 6/9 style.
_VA_TOKEN_RE = re.compile(r"\b(20|6)/(\d{1,3})\b")

# IOP-context integers near IOP/mmHg cue (5-60 range considered plausible
# tokens to verify, but we check the literal token regardless of range —
# range plausibility is a terminology concern, not a hallucination concern).
_IOP_CONTEXT_RE = re.compile(r"\b(IOP|mmHg)\b", re.IGNORECASE)
_IOP_NUMBER_RE = re.compile(r"\b(\d{1,3})\b")

# Diopter / sphere / cylinder signed decimals, e.g. -2.25, +2.00.
_DIOPTER_RE = re.compile(r"([+-]\d{1,2}(?:\.\d{1,2})?)")

# Axis values: integer 1-180 after "axis" or "x".
_AXIS_RE = re.compile(r"\b(?:axis\s*|x\s*)(\d{1,3})\b", re.IGNORECASE)

# Base curve / diameter values (contact-lens fitting), e.g. "8.6" in "base
# curve 8.6 mm" or "14.2" in "diameter 14.2 mm". Used by the semantic
# alignment numeric-facts check (_numeric_facts) so CL-fitting lines with
# matching BC/DIA values align even when phrasing differs.
_BC_DIA_RE = re.compile(r"\b(\d{1,2}\.\d{1,2})\b")


def _extract_numeric_tokens(text: str) -> list[str]:
    """Extract key numeric tokens (as plain digit strings, denominators
    included for VA) from a line's text per the hallucination spec:
    VA fractions, IOP-context integers, diopter/sphere/cylinder decimals,
    axis values."""
    tokens: list[str] = []

    for m in _VA_TOKEN_RE.finditer(text):
        tokens.append(m.group(1))
        tokens.append(m.group(2))

    if _IOP_CONTEXT_RE.search(text):
        for m in _IOP_NUMBER_RE.finditer(text):
            val = int(m.group(1))
            if 5 <= val <= 60:
                tokens.append(m.group(1))

    for m in _DIOPTER_RE.finditer(text):
        tokens.append(m.group(1))

    for m in _AXIS_RE.finditer(text):
        tokens.append(m.group(1))

    return tokens


def _token_supported(token: str, transcript_text: str, transcript_number_words: set[str]) -> bool:
    """A numeric token is supported if it literally appears in the
    transcript as digits, OR its spelled-out-word form appears (via the
    word-to-digit fallback mapping) — see module docstring for why this
    fallback is necessary."""
    # normalize token: strip leading '+'/'-' sign for comparisons (diopter
    # '+2.00'/'-2.00' digit search should still look for '2.00' or '2' in
    # transcript text/word-mapping, since transcripts speak magnitudes like
    # "minus three seventy-five" without literal sign punctuation).
    bare = token.lstrip("+-")
    if bare in transcript_text or token in transcript_text:
        return True
    # try integer-only form (e.g. '2.00' -> '2') against digits and words
    try:
        as_float = float(bare)
        if as_float == int(as_float):
            int_str = str(int(as_float))
            if int_str in transcript_text or int_str in transcript_number_words:
                return True
    except ValueError:
        pass
    return bare in transcript_number_words or token in transcript_number_words


def score_hallucination(
    generated: dict, transcript_text: str, alignment: dict
) -> tuple[int, str]:
    gen_lines = alignment["gen_lines"]
    gen_to_gold = alignment["gen_to_gold"]
    transcript_text = transcript_text or ""
    transcript_number_words = _spelled_numbers_to_digits(transcript_text)
    superseded_map = build_corrected_value_map(transcript_text)

    unsupported_numeric_lines = []  # (section, idx, tokens)
    unsupported_lines = []  # (section, idx) with no span/text support at all
    superseded_value_lines = []  # (section, idx, tokens) — asserted a corrected-away value

    for section, idx, line in gen_lines:
        text = _line_text(line)
        spans = line.get("spans") or [] if isinstance(line, dict) else []

        # Span support: concatenate transcript_text[start:end] for each span,
        # check overlap with the line's own text. We use word-set (Jaccard)
        # overlap of "significant" words rather than a raw character
        # SequenceMatcher ratio: golden/generated note lines routinely
        # compress verbose spoken prose into clinical shorthand (e.g. the
        # transcript span "I'll take the right to minus five fifty final...
        # left stays minus five point zero zero" vs. the note line
        # "Prescribe daily disposable soft CL: OD -5.50 ... OS -5.00") which
        # drives character-ratio as low as ~0.14 even for fully faithful,
        # transcript-supported golden content. Shared significant vocabulary
        # is a much more robust faithfulness signal for this style of
        # compression than literal character alignment.
        span_text_parts = []
        for span in spans:
            if (
                isinstance(span, (list, tuple))
                and len(span) == 2
                and isinstance(span[0], int)
                and isinstance(span[1], int)
                and 0 <= span[0] <= span[1] <= len(transcript_text)
            ):
                span_text_parts.append(transcript_text[span[0]:span[1]])
        span_text = " ".join(span_text_parts)
        if not span_text:
            has_span_support = False
        else:
            char_ratio = _ratio(span_text, text)
            span_words = _significant_words(span_text)
            line_words = _significant_words(text)
            if line_words:
                word_overlap = len(span_words & line_words) / len(line_words)
            else:
                word_overlap = 0.0
            has_span_support = (
                char_ratio >= _SPAN_SUPPORT_THRESHOLD or word_overlap >= _SPAN_WORD_OVERLAP_THRESHOLD
            )

        # Numeric token support check.
        numeric_tokens = _extract_numeric_tokens(text)
        fabricated_tokens = [
            tok for tok in numeric_tokens
            if not _token_supported(tok, transcript_text, transcript_number_words)
        ]

        # Correction-awareness: a numeric token that IS literally present
        # somewhere in the transcript (so it passes the plain support check
        # above) can still be clinically wrong if the transcript explicitly
        # superseded it with a corrected value later in the same dictation
        # (see build_corrected_value_map). Asserting the stale, corrected-
        # away value is exactly the naive "first-stated value" failure mode
        # the mock generator is seeded to sometimes produce on messy
        # transcripts — catch it here as its own hallucination category
        # (distinct from plain fabrication, since the digits DO appear in
        # the transcript, just not as the final clinical value).
        #
        # Exemptions (either is sufficient to NOT flag a token):
        #  1. The SAME line documents the correction as an explicit
        #     AUDIT-TRAIL statement (an arrow/"corrected to"/"moved ... to"/
        #     "changed to" marker between the stale and corrected values,
        #     e.g. golden style "-5.25 -> -5.50" or "(initially misspoken
        #     '-1.70' then corrected to -1.50)") — that is a deliberate,
        #     legible record of the correction, not a hallucination. This is
        #     narrower than "both values merely appear somewhere in the
        #     line": the naive mock drafter frequently parrots raw dictation
        #     disfluency verbatim ("-3.75, sorry, -3.50 sphere") which DOES
        #     leave both numbers in the line but WITHOUT resolving which one
        #     is final — that ambiguous juxtaposition is exactly the unsafe
        #     leakage this check exists to catch, so it must NOT be exempted
        #     just because the corrected value happens to also be present.
        #  2. The generated line's ALIGNED golden counterpart (per
        #     align_notes) asserts this same "stale" value WITHOUT pairing
        #     it with an audit-trail marker to its corrected sibling —
        #     signature of the value being a genuinely distinct,
        #     legitimately-documented fact in golden (e.g. contactlens_03's
        #     prior spectacle Rx "-5.25", documented as history, not as a
        #     re-litigated correction).
        _AUDIT_TRAIL_MARKER_RE = re.compile(
            r"->|—>|\bcorrected to\b|\bmoved\b.{0,20}\bto\b|\bchanged to\b|"
            r"\bthen corrected\b|\binitially misspoken\b",
            re.IGNORECASE,
        )

        def _has_audit_trail_marker(line_text: str) -> bool:
            return bool(_AUDIT_TRAIL_MARKER_RE.search(line_text))

        line_bare_tokens = {tok.lstrip("+-") for tok in numeric_tokens}
        gold_match = gen_to_gold.get((section, idx))
        aligned_gold_text = ""
        if gold_match:
            gold_section, gold_idx = gold_match[0], gold_match[1]
            gold_lines_list = alignment["gold_lines"]
            for gs, gi, gline in gold_lines_list:
                if gs == gold_section and gi == gold_idx:
                    aligned_gold_text = _line_text(gline)
                    break
        aligned_gold_tokens = {
            tok.lstrip("+-") for tok in _extract_numeric_tokens(aligned_gold_text)
        }

        superseded_tokens = []
        for tok in numeric_tokens:
            if tok in fabricated_tokens:
                continue
            bare = tok.lstrip("+-")
            corrected_by = superseded_map.get(bare)
            if not corrected_by:
                continue
            if corrected_by & line_bare_tokens and _has_audit_trail_marker(text):
                continue  # explicit audit-trail line -> legitimate, not an error
            if (
                bare in aligned_gold_tokens
                and not (corrected_by & aligned_gold_tokens and _has_audit_trail_marker(aligned_gold_text))
            ):
                continue  # golden line asserts this value as a standalone fact, not a re-litigated correction
            superseded_tokens.append(tok)

        # A line dense with clinical shorthand (e.g. "Over-refraction OD:
        # -0.25 improves from 20/25 to 20/20; OS 20/20 as is.") can have
        # near-zero prose word overlap with its cited transcript span even
        # though every one of its numeric claims is verified. When the line
        # carries numeric tokens and none are fabricated, treat that as
        # sufficient support in its own right rather than also requiring
        # word-level span overlap — the numeric check is the more precise
        # signal for this kind of line.
        if numeric_tokens and not fabricated_tokens:
            has_span_support = True

        if fabricated_tokens:
            unsupported_numeric_lines.append((section, idx, fabricated_tokens))
        elif superseded_tokens:
            superseded_value_lines.append((section, idx, superseded_tokens))
        elif not has_span_support:
            unsupported_lines.append((section, idx))

    unsupported_numeric_count = len(unsupported_numeric_lines)
    unsupported_line_count = len(unsupported_lines)
    superseded_count = len(superseded_value_lines)

    # Superseded-value assertions are treated with the same severity weight
    # as fabricated numeric values for scoring purposes: a specific,
    # confidently-stated wrong number is the safety-critical failure mode
    # the hallucination dimension exists to catch, per rubric.yaml, whether
    # the number was never spoken at all (fabricated) or was spoken and then
    # explicitly corrected away (superseded) — either way the note asserts a
    # clinical value the transcript does not support as final/true.
    severe_numeric_count = unsupported_numeric_count + superseded_count

    if severe_numeric_count == 0 and unsupported_line_count == 0:
        score = 5
    elif severe_numeric_count == 0 and unsupported_line_count == 1:
        score = 4
    elif severe_numeric_count == 0 and unsupported_line_count >= 2:
        score = 3
    elif severe_numeric_count == 1:
        score = 2
    else:  # severe_numeric_count >= 2
        score = 1

    if score == 5:
        rationale = "every generated line is transcript-supported; no fabricated numeric values."
    elif unsupported_numeric_count >= 1 and superseded_count >= 1:
        section, idx, tokens = superseded_value_lines[0]
        line_text = _line_text(gen_lines[[(s, i) for (s, i, _l) in gen_lines].index((section, idx))][2])
        rationale = (
            f"{unsupported_numeric_count} fabricated + {superseded_count} superseded-value "
            f"numeric issue(s) (e.g. asserted superseded value '{tokens[0]}' in "
            f"\"{line_text[:60]}\" — transcript explicitly corrected this value)."
        )
    elif superseded_count >= 1:
        section, idx, tokens = superseded_value_lines[0]
        line_text = _line_text(gen_lines[[(s, i) for (s, i, _l) in gen_lines].index((section, idx))][2])
        rationale = (
            f"{superseded_count} line(s) assert a superseded value "
            f"(e.g. '{tokens[0]}' in \"{line_text[:60]}\" was corrected in the "
            "transcript dictation but the note kept the pre-correction number); "
            + ("all other lines transcript-supported." if superseded_count == 1 else "")
        )
    elif unsupported_numeric_count >= 1:
        section, idx, tokens = unsupported_numeric_lines[0]
        line_text = _line_text(gen_lines[[(s, i) for (s, i, _l) in gen_lines].index((section, idx))][2])
        rationale = (
            f"{unsupported_numeric_count} fabricated numeric value(s) "
            f"(e.g. '{tokens[0]}' in \"{line_text[:60]}\" not found in transcript, "
            "incl. spelled-out-number check); "
            + ("all other lines transcript-supported." if unsupported_numeric_count == 1 else "")
        )
    else:
        rationale = (
            f"{unsupported_line_count} generated line(s) lack transcript span/text support "
            "(no fabricated numbers, but unverifiable content)."
        )

    return score, rationale


# ---------------------------------------------------------------------------
# 3. terminology
# ---------------------------------------------------------------------------

_PLAUSIBLE_SNELLEN_DENOMS = {"16", "20", "25", "30", "40", "50", "60", "70", "80", "100", "200", "400"}
_PLAUSIBLE_METRIC_PAIRS = {"6/6", "6/7.5", "6/9", "6/12", "6/18", "6/24", "6/30", "6/60"}

_LOWERCASE_LATERALITY_RE = re.compile(r"\b(od|os|ou)\b(?!\.)")
_MALFORMED_LATERALITY_RE = re.compile(r"\bO\.\s*D\.|\bO\.\s*S\.|\bO\.\s*U\.", re.IGNORECASE)


def _fallback_terminology_violations(generated: dict) -> list[dict]:
    """Internal regex-based terminology checker, used when
    scribegate.normalizer.check_note is unavailable (current state — the
    normalizer module is a stub). Mirrors the spirit of terminology.yaml:
    VA denominator plausibility, IOP range sanity, laterality casing, axis
    range."""
    violations: list[dict] = []
    soap = generated.get("soap", {}) if generated else {}

    for section, idx, line in _all_lines(soap):
        text = _line_text(line)
        if not text:
            continue

        # VA format sanity
        for m in re.finditer(r"\b(\d{1,3})\s*/\s*(\d{1,3}(?:\.\d+)?)\b", text):
            num, den = m.group(1), m.group(2)
            frac = f"{num}/{den}"
            if num == "20" and den not in _PLUS_DENOMS_SNELLEN_TWENTY:
                violations.append({"code": "VA_FORMAT", "severity": "warn",
                                    "message": f"implausible Snellen denominator in '{frac}'",
                                    "line_text": text})
            elif num == "6" and frac not in _PLAUSIBLE_METRIC_PAIRS:
                violations.append({"code": "VA_FORMAT", "severity": "warn",
                                    "message": f"implausible metric VA '{frac}'",
                                    "line_text": text})

        # IOP unit/range sanity
        if re.search(r"\b(IOP|mmHg)\b", text, re.IGNORECASE):
            for m in re.finditer(r"(\d{1,3}(?:\.\d+)?)\s*mmHg", text, re.IGNORECASE):
                val = float(m.group(1))
                if val < 3 or val > 60:
                    violations.append({"code": "IOP_RANGE", "severity": "error",
                                        "message": f"IOP value {val:g} mmHg is out of plausible range",
                                        "line_text": text})

        # Laterality cosmetic issues: lowercase od/os/ou, or "O.D."/"O.S."/"O.U."
        for m in _LOWERCASE_LATERALITY_RE.finditer(text):
            violations.append({"code": "LATERALITY_CONFLICT", "severity": "warn",
                                "message": f"lowercase laterality token '{m.group(0)}' should be uppercase",
                                "line_text": text})
        for m in _MALFORMED_LATERALITY_RE.finditer(text):
            violations.append({"code": "LATERALITY_CONFLICT", "severity": "warn",
                                "message": f"malformed laterality token '{m.group(0)}'",
                                "line_text": text})

        # Axis range
        for m in re.finditer(r"\bx\s*(\d{1,3})\b", text, re.IGNORECASE):
            axis = int(m.group(1))
            if axis < 1 or axis > 180:
                violations.append({"code": "AXIS_RANGE", "severity": "error",
                                    "message": f"axis value {axis} out of range 1-180",
                                    "line_text": text})

    return violations


_PLUS_DENOMS_SNELLEN_TWENTY = _PLAUSIBLE_SNELLEN_DENOMS


def _violation_severity(v) -> str | None:
    if isinstance(v, dict):
        return v.get("severity")
    return getattr(v, "severity", None)


def _violation_code(v) -> str:
    if isinstance(v, dict):
        return v.get("code", "?")
    return getattr(v, "code", "?")


def score_terminology(generated: dict, transcript_text: str) -> tuple[int, str]:
    violations = None
    used_normalizer = False
    try:
        from scribegate.normalizer import check_note  # defensive import

        try:
            violations = check_note(generated, transcript_text)
            if not isinstance(violations, list):
                violations = None
            else:
                used_normalizer = True
        except Exception:
            violations = None
    except (ImportError, AttributeError):
        violations = None
    except Exception:
        violations = None

    if violations is None:
        violations = _fallback_terminology_violations(generated)
        used_normalizer = False

    errors = 0
    warns = 0
    for v in violations:
        sev = _violation_severity(v)
        if sev == "error":
            errors += 1
        elif sev == "warn":
            warns += 1

    v_weighted = errors * 2 + warns * 1

    # NOTE on tolerance calibration: warn-severity violations (e.g. IOP_RANGE
    # "elevated" flags on a genuinely, correctly-documented high pressure
    # reading) are informational clinical flags, not notation/formatting
    # *errors* — per rubric.yaml's own anchor language, "cosmetic
    # inconsistency with no meaning change" still merits a 5. We therefore
    # only start deducting once weighted severity exceeds a small tolerance
    # (covers up to 2 warns with zero errors — what a correctly-documented
    # elevated-IOP golden line produces), or as soon as any error-severity
    # violation is present.
    if errors == 0 and v_weighted <= 2:
        score = 5
    elif v_weighted <= 2:
        score = 4
    elif v_weighted <= 4:
        score = 3
    elif v_weighted <= 6:
        score = 2
    else:
        score = 1

    source = "normalizer" if used_normalizer else "fallback checker"
    if v_weighted == 0:
        rationale = f"no terminology issues found ({source})."
    else:
        codes = sorted({_violation_code(v) for v in violations})
        rationale = (
            f"{source} flagged {errors} error(s) + {warns} warn(s) "
            f"({', '.join(codes[:3])})."
        )

    return score, rationale


# ---------------------------------------------------------------------------
# 4. coding_plausibility
# ---------------------------------------------------------------------------


def _total_line_len(soap: dict) -> int:
    return sum(len(_line_text(line)) for (_s, _i, line) in _all_lines(soap))


def score_coding_plausibility(generated: dict, golden: dict) -> tuple[int, str]:
    soap = generated.get("soap", {}) if generated else {}
    checks_passed = 0
    reasons = []
    failures = []

    # Check 1: all four sections non-empty
    sections_nonempty = all(bool(soap.get(s)) for s in SOAP_SECTIONS)
    if sections_nonempty:
        checks_passed += 1
        reasons.append("all 4 SOAP sections populated")
    else:
        empty_sections = [s for s in SOAP_SECTIONS if not soap.get(s)]
        failures.append(f"empty section(s): {', '.join(empty_sections)}")

    # Check 2: at least one P line references content/terms also in A or O
    p_lines = soap.get("P") or []
    ao_words: set[str] = set()
    for section in ("A", "O"):
        for line in soap.get(section) or []:
            ao_words |= _significant_words(_line_text(line))
    plan_references_assessment = False
    for line in p_lines:
        p_words = _significant_words(_line_text(line))
        if p_words & ao_words:
            plan_references_assessment = True
            break
    if plan_references_assessment:
        checks_passed += 1
        reasons.append("plan references assessment/objective content")
    else:
        failures.append("plan does not reference assessment/objective content")

    # Check 3: total line-text length within 0.5x-2x of golden's total length
    gen_len = _total_line_len(soap)
    gold_len = _total_line_len(golden.get("soap", {}) if golden else {})
    if gold_len == 0:
        length_in_band = True
    else:
        ratio = gen_len / gold_len
        length_in_band = 0.5 <= ratio <= 2.0
    if length_in_band:
        checks_passed += 1
        reasons.append("length in band vs golden")
    else:
        failures.append("length far outside golden's band")

    # Check 4: no copy-forward duplicates (near-identical lines within note)
    all_texts = [_line_text(line) for (_s, _i, line) in _all_lines(soap) if _line_text(line)]
    has_duplicate = False
    for i in range(len(all_texts)):
        for j in range(i + 1, len(all_texts)):
            if _ratio(all_texts[i], all_texts[j]) >= _DUPLICATE_THRESHOLD:
                has_duplicate = True
                break
        if has_duplicate:
            break
    if not has_duplicate:
        checks_passed += 1
        reasons.append("no copy-forward duplicates detected")
    else:
        failures.append("copy-forward duplicate lines detected")

    score_map = {4: 5, 3: 4, 2: 3, 1: 2, 0: 1}
    score = score_map[checks_passed]

    if checks_passed == 4:
        rationale = "; ".join(reasons) + "."
    else:
        rationale = f"scored {checks_passed}/4 checks; " + "; ".join(failures) + "."

    return score, rationale


# ---------------------------------------------------------------------------
# Aggregate + public entry point (deterministic mock judge)
# ---------------------------------------------------------------------------

def _aggregate(scores: dict) -> float:
    mean_score = sum(scores.values()) / len(scores)
    return (mean_score - 1) / 4


def _mock_judge_note(generated: dict, golden: dict, transcript_text: str) -> dict:
    generated = generated or {}
    golden = golden or {}
    transcript_text = transcript_text or ""

    alignment = align_notes(generated, golden)

    completeness_score, completeness_rationale = score_completeness(alignment)
    hallucination_score, hallucination_rationale = score_hallucination(
        generated, transcript_text, alignment
    )
    terminology_score, terminology_rationale = score_terminology(generated, transcript_text)
    coding_score, coding_rationale = score_coding_plausibility(generated, golden)

    scores = {
        "completeness": completeness_score,
        "hallucination": hallucination_score,
        "coding_plausibility": coding_score,
        "terminology": terminology_score,
    }
    rationales = {
        "completeness": completeness_rationale,
        "hallucination": hallucination_rationale,
        "coding_plausibility": coding_rationale,
        "terminology": terminology_rationale,
    }

    return {
        "scores": scores,
        "aggregate": _aggregate(scores),
        "rationales": rationales,
    }


# ---------------------------------------------------------------------------
# API judge (Haiku) — env-gated, inert by default
# ---------------------------------------------------------------------------

class APIJudge:
    """Claude Haiku-backed judge. Only ever constructed/used by `judge_note`
    when SCRIBEGATE_USE_API=1 and ANTHROPIC_API_KEY are both set. `anthropic`
    is imported lazily inside `judge` (never at module load) so this module
    never requires the package or network access in the default path.
    """

    def __init__(self, model: str = "claude-haiku-4-5"):
        self.model = model

    def _client(self):
        import anthropic  # lazy import — never at module load

        return anthropic.Anthropic()

    def _load_rubric_anchors(self) -> dict:
        """Load specs/rubric.yaml dimension anchors for prompt-building."""
        try:
            import yaml
        except ImportError:
            return {}
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "specs",
            "rubric.yaml",
        )
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if isinstance(data, dict):
                return data.get("dimensions", {}) or {}
        except Exception:
            return {}
        return {}

    def _build_prompt(self, generated: dict, golden: dict, transcript_text: str) -> str:
        anchors = self._load_rubric_anchors()
        anchor_lines = []
        for dim in DIMENSIONS:
            dim_spec = anchors.get(dim, {}) if isinstance(anchors, dict) else {}
            desc = dim_spec.get("description", "") if isinstance(dim_spec, dict) else ""
            dim_anchors = dim_spec.get("anchors", {}) if isinstance(dim_spec, dict) else {}
            anchor_lines.append(f"### {dim}\n{desc}")
            for level in ("1", "2", "3", "4", "5"):
                if level in dim_anchors:
                    anchor_lines.append(f"  {level}: {dim_anchors[level]}")

        anchors_text = "\n".join(anchor_lines)
        return (
            "You are ScribeGate's judge. Score the GENERATED note against the "
            "GOLDEN reference note and the source TRANSCRIPT, using this rubric:\n\n"
            f"{anchors_text}\n\n"
            f"TRANSCRIPT:\n{transcript_text}\n\n"
            f"GOLDEN NOTE:\n{golden}\n\n"
            f"GENERATED NOTE:\n{generated}\n\n"
            "Return strict JSON: "
            '{"scores": {"completeness": int, "hallucination": int, '
            '"coding_plausibility": int, "terminology": int}, '
            '"rationales": {"completeness": str, "hallucination": str, '
            '"coding_plausibility": str, "terminology": str}}'
        )

    def judge(self, generated: dict, golden: dict, transcript_text: str) -> dict:
        import json

        client = self._client()
        prompt = self._build_prompt(generated, golden, transcript_text)
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        parsed = json.loads(text)
        scores = {dim: int(parsed["scores"][dim]) for dim in DIMENSIONS}
        rationales = {dim: str(parsed["rationales"][dim]) for dim in DIMENSIONS}
        return {
            "scores": scores,
            "aggregate": _aggregate(scores),
            "rationales": rationales,
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def judge_note(generated: dict, golden: dict, transcript_text: str) -> dict:
    """Judge a generated SOAP note against a golden reference + transcript.

    Single public entry point matching specs/INTERFACES.md. Checks the API
    env gate first; if SCRIBEGATE_USE_API=1 AND ANTHROPIC_API_KEY is set,
    delegates to APIJudge (Haiku). Otherwise always runs the deterministic,
    offline mock path — this is the default and the only path exercised in
    CI/tests (no network, no API key required).
    """
    use_api = os.environ.get("SCRIBEGATE_USE_API") == "1"
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if use_api and has_key:
        return APIJudge().judge(generated, golden, transcript_text)
    return _mock_judge_note(generated, golden, transcript_text)
