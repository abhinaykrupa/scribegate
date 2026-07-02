"""normalizer.py (T3) — terminology/notation validation and canonicalization.

Contract (specs/INTERFACES.md):
    check_line(text, transcript=None) -> list[Violation]
    check_note(note, transcript=None) -> list[Violation]
    normalize_line(text) -> str

Rules are sourced from specs/terminology.yaml, loaded once at module import
time. Loading is best-effort: if the file is missing, unreadable, or
malformed, the module falls back to built-in defaults rather than raising at
import time (per repo convention — this module must never crash on import).

stdlib + pyyaml only. Deterministic. No network. Python >= 3.10.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml is a declared dependency
    yaml = None  # type: ignore[assignment]


SOAP_SECTIONS = ("S", "O", "A", "P")

_TERMINOLOGY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "specs",
    "terminology.yaml",
)

# ---------------------------------------------------------------------------
# Built-in fallback defaults (used if specs/terminology.yaml can't be loaded)
# ---------------------------------------------------------------------------

_DEFAULT_SNELLEN_METRIC = [
    ("20/20", "6/6"),
    ("20/25", "6/7.5"),
    ("20/30", "6/9"),
    ("20/40", "6/12"),
    ("20/60", "6/18"),
    ("20/80", "6/24"),
    ("20/100", "6/30"),
    ("20/200", "6/60"),
]

_DEFAULT_IOP_HYPOTONY = 6.0       # < 6 -> error
_DEFAULT_IOP_WARN_LOW = 21.0      # >21 -> warn (up to 60)
_DEFAULT_IOP_WARN_HIGH = 60.0     # >60 -> error
_DEFAULT_SPHERE_MAX = 20.00
_DEFAULT_CYL_MAX_MAGNITUDE = 6.00


def _load_terminology(path: str = _TERMINOLOGY_PATH) -> dict:
    """Load specs/terminology.yaml once. Returns {} on any failure so the
    module can fall back to built-in defaults without crashing at import."""
    if yaml is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, dict):
            return data
    except (OSError, IOError, ValueError):
        return {}
    except Exception:  # pragma: no cover - defensive catch-all, never crash
        return {}
    return {}


def _build_snellen_metric_table(terminology: dict) -> list[tuple[str, str]]:
    try:
        entries = terminology["visual_acuity"]["snellen_metric_equivalents"]
        pairs = []
        for entry in entries:
            snellen = str(entry["snellen"]).strip()
            metric = str(entry["metric"]).strip()
            pairs.append((snellen, metric))
        if pairs:
            return pairs
    except (KeyError, TypeError, ValueError):
        pass
    return list(_DEFAULT_SNELLEN_METRIC)


_TERMINOLOGY = _load_terminology()
_SNELLEN_METRIC_PAIRS = _build_snellen_metric_table(_TERMINOLOGY)
_METRIC_TO_SNELLEN = {metric: snellen for snellen, metric in _SNELLEN_METRIC_PAIRS}
_SNELLEN_SET = {snellen for snellen, _ in _SNELLEN_METRIC_PAIRS}


# ---------------------------------------------------------------------------
# Violation dataclass
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    code: str          # e.g. "VA_FORMAT", "IOP_RANGE", "LATERALITY_CONFLICT", "CYL_SIGN", "AXIS_RANGE"
    severity: str      # "error" | "warn"
    message: str
    line_text: str


# ---------------------------------------------------------------------------
# Shared regex building blocks
# ---------------------------------------------------------------------------

# Laterality tokens as standalone words only (case-insensitive). \b already
# prevents matches inside "wood"/"good"/"mood"/"hood"/"period"/"rodent" etc.
# because those words don't contain "od"/"os"/"ou" as a boundary-delimited
# substring (e.g. "wood" = w-o-o-d; scanning for "od" requires an 'o'
# followed by a 'd' with a word boundary immediately before the 'o' -- in
# "wood" the char before the first 'o' is 'w', a word character, so no
# boundary exists there; the only 'od'-shaped substring would need index
# alignment that "wood" doesn't have since it's w,o,o,d not w,o,d).
_LATERALITY_RE = re.compile(r"\b(OD|OS|OU)\b", re.IGNORECASE)

_VA_CUE_RE = re.compile(r"\b(VA|acuity|vision|sc|cc|PH)\b", re.IGNORECASE)

# A generic N/M numeric fraction (used to find VA-looking or IOP-looking slash pairs).
_FRACTION_RE = re.compile(r"\b(\d{1,3}(?:\.\d+)?)\s*/\s*(\d{0,3}(?:\.\d+)?)\b")

# Malformed VA attempts: cue word followed by digits then a hyphen where a
# slash is expected, e.g. "VA 20-40".
_VA_HYPHEN_RE = re.compile(
    r"\b(?:VA|acuity|vision)\b[^\n]{0,10}?\b(20|6)-(\d{1,3})\b", re.IGNORECASE
)

# Low-vision scale tokens; always considered valid VA notation.
_LOW_VISION_RE = re.compile(r"\b(CF|HM|LP|NLP)\b")

# IOP-looking number: a number immediately preceding/following "mmHg", or a
# number near the word IOP / GAT / Tonopen / iCare / NCT / "pressure".
_IOP_CUE_RE = re.compile(
    r"\b(IOP|GAT|Tonopen|iCare|NCT|pressure[s]?)\b", re.IGNORECASE
)
_IOP_NUMBER_WITH_UNIT_RE = re.compile(
    r"(\d{1,3}(?:\.\d+)?)\s*mmHg", re.IGNORECASE
)
_IOP_NUMBER_NEAR_CUE_RE = re.compile(
    r"\b(?:IOP)\b\s*(?::)?\s*(\d{1,3}(?:\.\d+)?)", re.IGNORECASE
)

# lens_rx pattern: SPH CYL x AXIS, e.g. "-2.25 -0.75 x 090"
_LENS_RX_RE = re.compile(
    r"(?P<sph>[+-]\d{1,2}(?:\.\d{1,2})?)\s+"
    r"(?P<cyl>[+-]\d{1,2}(?:\.\d{1,2})?)\s*[xX]\s*"
    r"(?P<axis>\d{1,3})\b"
)

# lens_rx pattern with missing/non-numeric axis token, to detect "axis missing"
_LENS_RX_NO_AXIS_RE = re.compile(
    r"(?P<sph>[+-]\d{1,2}(?:\.\d{1,2})?)\s+"
    r"(?P<cyl>[+-]\d{1,2}(?:\.\d{1,2})?)\s*[xX]\s*"
    r"(?P<axis>\S*)"
)


# ---------------------------------------------------------------------------
# normalize_line
# ---------------------------------------------------------------------------

def _uppercase_laterality(text: str) -> str:
    return _LATERALITY_RE.sub(lambda m: m.group(1).upper(), text)


def _canonicalize_metric_va(text: str) -> str:
    """Replace exact metric VA fractions (e.g. '6/9') with their Snellen
    equivalent ('20/30') per the terminology table. Exact matches only."""

    def _repl(m: re.Match) -> str:
        num, den = m.group(1), m.group(2)
        if not den:
            return m.group(0)
        metric_key = f"{num}/{den}"
        snellen = _METRIC_TO_SNELLEN.get(metric_key)
        if snellen:
            return snellen
        return m.group(0)

    return _FRACTION_RE.sub(_repl, text)


def _ensure_iop_units(text: str) -> str:
    """Insert 'mmHg' right after an IOP-looking number that lacks units.

    Built via manual scan-and-splice (not re.sub) because the "does mmHg
    already follow?" check must look at the ORIGINAL text right after the
    matched number, not just within the regex match itself — using re.sub's
    replacement callback with only `m.group(0)` would miss units that exist
    just outside the match, causing duplicate insertions on re-normalization.
    """
    if not _IOP_CUE_RE.search(text):
        return text

    pieces: list[str] = []
    last_end = 0
    for m in _IOP_NUMBER_NEAR_CUE_RE.finditer(text):
        num_end = m.end(1)
        pieces.append(text[last_end:num_end])
        tail = text[num_end:]
        if not re.match(r"\s*mmHg", tail, re.IGNORECASE):
            pieces.append(" mmHg")
        last_end = num_end
    pieces.append(text[last_end:])
    return "".join(pieces)


def normalize_line(text: str) -> str:
    """Canonicalize a note line: VA (metric -> Snellen 20/x, exact matches
    only), IOP units (insert 'mmHg' if missing near an IOP cue), and OD/OS/OU
    casing (uppercase standalone laterality tokens only).

    Idempotent: normalize_line(normalize_line(x)) == normalize_line(x).
    """
    if not text:
        return text
    result = text
    result = _canonicalize_metric_va(result)
    result = _ensure_iop_units(result)
    result = _uppercase_laterality(result)
    return result


# ---------------------------------------------------------------------------
# IOP checks
# ---------------------------------------------------------------------------

def _iop_bounds() -> tuple[float, float, float]:
    """Returns (hypotony_low, warn_low, warn_high) from terminology.yaml
    with fallback to built-in defaults."""
    return (_DEFAULT_IOP_HYPOTONY, _DEFAULT_IOP_WARN_LOW, _DEFAULT_IOP_WARN_HIGH)


def _check_iop(text: str) -> list[Violation]:
    violations: list[Violation] = []
    hypotony_low, warn_low, warn_high = _iop_bounds()

    # IOP_RANGE: numbers explicitly tagged with mmHg
    seen_spans: set[tuple[int, int]] = set()
    for m in _IOP_NUMBER_WITH_UNIT_RE.finditer(text):
        value = float(m.group(1))
        seen_spans.add(m.span())
        if value < hypotony_low or value > warn_high:
            violations.append(
                Violation(
                    code="IOP_RANGE",
                    severity="error",
                    message=(
                        f"IOP value {value:g} mmHg is outside the plausible "
                        f"range ({hypotony_low:g}-{warn_high:g} mmHg)."
                    ),
                    line_text=text,
                )
            )
        elif value > warn_low:
            violations.append(
                Violation(
                    code="IOP_RANGE",
                    severity="warn",
                    message=(
                        f"IOP value {value:g} mmHg is elevated (above "
                        f"{warn_low:g} mmHg)."
                    ),
                    line_text=text,
                )
            )

    # IOP_UNIT: an IOP-cue-adjacent number without "mmHg" nearby
    if _IOP_CUE_RE.search(text):
        for m in _IOP_NUMBER_NEAR_CUE_RE.finditer(text):
            num_start, num_end = m.start(1), m.end(1)
            if (num_start, num_end) in seen_spans:
                continue
            tail = text[num_end : num_end + 10]
            if re.match(r"\s*mmHg", tail, re.IGNORECASE):
                continue
            violations.append(
                Violation(
                    code="IOP_UNIT",
                    severity="warn",
                    message=(
                        f"IOP value '{m.group(1)}' found without 'mmHg' units nearby."
                    ),
                    line_text=text,
                )
            )
    return violations


# ---------------------------------------------------------------------------
# VA_FORMAT checks
# ---------------------------------------------------------------------------

def _has_va_cue_near(text: str, start: int, end: int, window: int = 25) -> bool:
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    return bool(_VA_CUE_RE.search(text[lo:hi]))


def _check_va_format(text: str) -> list[Violation]:
    violations: list[Violation] = []

    # Explicit hyphen-where-slash-expected malformed VA (e.g. "VA 20-40")
    for m in _VA_HYPHEN_RE.finditer(text):
        violations.append(
            Violation(
                code="VA_FORMAT",
                severity="error",
                message=(
                    f"Malformed visual acuity notation '{m.group(0)}' — "
                    "expected a slash (e.g. '20/40'), found a hyphen."
                ),
                line_text=text,
            )
        )

    # Missing-denominator forms near a VA cue, e.g. "VA 20/" or "20/ OD"
    for m in re.finditer(r"\b(20|6)\s*/\s*(?!\d)", text):
        if _has_va_cue_near(text, m.start(), m.end()):
            violations.append(
                Violation(
                    code="VA_FORMAT",
                    severity="error",
                    message=(
                        f"Malformed visual acuity notation '{m.group(0).strip()}' "
                        "is missing a denominator."
                    ),
                    line_text=text,
                )
            )

    # Fractions with a VA cue nearby: numerator must be 20 or 6, denominator
    # positive integer/decimal, and no trailing junk glued onto the
    # denominator (e.g. "20/45x" is unparseable).
    for m in _FRACTION_RE.finditer(text):
        num, den = m.group(1), m.group(2)
        if not den:
            continue  # handled above as missing-denominator
        if not _has_va_cue_near(text, m.start(), m.end()):
            continue
        if num not in ("20", "6"):
            continue
        # check for glued trailing junk right after the matched fraction
        trailing = text[m.end() : m.end() + 1]
        if trailing and trailing.isalpha():
            violations.append(
                Violation(
                    code="VA_FORMAT",
                    severity="error",
                    message=(
                        f"Malformed visual acuity notation "
                        f"'{m.group(0)}{trailing}' has unparseable trailing text."
                    ),
                    line_text=text,
                )
            )
            continue
        try:
            den_val = float(den)
        except ValueError:
            continue
        if den_val <= 0:
            violations.append(
                Violation(
                    code="VA_FORMAT",
                    severity="error",
                    message=(
                        f"Malformed visual acuity notation '{m.group(0)}' has "
                        "a non-positive denominator."
                    ),
                    line_text=text,
                )
            )

    return violations


# ---------------------------------------------------------------------------
# lens_rx checks: SPHERE_RANGE, CYL_SIGN, AXIS_RANGE
# ---------------------------------------------------------------------------

def _check_lens_rx(text: str) -> list[Violation]:
    violations: list[Violation] = []
    for m in _LENS_RX_RE.finditer(text):
        sph_text = m.group("sph")
        cyl_text = m.group("cyl")
        axis_text = m.group("axis")

        try:
            sphere = float(sph_text)
        except ValueError:
            sphere = None
        try:
            cylinder = float(cyl_text)
        except ValueError:
            cylinder = None

        if sphere is not None and abs(sphere) > _DEFAULT_SPHERE_MAX:
            violations.append(
                Violation(
                    code="SPHERE_RANGE",
                    severity="error",
                    message=(
                        f"Sphere value {sphere:+.2f} D exceeds the plausible "
                        f"range of ±{_DEFAULT_SPHERE_MAX:.2f} D."
                    ),
                    line_text=text,
                )
            )

        if cylinder is not None:
            cyl_msgs = []
            if cylinder > 0:
                cyl_msgs.append(
                    f"cylinder value {cylinder:+.2f} D is positive, violating "
                    "the minus-cylinder convention (sign error)"
                )
            if abs(cylinder) > _DEFAULT_CYL_MAX_MAGNITUDE:
                cyl_msgs.append(
                    f"cylinder magnitude {abs(cylinder):.2f} D is too large "
                    f"(magnitude too large; exceeds {_DEFAULT_CYL_MAX_MAGNITUDE:.2f} D)"
                )
            if cyl_msgs:
                violations.append(
                    Violation(
                        code="CYL_SIGN",
                        severity="error",
                        message="Cylinder issue: " + "; ".join(cyl_msgs) + ".",
                        line_text=text,
                    )
                )

        axis_ok = axis_text.isdigit()
        axis_val = int(axis_text) if axis_ok else None
        if not axis_ok or axis_val == 0 or (axis_val is not None and axis_val > 180):
            violations.append(
                Violation(
                    code="AXIS_RANGE",
                    severity="error",
                    message=(
                        f"Axis value '{axis_text}' is out of the plausible "
                        "range (1-180 degrees, integer)."
                    ),
                    line_text=text,
                )
            )

    # axis missing entirely when a cylinder value is present, e.g. "-2.25 -0.75 x"
    for m in re.finditer(
        r"(?P<sph>[+-]\d{1,2}(?:\.\d{1,2})?)\s+(?P<cyl>[+-]\d{1,2}(?:\.\d{1,2})?)\s*[xX](?!\s*\d)",
        text,
    ):
        violations.append(
            Violation(
                code="AXIS_RANGE",
                severity="error",
                message="Cylinder value present but axis is missing.",
                line_text=text,
            )
        )

    return violations


# ---------------------------------------------------------------------------
# LATERALITY_CONFLICT (transcript-aware heuristic)
# ---------------------------------------------------------------------------

_RIGHT_EYE_RE = re.compile(r"\b(right eye|OD)\b", re.IGNORECASE)
_LEFT_EYE_RE = re.compile(r"\b(left eye|OS)\b", re.IGNORECASE)
_FINDING_KEYWORD_RE = re.compile(
    r"\b(IOP|cup|C/D|field|RNFL|pressure)\b", re.IGNORECASE
)


def _laterality_conflict_for_line(text: str, transcript: str) -> list[Violation]:
    """Heuristic check for a note line's laterality vs. the transcript.

    Approach (deliberately simple, not exhaustive): for each numeric value in
    the note line that sits near a laterality token (OD/OS) or a finding
    keyword (IOP, cup, C/D, field, RNFL), search the transcript for the same
    number appearing near a "right eye"/OD or "left eye"/OS mention (within a
    ~90-character window on each side of the number, which typically spans a
    sentence in these transcripts). If the transcript's laterality for that
    number contradicts the note's stated laterality for the same number, a
    LATERALITY_CONFLICT violation is emitted.

    This is intentionally shallow: it only looks at numbers that literally
    reappear in the transcript, and only compares the *nearest* laterality
    mention on each side. False negatives are expected and acceptable — the
    goal is to catch obvious eye-flip transcription errors, not to perform
    exhaustive clinical reconciliation between note and transcript.
    """
    violations: list[Violation] = []
    if not transcript:
        return violations

    lat_tokens = list(_LATERALITY_RE.finditer(text))
    if not lat_tokens:
        return violations

    numbers = list(re.finditer(r"\d{1,3}(?:\.\d+)?", text))
    if not numbers:
        return violations

    window = 90
    for num_m in numbers:
        num_str = num_m.group(0)
        # find nearest laterality token to this number in the note line
        nearest_lat = min(
            lat_tokens, key=lambda lm: abs(lm.start() - num_m.start()), default=None
        )
        if nearest_lat is None:
            continue
        note_lat = nearest_lat.group(1).upper()
        if note_lat not in ("OD", "OS"):
            continue  # OU is not a flip candidate here

        # only bother if there's a finding-keyword or a laterality context
        # near the number in the note line (keeps this from firing on
        # unrelated stray numbers)
        lo = max(0, num_m.start() - window)
        hi = min(len(text), num_m.end() + window)
        local_context = text[lo:hi]
        if not (_FINDING_KEYWORD_RE.search(local_context) or nearest_lat):
            continue

        # search transcript for this same number
        for t_num_m in re.finditer(re.escape(num_str), transcript):
            t_lo = max(0, t_num_m.start() - window)
            t_hi = min(len(transcript), t_num_m.end() + window)
            t_window = transcript[t_lo:t_hi]

            right_m = _RIGHT_EYE_RE.search(t_window)
            left_m = _LEFT_EYE_RE.search(t_window)
            if right_m and left_m:
                # ambiguous window (both mentioned) — pick whichever is closer
                t_num_pos = t_num_m.start() - t_lo
                right_dist = abs(right_m.start() - t_num_pos)
                left_dist = abs(left_m.start() - t_num_pos)
                transcript_lat = "OD" if right_dist < left_dist else "OS"
            elif right_m:
                transcript_lat = "OD"
            elif left_m:
                transcript_lat = "OS"
            else:
                continue

            if transcript_lat != note_lat:
                violations.append(
                    Violation(
                        code="LATERALITY_CONFLICT",
                        severity="error",
                        message=(
                            f"Note attributes value '{num_str}' to {note_lat}, "
                            f"but transcript context near the same value "
                            f"suggests {transcript_lat} — possible laterality flip."
                        ),
                        line_text=text,
                    )
                )
                break  # one violation per number is enough

    return violations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_line(text: str, transcript: str | None = None) -> list[Violation]:
    """Check a single note line for terminology/notation violations.

    Runs VA_FORMAT, IOP_RANGE/IOP_UNIT, lens_rx (SPHERE_RANGE/CYL_SIGN/
    AXIS_RANGE) checks unconditionally, and LATERALITY_CONFLICT only when a
    transcript is supplied (transcript is not None).
    """
    if not text:
        return []

    violations: list[Violation] = []
    violations.extend(_check_va_format(text))
    violations.extend(_check_iop(text))
    violations.extend(_check_lens_rx(text))
    if transcript is not None:
        violations.extend(_laterality_conflict_for_line(text, transcript))
    return violations


def check_note(note: dict, transcript: str | None = None) -> list[Violation]:
    """Check every line in a Note dict's SOAP sections (S, O, A, P, in that
    order; within each section, in list order), concatenating all
    Violations found by check_line."""
    violations: list[Violation] = []
    soap = note.get("soap", {}) if note else {}
    for section in SOAP_SECTIONS:
        lines = soap.get(section) or []
        for line in lines:
            text = line.get("text", "") if isinstance(line, dict) else ""
            violations.extend(check_line(text, transcript=transcript))
    return violations
