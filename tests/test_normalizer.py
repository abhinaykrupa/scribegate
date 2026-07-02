import pytest

from scribegate.normalizer import Violation, check_line, check_note, normalize_line


def _codes(violations):
    return [v.code for v in violations]


# 1. A valid, clean VA/IOP/laterality line produces zero violations.
def test_clean_line_no_violations():
    text = "VA cc 20/25 OD, 20/30 OS; IOP 17 mmHg OD / 18 mmHg OS (GAT)."
    violations = check_line(text)
    assert violations == []


# 2. VA_FORMAT triggers on malformed Snellen.
@pytest.mark.parametrize(
    "text",
    [
        "VA 20/ OD, unable to complete chart.",
        "VA 20-40 OD today.",
    ],
)
def test_va_format_malformed(text):
    violations = check_line(text)
    assert "VA_FORMAT" in _codes(violations)


# 3. Valid CF/HM/LP/NLP notations pass (no VA_FORMAT).
@pytest.mark.parametrize(
    "text",
    [
        "VA CF at 2 ft OD.",
        "VA HM OS.",
        "VA LP with projection OD.",
        "VA NLP OS.",
    ],
)
def test_low_vision_scale_no_va_format(text):
    violations = check_line(text)
    assert "VA_FORMAT" not in _codes(violations)


# 4. IOP_RANGE error on IOP < 6.
def test_iop_range_error_low():
    violations = check_line("IOP 4 mmHg OD")
    range_v = [v for v in violations if v.code == "IOP_RANGE"]
    assert len(range_v) == 1
    assert range_v[0].severity == "error"


# 5. IOP_RANGE error on IOP > 60.
def test_iop_range_error_high():
    violations = check_line("IOP 65 mmHg OD")
    range_v = [v for v in violations if v.code == "IOP_RANGE"]
    assert len(range_v) == 1
    assert range_v[0].severity == "error"


# 6. IOP_RANGE warn "elevated" on IOP in [22,60].
@pytest.mark.parametrize("value", [22, 45, 60])
def test_iop_range_warn_elevated(value):
    violations = check_line(f"IOP {value} mmHg OD")
    range_v = [v for v in violations if v.code == "IOP_RANGE"]
    assert len(range_v) == 1
    assert range_v[0].severity == "warn"
    assert "elevated" in range_v[0].message.lower()


# 7. IOP boundary tests: 6/21/22/60/61 all tested explicitly.
def test_iop_boundary_6_no_error():
    violations = check_line("IOP 6 mmHg OD")
    assert "IOP_RANGE" not in _codes(violations)


def test_iop_boundary_21_no_warn():
    violations = check_line("IOP 21 mmHg OD")
    assert "IOP_RANGE" not in _codes(violations)


def test_iop_boundary_22_warns():
    violations = check_line("IOP 22 mmHg OD")
    range_v = [v for v in violations if v.code == "IOP_RANGE"]
    assert len(range_v) == 1
    assert range_v[0].severity == "warn"


def test_iop_boundary_60_warns():
    violations = check_line("IOP 60 mmHg OD")
    range_v = [v for v in violations if v.code == "IOP_RANGE"]
    assert len(range_v) == 1
    assert range_v[0].severity == "warn"


def test_iop_boundary_61_errors():
    violations = check_line("IOP 61 mmHg OD")
    range_v = [v for v in violations if v.code == "IOP_RANGE"]
    assert len(range_v) == 1
    assert range_v[0].severity == "error"


# 8. IOP_UNIT warn when mmHg missing from an IOP-looking line.
def test_iop_unit_warn_missing_units():
    violations = check_line("IOP 17 OD")
    unit_v = [v for v in violations if v.code == "IOP_UNIT"]
    assert len(unit_v) == 1
    assert unit_v[0].severity == "warn"


# 9. Laterality well-formed check: OD/OS/OU recognized; substrings in
# "wood"/"good"/"mood"/"hood" do NOT false-positive.
def test_laterality_no_false_positive_in_words():
    text = "Patient stood in the good, calm mood near the hood; wood floors."
    normalized = normalize_line(text)
    assert normalized == text  # no uppercasing/mangling of embedded substrings
    violations = check_line(text)
    assert "LATERALITY_CONFLICT" not in _codes(violations)


def test_laterality_recognizes_standalone_tokens():
    text = "iop 17 od"
    normalized = normalize_line(text)
    assert "OD" in normalized.split()


# 10. LATERALITY_CONFLICT triggers when note contradicts transcript.
def test_laterality_conflict_triggers_on_flip():
    transcript = (
        "DOCTOR: Let's check the pressure. TECH: Goldmann pressure in the "
        "left eye is 26 today."
    )
    note_line = "IOP 26 mmHg OD (GAT)."
    violations = check_line(note_line, transcript=transcript)
    assert "LATERALITY_CONFLICT" in _codes(violations)


# 11. LATERALITY_CONFLICT does NOT trigger when note matches transcript.
def test_laterality_conflict_absent_when_consistent():
    transcript = (
        "DOCTOR: Let's check the pressure. TECH: Goldmann pressure in the "
        "right eye is 26 today."
    )
    note_line = "IOP 26 mmHg OD (GAT)."
    violations = check_line(note_line, transcript=transcript)
    assert "LATERALITY_CONFLICT" not in _codes(violations)


# 12. CYL_SIGN triggers on positive cylinder.
def test_cyl_sign_positive_cylinder():
    violations = check_line("-2.00 +0.75 x 090")
    cyl_v = [v for v in violations if v.code == "CYL_SIGN"]
    assert len(cyl_v) == 1
    assert "sign" in cyl_v[0].message.lower()


# 13. CYL_SIGN triggers when cylinder magnitude > 6.00, message distinguishes
# it as a magnitude issue.
def test_cyl_sign_magnitude_too_large():
    violations = check_line("-2.00 -7.50 x 090")
    cyl_v = [v for v in violations if v.code == "CYL_SIGN"]
    assert len(cyl_v) == 1
    assert "magnitude" in cyl_v[0].message.lower()


# 14. AXIS_RANGE triggers on axis 0 and axis 181.
@pytest.mark.parametrize("axis", ["000", "181"])
def test_axis_range_out_of_bounds(axis):
    violations = check_line(f"-2.00 -0.75 x {axis}")
    assert "AXIS_RANGE" in _codes(violations)


# 15. AXIS_RANGE passes clean on axis 1 and axis 180 (boundary inclusive).
@pytest.mark.parametrize("axis", ["001", "180"])
def test_axis_range_boundary_ok(axis):
    violations = check_line(f"-2.00 -0.75 x {axis}")
    assert "AXIS_RANGE" not in _codes(violations)


# 16. SPHERE_RANGE triggers when sphere magnitude > 20D.
def test_sphere_range_triggers():
    violations = check_line("-22.00 -0.75 x 090")
    assert "SPHERE_RANGE" in _codes(violations)


# 17. 6/x -> 20/x normalize_line conversion.
def test_normalize_metric_to_snellen():
    result = normalize_line("6/9 od")
    assert "20/30" in result
    assert "OD" in result


# 18. normalize_line idempotence across >=3 different inputs.
@pytest.mark.parametrize(
    "text",
    [
        "6/9 od",
        "IOP 17 od",
        "va cc 6/6 os, iop 18 ou",
    ],
)
def test_normalize_line_idempotent(text):
    once = normalize_line(text)
    twice = normalize_line(once)
    assert once == twice


# 19. check_note aggregates violations across S/O/A/P sections.
def test_check_note_aggregates_across_sections():
    note = {
        "transcript_id": "unit_test_01",
        "visit_type": "glaucoma_followup",
        "synthetic": True,
        "soap": {
            "S": [{"text": "Patient reports mild discomfort OD.", "spans": [[0, 10]]}],
            "O": [
                {"text": "IOP 65 mmHg OD (GAT).", "spans": [[0, 10]]},
                {"text": "VA cc 20/25 OD, 20/30 OS.", "spans": [[0, 10]]},
            ],
            "A": [{"text": "Glaucoma OU, OD above target.", "spans": [[0, 10]]}],
            "P": [{"text": "Continue latanoprost OU.", "spans": [[0, 10]]}],
        },
    }
    violations = check_note(note)
    assert any(v.code == "IOP_RANGE" and v.severity == "error" for v in violations)
    # confirm it walked multiple sections: the flagged line came from O,
    # and no exception/short-circuit happened before A/P were processed.
    line_texts = {v.line_text for v in violations}
    assert "IOP 65 mmHg OD (GAT)." in line_texts


# Regression: real golden-note line must not misfire VA_FORMAT on IOP/baseline pairs.
def test_regression_glaucoma_05_no_va_format_false_positive():
    text = (
        "IOP 26 mmHg OD / 24 mmHg OS (GAT); 21/20 four months ago; "
        "baseline 34/32; above target."
    )
    violations = check_line(text)
    assert "VA_FORMAT" not in _codes(violations)
    # it's fine/expected to have IOP_RANGE warns for the elevated values
    assert any(v.code == "IOP_RANGE" for v in violations)


def test_violation_dataclass_fields():
    v = Violation(code="VA_FORMAT", severity="error", message="msg", line_text="text")
    assert v.code == "VA_FORMAT"
    assert v.severity == "error"
    assert v.message == "msg"
    assert v.line_text == "text"
