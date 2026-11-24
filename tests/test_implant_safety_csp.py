"""Unit tests for formal IEC 60601-2-33 / ASTM F2503 implant safety CSP verifier."""

from __future__ import annotations

import numpy as np
import pytest

from mrphys_guard.implant_safety_csp import (
    compute_sequence_rf_physics,
    list_implant_profiles,
    lookup_implant_profile,
    verify_implant_safety_csp,
)
from mrphys_guard.ir import (
    AbstentionCode,
    AbstentionError,
    ImplantConditionSpec,
    MRSafetyLabel,
    MRSequenceSpec,
    RFCoilType,
    ScannerFieldSpec,
)
from mrphys_guard.provenance import gate_implant_rf_violation


def test_rf_and_sar_exact_physics_scaling_laws() -> None:
    """Verify B1+rms ~ flip / sqrt(TR * tau_RF) and SAR ~ B0^2 * flip^2 / (TR * tau_RF)."""
    base = MRSequenceSpec(
        name="GRE_Base",
        sequence_type="GRE",
        b0_T=1.5,
        bandwidth_hz_per_px=300.0,
        flip_angle_deg=30.0,
        tr_ms=10.0,
        te_ms=3.0,
        rf_pulse_duration_ms=1.0,
    )
    e0 = compute_sequence_rf_physics(base)

    # Doubling flip angle -> 2x B1+rms, 4x SAR
    e_2flip = compute_sequence_rf_physics(base.model_copy(update={"flip_angle_deg": 60.0}))
    assert np.isclose(e_2flip.b1_rms_uT / e0.b1_rms_uT, 2.0, rtol=1e-12)
    assert np.isclose(e_2flip.wb_sar_W_per_kg / e0.wb_sar_W_per_kg, 4.0, rtol=1e-12)

    # Quadrupling TR -> 0.5x B1+rms, 0.25x SAR
    e_4tr = compute_sequence_rf_physics(base.model_copy(update={"tr_ms": 40.0}))
    assert np.isclose(e_4tr.b1_rms_uT / e0.b1_rms_uT, 0.5, rtol=1e-12)
    assert np.isclose(e_4tr.wb_sar_W_per_kg / e0.wb_sar_W_per_kg, 0.25, rtol=1e-12)

    # Doubling B0 (1.5T -> 3.0T) -> same B1+rms, 4x SAR
    e_3t = compute_sequence_rf_physics(base.model_copy(update={"b0_T": 3.0}))
    assert np.isclose(e_3t.b1_rms_uT, e0.b1_rms_uT, rtol=1e-12)
    assert np.isclose(e_3t.wb_sar_W_per_kg / e0.wb_sar_W_per_kg, 4.0, rtol=1e-12)

    # Quadrupling RF pulse duration tau_RF -> 0.5x B1+rms, 0.25x SAR
    e_4tau = compute_sequence_rf_physics(base.model_copy(update={"rf_pulse_duration_ms": 4.0}))
    assert np.isclose(e_4tau.b1_rms_uT / e0.b1_rms_uT, 0.5, rtol=1e-12)
    assert np.isclose(e_4tau.wb_sar_W_per_kg / e0.wb_sar_W_per_kg, 0.25, rtol=1e-12)


def test_csp_detects_pacemaker_violation_and_passes_safe_protocol() -> None:
    """High-SAR bSSFP violates 1.5T pacemaker B1+rms <= 2.0 uT; low-SAR GRE passes all checks."""
    scanner = ScannerFieldSpec(b0_T=1.5, max_spatial_gradient_T_per_m=17.0)
    pm = lookup_implant_profile("generic_cardiac_pacemaker_1p5t")

    unsafe_seq = MRSequenceSpec(
        name="HighSAR_bSSFP",
        sequence_type="bSSFP",
        b0_T=1.5,
        bandwidth_hz_per_px=600.0,
        flip_angle_deg=70.0,
        tr_ms=3.0,
        te_ms=1.5,
        rf_pulse_duration_ms=0.5,
    )
    verdict_fail = verify_implant_safety_csp(scanner, [unsafe_seq], pm)
    assert verdict_fail.verdict == "VIOLATION"
    assert any("max_b1_rms" in v for v in verdict_fail.violated_constraints)

    with pytest.raises(AbstentionError) as exc_rf:
        gate_implant_rf_violation(verdict_fail.violated_constraints, pm.device_id)
    assert exc_rf.value.code == AbstentionCode.IMPLANT_SAR_OR_B1RMS_VIOLATION

    safe_seq = MRSequenceSpec(
        name="LowSAR_GRE",
        sequence_type="GRE",
        b0_T=1.5,
        bandwidth_hz_per_px=350.0,
        flip_angle_deg=20.0,
        tr_ms=12.0,
        te_ms=4.0,
        rf_pulse_duration_ms=2.0,
        slew_rate_T_per_m_s=120.0,
        scan_duration_min=5.0,
    )
    verdict_pass = verify_implant_safety_csp(scanner, [safe_seq], pm)
    assert verdict_pass.verdict == "PASS"
    assert not verdict_pass.violated_constraints


def test_csp_hard_abstain_on_mr_unsafe_and_unspecified_implants() -> None:
    """MR_UNSAFE and under-specified MR_CONDITIONAL implants must trigger hard ABSTAIN."""
    scanner = ScannerFieldSpec(b0_T=1.5)
    seq = MRSequenceSpec(name="T1_GRE", b0_T=1.5, bandwidth_hz_per_px=300.0, tr_ms=15.0, te_ms=4.0)

    unsafe_clip = lookup_implant_profile("ferromagnetic_aneurysm_clip_unsafe")
    v_unsafe = verify_implant_safety_csp(scanner, [seq], unsafe_clip, raise_on_abstain=False)
    assert v_unsafe.verdict == "ABSTAIN"
    assert v_unsafe.abstention_code == AbstentionCode.MR_UNSAFE_DEVICE_DETECTED

    with pytest.raises(AbstentionError) as exc_unsafe:
        verify_implant_safety_csp(scanner, [seq], unsafe_clip, raise_on_abstain=True)
    assert exc_unsafe.value.code == AbstentionCode.MR_UNSAFE_DEVICE_DETECTED

    unknown_implant = ImplantConditionSpec(
        device_id="unknown_001",
        device_name="Unlabeled Metallic Fragment",
        category="unknown",
        mr_status=MRSafetyLabel.UNSPECIFIED,
    )
    v_unk = verify_implant_safety_csp(scanner, [seq], unknown_implant, raise_on_abstain=False)
    assert v_unk.verdict == "ABSTAIN"
    assert v_unk.abstention_code == AbstentionCode.UNSPECIFIED_IMPLANT_CONDITION

    incomplete_cond = ImplantConditionSpec(
        device_id="partial_cond_001",
        device_name="Incomplete Conditional Lead",
        category="lead",
        mr_status=MRSafetyLabel.MR_CONDITIONAL,
        allowed_b0_T=[1.5],
    )
    v_inc = verify_implant_safety_csp(scanner, [seq], incomplete_cond, raise_on_abstain=False)
    assert v_inc.verdict == "ABSTAIN"
    assert v_inc.abstention_code == AbstentionCode.UNSPECIFIED_IMPLANT_CONDITION


def test_implant_profile_registry_and_b0_coil_violation() -> None:
    """Built-in device registry has >= 6 profiles; 3.0T scanner violates 1.5T-only pacemaker."""
    profiles = list_implant_profiles()
    assert len(profiles) >= 6
    with pytest.raises(KeyError):
        lookup_implant_profile("nonexistent_device")

    scanner_3t = ScannerFieldSpec(scanner_name="3.0T MRI", b0_T=3.0)
    pm = lookup_implant_profile("generic_cardiac_pacemaker_1p5t")
    seq_3t = MRSequenceSpec(
        name="3T_GRE",
        b0_T=3.0,
        bandwidth_hz_per_px=300.0,
        flip_angle_deg=15.0,
        tr_ms=15.0,
        te_ms=4.0,
        rf_coil=RFCoilType.MULTI_TRANSMIT_PARALLEL,
    )
    v = verify_implant_safety_csp(scanner_3t, [seq_3t], pm)
    assert v.verdict == "VIOLATION"
    assert any("static_field_b0" in c for c in v.violated_constraints)
    assert any("rf_coil_type" in c for c in v.violated_constraints)
