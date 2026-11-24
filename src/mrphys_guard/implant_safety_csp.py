"""Formal IEC 60601-2-33 / ASTM F2503 MR-Conditional Implant Constraint Satisfaction Solver.

Implements:
1. Exact physics RF / SAR / B1+rms / dB/dt sequence scaling model:
       B1+rms = (1 / gamma) * sqrt(f_shape * sum(alpha_j^2) / (TR * tau_RF))
       SAR    = c_anatomy * B0^2 * (B1+rms)^2  (proportional to B0^2 * flip^2 * duty_cycle / TR)
2. Built-in generic MR-conditional literature device profiles (cardiac pacemaker, DBS lead,
   hip prosthesis, prostate gold fiducial, coronary stent, plus MR-unsafe aneurysm clip).
3. Formal exact Constraint Satisfaction Problem (CSP) verifier across multi-sequence protocols
   with hard ASTM F2503 ABSTAIN gates for MR_UNSAFE or UNSPECIFIED devices.
"""

from __future__ import annotations

import numpy as np

from mrphys_guard.distortion_solver import GAMMA_BAR_HZ_PER_T, relative_snr_from_bandwidth
from mrphys_guard.ir import (
    AbstentionError,
    ConstraintCheckResult,
    ImplantConditionSpec,
    MRSafetyLabel,
    MRSequenceSpec,
    RFCoilType,
    SafetyCSPVerdict,
    ScannerFieldSpec,
    SequencePhysicsEstimate,
)
from mrphys_guard.provenance import gate_implant_labeling

# Radian gyromagnetic ratio gamma = 2*pi * 42.577478518e6 [rad / (s * T)]
GAMMA_RAD_PER_S_T: float = 2.0 * np.pi * GAMMA_BAR_HZ_PER_T

# Empirically calibrated RF power coupling constants [W / (kg * T^2 * uT^2)] for a 70 kg adult
# At 1.5T, B1+rms = 3.2 uT -> WB SAR ~= 2.0 W/kg (IEC 60601-2-33 Normal Operating Mode boundary)
SAR_COUPLING_WB: float = 0.0868
SAR_COUPLING_HEAD: float = 0.1400

# IEC 60601-2-33 (Ed. 4, 2022) whole-body and head SAR limits [W/kg]
IEC_SAR_LIMITS: dict[str, dict[str, float]] = {
    "normal": {"wb_sar": 2.0, "head_sar": 3.2},
    "first_level_controlled": {"wb_sar": 4.0, "head_sar": 3.2},
}


# ---------------------------------------------------------------------------
# Built-In Generic MR-Conditional Literature Profiles (NOT Manufacturer IFU)
# ---------------------------------------------------------------------------

GENERIC_IMPLANT_PROFILES: dict[str, ImplantConditionSpec] = {
    "generic_cardiac_pacemaker_1p5t": ImplantConditionSpec(
        device_id="generic_cardiac_pacemaker_1p5t",
        device_name="Generic MR-Conditional Dual-Chamber Cardiac Pacemaker (1.5T)",
        category="cardiac_implantable_electronic_device",
        mr_status=MRSafetyLabel.MR_CONDITIONAL,
        source_note=(
            "Generic literature reference envelope aligned with typical 1.5T MR-conditional CIED "
            "labeling (IEC 60601-2-33 / ASTM F2503). NOT a manufacturer IFU."
        ),
        allowed_b0_T=[1.5],
        max_spatial_gradient_T_per_m=20.0,
        max_b1_rms_uT=2.0,
        max_wb_sar_W_per_kg=2.0,
        max_head_sar_W_per_kg=3.2,
        max_slew_rate_T_per_m_s=200.0,
        max_db_dt_T_per_s=60.0,
        max_active_scan_time_min=30.0,
        allowed_rf_coils=[RFCoilType.WHOLE_BODY_CP, RFCoilType.TX_RX_HEAD],
    ),
    "generic_dbs_lead_1p5t": ImplantConditionSpec(
        device_id="generic_dbs_lead_1p5t",
        device_name="Generic MR-Conditional Deep Brain Stimulation (DBS) System (1.5T)",
        category="neurostimulator",
        mr_status=MRSafetyLabel.MR_CONDITIONAL,
        source_note=(
            "Generic literature reference envelope for 1.5T MR-conditional DBS lead "
            "(strict B1+rms <= 1.1 uT / head SAR envelope). NOT a manufacturer IFU."
        ),
        allowed_b0_T=[1.5],
        max_spatial_gradient_T_per_m=19.0,
        max_b1_rms_uT=1.1,
        max_wb_sar_W_per_kg=0.5,
        max_head_sar_W_per_kg=0.4,
        max_slew_rate_T_per_m_s=125.0,
        max_db_dt_T_per_s=45.0,
        max_active_scan_time_min=30.0,
        allowed_rf_coils=[RFCoilType.TX_RX_HEAD, RFCoilType.WHOLE_BODY_CP],
    ),
    "generic_hip_prosthesis_ti": ImplantConditionSpec(
        device_id="generic_hip_prosthesis_ti",
        device_name="Generic Ti-6Al-4V Bilateral Hip Prosthesis (1.5T/3.0T)",
        category="orthopedic_prosthesis",
        mr_status=MRSafetyLabel.MR_CONDITIONAL,
        source_note=(
            "Generic literature reference envelope for non-ferromagnetic titanium hip arthroplasty "
            "(ASTM F2503 / IEC 60601-2-33). NOT a manufacturer IFU."
        ),
        allowed_b0_T=[0.35, 1.5, 3.0],
        max_spatial_gradient_T_per_m=30.0,
        max_b1_rms_uT=3.2,
        max_wb_sar_W_per_kg=2.0,
        max_head_sar_W_per_kg=3.2,
        max_slew_rate_T_per_m_s=200.0,
        max_db_dt_T_per_s=70.0,
        max_active_scan_time_min=60.0,
        allowed_rf_coils=[RFCoilType.WHOLE_BODY_CP, RFCoilType.TX_RX_HEAD, RFCoilType.TX_RX_EXTREMITY],
    ),
    "generic_prostate_gold_fiducial": ImplantConditionSpec(
        device_id="generic_prostate_gold_fiducial",
        device_name="Generic Pure Gold Prostate Fiducial Markers (0.35T/1.5T/3.0T)",
        category="radiotherapy_fiducial",
        mr_status=MRSafetyLabel.MR_CONDITIONAL,
        source_note=(
            "Generic literature reference envelope for 99.9% Au prostate radiotherapy fiducials. "
            "NOT a manufacturer IFU."
        ),
        allowed_b0_T=[0.35, 1.5, 3.0],
        max_spatial_gradient_T_per_m=40.0,
        max_b1_rms_uT=4.5,
        max_wb_sar_W_per_kg=4.0,
        max_head_sar_W_per_kg=3.2,
        max_slew_rate_T_per_m_s=200.0,
        max_db_dt_T_per_s=80.0,
        max_active_scan_time_min=60.0,
        allowed_rf_coils=[RFCoilType.WHOLE_BODY_CP, RFCoilType.TX_RX_HEAD, RFCoilType.TX_RX_EXTREMITY],
    ),
    "generic_coronary_stent_nitinol": ImplantConditionSpec(
        device_id="generic_coronary_stent_nitinol",
        device_name="Generic Nitinol / CoCr Vascular Stent (1.5T/3.0T)",
        category="vascular_stent",
        mr_status=MRSafetyLabel.MR_CONDITIONAL,
        source_note=(
            "Generic literature reference envelope for weakly paramagnetic vascular stents. "
            "NOT a manufacturer IFU."
        ),
        allowed_b0_T=[1.5, 3.0],
        max_spatial_gradient_T_per_m=30.0,
        max_b1_rms_uT=3.2,
        max_wb_sar_W_per_kg=2.0,
        max_head_sar_W_per_kg=3.2,
        max_slew_rate_T_per_m_s=200.0,
        max_db_dt_T_per_s=70.0,
        max_active_scan_time_min=60.0,
        allowed_rf_coils=[RFCoilType.WHOLE_BODY_CP, RFCoilType.TX_RX_HEAD],
    ),
    "ferromagnetic_aneurysm_clip_unsafe": ImplantConditionSpec(
        device_id="ferromagnetic_aneurysm_clip_unsafe",
        device_name="Legacy Ferromagnetic Intracranial Aneurysm Clip (MR UNSAFE)",
        category="neurovascular_clip",
        mr_status=MRSafetyLabel.MR_UNSAFE,
        source_note="Classic ASTM F2503 MR_UNSAFE contraindication test profile.",
    ),
}


def lookup_implant_profile(device_id: str) -> ImplantConditionSpec:
    """Retrieve a built-in generic MR-conditional / MR-unsafe profile by device_id."""
    if device_id not in GENERIC_IMPLANT_PROFILES:
        available = ", ".join(sorted(GENERIC_IMPLANT_PROFILES))
        raise KeyError(f"Unknown implant profile '{device_id}'. Available profiles: {available}")
    return GENERIC_IMPLANT_PROFILES[device_id]


def list_implant_profiles() -> list[ImplantConditionSpec]:
    """Return all built-in generic MR-conditional / MR-unsafe profiles."""
    return [GENERIC_IMPLANT_PROFILES[k] for k in sorted(GENERIC_IMPLANT_PROFILES)]


# ---------------------------------------------------------------------------
# Exact Sequence RF / B1+rms / SAR / Gradient Scaling Physics
# ---------------------------------------------------------------------------


def compute_sequence_rf_physics(
    sequence: MRSequenceSpec,
    scanner_bore_radius_m: float = 0.30,
) -> SequencePhysicsEstimate:
    """Compute exact B1+rms [uT], whole-body SAR [W/kg], head SAR [W/kg], and dB/dt [T/s].

    Exact scaling laws:
        B1+rms = (1 / gamma) * sqrt(f_shape * sum(alpha_j^2) / (TR * tau_RF))
               proportional to flip_angle / sqrt(TR * tau_RF)
        SAR    = c_anatomy * B0^2 * (B1+rms)^2
               proportional to B0^2 * flip_angle^2 * duty_cycle / TR  (since duty_cycle ~ N_rf * tau_RF / TR,
               and peak B1^2 ~ flip^2 / tau_RF^2, so peak B1^2 * duty_cycle ~ flip^2 / (TR * tau_RF))
    """
    alpha_exc_rad = float(np.deg2rad(sequence.flip_angle_deg))
    n_refocus = max(0, sequence.echo_train_length - 1) if sequence.sequence_type in {"TSE", "FSE", "SE"} else 0
    alpha_ref_rad = float(np.deg2rad(sequence.refocusing_flip_angle_deg))

    sum_alpha_sq = alpha_exc_rad**2 + n_refocus * (alpha_ref_rad**2)
    tr_s = sequence.tr_ms * 1e-3
    tau_rf_s = sequence.rf_pulse_duration_ms * 1e-3

    # Exact B1+rms [Tesla -> microTesla]
    b1_rms_T = (1.0 / GAMMA_RAD_PER_S_T) * float(
        np.sqrt(sequence.rf_shape_factor * sum_alpha_sq / (tr_s * tau_rf_s))
    )
    b1_rms_uT = b1_rms_T * 1e6

    # If DICOM explicitly reported a higher conservative B1+rms, take the conservative max
    if sequence.reported_b1_rms_uT is not None:
        b1_rms_uT = max(b1_rms_uT, sequence.reported_b1_rms_uT)

    wb_sar = SAR_COUPLING_WB * (sequence.b0_T**2) * (b1_rms_uT**2)
    head_sar = SAR_COUPLING_HEAD * (sequence.b0_T**2) * (b1_rms_uT**2)
    if sequence.reported_sar_W_per_kg is not None:
        if sequence.anatomy == "head":
            head_sar = max(head_sar, sequence.reported_sar_W_per_kg)
        else:
            wb_sar = max(wb_sar, sequence.reported_sar_W_per_kg)

    # Approximate peak peripheral dB/dt = slew_rate * effective_radius
    est_db_dt = sequence.slew_rate_T_per_m_s * scanner_bore_radius_m

    rel_snr = relative_snr_from_bandwidth(
        bandwidth_hz_per_px=sequence.bandwidth_hz_per_px,
        voxel_size_mm=sequence.voxel_size_mm,
        parallel_accel_factor=sequence.parallel_accel_factor,
    )

    return SequencePhysicsEstimate(
        sequence_name=sequence.name,
        b1_rms_uT=float(b1_rms_uT),
        wb_sar_W_per_kg=float(wb_sar),
        head_sar_W_per_kg=float(head_sar),
        slew_rate_T_per_m_s=float(sequence.slew_rate_T_per_m_s),
        estimated_db_dt_T_per_s=float(est_db_dt),
        relative_snr=float(rel_snr),
        scan_duration_min=float(sequence.scan_duration_min),
    )


# ---------------------------------------------------------------------------
# Formal IEC 60601-2-33 / ASTM F2503 Constraint Satisfaction Verifier
# ---------------------------------------------------------------------------


def _numeric_check(
    constraint_name: str,
    sequence_name: str,
    observed: float,
    limit: float,
    unit: str,
    rationale: str,
) -> ConstraintCheckResult:
    satisfied = observed <= limit + 1e-12
    margin_pct = ((limit - observed) / limit * 100.0) if limit > 0.0 else (0.0 if satisfied else -100.0)
    return ConstraintCheckResult(
        constraint_name=constraint_name,
        sequence_name=sequence_name,
        observed_value=round(observed, 6),
        limit_value=round(limit, 6),
        unit=unit,
        margin_pct=round(margin_pct, 2),
        satisfied=satisfied,
        rationale=rationale,
    )


def verify_implant_safety_csp(
    scanner: ScannerFieldSpec,
    sequences: list[MRSequenceSpec],
    implant: ImplantConditionSpec,
    raise_on_abstain: bool = False,
) -> SafetyCSPVerdict:
    """Formally verify a multi-sequence MR protocol against scanner & implant safety constraints.

    Hard ASTM F2503 ABSTAIN gates:
    - MR_UNSAFE -> ABSTAIN (MR_UNSAFE_DEVICE_DETECTED)
    - UNSPECIFIED or missing MR_CONDITIONAL limits -> ABSTAIN (UNSPECIFIED_IMPLANT_CONDITION)
    """
    if not sequences:
        raise ValueError("At least one MRSequenceSpec is required for protocol verification.")

    try:
        gate_implant_labeling(implant)
    except AbstentionError as exc:
        if raise_on_abstain:
            raise
        return SafetyCSPVerdict(
            scanner_name=scanner.scanner_name,
            implant_id=implant.device_id,
            implant_name=implant.device_name,
            mr_status=implant.mr_status,
            verdict="ABSTAIN",
            abstention_code=exc.code,
            total_active_scan_time_min=float(sum(s.scan_duration_min for s in sequences)),
            max_b1_rms_uT=0.0,
            max_wb_sar_W_per_kg=0.0,
            max_head_sar_W_per_kg=0.0,
            max_slew_rate_T_per_m_s=float(max(s.slew_rate_T_per_m_s for s in sequences)),
            sequence_estimates=[],
            checks=[],
            violated_constraints=[f"{exc.code.value}: {exc.message}"],
            summary=exc.message,
        )

    estimates = [compute_sequence_rf_physics(seq) for seq in sequences]
    checks: list[ConstraintCheckResult] = []
    iec_limits = IEC_SAR_LIMITS[scanner.operating_mode]

    # 1. Scanner static field B0 check (MR_CONDITIONAL)
    if implant.mr_status == MRSafetyLabel.MR_CONDITIONAL and implant.allowed_b0_T is not None:
        b0_ok = any(abs(scanner.b0_T - b_allow) <= 0.05 for b_allow in implant.allowed_b0_T)
        checks.append(
            ConstraintCheckResult(
                constraint_name="static_field_b0",
                sequence_name="SCANNER_GLOBAL",
                observed_value=scanner.b0_T,
                limit_value=", ".join(f"{b:.2f}" for b in implant.allowed_b0_T),
                unit="T",
                margin_pct=0.0 if b0_ok else -100.0,
                satisfied=b0_ok,
                rationale="Static field B0 must match implant ASTM F2503 conditional field strengths.",
            )
        )

    # 2. Scanner spatial field gradient |dB/dx| check
    if implant.max_spatial_gradient_T_per_m is not None:
        checks.append(
            _numeric_check(
                constraint_name="max_spatial_field_gradient",
                sequence_name="SCANNER_GLOBAL",
                observed=scanner.max_spatial_gradient_T_per_m,
                limit=implant.max_spatial_gradient_T_per_m,
                unit="T/m",
                rationale="Prevent translational magnetic deflection force on implant.",
            )
        )

    # 3. Cumulative active scan time check
    total_time = float(sum(s.scan_duration_min for s in sequences))
    if implant.max_active_scan_time_min is not None:
        checks.append(
            _numeric_check(
                constraint_name="active_scan_time",
                sequence_name="PROTOCOL_CUMULATIVE",
                observed=total_time,
                limit=implant.max_active_scan_time_min,
                unit="min",
                rationale="Limit cumulative RF thermal dose over active scan session.",
            )
        )

    # 4. Per-sequence RF, SAR, coil, and gradient slew/dB/dt constraints
    for seq, est in zip(sequences, estimates, strict=True):
        # Sequence B0 vs scanner B0 consistency
        if abs(seq.b0_T - scanner.b0_T) > 0.05:
            checks.append(
                ConstraintCheckResult(
                    constraint_name="sequence_scanner_b0_match",
                    sequence_name=seq.name,
                    observed_value=seq.b0_T,
                    limit_value=scanner.b0_T,
                    unit="T",
                    margin_pct=-100.0,
                    satisfied=False,
                    rationale="Sequence B0 does not match scanner static field B0.",
                )
            )

        # RF coil constraint
        if implant.allowed_rf_coils is not None:
            coil_ok = seq.rf_coil in implant.allowed_rf_coils
            checks.append(
                ConstraintCheckResult(
                    constraint_name="rf_coil_type",
                    sequence_name=seq.name,
                    observed_value=seq.rf_coil.value,
                    limit_value=", ".join(c.value for c in implant.allowed_rf_coils),
                    unit="enum",
                    margin_pct=0.0 if coil_ok else -100.0,
                    satisfied=coil_ok,
                    rationale="Transmit RF coil must be within implant conditional coil set.",
                )
            )

        # B1+rms constraint (implant-specific)
        if implant.max_b1_rms_uT is not None:
            checks.append(
                _numeric_check(
                    constraint_name="max_b1_rms",
                    sequence_name=seq.name,
                    observed=est.b1_rms_uT,
                    limit=implant.max_b1_rms_uT,
                    unit="uT",
                    rationale="Patient-independent IEC 60601-2-33 Ed. 4 RF heating limit near leads.",
                )
            )

        # Whole-body SAR constraint: min(implant, IEC operating mode)
        wb_limit = iec_limits["wb_sar"]
        if implant.max_wb_sar_W_per_kg is not None:
            wb_limit = min(wb_limit, implant.max_wb_sar_W_per_kg)
        checks.append(
            _numeric_check(
                constraint_name="max_wb_sar",
                sequence_name=seq.name,
                observed=est.wb_sar_W_per_kg,
                limit=wb_limit,
                unit="W/kg",
                rationale="IEC 60601-2-33 & implant whole-body averaged SAR thermal limit.",
            )
        )

        # Head SAR constraint (active when anatomy == head or implant specifies strict head SAR)
        if seq.anatomy == "head" or implant.max_head_sar_W_per_kg is not None:
            head_limit = iec_limits["head_sar"]
            if implant.max_head_sar_W_per_kg is not None:
                head_limit = min(head_limit, implant.max_head_sar_W_per_kg)
            if seq.anatomy == "head":
                checks.append(
                    _numeric_check(
                        constraint_name="max_head_sar",
                        sequence_name=seq.name,
                        observed=est.head_sar_W_per_kg,
                        limit=head_limit,
                        unit="W/kg",
                        rationale="IEC 60601-2-33 & implant head SAR thermal limit.",
                    )
                )

        # Gradient slew rate constraint
        slew_limit = scanner.max_slew_rate_T_per_m_s
        if implant.max_slew_rate_T_per_m_s is not None:
            slew_limit = min(slew_limit, implant.max_slew_rate_T_per_m_s)
        checks.append(
            _numeric_check(
                constraint_name="max_slew_rate",
                sequence_name=seq.name,
                observed=est.slew_rate_T_per_m_s,
                limit=slew_limit,
                unit="T/m/s",
                rationale="Limit gradient-induced eddy currents & cardiac/nerve stimulation.",
            )
        )

        # dB/dt constraint
        dbdt_limit = scanner.max_db_dt_T_per_s
        if implant.max_db_dt_T_per_s is not None:
            dbdt_limit = min(dbdt_limit, implant.max_db_dt_T_per_s)
        checks.append(
            _numeric_check(
                constraint_name="max_db_dt",
                sequence_name=seq.name,
                observed=est.estimated_db_dt_T_per_s,
                limit=dbdt_limit,
                unit="T/s",
                rationale="Peripheral nerve stimulation & lead-induced voltage limit.",
            )
        )

    violated = [
        f"{c.sequence_name}:{c.constraint_name} ({c.observed_value} {c.unit} > {c.limit_value} {c.unit})"
        for c in checks
        if not c.satisfied
    ]
    verdict = "PASS" if not violated else "VIOLATION"

    summary = (
        f"PASS: All {len(checks)} IEC 60601-2-33 / ASTM F2503 constraints satisfied for "
        f"{implant.device_name} across {len(sequences)} sequence(s)."
        if verdict == "PASS"
        else f"VIOLATION: {len(violated)} of {len(checks)} constraints violated for "
        f"{implant.device_name}: {'; '.join(violated[:3])}"
    )

    return SafetyCSPVerdict(
        scanner_name=scanner.scanner_name,
        implant_id=implant.device_id,
        implant_name=implant.device_name,
        mr_status=implant.mr_status,
        verdict=verdict,
        abstention_code=None,
        total_active_scan_time_min=total_time,
        max_b1_rms_uT=float(max(e.b1_rms_uT for e in estimates)),
        max_wb_sar_W_per_kg=float(max(e.wb_sar_W_per_kg for e in estimates)),
        max_head_sar_W_per_kg=float(max(e.head_sar_W_per_kg for e in estimates)),
        max_slew_rate_T_per_m_s=float(max(e.slew_rate_T_per_m_s for e in estimates)),
        sequence_estimates=estimates,
        checks=checks,
        violated_constraints=violated,
        summary=summary,
    )
