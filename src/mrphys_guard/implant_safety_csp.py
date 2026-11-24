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
    AbstentionCode,
    AbstentionError,
    ConflictCore,
    ConstraintCheckResult,
    ImplantConditionSpec,
    MRSafetyLabel,
    MRSequenceSpec,
    MultiImplantCSPVerdict,
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

# IEC 60601-2-33 (Ed. 4, 2022) whole-body/head SAR limits [W/kg] & PNS stimulation limits [%]
IEC_SAR_LIMITS: dict[str, dict[str, float]] = {
    "normal": {"wb_sar": 2.0, "head_sar": 3.2, "pns_pct": 80.0},
    "first_level_controlled": {"wb_sar": 4.0, "head_sar": 3.2, "pns_pct": 100.0},
}

# IEC 60601-2-33 default whole-body PNS hyperbolic curve constants
IEC_PNS_RHEOBASE_T_PER_S: float = 20.0
IEC_PNS_CHRONAXIE_MS: float = 0.36


def compute_iec_pns_percentage(
    slew_rate_T_per_m_s: float,
    peak_gradient_mT_per_m: float = 35.0,
    effective_radius_m: float = 0.22,
    rheobase_T_per_s: float = IEC_PNS_RHEOBASE_T_PER_S,
    chronaxie_ms: float = IEC_PNS_CHRONAXIE_MS,
) -> float:
    """Compute IEC 60601-2-33 Peripheral Nerve Stimulation (PNS) percentage [%].

    Uses the standard IEC hyperbolic strength-duration nerve excitation threshold:
        (dB/dt)_threshold = rheobase * (1 + chronaxie / tau_ramp)
        tau_ramp = G_max / slew_rate
        PNS [%] = 100 * (slew_rate * r_eff) / (dB/dt)_threshold

    IEC 60601-2-33 limits: <= 80% in Normal Operating Mode, <= 100% in First-Level Controlled Mode.
    """
    if slew_rate_T_per_m_s <= 0.0:
        return 0.0
    g_max_T_per_m = peak_gradient_mT_per_m * 1e-3
    tau_ramp_ms = (g_max_T_per_m / slew_rate_T_per_m_s) * 1e3
    dbdt_threshold = rheobase_T_per_s * (1.0 + chronaxie_ms / max(tau_ramp_ms, 1e-4))
    dbdt_eff = slew_rate_T_per_m_s * effective_radius_m
    return float(100.0 * dbdt_eff / dbdt_threshold)


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

    # Approximate peak peripheral dB/dt = slew_rate * effective_radius & IEC PNS %
    est_db_dt = sequence.slew_rate_T_per_m_s * scanner_bore_radius_m
    pns_pct = compute_iec_pns_percentage(sequence.slew_rate_T_per_m_s)

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
        pns_stimulation_pct=round(pns_pct, 2),
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

        # IEC 60601-2-33 chronaxie/rheobase PNS stimulation limit
        checks.append(
            _numeric_check(
                constraint_name="iec_pns_stimulation",
                sequence_name=seq.name,
                observed=est.pns_stimulation_pct,
                limit=iec_limits["pns_pct"],
                unit="%",
                rationale="IEC 60601-2-33 hyperbolic nerve stimulation threshold (80% normal / 100% first-level).",
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


def verify_multi_implant_safety_csp(
    scanner: ScannerFieldSpec,
    sequences: list[MRSequenceSpec],
    implants: list[ImplantConditionSpec],
) -> MultiImplantCSPVerdict:
    """Formally verify a protocol against multiple co-existing implants simultaneously.

    Computes the exact intersection of all ASTM F2503 / IEC 60601-2-33 safety envelopes,
    identifies which device governs each tightest ceiling, and emits minimal UNSAT conflict
    cores when devices have disjoint static-field / RF-coil requirements or protocol violations.
    """
    if not implants:
        raise ValueError("At least one ImplantConditionSpec is required.")

    per_verdicts: list[SafetyCSPVerdict] = []
    conflicts: list[ConflictCore] = []

    # Step 1: Evaluate each implant individually (preserving hard ASTM F2503 abstentions)
    for imp in implants:
        v = verify_implant_safety_csp(scanner, sequences, imp, raise_on_abstain=False)
        per_verdicts.append(v)
        if v.verdict == "ABSTAIN":
            conflicts.append(
                ConflictCore(
                    conflict_type="HARD_ASTM_F2503_ABSTAIN",
                    involved_devices=[imp.device_id],
                    constraint_name=v.abstention_code.value if v.abstention_code else "ABSTAIN",
                    explanation=v.summary,
                )
            )

    abstain_verdicts = [v for v in per_verdicts if v.verdict == "ABSTAIN"]
    if abstain_verdicts:
        first_code: AbstentionCode | None = abstain_verdicts[0].abstention_code
        return MultiImplantCSPVerdict(
            scanner_name=scanner.scanner_name,
            implant_ids=[i.device_id for i in implants],
            verdict="ABSTAIN",
            abstention_code=first_code,
            per_implant_verdicts=per_verdicts,
            conflict_cores=conflicts,
            summary=(
                f"ABSTAIN: {len(abstain_verdicts)} co-existing implant(s) triggered hard ASTM F2503 "
                f"abstention ({first_code.value if first_code else 'ABSTAIN'})."
            ),
        )

    # Step 2: Intersect static B0 fields and RF coils across all MR_CONDITIONAL devices
    b0_sets = [
        [round(b, 2) for b in imp.allowed_b0_T]
        for imp in implants
        if imp.allowed_b0_T is not None
    ]
    if b0_sets:
        common_b0 = sorted(set(b0_sets[0]).intersection(*b0_sets[1:]))
        if not common_b0:
            conflicts.append(
                ConflictCore(
                    conflict_type="MUTUALLY_INCOMPATIBLE_IMPLANTS",
                    involved_devices=[i.device_id for i in implants if i.allowed_b0_T is not None],
                    constraint_name="static_field_b0",
                    explanation=(
                        "Co-existing implants have disjoint permitted static magnetic fields B0; "
                        "no single scanner field strength can satisfy all devices simultaneously."
                    ),
                )
            )
    else:
        common_b0 = [scanner.b0_T]

    coil_sets = [set(imp.allowed_rf_coils) for imp in implants if imp.allowed_rf_coils is not None]
    if coil_sets and not set.intersection(*coil_sets):
        conflicts.append(
            ConflictCore(
                conflict_type="MUTUALLY_INCOMPATIBLE_IMPLANTS",
                involved_devices=[i.device_id for i in implants if i.allowed_rf_coils is not None],
                constraint_name="rf_coil_type",
                explanation="Co-existing implants have disjoint permitted transmit RF coil sets.",
            )
        )

    # Step 3: Compute tightest intersected scalar limits & identify governing bottleneck device
    governing: dict[str, str] = {}

    def _min_limit(attr: str, label: str) -> float | None:
        pairs = [(getattr(imp, attr), imp.device_id) for imp in implants if getattr(imp, attr) is not None]
        if not pairs:
            return None
        val, dev = min(pairs, key=lambda p: float(p[0]))
        governing[label] = dev
        return float(val)

    eff_b1 = _min_limit("max_b1_rms_uT", "max_b1_rms_uT")
    eff_wb_sar = _min_limit("max_wb_sar_W_per_kg", "max_wb_sar_W_per_kg")
    eff_slew = _min_limit("max_slew_rate_T_per_m_s", "max_slew_rate_T_per_m_s")
    eff_time = _min_limit("max_active_scan_time_min", "max_active_scan_time_min")
    _min_limit("max_spatial_gradient_T_per_m", "max_spatial_gradient_T_per_m")

    # Step 4: Collect minimal per-constraint violation conflict cores
    for v in per_verdicts:
        for chk in v.checks:
            if not chk.satisfied:
                conflicts.append(
                    ConflictCore(
                        conflict_type="PROTOCOL_CONSTRAINT_VIOLATION",
                        involved_devices=[v.implant_id],
                        constraint_name=f"{chk.sequence_name}:{chk.constraint_name}",
                        explanation=(
                            f"Device '{v.implant_id}' violated {chk.constraint_name} on "
                            f"'{chk.sequence_name}': observed {chk.observed_value} {chk.unit} > "
                            f"limit {chk.limit_value} {chk.unit}."
                        ),
                    )
                )

    verdict = "PASS" if not conflicts else "VIOLATION"
    summary = (
        f"PASS: Protocol simultaneously satisfies all {len(implants)} co-existing implants "
        f"(governing B1+rms: {governing.get('max_b1_rms_uT', 'IEC')}, "
        f"WB SAR: {governing.get('max_wb_sar_W_per_kg', 'IEC')})."
        if verdict == "PASS"
        else f"VIOLATION: {len(conflicts)} conflict core(s) detected across {len(implants)} implants."
    )

    return MultiImplantCSPVerdict(
        scanner_name=scanner.scanner_name,
        implant_ids=[i.device_id for i in implants],
        verdict=verdict,
        abstention_code=None,
        effective_b0_intersection_T=common_b0,
        effective_max_b1_rms_uT=eff_b1,
        effective_max_wb_sar_W_per_kg=eff_wb_sar,
        effective_max_slew_rate_T_per_m_s=eff_slew,
        effective_max_active_time_min=eff_time,
        governing_device_by_limit=governing,
        per_implant_verdicts=per_verdicts,
        conflict_cores=conflicts,
        summary=summary,
    )
