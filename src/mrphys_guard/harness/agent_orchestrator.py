"""Closed-Loop MR Sequence & Safety Co-Design Agent with Independent Critic-Verifier Gate.

Autonomously optimizes MRI pulse-sequence parameters (BW_px, flip angle, refocusing flip angle,
TR, RF pulse duration, EPI echo spacing, slew rate) to jointly satisfy:
1. Sub-millimeter target geometric fidelity (|Delta_r| <= srs_tolerance_mm, e.g. 0.5 mm for SRS)
2. Diagnostic SNR floor (relative_snr >= snr_floor)
3. Zero formal IEC 60601-2-33 / ASTM F2503 implant safety violations

Hard abstains on MR_UNSAFE or UNSPECIFIED implant conditions. Emits a SHA-256 provenance-chained
AAPM TG-284 / IEC 60601-2-33 MR-Physics Safety & Distortion Dossier (Markdown + JSON).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mrphys_guard.distortion_solver import (
    GAMMA_BAR_HZ_PER_T,
    SphericalHarmonicGNLModel,
    compute_3d_distortion_decomposition,
    relative_snr_from_bandwidth,
)
from mrphys_guard.implant_safety_csp import (
    compute_sequence_rf_physics,
    lookup_implant_profile,
    verify_implant_safety_csp,
)
from mrphys_guard.ir import (
    AbstentionCode,
    AbstentionError,
    ImplantConditionSpec,
    MRSequenceSpec,
    RFCoilType,
    SafetyCSPVerdict,
    ScannerFieldSpec,
)
from mrphys_guard.provenance import ProvenanceLedger, gate_implant_labeling


class CoDesignScenario(BaseModel):
    """High-risk clinical MRI / MR-Linac protocol co-design scenario."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    title: str
    clinical_context: str
    scanner: ScannerFieldSpec
    implant: ImplantConditionSpec
    initial_sequences: list[MRSequenceSpec]
    peak_target_delta_b0_ppm: float = Field(
        default=3.0,
        ge=0.0,
        le=40.0,
        description="Peak susceptibility field perturbation in target ROI [ppm]",
    )
    residual_gnl_target_mm: float = Field(
        default=0.18,
        ge=0.0,
        le=5.0,
        description="Residual system GNL error at target ROI after spherical-harmonic correction [mm]",
    )
    srs_tolerance_mm: float = Field(default=0.5, gt=0.0, le=5.0)
    snr_floor: float = Field(default=0.58, gt=0.1, le=2.0)
    enforce_srs_geometry: bool = Field(default=True)


class CoDesignStepTrace(BaseModel):
    """Single logged optimization iteration of the co-design agent."""

    model_config = ConfigDict(extra="forbid")

    iteration: int
    sequence_name: str
    action: str
    target_distortion_mm: float
    relative_snr: float
    b1_rms_uT: float
    wb_sar_W_per_kg: float
    csp_verdict: str


class CriticVerdict(BaseModel):
    """Independent Critic-Verifier Gate sign-off on the agent's proposed protocol."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["APPROVED", "REJECTED", "ABSTAINED"]
    checks_passed: int
    checks_total: int
    max_target_distortion_mm: float
    min_relative_snr: float
    csp_verdict: Literal["PASS", "VIOLATION", "ABSTAIN"]
    findings: list[str] = Field(default_factory=list)


class CoDesignResult(BaseModel):
    """Complete output of the Closed-Loop MR Sequence & Safety Co-Design Agent."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    title: str
    status: Literal["CONVERGED", "ABSTAIN", "INFEASIBLE"]
    abstention_code: AbstentionCode | None = None
    baseline_csp: SafetyCSPVerdict
    optimized_csp: SafetyCSPVerdict | None = None
    baseline_max_target_distortion_mm: float
    optimized_max_target_distortion_mm: float | None = None
    baseline_min_snr: float
    optimized_min_snr: float | None = None
    optimized_sequences: list[MRSequenceSpec] = Field(default_factory=list)
    trace: list[CoDesignStepTrace] = Field(default_factory=list)
    critic: CriticVerdict
    provenance_ledger: dict[str, Any]
    dossier_markdown: str


def _estimate_sequence_target_distortion_mm(
    seq: MRSequenceSpec,
    peak_delta_b0_ppm: float,
    residual_gnl_target_mm: float,
) -> float:
    """Compute exact worst-case target 3D displacement [mm] from Delta_B0(ppm), BW, EPI, and GNL."""
    delta_b0_T = (peak_delta_b0_ppm * 1e-6) * seq.b0_T
    delta_f_hz = GAMMA_BAR_HZ_PER_T * delta_b0_T
    dx_ro = seq.voxel_size_mm[seq.readout_axis]
    disp_ro = abs(delta_f_hz / seq.bandwidth_hz_per_px) * dx_ro

    disp_pe = 0.0
    if seq.sequence_type == "EPI" and seq.echo_spacing_ms is not None:
        dx_pe = seq.voxel_size_mm[seq.phase_encode_axis]
        t_esp_s = seq.echo_spacing_ms * 1e-3
        eff_pe_bw = seq.parallel_accel_factor / (seq.phase_encode_steps * t_esp_s)
        disp_pe = abs(delta_f_hz / eff_pe_bw) * dx_pe

    disp_b0 = float(np.hypot(disp_ro, disp_pe))
    return float(np.hypot(disp_b0, residual_gnl_target_mm))


def _optimize_single_sequence(
    seq: MRSequenceSpec,
    scanner: ScannerFieldSpec,
    implant: ImplantConditionSpec,
    peak_delta_b0_ppm: float,
    residual_gnl_target_mm: float,
    srs_tolerance_mm: float,
    snr_floor: float,
    enforce_srs_geometry: bool,
    trace: list[CoDesignStepTrace],
    iter_counter: list[int],
) -> tuple[MRSequenceSpec, bool]:
    """Physics-guided constrained optimizer for a single MR pulse sequence.

    Jointly tunes:
    - RF coil selection (if scanner supports a permitted coil)
    - Gradient slew rate (clamped to implant/scanner limit)
    - Receiver bandwidth BW_px (raised minimally to satisfy SRS distortion while maximizing SNR)
    - EPI echo spacing & parallel acceleration (when sequence_type == 'EPI')
    - RF thermal parameters (rf_pulse_duration_ms, refocusing_flip_angle_deg, flip_angle_deg, tr_ms)
      using exact scaling B1+rms ~ flip / sqrt(TR * tau_RF) to strictly clear B1+rms and SAR ceilings
      with a 5% engineering safety margin.
    """
    cur = seq.model_copy(deep=True)
    safety_margin = 0.94  # Target <= 94% of hard implant limits for robust clinical headroom

    # 1. Align B0 to scanner B0
    if abs(cur.b0_T - scanner.b0_T) > 1e-6:
        cur = cur.model_copy(update={"b0_T": scanner.b0_T})

    # 2. RF coil feasibility
    if implant.allowed_rf_coils and cur.rf_coil not in implant.allowed_rf_coils:
        common = [c for c in implant.allowed_rf_coils if c in scanner.available_coils]
        if not common:
            return cur, False
        cur = cur.model_copy(update={"rf_coil": common[0]})
        _log_step(cur, scanner, implant, peak_delta_b0_ppm, residual_gnl_target_mm,
                  f"Switched RF transmit coil to {common[0].value}", trace, iter_counter)

    # 3. Gradient slew rate & dB/dt clamping
    slew_cap = scanner.max_slew_rate_T_per_m_s
    if implant.max_slew_rate_T_per_m_s is not None:
        slew_cap = min(slew_cap, implant.max_slew_rate_T_per_m_s * safety_margin)
    dbdt_cap = scanner.max_db_dt_T_per_s
    if implant.max_db_dt_T_per_s is not None:
        dbdt_cap = min(dbdt_cap, implant.max_db_dt_T_per_s * safety_margin)
    slew_cap = min(slew_cap, dbdt_cap / 0.30)
    if cur.slew_rate_T_per_m_s > slew_cap:
        cur = cur.model_copy(update={"slew_rate_T_per_m_s": round(float(slew_cap), 2)})
        _log_step(cur, scanner, implant, peak_delta_b0_ppm, residual_gnl_target_mm,
                  f"Clamped slew rate to {cur.slew_rate_T_per_m_s:.1f} T/m/s (dB/dt safe)",
                  trace, iter_counter)

    # 4. Geometric fidelity vs SNR bandwidth co-design
    if enforce_srs_geometry:
        if residual_gnl_target_mm >= srs_tolerance_mm:
            return cur, False
        allowed_b0_disp = float(np.sqrt(max(srs_tolerance_mm**2 - residual_gnl_target_mm**2, 1e-12)))
        delta_f_hz = GAMMA_BAR_HZ_PER_T * (peak_delta_b0_ppm * 1e-6) * cur.b0_T
        dx_ro = cur.voxel_size_mm[cur.readout_axis]

        # If EPI, boost parallel acceleration & tighten echo spacing if needed
        if cur.sequence_type == "EPI" and cur.echo_spacing_ms is not None:
            cur = cur.model_copy(
                update={
                    "parallel_accel_factor": min(4.0, max(cur.parallel_accel_factor, 2.0)),
                    "echo_spacing_ms": min(cur.echo_spacing_ms, 0.45),
                }
            )
            _log_step(cur, scanner, implant, peak_delta_b0_ppm, residual_gnl_target_mm,
                      "Tightened EPI echo spacing and raised parallel acceleration",
                      trace, iter_counter)

        # Compute minimum readout BW that keeps |Delta_r| <= 0.96 * srs_tolerance_mm
        target_b0_disp = allowed_b0_disp * 0.95
        min_bw_needed = (delta_f_hz * dx_ro) / max(target_b0_disp, 1e-6)
        if cur.bandwidth_hz_per_px < min_bw_needed:
            new_bw = min(2000.0, float(np.ceil(min_bw_needed / 5.0) * 5.0))
            cur = cur.model_copy(update={"bandwidth_hz_per_px": new_bw})
            _log_step(cur, scanner, implant, peak_delta_b0_ppm, residual_gnl_target_mm,
                      f"Raised receiver BW_px to {new_bw:.1f} Hz/px for <= {srs_tolerance_mm:.2f} mm SRS fidelity",
                      trace, iter_counter)

        # Check SNR floor; if slightly below floor, check if voxel/parallel accel allows headroom
        snr_now = relative_snr_from_bandwidth(
            cur.bandwidth_hz_per_px,
            voxel_size_mm=cur.voxel_size_mm,
            parallel_accel_factor=cur.parallel_accel_factor,
        )
        if snr_now < snr_floor and cur.parallel_accel_factor > 1.0 and cur.sequence_type != "EPI":
            cur = cur.model_copy(update={"parallel_accel_factor": 1.0})
            _log_step(cur, scanner, implant, peak_delta_b0_ppm, residual_gnl_target_mm,
                      "Disabled parallel acceleration to recover SNR above floor",
                      trace, iter_counter)

    # 5. Exact RF thermal co-design (B1+rms, WB SAR, Head SAR)
    # Determine effective B1+rms cap implied by all RF/SAR limits
    b1_caps: list[float] = []
    if implant.max_b1_rms_uT is not None:
        b1_caps.append(implant.max_b1_rms_uT * safety_margin)

    wb_sar_lim = 2.0 if scanner.operating_mode == "normal" else 4.0
    if implant.max_wb_sar_W_per_kg is not None:
        wb_sar_lim = min(wb_sar_lim, implant.max_wb_sar_W_per_kg)
    b1_from_wb = float(np.sqrt((wb_sar_lim * safety_margin) / (0.0868 * (cur.b0_T**2))))
    b1_caps.append(b1_from_wb)

    if cur.anatomy == "head":
        head_sar_lim = 3.2
        if implant.max_head_sar_W_per_kg is not None:
            head_sar_lim = min(head_sar_lim, implant.max_head_sar_W_per_kg)
        b1_from_head = float(np.sqrt((head_sar_lim * safety_margin) / (0.1400 * (cur.b0_T**2))))
        b1_caps.append(b1_from_head)

    target_b1_uT = min(b1_caps)

    est = compute_sequence_rf_physics(cur)
    if est.b1_rms_uT > target_b1_uT:
        # Step 5a: Lengthen RF pulse duration tau_RF (zero contrast penalty, reduces B1+rms ~ 1/sqrt(tau_RF))
        ratio_sq = (est.b1_rms_uT / target_b1_uT) ** 2
        desired_tau = min(6.0, cur.rf_pulse_duration_ms * ratio_sq)
        if desired_tau > cur.rf_pulse_duration_ms + 0.05:
            cur = cur.model_copy(update={"rf_pulse_duration_ms": round(float(desired_tau), 2)})
            _log_step(cur, scanner, implant, peak_delta_b0_ppm, residual_gnl_target_mm,
                      f"Extended RF pulse duration tau_RF to {cur.rf_pulse_duration_ms:.2f} ms (low-SAR RF)",
                      trace, iter_counter)

    est = compute_sequence_rf_physics(cur)
    if est.b1_rms_uT > target_b1_uT and cur.sequence_type in {"TSE", "FSE"} and cur.echo_train_length > 1:
        # Step 5b: Reduce refocusing flip angle (clinical hyperecho / low-SAR TSE down to 110 deg)
        new_ref = max(110.0, cur.refocusing_flip_angle_deg * (target_b1_uT / est.b1_rms_uT))
        if new_ref < cur.refocusing_flip_angle_deg - 1.0:
            cur = cur.model_copy(update={"refocusing_flip_angle_deg": round(float(new_ref), 1)})
            _log_step(cur, scanner, implant, peak_delta_b0_ppm, residual_gnl_target_mm,
                      f"Reduced TSE refocusing flip angle to {cur.refocusing_flip_angle_deg:.1f} deg",
                      trace, iter_counter)

    est = compute_sequence_rf_physics(cur)
    if est.b1_rms_uT > target_b1_uT:
        # Step 5c: Reduce excitation flip angle within diagnostic range (down to 12 deg for GRE/bSSFP, 65 deg for SE)
        min_flip = 12.0 if cur.sequence_type in {"GRE", "bSSFP", "MPRAGE", "Dixon"} else 65.0
        scaled_flip = cur.flip_angle_deg * (target_b1_uT / est.b1_rms_uT)
        new_flip = max(min_flip, scaled_flip)
        if new_flip < cur.flip_angle_deg - 0.5:
            cur = cur.model_copy(update={"flip_angle_deg": round(float(new_flip), 1)})
            _log_step(cur, scanner, implant, peak_delta_b0_ppm, residual_gnl_target_mm,
                      f"Reduced excitation flip angle to {cur.flip_angle_deg:.1f} deg",
                      trace, iter_counter)

    est = compute_sequence_rf_physics(cur)
    if est.b1_rms_uT > target_b1_uT:
        # Step 5d: Increase TR (exact 1/sqrt(TR) scaling guarantees B1+rms convergence!)
        needed_tr = cur.tr_ms * ((est.b1_rms_uT / target_b1_uT) ** 2)
        new_tr = min(15000.0, float(np.ceil(needed_tr * 10.0) / 10.0))
        if new_tr > cur.tr_ms:
            cur = cur.model_copy(update={"tr_ms": new_tr})
            _log_step(cur, scanner, implant, peak_delta_b0_ppm, residual_gnl_target_mm,
                      f"Increased TR to {cur.tr_ms:.1f} ms to satisfy B1+rms <= {target_b1_uT:.2f} uT",
                      trace, iter_counter)

    # Clear stale DICOM-reported fields on newly synthesized sequence so physics reflects updated params
    cur = cur.model_copy(update={"reported_sar_W_per_kg": None, "reported_b1_rms_uT": None})

    # Verify single-sequence constraints
    dist_final = _estimate_sequence_target_distortion_mm(cur, peak_delta_b0_ppm, residual_gnl_target_mm)
    snr_final = relative_snr_from_bandwidth(
        cur.bandwidth_hz_per_px,
        voxel_size_mm=cur.voxel_size_mm,
        parallel_accel_factor=cur.parallel_accel_factor,
    )
    geom_ok = (not enforce_srs_geometry) or (dist_final <= srs_tolerance_mm + 1e-9)
    snr_ok = snr_final >= snr_floor - 1e-9
    return cur, bool(geom_ok and snr_ok)


def _log_step(
    seq: MRSequenceSpec,
    scanner: ScannerFieldSpec,
    implant: ImplantConditionSpec,
    peak_delta_b0_ppm: float,
    residual_gnl_target_mm: float,
    action: str,
    trace: list[CoDesignStepTrace],
    iter_counter: list[int],
) -> None:
    iter_counter[0] += 1
    est = compute_sequence_rf_physics(seq)
    dist = _estimate_sequence_target_distortion_mm(seq, peak_delta_b0_ppm, residual_gnl_target_mm)
    verdict = verify_implant_safety_csp(scanner, [seq], implant).verdict
    trace.append(
        CoDesignStepTrace(
            iteration=iter_counter[0],
            sequence_name=seq.name,
            action=action,
            target_distortion_mm=round(dist, 4),
            relative_snr=round(est.relative_snr, 4),
            b1_rms_uT=round(est.b1_rms_uT, 4),
            wb_sar_W_per_kg=round(est.wb_sar_W_per_kg, 4),
            csp_verdict=verdict,
        )
    )


def independent_critic_verify(
    scenario: CoDesignScenario,
    proposed_sequences: list[MRSequenceSpec],
) -> CriticVerdict:
    """Independent Critic-Verifier Gate re-evaluating all physics and IEC 60601-2-33 constraints."""
    findings: list[str] = []

    try:
        gate_implant_labeling(scenario.implant)
    except AbstentionError as exc:
        return CriticVerdict(
            status="ABSTAINED",
            checks_passed=0,
            checks_total=1,
            max_target_distortion_mm=0.0,
            min_relative_snr=0.0,
            csp_verdict="ABSTAIN",
            findings=[f"ASTM F2503 hard abstention confirmed: {exc.code.value}"],
        )

    csp = verify_implant_safety_csp(scenario.scanner, proposed_sequences, scenario.implant)
    dists = [
        _estimate_sequence_target_distortion_mm(
            s, scenario.peak_target_delta_b0_ppm, scenario.residual_gnl_target_mm
        )
        for s in proposed_sequences
    ]
    snrs = [
        relative_snr_from_bandwidth(
            s.bandwidth_hz_per_px,
            voxel_size_mm=s.voxel_size_mm,
            parallel_accel_factor=s.parallel_accel_factor,
        )
        for s in proposed_sequences
    ]

    max_dist = float(max(dists))
    min_snr = float(min(snrs))

    checks_total = len(csp.checks) + (2 if scenario.enforce_srs_geometry else 1)
    checks_passed = sum(1 for c in csp.checks if c.satisfied)

    if csp.verdict != "PASS":
        findings.append(f"CSP failed with {len(csp.violated_constraints)} violation(s).")
    else:
        findings.append(f"Formal IEC 60601-2-33 CSP verified: {len(csp.checks)}/{len(csp.checks)} PASS.")

    if scenario.enforce_srs_geometry:
        if max_dist <= scenario.srs_tolerance_mm + 1e-9:
            checks_passed += 1
            findings.append(
                f"SRS geometric fidelity verified: max |Delta_r| = {max_dist:.3f} mm <= {scenario.srs_tolerance_mm:.2f} mm."
            )
        else:
            findings.append(
                f"SRS geometric fidelity FAILED: max |Delta_r| = {max_dist:.3f} mm > {scenario.srs_tolerance_mm:.2f} mm."
            )

    if min_snr >= scenario.snr_floor - 1e-9:
        checks_passed += 1
        findings.append(f"Diagnostic SNR floor verified: min relSNR = {min_snr:.3f} >= {scenario.snr_floor:.2f}.")
    else:
        findings.append(f"Diagnostic SNR floor FAILED: min relSNR = {min_snr:.3f} < {scenario.snr_floor:.2f}.")

    status: Literal["APPROVED", "REJECTED", "ABSTAINED"] = (
        "APPROVED" if checks_passed == checks_total and csp.verdict == "PASS" else "REJECTED"
    )
    return CriticVerdict(
        status=status,
        checks_passed=checks_passed,
        checks_total=checks_total,
        max_target_distortion_mm=round(max_dist, 4),
        min_relative_snr=round(min_snr, 4),
        csp_verdict=csp.verdict,
        findings=findings,
    )


class MRSequenceCoDesignAgent:
    """Closed-Loop MR Sequence & Safety Co-Design Agent."""

    def run(
        self,
        scenario: CoDesignScenario,
        chi_si: np.ndarray | None = None,
        gnl_model: SphericalHarmonicGNLModel | None = None,
        target_roi_mask: np.ndarray | None = None,
        body_mask: np.ndarray | None = None,
    ) -> CoDesignResult:
        """Execute closed-loop co-design, independent Critic verification, and dossier emission."""
        ledger = ProvenanceLedger()

        # If a 3D susceptibility volume is provided, derive exact peak target Delta_B0 from 3D solver!
        effective_scenario = scenario
        if chi_si is not None:
            decomp, fields = compute_3d_distortion_decomposition(
                chi_si=chi_si,
                sequence=scenario.initial_sequences[0],
                gnl_model=gnl_model,
                target_roi_mask=target_roi_mask,
                body_mask=body_mask,
                srs_tolerance_mm=scenario.srs_tolerance_mm,
            )
            roi_mask = target_roi_mask if target_roi_mask is not None else np.ones(chi_si.shape, dtype=bool)
            peak_ppm = float(np.max(np.abs(fields["delta_b0_ppm"][roi_mask])))
            gnl_roi = float(np.max(fields["gnl_mag_mm"][roi_mask])) if gnl_model is not None else 0.0
            effective_scenario = scenario.model_copy(
                update={
                    "peak_target_delta_b0_ppm": peak_ppm,
                    "residual_gnl_target_mm": min(gnl_roi, scenario.residual_gnl_target_mm),
                }
            )
            ledger.append(
                operation="3d_susceptibility_gnl_baseline",
                inputs={"scenario_id": scenario.scenario_id, "shape": list(chi_si.shape)},
                outputs=decomp.model_dump(mode="json"),
                data_origin="SYNTHETIC_PHYSICS_FIXTURE",
            )

        # Baseline evaluation
        baseline_csp = verify_implant_safety_csp(
            effective_scenario.scanner,
            effective_scenario.initial_sequences,
            effective_scenario.implant,
            raise_on_abstain=False,
        )
        base_dists = [
            _estimate_sequence_target_distortion_mm(
                s,
                effective_scenario.peak_target_delta_b0_ppm,
                effective_scenario.residual_gnl_target_mm,
            )
            for s in effective_scenario.initial_sequences
        ]
        base_snrs = [
            relative_snr_from_bandwidth(
                s.bandwidth_hz_per_px,
                voxel_size_mm=s.voxel_size_mm,
                parallel_accel_factor=s.parallel_accel_factor,
            )
            for s in effective_scenario.initial_sequences
        ]
        base_max_dist = float(max(base_dists))
        base_min_snr = float(min(base_snrs))

        ledger.append(
            operation="baseline_csp_audit",
            inputs=effective_scenario.model_dump(mode="json"),
            outputs=baseline_csp.model_dump(mode="json"),
            data_origin="SYNTHETIC_PHYSICS_FIXTURE",
        )

        # Hard ASTM F2503 ABSTAIN gate
        if baseline_csp.verdict == "ABSTAIN":
            critic = independent_critic_verify(effective_scenario, effective_scenario.initial_sequences)
            ledger.append(
                operation="critic_verifier_abstain",
                inputs={"implant_id": effective_scenario.implant.device_id},
                outputs=critic.model_dump(mode="json"),
                data_origin="SYNTHETIC_PHYSICS_FIXTURE",
            )
            dossier = self._render_dossier(
                scenario=effective_scenario,
                status="ABSTAIN",
                baseline_csp=baseline_csp,
                optimized_csp=None,
                base_dist=base_max_dist,
                opt_dist=None,
                base_snr=base_min_snr,
                opt_snr=None,
                optimized_sequences=[],
                critic=critic,
                ledger_dict=ledger.to_json_dict(),
            )
            return CoDesignResult(
                scenario_id=effective_scenario.scenario_id,
                title=effective_scenario.title,
                status="ABSTAIN",
                abstention_code=baseline_csp.abstention_code,
                baseline_csp=baseline_csp,
                optimized_csp=None,
                baseline_max_target_distortion_mm=round(base_max_dist, 4),
                optimized_max_target_distortion_mm=None,
                baseline_min_snr=round(base_min_snr, 4),
                optimized_min_snr=None,
                optimized_sequences=[],
                trace=[],
                critic=critic,
                provenance_ledger=ledger.to_json_dict(),
                dossier_markdown=dossier,
            )

        # Closed-loop optimization over all sequences
        trace: list[CoDesignStepTrace] = []
        iter_counter = [0]
        optimized_seqs: list[MRSequenceSpec] = []
        all_seq_feasible = True

        for seq in effective_scenario.initial_sequences:
            opt_seq, seq_ok = _optimize_single_sequence(
                seq=seq,
                scanner=effective_scenario.scanner,
                implant=effective_scenario.implant,
                peak_delta_b0_ppm=effective_scenario.peak_target_delta_b0_ppm,
                residual_gnl_target_mm=effective_scenario.residual_gnl_target_mm,
                srs_tolerance_mm=effective_scenario.srs_tolerance_mm,
                snr_floor=effective_scenario.snr_floor,
                enforce_srs_geometry=effective_scenario.enforce_srs_geometry,
                trace=trace,
                iter_counter=iter_counter,
            )
            optimized_seqs.append(opt_seq)
            all_seq_feasible = all_seq_feasible and seq_ok

        # Protocol-level active scan time capping if cumulative time exceeds implant limit
        max_time = effective_scenario.implant.max_active_scan_time_min
        if max_time is not None:
            total_t = sum(s.scan_duration_min for s in optimized_seqs)
            if total_t > max_time * 0.95:
                scale_t = (max_time * 0.92) / total_t
                optimized_seqs = [
                    s.model_copy(update={"scan_duration_min": round(max(0.5, s.scan_duration_min * scale_t), 2)})
                    for s in optimized_seqs
                ]
                _log_step(
                    optimized_seqs[-1],
                    effective_scenario.scanner,
                    effective_scenario.implant,
                    effective_scenario.peak_target_delta_b0_ppm,
                    effective_scenario.residual_gnl_target_mm,
                    f"Scaled active scan durations to {sum(s.scan_duration_min for s in optimized_seqs):.1f} min <= {max_time:.1f} min",
                    trace,
                    iter_counter,
                )

        optimized_csp = verify_implant_safety_csp(
            effective_scenario.scanner,
            optimized_seqs,
            effective_scenario.implant,
            raise_on_abstain=False,
        )
        critic = independent_critic_verify(effective_scenario, optimized_seqs)

        status: Literal["CONVERGED", "ABSTAIN", "INFEASIBLE"] = (
            "CONVERGED" if (all_seq_feasible and critic.status == "APPROVED") else "INFEASIBLE"
        )

        ledger.append(
            operation="codesign_and_critic_verification",
            inputs={"optimized_sequences": [s.model_dump(mode="json") for s in optimized_seqs]},
            outputs={
                "status": status,
                "optimized_csp": optimized_csp.model_dump(mode="json"),
                "critic": critic.model_dump(mode="json"),
            },
            data_origin="SYNTHETIC_PHYSICS_FIXTURE",
        )

        dossier = self._render_dossier(
            scenario=effective_scenario,
            status=status,
            baseline_csp=baseline_csp,
            optimized_csp=optimized_csp,
            base_dist=base_max_dist,
            opt_dist=critic.max_target_distortion_mm,
            base_snr=base_min_snr,
            opt_snr=critic.min_relative_snr,
            optimized_sequences=optimized_seqs,
            critic=critic,
            ledger_dict=ledger.to_json_dict(),
        )

        return CoDesignResult(
            scenario_id=effective_scenario.scenario_id,
            title=effective_scenario.title,
            status=status,
            abstention_code=None,
            baseline_csp=baseline_csp,
            optimized_csp=optimized_csp,
            baseline_max_target_distortion_mm=round(base_max_dist, 4),
            optimized_max_target_distortion_mm=critic.max_target_distortion_mm,
            baseline_min_snr=round(base_min_snr, 4),
            optimized_min_snr=critic.min_relative_snr,
            optimized_sequences=optimized_seqs,
            trace=trace,
            critic=critic,
            provenance_ledger=ledger.to_json_dict(),
            dossier_markdown=dossier,
        )

    @staticmethod
    def _render_dossier(
        scenario: CoDesignScenario,
        status: str,
        baseline_csp: SafetyCSPVerdict,
        optimized_csp: SafetyCSPVerdict | None,
        base_dist: float,
        opt_dist: float | None,
        base_snr: float,
        opt_snr: float | None,
        optimized_sequences: list[MRSequenceSpec],
        critic: CriticVerdict,
        ledger_dict: dict[str, Any],
    ) -> str:
        lines = [
            "# AAPM TG-284 / IEC 60601-2-33 MR-Physics Safety & Distortion Dossier",
            "",
            "> **RESEARCH SOFTWARE — NOT A MEDICAL DEVICE.** For computational physics verification only.",
            "",
            f"- **Scenario ID:** `{scenario.scenario_id}`",
            f"- **Title:** {scenario.title}",
            f"- **Scanner:** {scenario.scanner.scanner_name} ($B_0 = {scenario.scanner.b0_T}\\text{{ T}}$, $|dB/dx|_{{\\max}} = {scenario.scanner.max_spatial_gradient_T_per_m}\\text{{ T/m}}$)",
            f"- **Implant:** {scenario.implant.device_name} (`{scenario.implant.mr_status.value}`)",
            f"- **Co-Design Status:** **{status}** | **Critic-Verifier Gate:** **{critic.status}** (`{critic.checks_passed}/{critic.checks_total}`)",
            "",
            "## Before vs. After Protocol Metrics",
            "",
            "| Metric | Baseline Protocol | Co-Designed Protocol | Target / Limit |",
            "| :--- | :---: | :---: | :---: |",
            f"| Formal CSP Verdict | `{baseline_csp.verdict}` | `{optimized_csp.verdict if optimized_csp else 'ABSTAIN'}` | `PASS` |",
            f"| Max Target Distortion | {base_dist:.3f} mm | {f'{opt_dist:.3f} mm' if opt_dist is not None else 'N/A'} | <= {scenario.srs_tolerance_mm:.2f} mm |",
            f"| Min Relative SNR | {base_snr:.3f} | {f'{opt_snr:.3f}' if opt_snr is not None else 'N/A'} | >= {scenario.snr_floor:.2f} |",
            f"| Peak $B_{{1+\\text{{rms}}}}$ | {baseline_csp.max_b1_rms_uT:.2f} uT | {f'{optimized_csp.max_b1_rms_uT:.2f} uT' if optimized_csp else 'N/A'} | <= {scenario.implant.max_b1_rms_uT} uT |",
            f"| Peak Whole-Body SAR | {baseline_csp.max_wb_sar_W_per_kg:.2f} W/kg | {f'{optimized_csp.max_wb_sar_W_per_kg:.2f} W/kg' if optimized_csp else 'N/A'} | <= {scenario.implant.max_wb_sar_W_per_kg} W/kg |",
            f"| Cumulative Active Time | {baseline_csp.total_active_scan_time_min:.1f} min | {f'{optimized_csp.total_active_scan_time_min:.1f} min' if optimized_csp else 'N/A'} | <= {scenario.implant.max_active_scan_time_min} min |",
            "",
            "## Critic-Verifier Findings",
            "",
        ]
        for finding in critic.findings:
            lines.append(f"- {finding}")

        if optimized_sequences:
            lines.extend([
                "",
                "## Co-Designed Sequence Parameters",
                "",
                "| Sequence | Type | BW (Hz/px) | Flip (deg) | Refocus (deg) | TR (ms) | tau_RF (ms) | Slew (T/m/s) |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ])
            for s in optimized_sequences:
                lines.append(
                    f"| {s.name} | {s.sequence_type} | {s.bandwidth_hz_per_px:.1f} | "
                    f"{s.flip_angle_deg:.1f} | {s.refocusing_flip_angle_deg:.1f} | "
                    f"{s.tr_ms:.1f} | {s.rf_pulse_duration_ms:.2f} | {s.slew_rate_T_per_m_s:.1f} |"
                )

        last_hash = (
            ledger_dict["records"][-1]["record_sha256"] if ledger_dict.get("records") else "N/A"
        )
        lines.extend([
            "",
            "## Cryptographic Provenance",
            "",
            f"- **Chain Valid:** `{ledger_dict.get('chain_valid')}` ({ledger_dict.get('record_count')} records)",
            f"- **Terminal Record SHA-256:** `{last_hash}`",
        ])
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Pre-Built Flagship Clinical Scenarios
# ---------------------------------------------------------------------------


class ParetoProtocolCandidate(BaseModel):
    """One verified operating point on the Distortion–CNR–SAR clinical Pareto frontier."""

    model_config = ConfigDict(extra="forbid")

    profile_name: Literal["LOWEST_DISTORTION", "BALANCED_CLINICAL_CNR", "LOWEST_RF_SAR"]
    max_target_distortion_mm: float
    min_relative_snr: float
    mean_bloch_cnr: float
    max_b1_rms_uT: float
    max_wb_sar_W_per_kg: float
    critic_status: Literal["APPROVED", "REJECTED", "ABSTAINED"]
    sequences: list[MRSequenceSpec]


def compute_bloch_tissue_cnr(
    seq: MRSequenceSpec,
    t1_tissue_a_ms: float | None = None,
    t2_tissue_a_ms: float | None = None,
    t1_tissue_b_ms: float | None = None,
    t2_tissue_b_ms: float | None = None,
) -> float:
    """Compute steady-state Bloch-equation tissue Contrast-to-Noise Ratio (CNR) proxy.

    Default relaxation times at 1.5T (literature values):
    - Pelvis/prostate: peripheral zone (T1=1317 ms, T2=110 ms) vs. tumor (T1=1050 ms, T2=68 ms)
    - Brain/head: gray matter (T1=1120 ms, T2=95 ms) vs. white matter (T1=660 ms, T2=72 ms)
    - Cardiac: myocardium (T1=1030 ms, T2=50 ms) vs. LV blood (T1=1480 ms, T2=250 ms)

    Combines exact spoiled-GRE / spin-echo Bloch steady-state signal contrast |S_A - S_B|
    with exact bandwidth/voxel relative SNR scaling (proportional to 1/sqrt(BW * R_PAT)).
    """
    defaults = {
        "pelvis": (1317.0, 110.0, 1050.0, 68.0),
        "head": (1120.0, 95.0, 660.0, 72.0),
        "cardiac": (1480.0, 250.0, 1030.0, 50.0),
    }
    da_t1, da_t2, db_t1, db_t2 = defaults.get(seq.anatomy, defaults["pelvis"])
    t1a = t1_tissue_a_ms if t1_tissue_a_ms is not None else da_t1
    t2a = t2_tissue_a_ms if t2_tissue_a_ms is not None else da_t2
    t1b = t1_tissue_b_ms if t1_tissue_b_ms is not None else db_t1
    t2b = t2_tissue_b_ms if t2_tissue_b_ms is not None else db_t2

    alpha = float(np.deg2rad(seq.flip_angle_deg))
    tr = seq.tr_ms
    te = seq.te_ms

    def _signal(t1: float, t2: float) -> float:
        e1 = float(np.exp(-tr / t1))
        e2 = float(np.exp(-te / t2))
        if seq.sequence_type in {"TSE", "FSE", "SE"}:
            return float(np.sin(alpha) * (1.0 - e1) * e2)
        if seq.sequence_type == "bSSFP":
            # Freeman-Hill steady-state bSSFP signal
            denom = 1.0 - (e1 - e2) * float(np.cos(alpha)) - e1 * e2
            return float(np.sin(alpha) * (1.0 - e1) * np.sqrt(e2) / max(denom, 1e-6))
        # Spoiled GRE / MPRAGE / Dixon steady-state Ernst equation
        denom = 1.0 - e1 * float(np.cos(alpha))
        return float(np.sin(alpha) * (1.0 - e1) / max(denom, 1e-6) * e2)

    contrast = abs(_signal(t1a, t2a) - _signal(t1b, t2b))
    snr = relative_snr_from_bandwidth(
        seq.bandwidth_hz_per_px,
        voxel_size_mm=seq.voxel_size_mm,
        parallel_accel_factor=seq.parallel_accel_factor,
    )
    return float(contrast * snr * 100.0)


def generate_pareto_protocol_candidates(
    scenario: CoDesignScenario,
) -> list[ParetoProtocolCandidate]:
    """Generate a 3-point verified Pareto frontier (Lowest Distortion, Balanced CNR, Lowest RF/SAR).

    Every candidate is independently verified by the Critic-Verifier Gate.
    """
    agent = MRSequenceCoDesignAgent()
    base_res = agent.run(scenario)
    if base_res.status != "CONVERGED" or not base_res.optimized_sequences:
        return []

    specs: list[tuple[Literal["LOWEST_DISTORTION", "BALANCED_CLINICAL_CNR", "LOWEST_RF_SAR"], float, float]] = [
        ("LOWEST_DISTORTION", 1.25, 1.00),
        ("BALANCED_CLINICAL_CNR", 1.00, 1.00),
        ("LOWEST_RF_SAR", 1.00, 1.45),
    ]
    candidates: list[ParetoProtocolCandidate] = []
    for label, bw_mult, tr_mult in specs:
        tweaked: list[MRSequenceSpec] = []
        for s in base_res.optimized_sequences:
            new_bw = min(2000.0, round(s.bandwidth_hz_per_px * bw_mult, 1))
            new_tr = min(15000.0, round(s.tr_ms * tr_mult, 1))
            new_tau = min(6.0, round(s.rf_pulse_duration_ms * (1.20 if label == "LOWEST_RF_SAR" else 1.0), 2))
            tweaked.append(
                s.model_copy(
                    update={
                        "bandwidth_hz_per_px": new_bw,
                        "tr_ms": new_tr,
                        "rf_pulse_duration_ms": new_tau,
                    }
                )
            )
        critic = independent_critic_verify(scenario, tweaked)
        csp = verify_implant_safety_csp(scenario.scanner, tweaked, scenario.implant)
        mean_cnr = float(np.mean([compute_bloch_tissue_cnr(s) for s in tweaked]))
        candidates.append(
            ParetoProtocolCandidate(
                profile_name=label,
                max_target_distortion_mm=critic.max_target_distortion_mm,
                min_relative_snr=critic.min_relative_snr,
                mean_bloch_cnr=round(mean_cnr, 3),
                max_b1_rms_uT=round(csp.max_b1_rms_uT, 4),
                max_wb_sar_W_per_kg=round(csp.max_wb_sar_W_per_kg, 4),
                critic_status=critic.status,
                sequences=tweaked,
            )
        )
    return candidates


def build_prostate_srs_hip_gas_scenario() -> CoDesignScenario:
    """1.5T MR-only prostate SRS with bilateral Ti hip prostheses and rectal gas pocket."""
    scanner = ScannerFieldSpec(
        scanner_name="1.5T MR-Linac (High-Field Adaptive RT)",
        b0_T=1.5,
        max_spatial_gradient_T_per_m=18.5,
        max_slew_rate_T_per_m_s=200.0,
        max_db_dt_T_per_s=65.0,
        operating_mode="normal",
    )
    implant = lookup_implant_profile("generic_hip_prosthesis_ti")
    sequences = [
        MRSequenceSpec(
            name="T2w_3D_TSE_Pelvis",
            sequence_type="TSE",
            b0_T=1.5,
            bandwidth_hz_per_px=185.0,
            voxel_size_mm=(1.0, 1.0, 1.5),
            flip_angle_deg=90.0,
            refocusing_flip_angle_deg=165.0,
            echo_train_length=24,
            tr_ms=320.0,
            te_ms=90.0,
            rf_pulse_duration_ms=1.2,
            slew_rate_T_per_m_s=175.0,
            scan_duration_min=5.5,
            rf_coil=RFCoilType.WHOLE_BODY_CP,
            anatomy="pelvis",
        ),
        MRSequenceSpec(
            name="Dixon_3D_sCT_Anchor",
            sequence_type="Dixon",
            b0_T=1.5,
            bandwidth_hz_per_px=240.0,
            voxel_size_mm=(1.2, 1.2, 2.0),
            flip_angle_deg=25.0,
            tr_ms=6.5,
            te_ms=2.3,
            rf_pulse_duration_ms=0.7,
            slew_rate_T_per_m_s=180.0,
            scan_duration_min=3.0,
            rf_coil=RFCoilType.WHOLE_BODY_CP,
            anatomy="pelvis",
        ),
    ]
    return CoDesignScenario(
        scenario_id="prostate_srs_hip_gas_1p5t",
        title="1.5T MR-Only Prostate SRS with Bilateral Ti Hip Prostheses & Rectal Gas",
        clinical_context=(
            "High-precision MR-only stereotactic prostate radiotherapy requiring <= 0.5 mm target "
            "geometric fidelity adjacent to rectal gas and bilateral titanium hip arthroplasties."
        ),
        scanner=scanner,
        implant=implant,
        initial_sequences=sequences,
        peak_target_delta_b0_ppm=3.1,
        residual_gnl_target_mm=0.16,
        srs_tolerance_mm=0.50,
        snr_floor=0.60,
        enforce_srs_geometry=True,
    )


def build_pacemaker_cardiac_brain_scenario() -> CoDesignScenario:
    """1.5T clinical cardiac + brain MRI protocol with MR-conditional dual-chamber pacemaker."""
    scanner = ScannerFieldSpec(
        scanner_name="1.5T Diagnostic MRI (IEC Normal Mode)",
        b0_T=1.5,
        max_spatial_gradient_T_per_m=17.0,
        max_slew_rate_T_per_m_s=200.0,
        max_db_dt_T_per_s=60.0,
        operating_mode="normal",
    )
    implant = lookup_implant_profile("generic_cardiac_pacemaker_1p5t")
    sequences = [
        MRSequenceSpec(
            name="Cardiac_Cine_bSSFP",
            sequence_type="bSSFP",
            b0_T=1.5,
            bandwidth_hz_per_px=650.0,
            voxel_size_mm=(1.4, 1.4, 6.0),
            flip_angle_deg=65.0,
            tr_ms=3.2,
            te_ms=1.5,
            rf_pulse_duration_ms=0.6,
            slew_rate_T_per_m_s=195.0,
            scan_duration_min=8.0,
            rf_coil=RFCoilType.WHOLE_BODY_CP,
            anatomy="cardiac",
        ),
        MRSequenceSpec(
            name="Brain_T2_TSE_HighSAR",
            sequence_type="TSE",
            b0_T=1.5,
            bandwidth_hz_per_px=220.0,
            voxel_size_mm=(1.0, 1.0, 3.0),
            flip_angle_deg=90.0,
            refocusing_flip_angle_deg=170.0,
            echo_train_length=20,
            tr_ms=420.0,
            te_ms=95.0,
            rf_pulse_duration_ms=1.1,
            slew_rate_T_per_m_s=180.0,
            scan_duration_min=6.5,
            rf_coil=RFCoilType.WHOLE_BODY_CP,
            anatomy="head",
        ),
        MRSequenceSpec(
            name="Brain_3D_MPRAGE_T1",
            sequence_type="MPRAGE",
            b0_T=1.5,
            bandwidth_hz_per_px=210.0,
            voxel_size_mm=(1.0, 1.0, 1.0),
            flip_angle_deg=15.0,
            tr_ms=8.5,
            te_ms=3.8,
            rf_pulse_duration_ms=1.2,
            slew_rate_T_per_m_s=150.0,
            scan_duration_min=5.5,
            rf_coil=RFCoilType.WHOLE_BODY_CP,
            anatomy="head",
        ),
    ]
    return CoDesignScenario(
        scenario_id="pacemaker_cardiac_brain_1p5t",
        title="1.5T Cardiac & Brain MRI Protocol with MR-Conditional Dual-Chamber Pacemaker",
        clinical_context=(
            "Multi-sequence cardiac cine bSSFP + brain T2 TSE + 3D MPRAGE protocol audited against "
            "IEC 60601-2-33 Ed. 4 B1+rms <= 2.0 uT, WB SAR <= 2.0 W/kg, and 30 min active time."
        ),
        scanner=scanner,
        implant=implant,
        initial_sequences=sequences,
        peak_target_delta_b0_ppm=2.2,
        residual_gnl_target_mm=0.22,
        srs_tolerance_mm=1.0,
        snr_floor=0.55,
        enforce_srs_geometry=False,
    )
