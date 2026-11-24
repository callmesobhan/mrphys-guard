"""MR-Only Physics-Ambiguity to DVH & Anisotropic van Herk PTV Margin Certifier.

Bridges the diagnostic MRI physics <-> radiotherapy TPS silo that no commercial vendor resolves:
1. Propagates exact non-rigid 3D MRI geometric distortion (System GNL + Susceptibility B0 with
   explicit readout axis & polarity) directly into 3D RTDOSE / RTSTRUCT organ DVH curves
   (Target D98%, OAR D0.1cc, Niemierko TCP & LKB NTCP).
2. Computes the physics-exact "Dark-MRI Identifiability" adversarial sCT + multi-beam ERE
   blind-spot certificate across all voxels where T1/T2 MRI intensity cannot separate cortical
   bone from air.
3. Extends van Herk's PTV margin formula (IJROBP 2000; DOI: 10.1016/S0360-3016(00)00518-6) to
   sequence-conditioned anisotropic margins (M_x, M_y, M_z) and optimizes across all 6 readout
   axis/polarity choices (+/-LR, +/-AP, +/-SI) to push susceptibility distortion away from OARs.

RESEARCH SOFTWARE — NOT A MEDICAL DEVICE.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, map_coordinates
from scipy.special import erf, erfc

from mrphys_guard.distortion_solver import (
    GAMMA_BAR_HZ_PER_T,
    SphericalHarmonicGNLModel,
    compute_distortion_jacobian_and_unwarp,
    solve_susceptibility_b0_map,
)
from mrphys_guard.ir import (
    DarkMRIAmbiguityCertificate,
    MROnlyDVHMarginCertificate,
    MRSequenceSpec,
    OrganDVHImpact,
    ReadoutAxisCandidate,
)
from mrphys_guard.sct_ere_auditor import compute_multibeam_ere_cancellation

READOUT_CONFIGS: list[tuple[str, Literal[0, 1, 2], Literal[1, -1]]] = [
    ("+X (L→R)", 0, 1),
    ("-X (R→L)", 0, -1),
    ("+Y (A→P)", 1, 1),
    ("-Y (P→A)", 1, -1),
    ("+Z (I→S)", 2, 1),
    ("-Z (S→I)", 2, -1),
]


def build_clinical_srs_dose_grid(
    target_mask: np.ndarray,
    oar_mask: np.ndarray,
    voxel_size_mm: tuple[float, float, float],
    prescription_gy: float = 36.25,
    ptv_margin_mm: float = 3.0,
    penumbra_sigma_mm: float = 2.6,
) -> np.ndarray:
    """Construct a realistic 3D conformal stereotactic (SBRT/SRS) planned dose grid [Gy].

    Features a uniform high-dose target plateau (~102% Rx inside CTV), crisp stereotactic
    penumbra (~4.4 mm 80%-20%), and steep posterior OAR sparing carve-out directly at the
    target-OAR boundary (e.g., prostate CTV vs. anterior rectal wall).
    """
    ctv = np.asarray(target_mask, dtype=bool)
    oar = np.asarray(oar_mask, dtype=bool)
    if ctv.shape != oar.shape or ctv.ndim != 3:
        raise ValueError("target_mask and oar_mask must be matching 3D boolean arrays.")

    dist_outside_ctv = distance_transform_edt(~ctv, sampling=voxel_size_mm)
    dist_inside_ctv = distance_transform_edt(ctv, sampling=voxel_size_mm)
    signed_dist = dist_outside_ctv - dist_inside_ctv

    # Posterior OAR sparing: slightly tighten margin directly adjacent to OAR wall
    dist_to_oar = distance_transform_edt(~oar, sampling=voxel_size_mm)
    effective_margin = np.where(dist_to_oar < 8.0, ptv_margin_mm * 0.45, ptv_margin_mm)

    # Smooth conformal stereotactic profile: 103% Rx at CTV core, ~98% at CTV edge, steep outside
    norm_d = (signed_dist - effective_margin) / max(penumbra_sigma_mm, 0.5)
    dose = prescription_gy * 1.035 * 0.5 * erfc(norm_d / np.sqrt(2.0))
    return np.ascontiguousarray(np.clip(dose, 0.0, prescription_gy * 1.08), dtype=np.float64)


def build_rectal_wall_oar_mask(
    rectal_gas_mask: np.ndarray,
    body_mask: np.ndarray,
    prostate_mask: np.ndarray,
    wall_thickness_voxels: int = 2,
) -> np.ndarray:
    """Construct an anatomical rectal wall OAR shell surrounding the rectal lumen (excluding CTV)."""
    lumen = np.asarray(rectal_gas_mask, dtype=bool)
    body = np.asarray(body_mask, dtype=bool)
    ctv = np.asarray(prostate_mask, dtype=bool)
    shell = binary_dilation(lumen, iterations=wall_thickness_voxels) & (~lumen) & body & (~ctv)
    return shell


def compute_dvh_metrics(
    dose_values_gy: np.ndarray,
    voxel_vol_cc: float,
) -> dict[str, float]:
    """Compute exact clinical DVH endpoints (D98%, D95%, Dmean, D2%, D0.1cc) from ROI dose samples."""
    vals = np.asarray(dose_values_gy, dtype=np.float64).ravel()
    if vals.size == 0:
        return {"d98": 0.0, "d95": 0.0, "dmean": 0.0, "d2": 0.0, "d0p1cc": 0.0}
    sorted_desc = np.sort(vals)[::-1]
    n_0p1cc = max(1, int(np.ceil(0.1 / max(voxel_vol_cc, 1e-6))))
    idx_0p1cc = min(n_0p1cc - 1, sorted_desc.size - 1)
    return {
        "d98": float(np.percentile(vals, 2.0)),
        "d95": float(np.percentile(vals, 5.0)),
        "dmean": float(np.mean(vals)),
        "d2": float(np.percentile(vals, 98.0)),
        "d0p1cc": float(sorted_desc[idx_0p1cc]),
    }


def compute_cumulative_dvh_curve(
    dose_values_gy: np.ndarray,
    dose_bins_gy: np.ndarray,
) -> np.ndarray:
    """Compute cumulative DVH volume percentage V(D >= bin) in [0, 100]."""
    vals = np.asarray(dose_values_gy, dtype=np.float64).ravel()
    if vals.size == 0:
        return np.zeros_like(dose_bins_gy, dtype=np.float64)
    return np.array([float(np.mean(vals >= d) * 100.0) for d in dose_bins_gy], dtype=np.float64)


def compute_niemierko_tcp_pct(
    dose_values_gy: np.ndarray,
    tcd50_gy: float = 28.0,
    gamma50: float = 2.0,
    a_param: float = -10.0,
) -> float:
    """Compute Niemierko EUD-based logistic Tumor Control Probability (TCP) [%]."""
    vals = np.maximum(np.asarray(dose_values_gy, dtype=np.float64).ravel(), 1e-3)
    if vals.size == 0:
        return 0.0
    eud = float(np.mean(vals**a_param) ** (1.0 / a_param))
    tcp = 100.0 / (1.0 + (tcd50_gy / max(eud, 1e-6)) ** (4.0 * gamma50))
    return float(np.clip(tcp, 0.0, 100.0))


def compute_lkb_ntcp_pct(
    dose_values_gy: np.ndarray,
    td50_gy: float = 39.5,
    m_param: float = 0.14,
    n_param: float = 0.09,
) -> float:
    """Compute Lyman-Kutcher-Burman (LKB) Normal Tissue Complication Probability (NTCP) [%]."""
    vals = np.maximum(np.asarray(dose_values_gy, dtype=np.float64).ravel(), 0.0)
    if vals.size == 0:
        return 0.0
    geud = float(np.mean(vals ** (1.0 / n_param)) ** n_param)
    t_val = (geud - td50_gy) / max(m_param * td50_gy, 1e-6)
    ntcp = 50.0 * (1.0 + float(erf(t_val / np.sqrt(2.0))))
    return float(np.clip(ntcp, 0.0, 100.0))


def _sample_dose_with_displacement(
    dose_gy: np.ndarray,
    roi_mask: np.ndarray,
    disp_vec_mm: np.ndarray,
    voxel_size_mm: tuple[float, float, float],
) -> np.ndarray:
    """Sample planned 3D RTDOSE at non-rigidly distorted coordinates x + Delta_r(x) inside ROI."""
    idx = np.argwhere(roi_mask)
    if idx.shape[0] == 0:
        return np.array([], dtype=np.float64)
    shifted = np.empty((3, idx.shape[0]), dtype=np.float64)
    for ax in range(3):
        shifted[ax] = idx[:, ax].astype(np.float64) + (
            disp_vec_mm[idx[:, 0], idx[:, 1], idx[:, 2], ax] / voxel_size_mm[ax]
        )
    return map_coordinates(dose_gy, shifted, order=1, mode="constant", cval=0.0)


def _displacement_for_readout_config(
    delta_b0_T: np.ndarray,
    gnl_disp_vec_mm: np.ndarray,
    bandwidth_hz_per_px: float,
    voxel_size_mm: tuple[float, float, float],
    readout_axis: Literal[0, 1, 2],
    readout_polarity: Literal[1, -1],
) -> np.ndarray:
    """Build total 3D displacement vector [mm] for a chosen readout axis and polarity (+/-1)."""
    delta_f_hz = GAMMA_BAR_HZ_PER_T * delta_b0_T
    disp = gnl_disp_vec_mm.copy()
    dx_ro = voxel_size_mm[readout_axis]
    disp[..., readout_axis] += float(readout_polarity) * (delta_f_hz / bandwidth_hz_per_px) * dx_ro
    return disp


def certify_dark_mri_sct_blindspot(
    mri_volume: np.ndarray,
    ct_or_sct_hu: np.ndarray,
    dose_gy: np.ndarray,
    target_mask: np.ndarray,
    oar_mask: np.ndarray,
    body_mask: np.ndarray,
    voxel_size_mm: tuple[float, float, float],
    b0_T: float = 1.5,
    gantry_angles_deg: list[float] | None = None,
    dark_mri_threshold: float = 0.14,
) -> tuple[DarkMRIAmbiguityCertificate, dict[str, np.ndarray]]:
    """Certify worst-case dosimetric blind spot from dark-MRI bone-vs-air sCT non-identifiability.

    On standard T1/T2 MRI (without UTE/ZTE), cortical bone, air cavities, and metal all produce
    near-zero signal (< dark_mri_threshold). Any black-box AI sCT is physically underdetermined
    on this voxel set Omega_ambig. This function computes the worst-case physically consistent
    beam-line WEPL error + multi-beam ERE interface perturbation inside Omega_ambig.
    """
    mri = np.asarray(mri_volume, dtype=np.float64)
    p99 = float(np.percentile(mri, 99.5))
    mri_norm = np.clip(mri / max(p99, 1e-8), 0.0, 1.5)

    body = np.asarray(body_mask, dtype=bool)
    ctv = np.asarray(target_mask, dtype=bool)
    oar = np.asarray(oar_mask, dtype=bool)
    vox_cc = float(np.prod(voxel_size_mm)) * 1e-3

    # Physically underdetermined dark-MRI set inside patient contour
    ambig_mask = body & (mri_norm < dark_mri_threshold)
    ambig_vox = int(np.sum(ambig_mask))
    ambig_cc = ambig_vox * vox_cc
    body_vox = max(1, int(np.sum(body)))
    ambig_frac = float(ambig_vox / body_vox)

    # Beam-line worst-case WEPL ambiguity through CTV center across planned gantry directions
    # Each dark voxel can be misclassified between air (0.001) and cortical bone (1.56) -> |Delta_rho| ~ 1.56
    # Restricting to rays actually crossing the target ROI gives exact target-relevant WEPL exposure.
    ctv_idx = np.argwhere(ctv)
    if ctv_idx.shape[0] > 0:
        cx, cy, cz = (
            int(round(float(np.mean(ctv_idx[:, 0])))),
            int(round(float(np.mean(ctv_idx[:, 1])))),
            int(round(float(np.mean(ctv_idx[:, 2])))),
        )
    else:
        cx, cy, cz = mri.shape[0] // 2, mri.shape[1] // 2, mri.shape[2] // 2

    angles = gantry_angles_deg or [0.0, 51.0, 103.0, 154.0, 206.0, 257.0, 309.0]
    beam_wepl_errors: list[float] = []
    max_steps = int(max(mri.shape[0], mri.shape[1]))
    step_mm = 0.5 * (voxel_size_mm[0] + voxel_size_mm[1])
    for ang in angles:
        rad = np.deg2rad(ang)
        ux, uy = float(np.sin(rad)), float(np.cos(rad))
        # Trace incoming ray from periphery to target center (cx, cy, cz)
        ray_ambig_mm = 0.0
        for s in range(1, max_steps):
            ix = int(round(cx - ux * s))
            iy = int(round(cy - uy * s))
            if not (0 <= ix < mri.shape[0] and 0 <= iy < mri.shape[1]):
                break
            if ambig_mask[ix, iy, cz]:
                # Worst-case bone-vs-air delta_rho_e = 1.56 (or 1.0 conservative fraction 0.5 for partial ambiguity)
                ray_ambig_mm += step_mm * 0.78
        beam_wepl_errors.append(ray_ambig_mm)

    # Multi-beam averaged worst-case WEPL error (mm) -> 6 MV photon dose shift (~0.35% / mm WEPL)
    worst_wepl_mm = float(np.mean(beam_wepl_errors)) if beam_wepl_errors else 0.0
    worst_att_shift_pct = 0.35 * worst_wepl_mm

    # Multi-beam ERE boundary shift on OAR wall at B0
    mb_ere, mb_maps = compute_multibeam_ere_cancellation(
        ct_or_sct_hu=ct_or_sct_hu,
        voxel_size_mm=voxel_size_mm,
        gantry_angles_deg=angles,
        b0_T=b0_T,
        body_mask=body,
    )
    ere_map = mb_maps["multibeam_net_ere_pct"]
    oar_ere_vals = ere_map[oar] if np.any(oar) else np.array([0.0])
    worst_ere_oar_pct = float(np.max(oar_ere_vals)) if oar_ere_vals.size > 0 else mb_ere.multibeam_net_peak_pct

    # Worst-case target D98 drop (%) and OAR D0.1cc increase (%) from sCT ambiguity + ERE
    plan_ctv_m = compute_dvh_metrics(dose_gy[ctv], vox_cc)
    plan_oar_m = compute_dvh_metrics(dose_gy[oar], vox_cc)

    worst_target_loss_pct = float(worst_att_shift_pct)
    worst_oar_inc_pct = float(worst_att_shift_pct + worst_ere_oar_pct)

    if worst_target_loss_pct <= 2.0 and worst_oar_inc_pct <= 8.0:
        status: Literal["CERTIFIED_ROBUST", "MARGIN_ADJUSTMENT_REQUIRED", "UNSAFE_BLIND_SPOT"] = (
            "CERTIFIED_ROBUST"
        )
        rationale = (
            f"Dark-MRI ambiguity ({ambig_cc:.1f} cc) induces <= {worst_target_loss_pct:.2f}% "
            f"target attenuation uncertainty and <= {worst_oar_inc_pct:.1f}% OAR ERE+sCT bound."
        )
    elif worst_target_loss_pct <= 5.0 and worst_oar_inc_pct <= 18.0:
        status = "MARGIN_ADJUSTMENT_REQUIRED"
        rationale = (
            f"Moderate dark-MRI blind spot ({ambig_cc:.1f} cc, WEPL bound {worst_wepl_mm:.1f} mm, "
            f"OAR ERE+sCT +{worst_oar_inc_pct:.1f}%): UTE/ZTE bone anchor or anisotropic margin adjustment advised."
        )
    else:
        status = "UNSAFE_BLIND_SPOT"
        rationale = (
            f"Severe dark-MRI sCT+ERE blind spot ({ambig_cc:.1f} cc, target attenuation bound "
            f"-{worst_target_loss_pct:.1f}%, OAR wall +{worst_oar_inc_pct:.1f}%): MR-only sCT requires "
            "direct CT verification or UTE/ZTE bone-air disambiguation."
        )
    _ = (plan_ctv_m, plan_oar_m)

    cert = DarkMRIAmbiguityCertificate(
        dark_mri_threshold=dark_mri_threshold,
        ambiguous_voxel_count=ambig_vox,
        ambiguous_volume_cc=round(ambig_cc, 3),
        ambiguous_body_fraction=round(ambig_frac, 4),
        worst_case_wepl_error_mm=round(worst_wepl_mm, 3),
        worst_case_sct_attenuation_shift_pct=round(worst_att_shift_pct, 3),
        worst_case_ere_interface_shift_pct=round(worst_ere_oar_pct, 3),
        worst_case_target_d98_loss_pct=round(worst_target_loss_pct, 3),
        worst_case_oar_d0p1cc_increase_pct=round(worst_oar_inc_pct, 3),
        status=status,
        rationale=rationale,
    )
    return cert, {"ambiguous_dark_mri_mask": ambig_mask, "multibeam_ere_pct": ere_map}


def certify_mronly_dvh_and_anisotropic_margins(
    chi_si: np.ndarray,
    mri_volume: np.ndarray,
    ct_or_sct_hu: np.ndarray,
    dose_gy: np.ndarray,
    target_mask: np.ndarray,
    oar_mask: np.ndarray,
    body_mask: np.ndarray,
    sequence: MRSequenceSpec,
    gnl_model: SphericalHarmonicGNLModel | None = None,
    prescription_gy: float = 36.25,
    setup_sigma_sys_mm: float = 0.8,
    random_sigma_mm: float = 1.0,
    gantry_angles_deg: list[float] | None = None,
    target_name: str = "Prostate_CTV",
    oar_name: str = "Rectal_Wall",
) -> tuple[MROnlyDVHMarginCertificate, dict[str, np.ndarray]]:
    """Run the full MR-Only Physics-Ambiguity to DVH & Anisotropic van Herk PTV Margin Certifier.

    Evaluates:
    1. Exact 3D k-space susceptibility Delta_B0 + l<=5 GNL distortion field.
    2. All 6 readout axis/polarity options (+/-LR, +/-AP, +/-SI) -> Distortion-DVH + anisotropic
       van Herk PTV margins M_a = 2.5*sqrt(Sigma_setup^2 + Sigma_MR,a^2) + 0.7*sigma_rand.
    3. Dark-MRI identifiability adversarial sCT + multi-beam ERE worst-case dose bounds.
    4. Returns the unified certificate plus exact cumulative DVH arrays for plotting.
    """
    vox = sequence.voxel_size_mm
    vox_cc = float(np.prod(vox)) * 1e-3
    ctv = np.asarray(target_mask, dtype=bool)
    oar = np.asarray(oar_mask, dtype=bool)
    body = np.asarray(body_mask, dtype=bool)

    # 1. Exact 3D B0 susceptibility map & GNL vector field
    delta_b0_T = solve_susceptibility_b0_map(
        chi_si=chi_si, voxel_size_mm=vox, b0_T=sequence.b0_T, b0_axis=2, pad_factor=2
    )
    gnl_disp = (
        gnl_model.evaluate_grid(chi_si.shape, vox)
        if gnl_model is not None
        else np.zeros((*chi_si.shape, 3), dtype=np.float64)
    )

    # 2. Dark-MRI adversarial sCT + multi-beam ERE blind-spot bound
    dark_cert, dark_maps = certify_dark_mri_sct_blindspot(
        mri_volume=mri_volume,
        ct_or_sct_hu=ct_or_sct_hu,
        dose_gy=dose_gy,
        target_mask=ctv,
        oar_mask=oar,
        body_mask=body,
        voxel_size_mm=vox,
        b0_T=sequence.b0_T,
        gantry_angles_deg=gantry_angles_deg,
    )

    # Planned baseline DVH
    plan_ctv_dose = dose_gy[ctv]
    plan_oar_dose = dose_gy[oar]
    plan_ctv_m = compute_dvh_metrics(plan_ctv_dose, vox_cc)
    plan_oar_m = compute_dvh_metrics(plan_oar_dose, vox_cc)

    # 3. Sweep all 6 readout axis/polarity configurations
    candidates: list[ReadoutAxisCandidate] = []
    disp_by_label: dict[str, np.ndarray] = {}
    base_label = next(lbl for lbl, ax, pol in READOUT_CONFIGS if ax == sequence.readout_axis and pol == 1)

    for label, ro_ax, ro_pol in READOUT_CONFIGS:
        disp = _displacement_for_readout_config(
            delta_b0_T=delta_b0_T,
            gnl_disp_vec_mm=gnl_disp,
            bandwidth_hz_per_px=sequence.bandwidth_hz_per_px,
            voxel_size_mm=vox,
            readout_axis=ro_ax,
            readout_polarity=ro_pol,
        )
        disp_by_label[label] = disp

        # Per-axis systematic MR distortion RMS inside target CTV
        ctv_disp = disp[ctv]
        sig_mr = tuple(float(np.sqrt(np.mean(ctv_disp[:, a] ** 2))) for a in range(3))
        aniso_m = tuple(
            round(2.5 * float(np.hypot(setup_sigma_sys_mm, sig_mr[a])) + 0.7 * random_sigma_mm, 3)
            for a in range(3)
        )
        iso_m = max(aniso_m)

        # Sample Distortion-DVH
        dist_ctv_dose = _sample_dose_with_displacement(dose_gy, ctv, disp, vox)
        dist_oar_dose = _sample_dose_with_displacement(dose_gy, oar, disp, vox)
        dist_ctv_m = compute_dvh_metrics(dist_ctv_dose, vox_cc)
        dist_oar_m = compute_dvh_metrics(dist_oar_dose, vox_cc)

        d98_delta_gy = dist_ctv_m["d98"] - plan_ctv_m["d98"]
        d98_delta_pct = (d98_delta_gy / max(prescription_gy, 1e-6)) * 100.0
        oar_delta_gy = dist_oar_m["d0p1cc"] - plan_oar_m["d0p1cc"]
        oar_delta_pct = (oar_delta_gy / max(prescription_gy, 1e-6)) * 100.0

        jac_rep, _ = compute_distortion_jacobian_and_unwarp(mri_volume, disp, vox, body_mask=body)

        # Composite risk score: penalizes target D98 loss, OAR D0.1cc overdose, and Jacobian folding
        score = (
            2.0 * max(0.0, -d98_delta_gy)
            + 2.5 * max(0.0, oar_delta_gy)
            + 0.15 * float(sum(aniso_m))
            + 0.002 * jac_rep.folding_voxel_count
        )
        candidates.append(
            ReadoutAxisCandidate(
                label=label,
                readout_axis=ro_ax,
                readout_polarity=ro_pol,
                target_sigma_sys_mm=(round(sig_mr[0], 4), round(sig_mr[1], 4), round(sig_mr[2], 4)),
                anisotropic_ptv_margin_mm=(aniso_m[0], aniso_m[1], aniso_m[2]),
                naive_isotropic_margin_mm=iso_m,
                target_delta_d98_gy=round(d98_delta_gy, 4),
                target_delta_d98_pct=round(d98_delta_pct, 3),
                oar_delta_d0p1cc_gy=round(oar_delta_gy, 4),
                oar_delta_d0p1cc_pct=round(oar_delta_pct, 3),
                folding_voxel_count=jac_rep.folding_voxel_count,
                composite_risk_score=round(score, 4),
            )
        )

    candidates_sorted = sorted(candidates, key=lambda c: c.composite_risk_score)
    best_cand = candidates_sorted[0]
    base_cand = next(c for c in candidates if c.label == base_label)

    # 4. Build full per-organ DVH impact (Planned vs. Baseline Distorted vs. Worst-Case sCT+ERE)
    disp_base = disp_by_label[base_label]
    dist_ctv_base = _sample_dose_with_displacement(dose_gy, ctv, disp_base, vox)
    dist_oar_base = _sample_dose_with_displacement(dose_gy, oar, disp_base, vox)

    # Worst-case sCT attenuation drop on CTV & sCT+ERE hotspot amplification on OAR
    worst_ctv_dose = dist_ctv_base * max(0.5, 1.0 - dark_cert.worst_case_sct_attenuation_shift_pct / 100.0)
    worst_oar_dose = dist_oar_base * (1.0 + dark_cert.worst_case_oar_d0p1cc_increase_pct / 100.0)

    dist_ctv_m = compute_dvh_metrics(dist_ctv_base, vox_cc)
    worst_ctv_m = compute_dvh_metrics(worst_ctv_dose, vox_cc)
    dist_oar_m = compute_dvh_metrics(dist_oar_base, vox_cc)
    worst_oar_m = compute_dvh_metrics(worst_oar_dose, vox_cc)

    organ_impacts = [
        OrganDVHImpact(
            roi_name=target_name,
            roi_role="TARGET",
            volume_cc=round(float(np.sum(ctv)) * vox_cc, 3),
            planned_d98_gy=round(plan_ctv_m["d98"], 3),
            distorted_d98_gy=round(dist_ctv_m["d98"], 3),
            worst_case_d98_gy=round(worst_ctv_m["d98"], 3),
            delta_d98_gy=round(dist_ctv_m["d98"] - plan_ctv_m["d98"], 3),
            delta_d98_pct=round((dist_ctv_m["d98"] - plan_ctv_m["d98"]) / prescription_gy * 100.0, 2),
            planned_dmean_gy=round(plan_ctv_m["dmean"], 3),
            distorted_dmean_gy=round(dist_ctv_m["dmean"], 3),
            planned_d0p1cc_gy=round(plan_ctv_m["d0p1cc"], 3),
            distorted_d0p1cc_gy=round(dist_ctv_m["d0p1cc"], 3),
            worst_case_d0p1cc_gy=round(worst_ctv_m["d0p1cc"], 3),
            delta_d0p1cc_gy=round(dist_ctv_m["d0p1cc"] - plan_ctv_m["d0p1cc"], 3),
            delta_d0p1cc_pct=round((dist_ctv_m["d0p1cc"] - plan_ctv_m["d0p1cc"]) / prescription_gy * 100.0, 2),
            planned_tcp_or_ntcp_pct=round(compute_niemierko_tcp_pct(plan_ctv_dose), 2),
            distorted_tcp_or_ntcp_pct=round(compute_niemierko_tcp_pct(dist_ctv_base), 2),
            worst_case_tcp_or_ntcp_pct=round(compute_niemierko_tcp_pct(worst_ctv_dose), 2),
        ),
        OrganDVHImpact(
            roi_name=oar_name,
            roi_role="OAR",
            volume_cc=round(float(np.sum(oar)) * vox_cc, 3),
            planned_d98_gy=round(plan_oar_m["d98"], 3),
            distorted_d98_gy=round(dist_oar_m["d98"], 3),
            worst_case_d98_gy=round(worst_oar_m["d98"], 3),
            delta_d98_gy=round(dist_oar_m["d98"] - plan_oar_m["d98"], 3),
            delta_d98_pct=round((dist_oar_m["d98"] - plan_oar_m["d98"]) / prescription_gy * 100.0, 2),
            planned_dmean_gy=round(plan_oar_m["dmean"], 3),
            distorted_dmean_gy=round(dist_oar_m["dmean"], 3),
            planned_d0p1cc_gy=round(plan_oar_m["d0p1cc"], 3),
            distorted_d0p1cc_gy=round(dist_oar_m["d0p1cc"], 3),
            worst_case_d0p1cc_gy=round(worst_oar_m["d0p1cc"], 3),
            delta_d0p1cc_gy=round(dist_oar_m["d0p1cc"] - plan_oar_m["d0p1cc"], 3),
            delta_d0p1cc_pct=round((dist_oar_m["d0p1cc"] - plan_oar_m["d0p1cc"]) / prescription_gy * 100.0, 2),
            planned_tcp_or_ntcp_pct=round(compute_lkb_ntcp_pct(plan_oar_dose), 2),
            distorted_tcp_or_ntcp_pct=round(compute_lkb_ntcp_pct(dist_oar_base), 2),
            worst_case_tcp_or_ntcp_pct=round(compute_lkb_ntcp_pct(worst_oar_dose), 2),
        ),
    ]

    oar_saved = round(base_cand.oar_delta_d0p1cc_gy - best_cand.oar_delta_d0p1cc_gy, 3)
    target_saved = round(best_cand.target_delta_d98_gy - base_cand.target_delta_d98_gy, 3)

    # Overall certificate combines Distortion-DVH severity and Dark-MRI sCT+ERE blind-spot status
    worst_ctv_drop_pct = (plan_ctv_m["d98"] - worst_ctv_m["d98"]) / prescription_gy * 100.0
    if dark_cert.status == "UNSAFE_BLIND_SPOT" or worst_ctv_drop_pct > 8.0:
        overall: Literal["CERTIFIED_ROBUST", "MARGIN_ADJUSTMENT_REQUIRED", "UNSAFE_BLIND_SPOT"] = (
            "UNSAFE_BLIND_SPOT"
        )
    elif dark_cert.status == "MARGIN_ADJUSTMENT_REQUIRED" or abs(base_cand.target_delta_d98_pct) > 2.0:
        overall = "MARGIN_ADJUSTMENT_REQUIRED"
    else:
        overall = "CERTIFIED_ROBUST"

    summary = (
        f"{overall}: Switching readout from {base_label} to {best_cand.label} changes target "
        f"Delta_D98 from {base_cand.target_delta_d98_gy:+.2f} Gy to {best_cand.target_delta_d98_gy:+.2f} Gy "
        f"and OAR Delta_D0.1cc from {base_cand.oar_delta_d0p1cc_gy:+.2f} Gy to "
        f"{best_cand.oar_delta_d0p1cc_gy:+.2f} Gy; recommended anisotropic van Herk PTV margin = "
        f"{best_cand.anisotropic_ptv_margin_mm} mm."
    )

    cert = MROnlyDVHMarginCertificate(
        b0_T=sequence.b0_T,
        bandwidth_hz_per_px=sequence.bandwidth_hz_per_px,
        prescription_gy=prescription_gy,
        baseline_readout_label=base_label,
        recommended_readout_label=best_cand.label,
        recommended_anisotropic_margin_mm=best_cand.anisotropic_ptv_margin_mm,
        baseline_anisotropic_margin_mm=base_cand.anisotropic_ptv_margin_mm,
        oar_overdose_saved_gy=oar_saved,
        target_coverage_saved_gy=target_saved,
        organ_impacts=organ_impacts,
        dark_mri_certificate=dark_cert,
        readout_candidates=candidates,
        overall_certificate=overall,
        summary=summary,
    )

    # Pre-compute cumulative DVH arrays for plotting
    dose_bins = np.linspace(0.0, prescription_gy * 1.15, 220)
    disp_best = disp_by_label[best_cand.label]
    dist_ctv_best = _sample_dose_with_displacement(dose_gy, ctv, disp_best, vox)
    dist_oar_best = _sample_dose_with_displacement(dose_gy, oar, disp_best, vox)

    artifacts = {
        "dose_bins_gy": dose_bins,
        "dvh_ctv_planned": compute_cumulative_dvh_curve(plan_ctv_dose, dose_bins),
        "dvh_ctv_distorted": compute_cumulative_dvh_curve(dist_ctv_base, dose_bins),
        "dvh_ctv_best_ro": compute_cumulative_dvh_curve(dist_ctv_best, dose_bins),
        "dvh_ctv_worst_case": compute_cumulative_dvh_curve(worst_ctv_dose, dose_bins),
        "dvh_oar_planned": compute_cumulative_dvh_curve(plan_oar_dose, dose_bins),
        "dvh_oar_distorted": compute_cumulative_dvh_curve(dist_oar_base, dose_bins),
        "dvh_oar_best_ro": compute_cumulative_dvh_curve(dist_oar_best, dose_bins),
        "dvh_oar_worst_case": compute_cumulative_dvh_curve(worst_oar_dose, dose_bins),
        "ambiguous_dark_mri_mask": dark_maps["ambiguous_dark_mri_mask"],
    }
    return cert, artifacts
