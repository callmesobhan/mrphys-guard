"""Synthetic-CT (sCT) Physics-Plausibility Auditor & Lorentz-Force Electron-Return-Effect (ERE)
Boundary Risk Kernel for 1.5T / 0.35T MR-Linacs.

References:
- Raaijmakers, Raaymakers & Lagendijk (2005), Phys Med Biol 50(7):1363-1376.
  DOI: 10.1088/0031-9155/50/7/002
- Raaijmakers, Raaymakers & Lagendijk (2008), Phys Med Biol 53(4):909-923.
  DOI: 10.1088/0031-9155/53/4/006
- Thummerer et al. (2023), SynthRAD2023, Med Phys 50(7):4664-4674.
  DOI: 10.1002/mp.16529
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

from mrphys_guard.ir import EREBoundaryRisk, SCTAnomalyFlag, SCTAuditReport

# Physical constants (CODATA 2018/2022)
ELECTRON_REST_ENERGY_MEV: float = 0.51099895000
SPEED_OF_LIGHT_M_S: float = 2.99792458e8

# Piecewise-linear clinical CT HU -> relative electron density (rho_e / rho_e,water) lookup
_HU_ANCHORS = np.array([-1000.0, -700.0, -100.0, 0.0, 50.0, 300.0, 1000.0, 3000.0], dtype=np.float64)
_RED_ANCHORS = np.array([0.0012, 0.30, 0.90, 1.00, 1.045, 1.18, 1.56, 3.70], dtype=np.float64)


def hu_to_relative_electron_density(hu: np.ndarray) -> np.ndarray:
    """Convert CT/sCT Hounsfield Units (HU) to relative electron density (rho_e / rho_e,water)."""
    arr = np.asarray(hu, dtype=np.float64)
    return np.interp(arr, _HU_ANCHORS, _RED_ANCHORS)


def inject_sct_failure_modes(
    ct_hu: np.ndarray,
    bone_mask: np.ndarray | None = None,
    metal_mask: np.ndarray | None = None,
    body_mask: np.ndarray | None = None,
    swap_bone_fraction: float = 0.25,
    drop_metal_to_tissue: bool = True,
    truncate_body_border_voxels: int = 2,
    hu_noise_sigma: float = 18.0,
    seed: int = 7,
) -> np.ndarray:
    """Synthesize clinically realistic AI synthetic-CT failure modes for benchmark auditing.

    Produces:
    - Silent bone-to-air swaps (low-MRI-signal cortical bone misclassified as air -950 HU)
    - Missing metal artifact (high-Z implant replaced by soft tissue +45 HU)
    - Body-contour truncation at periphery (skin border set to -1000 HU)
    - Moderate soft-tissue HU bias/noise
    """
    rng = np.random.default_rng(seed)
    sct = np.array(ct_hu, dtype=np.float64, copy=True)

    # Smooth mild synthesis noise inside body
    if body_mask is None:
        body_mask = ct_hu > -500.0
    if hu_noise_sigma > 0.0:
        sct[body_mask] += rng.normal(0.0, hu_noise_sigma, size=int(np.sum(body_mask)))

    # 1. Silent bone-to-air swap in a contiguous posterior/lateral sub-region of cortical bone
    if bone_mask is None:
        bone_mask = ct_hu > 350.0
    bone_idx = np.argwhere(bone_mask)
    if bone_idx.shape[0] > 0 and swap_bone_fraction > 0.0:
        # Select a coherent anatomical bone cluster spanning mid-axial slices (high y, right side)
        n_swap = max(1, int(round(bone_idx.shape[0] * swap_bone_fraction)))
        order = np.argsort(bone_idx[:, 1] * 1000 + bone_idx[:, 0])
        chosen = bone_idx[order[-n_swap:]]
        sct[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = -950.0

    # 2. Missing metal implant (AI model hallucinates normal soft tissue)
    if metal_mask is None:
        metal_mask = ct_hu > 1800.0
    if drop_metal_to_tissue and np.any(metal_mask):
        sct[metal_mask] = 45.0

    # 3. Body contour truncation on lateral/anterior outer skin rim
    if truncate_body_border_voxels > 0 and np.any(body_mask):
        eroded = binary_erosion(body_mask, iterations=truncate_body_border_voxels)
        rim = body_mask & (~eroded)
        # Truncate anterior/left quadrant of skin rim
        nx = sct.shape[0]
        trunc = np.zeros_like(body_mask)
        trunc[: nx // 3, :, :] = rim[: nx // 3, :, :]
        sct[trunc] = -1000.0

    return np.clip(sct, -1024.0, 4000.0)


def audit_synthetic_ct(
    mri_volume: np.ndarray,
    sct_hu: np.ndarray,
    voxel_size_mm: tuple[float, float, float],
    ground_truth_ct_hu: np.ndarray | None = None,
    body_mask: np.ndarray | None = None,
    bone_prior_mask: np.ndarray | None = None,
    metal_prior_mask: np.ndarray | None = None,
) -> tuple[SCTAuditReport, dict[str, np.ndarray]]:
    """Audit a synthetic CT (sCT) volume against MRI intensity-physics priors and optional GT CT.

    Detects:
    1. Silent bone-to-air swaps (internal low-MRI-signal region predicted as air in bone corridor
       or mismatched against MRI/GT bone prior).
    2. Missing dental/hip metal artifacts (MR signal void + sharp surrounding gradient where sCT
       fails to predict high-Z metal > 1500 HU).
    3. Body-contour truncation (non-zero MRI soft-tissue signal where sCT predicts air < -500 HU).
    4. Water-equivalent path length (WEPL) error & approximate 6 MV photon dose error (~0.35%/mm).
    """
    mri = np.asarray(mri_volume, dtype=np.float64)
    sct = np.asarray(sct_hu, dtype=np.float64)
    if mri.shape != sct.shape or mri.ndim != 3:
        raise ValueError(f"mri_volume {mri.shape} and sct_hu {sct.shape} must be matching 3D arrays.")

    # Normalize MRI to [0, 1] robustly
    p99 = float(np.percentile(mri, 99.5))
    mri_norm = np.clip(mri / max(p99, 1e-8), 0.0, 1.5)

    voxel_vol_cc = float(np.prod(voxel_size_mm)) * 1e-3  # mm^3 -> cm^3 (cc)

    # Derive MRI body mask from signal threshold if not provided
    mri_body = body_mask.astype(bool) if body_mask is not None else (mri_norm > 0.12)
    sct_body = sct > -500.0
    eval_body = mri_body | sct_body

    # 1. Body-contour truncation: MRI has clear tissue signal (> 0.25) while sCT predicts air (<-500 HU)
    trunc_mask = mri_body & (mri_norm > 0.25) & (sct < -500.0)
    trunc_voxels = int(np.sum(trunc_mask))

    # 2. Bone-to-air swap detection
    # If explicit bone_prior_mask or GT CT is available, use it; otherwise infer internal dark-MRI
    # pockets embedded inside deep interior (surrounded by tissue and high-HU bone neighbors).
    if bone_prior_mask is not None:
        bone_ref = bone_prior_mask.astype(bool)
    elif ground_truth_ct_hu is not None:
        bone_ref = (ground_truth_ct_hu > 300.0) & (ground_truth_ct_hu <= 1800.0)
    else:
        # Physics heuristic: internal low-MRI voxels adjacent to sCT cortical bone (> 300 HU)
        # that suddenly drop to air (< -500 HU) despite being enclosed inside body
        sct_bone_dil = binary_dilation(sct > 300.0, iterations=2)
        deep_body = binary_erosion(mri_body, iterations=2)
        bone_ref = deep_body & sct_bone_dil & (mri_norm < 0.18)

    bone_to_air_mask = bone_ref & (sct < -400.0)
    bone_to_air_voxels = int(np.sum(bone_to_air_mask))

    # 3. Missing metal artifact detection
    if metal_prior_mask is not None:
        metal_ref = metal_prior_mask.astype(bool)
    elif ground_truth_ct_hu is not None:
        metal_ref = ground_truth_ct_hu > 1800.0
    else:
        # Deep interior complete MR signal void (< 0.025) surrounded by soft tissue where sCT is soft tissue
        deep_body = binary_erosion(mri_body, iterations=3)
        metal_ref = deep_body & (mri_norm < 0.025) & (sct > -200.0) & (sct < 250.0)

    missing_metal_mask = metal_ref & (sct < 1200.0)
    missing_metal_voxels = int(np.sum(missing_metal_mask))

    # Compute error metrics against GT CT when available, else MRI-inferred physics discrepancy
    red_sct = hu_to_relative_electron_density(sct)
    if ground_truth_ct_hu is not None:
        gt = np.asarray(ground_truth_ct_hu, dtype=np.float64)
        hu_err_map = sct - gt
        red_gt = hu_to_relative_electron_density(gt)
        red_err_map = red_sct - red_gt
        mae_body = float(np.mean(np.abs(hu_err_map[eval_body]))) if np.any(eval_body) else 0.0
        gt_bone = (gt > 300.0) & (gt <= 1800.0)
        gt_air = (gt < -500.0) & eval_body
        mae_bone = float(np.mean(np.abs(hu_err_map[gt_bone]))) if np.any(gt_bone) else 0.0
        mae_air = float(np.mean(np.abs(hu_err_map[gt_air]))) if np.any(gt_air) else 0.0
        mean_red_err = float(np.mean(np.abs(red_err_map[eval_body]))) if np.any(eval_body) else 0.0
    else:
        # Construct conservative synthetic residual map from detected physics-violation masks
        hu_err_map = np.zeros_like(sct)
        hu_err_map[bone_to_air_mask] = -1800.0
        hu_err_map[missing_metal_mask] = -2950.0
        hu_err_map[trunc_mask] = -1045.0
        red_err_map = np.zeros_like(sct)
        red_err_map[bone_to_air_mask] = -1.55
        red_err_map[missing_metal_mask] = -2.65
        red_err_map[trunc_mask] = -1.04
        mae_body = None
        mae_bone = None
        mae_air = None
        mean_red_err = None

    # Estimate maximum beam-line WEPL error across AP (axis 1) and LR (axis 0) projections [mm]
    wepl_err_ap = np.abs(np.sum(red_err_map, axis=1)) * voxel_size_mm[1]
    wepl_err_lr = np.abs(np.sum(red_err_map, axis=0)) * voxel_size_mm[0]
    max_wepl_error_mm = float(max(np.max(wepl_err_ap), np.max(wepl_err_lr)))
    # 6 MV photon attenuation sensitivity ~0.35% dose change per mm of water-equivalent path error
    est_dose_error_pct = 0.35 * max_wepl_error_mm

    anomalies: list[SCTAnomalyFlag] = []
    if bone_to_air_voxels > 0:
        vol_cc = bone_to_air_voxels * voxel_vol_cc
        wepl_b = float(
            max(
                np.max(np.sum(bone_to_air_mask, axis=1)) * voxel_size_mm[1],
                np.max(np.sum(bone_to_air_mask, axis=0)) * voxel_size_mm[0],
            )
            * 1.55
        )
        anomalies.append(
            SCTAnomalyFlag(
                anomaly_type="BONE_TO_AIR_SWAP",
                severity="CRITICAL" if wepl_b >= 5.0 else "HIGH",
                voxel_count=bone_to_air_voxels,
                volume_cc=vol_cc,
                mean_hu_discrepancy=float(np.mean(hu_err_map[bone_to_air_mask])),
                estimated_wepl_error_mm=wepl_b,
                description=(
                    f"Detected {bone_to_air_voxels} voxels ({vol_cc:.2f} cc) of cortical bone "
                    "silently misclassified as air (< -400 HU)."
                ),
            )
        )

    if missing_metal_voxels > 0:
        vol_cc = missing_metal_voxels * voxel_vol_cc
        wepl_m = float(
            max(
                np.max(np.sum(missing_metal_mask, axis=1)) * voxel_size_mm[1],
                np.max(np.sum(missing_metal_mask, axis=0)) * voxel_size_mm[0],
            )
            * 2.65
        )
        anomalies.append(
            SCTAnomalyFlag(
                anomaly_type="MISSING_METAL_ARTIFACT",
                severity="CRITICAL" if vol_cc >= 1.0 else "HIGH",
                voxel_count=missing_metal_voxels,
                volume_cc=vol_cc,
                mean_hu_discrepancy=float(np.mean(hu_err_map[missing_metal_mask])),
                estimated_wepl_error_mm=wepl_m,
                description=(
                    f"Detected {missing_metal_voxels} voxels ({vol_cc:.2f} cc) of high-Z implant "
                    "missing from sCT (replaced by soft-tissue HU)."
                ),
            )
        )

    if trunc_voxels > 0:
        vol_cc = trunc_voxels * voxel_vol_cc
        wepl_t = float(
            max(
                np.max(np.sum(trunc_mask, axis=1)) * voxel_size_mm[1],
                np.max(np.sum(trunc_mask, axis=0)) * voxel_size_mm[0],
            )
            * 1.04
        )
        anomalies.append(
            SCTAnomalyFlag(
                anomaly_type="BODY_CONTOUR_TRUNCATION",
                severity="HIGH" if wepl_t >= 4.0 else "MODERATE",
                voxel_count=trunc_voxels,
                volume_cc=vol_cc,
                mean_hu_discrepancy=float(np.mean(hu_err_map[trunc_mask])),
                estimated_wepl_error_mm=wepl_t,
                description=(
                    f"Detected {trunc_voxels} voxels ({vol_cc:.2f} cc) of truncated skin/body "
                    "contour where MRI shows tissue but sCT predicts air."
                ),
            )
        )

    if any(a.severity == "CRITICAL" for a in anomalies) or max_wepl_error_mm >= 6.0:
        status = "FAIL"
    elif anomalies or max_wepl_error_mm >= 2.0:
        status = "WARNING"
    else:
        status = "PASS"

    report = SCTAuditReport(
        volume_shape=tuple(int(s) for s in sct.shape),  # type: ignore[arg-type]
        voxel_size_mm=voxel_size_mm,
        body_volume_cc=float(np.sum(eval_body)) * voxel_vol_cc,
        mae_hu_body=mae_body,
        mae_hu_bone=mae_bone,
        mae_hu_air=mae_air,
        mean_rel_ed_error=mean_red_err,
        max_wepl_error_mm=max_wepl_error_mm,
        estimated_6mv_dose_error_pct=est_dose_error_pct,
        bone_to_air_swap_voxels=bone_to_air_voxels,
        missing_metal_voxels=missing_metal_voxels,
        contour_truncation_voxels=trunc_voxels,
        anomalies=anomalies,
        overall_status=status,
    )

    maps = {
        "hu_error_map": hu_err_map,
        "red_error_map": red_err_map,
        "bone_to_air_mask": bone_to_air_mask,
        "missing_metal_mask": missing_metal_mask,
        "truncation_mask": trunc_mask,
        "rel_electron_density_sct": red_sct,
    }
    return report, maps


# ---------------------------------------------------------------------------
# Lorentz-Force Electron-Return-Effect (ERE) Boundary Risk Kernel
# ---------------------------------------------------------------------------


def electron_gyroradius_mm(
    kinetic_energy_mev: float,
    b0_T: float,
    beam_to_b0_angle_deg: float = 90.0,
) -> float | None:
    """Compute relativistic transverse Lorentz gyroradius r_g = p_perp / (e * B0) [mm].

    For a secondary electron of kinetic energy T_e [MeV]:
        p * c = sqrt(T_e * (T_e + 2 * m_e * c^2))   [MeV]
        p_perp = p * sin(alpha)
        r_g = (p_perp * c [eV]) / (c [m/s] * B0 [T]) * 1000  [mm]

    Returns None when B0 == 0 or sin(alpha) == 0 (no Lorentz curvature).
    """
    sin_alpha = abs(float(np.sin(np.deg2rad(beam_to_b0_angle_deg))))
    if b0_T <= 0.0 or sin_alpha <= 1e-12:
        return None
    pc_mev = float(np.sqrt(kinetic_energy_mev * (kinetic_energy_mev + 2.0 * ELECTRON_REST_ENERGY_MEV)))
    p_over_e_Tm = (pc_mev * 1e6) / SPEED_OF_LIGHT_M_S
    r_g_m = (p_over_e_Tm * sin_alpha) / b0_T
    return float(r_g_m * 1000.0)


def compute_ere_boundary_risk(
    ct_or_sct_hu: np.ndarray,
    voxel_size_mm: tuple[float, float, float],
    b0_T: float = 1.5,
    beam_direction: tuple[float, float, float] = (0.0, 1.0, 0.0),
    b0_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    beam_energy_mv: float = 6.0,
    mean_secondary_electron_mev: float = 1.5,
    body_mask: np.ndarray | None = None,
) -> tuple[EREBoundaryRisk, dict[str, np.ndarray]]:
    """Evaluate 3D Lorentz-force Electron-Return-Effect (ERE) boundary dose-hotspot risk.

    Models secondary Compton electrons exiting tissue into air (rectum, trachea, bowel, skin)
    in a transverse magnetic field B0 (Raaijmakers et al. 2005, 2008).

    Physics scaling:
    - Relativistic gyroradius r_g = p_perp / (e * B0).
      At 1.5 MeV, 90 deg: r_g = 4.33 mm (1.5T), 18.54 mm (0.35T), inf (0T).
    - Tissue-to-air exit gradient alignment: max(0, u_beam . (-grad(rho_e))).
    - Air-gap clearance return probability: eta_return = 1 - exp(-L_air / (2 * r_g)),
      where L_air is the local contiguous air-cavity / external-air depth along the exit ray.
    - Magnetic field return-focusing saturation factor matching Raaijmakers (2008) 0T/0.35T/1.5T
      Monte Carlo trends (+0% at 0T, ~+10-16% at 0.35T, ~+28-40% at 1.5T on single-beam exit).
    """
    hu = np.asarray(ct_or_sct_hu, dtype=np.float64)
    if hu.ndim != 3:
        raise ValueError("ct_or_sct_hu must be a 3D array.")

    u_beam = np.asarray(beam_direction, dtype=np.float64)
    u_beam = u_beam / max(float(np.linalg.norm(u_beam)), 1e-12)
    u_b0 = np.asarray(b0_direction, dtype=np.float64)
    u_b0 = u_b0 / max(float(np.linalg.norm(u_b0)), 1e-12)
    cos_alpha = float(np.clip(np.dot(u_beam, u_b0), -1.0, 1.0))
    angle_deg = float(np.rad2deg(np.arccos(abs(cos_alpha))))
    sin_alpha = float(np.sin(np.deg2rad(angle_deg)))

    r_g_mm = electron_gyroradius_mm(
        kinetic_energy_mev=mean_secondary_electron_mev,
        b0_T=b0_T,
        beam_to_b0_angle_deg=angle_deg,
    )

    red = hu_to_relative_electron_density(hu)
    air_mask = hu < -450.0
    tissue_mask = hu >= -200.0
    if body_mask is None:
        body_mask = ~air_mask

    # Identify tissue-side interface voxels directly adjacent to air
    air_dilated = binary_dilation(air_mask, iterations=1)
    interface_mask = tissue_mask & air_dilated

    # Compute 3D electron-density gradient (1/mm)
    gx, gy, gz = np.gradient(red, voxel_size_mm[0], voxel_size_mm[1], voxel_size_mm[2])
    grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
    # Outward normal from tissue into air is along -grad(red)
    exit_projection = -(gx * u_beam[0] + gy * u_beam[1] + gz * u_beam[2])
    # Normalized alignment in [0, 1] plus lateral grazing component (Raaijmakers 2008 shows both
    # exit and oblique cavity walls experience ERE, strongest at perpendicular exit)
    norm_exit = np.divide(exit_projection, grad_mag, out=np.zeros_like(grad_mag), where=grad_mag > 1e-6)
    directional_weight = np.clip(0.30 + 0.70 * np.clip(norm_exit, 0.0, 1.0), 0.0, 1.0)

    # Local air-cavity clearance depth L_air [mm] via Euclidean distance transform in air
    # Diameter of air pocket adjacent to each tissue interface voxel (~2 * max EDT in air neighborhood)
    air_edt_mm = distance_transform_edt(air_mask, sampling=voxel_size_mm)
    # Propagate neighboring air depth onto the adjacent tissue interface layer
    local_air_depth_mm = np.zeros_like(hu)
    for shift_ax in range(3):
        for sgn in (-1, 1):
            shifted = np.roll(air_edt_mm, shift=sgn, axis=shift_ax)
            local_air_depth_mm = np.maximum(local_air_depth_mm, shifted)
    # Scale by 2.0 because EDT at wall is 1 voxel and cavity width is ~2*radius; cap at 60 mm (free air)
    cavity_clearance_mm = np.clip(local_air_depth_mm * 4.0, 0.0, 60.0)

    # Local density jump Delta_rho across the interface (in [0, 1.5])
    delta_rho = np.clip(red - 0.0012, 0.0, 1.2)

    ere_enhancement_pct = np.zeros_like(hu, dtype=np.float64)
    if r_g_mm is not None and b0_T > 0.0 and np.any(interface_mask):
        # Cavity clearance return probability: approaches 1 when air gap >> 2*r_g
        eta_return = 1.0 - np.exp(-cavity_clearance_mm / (2.0 * r_g_mm))
        # Field-strength focusing factor (Raaijmakers 2008: 0.35T ~ 0.45x of 1.5T peak)
        field_focus = (b0_T / (b0_T + 0.45)) * (sin_alpha**2)
        # Calibrated baseline single-beam perpendicular exit peak (~42% max at 1.5T, delta_rho=1.0)
        peak_scale_pct = 52.0 * (beam_energy_mv / 6.0) ** 0.25
        raw_ere = peak_scale_pct * field_focus * delta_rho * eta_return * directional_weight
        ere_enhancement_pct[interface_mask] = np.clip(raw_ere[interface_mask], 0.0, 65.0)

    iface_vals = ere_enhancement_pct[interface_mask]
    iface_count = int(np.sum(interface_mask))
    high_risk_count = int(np.sum(ere_enhancement_pct >= 20.0))

    if iface_count > 0:
        peak_pct = float(np.max(iface_vals))
        p95_pct = float(np.percentile(iface_vals, 95))
        mean_pct = float(np.mean(iface_vals))
    else:
        peak_pct = 0.0
        p95_pct = 0.0
        mean_pct = 0.0

    # Classify internal air cavity vs external skin exit interfaces
    internal_air = air_mask & binary_dilation(body_mask, iterations=1) & (
        binary_erosion(binary_dilation(body_mask, iterations=4), iterations=4)
    )
    dominant: list[str] = []
    if np.any(interface_mask & binary_dilation(internal_air, iterations=1)) and peak_pct > 5.0:
        dominant.append("internal_air_cavity_wall (rectum/trachea/sinus)")
    if np.any(interface_mask) and peak_pct > 5.0:
        dominant.append("external_skin_exit_surface")
    if not dominant:
        dominant.append("none_detected")

    if peak_pct < 1.0:
        tier = "NONE"
    elif peak_pct < 10.0:
        tier = "LOW"
    elif peak_pct < 20.0:
        tier = "MODERATE"
    elif peak_pct < 30.0:
        tier = "HIGH"
    else:
        tier = "CRITICAL"

    risk = EREBoundaryRisk(
        b0_T=b0_T,
        beam_energy_mv=beam_energy_mv,
        mean_secondary_electron_mev=mean_secondary_electron_mev,
        beam_to_b0_angle_deg=angle_deg,
        gyroradius_mm=r_g_mm,
        interface_voxel_count=iface_count,
        high_risk_voxel_count=high_risk_count,
        peak_dose_enhancement_pct=peak_pct,
        p95_interface_enhancement_pct=p95_pct,
        mean_interface_enhancement_pct=mean_pct,
        dominant_interfaces=dominant,
        risk_tier=tier,
    )

    maps = {
        "ere_enhancement_pct": ere_enhancement_pct,
        "interface_mask": interface_mask,
        "rel_electron_density": red,
        "cavity_clearance_mm": cavity_clearance_mm,
    }
    return risk, maps
