"""3D MRI Geometric Distortion Solver:
1. System Gradient Nonlinearity (GNL) Kernel via 3D Real Solid Spherical Harmonics (l <= 5).
2. Patient Susceptibility B0 Dipole-Convolution Solver in k-space via exact Lorentz-sphere kernel:
       Delta_B0(k) = B0 * (1/3 - k_z^2 / |k|^2) * chi(k)
   (Marques & Bowtell, Concepts Magn Reson Part B, 2005; DOI: 10.1002/cmr.b.20034).
3. Exact Readout (1/BW_px) and EPI Phase-Encode Distortion Mapping & SNR Pareto Tradeoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import lpmv

from mrphys_guard.ir import DistortionDecomposition, MRSequenceSpec

# CODATA 2018/2022 proton gyromagnetic ratio gamma / (2*pi) [Hz/T]
GAMMA_BAR_HZ_PER_T: float = 42.577478518e6

# Canonical volumetric magnetic susceptibilities (SI, dimensionless, relative to vacuum)
CHI_AIR_SI: float = 0.36e-6          # Air (+0.36 ppm, paramagnetic O2)
CHI_WATER_SI: float = -9.05e-6       # Soft tissue / water (-9.05 ppm)
CHI_FAT_SI: float = -7.79e-6         # Adipose tissue (-7.79 ppm)
CHI_CORTICAL_BONE_SI: float = -11.3e-6  # Cortical bone (-11.3 ppm)
CHI_TITANIUM_SI: float = 182.0e-6    # Ti-6Al-4V hip/dental alloy (+182 ppm)


# ---------------------------------------------------------------------------
# 1. 3D Real Solid Spherical Harmonics (l <= 5) for Gradient Nonlinearity
# ---------------------------------------------------------------------------


def _real_solid_harmonics_matrix(
    coords_mm: np.ndarray,
    lmax: int = 5,
    r0_mm: float = 200.0,
) -> np.ndarray:
    """Construct design matrix of real regular solid harmonics R_l^m(x, y, z) for 0 <= l <= lmax.

    Basis functions are normalized by r0_mm^l and orthonormal spherical-harmonic factors so
    least-squares fitting on 3D grid-phantom fiducials is numerically well-conditioned.
    Total columns = (lmax + 1)^2.
    """
    pts = np.asarray(coords_mm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"coords_mm must have shape (N, 3); got {pts.shape}.")

    x = pts[:, 0] / r0_mm
    y = pts[:, 1] / r0_mm
    z = pts[:, 2] / r0_mm
    r = np.sqrt(x * x + y * y + z * z)
    cos_theta = np.divide(z, r, out=np.ones_like(r), where=r > 1e-15)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    phi = np.arctan2(y, x)

    n_pts = pts.shape[0]
    n_basis = (lmax + 1) ** 2
    design = np.empty((n_pts, n_basis), dtype=np.float64)

    col = 0
    for ell in range(lmax + 1):
        r_pow = r**ell
        for m in range(-ell, ell + 1):
            abs_m = abs(m)
            # Associated Legendre polynomial P_l^|m|(cos theta), Condon-Shortley phase stripped
            p_lm = lpmv(abs_m, ell, cos_theta) * ((-1.0) ** abs_m)
            # Normalization factor sqrt((2l+1)/(4pi) * (l-|m|)!/(l+|m|)!)
            fact_ratio = 1.0
            for k in range(ell - abs_m + 1, ell + abs_m + 1):
                fact_ratio /= float(k)
            norm = np.sqrt(((2 * ell + 1) / (4.0 * np.pi)) * fact_ratio)
            if m == 0:
                val = norm * r_pow * p_lm
            elif m > 0:
                val = np.sqrt(2.0) * norm * r_pow * p_lm * np.cos(abs_m * phi)
            else:
                val = np.sqrt(2.0) * norm * r_pow * p_lm * np.sin(abs_m * phi)
            design[:, col] = val
            col += 1
    return design


@dataclass(frozen=True)
class SphericalHarmonicGNLModel:
    """Fitted 3D real solid spherical-harmonic model for scanner gradient nonlinearity."""

    lmax: int
    r0_mm: float
    coeffs_xyz: np.ndarray  # shape ((lmax + 1)^2, 3) -> [dx, dy, dz] in mm
    fit_residual_rms_mm: float
    fit_residual_max_mm: float

    def predict(self, coords_mm: np.ndarray) -> np.ndarray:
        """Predict 3D system GNL displacement vectors (dx, dy, dz) [mm] at N coordinates."""
        design = _real_solid_harmonics_matrix(coords_mm, lmax=self.lmax, r0_mm=self.r0_mm)
        return design @ self.coeffs_xyz

    def evaluate_grid(
        self,
        shape: tuple[int, int, int],
        voxel_size_mm: tuple[float, float, float],
        isocenter_index: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        """Evaluate 3D GNL displacement field on a regular voxel grid -> shape (*shape, 3) [mm]."""
        if isocenter_index is None:
            isocenter_index = ((shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0, (shape[2] - 1) / 2.0)
        xs = (np.arange(shape[0], dtype=np.float64) - isocenter_index[0]) * voxel_size_mm[0]
        ys = (np.arange(shape[1], dtype=np.float64) - isocenter_index[1]) * voxel_size_mm[1]
        zs = (np.arange(shape[2], dtype=np.float64) - isocenter_index[2]) * voxel_size_mm[2]
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        coords = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        disp = self.predict(coords)
        return disp.reshape((*shape, 3))


def fit_spherical_harmonic_gnl(
    true_fiducials_mm: np.ndarray,
    measured_fiducials_mm: np.ndarray,
    lmax: int = 5,
    r0_mm: float = 200.0,
    ridge_alpha: float = 1e-10,
) -> SphericalHarmonicGNLModel:
    """Fit 3D solid spherical harmonics (l <= lmax) to 3D grid-phantom fiducial displacements.

    Displacement is defined as delta_r = r_measured - r_true [mm].
    """
    true_pts = np.asarray(true_fiducials_mm, dtype=np.float64)
    meas_pts = np.asarray(measured_fiducials_mm, dtype=np.float64)
    if true_pts.shape != meas_pts.shape or true_pts.ndim != 2 or true_pts.shape[1] != 3:
        raise ValueError("true_fiducials_mm and measured_fiducials_mm must both have shape (N, 3).")
    n_basis = (lmax + 1) ** 2
    if true_pts.shape[0] < n_basis:
        raise ValueError(
            f"Need at least {n_basis} fiducials to fit lmax={lmax}; got {true_pts.shape[0]}."
        )

    target_disp = meas_pts - true_pts
    design = _real_solid_harmonics_matrix(true_pts, lmax=lmax, r0_mm=r0_mm)

    if ridge_alpha > 0.0:
        ata = design.T @ design + ridge_alpha * np.eye(n_basis, dtype=np.float64)
        coeffs = np.linalg.solve(ata, design.T @ target_disp)
    else:
        coeffs, *_ = np.linalg.lstsq(design, target_disp, rcond=None)

    pred = design @ coeffs
    residuals = np.linalg.norm(pred - target_disp, axis=1)
    return SphericalHarmonicGNLModel(
        lmax=lmax,
        r0_mm=r0_mm,
        coeffs_xyz=coeffs,
        fit_residual_rms_mm=float(np.sqrt(np.mean(residuals**2))),
        fit_residual_max_mm=float(np.max(residuals)),
    )


def generate_synthetic_grid_phantom(
    extent_mm: float = 180.0,
    spacing_mm: float = 30.0,
    max_edge_distortion_mm: float = 2.8,
    noise_sigma_mm: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a realistic 3D AAPM TG-284 large-FOV gradient-nonlinearity grid phantom.

    Uses physically motivated odd-order spherical-harmonic coil roll-off (l=3, l=5 barrel/pincushion
    distortion symmetric about the magnet bore axis z), matching clinical 1.5T/3T cylindrical bore
    gradient coils (displacement grows roughly as r^3 and r^5 away from isocenter).

    Returns (true_fiducials_mm, measured_fiducials_mm, noiseless_displacement_mm).
    """
    coords_1d = np.arange(-extent_mm, extent_mm + 1e-6, spacing_mm, dtype=np.float64)
    gx, gy, gz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    # Keep fiducials inside cylindrical/spherical phantom envelope (R <= 210 mm)
    radii = np.linalg.norm(pts, axis=1)
    pts = pts[radii <= extent_mm * 1.15]

    r0 = 200.0
    xn = pts[:, 0] / r0
    yn = pts[:, 1] / r0
    zn = pts[:, 2] / r0
    rho2 = xn * xn + yn * yn
    z2 = zn * zn

    # Exact Maxwell-Laplace source-free cylindrical gradient-coil solid spherical harmonics
    # (every component satisfies Laplace's equation nabla^2(Delta r_i) = 0 identically!):
    #   l = 3: R_3^1  = x*(4*z^2 - rho^2),  R_3^-1 = y*(4*z^2 - rho^2),  R_3^0 = z*(2*z^2 - 3*rho^2)
    #   l = 5: R_5^1  = x*(8*z^4 - 12*z^2*rho^2 + rho^4),
    #          R_5^-1 = y*(8*z^4 - 12*z^2*rho^2 + rho^4),
    #          R_5^0  = z*(8*z^4 - 40*z^2*rho^2 + 15*rho^4)
    h3_xy = 4.0 * z2 - rho2
    h5_xy = 8.0 * (z2**2) - 12.0 * z2 * rho2 + (rho2**2)
    h3_z = 2.0 * z2 - 3.0 * rho2
    h5_z = 8.0 * (z2**2) - 40.0 * z2 * rho2 + 15.0 * (rho2**2)

    raw_dx = 0.55 * xn * h3_xy + 0.12 * xn * h5_xy
    raw_dy = 0.55 * yn * h3_xy + 0.12 * yn * h5_xy
    raw_dz = 0.45 * zn * h3_z - 0.045 * zn * h5_z
    raw_disp = np.column_stack([raw_dx, raw_dy, raw_dz])

    scale = max_edge_distortion_mm / float(np.max(np.linalg.norm(raw_disp, axis=1)))
    true_disp = raw_disp * scale

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_sigma_mm, size=true_disp.shape) if noise_sigma_mm > 0.0 else 0.0
    measured = pts + true_disp + noise
    return pts, measured, true_disp


# ---------------------------------------------------------------------------
# 2. 3D Magnetostatic Susceptibility Dipole-Convolution Solver (k-space)
# ---------------------------------------------------------------------------


def dipole_kernel_kspace(
    shape: tuple[int, int, int],
    voxel_size_mm: tuple[float, float, float],
    b0_axis: int = 2,
) -> np.ndarray:
    """Compute exact 3D Lorentz-sphere dipole kernel D(k) = 1/3 - k_z^2 / |k|^2 in k-space.

    At k = 0, D(0) = 0 (zero-mean macroscopic boundary convention).
    Ref: Marques & Bowtell (2005), Concepts Magn Reson Part B 25B(1):65-78.
    """
    if b0_axis not in (0, 1, 2):
        raise ValueError(f"b0_axis must be 0, 1, or 2; got {b0_axis}.")
    kx = np.fft.fftfreq(shape[0], d=voxel_size_mm[0])
    ky = np.fft.fftfreq(shape[1], d=voxel_size_mm[1])
    kz = np.fft.fftfreq(shape[2], d=voxel_size_mm[2])
    k_grids = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = k_grids[0] ** 2 + k_grids[1] ** 2 + k_grids[2] ** 2
    kb0_sq = k_grids[b0_axis] ** 2

    kernel = np.zeros(shape, dtype=np.float64)
    nonzero = k2 > 0.0
    kernel[nonzero] = (1.0 / 3.0) - (kb0_sq[nonzero] / k2[nonzero])
    return kernel


def solve_susceptibility_b0_map(
    chi_si: np.ndarray,
    voxel_size_mm: tuple[float, float, float],
    b0_T: float = 1.5,
    b0_axis: int = 2,
    pad_factor: int = 2,
) -> np.ndarray:
    """Solve 3D magnetostatic field perturbation Delta_B0(x) [Tesla] from susceptibility chi(x).

    Uses exact zero-padded 3D FFT dipole convolution:
        Delta_B0(k) = B0 * (1/3 - k_z^2 / |k|^2) * chi(k)

    Zero-padding (default pad_factor=2) suppresses periodic wrap-around aliasing so isolated
    objects (e.g. analytical sphere-in-uniform-field or patient anatomy in air) obey free-space
    dipole boundary conditions.
    """
    chi = np.asarray(chi_si, dtype=np.float64)
    if chi.ndim != 3:
        raise ValueError(f"chi_si must be a 3D array; got ndim={chi.ndim}.")
    if pad_factor < 1:
        raise ValueError("pad_factor must be >= 1.")

    orig_shape = chi.shape
    if pad_factor > 1:
        # Pad with background susceptibility (ambient air / boundary median) to avoid edge step
        bg_val = float(chi[0, 0, 0])
        pad_shape = tuple(int(s * pad_factor) for s in orig_shape)
        chi_pad = np.full(pad_shape, bg_val, dtype=np.float64)
        starts = tuple((ps - os) // 2 for ps, os in zip(pad_shape, orig_shape, strict=True))
        slices = tuple(slice(st, st + os) for st, os in zip(starts, orig_shape, strict=True))
        chi_pad[slices] = chi
        # Shift by background constant (D(0)=0 means constant chi shift adds zero Delta_B0)
        chi_work = chi_pad - bg_val
    else:
        chi_work = chi - float(chi[0, 0, 0])
        slices = (slice(None), slice(None), slice(None))

    kernel = dipole_kernel_kspace(chi_work.shape, voxel_size_mm=voxel_size_mm, b0_axis=b0_axis)
    chi_k = np.fft.fftn(chi_work)
    delta_b0_pad = np.fft.ifftn(b0_T * kernel * chi_k).real
    return np.ascontiguousarray(delta_b0_pad[slices], dtype=np.float64)


def susceptibility_displacement_mm(
    delta_b0_T: np.ndarray,
    sequence: MRSequenceSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map Delta_B0(x) [T] to frequency shift [Hz] and exact 3D voxel displacement [mm].

    Readout axis displacement:
        Delta_r_RO(x) = (Delta_f(x) / BW_px) * dx_RO
    EPI phase-encode axis displacement (when sequence_type == 'EPI' and echo_spacing_ms is set):
        Delta_r_PE(x) = Delta_f(x) * (N_PE * t_esp / R_PAT) * dx_PE

    Returns (delta_f_hz, disp_vec_mm [shape (*shape, 3)], epi_pe_disp_mm [shape shape]).
    """
    delta_f_hz = GAMMA_BAR_HZ_PER_T * np.asarray(delta_b0_T, dtype=np.float64)
    disp_vec = np.zeros((*delta_f_hz.shape, 3), dtype=np.float64)

    ro_ax = sequence.readout_axis
    dx_ro = sequence.voxel_size_mm[ro_ax]
    disp_vec[..., ro_ax] = (delta_f_hz / sequence.bandwidth_hz_per_px) * dx_ro

    epi_pe_disp = np.zeros_like(delta_f_hz)
    if sequence.sequence_type == "EPI" and sequence.echo_spacing_ms is not None:
        pe_ax = sequence.phase_encode_axis
        dx_pe = sequence.voxel_size_mm[pe_ax]
        t_esp_s = sequence.echo_spacing_ms * 1e-3
        eff_pe_bw = sequence.parallel_accel_factor / (sequence.phase_encode_steps * t_esp_s)
        epi_pe_disp = (delta_f_hz / eff_pe_bw) * dx_pe
        disp_vec[..., pe_ax] = epi_pe_disp

    return delta_f_hz, disp_vec, epi_pe_disp


def relative_snr_from_bandwidth(
    bandwidth_hz_per_px: float,
    reference_bw_hz_per_px: float = 250.0,
    voxel_size_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    reference_voxel_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    parallel_accel_factor: float = 1.0,
) -> float:
    """Compute exact physics relative SNR scaling: SNR proportional to voxel_vol / sqrt(BW * R)."""
    vol_ratio = float(np.prod(voxel_size_mm)) / float(np.prod(reference_voxel_mm))
    return float(
        vol_ratio * np.sqrt(reference_bw_hz_per_px / (bandwidth_hz_per_px * parallel_accel_factor))
    )


def compute_3d_distortion_decomposition(
    chi_si: np.ndarray,
    sequence: MRSequenceSpec,
    gnl_model: SphericalHarmonicGNLModel | None = None,
    target_roi_mask: np.ndarray | None = None,
    body_mask: np.ndarray | None = None,
    b0_axis: int = 2,
    srs_tolerance_mm: float = 0.5,
    pad_factor: int = 2,
) -> tuple[DistortionDecomposition, dict[str, np.ndarray]]:
    """Run full 3D MRI distortion decomposition (System GNL + Patient Susceptibility B0).

    Returns (DistortionDecomposition summary IR, dict of 3D spatial fields).
    """
    shape = chi_si.shape
    delta_b0_T = solve_susceptibility_b0_map(
        chi_si=chi_si,
        voxel_size_mm=sequence.voxel_size_mm,
        b0_T=sequence.b0_T,
        b0_axis=b0_axis,
        pad_factor=pad_factor,
    )
    delta_b0_ppm = (delta_b0_T / sequence.b0_T) * 1e6
    delta_f_hz, b0_disp_vec, epi_pe_disp = susceptibility_displacement_mm(delta_b0_T, sequence)

    if gnl_model is not None:
        gnl_disp_vec = gnl_model.evaluate_grid(shape, sequence.voxel_size_mm)
        lmax = gnl_model.lmax
        fit_rms = gnl_model.fit_residual_rms_mm
    else:
        gnl_disp_vec = np.zeros((*shape, 3), dtype=np.float64)
        lmax = 0
        fit_rms = 0.0

    total_disp_vec = gnl_disp_vec + b0_disp_vec
    gnl_mag = np.linalg.norm(gnl_disp_vec, axis=-1)
    b0_mag = np.linalg.norm(b0_disp_vec, axis=-1)
    total_mag = np.linalg.norm(total_disp_vec, axis=-1)

    eval_mask = body_mask.astype(bool) if body_mask is not None else np.ones(shape, dtype=bool)
    if not np.any(eval_mask):
        eval_mask = np.ones(shape, dtype=bool)

    roi_mask = target_roi_mask.astype(bool) if target_roi_mask is not None else eval_mask
    if not np.any(roi_mask):
        roi_mask = eval_mask

    gnl_vals = gnl_mag[eval_mask]
    b0_vals = b0_mag[eval_mask]
    tot_vals = total_mag[eval_mask]
    ppm_vals = delta_b0_ppm[eval_mask]
    roi_vals = total_mag[roi_mask]

    rel_snr = relative_snr_from_bandwidth(
        bandwidth_hz_per_px=sequence.bandwidth_hz_per_px,
        voxel_size_mm=sequence.voxel_size_mm,
        parallel_accel_factor=sequence.parallel_accel_factor,
    )

    target_max = float(np.max(roi_vals))
    total_max = float(np.max(tot_vals))

    summary = DistortionDecomposition(
        b0_T=sequence.b0_T,
        bandwidth_hz_per_px=sequence.bandwidth_hz_per_px,
        voxel_size_mm=sequence.voxel_size_mm,
        gnl_max_mm=float(np.max(gnl_vals)),
        gnl_rms_mm=float(np.sqrt(np.mean(gnl_vals**2))),
        gnl_p95_mm=float(np.percentile(gnl_vals, 95)),
        susceptibility_delta_b0_min_ppm=float(np.min(ppm_vals)),
        susceptibility_delta_b0_max_ppm=float(np.max(ppm_vals)),
        susceptibility_max_mm=float(np.max(b0_vals)),
        susceptibility_rms_mm=float(np.sqrt(np.mean(b0_vals**2))),
        epi_phase_encode_max_mm=float(np.max(np.abs(epi_pe_disp[eval_mask]))),
        total_max_mm=total_max,
        total_rms_mm=float(np.sqrt(np.mean(tot_vals**2))),
        total_p95_mm=float(np.percentile(tot_vals, 95)),
        target_roi_max_mm=target_max,
        target_roi_p95_mm=float(np.percentile(roi_vals, 95)),
        fraction_exceeding_0p5mm=float(np.mean(tot_vals > 0.5)),
        fraction_exceeding_1p0mm=float(np.mean(tot_vals > 1.0)),
        srs_tolerance_mm=srs_tolerance_mm,
        passes_srs_tolerance=bool(target_max <= srs_tolerance_mm),
        passes_tg284_2mm_dsv=bool(total_max <= 2.0),
        relative_snr=rel_snr,
        harmonic_order_lmax=lmax,
        harmonic_fit_residual_rms_mm=fit_rms,
    )

    fields = {
        "delta_b0_T": delta_b0_T,
        "delta_b0_ppm": delta_b0_ppm,
        "delta_f_hz": delta_f_hz,
        "gnl_disp_vec_mm": gnl_disp_vec,
        "gnl_mag_mm": gnl_mag,
        "b0_disp_vec_mm": b0_disp_vec,
        "b0_mag_mm": b0_mag,
        "total_disp_vec_mm": total_disp_vec,
        "total_mag_mm": total_mag,
    }
    return summary, fields


def build_pelvic_prostate_susceptibility_phantom(
    shape: tuple[int, int, int] = (64, 64, 48),
    voxel_size_mm: tuple[float, float, float] = (2.5, 2.5, 2.5),
    include_rectal_gas: bool = True,
    include_bilateral_hip_prostheses: bool = True,
    rectal_gas_radius_mm: float = 13.0,
) -> dict[str, np.ndarray]:
    """Construct a realistic 3D MR-only prostate radiotherapy anatomical phantom.

    Contains:
    - Ellipsoidal pelvic body contour (soft tissue, chi = -9.05 ppm, HU = +40)
    - Prostate CTV target ROI at isocenter
    - Posterior rectal gas pocket (chi = +0.36 ppm, HU = -1000) creating steep Delta_B0
    - Bilateral femoral heads / optional titanium hip prostheses (chi = +182 ppm, HU = +3000)
    - Pelvic bone ring (chi = -11.3 ppm, HU = +850)
    - Synthetic T1-weighted MRI contrast for sCT plausibility auditing
    """
    nx, ny, nz = shape
    dx, dy, dz = voxel_size_mm
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy
    z = (np.arange(nz) - (nz - 1) / 2.0) * dz
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")

    # Body contour: anterior-posterior y (~110 mm), left-right x (~150 mm), sup-inf z
    body_mask = ((gx / 72.0) ** 2 + (gy / 54.0) ** 2 + (gz / 56.0) ** 2) <= 1.0
    subcut_fat = body_mask & (((gx / 64.0) ** 2 + (gy / 46.0) ** 2 + (gz / 52.0) ** 2) > 1.0)

    # Prostate target at isocenter (radius ~18 mm)
    prostate_mask = ((gx / 18.0) ** 2 + ((gy + 2.0) / 15.0) ** 2 + (gz / 16.0) ** 2) <= 1.0

    # Rectal gas cavity directly posterior to prostate (y > 16 mm)
    if include_rectal_gas:
        rectal_gas_mask = (
            ((gx / (rectal_gas_radius_mm * 0.95)) ** 2
             + ((gy - 24.0) / rectal_gas_radius_mm) ** 2
             + (gz / 26.0) ** 2)
            <= 1.0
        ) & body_mask
    else:
        rectal_gas_mask = np.zeros(shape, dtype=bool)

    # Pelvic bone cortex / bilateral femoral heads
    left_hip = (((gx + 46.0) / 14.0) ** 2 + ((gy + 2.0) / 14.0) ** 2 + (gz / 18.0) ** 2) <= 1.0
    right_hip = (((gx - 46.0) / 14.0) ** 2 + ((gy + 2.0) / 14.0) ** 2 + (gz / 18.0) ** 2) <= 1.0
    sacrum = (((gx / 24.0) ** 2 + ((gy - 40.0) / 7.5) ** 2 + (gz / 30.0) ** 2) <= 1.0) & body_mask
    pubic_symph = (
        ((gx / 22.0) ** 2 + ((gy + 36.0) / 6.5) ** 2 + (gz / 12.0) ** 2) <= 1.0
    ) & body_mask

    bone_mask = (sacrum | pubic_symph) & body_mask
    if include_bilateral_hip_prostheses:
        metal_mask = (left_hip | right_hip) & body_mask
    else:
        metal_mask = np.zeros(shape, dtype=bool)
        bone_mask = bone_mask | ((left_hip | right_hip) & body_mask)

    # Susceptibility map [SI]
    chi = np.full(shape, CHI_AIR_SI, dtype=np.float64)
    chi[body_mask] = CHI_WATER_SI
    chi[subcut_fat] = CHI_FAT_SI
    chi[bone_mask] = CHI_CORTICAL_BONE_SI
    chi[rectal_gas_mask] = CHI_AIR_SI
    chi[metal_mask] = CHI_TITANIUM_SI

    # Ground-truth CT HU map
    ct_hu = np.full(shape, -1000.0, dtype=np.float64)
    ct_hu[body_mask] = 45.0
    ct_hu[subcut_fat] = -95.0
    ct_hu[prostate_mask] = 55.0
    ct_hu[bone_mask] = 850.0
    ct_hu[rectal_gas_mask] = -980.0
    ct_hu[metal_mask] = 3000.0

    # Synthetic T1-weighted MRI intensity [0..1] (high fat, intermediate muscle/prostate, dark bone/air/metal)
    mri_t1 = np.zeros(shape, dtype=np.float64)
    mri_t1[body_mask] = 0.55
    mri_t1[subcut_fat] = 0.88
    mri_t1[prostate_mask] = 0.62
    mri_t1[bone_mask] = 0.08
    mri_t1[rectal_gas_mask] = 0.03
    mri_t1[metal_mask] = 0.01

    return {
        "chi_si": chi,
        "ct_hu": ct_hu,
        "mri_t1": mri_t1,
        "body_mask": body_mask,
        "prostate_mask": prostate_mask,
        "rectal_gas_mask": rectal_gas_mask,
        "bone_mask": bone_mask,
        "metal_mask": metal_mask,
    }
