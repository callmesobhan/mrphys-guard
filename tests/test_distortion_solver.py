"""Unit and physics-invariant tests for 3D MRI distortion solver (GNL + k-space B0 dipole)."""

from __future__ import annotations

import numpy as np
import pytest

from mrphys_guard.distortion_solver import (
    CHI_AIR_SI,
    CHI_WATER_SI,
    GAMMA_BAR_HZ_PER_T,
    build_pelvic_prostate_susceptibility_phantom,
    compute_3d_distortion_decomposition,
    dipole_kernel_kspace,
    fit_spherical_harmonic_gnl,
    generate_synthetic_grid_phantom,
    relative_snr_from_bandwidth,
    solve_susceptibility_b0_map,
    susceptibility_displacement_mm,
)
from mrphys_guard.ir import MRSequenceSpec
from mrphys_guard.provenance import AbstentionCode, AbstentionError, gate_b0_phase_unwrappability


def test_exact_lorentz_sphere_interior_zero_and_exterior_dipole_decay() -> None:
    """Analytical identity: Delta_B0 == 0 inside uniform sphere; exterior decays as 1/r^3 (3cos^2-1)."""
    n = 64
    vox = (1.0, 1.0, 1.0)
    coords = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
    r = np.sqrt(gx**2 + gy**2 + gz**2)

    radius = 12.0
    chi0 = CHI_WATER_SI - CHI_AIR_SI  # -9.41 ppm
    b0 = 1.5
    chi = np.where(r <= radius, chi0, 0.0)

    db0 = solve_susceptibility_b0_map(chi, voxel_size_mm=vox, b0_T=b0, b0_axis=2, pad_factor=2)

    # Interior Lorentz-sphere cancellation: Delta_B0 / (B0 * chi0) ~= 0
    interior = r <= (radius - 2.5)
    interior_rel_mean = abs(float(np.mean(db0[interior]))) / (b0 * abs(chi0))
    interior_rel_rms = float(np.sqrt(np.mean(db0[interior] ** 2))) / (b0 * abs(chi0))
    assert interior_rel_mean < 5e-3
    assert interior_rel_rms < 1.5e-2

    # Exterior dipole analytical solution: (1/3) * B0 * chi0 * (R/r)^3 * (3 cos^2 theta - 1)
    ext = (r >= radius + 4.0) & (r <= radius + 14.0)
    cos_theta = np.divide(gz, r, out=np.zeros_like(r), where=r > 0)
    analytic = (1.0 / 3.0) * b0 * chi0 * ((radius / np.maximum(r, 1e-6)) ** 3) * (3.0 * cos_theta**2 - 1.0)
    rel_l2 = float(np.linalg.norm(db0[ext] - analytic[ext]) / np.linalg.norm(analytic[ext]))
    assert rel_l2 < 0.04

    # Verify exact 1/r^3 power-law slope along +z polar axis
    mid = n // 2
    z_offsets = np.array([17, 20, 23, 26], dtype=np.float64)
    vals = np.array([abs(db0[mid, mid, int(mid + dz)]) for dz in z_offsets])
    slope, _ = np.polyfit(np.log(z_offsets), np.log(vals), 1)
    assert abs(slope - (-3.0)) < 0.15


def test_bandwidth_scaling_law_machine_precision() -> None:
    """Readout susceptibility displacement must scale as 1 / BW_px to machine precision."""
    rng = np.random.default_rng(123)
    db0 = rng.normal(0.0, 2.0e-6, size=(12, 12, 12))

    seq_a = MRSequenceSpec(
        name="BW200", b0_T=1.5, bandwidth_hz_per_px=200.0, voxel_size_mm=(1.2, 1.2, 2.0), tr_ms=10.0, te_ms=3.0
    )
    seq_b = seq_a.model_copy(update={"name": "BW600", "bandwidth_hz_per_px": 600.0})

    df_a, disp_a, _ = susceptibility_displacement_mm(db0, seq_a)
    df_b, disp_b, _ = susceptibility_displacement_mm(db0, seq_b)

    np.testing.assert_allclose(df_a, df_b, rtol=1e-15, atol=1e-15)
    # 3x higher BW -> exact 1/3 displacement to IEEE-754 machine precision
    np.testing.assert_allclose(disp_a / 3.0, disp_b, rtol=1e-14, atol=1e-15)


def test_epi_phase_encode_distortion_amplification() -> None:
    """EPI phase-encode axis must exhibit much larger distortion than readout due to low effective PE BW."""
    db0 = np.full((8, 8, 8), 1.5e-6)  # 1 ppm at 1.5T -> ~63.87 Hz
    seq_epi = MRSequenceSpec(
        name="EPI_DWI",
        sequence_type="EPI",
        b0_T=1.5,
        bandwidth_hz_per_px=1500.0,
        voxel_size_mm=(2.0, 2.0, 2.0),
        echo_spacing_ms=0.6,
        phase_encode_steps=128,
        parallel_accel_factor=2.0,
        tr_ms=4000.0,
        te_ms=70.0,
    )
    _, disp_vec, epi_pe = susceptibility_displacement_mm(db0, seq_epi)
    ro_disp = float(np.mean(np.abs(disp_vec[..., 0])))
    pe_disp = float(np.mean(np.abs(epi_pe)))
    expected_pe = GAMMA_BAR_HZ_PER_T * 1.5e-6 * (128 * 0.6e-3 / 2.0) * 2.0
    assert np.isclose(pe_disp, expected_pe, rtol=1e-12)
    assert pe_disp > 20.0 * ro_disp


def test_spherical_harmonic_gnl_fit_exact_recovery_and_noisy_robustness() -> None:
    """l <= 5 real solid spherical harmonics must recover synthetic odd-harmonic GNL to < 1e-6 mm."""
    true_pts, meas_pts, true_disp = generate_synthetic_grid_phantom(
        extent_mm=180.0, spacing_mm=36.0, max_edge_distortion_mm=3.0, noise_sigma_mm=0.0
    )
    model_l5 = fit_spherical_harmonic_gnl(true_pts, meas_pts, lmax=5, ridge_alpha=0.0)
    assert model_l5.fit_residual_rms_mm < 1e-6
    assert model_l5.fit_residual_max_mm < 1e-5

    # Compare lmax=3 vs lmax=5: lmax=5 captures the r^5 coil roll-off much better
    model_l3 = fit_spherical_harmonic_gnl(true_pts, meas_pts, lmax=3, ridge_alpha=0.0)
    assert model_l5.fit_residual_rms_mm < model_l3.fit_residual_rms_mm

    # With 0.03 mm realistic fiducial localization noise, RMS should stay near noise floor
    true_n, meas_n, _ = generate_synthetic_grid_phantom(
        extent_mm=180.0, spacing_mm=36.0, max_edge_distortion_mm=2.8, noise_sigma_mm=0.03, seed=99
    )
    model_noisy = fit_spherical_harmonic_gnl(true_n, meas_n, lmax=5)
    assert model_noisy.fit_residual_rms_mm < 0.08
    pred_n = model_noisy.predict(true_n)
    assert pred_n.shape == true_disp.shape


def test_pelvic_3d_decomposition_and_snr_scaling() -> None:
    """End-to-end 3D decomposition on pelvic phantom checks BW monotonicity and SNR law."""
    phantom = build_pelvic_prostate_susceptibility_phantom(shape=(32, 32, 24), voxel_size_mm=(5.0, 5.0, 5.0))
    seq_lo = MRSequenceSpec(
        name="LoBW", b0_T=1.5, bandwidth_hz_per_px=150.0, voxel_size_mm=(5.0, 5.0, 5.0), tr_ms=20.0, te_ms=5.0
    )
    seq_hi = seq_lo.model_copy(update={"name": "HiBW", "bandwidth_hz_per_px": 600.0})

    dec_lo, _ = compute_3d_distortion_decomposition(
        phantom["chi_si"], seq_lo, target_roi_mask=phantom["prostate_mask"], body_mask=phantom["body_mask"]
    )
    dec_hi, _ = compute_3d_distortion_decomposition(
        phantom["chi_si"], seq_hi, target_roi_mask=phantom["prostate_mask"], body_mask=phantom["body_mask"]
    )
    assert np.isclose(dec_lo.susceptibility_max_mm / 4.0, dec_hi.susceptibility_max_mm, rtol=1e-12)
    assert np.isclose(dec_lo.relative_snr / 2.0, dec_hi.relative_snr, rtol=1e-12)
    assert relative_snr_from_bandwidth(250.0) == pytest.approx(1.0)


def test_dipole_kernel_dc_zero_and_phase_unwrapping_gate() -> None:
    """DC component of dipole kernel is zero; severe B0 gradient triggers UNWRAPPABLE_B0_PHASE."""
    k = dipole_kernel_kspace((16, 16, 16), (1.0, 1.0, 1.0), b0_axis=2)
    assert k[0, 0, 0] == 0.0

    # Steep phase jump > pi rad at delta_TE = 5 ms
    steep_db0 = np.zeros((8, 8, 8), dtype=np.float64)
    steep_db0[4:, :, :] = 4.0e-6  # ~170 Hz -> 170 * 2pi * 0.005 = 5.35 rad > pi
    with pytest.raises(AbstentionError) as exc_info:
        gate_b0_phase_unwrappability(steep_db0, delta_te_s=5e-3, voxel_size_mm=(1.0, 1.0, 1.0))
    assert exc_info.value.code == AbstentionCode.UNWRAPPABLE_B0_PHASE
