"""Unit tests for Synthetic-CT physics-plausibility auditor and Lorentz-force ERE boundary kernel."""

from __future__ import annotations

import numpy as np
import pytest

from mrphys_guard.distortion_solver import build_pelvic_prostate_susceptibility_phantom
from mrphys_guard.sct_ere_auditor import (
    audit_synthetic_ct,
    compute_ere_boundary_risk,
    electron_gyroradius_mm,
    hu_to_relative_electron_density,
    inject_sct_failure_modes,
)


def test_hu_to_relative_electron_density_monotonicity() -> None:
    """Stoichiometric HU -> relative electron density mapping must be strictly increasing."""
    hus = np.array([-1000.0, -700.0, -100.0, 0.0, 50.0, 300.0, 1000.0, 3000.0])
    reds = hu_to_relative_electron_density(hus)
    assert np.all(np.diff(reds) > 0.0)
    assert reds[0] == pytest.approx(0.0012, rel=1e-4)
    assert reds[3] == pytest.approx(1.0, rel=1e-6)
    assert reds[-1] == pytest.approx(3.70, rel=1e-4)


def test_sct_auditor_flags_bone_swap_missing_metal_and_truncation() -> None:
    """Perturbed sCT must trigger all three clinical failure modes and FAIL overall status."""
    phantom = build_pelvic_prostate_susceptibility_phantom(shape=(40, 40, 32), voxel_size_mm=(4.0, 4.0, 4.0))
    sct = inject_sct_failure_modes(
        ct_hu=phantom["ct_hu"],
        bone_mask=phantom["bone_mask"],
        metal_mask=phantom["metal_mask"],
        body_mask=phantom["body_mask"],
        swap_bone_fraction=0.35,
        drop_metal_to_tissue=True,
        truncate_body_border_voxels=2,
        hu_noise_sigma=10.0,
    )
    report, maps = audit_synthetic_ct(
        mri_volume=phantom["mri_t1"],
        sct_hu=sct,
        voxel_size_mm=(4.0, 4.0, 4.0),
        ground_truth_ct_hu=phantom["ct_hu"],
        body_mask=phantom["body_mask"],
    )
    assert report.overall_status == "FAIL"
    assert report.bone_to_air_swap_voxels > 0
    assert report.missing_metal_voxels > 0
    assert report.contour_truncation_voxels > 0
    assert report.max_wepl_error_mm > 10.0
    assert report.estimated_6mv_dose_error_pct > 3.0
    types = {a.anomaly_type for a in report.anomalies}
    assert {"BONE_TO_AIR_SWAP", "MISSING_METAL_ARTIFACT", "BODY_CONTOUR_TRUNCATION"}.issubset(types)
    assert maps["hu_error_map"].shape == sct.shape


def test_sct_auditor_passes_clean_ct_and_works_without_ground_truth() -> None:
    """Unperturbed CT passes; reference-free MRI-only audit still detects injected errors."""
    phantom = build_pelvic_prostate_susceptibility_phantom(shape=(36, 36, 28), voxel_size_mm=(4.0, 4.0, 4.0))
    clean_rep, _ = audit_synthetic_ct(
        mri_volume=phantom["mri_t1"],
        sct_hu=phantom["ct_hu"],
        voxel_size_mm=(4.0, 4.0, 4.0),
        ground_truth_ct_hu=phantom["ct_hu"],
        body_mask=phantom["body_mask"],
    )
    assert clean_rep.overall_status == "PASS"
    assert clean_rep.max_wepl_error_mm == pytest.approx(0.0, abs=1e-9)

    # Reference-free audit (ground_truth_ct_hu=None) still catches missing metal & truncation
    sct_bad = inject_sct_failure_modes(
        ct_hu=phantom["ct_hu"],
        bone_mask=phantom["bone_mask"],
        metal_mask=phantom["metal_mask"],
        body_mask=phantom["body_mask"],
        drop_metal_to_tissue=True,
        truncate_body_border_voxels=2,
    )
    ref_free_rep, _ = audit_synthetic_ct(
        mri_volume=phantom["mri_t1"],
        sct_hu=sct_bad,
        voxel_size_mm=(4.0, 4.0, 4.0),
        ground_truth_ct_hu=None,
        body_mask=phantom["body_mask"],
    )
    assert ref_free_rep.mae_hu_body is None
    assert ref_free_rep.missing_metal_voxels > 0
    assert ref_free_rep.contour_truncation_voxels > 0
    assert ref_free_rep.overall_status in {"WARNING", "FAIL"}


def test_relativistic_electron_gyroradius_exact_values() -> None:
    """Verify exact relativistic Lorentz gyroradius r_g = p_perp / (e B0) at 1.5T, 0.35T, 0T."""
    rg_1p5 = electron_gyroradius_mm(1.5, 1.5, 90.0)
    rg_0p35 = electron_gyroradius_mm(1.5, 0.35, 90.0)
    rg_0 = electron_gyroradius_mm(1.5, 0.0, 90.0)
    rg_parallel = electron_gyroradius_mm(1.5, 1.5, 0.0)

    assert rg_1p5 is not None and rg_0p35 is not None
    assert rg_1p5 == pytest.approx(4.325, rel=1e-2)
    assert rg_0p35 == pytest.approx(18.536, rel=1e-2)
    assert np.isclose(rg_0p35 / rg_1p5, 1.5 / 0.35, rtol=1e-12)
    assert rg_0 is None
    assert rg_parallel is None


def test_ere_field_strength_ordering_1p5t_gt_0p35t_gt_0t() -> None:
    """Lorentz-force ERE boundary dose spikes must strictly obey 1.5T > 0.35T > 0T = 0%."""
    phantom = build_pelvic_prostate_susceptibility_phantom(shape=(40, 40, 28), voxel_size_mm=(4.0, 4.0, 4.0))
    ere_0, _ = compute_ere_boundary_risk(phantom["ct_hu"], (4.0, 4.0, 4.0), b0_T=0.0, body_mask=phantom["body_mask"])
    ere_035, _ = compute_ere_boundary_risk(phantom["ct_hu"], (4.0, 4.0, 4.0), b0_T=0.35, body_mask=phantom["body_mask"])
    ere_15, _ = compute_ere_boundary_risk(phantom["ct_hu"], (4.0, 4.0, 4.0), b0_T=1.5, body_mask=phantom["body_mask"])

    assert ere_0.peak_dose_enhancement_pct == 0.0
    assert ere_0.risk_tier == "NONE"
    assert ere_035.peak_dose_enhancement_pct > 5.0
    assert ere_15.peak_dose_enhancement_pct > ere_035.peak_dose_enhancement_pct > ere_0.peak_dose_enhancement_pct
    assert ere_15.peak_dose_enhancement_pct >= 20.0
    assert ere_15.risk_tier in {"HIGH", "CRITICAL"}
