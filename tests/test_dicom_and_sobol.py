"""Unit tests for DICOM MR/CT lossless round-trip, NIfTI/NRRD I/O, and Saltelli-Jansen Sobol'."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mrphys_guard.dicom_io import (
    build_ct_dicom_dataset,
    build_mr_dicom_dataset,
    load_nifti_volume,
    load_nrrd_volume,
    read_dicom_physics,
    save_dicom,
    sequence_spec_from_dicom,
)
from mrphys_guard.ir import MRSequenceSpec
from mrphys_guard.sobol_sensitivity import (
    SobolParameterSpec,
    compute_sobol_saltelli_jansen,
    run_mr_uncertainty_sobol_suite,
)


def test_mr_and_ct_dicom_lossless_roundtrip(tmp_path: Path) -> None:
    """Verify pydicom MR & CT DICOM serialization preserves pixels and MR physics tags."""
    rng = np.random.default_rng(7)
    mr_slice = rng.uniform(0.0, 1.0, size=(32, 32))
    seq = MRSequenceSpec(
        name="Pelvis_T2_TSE",
        sequence_type="TSE",
        b0_T=1.5,
        bandwidth_hz_per_px=455.0,
        voxel_size_mm=(1.25, 1.25, 2.5),
        flip_angle_deg=90.0,
        echo_train_length=16,
        tr_ms=3200.0,
        te_ms=85.0,
        reported_sar_W_per_kg=1.42,
        reported_b1_rms_uT=1.85,
    )
    mr_ds = build_mr_dicom_dataset(mr_slice, seq)
    mr_path = save_dicom(mr_ds, tmp_path / "test_mr.dcm")

    arr_out, hdr = read_dicom_physics(mr_path)
    assert np.max(np.abs(arr_out - mr_slice)) < 1e-4
    assert hdr["Modality"] == "MR"
    assert hdr["MagneticFieldStrength"] == pytest.approx(1.5)
    assert hdr["PixelBandwidth"] == pytest.approx(455.0)
    assert hdr["SAR"] == pytest.approx(1.42)
    assert hdr["B1rms"] == pytest.approx(1.85)

    parsed_seq = sequence_spec_from_dicom(mr_path)
    assert parsed_seq.bandwidth_hz_per_px == pytest.approx(455.0)
    assert parsed_seq.tr_ms == pytest.approx(3200.0)
    assert parsed_seq.reported_b1_rms_uT == pytest.approx(1.85)

    # Exact integer HU round-trip for CT/sCT DICOM
    ct_slice = np.round(rng.uniform(-1000.0, 2800.0, size=(32, 32)))
    ct_ds = build_ct_dicom_dataset(ct_slice, voxel_size_mm=(1.0, 1.0, 2.0))
    ct_path = save_dicom(ct_ds, tmp_path / "test_sct.dcm")
    ct_out, ct_hdr = read_dicom_physics(ct_path)
    np.testing.assert_allclose(ct_out, ct_slice, atol=1e-9)
    assert ct_hdr["Modality"] == "CT"


def test_nifti_loader_roundtrip(tmp_path: Path) -> None:
    """NIfTI loader parses SynthRAD2023-style 3D .nii.gz volumes and exact voxel zooms."""
    vol = np.linspace(-1000.0, 1200.0, 16 * 16 * 12).reshape((16, 16, 12))
    affine = np.diag([1.0, 1.5, 2.5, 1.0])
    p = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(vol.astype(np.float32), affine), str(p))

    loaded = load_nifti_volume(p, modality="CT")
    assert loaded.array.shape == (16, 16, 12)
    assert loaded.voxel_size_mm == pytest.approx((1.0, 1.5, 2.5))
    np.testing.assert_allclose(loaded.array, vol, rtol=1e-5, atol=1e-3)


def test_sobol_ishigami_analytical_benchmark_and_mr_suite() -> None:
    """Validate Saltelli-Jansen QMC estimator on analytical Ishigami function and MR suite."""
    # Analytical Ishigami function: f(x1,x2,x3) = sin(x1) + 7 sin^2(x2) + 0.1 x3^4 sin(x1)
    # Exact Sobol indices: S1 = [0.3139, 0.4424, 0.0], ST = [0.5576, 0.4424, 0.2437]
    params = [
        SobolParameterSpec(name="x1", unit="rad", low=-np.pi, high=np.pi),
        SobolParameterSpec(name="x2", unit="rad", low=-np.pi, high=np.pi),
        SobolParameterSpec(name="x3", unit="rad", low=-np.pi, high=np.pi),
    ]

    def ishigami(x: np.ndarray) -> np.ndarray:
        return np.sin(x[:, 0]) + 7.0 * (np.sin(x[:, 1]) ** 2) + 0.1 * (x[:, 2] ** 4) * np.sin(x[:, 0])

    rep = compute_sobol_saltelli_jansen(
        ishigami, params, output_name="Ishigami", output_unit="a.u.", n_base=1024, seed=42
    )
    s1 = [r.s1 for r in rep.indices]
    st = [r.st for r in rep.indices]
    assert s1[0] == pytest.approx(0.3139, abs=0.04)
    assert s1[1] == pytest.approx(0.4424, abs=0.04)
    assert abs(s1[2]) < 0.04
    assert st[0] == pytest.approx(0.5576, abs=0.04)
    assert st[1] == pytest.approx(0.4424, abs=0.04)
    assert st[2] == pytest.approx(0.2437, abs=0.04)

    # MR-RT sensitivity suite invariants: 0 <= S1 <= ST + tol, sum(S1) <= 1.05, sum(ST) >= 0.95
    mr_suite = run_mr_uncertainty_sobol_suite(n_base=512, b0_T=1.5, seed=2026)
    for report in mr_suite.values():
        assert 0.85 <= report.sum_s1 <= 1.08
        assert report.sum_st >= 0.95
        for idx in report.indices:
            assert -0.05 <= idx.s1 <= 1.05
            assert 0.0 <= idx.st <= 1.05
            assert idx.st >= idx.s1 - 0.04


def test_public_ctmr_brain_nrrd_fixture_integrity() -> None:
    """Verify bundled 3D Slicer CT-MR Brain fixture loads cleanly and obeys ERE field ordering."""
    data_dir = Path(__file__).resolve().parents[1] / "data" / "public_ctmr_brain"
    ct_path = data_dir / "CT-brain.nrrd"
    mr_path = data_dir / "MR-brain-T1.nrrd"
    if not ct_path.exists():
        pytest.skip("Public CT-MR Brain fixture not fetched.")

    assert (data_dir / "LICENSE").exists()
    assert (data_dir / "MANIFEST.json").exists()
    assert (data_dir / "ATTRIBUTION.md").exists()

    ct_vol = load_nrrd_volume(ct_path, modality="CT")
    mr_vol = load_nrrd_volume(mr_path, modality="MR")
    assert ct_vol.array.ndim == 3 and ct_vol.array.shape[0] >= 256 and ct_vol.array.shape[2] >= 64
    assert mr_vol.array.ndim == 3 and mr_vol.array.shape[0] >= 256 and mr_vol.array.shape[2] >= 16
    assert all(v > 0.0 for v in ct_vol.voxel_size_mm)
    assert all(v > 0.0 for v in mr_vol.voxel_size_mm)
