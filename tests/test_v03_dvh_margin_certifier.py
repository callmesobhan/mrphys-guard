"""Tests for v0.3.0 MR-Only Ambiguity-to-DVH & Anisotropic van Herk PTV Margin Certifier
and real pydicom RTDOSE / RTSTRUCT round-trip I/O.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mrphys_guard.cli import main as cli_main
from mrphys_guard.dicom_io import (
    build_rtdose_dicom_dataset,
    build_rtstruct_dicom_dataset,
    read_rtdose_dicom,
    read_rtstruct_dicom,
    save_dicom,
)
from mrphys_guard.distortion_solver import (
    build_pelvic_prostate_susceptibility_phantom,
    fit_spherical_harmonic_gnl,
    generate_synthetic_grid_phantom,
)
from mrphys_guard.dvh_margin_certifier import (
    build_clinical_srs_dose_grid,
    build_rectal_wall_oar_mask,
    certify_mronly_dvh_and_anisotropic_margins,
    compute_dvh_metrics,
    compute_lkb_ntcp_pct,
    compute_niemierko_tcp_pct,
)
from mrphys_guard.harness.mcp_server import handle_jsonrpc_request
from mrphys_guard.ir import MRSequenceSpec


def test_rtdose_and_rtstruct_dicom_roundtrip(tmp_path: Path) -> None:
    """Lossless 32-bit RTDOSE and CLOSED_PLANAR RTSTRUCT DICOM round-trip."""
    vox = (3.0, 3.0, 3.0)
    phantom = build_pelvic_prostate_susceptibility_phantom(shape=(32, 32, 24), voxel_size_mm=vox)
    oar_wall = build_rectal_wall_oar_mask(
        phantom["rectal_gas_mask"], phantom["body_mask"], phantom["prostate_mask"]
    )
    dose = build_clinical_srs_dose_grid(phantom["prostate_mask"], oar_wall, vox, prescription_gy=36.25)

    rtdose_path = save_dicom(build_rtdose_dicom_dataset(dose, vox), tmp_path / "rtdose.dcm")
    loaded_dose = read_rtdose_dicom(rtdose_path)
    assert loaded_dose.modality == "RTDOSE"
    assert loaded_dose.voxel_size_mm == pytest.approx(vox)
    assert float(np.max(np.abs(loaded_dose.array - dose))) < 1e-5

    rtstruct_path = save_dicom(
        build_rtstruct_dicom_dataset({"Prostate_CTV": phantom["prostate_mask"], "Rectal_Wall": oar_wall}, vox),
        tmp_path / "rtstruct.dcm",
    )
    recovered = read_rtstruct_dicom(rtstruct_path, phantom["prostate_mask"].shape, vox)
    assert set(recovered) == {"Prostate_CTV", "Rectal_Wall"}
    inter = int(np.sum(recovered["Prostate_CTV"] & phantom["prostate_mask"]))
    dice = 2.0 * inter / max(1, int(np.sum(recovered["Prostate_CTV"]) + np.sum(phantom["prostate_mask"])))
    assert dice > 0.90


def test_dvh_metrics_and_tcp_ntcp_monotonicity() -> None:
    """TCP increases with target dose; LKB NTCP increases with OAR dose."""
    d_lo = np.full(500, 26.0)
    d_hi = np.full(500, 37.0)
    assert compute_niemierko_tcp_pct(d_hi) > compute_niemierko_tcp_pct(d_lo)
    assert compute_lkb_ntcp_pct(d_hi) > compute_lkb_ntcp_pct(d_lo)

    m = compute_dvh_metrics(np.linspace(20.0, 40.0, 200), voxel_vol_cc=0.01)
    assert m["d0p1cc"] >= m["dmean"] >= m["d98"]


def test_mronly_ambiguity_dvh_certifier_and_readout_axis_optimization() -> None:
    """Certifier evaluates all 6 readout axis/polarities, anisotropic van Herk margins, and dark-MRI blind spot."""
    vox = (4.0, 4.0, 4.0)
    phantom = build_pelvic_prostate_susceptibility_phantom(shape=(36, 36, 28), voxel_size_mm=vox)
    oar_wall = build_rectal_wall_oar_mask(
        phantom["rectal_gas_mask"], phantom["body_mask"], phantom["prostate_mask"]
    )
    dose = build_clinical_srs_dose_grid(phantom["prostate_mask"], oar_wall, vox, prescription_gy=36.25)
    true_pts, meas_pts, _ = generate_synthetic_grid_phantom(extent_mm=150.0, spacing_mm=50.0)
    gnl = fit_spherical_harmonic_gnl(true_pts, meas_pts, lmax=5)

    seq_low_bw = MRSequenceSpec(
        name="Pelvis_LowBW",
        b0_T=1.5,
        bandwidth_hz_per_px=220.0,
        voxel_size_mm=vox,
        readout_axis=1,
        phase_encode_axis=0,
        tr_ms=15.0,
        te_ms=4.5,
    )
    cert_lo, art = certify_mronly_dvh_and_anisotropic_margins(
        chi_si=phantom["chi_si"],
        mri_volume=phantom["mri_t1"],
        ct_or_sct_hu=phantom["ct_hu"],
        dose_gy=dose,
        target_mask=phantom["prostate_mask"],
        oar_mask=oar_wall,
        body_mask=phantom["body_mask"],
        sequence=seq_low_bw,
        gnl_model=gnl,
        prescription_gy=36.25,
    )

    assert len(cert_lo.readout_candidates) == 6
    assert len(cert_lo.organ_impacts) == 2
    assert cert_lo.dark_mri_certificate.ambiguous_voxel_count > 0
    assert cert_lo.dark_mri_certificate.worst_case_wepl_error_mm > 0.0
    assert cert_lo.organ_impacts[0].worst_case_d98_gy <= cert_lo.organ_impacts[0].distorted_d98_gy
    assert cert_lo.organ_impacts[1].worst_case_d0p1cc_gy >= cert_lo.organ_impacts[1].distorted_d0p1cc_gy
    assert art["dvh_ctv_planned"].shape == art["dose_bins_gy"].shape

    # Raising receiver BW shrinks anisotropic PTV margin along the readout axis!
    seq_hi_bw = seq_low_bw.model_copy(update={"bandwidth_hz_per_px": 700.0})
    cert_hi, _ = certify_mronly_dvh_and_anisotropic_margins(
        chi_si=phantom["chi_si"],
        mri_volume=phantom["mri_t1"],
        ct_or_sct_hu=phantom["ct_hu"],
        dose_gy=dose,
        target_mask=phantom["prostate_mask"],
        oar_mask=oar_wall,
        body_mask=phantom["body_mask"],
        sequence=seq_hi_bw,
        gnl_model=gnl,
        prescription_gy=36.25,
    )
    assert cert_hi.baseline_anisotropic_margin_mm[1] < cert_lo.baseline_anisotropic_margin_mm[1]


def test_mcp_tool_9_and_cli_certify_dvh(capsys: pytest.CaptureFixture[str]) -> None:
    """MCP tool mrphys_certify_mronly_dvh_margins and CLI certify-dvh return valid certificates."""
    resp = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "tools/call",
            "params": {
                "name": "mrphys_certify_mronly_dvh_margins",
                "arguments": {"grid_size": 24, "bandwidth_hz_per_px": 260.0, "b0_T": 1.5},
            },
        }
    )
    assert resp is not None
    out = resp["result"]["structuredContent"]["result"]
    assert len(out["readout_candidates"]) == 6
    assert out["overall_certificate"] in {"CERTIFIED_ROBUST", "MARGIN_ADJUSTMENT_REQUIRED", "UNSAFE_BLIND_SPOT"}

    assert cli_main(["certify-dvh", "--bw", "260", "--b0", "1.5"]) == 0
    cli_out = json.loads(capsys.readouterr().out)
    assert "recommended_readout_label" in cli_out
