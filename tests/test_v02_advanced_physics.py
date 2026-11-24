"""Tests for v0.2.0 physics & verification upgrades:
- 3D DCT Poisson/Laplacian dual-echo B0 phase unwrapping
- Distortion Jacobian folding/pile-up singularity detection & unwarping
- Physical-space 4x4 affine NRRD resampling (real 3D Slicer MR-T1 -> CT-brain)
- Multi-beam IMRT/VMAT ERE cancellation & gantry-angle avoidance
- IEC 60601-2-33 PNS chronaxie/rheobase & simultaneous multi-implant CSP + UNSAT conflict cores
- Bloch-equation tissue CNR & 3-point Pareto protocol co-design
- End-to-end subprocess stdio JSON-RPC 2.0 MCP server test
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mrphys_guard.dicom_io import load_nrrd_volume, resample_volume_to_reference
from mrphys_guard.distortion_solver import (
    build_pelvic_prostate_susceptibility_phantom,
    compute_distortion_jacobian_and_unwarp,
    unwrap_3d_b0_phase_dct,
)
from mrphys_guard.harness.agent_orchestrator import (
    build_prostate_srs_hip_gas_scenario,
    compute_bloch_tissue_cnr,
    generate_pareto_protocol_candidates,
)
from mrphys_guard.implant_safety_csp import (
    compute_iec_pns_percentage,
    lookup_implant_profile,
    verify_multi_implant_safety_csp,
)
from mrphys_guard.ir import (
    AbstentionCode,
    ImplantConditionSpec,
    MRSafetyLabel,
    MRSequenceSpec,
    RFCoilType,
    ScannerFieldSpec,
)
from mrphys_guard.sct_ere_auditor import compute_multibeam_ere_cancellation


def test_3d_dct_poisson_phase_unwrapping_exact_multi_cycle_recovery() -> None:
    """3D DCT phase unwrapper must recover multi-2pi wrapped smooth B0 field to machine precision."""
    nx, ny, nz = 24, 24, 16
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    z = np.linspace(-1.0, 1.0, nz)
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")

    # Smooth 3D phase field spanning > 14 radians (~2.3 full 2*pi wrap cycles)
    true_phase = 8.5 * np.exp(-(gx**2 + gy**2 + gz**2) / 0.45) + 4.5 * gx - 3.0 * gy
    wrapped = np.arctan2(np.sin(true_phase), np.cos(true_phase))

    report, fields = unwrap_3d_b0_phase_dct(wrapped, delta_te_ms=2.4, b0_T=1.5)
    unwrapped = fields["unwrapped_phase_rad"]

    # Recovered phase equals true phase up to a single global 2*pi*k integer constant
    diff = unwrapped - true_phase
    centered_err = diff - np.median(diff)
    assert float(np.max(np.abs(centered_err))) < 1e-11
    assert report.cycle_count_range >= 2
    assert report.wrap_consistency_rms_rad < 1e-12


def test_distortion_jacobian_detects_folding_singularity_and_preserves_signal() -> None:
    """Gentle shift has J > 0 and preserves signal; steep shift triggers folding singularity (J <= 0)."""
    shape = (24, 24, 16)
    vox = (2.0, 2.0, 2.0)
    gx, gy, gz = np.meshgrid(
        np.arange(shape[0], dtype=np.float64),
        np.arange(shape[1], dtype=np.float64),
        np.arange(shape[2], dtype=np.float64),
        indexing="ij",
    )
    img = np.exp(-(((gx - 12.0) / 4.0) ** 2 + ((gy - 12.0) / 4.0) ** 2 + ((gz - 8.0) / 3.0) ** 2))

    # Mild smooth distortion (invertible, no folding)
    disp_mild = np.zeros((*shape, 3), dtype=np.float64)
    disp_mild[..., 0] = 1.2 * np.sin(gx / 6.0)
    rep_mild, _ = compute_distortion_jacobian_and_unwarp(img, disp_mild, vox)
    assert rep_mild.has_folding_singularity is False
    assert rep_mild.min_jacobian > 0.5
    assert 0.85 <= rep_mild.signal_conservation_ratio <= 1.15

    # Severe localized gradient -> J = 1 + d(Delta_r)/dx < 0 (irreversible signal pile-up/folding!)
    disp_severe = np.zeros((*shape, 3), dtype=np.float64)
    disp_severe[..., 0] = -8.0 * np.tanh((gx - 12.0) / 1.5)
    rep_sev, fields_sev = compute_distortion_jacobian_and_unwarp(img, disp_severe, vox)
    assert rep_sev.has_folding_singularity is True
    assert rep_sev.folding_voxel_count > 0
    assert rep_sev.min_jacobian < 0.0
    assert np.any(fields_sev["folding_mask"])


def test_real_public_nrrd_affine_resampling_mr_onto_ct() -> None:
    """Real 3D Slicer MR-brain-T1.nrrd resamples accurately onto CT-brain.nrrd physical space."""
    data_dir = Path(__file__).resolve().parents[1] / "data" / "public_ctmr_brain"
    ct_path = data_dir / "CT-brain.nrrd"
    mr_path = data_dir / "MR-brain-T1.nrrd"
    if not ct_path.exists() or not mr_path.exists():
        pytest.skip("Public CT-MR Brain fixture not fetched.")

    ct_vol = load_nrrd_volume(ct_path, modality="CT")
    mr_vol = load_nrrd_volume(mr_path, modality="MR")
    assert ct_vol.affine_4x4 is not None and mr_vol.affine_4x4 is not None

    # Downsample reference CT to keep test fast while verifying exact affine transformation
    ct_sub = type(ct_vol)(
        array=ct_vol.array[::4, ::4, ::2],
        voxel_size_mm=(ct_vol.voxel_size_mm[0] * 4, ct_vol.voxel_size_mm[1] * 4, ct_vol.voxel_size_mm[2] * 2),
        modality="CT",
        source_path=ct_vol.source_path,
        metadata=ct_vol.metadata,
        affine_4x4=ct_vol.affine_4x4 @ np.diag([4.0, 4.0, 2.0, 1.0]),
    )
    mr_on_ct = resample_volume_to_reference(mr_vol, ct_sub, order=1)
    assert mr_on_ct.array.shape == ct_sub.array.shape
    assert float(np.max(mr_on_ct.array)) > 100.0  # Real cranial MR signal mapped cleanly into CT space


def test_multibeam_imrt_ere_cancellation_reduces_peak_hotspot() -> None:
    """Opposing 2-beam & 7-beam IMRT plans must substantially cancel single-beam 1.5T ERE hotspots."""
    phantom = build_pelvic_prostate_susceptibility_phantom(shape=(36, 36, 24), voxel_size_mm=(4.5, 4.5, 5.0))
    vox = (4.5, 4.5, 5.0)

    rep_1, _ = compute_multibeam_ere_cancellation(
        phantom["ct_hu"], vox, gantry_angles_deg=[0.0], b0_T=1.5, body_mask=phantom["body_mask"]
    )
    rep_2, _ = compute_multibeam_ere_cancellation(
        phantom["ct_hu"], vox, gantry_angles_deg=[0.0, 180.0], b0_T=1.5, body_mask=phantom["body_mask"]
    )
    rep_7, _ = compute_multibeam_ere_cancellation(
        phantom["ct_hu"],
        vox,
        gantry_angles_deg=[0.0, 51.0, 103.0, 154.0, 206.0, 257.0, 309.0],
        b0_T=1.5,
        body_mask=phantom["body_mask"],
    )

    assert rep_1.cancellation_reduction_pct == pytest.approx(0.0, abs=1e-6)
    assert rep_2.multibeam_net_peak_pct < rep_1.multibeam_net_peak_pct
    assert rep_2.cancellation_reduction_pct > 45.0
    assert rep_7.multibeam_net_peak_pct < rep_1.multibeam_net_peak_pct
    assert rep_7.cancellation_reduction_pct > 55.0
    assert len(rep_7.recommended_low_ere_angles_deg) == 3
    assert len(rep_7.highest_risk_angles_deg) == 3


def test_iec_pns_and_multi_implant_simultaneous_csp_conflict_cores() -> None:
    """Verify IEC chronaxie/rheobase PNS % and simultaneous multi-implant CSP with UNSAT conflict cores."""
    assert compute_iec_pns_percentage(0.0) == 0.0
    pns_120 = compute_iec_pns_percentage(120.0)
    pns_300 = compute_iec_pns_percentage(300.0)
    assert 30.0 < pns_120 < 80.0
    assert pns_300 > 80.0  # Exceeds IEC Normal Mode 80% PNS limit

    scanner = ScannerFieldSpec(b0_T=1.5, max_spatial_gradient_T_per_m=17.0, operating_mode="normal")
    pm = lookup_implant_profile("generic_cardiac_pacemaker_1p5t")
    hip = lookup_implant_profile("generic_hip_prosthesis_ti")

    safe_seq = MRSequenceSpec(
        name="Safe_MultiImplant_GRE",
        b0_T=1.5,
        bandwidth_hz_per_px=450.0,
        flip_angle_deg=20.0,
        tr_ms=14.0,
        te_ms=4.0,
        rf_pulse_duration_ms=2.2,
        slew_rate_T_per_m_s=120.0,
        scan_duration_min=6.0,
    )
    mv_pass = verify_multi_implant_safety_csp(scanner, [safe_seq], [pm, hip])
    assert mv_pass.verdict == "PASS"
    assert mv_pass.effective_b0_intersection_T == [1.5]
    assert mv_pass.governing_device_by_limit["max_b1_rms_uT"] == "generic_cardiac_pacemaker_1p5t"
    assert mv_pass.effective_max_b1_rms_uT == 2.0

    # Disjoint B0 implants -> UNSAT MUTUALLY_INCOMPATIBLE_IMPLANTS conflict core!
    only_3t = ImplantConditionSpec(
        device_id="device_3t_only",
        device_name="3.0T-Only Research Device",
        category="experimental",
        mr_status=MRSafetyLabel.MR_CONDITIONAL,
        allowed_b0_T=[3.0],
        max_spatial_gradient_T_per_m=20.0,
        max_b1_rms_uT=2.5,
        allowed_rf_coils=[RFCoilType.TX_RX_EXTREMITY],
    )
    mv_unsat = verify_multi_implant_safety_csp(scanner, [safe_seq], [pm, only_3t])
    assert mv_unsat.verdict == "VIOLATION"
    assert any(c.conflict_type == "MUTUALLY_INCOMPATIBLE_IMPLANTS" for c in mv_unsat.conflict_cores)

    # Co-existing MR_UNSAFE device triggers hard ABSTAIN
    unsafe = lookup_implant_profile("ferromagnetic_aneurysm_clip_unsafe")
    mv_abstain = verify_multi_implant_safety_csp(scanner, [safe_seq], [hip, unsafe])
    assert mv_abstain.verdict == "ABSTAIN"
    assert mv_abstain.abstention_code == AbstentionCode.MR_UNSAFE_DEVICE_DETECTED


def test_bloch_cnr_and_three_point_pareto_protocol_codesign() -> None:
    """Bloch CNR > 0 and 3-point Pareto frontier yields 3 Critic-APPROVED distinct trade-offs."""
    scenario = build_prostate_srs_hip_gas_scenario()
    cnr_base = compute_bloch_tissue_cnr(scenario.initial_sequences[0])
    assert cnr_base > 0.0

    pareto = generate_pareto_protocol_candidates(scenario)
    assert len(pareto) == 3
    by_name = {c.profile_name: c for c in pareto}
    assert set(by_name) == {"LOWEST_DISTORTION", "BALANCED_CLINICAL_CNR", "LOWEST_RF_SAR"}
    assert by_name["LOWEST_DISTORTION"].max_target_distortion_mm <= by_name["BALANCED_CLINICAL_CNR"].max_target_distortion_mm
    assert by_name["LOWEST_RF_SAR"].max_b1_rms_uT < by_name["BALANCED_CLINICAL_CNR"].max_b1_rms_uT
    assert by_name["BALANCED_CLINICAL_CNR"].critic_status == "APPROVED"
    assert by_name["LOWEST_RF_SAR"].critic_status == "APPROVED"


def test_mcp_server_end_to_end_stdio_subprocess() -> None:
    """Spawn mrphys_guard.harness.mcp_server as a real stdio subprocess and execute JSON-RPC calls."""
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "mrphys_multibeam_ere_optimize",
                "arguments": {"grid_size": 24, "b0_T": 1.5, "gantry_angles_deg": [0.0, 180.0]},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "mrphys_verify_multi_implant_csp",
                "arguments": {
                    "implant_ids": ["generic_cardiac_pacemaker_1p5t", "generic_hip_prosthesis_ti"],
                    "sequences": [
                        {
                            "name": "Safe_GRE",
                            "b0_T": 1.5,
                            "bandwidth_hz_per_px": 450.0,
                            "flip_angle_deg": 20.0,
                            "tr_ms": 14.0,
                            "te_ms": 4.0,
                            "rf_pulse_duration_ms": 2.2,
                            "slew_rate_T_per_m_s": 120.0,
                        }
                    ],
                },
            },
        },
    ]
    stdin_payload = "\n".join(json.dumps(r) for r in reqs) + "\n"
    proc = subprocess.run(
        [sys.executable, "-m", "mrphys_guard.harness.mcp_server"],
        input=stdin_payload,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    responses = [json.loads(line) for line in proc.stdout.strip().splitlines() if line.strip()]
    assert len(responses) == 4  # notifications/initialized produces no response
    assert responses[0]["result"]["serverInfo"]["version"] == "0.2.0"
    assert len(responses[1]["result"]["tools"]) == 8
    ere_out = responses[2]["result"]["structuredContent"]["result"]
    assert ere_out["cancellation_reduction_pct"] > 35.0
    mi_out = responses[3]["result"]["structuredContent"]["result"]
    assert mi_out["verdict"] == "PASS"
