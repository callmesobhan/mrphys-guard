"""End-to-end physics verification, public CT-MR Brain benchmark, Sobol' sensitivity,
and closed-loop MR sequence co-design benchmark suite for mrphys-guard.

Writes structured JSON ledgers and Markdown clinical dossiers into `results/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mrphys_guard.dicom_io import (
    LoadedMedicalVolume,
    build_ct_dicom_dataset,
    build_mr_dicom_dataset,
    build_rtdose_dicom_dataset,
    build_rtstruct_dicom_dataset,
    load_nrrd_volume,
    read_dicom_physics,
    read_rtdose_dicom,
    read_rtstruct_dicom,
    resample_volume_to_reference,
    save_dicom,
)
from mrphys_guard.distortion_solver import (
    CHI_AIR_SI,
    CHI_WATER_SI,
    build_pelvic_prostate_susceptibility_phantom,
    compute_3d_distortion_decomposition,
    compute_distortion_jacobian_and_unwarp,
    fit_spherical_harmonic_gnl,
    generate_synthetic_grid_phantom,
    solve_susceptibility_b0_map,
    unwrap_3d_b0_phase_dct,
)
from mrphys_guard.dvh_margin_certifier import (
    build_clinical_srs_dose_grid,
    build_rectal_wall_oar_mask,
    certify_mronly_dvh_and_anisotropic_margins,
)
from mrphys_guard.harness.agent_orchestrator import (
    MRSequenceCoDesignAgent,
    build_pacemaker_cardiac_brain_scenario,
    build_prostate_srs_hip_gas_scenario,
    generate_pareto_protocol_candidates,
)
from mrphys_guard.implant_safety_csp import (
    lookup_implant_profile,
    verify_implant_safety_csp,
    verify_multi_implant_safety_csp,
)
from mrphys_guard.ir import MRSequenceSpec, ScannerFieldSpec
from mrphys_guard.provenance import ProvenanceLedger
from mrphys_guard.sct_ere_auditor import (
    audit_synthetic_ct,
    compute_ere_boundary_risk,
    compute_multibeam_ere_cancellation,
    inject_sct_failure_modes,
)
from mrphys_guard.sobol_sensitivity import run_mr_uncertainty_sobol_suite

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PUBLIC_DATA_DIR = REPO_ROOT / "data" / "public_ctmr_brain"


def run_analytical_sphere_benchmark() -> dict[str, float]:
    """Verify exact Lorentz-sphere analytical identity (interior Delta_B0 == 0, exterior 1/r^3)."""
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

    interior = r <= (radius - 2.5)
    interior_mean_ppm = float(np.mean(db0[interior] / b0 * 1e6))
    interior_rms_ppm = float(np.sqrt(np.mean((db0[interior] / b0 * 1e6) ** 2)))

    # Exterior axial comparison along +z axis (theta = 0 -> 3*cos^2(0) - 1 = 2)
    ext_mask = (r >= radius + 4.0) & (r <= radius + 14.0)
    cos_theta = np.divide(gz, r, out=np.zeros_like(r), where=r > 0)
    analytic_ext = (1.0 / 3.0) * b0 * chi0 * ((radius / np.maximum(r, 1e-6)) ** 3) * (3.0 * cos_theta**2 - 1.0)
    rel_l2_ext = float(
        np.linalg.norm(db0[ext_mask] - analytic_ext[ext_mask]) / np.linalg.norm(analytic_ext[ext_mask])
    )
    return {
        "sphere_radius_mm": radius,
        "interior_mean_ppm": round(interior_mean_ppm, 6),
        "interior_rms_ppm": round(interior_rms_ppm, 6),
        "exterior_relative_l2_error": round(rel_l2_ext, 5),
    }


def run_public_ctmr_brain_benchmark() -> dict[str, object]:
    """Run real-volume ERE & derived-perturbation sCT audit on 3D Slicer CT-MR Brain."""
    ct_path = PUBLIC_DATA_DIR / "CT-brain.nrrd"
    mr_path = PUBLIC_DATA_DIR / "MR-brain-T1.nrrd"
    if not ct_path.exists() or not mr_path.exists():
        return {"available": False, "note": "Run scripts/fetch_public_mri_sample.py first."}

    ct_vol = load_nrrd_volume(ct_path, modality="CT")
    mr_vol = load_nrrd_volume(mr_path, modality="MR")

    # Downsample CT cleanly by stride (2, 2, 2) for fast 3D ERE benchmarking while preserving real anatomy
    ct_ds = ct_vol.array[::2, ::2, ::2]
    vox_ds = tuple(v * 2.0 for v in ct_vol.voxel_size_mm)

    ere_by_b0 = {}
    for b0 in (0.0, 0.35, 1.5, 3.0):
        risk, _ = compute_ere_boundary_risk(
            ct_or_sct_hu=ct_ds,
            voxel_size_mm=vox_ds,
            b0_T=b0,
            beam_energy_mv=6.0,
        )
        ere_by_b0[f"{b0:.2f}T"] = risk.model_dump(mode="json")

    # Co-register real clinical MR-brain-T1.nrrd directly onto downsampled CT-brain.nrrd physical grid!
    assert ct_vol.affine_4x4 is not None
    ct_ds_vol = LoadedMedicalVolume(
        array=ct_ds,
        voxel_size_mm=vox_ds,
        modality="CT",
        source_path=ct_vol.source_path,
        metadata=ct_vol.metadata,
        affine_4x4=ct_vol.affine_4x4 @ np.diag([2.0, 2.0, 2.0, 1.0]),
    )
    mr_on_ct = resample_volume_to_reference(mr_vol, ct_ds_vol, order=1)

    ct_body = ct_ds > -450.0
    ct_bone = ct_ds > 350.0
    sct_perturbed = inject_sct_failure_modes(
        ct_hu=ct_ds,
        bone_mask=ct_bone,
        metal_mask=np.zeros_like(ct_bone),
        body_mask=ct_body,
        swap_bone_fraction=0.12,
        drop_metal_to_tissue=False,
        truncate_body_border_voxels=2,
        hu_noise_sigma=15.0,
    )
    sct_report, _ = audit_synthetic_ct(
        mri_volume=mr_on_ct.array,
        sct_hu=sct_perturbed,
        voxel_size_mm=vox_ds,
        ground_truth_ct_hu=ct_ds,
        body_mask=ct_body,
    )

    return {
        "available": True,
        "provenance_tag_raw": "PUBLIC_OPEN_CLINICAL",
        "provenance_tag_sct_perturbation": "DERIVED_PERTURBATION",
        "ct_native_shape": list(ct_vol.array.shape),
        "ct_native_voxel_mm": [round(v, 4) for v in ct_vol.voxel_size_mm],
        "mr_t1_native_shape": list(mr_vol.array.shape),
        "mr_t1_native_voxel_mm": [round(v, 4) for v in mr_vol.voxel_size_mm],
        "ere_by_field_strength": ere_by_b0,
        "derived_sct_audit": sct_report.model_dump(mode="json"),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ledger = ProvenanceLedger()

    print("1/6 Running analytical Lorentz-sphere dipole benchmark...")
    sphere_metrics = run_analytical_sphere_benchmark()
    ledger.append("benchmark_lorentz_sphere", {"radius_mm": 12.0}, sphere_metrics)

    print("2/6 Running 3D pelvic prostate distortion + spherical harmonics benchmark...")
    true_pts, meas_pts, _ = generate_synthetic_grid_phantom(
        extent_mm=180.0, spacing_mm=30.0, max_edge_distortion_mm=2.8, noise_sigma_mm=0.02
    )
    gnl_l3 = fit_spherical_harmonic_gnl(true_pts, meas_pts, lmax=3)
    gnl_l5 = fit_spherical_harmonic_gnl(true_pts, meas_pts, lmax=5)

    phantom = build_pelvic_prostate_susceptibility_phantom(
        shape=(64, 64, 48), voxel_size_mm=(2.5, 2.5, 2.5)
    )
    seq_low_bw = MRSequenceSpec(
        name="Pelvis_LowBW_Baseline",
        b0_T=1.5,
        bandwidth_hz_per_px=185.0,
        voxel_size_mm=(2.5, 2.5, 2.5),
        tr_ms=15.0,
        te_ms=4.5,
    )
    seq_high_bw = seq_low_bw.model_copy(
        update={"name": "Pelvis_HighBW_CoDesigned", "bandwidth_hz_per_px": 560.0}
    )
    decomp_low, _ = compute_3d_distortion_decomposition(
        chi_si=phantom["chi_si"],
        sequence=seq_low_bw,
        gnl_model=gnl_l5,
        target_roi_mask=phantom["prostate_mask"],
        body_mask=phantom["body_mask"],
    )
    decomp_high, _ = compute_3d_distortion_decomposition(
        chi_si=phantom["chi_si"],
        sequence=seq_high_bw,
        gnl_model=gnl_l5,
        target_roi_mask=phantom["prostate_mask"],
        body_mask=phantom["body_mask"],
    )
    ledger.append("benchmark_3d_distortion", {"bw_low": 185.0, "bw_high": 560.0}, {
        "low_bw": decomp_low.model_dump(mode="json"),
        "high_bw": decomp_high.model_dump(mode="json"),
    })

    print("3/6 Running pelvic sCT plausibility audit & 0T/0.35T/1.5T/3.0T ERE sweep...")
    sct_pelvis = inject_sct_failure_modes(
        ct_hu=phantom["ct_hu"],
        bone_mask=phantom["bone_mask"],
        metal_mask=phantom["metal_mask"],
        body_mask=phantom["body_mask"],
        swap_bone_fraction=0.28,
        drop_metal_to_tissue=True,
        truncate_body_border_voxels=2,
    )
    sct_pelvis_report, _ = audit_synthetic_ct(
        mri_volume=phantom["mri_t1"],
        sct_hu=sct_pelvis,
        voxel_size_mm=(2.5, 2.5, 2.5),
        ground_truth_ct_hu=phantom["ct_hu"],
        body_mask=phantom["body_mask"],
    )
    pelvis_ere = {}
    for b0 in (0.0, 0.35, 1.5, 3.0):
        er, _ = compute_ere_boundary_risk(
            ct_or_sct_hu=phantom["ct_hu"],
            voxel_size_mm=(2.5, 2.5, 2.5),
            b0_T=b0,
            body_mask=phantom["body_mask"],
        )
        pelvis_ere[f"{b0:.2f}T"] = er.model_dump(mode="json")

    print("4/6 Running real open-license 3D Slicer CT-MR Brain benchmark...")
    public_ctmr = run_public_ctmr_brain_benchmark()
    ledger.append(
        "benchmark_public_ctmr_brain",
        {"dataset": "slicer_ctmr_brain_paired"},
        public_ctmr,
        data_origin="PUBLIC_OPEN_CLINICAL",
    )

    print("5/6 Running closed-loop MR Sequence Co-Design Agent & ABSTAIN verification...")
    agent = MRSequenceCoDesignAgent()
    res_prostate = agent.run(build_prostate_srs_hip_gas_scenario())
    res_pacemaker = agent.run(build_pacemaker_cardiac_brain_scenario())

    # Verify MR_UNSAFE abstention
    unsafe_verdict = verify_implant_safety_csp(
        ScannerFieldSpec(),
        build_pacemaker_cardiac_brain_scenario().initial_sequences,
        lookup_implant_profile("ferromagnetic_aneurysm_clip_unsafe"),
    )

    (RESULTS_DIR / "dossier_prostate_srs_hip_gas.md").write_text(
        res_prostate.dossier_markdown, encoding="utf-8"
    )
    (RESULTS_DIR / "dossier_pacemaker_cardiac_brain.md").write_text(
        res_pacemaker.dossier_markdown, encoding="utf-8"
    )

    print("6/6 Running Saltelli-Jansen QMC Sobol' global sensitivity (N=1024) + DICOM round-trip...")
    sobol_reports = run_mr_uncertainty_sobol_suite(n_base=1024, b0_T=1.5, seed=2026)

    # Lossless MR & CT DICOM round-trip check
    dcm_dir = RESULTS_DIR / "dicom_roundtrip"
    mid_z = phantom["mri_t1"].shape[2] // 2
    mr_ds = build_mr_dicom_dataset(phantom["mri_t1"][:, :, mid_z], seq_high_bw)
    ct_ds = build_ct_dicom_dataset(phantom["ct_hu"][:, :, mid_z], voxel_size_mm=(2.5, 2.5, 2.5))
    mr_dcm_path = save_dicom(mr_ds, dcm_dir / "synthetic_pelvis_mr.dcm")
    ct_dcm_path = save_dicom(ct_ds, dcm_dir / "synthetic_pelvis_sct.dcm")
    mr_rt, mr_hdr = read_dicom_physics(mr_dcm_path)
    ct_rt, ct_hdr = read_dicom_physics(ct_dcm_path)
    dicom_metrics = {
        "mr_max_abs_err": float(np.max(np.abs(mr_rt - phantom["mri_t1"][:, :, mid_z]))),
        "ct_max_abs_err_hu": float(np.max(np.abs(ct_rt - phantom["ct_hu"][:, :, mid_z]))),
        "mr_bandwidth_hz_per_px": mr_hdr["PixelBandwidth"],
        "mr_b0_T": mr_hdr["MagneticFieldStrength"],
        "ct_modality": ct_hdr["Modality"],
    }

    # v0.2.0 Advanced Physics Benchmarks: Multi-beam ERE, DCT Phase Unwrap, Jacobian, Multi-Implant & Pareto
    mb_ere_7, _ = compute_multibeam_ere_cancellation(
        ct_or_sct_hu=phantom["ct_hu"],
        voxel_size_mm=(2.5, 2.5, 2.5),
        gantry_angles_deg=[0.0, 51.0, 103.0, 154.0, 206.0, 257.0, 309.0],
        b0_T=1.5,
        body_mask=phantom["body_mask"],
    )
    # 3D DCT phase unwrap benchmark
    ux, uy, uz = np.meshgrid(
        np.linspace(-1, 1, 24), np.linspace(-1, 1, 24), np.linspace(-1, 1, 16), indexing="ij"
    )
    true_phi = 8.5 * np.exp(-(ux**2 + uy**2 + uz**2) / 0.45) + 4.5 * ux - 3.0 * uy
    wrapped_phi = np.arctan2(np.sin(true_phi), np.cos(true_phi))
    unwrap_rep, unwrap_fields = unwrap_3d_b0_phase_dct(wrapped_phi, delta_te_ms=2.4, b0_T=1.5)
    unwrap_diff = unwrap_fields["unwrapped_phase_rad"] - true_phi
    unwrap_max_err = float(np.max(np.abs(unwrap_diff - np.median(unwrap_diff))))

    # Distortion Jacobian singularity check at low BW
    _, fields_lo = compute_3d_distortion_decomposition(
        chi_si=phantom["chi_si"],
        sequence=seq_low_bw,
        gnl_model=gnl_l5,
        target_roi_mask=phantom["prostate_mask"],
        body_mask=phantom["body_mask"],
    )
    jac_rep, _ = compute_distortion_jacobian_and_unwarp(
        phantom["mri_t1"], fields_lo["total_disp_vec_mm"], (2.5, 2.5, 2.5), body_mask=phantom["body_mask"]
    )

    # Multi-implant simultaneous CSP (pacemaker + Ti hip prosthesis)
    mi_verdict = verify_multi_implant_safety_csp(
        ScannerFieldSpec(b0_T=1.5, max_spatial_gradient_T_per_m=17.0),
        res_pacemaker.optimized_sequences,
        [
            lookup_implant_profile("generic_cardiac_pacemaker_1p5t"),
            lookup_implant_profile("generic_hip_prosthesis_ti"),
        ],
    )
    pareto_cands = generate_pareto_protocol_candidates(build_prostate_srs_hip_gas_scenario())

    # v0.3.0 MR-Only Ambiguity-to-DVH & Anisotropic van Herk PTV Margin Certifier + RTDOSE/RTSTRUCT round-trip
    oar_wall = build_rectal_wall_oar_mask(
        phantom["rectal_gas_mask"], phantom["body_mask"], phantom["prostate_mask"]
    )
    dose_gy = build_clinical_srs_dose_grid(
        phantom["prostate_mask"], oar_wall, (2.5, 2.5, 2.5), prescription_gy=36.25
    )
    seq_dvh = MRSequenceSpec(
        name="Pelvis_SBRT_BW240_AP",
        b0_T=1.5,
        bandwidth_hz_per_px=240.0,
        voxel_size_mm=(2.5, 2.5, 2.5),
        readout_axis=1,
        phase_encode_axis=0,
        tr_ms=15.0,
        te_ms=4.5,
    )
    dvh_cert, _ = certify_mronly_dvh_and_anisotropic_margins(
        chi_si=phantom["chi_si"],
        mri_volume=phantom["mri_t1"],
        ct_or_sct_hu=phantom["ct_hu"],
        dose_gy=dose_gy,
        target_mask=phantom["prostate_mask"],
        oar_mask=oar_wall,
        body_mask=phantom["body_mask"],
        sequence=seq_dvh,
        gnl_model=gnl_l5,
        prescription_gy=36.25,
    )
    # Real DICOM RTDOSE & RTSTRUCT lossless round-trip check
    rtdose_ds = build_rtdose_dicom_dataset(dose_gy, (2.5, 2.5, 2.5))
    rtstruct_ds = build_rtstruct_dicom_dataset(
        {"Prostate_CTV": phantom["prostate_mask"], "Rectal_Wall": oar_wall}, (2.5, 2.5, 2.5)
    )
    rtdose_path = save_dicom(rtdose_ds, dcm_dir / "synthetic_pelvis_rtdose.dcm")
    rtstruct_path = save_dicom(rtstruct_ds, dcm_dir / "synthetic_pelvis_rtstruct.dcm")
    rtdose_rt = read_rtdose_dicom(rtdose_path)
    rtstruct_rt = read_rtstruct_dicom(rtstruct_path, phantom["prostate_mask"].shape, (2.5, 2.5, 2.5))
    ctv_inter = int(np.sum(rtstruct_rt["Prostate_CTV"] & phantom["prostate_mask"]))
    ctv_dice = 2.0 * ctv_inter / max(1, int(np.sum(rtstruct_rt["Prostate_CTV"]) + np.sum(phantom["prostate_mask"])))
    rt_dicom_metrics = {
        "rtdose_max_abs_err_gy": float(np.max(np.abs(rtdose_rt.array - dose_gy))),
        "rtstruct_ctv_dice": round(ctv_dice, 4),
        "rtstruct_rois_recovered": sorted(rtstruct_rt.keys()),
    }

    summary = {
        "research_only_notice": "RESEARCH SOFTWARE — NOT A MEDICAL DEVICE.",
        "analytical_lorentz_sphere": sphere_metrics,
        "spherical_harmonic_gnl": {
            "lmax_3_rms_mm": round(gnl_l3.fit_residual_rms_mm, 5),
            "lmax_5_rms_mm": round(gnl_l5.fit_residual_rms_mm, 5),
            "lmax_5_max_mm": round(gnl_l5.fit_residual_max_mm, 5),
        },
        "v03_mronly_dvh_margin_certifier": {
            "certificate": dvh_cert.model_dump(mode="json"),
            "dicom_rt_roundtrip": rt_dicom_metrics,
        },
        "v02_advanced_physics": {
            "multibeam_7field_ere": mb_ere_7.model_dump(mode="json"),
            "dct_phase_unwrap_max_err_rad": unwrap_max_err,
            "dct_phase_unwrap_cycles": unwrap_rep.cycle_count_range,
            "low_bw_jacobian": jac_rep.model_dump(mode="json"),
            "multi_implant_csp": {
                "verdict": mi_verdict.verdict,
                "governing_device_by_limit": mi_verdict.governing_device_by_limit,
                "effective_max_b1_rms_uT": mi_verdict.effective_max_b1_rms_uT,
            },
            "bloch_pareto_frontier": [
                {
                    "profile_name": c.profile_name,
                    "max_target_distortion_mm": c.max_target_distortion_mm,
                    "min_relative_snr": c.min_relative_snr,
                    "mean_bloch_cnr": c.mean_bloch_cnr,
                    "max_b1_rms_uT": c.max_b1_rms_uT,
                    "critic_status": c.critic_status,
                }
                for c in pareto_cands
            ],
        },
        "pelvis_3d_distortion": {
            "low_bw_185": decomp_low.model_dump(mode="json"),
            "high_bw_560": decomp_high.model_dump(mode="json"),
        },
        "pelvis_sct_audit": sct_pelvis_report.model_dump(mode="json"),
        "pelvis_ere_by_field": pelvis_ere,
        "public_ctmr_brain": public_ctmr,
        "agent_codesign": {
            "prostate_srs_hip_gas": {
                "status": res_prostate.status,
                "critic_status": res_prostate.critic.status,
                "baseline_csp": res_prostate.baseline_csp.verdict,
                "optimized_csp": res_prostate.optimized_csp.verdict if res_prostate.optimized_csp else None,
                "baseline_dist_mm": res_prostate.baseline_max_target_distortion_mm,
                "optimized_dist_mm": res_prostate.optimized_max_target_distortion_mm,
                "baseline_b1_uT": round(res_prostate.baseline_csp.max_b1_rms_uT, 3),
                "optimized_b1_uT": round(res_prostate.optimized_csp.max_b1_rms_uT, 3) if res_prostate.optimized_csp else None,
            },
            "pacemaker_cardiac_brain": {
                "status": res_pacemaker.status,
                "critic_status": res_pacemaker.critic.status,
                "baseline_csp": res_pacemaker.baseline_csp.verdict,
                "optimized_csp": res_pacemaker.optimized_csp.verdict if res_pacemaker.optimized_csp else None,
                "baseline_b1_uT": round(res_pacemaker.baseline_csp.max_b1_rms_uT, 3),
                "optimized_b1_uT": round(res_pacemaker.optimized_csp.max_b1_rms_uT, 3) if res_pacemaker.optimized_csp else None,
                "baseline_wb_sar": round(res_pacemaker.baseline_csp.max_wb_sar_W_per_kg, 3),
                "optimized_wb_sar": round(res_pacemaker.optimized_csp.max_wb_sar_W_per_kg, 3) if res_pacemaker.optimized_csp else None,
            },
            "mr_unsafe_abstention": {
                "verdict": unsafe_verdict.verdict,
                "abstention_code": unsafe_verdict.abstention_code.value if unsafe_verdict.abstention_code else None,
            },
        },
        "sobol_sensitivity": {k: v.model_dump(mode="json") for k, v in sobol_reports.items()},
        "dicom_roundtrip": dicom_metrics,
        "provenance_ledger": ledger.to_json_dict(),
    }

    out_json = RESULTS_DIR / "benchmark_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Benchmark suite complete -> {out_json}")


if __name__ == "__main__":
    main()
