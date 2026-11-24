"""End-to-end physics verification, public CT-MR Brain benchmark, Sobol' sensitivity,
and closed-loop MR sequence co-design benchmark suite for mrphys-guard.

Writes structured JSON ledgers and Markdown clinical dossiers into `results/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mrphys_guard.dicom_io import (
    build_ct_dicom_dataset,
    build_mr_dicom_dataset,
    load_nrrd_volume,
    read_dicom_physics,
    save_dicom,
)
from mrphys_guard.distortion_solver import (
    CHI_AIR_SI,
    CHI_WATER_SI,
    build_pelvic_prostate_susceptibility_phantom,
    compute_3d_distortion_decomposition,
    fit_spherical_harmonic_gnl,
    generate_synthetic_grid_phantom,
    solve_susceptibility_b0_map,
)
from mrphys_guard.harness.agent_orchestrator import (
    MRSequenceCoDesignAgent,
    build_pacemaker_cardiac_brain_scenario,
    build_prostate_srs_hip_gas_scenario,
)
from mrphys_guard.implant_safety_csp import lookup_implant_profile, verify_implant_safety_csp
from mrphys_guard.ir import MRSequenceSpec, ScannerFieldSpec
from mrphys_guard.provenance import ProvenanceLedger
from mrphys_guard.sct_ere_auditor import (
    audit_synthetic_ct,
    compute_ere_boundary_risk,
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

    # Derived perturbation audit on real clinical CT (tagged DERIVED_PERTURBATION)
    # Build an MRI-like intensity proxy from real CT soft tissue / bone / air contrast on the exact CT grid
    # (CT-brain and MR-brain-T1 have different native voxel grids; report both native shapes explicitly)
    ct_body = ct_ds > -450.0
    ct_bone = ct_ds > 350.0
    mri_proxy = np.where(ct_body, 0.62, 0.01)
    mri_proxy[ct_bone] = 0.08
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
        mri_volume=mri_proxy,
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

    summary = {
        "research_only_notice": "RESEARCH SOFTWARE — NOT A MEDICAL DEVICE.",
        "analytical_lorentz_sphere": sphere_metrics,
        "spherical_harmonic_gnl": {
            "lmax_3_rms_mm": round(gnl_l3.fit_residual_rms_mm, 5),
            "lmax_5_rms_mm": round(gnl_l5.fit_residual_rms_mm, 5),
            "lmax_5_max_mm": round(gnl_l5.fit_residual_max_mm, 5),
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
