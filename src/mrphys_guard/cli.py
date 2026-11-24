"""Command-line interface for mrphys-guard."""

from __future__ import annotations

import argparse
import json
import sys

from mrphys_guard.distortion_solver import build_pelvic_prostate_susceptibility_phantom
from mrphys_guard.dvh_margin_certifier import (
    build_clinical_srs_dose_grid,
    build_rectal_wall_oar_mask,
    certify_mronly_dvh_and_anisotropic_margins,
)
from mrphys_guard.harness.agent_orchestrator import (
    MRSequenceCoDesignAgent,
    build_pacemaker_cardiac_brain_scenario,
    build_prostate_srs_hip_gas_scenario,
)
from mrphys_guard.harness.mcp_server import main as run_mcp_server
from mrphys_guard.implant_safety_csp import list_implant_profiles
from mrphys_guard.ir import MRSequenceSpec
from mrphys_guard.sobol_sensitivity import run_mr_uncertainty_sobol_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mrphys-guard",
        description=(
            "mrphys-guard: 3D MRI Distortion, sCT/ERE Physics Audit & IEC 60601-2-33 "
            "Implant Safety Engine (Research Software — NOT A MEDICAL DEVICE)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-implants", help="List built-in generic MR-conditional device profiles.")

    cd = sub.add_parser("codesign", help="Run closed-loop MR sequence & safety co-design agent.")
    cd.add_argument(
        "--preset",
        choices=["prostate_srs_hip_gas", "pacemaker_cardiac_brain"],
        default="prostate_srs_hip_gas",
    )
    cd.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown dossier.")

    sb = sub.add_parser("sobol", help="Run Saltelli-Jansen QMC global Sobol' sensitivity suite.")
    sb.add_argument("--n-base", type=int, default=512)
    sb.add_argument("--b0", type=float, default=1.5)

    dv = sub.add_parser(
        "certify-dvh",
        help="Run MR-Only Ambiguity-to-DVH, Readout-Axis Optimizer & Anisotropic van Herk PTV Margin Certifier.",
    )
    dv.add_argument("--bw", type=float, default=240.0, help="Receiver bandwidth [Hz/px]")
    dv.add_argument("--b0", type=float, default=1.5, help="Static field B0 [T]")
    dv.add_argument("--rx", type=float, default=36.25, help="Prescription dose [Gy]")

    sub.add_parser("mcp", help="Start JSON-RPC 2.0 stdio MCP server.")

    args = parser.parse_args(argv)

    if args.command == "list-implants":
        profiles = [p.model_dump(mode="json") for p in list_implant_profiles()]
        print(json.dumps(profiles, indent=2))
        return 0

    if args.command == "codesign":
        scenario = (
            build_pacemaker_cardiac_brain_scenario()
            if args.preset == "pacemaker_cardiac_brain"
            else build_prostate_srs_hip_gas_scenario()
        )
        res = MRSequenceCoDesignAgent().run(scenario)
        if args.json:
            print(json.dumps(res.model_dump(mode="json"), indent=2))
        else:
            print(res.dossier_markdown)
        return 0

    if args.command == "sobol":
        reports = run_mr_uncertainty_sobol_suite(n_base=args.n_base, b0_T=args.b0)
        print(json.dumps({k: v.model_dump(mode="json") for k, v in reports.items()}, indent=2))
        return 0

    if args.command == "certify-dvh":
        vox = (4.0, 4.0, 4.0)
        phantom = build_pelvic_prostate_susceptibility_phantom(shape=(40, 40, 30), voxel_size_mm=vox)
        oar_wall = build_rectal_wall_oar_mask(
            phantom["rectal_gas_mask"], phantom["body_mask"], phantom["prostate_mask"]
        )
        dose = build_clinical_srs_dose_grid(phantom["prostate_mask"], oar_wall, vox, prescription_gy=args.rx)
        seq = MRSequenceSpec(
            name="CLI_Certify_DVH",
            b0_T=args.b0,
            bandwidth_hz_per_px=args.bw,
            voxel_size_mm=vox,
            readout_axis=1,
            phase_encode_axis=0,
            tr_ms=15.0,
            te_ms=4.5,
        )
        cert, _ = certify_mronly_dvh_and_anisotropic_margins(
            chi_si=phantom["chi_si"],
            mri_volume=phantom["mri_t1"],
            ct_or_sct_hu=phantom["ct_hu"],
            dose_gy=dose,
            target_mask=phantom["prostate_mask"],
            oar_mask=oar_wall,
            body_mask=phantom["body_mask"],
            sequence=seq,
            prescription_gy=args.rx,
        )
        print(json.dumps(cert.model_dump(mode="json"), indent=2))
        return 0

    if args.command == "mcp":
        run_mcp_server()
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
