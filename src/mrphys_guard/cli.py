"""Command-line interface for mrphys-guard."""

from __future__ import annotations

import argparse
import json
import sys

from mrphys_guard.harness.agent_orchestrator import (
    MRSequenceCoDesignAgent,
    build_pacemaker_cardiac_brain_scenario,
    build_prostate_srs_hip_gas_scenario,
)
from mrphys_guard.harness.mcp_server import main as run_mcp_server
from mrphys_guard.implant_safety_csp import list_implant_profiles
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

    if args.command == "mcp":
        run_mcp_server()
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
