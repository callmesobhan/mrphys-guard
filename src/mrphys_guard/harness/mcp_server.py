"""JSON-RPC 2.0 stdio Model Context Protocol (MCP) Server for mrphys-guard.

Exposes 6 typed, provenance-gated medical physics tools:
1. mrphys_solve_3d_distortion
2. mrphys_audit_sct_and_ere
3. mrphys_verify_implant_csp
4. mrphys_codesign_safe_sequence
5. mrphys_lookup_implant_profile
6. mrphys_run_sobol_sensitivity
"""

from __future__ import annotations

import json
import sys
from typing import Any

from mrphys_guard.distortion_solver import (
    build_pelvic_prostate_susceptibility_phantom,
    compute_3d_distortion_decomposition,
    fit_spherical_harmonic_gnl,
    generate_synthetic_grid_phantom,
)
from mrphys_guard.harness.agent_orchestrator import (
    CoDesignScenario,
    MRSequenceCoDesignAgent,
    build_pacemaker_cardiac_brain_scenario,
    build_prostate_srs_hip_gas_scenario,
)
from mrphys_guard.implant_safety_csp import (
    list_implant_profiles,
    lookup_implant_profile,
    verify_implant_safety_csp,
    verify_multi_implant_safety_csp,
)
from mrphys_guard.ir import ImplantConditionSpec, MRSequenceSpec, ScannerFieldSpec
from mrphys_guard.provenance import ProvenanceLedger
from mrphys_guard.sct_ere_auditor import (
    audit_synthetic_ct,
    compute_ere_boundary_risk,
    compute_multibeam_ere_cancellation,
    inject_sct_failure_modes,
)
from mrphys_guard.sobol_sensitivity import run_mr_uncertainty_sobol_suite

SERVER_INFO = {
    "name": "mrphys-guard-mcp",
    "version": "0.2.0",
    "notice": "RESEARCH SOFTWARE — NOT A MEDICAL DEVICE.",
}

MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "mrphys_solve_3d_distortion",
        "description": (
            "Solve 3D MRI gradient nonlinearity (l<=5 solid spherical harmonics) and k-space "
            "Lorentz-sphere susceptibility B0 dipole convolution on a pelvic prostate MR-only phantom."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "b0_T": {"type": "number", "default": 1.5},
                "bandwidth_hz_per_px": {"type": "number", "default": 250.0},
                "include_rectal_gas": {"type": "boolean", "default": True},
                "include_bilateral_hip_prostheses": {"type": "boolean", "default": True},
                "srs_tolerance_mm": {"type": "number", "default": 0.5},
                "grid_size": {"type": "integer", "default": 32},
            },
        },
    },
    {
        "name": "mrphys_audit_sct_and_ere",
        "description": (
            "Audit synthetic-CT (sCT) physics plausibility (bone-to-air swaps, missing metal, "
            "body-contour truncation) and compute 1.5T/0.35T Lorentz-force ERE boundary hotspot risk."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "b0_T": {"type": "number", "default": 1.5},
                "beam_energy_mv": {"type": "number", "default": 6.0},
                "swap_bone_fraction": {"type": "number", "default": 0.25},
                "drop_metal_to_tissue": {"type": "boolean", "default": True},
                "grid_size": {"type": "integer", "default": 32},
            },
        },
    },
    {
        "name": "mrphys_verify_implant_csp",
        "description": (
            "Formally verify a multi-sequence MRI protocol against IEC 60601-2-33 / ASTM F2503 "
            "implant constraints (B0, dB/dx, B1+rms, SAR, slew rate, dB/dt, active scan time)."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["implant_id", "sequences"],
            "properties": {
                "scanner": {"type": "object"},
                "implant_id": {"type": "string"},
                "implant_override": {"type": "object"},
                "sequences": {"type": "array", "items": {"type": "object"}},
            },
        },
    },
    {
        "name": "mrphys_codesign_safe_sequence",
        "description": (
            "Run the closed-loop MR sequence & safety co-design agent with independent Critic-Verifier "
            "gate on a preset or custom high-risk clinical scenario."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "preset": {
                    "type": "string",
                    "enum": ["prostate_srs_hip_gas", "pacemaker_cardiac_brain"],
                    "default": "prostate_srs_hip_gas",
                },
                "custom_scenario": {"type": "object"},
            },
        },
    },
    {
        "name": "mrphys_lookup_implant_profile",
        "description": (
            "Lookup built-in generic ASTM F2503 / IEC 60601-2-33 MR-conditional literature device "
            "profiles (pass device_id or omit to list all profiles)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "device_id": {"type": "string"},
            },
        },
    },
    {
        "name": "mrphys_run_sobol_sensitivity",
        "description": (
            "Run exact Saltelli-Jansen QMC global Sobol' sensitivity analysis (S_1, S_T) over "
            "receiver bandwidth, air-cavity susceptibility, GNL residual, and sCT HU error."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "n_base": {"type": "integer", "default": 256},
                "b0_T": {"type": "number", "default": 1.5},
                "seed": {"type": "integer", "default": 2026},
            },
        },
    },
    {
        "name": "mrphys_multibeam_ere_optimize",
        "description": (
            "Evaluate multi-beam IMRT/VMAT opposing-beam Lorentz-force ERE cancellation and "
            "recommend lowest-ERE gantry angles at 1.5T or 0.35T."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "b0_T": {"type": "number", "default": 1.5},
                "gantry_angles_deg": {
                    "type": "array",
                    "items": {"type": "number"},
                    "default": [0.0, 51.0, 103.0, 154.0, 206.0, 257.0, 309.0],
                },
                "grid_size": {"type": "integer", "default": 32},
            },
        },
    },
    {
        "name": "mrphys_verify_multi_implant_csp",
        "description": (
            "Formally verify a multi-sequence MR protocol simultaneously across multiple "
            "co-existing implants, returning intersected limits, IEC PNS %, and UNSAT conflict cores."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["implant_ids", "sequences"],
            "properties": {
                "scanner": {"type": "object"},
                "implant_ids": {"type": "array", "items": {"type": "string"}},
                "sequences": {"type": "array", "items": {"type": "object"}},
            },
        },
    },
]


def _dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a typed mrphys-guard tool and wrap output with SHA-256 provenance."""
    ledger = ProvenanceLedger()

    if name == "mrphys_solve_3d_distortion":
        n = int(arguments.get("grid_size", 32))
        vox = (160.0 / n, 160.0 / n, 120.0 / max(16, int(n * 0.75)))
        shape = (n, n, max(16, int(n * 0.75)))
        phantom = build_pelvic_prostate_susceptibility_phantom(
            shape=shape,
            voxel_size_mm=vox,
            include_rectal_gas=bool(arguments.get("include_rectal_gas", True)),
            include_bilateral_hip_prostheses=bool(
                arguments.get("include_bilateral_hip_prostheses", True)
            ),
        )
        true_pts, meas_pts, _ = generate_synthetic_grid_phantom(extent_mm=150.0, spacing_mm=50.0)
        gnl_model = fit_spherical_harmonic_gnl(true_pts, meas_pts, lmax=5)
        seq = MRSequenceSpec(
            name="MCP_Pelvis_GRE",
            b0_T=float(arguments.get("b0_T", 1.5)),
            bandwidth_hz_per_px=float(arguments.get("bandwidth_hz_per_px", 250.0)),
            voxel_size_mm=vox,
            tr_ms=15.0,
            te_ms=4.5,
        )
        decomp, _ = compute_3d_distortion_decomposition(
            chi_si=phantom["chi_si"],
            sequence=seq,
            gnl_model=gnl_model,
            target_roi_mask=phantom["prostate_mask"],
            body_mask=phantom["body_mask"],
            srs_tolerance_mm=float(arguments.get("srs_tolerance_mm", 0.5)),
        )
        result = decomp.model_dump(mode="json")

    elif name == "mrphys_audit_sct_and_ere":
        n = int(arguments.get("grid_size", 32))
        vox = (160.0 / n, 160.0 / n, 120.0 / max(16, int(n * 0.75)))
        shape = (n, n, max(16, int(n * 0.75)))
        phantom = build_pelvic_prostate_susceptibility_phantom(shape=shape, voxel_size_mm=vox)
        sct_hu = inject_sct_failure_modes(
            ct_hu=phantom["ct_hu"],
            bone_mask=phantom["bone_mask"],
            metal_mask=phantom["metal_mask"],
            body_mask=phantom["body_mask"],
            swap_bone_fraction=float(arguments.get("swap_bone_fraction", 0.25)),
            drop_metal_to_tissue=bool(arguments.get("drop_metal_to_tissue", True)),
        )
        sct_report, _ = audit_synthetic_ct(
            mri_volume=phantom["mri_t1"],
            sct_hu=sct_hu,
            voxel_size_mm=vox,
            ground_truth_ct_hu=phantom["ct_hu"],
            body_mask=phantom["body_mask"],
        )
        ere_risk, _ = compute_ere_boundary_risk(
            ct_or_sct_hu=phantom["ct_hu"],
            voxel_size_mm=vox,
            b0_T=float(arguments.get("b0_T", 1.5)),
            beam_energy_mv=float(arguments.get("beam_energy_mv", 6.0)),
            body_mask=phantom["body_mask"],
        )
        result = {
            "sct_audit": sct_report.model_dump(mode="json"),
            "ere_boundary_risk": ere_risk.model_dump(mode="json"),
        }

    elif name == "mrphys_verify_implant_csp":
        scanner = ScannerFieldSpec(**arguments.get("scanner", {}))
        if "implant_override" in arguments and arguments["implant_override"] is not None:
            implant = ImplantConditionSpec(**arguments["implant_override"])
        else:
            implant = lookup_implant_profile(str(arguments["implant_id"]))
        sequences = [MRSequenceSpec(**s) for s in arguments["sequences"]]
        verdict = verify_implant_safety_csp(scanner, sequences, implant, raise_on_abstain=False)
        result = verdict.model_dump(mode="json")

    elif name == "mrphys_codesign_safe_sequence":
        if "custom_scenario" in arguments and arguments["custom_scenario"] is not None:
            scenario = CoDesignScenario(**arguments["custom_scenario"])
        else:
            preset = str(arguments.get("preset", "prostate_srs_hip_gas"))
            if preset == "pacemaker_cardiac_brain":
                scenario = build_pacemaker_cardiac_brain_scenario()
            else:
                scenario = build_prostate_srs_hip_gas_scenario()
        agent = MRSequenceCoDesignAgent()
        res = agent.run(scenario)
        result = res.model_dump(mode="json")

    elif name == "mrphys_lookup_implant_profile":
        dev_id = arguments.get("device_id")
        if dev_id:
            result = lookup_implant_profile(str(dev_id)).model_dump(mode="json")
        else:
            result = {"profiles": [p.model_dump(mode="json") for p in list_implant_profiles()]}

    elif name == "mrphys_run_sobol_sensitivity":
        reports = run_mr_uncertainty_sobol_suite(
            n_base=int(arguments.get("n_base", 256)),
            b0_T=float(arguments.get("b0_T", 1.5)),
            seed=int(arguments.get("seed", 2026)),
        )
        result = {k: v.model_dump(mode="json") for k, v in reports.items()}

    elif name == "mrphys_multibeam_ere_optimize":
        n = int(arguments.get("grid_size", 32))
        vox = (160.0 / n, 160.0 / n, 120.0 / max(16, int(n * 0.75)))
        shape = (n, n, max(16, int(n * 0.75)))
        phantom = build_pelvic_prostate_susceptibility_phantom(shape=shape, voxel_size_mm=vox)
        angles = [float(a) for a in arguments.get("gantry_angles_deg", [0.0, 51.0, 103.0, 154.0, 206.0, 257.0, 309.0])]
        mb_report, _ = compute_multibeam_ere_cancellation(
            ct_or_sct_hu=phantom["ct_hu"],
            voxel_size_mm=vox,
            gantry_angles_deg=angles,
            b0_T=float(arguments.get("b0_T", 1.5)),
            body_mask=phantom["body_mask"],
        )
        result = mb_report.model_dump(mode="json")

    elif name == "mrphys_verify_multi_implant_csp":
        scanner = ScannerFieldSpec(**arguments.get("scanner", {}))
        implants = [lookup_implant_profile(str(i)) for i in arguments["implant_ids"]]
        sequences = [MRSequenceSpec(**s) for s in arguments["sequences"]]
        mv = verify_multi_implant_safety_csp(scanner, sequences, implants)
        result = mv.model_dump(mode="json")

    else:
        raise KeyError(f"Unknown MCP tool: {name}")

    rec = ledger.append(
        operation=f"mcp:{name}",
        inputs=arguments,
        outputs=result,
        data_origin="SYNTHETIC_PHYSICS_FIXTURE",
    )
    return {
        "research_only_notice": "RESEARCH SOFTWARE — NOT A MEDICAL DEVICE.",
        "tool": name,
        "result": result,
        "provenance": rec.model_dump(mode="json"),
    }


def handle_jsonrpc_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """Handle a single JSON-RPC 2.0 MCP request dictionary and return response dict (or None for notifications)."""
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    if method == "notifications/initialized":
        return None

    try:
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": SERVER_INFO,
                    "capabilities": {"tools": {}},
                },
            }
        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": MCP_TOOLS}}
        if method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments") or {}
            payload = _dispatch_tool(str(tool_name), dict(tool_args))
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
                    "structuredContent": payload,
                    "isError": False,
                },
            }
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32602, "message": str(exc)},
        }


def main() -> None:
    """Run JSON-RPC 2.0 MCP server over newline-delimited stdin/stdout."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            err = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()
            continue
        resp = handle_jsonrpc_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
