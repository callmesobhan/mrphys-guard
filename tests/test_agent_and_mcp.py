"""End-to-end tests for Closed-Loop Co-Design Agent, Critic-Verifier Gate, Provenance Ledger,
CLI, and JSON-RPC 2.0 MCP Server (all 6 tools).
"""

from __future__ import annotations

import json

import pytest

from mrphys_guard.cli import main as cli_main
from mrphys_guard.distortion_solver import (
    build_pelvic_prostate_susceptibility_phantom,
    fit_spherical_harmonic_gnl,
    generate_synthetic_grid_phantom,
)
from mrphys_guard.harness.agent_orchestrator import (
    MRSequenceCoDesignAgent,
    build_pacemaker_cardiac_brain_scenario,
    build_prostate_srs_hip_gas_scenario,
)
from mrphys_guard.harness.mcp_server import MCP_TOOLS, handle_jsonrpc_request
from mrphys_guard.implant_safety_csp import lookup_implant_profile
from mrphys_guard.ir import AbstentionCode, AbstentionError, ImplantConditionSpec, MRSafetyLabel
from mrphys_guard.provenance import ProvenanceLedger, gate_srs_distortion_tolerance


def test_agent_codesign_prostate_srs_converges_and_critic_approves() -> None:
    """Prostate SRS with bilateral hip prostheses & rectal gas transitions VIOLATION -> CONVERGED/APPROVED."""
    scenario = build_prostate_srs_hip_gas_scenario()
    res = MRSequenceCoDesignAgent().run(scenario)

    assert res.baseline_csp.verdict == "VIOLATION"
    assert res.baseline_max_target_distortion_mm > 0.5
    assert res.status == "CONVERGED"
    assert res.critic.status == "APPROVED"
    assert res.optimized_csp is not None and res.optimized_csp.verdict == "PASS"
    assert res.optimized_max_target_distortion_mm is not None
    assert res.optimized_max_target_distortion_mm <= scenario.srs_tolerance_mm
    assert res.optimized_min_snr is not None and res.optimized_min_snr >= scenario.snr_floor
    assert res.provenance_ledger["chain_valid"] is True
    assert "AAPM TG-284" in res.dossier_markdown


def test_agent_codesign_pacemaker_protocol_converges_and_critic_approves() -> None:
    """1.5T cardiac + brain pacemaker protocol transitions VIOLATION -> PASS with B1+rms <= 2.0 uT."""
    scenario = build_pacemaker_cardiac_brain_scenario()
    res = MRSequenceCoDesignAgent().run(scenario)

    assert res.baseline_csp.verdict == "VIOLATION"
    assert res.status == "CONVERGED"
    assert res.critic.status == "APPROVED"
    assert res.optimized_csp is not None and res.optimized_csp.verdict == "PASS"
    assert res.optimized_csp.max_b1_rms_uT <= 2.0
    assert res.optimized_csp.max_wb_sar_W_per_kg <= 2.0
    assert len(res.trace) >= 2


def test_agent_abstains_on_mr_unsafe_and_unspecified_implant() -> None:
    """Co-design agent must hard ABSTAIN when implant is MR_UNSAFE or UNSPECIFIED."""
    base = build_pacemaker_cardiac_brain_scenario()
    unsafe_scenario = base.model_copy(
        update={"implant": lookup_implant_profile("ferromagnetic_aneurysm_clip_unsafe")}
    )
    res_unsafe = MRSequenceCoDesignAgent().run(unsafe_scenario)
    assert res_unsafe.status == "ABSTAIN"
    assert res_unsafe.abstention_code == AbstentionCode.MR_UNSAFE_DEVICE_DETECTED
    assert res_unsafe.critic.status == "ABSTAINED"
    assert res_unsafe.optimized_sequences == []

    unk_scenario = base.model_copy(
        update={
            "implant": ImplantConditionSpec(
                device_id="unk_clip",
                device_name="Unknown Surgical Clip",
                category="clip",
                mr_status=MRSafetyLabel.UNSPECIFIED,
            )
        }
    )
    res_unk = MRSequenceCoDesignAgent().run(unk_scenario)
    assert res_unk.status == "ABSTAIN"
    assert res_unk.abstention_code == AbstentionCode.UNSPECIFIED_IMPLANT_CONDITION


def test_agent_accepts_3d_susceptibility_volume_and_srs_gate() -> None:
    """Agent can ingest a 3D susceptibility volume + GNL model; SRS gate raises on excess distortion."""
    phantom = build_pelvic_prostate_susceptibility_phantom(
        shape=(24, 24, 20), voxel_size_mm=(5.0, 5.0, 5.0), include_bilateral_hip_prostheses=False
    )
    true_pts, meas_pts, _ = generate_synthetic_grid_phantom(extent_mm=150.0, spacing_mm=50.0)
    gnl = fit_spherical_harmonic_gnl(true_pts, meas_pts, lmax=5)
    scenario = build_prostate_srs_hip_gas_scenario().model_copy(
        update={"srs_tolerance_mm": 1.2, "snr_floor": 0.45}
    )
    res = MRSequenceCoDesignAgent().run(
        scenario,
        chi_si=phantom["chi_si"],
        gnl_model=gnl,
        target_roi_mask=phantom["prostate_mask"],
        body_mask=phantom["body_mask"],
    )
    assert res.provenance_ledger["record_count"] >= 3

    with pytest.raises(AbstentionError) as exc:
        gate_srs_distortion_tolerance(0.85, srs_tolerance_mm=0.50)
    assert exc.value.code == AbstentionCode.SRS_DISTORTION_EXCEEDS_TOLERANCE


def test_provenance_ledger_tamper_detection() -> None:
    """SHA-256 hash chain validates intact logs and detects tampered hashes."""
    ledger = ProvenanceLedger()
    ledger.append("step_1", {"a": 1}, {"b": 2})
    ledger.append("step_2", {"b": 2}, {"c": 3})
    assert ledger.verify_chain() is True

    # Tamper with record 0 output hash
    ledger._records[0].output_sha256 = "0" * 64
    assert ledger.verify_chain() is False


def test_mcp_jsonrpc_lifecycle_and_all_six_tools() -> None:
    """Verify MCP initialize, ping, tools/list, and every registered tool over JSON-RPC 2.0."""
    init_resp = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_resp is not None and init_resp["result"]["serverInfo"]["name"] == "mrphys-guard-mcp"

    assert handle_jsonrpc_request({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert handle_jsonrpc_request({"jsonrpc": "2.0", "id": 2, "method": "ping"}) == {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {},
    }

    list_resp = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
    assert list_resp is not None
    tools = list_resp["result"]["tools"]
    assert len(tools) == len(MCP_TOOLS) == 8
    tool_names = {t["name"] for t in tools}
    assert tool_names == {
        "mrphys_solve_3d_distortion",
        "mrphys_audit_sct_and_ere",
        "mrphys_verify_implant_csp",
        "mrphys_codesign_safe_sequence",
        "mrphys_lookup_implant_profile",
        "mrphys_run_sobol_sensitivity",
        "mrphys_multibeam_ere_optimize",
        "mrphys_verify_multi_implant_csp",
    }

    # 1. Lookup implant profile
    r1 = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {"name": "mrphys_lookup_implant_profile", "arguments": {"device_id": "generic_dbs_lead_1p5t"}},
        }
    )
    assert r1 is not None and r1["result"]["structuredContent"]["result"]["max_b1_rms_uT"] == 1.1

    # 2. Solve 3D distortion
    r2 = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {"name": "mrphys_solve_3d_distortion", "arguments": {"grid_size": 24, "bandwidth_hz_per_px": 400.0}},
        }
    )
    assert r2 is not None and r2["result"]["structuredContent"]["result"]["bandwidth_hz_per_px"] == 400.0

    # 3. Audit sCT and ERE
    r3 = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {"name": "mrphys_audit_sct_and_ere", "arguments": {"grid_size": 24, "b0_T": 1.5}},
        }
    )
    assert r3 is not None
    sc3 = r3["result"]["structuredContent"]["result"]
    assert sc3["sct_audit"]["overall_status"] == "FAIL"
    assert sc3["ere_boundary_risk"]["peak_dose_enhancement_pct"] > 10.0

    # 4. Verify implant CSP
    r4 = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 13,
            "method": "tools/call",
            "params": {
                "name": "mrphys_verify_implant_csp",
                "arguments": {
                    "implant_id": "ferromagnetic_aneurysm_clip_unsafe",
                    "sequences": [{"name": "GRE", "bandwidth_hz_per_px": 300.0, "tr_ms": 15.0, "te_ms": 4.0}],
                },
            },
        }
    )
    assert r4 is not None and r4["result"]["structuredContent"]["result"]["verdict"] == "ABSTAIN"

    # 5. Co-design safe sequence
    r5 = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 14,
            "method": "tools/call",
            "params": {"name": "mrphys_codesign_safe_sequence", "arguments": {"preset": "pacemaker_cardiac_brain"}},
        }
    )
    assert r5 is not None and r5["result"]["structuredContent"]["result"]["status"] == "CONVERGED"

    # 6. Run Sobol sensitivity
    r6 = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 15,
            "method": "tools/call",
            "params": {"name": "mrphys_run_sobol_sensitivity", "arguments": {"n_base": 64}},
        }
    )
    assert r6 is not None and "geometric_error" in r6["result"]["structuredContent"]["result"]

    # 7. Multi-beam ERE optimization
    r7 = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 16,
            "method": "tools/call",
            "params": {"name": "mrphys_multibeam_ere_optimize", "arguments": {"grid_size": 24, "b0_T": 1.5}},
        }
    )
    assert r7 is not None and r7["result"]["structuredContent"]["result"]["cancellation_reduction_pct"] > 35.0

    # 8. Simultaneous multi-implant CSP
    r8 = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 17,
            "method": "tools/call",
            "params": {
                "name": "mrphys_verify_multi_implant_csp",
                "arguments": {
                    "implant_ids": ["generic_cardiac_pacemaker_1p5t", "generic_hip_prosthesis_ti"],
                    "sequences": [{"name": "GRE", "bandwidth_hz_per_px": 450.0, "flip_angle_deg": 20.0, "tr_ms": 14.0, "te_ms": 4.0}],
                },
            },
        }
    )
    assert r8 is not None and r8["result"]["structuredContent"]["result"]["verdict"] == "PASS"

    # Unknown method returns standard JSON-RPC -32601 error
    err_resp = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 99, "method": "unknown/method"})
    assert err_resp is not None and err_resp["error"]["code"] == -32601


def test_cli_entrypoints(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI list-implants, codesign, and sobol subcommands exit 0 cleanly."""
    assert cli_main(["list-implants"]) == 0
    out1 = json.loads(capsys.readouterr().out)
    assert len(out1) >= 6

    assert cli_main(["codesign", "--preset", "prostate_srs_hip_gas", "--json"]) == 0
    out2 = json.loads(capsys.readouterr().out)
    assert out2["status"] == "CONVERGED"

    assert cli_main(["sobol", "--n-base", "64"]) == 0
    out3 = json.loads(capsys.readouterr().out)
    assert "dosimetric_uncertainty" in out3
