"""Frontier Agent Harness & JSON-RPC 2.0 MCP Server for mrphys-guard."""

from mrphys_guard.harness.agent_orchestrator import (
    CoDesignResult,
    CoDesignScenario,
    CriticVerdict,
    MRSequenceCoDesignAgent,
    ParetoProtocolCandidate,
    build_pacemaker_cardiac_brain_scenario,
    build_prostate_srs_hip_gas_scenario,
    compute_bloch_tissue_cnr,
    generate_pareto_protocol_candidates,
)
from mrphys_guard.harness.mcp_server import MCP_TOOLS, handle_jsonrpc_request

__all__ = [
    "CoDesignResult",
    "CoDesignScenario",
    "CriticVerdict",
    "MRSequenceCoDesignAgent",
    "ParetoProtocolCandidate",
    "MCP_TOOLS",
    "build_pacemaker_cardiac_brain_scenario",
    "build_prostate_srs_hip_gas_scenario",
    "compute_bloch_tissue_cnr",
    "generate_pareto_protocol_candidates",
    "handle_jsonrpc_request",
]
