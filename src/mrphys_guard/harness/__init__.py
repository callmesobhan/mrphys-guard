"""Frontier Agent Harness & JSON-RPC 2.0 MCP Server for mrphys-guard."""

from mrphys_guard.harness.agent_orchestrator import (
    CoDesignResult,
    CoDesignScenario,
    CriticVerdict,
    MRSequenceCoDesignAgent,
    build_pacemaker_cardiac_brain_scenario,
    build_prostate_srs_hip_gas_scenario,
)
from mrphys_guard.harness.mcp_server import MCP_TOOLS, handle_jsonrpc_request

__all__ = [
    "CoDesignResult",
    "CoDesignScenario",
    "CriticVerdict",
    "MRSequenceCoDesignAgent",
    "MCP_TOOLS",
    "build_pacemaker_cardiac_brain_scenario",
    "build_prostate_srs_hip_gas_scenario",
    "handle_jsonrpc_request",
]
