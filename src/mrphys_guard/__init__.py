"""mrphys-guard: 3D MRI Geometric Distortion, Synthetic-CT Physics Audit,
Electron-Return-Effect (ERE) & Formal IEC 60601-2-33 RF/Implant Safety Engine.

RESEARCH SOFTWARE — NOT A MEDICAL DEVICE.
"""

from __future__ import annotations

from mrphys_guard.ir import (
    AbstentionCode,
    AbstentionError,
    ConflictCore,
    DarkMRIAmbiguityCertificate,
    DistortionDecomposition,
    EREBoundaryRisk,
    ImplantConditionSpec,
    JacobianUnwarpReport,
    MROnlyDVHMarginCertificate,
    MRSafetyLabel,
    MRSequenceSpec,
    MultiBeamEREReport,
    MultiImplantCSPVerdict,
    OrganDVHImpact,
    PhaseUnwrapReport,
    ProvenanceRecord,
    ReadoutAxisCandidate,
    SafetyCSPVerdict,
    ScannerFieldSpec,
    SCTAuditReport,
)

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "AbstentionCode",
    "AbstentionError",
    "ConflictCore",
    "DarkMRIAmbiguityCertificate",
    "DistortionDecomposition",
    "EREBoundaryRisk",
    "ImplantConditionSpec",
    "JacobianUnwarpReport",
    "MRSafetyLabel",
    "MRSequenceSpec",
    "MROnlyDVHMarginCertificate",
    "MultiBeamEREReport",
    "MultiImplantCSPVerdict",
    "OrganDVHImpact",
    "PhaseUnwrapReport",
    "ProvenanceRecord",
    "ReadoutAxisCandidate",
    "SafetyCSPVerdict",
    "ScannerFieldSpec",
    "SCTAuditReport",
]
