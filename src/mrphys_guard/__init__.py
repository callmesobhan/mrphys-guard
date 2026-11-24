"""mrphys-guard: 3D MRI Geometric Distortion, Synthetic-CT Physics Audit,
Electron-Return-Effect (ERE) & Formal IEC 60601-2-33 RF/Implant Safety Engine.

RESEARCH SOFTWARE — NOT A MEDICAL DEVICE.
"""

from __future__ import annotations

from mrphys_guard.ir import (
    AbstentionCode,
    AbstentionError,
    ConflictCore,
    DistortionDecomposition,
    EREBoundaryRisk,
    ImplantConditionSpec,
    JacobianUnwarpReport,
    MRSafetyLabel,
    MRSequenceSpec,
    MultiBeamEREReport,
    MultiImplantCSPVerdict,
    PhaseUnwrapReport,
    ProvenanceRecord,
    SafetyCSPVerdict,
    ScannerFieldSpec,
    SCTAuditReport,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "AbstentionCode",
    "AbstentionError",
    "ConflictCore",
    "DistortionDecomposition",
    "EREBoundaryRisk",
    "ImplantConditionSpec",
    "JacobianUnwarpReport",
    "MRSafetyLabel",
    "MRSequenceSpec",
    "MultiBeamEREReport",
    "MultiImplantCSPVerdict",
    "PhaseUnwrapReport",
    "ProvenanceRecord",
    "SafetyCSPVerdict",
    "ScannerFieldSpec",
    "SCTAuditReport",
]
