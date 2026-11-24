"""SHA-256 cryptographic provenance ledger and hard physics/safety abstention gates."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Literal

import numpy as np

from mrphys_guard.ir import (
    AbstentionCode,
    AbstentionError,
    ImplantConditionSpec,
    MRSafetyLabel,
    ProvenanceRecord,
)


def _canonical_json(obj: Any) -> str:
    """Serialize arbitrary JSON-compatible / Pydantic / NumPy payloads deterministically."""

    def _default(item: Any) -> Any:
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json")
        if isinstance(item, np.ndarray):
            return {
                "__ndarray__": True,
                "shape": list(item.shape),
                "dtype": str(item.dtype),
                "sha256": hashlib.sha256(np.ascontiguousarray(item).tobytes()).hexdigest(),
            }
        if isinstance(item, (np.floating, np.integer)):
            return item.item()
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def sha256_payload(payload: Any) -> str:
    """Compute canonical SHA-256 hex digest of any structured payload or NumPy array."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    """Compute SHA-256 hex digest of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class ProvenanceLedger:
    """Append-only hash-chained SHA-256 provenance ledger."""

    def __init__(self, engine_version: str = "0.1.0") -> None:
        self.engine_version = engine_version
        self._records: list[ProvenanceRecord] = []

    @property
    def records(self) -> list[ProvenanceRecord]:
        return list(self._records)

    def append(
        self,
        operation: str,
        inputs: Any,
        outputs: Any,
        data_origin: Literal[
            "PUBLIC_OPEN_CLINICAL",
            "SYNTHETIC_PHYSICS_FIXTURE",
            "DERIVED_PERTURBATION",
        ] = "SYNTHETIC_PHYSICS_FIXTURE",
        metadata: dict[str, Any] | None = None,
    ) -> ProvenanceRecord:
        in_hash = sha256_payload(inputs)
        out_hash = sha256_payload(outputs)
        prev_hash = self._records[-1].record_sha256 if self._records else None
        record_id = f"mrpg-{len(self._records) + 1:04d}-{uuid.uuid4().hex[:8]}"
        body = {
            "record_id": record_id,
            "engine_version": self.engine_version,
            "operation": operation,
            "data_origin": data_origin,
            "input_sha256": in_hash,
            "output_sha256": out_hash,
            "previous_record_sha256": prev_hash,
            "metadata": metadata or {},
        }
        rec_hash = sha256_payload(body)
        record = ProvenanceRecord(
            record_id=record_id,
            engine_version=self.engine_version,
            operation=operation,
            data_origin=data_origin,
            input_sha256=in_hash,
            output_sha256=out_hash,
            previous_record_sha256=prev_hash,
            record_sha256=rec_hash,
            metadata=metadata or {},
        )
        self._records.append(record)
        return record

    def verify_chain(self) -> bool:
        """Verify hash-chain integrity across all appended provenance records."""
        prev: str | None = None
        for rec in self._records:
            if rec.previous_record_sha256 != prev:
                return False
            body = {
                "record_id": rec.record_id,
                "engine_version": rec.engine_version,
                "operation": rec.operation,
                "data_origin": rec.data_origin,
                "input_sha256": rec.input_sha256,
                "output_sha256": rec.output_sha256,
                "previous_record_sha256": rec.previous_record_sha256,
                "metadata": rec.metadata,
            }
            if sha256_payload(body) != rec.record_sha256:
                return False
            prev = rec.record_sha256
        return True

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "engine_version": self.engine_version,
            "chain_valid": self.verify_chain(),
            "record_count": len(self._records),
            "records": [r.model_dump(mode="json") for r in self._records],
        }


# ---------------------------------------------------------------------------
# Hard Abstention Gates
# ---------------------------------------------------------------------------


def gate_implant_labeling(implant: ImplantConditionSpec) -> None:
    """Enforce ASTM F2503 hard gates before running CSP evaluation."""
    if implant.mr_status == MRSafetyLabel.MR_UNSAFE:
        raise AbstentionError(
            code=AbstentionCode.MR_UNSAFE_DEVICE_DETECTED,
            message=(
                f"Device '{implant.device_name}' ({implant.device_id}) is classified "
                f"MR_UNSAFE under ASTM F2503. MRI scanning is strictly contraindicated."
            ),
            details={"device_id": implant.device_id, "mr_status": implant.mr_status.value},
        )
    if implant.mr_status == MRSafetyLabel.UNSPECIFIED:
        raise AbstentionError(
            code=AbstentionCode.UNSPECIFIED_IMPLANT_CONDITION,
            message=(
                f"Device '{implant.device_name}' ({implant.device_id}) has UNSPECIFIED "
                "MR safety labeling. Cannot clear protocol without verified ASTM F2503 conditions."
            ),
            details={"device_id": implant.device_id},
        )
    if implant.mr_status == MRSafetyLabel.MR_CONDITIONAL:
        missing: list[str] = []
        if not implant.allowed_b0_T:
            missing.append("allowed_b0_T")
        if implant.max_spatial_gradient_T_per_m is None:
            missing.append("max_spatial_gradient_T_per_m")
        if (
            implant.max_b1_rms_uT is None
            and implant.max_wb_sar_W_per_kg is None
            and implant.max_head_sar_W_per_kg is None
        ):
            missing.append("max_b1_rms_uT or max_wb_sar_W_per_kg/max_head_sar_W_per_kg")
        if missing:
            raise AbstentionError(
                code=AbstentionCode.UNSPECIFIED_IMPLANT_CONDITION,
                message=(
                    f"MR_CONDITIONAL device '{implant.device_name}' is missing required "
                    f"conditional envelope fields: {', '.join(missing)}."
                ),
                details={"device_id": implant.device_id, "missing_fields": missing},
            )


def gate_b0_phase_unwrappability(
    delta_b0_T: np.ndarray,
    delta_te_s: float,
    voxel_size_mm: tuple[float, float, float],
    max_phase_step_rad: float = np.pi,
) -> float:
    """Verify dual-echo B0 phase map does not violate Nyquist phase-unwrapping condition.

    Returns the maximum inter-voxel phase jump [rad]; raises UNWRAPPABLE_B0_PHASE if >= pi.
    """
    gamma_rad_s_T = 2.0 * np.pi * 42.577478518e6
    phase_rad = gamma_rad_s_T * delta_b0_T * delta_te_s
    max_jump = 0.0
    for axis in range(3):
        if phase_rad.shape[axis] > 1:
            diff = np.abs(np.diff(phase_rad, axis=axis))
            axis_max = float(np.max(diff))
            if axis_max > max_jump:
                max_jump = axis_max
    if max_jump >= max_phase_step_rad:
        raise AbstentionError(
            code=AbstentionCode.UNWRAPPABLE_B0_PHASE,
            message=(
                f"Inter-voxel B0 phase step ({max_jump:.2f} rad) reaches/exceeds Nyquist "
                f"unwrapping limit ({max_phase_step_rad:.2f} rad) at delta_TE={delta_te_s * 1e3:.2f} ms."
            ),
            details={
                "max_phase_step_rad": max_jump,
                "delta_te_ms": delta_te_s * 1e3,
                "voxel_size_mm": list(voxel_size_mm),
            },
        )
    return max_jump


def gate_srs_distortion_tolerance(
    target_max_mm: float,
    srs_tolerance_mm: float = 0.5,
) -> None:
    """Enforce sub-millimeter stereotactic radiosurgery (SRS) target geometric tolerance."""
    if target_max_mm > srs_tolerance_mm:
        raise AbstentionError(
            code=AbstentionCode.SRS_DISTORTION_EXCEEDS_TOLERANCE,
            message=(
                f"Target ROI maximum 3D geometric distortion ({target_max_mm:.3f} mm) "
                f"exceeds SRS tolerance ({srs_tolerance_mm:.3f} mm)."
            ),
            details={
                "target_max_mm": target_max_mm,
                "srs_tolerance_mm": srs_tolerance_mm,
            },
        )


def gate_implant_rf_violation(violated_constraints: list[str], implant_id: str) -> None:
    """Raise hard abstention when a protocol violates RF B1+rms or SAR limits."""
    rf_violations = [
        v for v in violated_constraints if ("b1_rms" in v.lower() or "sar" in v.lower())
    ]
    if rf_violations:
        raise AbstentionError(
            code=AbstentionCode.IMPLANT_SAR_OR_B1RMS_VIOLATION,
            message=(
                f"Protocol violates implant RF thermal limits for '{implant_id}': "
                f"{'; '.join(rf_violations)}"
            ),
            details={"implant_id": implant_id, "violations": rf_violations},
        )
