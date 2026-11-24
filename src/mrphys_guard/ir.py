"""Strictly typed Pydantic v2 Physics Intermediate Representation (IR) for mrphys-guard.

Defines immutable-by-convention scientific records for MRI pulse sequences, scanner hardware
envelopes, ASTM F2503 / IEC 60601-2-33 MR-conditional implants, 3D distortion decompositions,
synthetic-CT physics plausibility audits, Lorentz-force electron-return-effect (ERE) boundary
risk assessments, formal safety CSP verdicts, SHA-256 provenance records, and hard abstentions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MRSafetyLabel(StrEnum):
    """ASTM F2503 MR safety classification."""

    MR_SAFE = "MR_SAFE"
    MR_CONDITIONAL = "MR_CONDITIONAL"
    MR_UNSAFE = "MR_UNSAFE"
    UNSPECIFIED = "UNSPECIFIED"


class RFCoilType(StrEnum):
    """Transmit RF coil configuration per IEC 60601-2-33."""

    WHOLE_BODY_CP = "whole_body_circularly_polarized"
    TX_RX_HEAD = "tx_rx_head_only"
    TX_RX_EXTREMITY = "tx_rx_extremity"
    MULTI_TRANSMIT_PARALLEL = "multi_transmit_parallel"


class AbstentionCode(StrEnum):
    """Hard safety and physics abstention codes."""

    UNSPECIFIED_IMPLANT_CONDITION = "UNSPECIFIED_IMPLANT_CONDITION"
    MR_UNSAFE_DEVICE_DETECTED = "MR_UNSAFE_DEVICE_DETECTED"
    IMPLANT_SAR_OR_B1RMS_VIOLATION = "IMPLANT_SAR_OR_B1RMS_VIOLATION"
    UNWRAPPABLE_B0_PHASE = "UNWRAPPABLE_B0_PHASE"
    SRS_DISTORTION_EXCEEDS_TOLERANCE = "SRS_DISTORTION_EXCEEDS_TOLERANCE"


class AbstentionError(RuntimeError):
    """Raised when a hard physics or clinical safety gate prohibits emitting a passing result."""

    def __init__(
        self,
        code: AbstentionCode,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"[{code.value}] {message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "abstained": True,
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }


class MRSequenceSpec(BaseModel):
    """Pulse sequence specification for distortion, SNR, and RF/SAR/gradient safety physics."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Human-readable sequence identifier")
    sequence_type: Literal["GRE", "TSE", "FSE", "bSSFP", "EPI", "MPRAGE", "Dixon", "SE"] = Field(
        default="GRE",
        description="Pulse sequence family",
    )
    b0_T: float = Field(default=1.5, gt=0.0, le=14.0, description="Static magnetic field B0 [T]")
    bandwidth_hz_per_px: float = Field(
        ...,
        gt=10.0,
        le=5000.0,
        description="Receiver readout bandwidth per pixel [Hz/px]",
    )
    voxel_size_mm: tuple[float, float, float] = Field(
        default=(1.0, 1.0, 1.0),
        description="Voxel spacing (readout, phase_encode, slice) [mm]",
    )
    readout_axis: Literal[0, 1, 2] = Field(
        default=0,
        description="Axis index of frequency-encoding (readout) direction",
    )
    phase_encode_axis: Literal[0, 1, 2] = Field(
        default=1,
        description="Axis index of phase-encoding direction",
    )
    flip_angle_deg: float = Field(
        default=30.0,
        gt=0.0,
        le=180.0,
        description="Excitation flip angle [degrees]",
    )
    refocusing_flip_angle_deg: float = Field(
        default=150.0,
        ge=0.0,
        le=180.0,
        description="Refocusing flip angle for spin-echo/TSE trains [degrees]",
    )
    echo_train_length: int = Field(
        default=1,
        ge=1,
        le=256,
        description="Number of echoes per TR (refocusing pulses = ETL - 1 for TSE/FSE)",
    )
    tr_ms: float = Field(..., gt=0.5, le=20000.0, description="Repetition time TR [ms]")
    te_ms: float = Field(..., gt=0.1, le=500.0, description="Echo time TE [ms]")
    rf_pulse_duration_ms: float = Field(
        default=2.0,
        gt=0.1,
        le=20.0,
        description="RF pulse duration tau_RF [ms]",
    )
    rf_shape_factor: float = Field(
        default=1.6,
        ge=1.0,
        le=6.0,
        description="Dimensionless RF waveform power factor (1.0=hard rect, ~1.5-2.0=apodized sinc)",
    )
    echo_spacing_ms: float | None = Field(
        default=None,
        gt=0.05,
        le=10.0,
        description="EPI echo spacing [ms] (required for EPI phase-encode distortion)",
    )
    phase_encode_steps: int = Field(
        default=128,
        ge=16,
        le=1024,
        description="Phase-encode matrix size N_PE",
    )
    parallel_accel_factor: float = Field(
        default=1.0,
        ge=1.0,
        le=8.0,
        description="Parallel imaging acceleration factor R_PAT (SENSE/GRAPPA)",
    )
    slew_rate_T_per_m_s: float = Field(
        default=120.0,
        ge=0.0,
        le=400.0,
        description="Effective gradient slew rate [T/(m*s)]",
    )
    scan_duration_min: float = Field(
        default=4.0,
        gt=0.0,
        le=120.0,
        description="Active scan duration [min]",
    )
    rf_coil: RFCoilType = Field(
        default=RFCoilType.WHOLE_BODY_CP,
        description="Transmit RF coil configuration",
    )
    anatomy: Literal["whole_body", "head", "pelvis", "cardiac", "extremity"] = Field(
        default="pelvis",
        description="Target anatomical region for SAR model selection",
    )
    reported_sar_W_per_kg: float | None = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description="Optional DICOM-reported whole-body or head SAR [W/kg]",
    )
    reported_b1_rms_uT: float | None = Field(
        default=None,
        ge=0.0,
        le=30.0,
        description="Optional DICOM-reported B1+rms [uT]",
    )

    @model_validator(mode="after")
    def _validate_axes_and_te(self) -> MRSequenceSpec:
        if self.readout_axis == self.phase_encode_axis:
            raise ValueError("readout_axis and phase_encode_axis must be distinct.")
        if self.te_ms > self.tr_ms:
            raise ValueError(f"te_ms ({self.te_ms}) cannot exceed tr_ms ({self.tr_ms}).")
        for s in self.voxel_size_mm:
            if s <= 0.0:
                raise ValueError("All voxel_size_mm entries must be strictly positive.")
        return self


class ScannerFieldSpec(BaseModel):
    """MRI / MR-Linac scanner hardware and static/gradient field specification."""

    model_config = ConfigDict(extra="forbid")

    scanner_name: str = Field(default="Generic 1.5T MR-Linac", min_length=1)
    b0_T: float = Field(default=1.5, gt=0.0, le=14.0, description="Nominal static field B0 [T]")
    max_spatial_gradient_T_per_m: float = Field(
        default=18.0,
        ge=0.0,
        le=200.0,
        description="Maximum accessible spatial magnetic field gradient |dB/dx| [T/m]",
    )
    max_slew_rate_T_per_m_s: float = Field(
        default=200.0,
        gt=0.0,
        le=500.0,
        description="Maximum gradient system slew rate [T/(m*s)]",
    )
    max_db_dt_T_per_s: float = Field(
        default=60.0,
        ge=0.0,
        le=300.0,
        description="Maximum dB/dt at patient bore radius [T/s]",
    )
    dsv_radius_mm: float = Field(
        default=200.0,
        gt=20.0,
        le=350.0,
        description="Diameter-of-spherical-volume (DSV) half-radius [mm]",
    )
    available_coils: list[RFCoilType] = Field(
        default_factory=lambda: [RFCoilType.WHOLE_BODY_CP, RFCoilType.TX_RX_HEAD],
    )
    operating_mode: Literal["normal", "first_level_controlled"] = Field(
        default="normal",
        description="IEC 60601-2-33 RF operating mode",
    )


class ImplantConditionSpec(BaseModel):
    """ASTM F2503 / IEC 60601-2-33 MR-Conditional implant safety envelope."""

    model_config = ConfigDict(extra="forbid")

    device_id: str = Field(..., min_length=1)
    device_name: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    mr_status: MRSafetyLabel = Field(...)
    source_note: str = Field(
        default="Generic literature reference profile — NOT a manufacturer IFU.",
    )
    allowed_b0_T: list[float] | None = Field(
        default=None,
        description="Permitted static magnetic field strengths [T]",
    )
    max_spatial_gradient_T_per_m: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum permitted spatial field gradient |dB/dx| [T/m]",
    )
    max_b1_rms_uT: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum permitted B1+rms [uT]",
    )
    max_wb_sar_W_per_kg: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum permitted whole-body averaged SAR [W/kg]",
    )
    max_head_sar_W_per_kg: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum permitted head SAR [W/kg]",
    )
    max_slew_rate_T_per_m_s: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum permitted gradient slew rate per axis [T/(m*s)]",
    )
    max_db_dt_T_per_s: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum permitted dB/dt [T/s]",
    )
    max_active_scan_time_min: float | None = Field(
        default=None,
        gt=0.0,
        description="Maximum cumulative active RF scan time per session [min]",
    )
    allowed_rf_coils: list[RFCoilType] | None = Field(
        default=None,
        description="Permitted transmit RF coil configurations",
    )
    requires_isocenter_exclusion: bool = Field(
        default=False,
        description="Whether the device requires anatomical landmark exclusion",
    )


class DistortionDecomposition(BaseModel):
    """Structured summary of 3D geometric distortion decomposition."""

    model_config = ConfigDict(extra="forbid")

    b0_T: float
    bandwidth_hz_per_px: float
    voxel_size_mm: tuple[float, float, float]
    gnl_max_mm: float = Field(..., ge=0.0)
    gnl_rms_mm: float = Field(..., ge=0.0)
    gnl_p95_mm: float = Field(..., ge=0.0)
    susceptibility_delta_b0_min_ppm: float
    susceptibility_delta_b0_max_ppm: float
    susceptibility_max_mm: float = Field(..., ge=0.0)
    susceptibility_rms_mm: float = Field(..., ge=0.0)
    epi_phase_encode_max_mm: float = Field(default=0.0, ge=0.0)
    total_max_mm: float = Field(..., ge=0.0)
    total_rms_mm: float = Field(..., ge=0.0)
    total_p95_mm: float = Field(..., ge=0.0)
    target_roi_max_mm: float = Field(..., ge=0.0)
    target_roi_p95_mm: float = Field(..., ge=0.0)
    fraction_exceeding_0p5mm: float = Field(..., ge=0.0, le=1.0)
    fraction_exceeding_1p0mm: float = Field(..., ge=0.0, le=1.0)
    srs_tolerance_mm: float = Field(default=0.5, gt=0.0)
    passes_srs_tolerance: bool
    passes_tg284_2mm_dsv: bool
    relative_snr: float = Field(..., gt=0.0)
    harmonic_order_lmax: int = Field(default=5, ge=0)
    harmonic_fit_residual_rms_mm: float = Field(default=0.0, ge=0.0)


class SCTAnomalyFlag(BaseModel):
    """Single detected synthetic-CT physics plausibility anomaly."""

    model_config = ConfigDict(extra="forbid")

    anomaly_type: Literal[
        "BONE_TO_AIR_SWAP",
        "AIR_TO_BONE_SWAP",
        "MISSING_METAL_ARTIFACT",
        "BODY_CONTOUR_TRUNCATION",
        "SEVERE_HU_BIAS",
    ]
    severity: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    voxel_count: int = Field(..., ge=0)
    volume_cc: float = Field(..., ge=0.0)
    mean_hu_discrepancy: float
    estimated_wepl_error_mm: float = Field(..., ge=0.0)
    description: str


class SCTAuditReport(BaseModel):
    """Synthetic-CT physics plausibility audit summary."""

    model_config = ConfigDict(extra="forbid")

    volume_shape: tuple[int, int, int]
    voxel_size_mm: tuple[float, float, float]
    body_volume_cc: float = Field(..., ge=0.0)
    mae_hu_body: float | None = Field(default=None, ge=0.0)
    mae_hu_bone: float | None = Field(default=None, ge=0.0)
    mae_hu_air: float | None = Field(default=None, ge=0.0)
    mean_rel_ed_error: float | None = Field(default=None, ge=0.0)
    max_wepl_error_mm: float = Field(..., ge=0.0)
    estimated_6mv_dose_error_pct: float = Field(..., ge=0.0)
    bone_to_air_swap_voxels: int = Field(..., ge=0)
    missing_metal_voxels: int = Field(..., ge=0)
    contour_truncation_voxels: int = Field(..., ge=0)
    anomalies: list[SCTAnomalyFlag] = Field(default_factory=list)
    overall_status: Literal["PASS", "WARNING", "FAIL"]


class EREBoundaryRisk(BaseModel):
    """Lorentz-force Electron-Return-Effect (ERE) boundary risk assessment."""

    model_config = ConfigDict(extra="forbid")

    b0_T: float = Field(..., ge=0.0)
    beam_energy_mv: float = Field(default=6.0, gt=0.0)
    mean_secondary_electron_mev: float = Field(..., gt=0.0)
    beam_to_b0_angle_deg: float = Field(..., ge=0.0, le=180.0)
    gyroradius_mm: float | None = Field(
        default=None,
        description="Transverse Lorentz gyroradius r_g [mm]; None when B0=0 or parallel",
    )
    interface_voxel_count: int = Field(..., ge=0)
    high_risk_voxel_count: int = Field(..., ge=0)
    peak_dose_enhancement_pct: float = Field(..., ge=0.0)
    p95_interface_enhancement_pct: float = Field(..., ge=0.0)
    mean_interface_enhancement_pct: float = Field(..., ge=0.0)
    dominant_interfaces: list[str] = Field(default_factory=list)
    risk_tier: Literal["NONE", "LOW", "MODERATE", "HIGH", "CRITICAL"]


class ConstraintCheckResult(BaseModel):
    """Evaluation of one formal IEC 60601-2-33 / ASTM F2503 safety constraint."""

    model_config = ConfigDict(extra="forbid")

    constraint_name: str
    sequence_name: str
    observed_value: float | str
    limit_value: float | str
    unit: str
    margin_pct: float | None = Field(
        default=None,
        description="Positive = safe headroom (%); negative = violation magnitude (%)",
    )
    satisfied: bool
    rationale: str


class SequencePhysicsEstimate(BaseModel):
    """Computed RF, SAR, gradient, PNS, and SNR physics for a single MRI sequence."""

    model_config = ConfigDict(extra="forbid")

    sequence_name: str
    b1_rms_uT: float = Field(..., ge=0.0)
    wb_sar_W_per_kg: float = Field(..., ge=0.0)
    head_sar_W_per_kg: float = Field(..., ge=0.0)
    slew_rate_T_per_m_s: float = Field(..., ge=0.0)
    estimated_db_dt_T_per_s: float = Field(..., ge=0.0)
    pns_stimulation_pct: float = Field(
        default=0.0,
        ge=0.0,
        description="IEC 60601-2-33 chronaxie/rheobase PNS stimulation percentage (80%=normal, 100%=first-level)",
    )
    relative_snr: float = Field(..., gt=0.0)
    scan_duration_min: float = Field(..., gt=0.0)


class ConflictCore(BaseModel):
    """Minimal conflict explanation when simultaneous multi-implant CSP is violated or infeasible."""

    model_config = ConfigDict(extra="forbid")

    conflict_type: str
    involved_devices: list[str]
    constraint_name: str
    explanation: str


class MultiImplantCSPVerdict(BaseModel):
    """Simultaneous formal IEC 60601-2-33 / ASTM F2503 CSP verdict across multiple co-existing implants."""

    model_config = ConfigDict(extra="forbid")

    scanner_name: str
    implant_ids: list[str]
    verdict: Literal["PASS", "VIOLATION", "ABSTAIN"]
    abstention_code: AbstentionCode | None = None
    effective_b0_intersection_T: list[float] = Field(default_factory=list)
    effective_max_b1_rms_uT: float | None = None
    effective_max_wb_sar_W_per_kg: float | None = None
    effective_max_slew_rate_T_per_m_s: float | None = None
    effective_max_active_time_min: float | None = None
    governing_device_by_limit: dict[str, str] = Field(default_factory=dict)
    per_implant_verdicts: list[SafetyCSPVerdict] = Field(default_factory=list)
    conflict_cores: list[ConflictCore] = Field(default_factory=list)
    summary: str


class MultiBeamEREReport(BaseModel):
    """Multi-beam IMRT/VMAT Lorentz-force ERE cancellation & gantry-avoidance analysis."""

    model_config = ConfigDict(extra="forbid")

    b0_T: float = Field(..., ge=0.0)
    beam_energy_mv: float = Field(default=6.0, gt=0.0)
    gantry_angles_deg: list[float]
    beam_weights: list[float]
    single_beam_peak_pct: float = Field(..., ge=0.0)
    multibeam_net_peak_pct: float = Field(..., ge=0.0)
    multibeam_p95_pct: float = Field(..., ge=0.0)
    cancellation_reduction_pct: float = Field(
        ...,
        description="Relative reduction in peak ERE spike via opposing/multi-beam cancellation [%]",
    )
    recommended_low_ere_angles_deg: list[float] = Field(default_factory=list)
    highest_risk_angles_deg: list[float] = Field(default_factory=list)
    angle_peak_ere_map: dict[str, float] = Field(default_factory=dict)


class PhaseUnwrapReport(BaseModel):
    """3D DCT Poisson/Laplacian dual-echo B0 phase unwrapping & Jacobian singularity summary."""

    model_config = ConfigDict(extra="forbid")

    volume_shape: tuple[int, int, int]
    delta_te_ms: float = Field(..., gt=0.0)
    max_abs_unwrapped_rad: float = Field(..., ge=0.0)
    cycle_count_range: int = Field(..., ge=0)
    delta_b0_min_ppm: float
    delta_b0_max_ppm: float
    wrap_consistency_rms_rad: float = Field(..., ge=0.0)


class JacobianUnwarpReport(BaseModel):
    """Distortion Jacobian determinant & folding/pile-up singularity report."""

    model_config = ConfigDict(extra="forbid")

    min_jacobian: float
    max_jacobian: float
    folding_voxel_count: int = Field(..., ge=0)
    pileup_voxel_count: int = Field(..., ge=0)
    has_folding_singularity: bool
    signal_conservation_ratio: float = Field(..., gt=0.0)


class SafetyCSPVerdict(BaseModel):
    """Formal Constraint Satisfaction Problem (CSP) verdict for an MR protocol + implant."""

    model_config = ConfigDict(extra="forbid")

    scanner_name: str
    implant_id: str
    implant_name: str
    mr_status: MRSafetyLabel
    verdict: Literal["PASS", "VIOLATION", "ABSTAIN"]
    abstention_code: AbstentionCode | None = None
    total_active_scan_time_min: float = Field(..., ge=0.0)
    max_b1_rms_uT: float = Field(..., ge=0.0)
    max_wb_sar_W_per_kg: float = Field(..., ge=0.0)
    max_head_sar_W_per_kg: float = Field(..., ge=0.0)
    max_slew_rate_T_per_m_s: float = Field(..., ge=0.0)
    sequence_estimates: list[SequencePhysicsEstimate] = Field(default_factory=list)
    checks: list[ConstraintCheckResult] = Field(default_factory=list)
    violated_constraints: list[str] = Field(default_factory=list)
    summary: str


class ProvenanceRecord(BaseModel):
    """Cryptographic SHA-256 provenance record for auditability and reproducibility."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )
    engine_version: str = Field(default="0.1.0")
    operation: str
    data_origin: Literal["PUBLIC_OPEN_CLINICAL", "SYNTHETIC_PHYSICS_FIXTURE", "DERIVED_PERTURBATION"]
    input_sha256: str = Field(..., min_length=64, max_length=64)
    output_sha256: str = Field(..., min_length=64, max_length=64)
    previous_record_sha256: str | None = None
    record_sha256: str = Field(..., min_length=64, max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)
