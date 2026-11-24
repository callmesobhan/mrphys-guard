"""Real pydicom MR / CT / NIfTI / NRRD loader and lossless synthetic MR/sCT DICOM round-trip builder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import nrrd
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    MRImageStorage,
    generate_uid,
)

from mrphys_guard.ir import MRSequenceSpec


@dataclass(frozen=True)
class LoadedMedicalVolume:
    """Standardized 3D medical volume with voxel spacing and extracted DICOM/header metadata."""

    array: np.ndarray
    voxel_size_mm: tuple[float, float, float]
    modality: str
    source_path: str
    metadata: dict[str, Any]


def load_nifti_volume(path: str | Path, modality: str = "MR") -> LoadedMedicalVolume:
    """Load a 3D NIfTI (.nii / .nii.gz) volume with voxel spacing."""
    p = Path(path)
    img = nib.load(str(p))
    arr = np.asarray(img.get_fdata(), dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI volume at {p}; got shape {arr.shape}.")
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return LoadedMedicalVolume(
        array=arr,
        voxel_size_mm=(zooms[0], zooms[1], zooms[2]),
        modality=modality,
        source_path=str(p),
        metadata={"format": "NIfTI", "shape": list(arr.shape)},
    )


def load_nrrd_volume(path: str | Path, modality: str = "MR") -> LoadedMedicalVolume:
    """Load a 3D NRRD (.nrrd) volume (such as 3D Slicer CT-MR Brain) with exact voxel spacing."""
    p = Path(path)
    data, header = nrrd.read(str(p))
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D NRRD volume at {p}; got shape {arr.shape}.")
    if "space directions" in header:
        dirs = np.asarray(header["space directions"], dtype=np.float64)
        spacings = tuple(float(np.linalg.norm(dirs[i])) for i in range(3))
    elif "spacings" in header:
        sp = header["spacings"]
        spacings = (float(sp[0]), float(sp[1]), float(sp[2]))
    else:
        spacings = (1.0, 1.0, 1.0)
    return LoadedMedicalVolume(
        array=arr,
        voxel_size_mm=spacings,
        modality=modality,
        source_path=str(p),
        metadata={
            "format": "NRRD",
            "shape": list(arr.shape),
            "space": str(header.get("space", "unknown")),
        },
    )


def build_mr_dicom_dataset(
    slice_2d: np.ndarray,
    sequence: MRSequenceSpec,
    patient_id: str = "MRPG-SYNTH-001",
    series_description: str | None = None,
    slice_location_mm: float = 0.0,
) -> FileDataset:
    """Build a standards-compliant lossless MR DICOM FileDataset encoding MR physics headers.

    Encodes:
    - MagneticFieldStrength (0018,0087)
    - PixelBandwidth (0018,0095)
    - FlipAngle (0018,1314)
    - RepetitionTime (0018,0080)
    - EchoTime (0018,0081)
    - SAR (0018,1316)
    - B1rms (0018,1320)
    - PixelSpacing (0028,0030) and SliceThickness (0018,0050)
    """
    arr = np.asarray(slice_2d, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"slice_2d must be 2D; got shape {arr.shape}.")

    # Lossless 16-bit unsigned scaling with exact RescaleSlope / RescaleIntercept
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    span = max(vmax - vmin, 1e-9)
    slope = span / 65535.0
    intercept = vmin
    scaled = np.clip(np.round((arr - intercept) / slope), 0, 65535).astype(np.uint16)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset("", {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "MR"
    ds.PatientID = patient_id
    ds.PatientName = f"SYNTHETIC^{patient_id}"
    ds.SeriesDescription = series_description or sequence.name
    ds.ScanningSequence = "GR" if sequence.sequence_type in {"GRE", "bSSFP", "Dixon", "MPRAGE"} else "SE"
    ds.SequenceVariant = "NONE"

    # Core MR Physics tags
    ds.MagneticFieldStrength = float(sequence.b0_T)
    ds.PixelBandwidth = float(sequence.bandwidth_hz_per_px)
    ds.FlipAngle = float(sequence.flip_angle_deg)
    ds.RepetitionTime = float(sequence.tr_ms)
    ds.EchoTime = float(sequence.te_ms)
    ds.EchoTrainLength = int(sequence.echo_train_length)
    if sequence.reported_sar_W_per_kg is not None:
        ds.SAR = float(sequence.reported_sar_W_per_kg)
    if sequence.reported_b1_rms_uT is not None:
        ds.B1rms = float(sequence.reported_b1_rms_uT)

    # Spatial geometry
    ds.Rows, ds.Columns = int(scaled.shape[0]), int(scaled.shape[1])
    ds.PixelSpacing = [float(sequence.voxel_size_mm[0]), float(sequence.voxel_size_mm[1])]
    ds.SliceThickness = float(sequence.voxel_size_mm[2])
    ds.SliceLocation = float(slice_location_mm)

    # Pixel formatting
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = float(slope)
    ds.RescaleIntercept = float(intercept)
    ds.PixelData = scaled.tobytes()
    return ds


def build_ct_dicom_dataset(
    hu_slice_2d: np.ndarray,
    voxel_size_mm: tuple[float, float, float] = (1.0, 1.0, 2.0),
    patient_id: str = "MRPG-SYNTH-SCT-001",
    series_description: str = "SyntheticCT_MRPhysGuard",
    slice_location_mm: float = 0.0,
) -> FileDataset:
    """Build a lossless CT/sCT DICOM FileDataset storing exact integer Hounsfield Units."""
    hu = np.asarray(hu_slice_2d, dtype=np.float64)
    if hu.ndim != 2:
        raise ValueError(f"hu_slice_2d must be 2D; got shape {hu.shape}.")

    intercept = -1024.0
    slope = 1.0
    stored = np.clip(np.round((hu - intercept) / slope), 0, 65535).astype(np.uint16)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset("", {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "CT"
    ds.PatientID = patient_id
    ds.PatientName = f"SYNTHETIC^{patient_id}"
    ds.SeriesDescription = series_description
    ds.Rows, ds.Columns = int(stored.shape[0]), int(stored.shape[1])
    ds.PixelSpacing = [float(voxel_size_mm[0]), float(voxel_size_mm[1])]
    ds.SliceThickness = float(voxel_size_mm[2])
    ds.SliceLocation = float(slice_location_mm)

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = float(slope)
    ds.RescaleIntercept = float(intercept)
    ds.PixelData = stored.tobytes()
    return ds


def save_dicom(ds: FileDataset, path: str | Path) -> Path:
    """Write a pydicom FileDataset cleanly without implicit-VR deprecation warnings."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(out), enforce_file_format=True)
    return out


def read_dicom_physics(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read an MR or CT DICOM file, returning rescaled float64 pixel array and physics headers."""
    ds: Dataset = pydicom.dcmread(str(path))
    raw = ds.pixel_array.astype(np.float64)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = raw * slope + intercept

    px_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    voxel_size = (
        float(px_spacing[0]),
        float(px_spacing[1]),
        float(getattr(ds, "SliceThickness", 1.0)),
    )

    headers: dict[str, Any] = {
        "Modality": str(getattr(ds, "Modality", "UNKNOWN")),
        "SeriesDescription": str(getattr(ds, "SeriesDescription", "")),
        "PatientID": str(getattr(ds, "PatientID", "")),
        "VoxelSizeMM": voxel_size,
        "MagneticFieldStrength": _opt_float(ds, "MagneticFieldStrength"),
        "PixelBandwidth": _opt_float(ds, "PixelBandwidth"),
        "FlipAngle": _opt_float(ds, "FlipAngle"),
        "RepetitionTime": _opt_float(ds, "RepetitionTime"),
        "EchoTime": _opt_float(ds, "EchoTime"),
        "EchoTrainLength": _opt_int(ds, "EchoTrainLength"),
        "SAR": _opt_float(ds, "SAR"),
        "B1rms": _opt_float(ds, "B1rms"),
    }
    return arr, headers


def sequence_spec_from_dicom(path: str | Path) -> MRSequenceSpec:
    """Parse an MR DICOM file directly into a validated MRSequenceSpec IR object."""
    _, hdr = read_dicom_physics(path)
    if hdr["Modality"] != "MR":
        raise ValueError(f"Expected MR DICOM at {path}; got Modality={hdr['Modality']}.")
    return MRSequenceSpec(
        name=hdr["SeriesDescription"] or Path(path).stem,
        b0_T=hdr["MagneticFieldStrength"] or 1.5,
        bandwidth_hz_per_px=hdr["PixelBandwidth"] or 250.0,
        voxel_size_mm=hdr["VoxelSizeMM"],
        flip_angle_deg=hdr["FlipAngle"] or 30.0,
        echo_train_length=hdr["EchoTrainLength"] or 1,
        tr_ms=hdr["RepetitionTime"] or 500.0,
        te_ms=hdr["EchoTime"] or 10.0,
        reported_sar_W_per_kg=hdr["SAR"],
        reported_b1_rms_uT=hdr["B1rms"],
    )


def _opt_float(ds: Dataset, attr: str) -> float | None:
    val = getattr(ds, attr, None)
    return float(val) if val is not None else None


def _opt_int(ds: Dataset, attr: str) -> int | None:
    val = getattr(ds, attr, None)
    return int(val) if val is not None else None
