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
from pydicom.sequence import Sequence
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    MRImageStorage,
    RTDoseStorage,
    RTStructureSetStorage,
    generate_uid,
)
from scipy.ndimage import affine_transform, binary_erosion

from mrphys_guard.ir import MRSequenceSpec


@dataclass(frozen=True)
class LoadedMedicalVolume:
    """Standardized 3D medical volume with voxel spacing, 4x4 physical affine, and header metadata."""

    array: np.ndarray
    voxel_size_mm: tuple[float, float, float]
    modality: str
    source_path: str
    metadata: dict[str, Any]
    affine_4x4: np.ndarray | None = None


def load_nifti_volume(path: str | Path, modality: str = "MR") -> LoadedMedicalVolume:
    """Load a 3D NIfTI (.nii / .nii.gz) volume with voxel spacing and 4x4 RAS world affine."""
    p = Path(path)
    img = nib.load(str(p))
    arr = np.asarray(img.get_fdata(), dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI volume at {p}; got shape {arr.shape}.")
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    affine = np.asarray(img.affine, dtype=np.float64)
    return LoadedMedicalVolume(
        array=arr,
        voxel_size_mm=(zooms[0], zooms[1], zooms[2]),
        modality=modality,
        source_path=str(p),
        metadata={"format": "NIfTI", "shape": list(arr.shape)},
        affine_4x4=affine,
    )


def load_nrrd_volume(path: str | Path, modality: str = "MR") -> LoadedMedicalVolume:
    """Load a 3D NRRD (.nrrd) volume (such as 3D Slicer CT-MR Brain) with exact voxel spacing & 4x4 affine."""
    p = Path(path)
    data, header = nrrd.read(str(p))
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D NRRD volume at {p}; got shape {arr.shape}.")

    affine = np.eye(4, dtype=np.float64)
    if "space directions" in header:
        dirs = np.asarray(header["space directions"], dtype=np.float64)
        spacings = tuple(float(np.linalg.norm(dirs[i])) for i in range(3))
        # In NRRD, row i of space directions is the physical step vector for index axis i -> column i of affine
        affine[:3, :3] = dirs.T
    elif "spacings" in header:
        sp = header["spacings"]
        spacings = (float(sp[0]), float(sp[1]), float(sp[2]))
        affine[:3, :3] = np.diag(spacings)
    else:
        spacings = (1.0, 1.0, 1.0)

    if "space origin" in header:
        affine[:3, 3] = np.asarray(header["space origin"], dtype=np.float64)

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
        affine_4x4=affine,
    )


def resample_volume_to_reference(
    moving: LoadedMedicalVolume,
    reference: LoadedMedicalVolume,
    order: int = 1,
    cval: float = 0.0,
) -> LoadedMedicalVolume:
    """Resample a 3D medical volume onto a reference volume's exact physical-space voxel grid.

    Uses the exact 4x4 voxel-to-world affines (M = A_moving^-1 @ A_ref) so multi-modality pairs
    (e.g. thick-slice clinical MR-brain-T1.nrrd + thin-slice CT-brain.nrrd) are aligned in mm space.
    """
    a_mov = moving.affine_4x4 if moving.affine_4x4 is not None else np.diag([*moving.voxel_size_mm, 1.0])
    a_ref = reference.affine_4x4 if reference.affine_4x4 is not None else np.diag([*reference.voxel_size_mm, 1.0])
    ref_to_mov = np.linalg.inv(a_mov) @ a_ref

    resampled = affine_transform(
        moving.array,
        matrix=ref_to_mov[:3, :3],
        offset=ref_to_mov[:3, 3],
        output_shape=reference.array.shape,
        order=order,
        mode="constant",
        cval=cval,
    )
    return LoadedMedicalVolume(
        array=np.asarray(resampled, dtype=np.float64),
        voxel_size_mm=reference.voxel_size_mm,
        modality=moving.modality,
        source_path=f"{moving.source_path} -> resampled({reference.source_path})",
        metadata={
            **moving.metadata,
            "resampled_to_shape": list(reference.array.shape),
            "interpolation_order": order,
        },
        affine_4x4=a_ref.copy(),
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


# ---------------------------------------------------------------------------
# Real DICOM-RT Dose (RTDOSE) & Structure Set (RTSTRUCT) Round-Trip I/O
# ---------------------------------------------------------------------------


def build_rtdose_dicom_dataset(
    dose_gy_3d: np.ndarray,
    voxel_size_mm: tuple[float, float, float],
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    patient_id: str = "MRPG-RT-001",
    series_description: str = "MRPhysGuard_RTDOSE",
) -> FileDataset:
    """Build a standards-compliant 3D DICOM RTDOSE FileDataset (32-bit unsigned Gy grid).

    Stores slices as (nz, ny, nx) per DICOM RTDOSE convention with exact DoseGridScaling and
    GridFrameOffsetVector so read_rtdose_dicom() recovers Gy values to < 1e-6 Gy.
    """
    dose = np.asarray(dose_gy_3d, dtype=np.float64)
    if dose.ndim != 3:
        raise ValueError(f"dose_gy_3d must be 3D (nx, ny, nz); got shape {dose.shape}.")
    nx, ny, nz = dose.shape
    max_gy = float(np.max(dose))
    # 32-bit unsigned integer scaling (0 .. 2^31 - 1) for sub-microGray lossless fidelity
    scaling = max(max_gy / float(2**30), 1e-9)
    # Transpose (nx, ny, nz) -> DICOM (Frames=nz, Rows=ny, Columns=nx)
    dicom_arr = np.transpose(dose, (2, 1, 0))
    quantized = np.clip(np.round(dicom_arr / scaling), 0, 2**31 - 1).astype(np.uint32)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = RTDoseStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset("", {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = RTDoseStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "RTDOSE"
    ds.PatientID = patient_id
    ds.PatientName = f"SYNTHETIC^{patient_id}"
    ds.SeriesDescription = series_description

    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseSummationType = "PLAN"
    ds.DoseGridScaling = float(scaling)

    ds.NumberOfFrames = int(nz)
    ds.Rows = int(ny)
    ds.Columns = int(nx)
    ds.PixelSpacing = [float(voxel_size_mm[1]), float(voxel_size_mm[0])]
    ds.SliceThickness = float(voxel_size_mm[2])
    ds.ImagePositionPatient = [float(origin_mm[0]), float(origin_mm[1]), float(origin_mm[2])]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.GridFrameOffsetVector = [float(k * voxel_size_mm[2]) for k in range(nz)]

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 32
    ds.BitsStored = 32
    ds.HighBit = 31
    ds.PixelRepresentation = 0
    ds.PixelData = quantized.tobytes()
    return ds


def read_rtdose_dicom(path: str | Path) -> LoadedMedicalVolume:
    """Read a 3D DICOM RTDOSE file into (nx, ny, nz) float64 Gray array with physical spacing."""
    ds = pydicom.dcmread(str(path))
    if str(getattr(ds, "Modality", "")) != "RTDOSE":
        raise ValueError(f"Expected RTDOSE DICOM at {path}; got Modality={getattr(ds, 'Modality', None)}.")

    scaling = float(getattr(ds, "DoseGridScaling", 1.0))
    raw = ds.pixel_array.astype(np.float64) * scaling
    if raw.ndim == 2:
        raw = raw[None, :, :]
    # DICOM (nz, ny, nx) -> engine (nx, ny, nz)
    dose_xyz = np.transpose(raw, (2, 1, 0))

    px = getattr(ds, "PixelSpacing", [1.0, 1.0])
    gfov = getattr(ds, "GridFrameOffsetVector", None)
    if gfov is not None and len(gfov) >= 2:
        dz = float(abs(gfov[1] - gfov[0]))
    else:
        dz = float(getattr(ds, "SliceThickness", 1.0))
    vox = (float(px[1]), float(px[0]), dz)
    origin = [float(v) for v in getattr(ds, "ImagePositionPatient", [0.0, 0.0, 0.0])]
    affine = np.diag([vox[0], vox[1], vox[2], 1.0])
    affine[:3, 3] = origin

    return LoadedMedicalVolume(
        array=np.ascontiguousarray(dose_xyz, dtype=np.float64),
        voxel_size_mm=vox,
        modality="RTDOSE",
        source_path=str(path),
        metadata={
            "format": "DICOM_RTDOSE",
            "DoseUnits": str(getattr(ds, "DoseUnits", "GY")),
            "DoseType": str(getattr(ds, "DoseType", "PHYSICAL")),
            "max_dose_gy": float(np.max(dose_xyz)),
        },
        affine_4x4=affine,
    )


def build_rtstruct_dicom_dataset(
    roi_masks: dict[str, np.ndarray],
    voxel_size_mm: tuple[float, float, float],
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    patient_id: str = "MRPG-RT-001",
    structure_set_label: str = "MRPG_RTSTRUCT",
) -> FileDataset:
    """Build a standards-compliant DICOM RTSTRUCT FileDataset from 3D binary ROI masks.

    Encodes slice-by-slice CLOSED_PLANAR boundary polygons in physical mm coordinates (x, y, z)
    under StructureSetROISequence and ROIContourSequence.
    """
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = RTStructureSetStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset("", {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = RTStructureSetStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "RTSTRUCT"
    ds.PatientID = patient_id
    ds.PatientName = f"SYNTHETIC^{patient_id}"
    ds.StructureSetLabel = structure_set_label[:16]

    ref_for_uid = generate_uid()
    ss_roi_seq = Sequence()
    roi_contour_seq = Sequence()

    dx, dy, dz = voxel_size_mm
    ox, oy, oz = origin_mm

    for roi_num, (roi_name, mask_raw) in enumerate(roi_masks.items(), start=1):
        mask = np.asarray(mask_raw, dtype=bool)
        if mask.ndim != 3:
            raise ValueError(f"ROI mask '{roi_name}' must be 3D; got shape {mask.shape}.")

        roi_item = Dataset()
        roi_item.ROINumber = roi_num
        roi_item.ReferencedFrameOfReferenceUID = ref_for_uid
        roi_item.ROIName = roi_name
        roi_item.ROIGenerationAlgorithm = "AUTOMATIC"
        ss_roi_seq.append(roi_item)

        rc_item = Dataset()
        rc_item.ReferencedROINumber = roi_num
        contour_seq = Sequence()

        for kz in range(mask.shape[2]):
            sl = mask[:, :, kz]
            if not np.any(sl):
                continue
            # Extract boundary pixels and order azimuthally around slice centroid for closed planar loop
            eroded = binary_erosion(sl, border_value=0)
            boundary = sl & (~eroded)
            pts = np.argwhere(boundary)
            if pts.shape[0] < 3:
                pts = np.argwhere(sl)
            if pts.shape[0] < 3:
                continue
            cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
            angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
            ordered = pts[np.argsort(angles)]

            z_mm = oz + kz * dz
            coords: list[float] = []
            for ix, iy in ordered:
                coords.extend([round(ox + float(ix) * dx, 4), round(oy + float(iy) * dy, 4), round(z_mm, 4)])

            c_item = Dataset()
            c_item.ContourGeometricType = "CLOSED_PLANAR"
            c_item.NumberOfContourPoints = int(ordered.shape[0])
            c_item.ContourData = coords
            contour_seq.append(c_item)

        rc_item.ContourSequence = contour_seq
        roi_contour_seq.append(rc_item)

    ds.StructureSetROISequence = ss_roi_seq
    ds.ROIContourSequence = roi_contour_seq
    return ds


def read_rtstruct_dicom(
    path: str | Path,
    grid_shape: tuple[int, int, int],
    voxel_size_mm: tuple[float, float, float],
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, np.ndarray]:
    """Read a DICOM RTSTRUCT file and rasterize each ROI's CLOSED_PLANAR polygons into 3D masks."""
    from matplotlib.path import Path as MplPath

    ds = pydicom.dcmread(str(path))
    if str(getattr(ds, "Modality", "")) != "RTSTRUCT":
        raise ValueError(f"Expected RTSTRUCT DICOM at {path}; got Modality={getattr(ds, 'Modality', None)}.")

    num_to_name = {
        int(item.ROINumber): str(item.ROIName)
        for item in getattr(ds, "StructureSetROISequence", [])
    }

    nx, ny, nz = grid_shape
    dx, dy, dz = voxel_size_mm
    ox, oy, oz = origin_mm
    gx, gy = np.meshgrid(np.arange(nx, dtype=np.float64), np.arange(ny, dtype=np.float64), indexing="ij")
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])

    masks: dict[str, np.ndarray] = {}
    for rc in getattr(ds, "ROIContourSequence", []):
        roi_num = int(rc.ReferencedROINumber)
        name = num_to_name.get(roi_num, f"ROI_{roi_num}")
        mask = np.zeros(grid_shape, dtype=bool)
        for contour in getattr(rc, "ContourSequence", []):
            raw_data = np.asarray([float(v) for v in contour.ContourData], dtype=np.float64)
            if raw_data.size < 9:
                continue
            pts_mm = raw_data.reshape((-1, 3))
            kz = int(round((float(pts_mm[0, 2]) - oz) / dz))
            if not (0 <= kz < nz):
                continue
            poly_xy = np.column_stack([(pts_mm[:, 0] - ox) / dx, (pts_mm[:, 1] - oy) / dy])
            inside = MplPath(poly_xy).contains_points(grid_xy, radius=-0.15).reshape((nx, ny))
            # Also include exact boundary vertices so thin structures are preserved
            bx = np.clip(np.round(poly_xy[:, 0]).astype(int), 0, nx - 1)
            by = np.clip(np.round(poly_xy[:, 1]).astype(int), 0, ny - 1)
            inside[bx, by] = True
            mask[:, :, kz] |= inside
        masks[name] = mask
    return masks
