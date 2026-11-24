"""Fetch real open-access paired clinical Brain CT + T1/T2 MRI benchmark volumes.

Downloads the 3D Slicer CT-MR Brain paired clinical dataset (CT-brain.nrrd, MR-brain-T1.nrrd,
MR-brain-T2.nrrd) from Slicer/SlicerTestingData content-addressed SHA-256 GitHub releases,
along with the verbatim upstream 3D Slicer LICENSE, verified SHA-256 MANIFEST.json, and
ATTRIBUTION.md.

Upstream License: 3D Slicer Software License (permissive BSD-style; sample data donated to
the 3D Slicer project to be used without restrictions).
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "public_ctmr_brain"

UPSTREAM_FILES = [
    {
        "filename": "CT-brain.nrrd",
        "url": "https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/6a5b6caccb76576a863beb095e3bfb910c50ca78f4c9bf043aa42f976cfa53d1",
        "expected_sha256": "6a5b6caccb76576a863beb095e3bfb910c50ca78f4c9bf043aa42f976cfa53d1",
        "modality": "CT",
        "description": "Paired clinical head CT volume (3D Slicer CTMRBrain sample)",
    },
    {
        "filename": "MR-brain-T1.nrrd",
        "url": "https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/2da3f655ed20356ee8cdf32aa0f8f9420385de4b6e407d28e67f9974d7ce1593",
        "expected_sha256": "2da3f655ed20356ee8cdf32aa0f8f9420385de4b6e407d28e67f9974d7ce1593",
        "modality": "MR_T1",
        "description": "Paired clinical head T1-weighted MRI volume (3D Slicer CTMRBrain sample)",
    },
    {
        "filename": "MR-brain-T2.nrrd",
        "url": "https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/fa1fe5910a69182f2b03c0150d8151ac6c75df986449fb5a6c5ae67141e0f5e7",
        "expected_sha256": "fa1fe5910a69182f2b03c0150d8151ac6c75df986449fb5a6c5ae67141e0f5e7",
        "modality": "MR_T2",
        "description": "Paired clinical head T2-weighted MRI volume (3D Slicer CTMRBrain sample)",
    },
]

LICENSE_URL = "https://raw.githubusercontent.com/Slicer/Slicer/main/License.txt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "mrphys-guard/0.1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_files = []

    # Fetch verbatim upstream 3D Slicer License.txt
    lic_path = DATA_DIR / "LICENSE"
    if not lic_path.exists():
        print(f"Downloading upstream LICENSE from {LICENSE_URL} ...")
        download_file(LICENSE_URL, lic_path)
    lic_sha = sha256_file(lic_path)
    manifest_files.append(
        {
            "filename": "LICENSE",
            "url": LICENSE_URL,
            "sha256": lic_sha,
            "size_bytes": lic_path.stat().st_size,
            "modality": "LICENSE",
            "description": "Verbatim 3D Slicer Software License Agreement",
        }
    )

    for entry in UPSTREAM_FILES:
        dest = DATA_DIR / entry["filename"]
        if not dest.exists():
            print(f"Downloading {entry['filename']} ...")
            download_file(entry["url"], dest)
        actual_sha = sha256_file(dest)
        if actual_sha != entry["expected_sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for {entry['filename']}: "
                f"expected {entry['expected_sha256']}, got {actual_sha}"
            )
        print(f"  Verified {entry['filename']} ({dest.stat().st_size / 1e6:.2f} MB, SHA-256 OK)")
        manifest_files.append(
            {
                "filename": entry["filename"],
                "url": entry["url"],
                "sha256": actual_sha,
                "size_bytes": dest.stat().st_size,
                "modality": entry["modality"],
                "description": entry["description"],
            }
        )

    manifest = {
        "dataset_id": "slicer_ctmr_brain_paired",
        "upstream_project": "3D Slicer (Slicer/SlicerTestingData)",
        "upstream_license": "3D Slicer Software License (permissive BSD-style; unrestricted sample data)",
        "fetched_utc": datetime.now(UTC).isoformat(),
        "provenance_tag": "PUBLIC_OPEN_CLINICAL",
        "files": manifest_files,
        "notes": (
            "Real clinical paired brain CT + T1/T2 MRI volumes. Any synthetic-CT failure "
            "perturbations or MR-Linac implant protocols applied during mrphys-guard benchmarks "
            "are explicitly tagged DERIVED_PERTURBATION or SYNTHETIC_PHYSICS_FIXTURE in the "
            "cryptographic provenance ledger."
        ),
    }
    manifest_path = DATA_DIR / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    attribution_md = """# Public Open-License Dataset Attribution (`public_ctmr_brain`)

- **Dataset Name:** 3D Slicer `CTMRBrain` (Paired Clinical Brain CT + T1-weighted MRI + T2-weighted MRI)
- **Upstream Source:** `https://github.com/Slicer/SlicerTestingData` (referenced by 3D Slicer `SampleData.py`)
- **Upstream License:** 3D Slicer Software License (BSD-style permissive open-source license; verbatim copy archived in [`LICENSE`](./LICENSE)). Sample datasets in 3D Slicer are donated for unrestricted research, educational, and software testing use.
- **Provenance Tag:** `PUBLIC_OPEN_CLINICAL`
- **Verification:** All files are content-addressed and SHA-256 verified in [`MANIFEST.json`](./MANIFEST.json).

## Benchmark Usage & Derived Perturbation Disclosure

In `mrphys-guard`, the unmodified `CT-brain.nrrd` and `MR-brain-T1.nrrd` volumes serve as a real clinical paired CT-MRI reference for testing:
1. NRRD volumetric I/O and physical voxel-spacing parsing (`dicom_io.py`),
2. 3D Lorentz-force Electron-Return-Effect (ERE) boundary risk computation on real cranial sinus and external skin tissue-air interfaces (`sct_ere_auditor.py`),
3. Synthetic-CT physics-plausibility auditing using controlled, explicitly labeled perturbations (`DERIVED_PERTURBATION`) of the real clinical CT volume.

## Companion Radiotherapy Benchmark Standard

`mrphys-guard` also directly supports the NIfTI layout (`mr.nii.gz`, `ct.nii.gz`, `mask.nii.gz`) of the **SynthRAD2023 Grand Challenge** (Thummerer et al., *Medical Physics* 2023, DOI `10.1002/mp.16529`; Zenodo `10.5281/zenodo.7260705`). Because the multi-center SynthRAD2023 Zenodo archive is several gigabytes and distributed under CC BY-NC 4.0, this repository bundles only the compact, permissively licensed 3D Slicer paired CT-MR clinical fixture while providing full `nibabel` NIfTI support for local SynthRAD2023 evaluation.

## Research Notice

**RESEARCH SOFTWARE — NOT A MEDICAL DEVICE.** These open clinical sample volumes are for algorithm verification and education only.
"""
    (DATA_DIR / "ATTRIBUTION.md").write_text(attribution_md, encoding="utf-8")
    print(f"Wrote {manifest_path} and ATTRIBUTION.md successfully.")


if __name__ == "__main__":
    main()
