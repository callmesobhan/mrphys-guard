# Public Open-License Dataset Attribution (`public_ctmr_brain`)

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
