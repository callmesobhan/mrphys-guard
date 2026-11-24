# Data Sources, Standards & Scientific Citations

> **RESEARCH SOFTWARE — NOT A MEDICAL DEVICE.** All datasets, standards references, and literature models cited below are used strictly for computational physics benchmarking and education.

## 1. Bundled Open-License Clinical Benchmark Fixture

| Dataset | Location | Format | Upstream License | Provenance Tag |
| :--- | :--- | :---: | :--- | :---: |
| **3D Slicer `CTMRBrain`** (`CT-brain.nrrd`, `MR-brain-T1.nrrd`, `MR-brain-T2.nrrd`) | `data/public_ctmr_brain/` | NRRD | 3D Slicer Software License (BSD-style permissive; sample data donated without restrictions; verbatim `LICENSE` bundled) | `PUBLIC_OPEN_CLINICAL` |

- **Download Script:** [`scripts/fetch_public_mri_sample.py`](scripts/fetch_public_mri_sample.py) (content-addressed SHA-256 GitHub release URLs from `Slicer/SlicerTestingData`, verified in [`data/public_ctmr_brain/MANIFEST.json`](data/public_ctmr_brain/MANIFEST.json)).
- **Derived Perturbation Disclosure:** Any synthetic-CT failure modes (bone-to-air swaps, missing metal, contour truncation) applied to the clinical CT volume are generated deterministically by `inject_sct_failure_modes()` and explicitly tagged `DERIVED_PERTURBATION` in the SHA-256 provenance ledger. Synthetic pelvic SRS phantoms and generic literature implant profiles are tagged `SYNTHETIC_PHYSICS_FIXTURE`.

## 2. Companion External Open-Access Dataset Compatibility

- **SynthRAD2023 Grand Challenge Dataset** (Thummerer et al., *Medical Physics* 2023, DOI: `10.1002/mp.16529`):
  - Multi-center paired MRI/CBCT + CT NIfTI dataset (`mr.nii.gz`, `ct.nii.gz`, `mask.nii.gz`) hosted on Zenodo (Training: `10.5281/zenodo.7260705`; Validation: `10.5281/zenodo.7868168`).
  - Note on upstream licensing: while the *Medical Physics* data descriptor is open access (CC BY 4.0), the multi-gigabyte Zenodo image archives are released under CC BY-NC 4.0. `mrphys-guard` provides native `nibabel` NIfTI loaders (`load_nifti_volume`) compatible with SynthRAD2023 directory layouts without redistributing large upstream archives inside this repository.

## 3. Verified Literature & Standards References (BibTeX)

```bibtex
@article{glidehurst2021aapm_tg284,
  author  = {Glide-Hurst, Carri K. and Paulson, Eric S. and McGee, Kiaran and Tyagi, Neelam and Hu, Yanle and Balter, James and Bayouth, John},
  title   = {Task Group 284 report: Magnetic resonance imaging simulation in radiotherapy: Considerations for clinical implementation, optimization, and quality assurance},
  journal = {Medical Physics},
  year    = {2021},
  volume  = {48},
  number  = {7},
  pages   = {e636--e670},
  doi     = {10.1002/mp.14695}
}

@article{thummerer2023synthrad,
  author  = {Thummerer, Adrian and van der Bijl, Erik and Galapon, Arthur and Verhoeff, Joost J. C. and Langendijk, Johannes A. and Both, Stefan and van den Berg, Cornelis A. T. and Maspero, Matteo},
  title   = {SynthRAD2023 Grand Challenge dataset: Generating synthetic CT for radiotherapy},
  journal = {Medical Physics},
  year    = {2023},
  volume  = {50},
  number  = {7},
  pages   = {4664--4674},
  doi     = {10.1002/mp.16529}
}

@article{marques2005fourier_susceptibility,
  author  = {Marques, Jos{\'e} P. and Bowtell, Richard},
  title   = {Application of a Fourier-based method for rapid calculation of field inhomogeneity due to spatial variation of magnetic susceptibility},
  journal = {Concepts in Magnetic Resonance Part B: Magnetic Resonance Engineering},
  year    = {2005},
  volume  = {25B},
  number  = {1},
  pages   = {65--78},
  doi     = {10.1002/cmr.b.20034}
}

@article{raaijmakers2005ere,
  author  = {Raaijmakers, A. J. E. and Raaymakers, B. W. and Lagendijk, J. J. W.},
  title   = {Integrating a MRI scanner with a 6 MV radiotherapy accelerator: dose increase at tissue--air interfaces in a lateral magnetic field due to returning electrons},
  journal = {Physics in Medicine and Biology},
  year    = {2005},
  volume  = {50},
  number  = {7},
  pages   = {1363--1376},
  doi     = {10.1088/0031-9155/50/7/002}
}

@article{raaijmakers2008ere_b0,
  author  = {Raaijmakers, A. J. E. and Raaymakers, B. W. and Lagendijk, J. J. W.},
  title   = {Magnetic-field-induced dose effects in MR-guided radiotherapy systems: dependence on the magnetic field strength},
  journal = {Physics in Medicine and Biology},
  year    = {2008},
  volume  = {53},
  number  = {4},
  pages   = {909--923},
  doi     = {10.1088/0031-9155/53/4/006}
}

@article{saltelli2010sobol,
  author  = {Saltelli, Andrea and Annoni, Paola and Azzini, Ivano and Campolongo, Francesca and Ratto, Marco and Tarantola, Stefano},
  title   = {Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index},
  journal = {Computer Physics Communications},
  year    = {2010},
  volume  = {181},
  number  = {2},
  pages   = {259--270},
  doi     = {10.1016/j.cpc.2009.09.018}
}

@techreport{iec60601_2_33_2022,
  author      = {{International Electrotechnical Commission}},
  title       = {IEC 60601-2-33:2022 (Edition 4.0) Medical electrical equipment -- Part 2-33: Particular requirements for the basic safety and essential performance of magnetic resonance equipment for medical diagnosis},
  institution = {IEC},
  year        = {2022}
}

@techreport{astm_f2503,
  author      = {{ASTM International}},
  title       = {ASTM F2503 Standard Practice for Marking Medical Devices and Other Items for Safety in the Magnetic Resonance Environment},
  institution = {ASTM International},
  year        = {2020}
}

@article{vanherk2000margins,
  author  = {van Herk, Marcel and Remeijer, Peter and Rasch, Coen and Lebesque, Joos V.},
  title   = {The probability of correct target dosage: dose-population histograms for deriving treatment margins in radiotherapy},
  journal = {International Journal of Radiation Oncology, Biology, Physics},
  year    = {2000},
  volume  = {47},
  number  = {4},
  pages   = {1121--1135},
  doi     = {10.1016/S0360-3016(00)00518-6}
}

@article{ghiglia1994dct_unwrap,
  author  = {Ghiglia, Dennis C. and Romero, Louis A.},
  title   = {Robust two-dimensional weighted and unweighted phase unwrapping that uses fast transforms and iterative methods},
  journal = {Journal of the Optical Society of America A},
  year    = {1994},
  volume  = {11},
  number  = {1},
  pages   = {107--117},
  doi     = {10.1364/JOSAA.11.000107}
}
```
