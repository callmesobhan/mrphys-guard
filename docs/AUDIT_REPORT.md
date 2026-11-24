# Independent Engineering, Physics & Visual Audit Report

> **RESEARCH SOFTWARE — NOT A MEDICAL DEVICE.**

## 1. Executive Audit Summary

| Audit Dimension | Status | Verified Result |
| :--- | :---: | :--- |
| **Analytical Lorentz-Sphere Identity** | **PASS** | Interior mean $\Delta B_0 = 0.000000\text{ ppm}$ (RMS `0.019 ppm` out of `9.41 ppm`); exterior relative $L_2$ error = `0.54%` (`0.0054`) vs. exact $\frac{1}{3} B_0 \chi_0 (R/r)^3 (3\cos^2\theta - 1)$ |
| **Exact `1/BW` Scaling Law** | **PASS** | Verified to IEEE-754 machine precision (`rtol = 1e-14`) across $3\times$ bandwidth scaling |
| **Maxwell Solid Spherical Harmonics ($\ell \le 5$)** | **PASS** | Noiseless recovery `< 1e-6 mm`; noisy (`0.02 mm` $\sigma$) fit RMS = `0.034 mm` vs. $\ell \le 3$ RMS = `0.217 mm` |
| **ERE Magnetic Field Dependence** | **PASS** | Strictly obeys `1.5T > 0.35T > 0T = 0%` on both Pelvic Phantom (`28.7% > 5.6% > 0%`) and Real 3D Slicer Clinical Brain CT (`43.2% > 14.6% > 0%`) |
| **Formal IEC 60601-2-33 / ASTM F2503 CSP** | **PASS** | Exact RF scaling ($B_{1+\text{rms}} \propto \alpha/\sqrt{\mathrm{TR}\,\tau_{\text{RF}}}$, $\mathrm{SAR} \propto B_0^2 B_{1+\text{rms}}^2$); hard `ABSTAIN` on `MR_UNSAFE` and `UNSPECIFIED` implants |
| **Closed-Loop Co-Design Agent + Critic Gate** | **PASS** | Both flagship scenarios transition `VIOLATION -> CONVERGED` with Critic-Verifier `APPROVED`; SHA-256 hash chain `chain_valid = True` |
| **Saltelli–Jansen QMC Sobol' Estimator** | **PASS** | Recovers analytical Ishigami $S_1, S_T$ within `±0.04`; MR-RT suite $\sum S_1 \in [0.95, 0.99]$, $\sum S_T \in [1.01, 1.05]$ |
| **Real DICOM & Public NRRD I/O** | **PASS** | Lossless CT/sCT DICOM round-trip `0.0 HU` error; MR DICOM round-trip `5.2e-6` error; 3D Slicer `CTMRBrain` SHA-256 verified |
| **Automated Verification & Lint** | **PASS** | `26/26` pytest passing, `0` warnings (`-W error`), `92.1%` total coverage, `0` ruff errors |
| **Publication Figure Visual Audit** | **PASS** | All 6 300-DPI PNGs visually inspected with `read_file`; defects found and fixed on second pass |

## 2. Physics & Implementation Defects Caught and Resolved During Audit

### 2.1 Non-Harmonic Polynomial Term in Synthetic GNL Grid Phantom
- **Defect Caught:** Initial `generate_synthetic_grid_phantom()` included a radial term $x r^4$ whose Laplacian $\nabla^2(x r^4) \neq 0$. Because real cylindrical MRI gradient coils operate in a source-free bore ($\nabla^2 B_z = 0$), solid spherical harmonics ($\ell \le 5$) cannot fit non-harmonic polynomials, causing an artificial `0.152 mm` residual plateau in both $\ell \le 3$ and $\ell \le 5$ fits.
- **Root-Cause Fix:** Replaced the synthetic coil roll-off in [`distortion_solver.py`](../src/mrphys_guard/distortion_solver.py) with exact Maxwell-Laplace odd-order regular solid harmonics ($R_3^1, R_5^1$ for $x$; $R_3^{-1}, R_5^{-1}$ for $y$; $R_3^0, R_5^0$ for $z$).
- **Verification:** Noiseless $\ell \le 5$ recovery improved from `0.152 mm` to `< 1e-6 mm`; noisy (`0.02 mm` $\sigma$) fit RMS dropped to `0.034 mm` (exact $\sqrt{3}\sigma$ theoretical noise floor) while $\ell \le 3$ cleanly shows the expected `0.217 mm` $\ell = 5$ truncation residual.

### 2.2 Visual Audit Defects Found & Fixed Across the 6 Publication Figures
All 6 300-DPI PNGs were visually inspected via `read_file` after initial generation, revealing 6 specific publication-quality defects that were systematically resolved and re-verified:
1. **`fig1_3d_distortion_decomposition.png`:** Panel B GNL colorbar (`0–1.2 mm`) hid mid-FOV radial gradients, and Panel D saturated at `BW = 220 Hz/px`. Re-scaled Panel B to `[0, 0.35 mm]` and evaluated Panel D at `BW = 500 Hz/px` so the `0.5 mm` and `1.0 mm` SRS contours weave crisply across the prostate and rectal-gas interface.
2. **`fig2_bandwidth_snr_distortion_pareto.png`:** Panel B `"Agent Move"` text touched the baseline orange marker. Replaced with a boxed `"Agent Co-Design"` callout cleanly positioned above the curved arrow.
3. **`fig3_sct_audit_and_ere_risk.png`:** Panel A white scalebar lacked contrast on the light gray background and missed the bone-to-air swap slice; Panel D legend hid panel label `D`, and `0.00 T` bar labels touched. Updated bone-swap clustering to span mid-z (so all 3 sCT anomalies appear simultaneously in Panel A), switched Panel A scalebar to dark ink, offset Panel D legend below `D`, and separated bar labels.
4. **`fig4_implant_safety_csp_envelope.png`:** Panel A `0.94×` text overlapped the `1.0×` dashed limit; Panel B legend covered `B` and `"Cardiac Cine bSSFP"` clipped the right border. Placed near-limit green labels cleanly inside the top of bars in white ink, moved Panel B legend below `B`, and left-aligned `"Cardiac Cine bSSFP"` inside the axis frame.
5. **`fig5_phantom_spherical_harmonics_qa.png`:** Panel B legend hid `B`, and $\ell \le 3$ overlapped $\ell \le 5$ prior to the Maxwell-harmonic fix. Re-positioned legend cleanly below `B` and verified crisp separation between Uncorrected (`2.83 mm`), $\ell \le 3$ (`0.217 mm` RMS), and $\ell \le 5$ (`0.034 mm` RMS).
6. **`fig6_sobol_mr_uncertainty.png`:** Numeric labels sat directly on top of the 95% bootstrap error bars. Repositioned every value label cleanly above its upper 95% CI cap (`max(val, ci_high) + 0.025`).

## 3. Known Limitations & Explicit Non-Clinical Scope
- **Lorentz-Force ERE Kernel:** Analytic/geometric boundary-risk kernel calibrated to Raaijmakers et al. (2005, 2008) Monte Carlo trends; does not replace full coupled photon–electron Monte Carlo dose calculation (e.g., EGSnrc/Geant4/Monaco GPUMCD) in magnetic fields.
- **RF / SAR Thermal Model:** Uses a calibrated 70 kg adult coupling constant ($c_{\text{WB}}, c_{\text{head}}$) for sequence-level scaling rather than patient-specific FDTD electromagnetic simulations (Sim4Life/CST) around implant lead trajectories.
- **Generic Implant Profiles:** Built-in profiles are literature reference envelopes for constraint-solver testing and NEVER replace manufacturer IFU documentation or qualified MR Safety Officer (MRSO) review.
