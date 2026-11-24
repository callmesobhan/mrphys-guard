# mrphys-guard Engine Specification

> **RESEARCH SOFTWARE — NOT A MEDICAL DEVICE.**

## 1. Physical Constants & Units

| Constant | Symbol | Value | Units | Source |
| :--- | :---: | :---: | :---: | :--- |
| $^{1}\text{H}$ Gyromagnetic Ratio | $\gamma / (2\pi)$ | $42.577478518 \times 10^6$ | Hz/T | CODATA 2018/2022 |
| Electron Rest Energy | $m_e c^2$ | $0.51099895$ | MeV | CODATA 2018/2022 |
| Speed of Light | $c$ | $2.99792458 \times 10^8$ | m/s | SI exact |
| Air Volumetric Susceptibility | $\chi_{\text{air}}$ | $+0.36 \times 10^{-6}$ | SI | Literature |
| Water / Soft Tissue Susceptibility | $\chi_{\text{water}}$ | $-9.05 \times 10^{-6}$ | SI | Literature |
| Cortical Bone Susceptibility | $\chi_{\text{bone}}$ | $-11.3 \times 10^{-6}$ | SI | Literature |
| Ti-6Al-4V Prosthesis Susceptibility | $\chi_{\text{Ti}}$ | $+182.0 \times 10^{-6}$ | SI | Literature |

## 2. 3D Geometric Distortion Solver (`distortion_solver.py`)

### 2.1 System Gradient Nonlinearity (GNL)
In a source-free cylindrical MRI bore, each gradient field component satisfies Laplace's equation $\nabla^2 B_z^{(i)} = 0$. Consequently, the Cartesian system geometric displacement vector $\Delta\mathbf{r}_{\text{GNL}}(\mathbf{x}) = (\Delta x, \Delta y, \Delta z)$ is expanded in normalized real regular solid spherical harmonics up to order $\ell_{\max} = 5$:
$$\Delta r_j(\mathbf{x}) = \sum_{\ell=0}^{5} \sum_{m=-\ell}^{\ell} c_{j,\ell m} \left(\frac{r}{R_0}\right)^\ell \tilde{Y}_\ell^m(\theta, \phi), \qquad R_0 = 200\text{ mm}$$
yielding $(5 + 1)^2 = 36$ basis functions per Cartesian axis, fitted to 3D grid-phantom fiducial displacements $\mathbf{r}_{\text{meas}} - \mathbf{r}_{\text{true}}$ via conditioned SVD / ridge least squares.

### 2.2 Patient Susceptibility $B_0$ Dipole Convolution
Following Marques & Bowtell (2005, DOI `10.1002/cmr.b.20034`), the local static magnetic field perturbation in k-space with Lorentz-sphere correction is:
$$\Delta B_0(\mathbf{k}) = B_0 \left(\frac{1}{3} - \frac{k_z^2}{|\mathbf{k}|^2}\right) \chi(\mathbf{k}), \qquad D(\mathbf{0}) = 0$$
evaluated on a $2\times$ zero-padded 3D FFT grid to enforce free-space dipole boundary conditions. Analytical invariants verified in unit tests:
1. Inside a uniform sphere of susceptibility $\chi_0$, $\Delta B_{0,\text{in}} = 0$ (exact cancellation of demagnetizing factor $N = 1/3$ and Lorentz cavity $+1/3$).
2. Outside a sphere of radius $R$, $\Delta B_{0,\text{out}}(r, \theta) = \frac{1}{3} B_0 \chi_0 (R/r)^3 (3\cos^2\theta - 1)$.

### 2.3 Readout & EPI Phase-Encode Geometric Shift and SNR Tradeoff
$$\Delta f(\mathbf{x}) = \frac{\gamma}{2\pi}\Delta B_0(\mathbf{x}), \qquad \Delta r_{\text{RO}}(\mathbf{x}) = \frac{\Delta f(\mathbf{x})}{\mathrm{BW}_{\mathrm{px}}} \cdot \Delta x_{\text{RO}}$$
For echo-planar imaging (EPI) with echo spacing $t_{\text{esp}}$, phase-encode steps $N_{\text{PE}}$, and parallel acceleration $R_{\text{PAT}}$:
$$\Delta r_{\text{PE}}(\mathbf{x}) = \Delta f(\mathbf{x}) \cdot \frac{N_{\text{PE}} \cdot t_{\text{esp}}}{R_{\text{PAT}}} \cdot \Delta x_{\text{PE}}$$
Relative signal-to-noise ratio obeys the exact physics scaling:
$$\mathrm{SNR}_{\text{rel}} = \frac{V_{\text{vox}}}{V_{\text{ref}}} \sqrt{\frac{\mathrm{BW}_{\text{ref}}}{\mathrm{BW}_{\mathrm{px}} \cdot R_{\text{PAT}}}}$$

## 3. Synthetic-CT Physics Plausibility & Lorentz-Force ERE Kernel (`sct_ere_auditor.py`)

### 3.1 Synthetic-CT Plausibility Audit
- Stoichiometric piecewise-linear HU $\to$ relative electron density ($\rho_e / \rho_{e,w}$) conversion.
- Cross-checks sCT against MRI intensity-physics priors (and optional ground-truth CT) to flag:
  - `BONE_TO_AIR_SWAP` (cortical bone misclassified $< -400\text{ HU}$)
  - `MISSING_METAL_ARTIFACT` (high-Z metallic implant replaced by soft tissue)
  - `BODY_CONTOUR_TRUNCATION` (MRI tissue signal extending beyond sCT body mask)
- Quantifies maximum AP/LR beam-line water-equivalent path length (WEPL) error [mm] and conservative 6 MV photon dose error ($\approx 0.35\%/\text{mm}$ WEPL).

### 3.2 Lorentz-Force Electron-Return-Effect (ERE) Boundary Risk
Following Raaijmakers et al. (2005, 2008), relativistic secondary Compton electrons of kinetic energy $T_e$ exiting tissue into air at angle $\alpha$ to $\mathbf{B}_0$ curve with transverse gyroradius:
$$p c = \sqrt{T_e(T_e + 2 m_e c^2)}, \qquad r_g = \frac{p \sin\alpha}{e B_0}$$
At $T_e = 1.5\text{ MeV}$ and $\alpha = 90^\circ$: $r_g = 4.33\text{ mm}$ ($1.5\text{ T}$), $18.54\text{ mm}$ ($0.35\text{ T}$), $\infty$ ($0\text{ T}$).
Local tissue–air exit interfaces combine density jump $\Delta\rho_e$, beam exit alignment $\hat{\mathbf{u}}_{\text{beam}}\cdot(-\nabla\rho_e)$, air-cavity clearance return fraction $\eta_{\text{return}} = 1 - \exp(-L_{\text{air}} / (2 r_g))$, and field-strength focusing to estimate single-beam exit ERE dose enhancement [%].

## 4. Formal IEC 60601-2-33 / ASTM F2503 Implant CSP (`implant_safety_csp.py`)

Exact pulse-sequence RF scaling:
$$B_{1+\text{rms}} = \frac{1}{\gamma}\sqrt{\frac{f_{\text{shape}} \sum_j \alpha_j^2}{\mathrm{TR} \cdot \tau_{\text{RF}}}}, \qquad \mathrm{SAR} = c_{\text{anatomy}} B_0^2 (B_{1+\text{rms}})^2$$
Hard abstention gates (`provenance.py`) fire before solver clearance on:
- `MR_UNSAFE_DEVICE_DETECTED`
- `UNSPECIFIED_IMPLANT_CONDITION`
- `IMPLANT_SAR_OR_B1RMS_VIOLATION`
- `UNWRAPPABLE_B0_PHASE`
- `SRS_DISTORTION_EXCEEDS_TOLERANCE`

## 5. Saltelli–Jansen Global Sobol' Sensitivity (`sobol_sensitivity.py`)
Scrambled base-2 Sobol' QMC (`scipy.stats.qmc.Sobol`) with Saltelli (2010) Table 2 Eq. (b) for $S_1$ and Jansen (1999) / Saltelli (2010) Eq. (f) for $S_T$, plus 200-replicate bootstrap 95% CIs.
