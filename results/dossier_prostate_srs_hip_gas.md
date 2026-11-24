# AAPM TG-284 / IEC 60601-2-33 MR-Physics Safety & Distortion Dossier

> **RESEARCH SOFTWARE — NOT A MEDICAL DEVICE.** For computational physics verification only.

- **Scenario ID:** `prostate_srs_hip_gas_1p5t`
- **Title:** 1.5T MR-Only Prostate SRS with Bilateral Ti Hip Prostheses & Rectal Gas
- **Scanner:** 1.5T MR-Linac (High-Field Adaptive RT) ($B_0 = 1.5\text{ T}$, $|dB/dx|_{\max} = 18.5\text{ T/m}$)
- **Implant:** Generic Ti-6Al-4V Bilateral Hip Prosthesis (1.5T/3.0T) (`MR_CONDITIONAL`)
- **Co-Design Status:** **CONVERGED** | **Critic-Verifier Gate:** **APPROVED** (`15/15`)

## Before vs. After Protocol Metrics

| Metric | Baseline Protocol | Co-Designed Protocol | Target / Limit |
| :--- | :---: | :---: | :---: |
| Formal CSP Verdict | `VIOLATION` | `PASS` | `PASS` |
| Max Target Distortion | 1.082 mm | 0.478 mm | <= 0.50 mm |
| Min Relative SNR | 1.744 | 1.131 | >= 0.60 |
| Peak $B_{1+\text{rms}}$ | 3.35 uT | 3.01 uT | <= 3.2 uT |
| Peak Whole-Body SAR | 2.20 W/kg | 1.77 W/kg | <= 2.0 W/kg |
| Cumulative Active Time | 8.5 min | 8.5 min | <= 60.0 min |

## Critic-Verifier Findings

- Formal IEC 60601-2-33 CSP verified: 13/13 PASS.
- SRS geometric fidelity verified: max |Delta_r| = 0.478 mm <= 0.50 mm.
- Diagnostic SNR floor verified: min relSNR = 1.131 >= 0.60.

## Co-Designed Sequence Parameters

| Sequence | Type | BW (Hz/px) | Flip (deg) | Refocus (deg) | TR (ms) | tau_RF (ms) | Slew (T/m/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| T2w_3D_TSE_Pelvis | TSE | 440.0 | 90.0 | 165.0 | 320.4 | 1.49 | 175.0 |
| Dixon_3D_sCT_Anchor | Dixon | 530.0 | 25.0 | 150.0 | 6.5 | 0.70 | 180.0 |

## Cryptographic Provenance

- **Chain Valid:** `True` (2 records)
- **Terminal Record SHA-256:** `e8537bca0285052a7b785074e0379b4b50224f6ce32bc2b0033e087e7500d632`
