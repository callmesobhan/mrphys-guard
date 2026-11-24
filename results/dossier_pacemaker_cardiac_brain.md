# AAPM TG-284 / IEC 60601-2-33 MR-Physics Safety & Distortion Dossier

> **RESEARCH SOFTWARE — NOT A MEDICAL DEVICE.** For computational physics verification only.

- **Scenario ID:** `pacemaker_cardiac_brain_1p5t`
- **Title:** 1.5T Cardiac & Brain MRI Protocol with MR-Conditional Dual-Chamber Pacemaker
- **Scanner:** 1.5T Diagnostic MRI (IEC Normal Mode) ($B_0 = 1.5\text{ T}$, $|dB/dx|_{\max} = 17.0\text{ T/m}$)
- **Implant:** Generic MR-Conditional Dual-Chamber Cardiac Pacemaker (1.5T) (`MR_CONDITIONAL`)
- **Co-Design Status:** **CONVERGED** | **Critic-Verifier Gate:** **APPROVED** (`24/24`)

## Before vs. After Protocol Metrics

| Metric | Baseline Protocol | Co-Designed Protocol | Target / Limit |
| :--- | :---: | :---: | :---: |
| Formal CSP Verdict | `VIOLATION` | `PASS` | `PASS` |
| Max Target Distortion | 0.704 mm | 0.704 mm | <= 1.00 mm |
| Min Relative SNR | 1.091 | 1.091 | >= 0.55 |
| Peak $B_{1+\text{rms}}$ | 3.87 uT | 1.88 uT | <= 2.0 uT |
| Peak Whole-Body SAR | 2.93 W/kg | 0.69 W/kg | <= 2.0 W/kg |
| Cumulative Active Time | 20.0 min | 20.0 min | <= 30.0 min |

## Critic-Verifier Findings

- Formal IEC 60601-2-33 CSP verified: 23/23 PASS.
- Diagnostic SNR floor verified: min relSNR = 1.091 >= 0.55.

## Co-Designed Sequence Parameters

| Sequence | Type | BW (Hz/px) | Flip (deg) | Refocus (deg) | TR (ms) | tau_RF (ms) | Slew (T/m/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Cardiac_Cine_bSSFP | bSSFP | 650.0 | 65.0 | 150.0 | 3.3 | 2.54 | 188.0 |
| Brain_T2_TSE_HighSAR | TSE | 220.0 | 90.0 | 170.0 | 420.0 | 2.56 | 180.0 |
| Brain_3D_MPRAGE_T1 | MPRAGE | 210.0 | 15.0 | 150.0 | 8.5 | 1.20 | 150.0 |

## Cryptographic Provenance

- **Chain Valid:** `True` (2 records)
- **Terminal Record SHA-256:** `4d4686f90db14794c2dc35b13f32ad9ee16e0594e1db9c78bc5997e28bf1810a`
