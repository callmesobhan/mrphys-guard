"""Exact Saltelli–Jansen Quasi-Monte Carlo (QMC) Global Sobol' Sensitivity Analysis.

Implements the variance-based global sensitivity design and estimators from:
Saltelli, Annoni, Azzini, Campolongo, Ratto & Tarantola (2010),
"Variance based sensitivity analysis of model output. Design and estimator for the total
sensitivity index", Computer Physics Communications 181(2):259-270.
DOI: 10.1016/j.cpc.2009.09.018
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import qmc

from mrphys_guard.distortion_solver import GAMMA_BAR_HZ_PER_T


class SobolParameterSpec(BaseModel):
    """Uniformly distributed uncertain physics parameter for global Sobol' analysis."""

    model_config = ConfigDict(extra="forbid")

    name: str
    unit: str
    low: float
    high: float


class SobolIndexResult(BaseModel):
    """First-order (S_1) and Total-effect (S_T) Sobol' sensitivity indices with 95% CIs."""

    model_config = ConfigDict(extra="forbid")

    parameter: str
    unit: str
    s1: float
    s1_ci95: tuple[float, float]
    st: float
    st_ci95: tuple[float, float]
    interaction_index: float


class SobolAnalysisReport(BaseModel):
    """Complete Saltelli–Jansen QMC Sobol' global sensitivity report."""

    model_config = ConfigDict(extra="forbid")

    output_name: str
    output_unit: str
    n_base_samples: int = Field(..., ge=16)
    total_evaluations: int = Field(..., ge=32)
    output_mean: float
    output_std: float
    output_p05: float
    output_p95: float
    sum_s1: float
    sum_st: float
    indices: list[SobolIndexResult]


@dataclass(frozen=True)
class SaltelliDesign:
    """Saltelli radial QMC sample matrices A, B, and A_B^(i)."""

    a_unit: np.ndarray
    b_unit: np.ndarray
    a_phys: np.ndarray
    b_phys: np.ndarray
    ab_phys: list[np.ndarray]


def build_saltelli_qmc_design(
    parameters: list[SobolParameterSpec],
    n_base_power_of_two: int = 1024,
    seed: int = 2026,
) -> SaltelliDesign:
    """Construct exact Saltelli (2010) scrambled Sobol' QMC sample matrices A, B, and A_B^(i)."""
    if n_base_power_of_two < 16 or (n_base_power_of_two & (n_base_power_of_two - 1)) != 0:
        raise ValueError("n_base_power_of_two must be a power of 2 >= 16 (e.g. 256, 512, 1024).")
    k = len(parameters)
    if k < 2:
        raise ValueError("At least 2 parameters are required for Sobol' sensitivity analysis.")

    sampler = qmc.Sobol(d=2 * k, scramble=True, seed=seed)
    m_exp = int(np.log2(n_base_power_of_two))
    u = sampler.random_base2(m=m_exp)
    a_unit = u[:, :k]
    b_unit = u[:, k:]

    lows = np.array([p.low for p in parameters], dtype=np.float64)
    highs = np.array([p.high for p in parameters], dtype=np.float64)
    span = highs - lows

    a_phys = lows + a_unit * span
    b_phys = lows + b_unit * span
    ab_phys: list[np.ndarray] = []
    for i in range(k):
        mat = a_phys.copy()
        mat[:, i] = b_phys[:, i]
        ab_phys.append(mat)

    return SaltelliDesign(
        a_unit=a_unit,
        b_unit=b_unit,
        a_phys=a_phys,
        b_phys=b_phys,
        ab_phys=ab_phys,
    )


def compute_sobol_saltelli_jansen(
    model_fn: Callable[[np.ndarray], np.ndarray],
    parameters: list[SobolParameterSpec],
    output_name: str,
    output_unit: str,
    n_base: int = 1024,
    n_bootstrap: int = 200,
    seed: int = 2026,
) -> SobolAnalysisReport:
    """Evaluate exact Saltelli (2010) S_1 and Jansen (1999)/Saltelli (2010) S_T Sobol' indices.

    Formulas (Saltelli et al. 2010, Table 2):
        V_i  = (1 / N) * sum_j f(B)_j * (f(A_B^(i))_j - f(A)_j)    -> S_1,i = V_i / Var(Y)
        E_i  = (1 / (2N)) * sum_j (f(A)_j - f(A_B^(i))_j)^2        -> S_T,i = E_i / Var(Y)
    """
    design = build_saltelli_qmc_design(parameters, n_base_power_of_two=n_base, seed=seed)
    k = len(parameters)

    ya = np.asarray(model_fn(design.a_phys), dtype=np.float64).ravel()
    yb = np.asarray(model_fn(design.b_phys), dtype=np.float64).ravel()
    yab = [np.asarray(model_fn(mat), dtype=np.float64).ravel() for mat in design.ab_phys]

    y_all = np.concatenate([ya, yb])
    var_y = float(np.var(y_all, ddof=1))
    if var_y <= 1e-18:
        raise ValueError("Model output variance is effectively zero; cannot compute Sobol' indices.")

    rng = np.random.default_rng(seed + 17)
    boot_idx = rng.integers(0, n_base, size=(n_bootstrap, n_base))

    results: list[SobolIndexResult] = []
    for i, param in enumerate(parameters):
        yi = yab[i]
        # Saltelli 2010 Eq. (b) first-order estimator
        s1_val = float(np.mean(yb * (yi - ya)) / var_y)
        # Jansen 1999 / Saltelli 2010 Eq. (f) total-effect estimator
        st_val = float(0.5 * np.mean((ya - yi) ** 2) / var_y)

        # Bootstrap 95% CIs
        yb_b = yb[boot_idx]
        ya_b = ya[boot_idx]
        yi_b = yi[boot_idx]
        var_b = np.var(np.concatenate([ya_b, yb_b], axis=1), axis=1, ddof=1)
        var_b = np.maximum(var_b, 1e-18)
        s1_boot = np.mean(yb_b * (yi_b - ya_b), axis=1) / var_b
        st_boot = 0.5 * np.mean((ya_b - yi_b) ** 2, axis=1) / var_b

        s1_ci = (float(np.percentile(s1_boot, 2.5)), float(np.percentile(s1_boot, 97.5)))
        st_ci = (float(np.percentile(st_boot, 2.5)), float(np.percentile(st_boot, 97.5)))

        results.append(
            SobolIndexResult(
                parameter=param.name,
                unit=param.unit,
                s1=s1_val,
                s1_ci95=s1_ci,
                st=st_val,
                st_ci95=st_ci,
                interaction_index=float(max(0.0, st_val - s1_val)),
            )
        )

    return SobolAnalysisReport(
        output_name=output_name,
        output_unit=output_unit,
        n_base_samples=n_base,
        total_evaluations=n_base * (k + 2),
        output_mean=float(np.mean(y_all)),
        output_std=float(np.std(y_all, ddof=1)),
        output_p05=float(np.percentile(y_all, 5)),
        output_p95=float(np.percentile(y_all, 95)),
        sum_s1=float(sum(r.s1 for r in results)),
        sum_st=float(sum(r.st for r in results)),
        indices=results,
    )


# ---------------------------------------------------------------------------
# MR-Only Radiotherapy Physics Surrogate Models for Global Sobol' Analysis
# ---------------------------------------------------------------------------


def default_mr_rt_uncertainty_parameters() -> list[SobolParameterSpec]:
    """Standard uncertain physical parameters in 1.5T MR-only radiotherapy simulation."""
    return [
        SobolParameterSpec(name="bandwidth_hz_per_px", unit="Hz/px", low=150.0, high=650.0),
        SobolParameterSpec(name="air_cavity_delta_chi_ppm", unit="ppm", low=4.5, high=11.0),
        SobolParameterSpec(name="gnl_calibration_residual_mm", unit="mm", low=0.10, high=0.75),
        SobolParameterSpec(name="sct_hu_error_hu", unit="HU", low=15.0, high=160.0),
    ]


def mr_target_geometric_error_model(
    x: np.ndarray,
    b0_T: float = 1.5,
    voxel_size_mm: float = 1.0,
    dipole_geometry_factor: float = 0.28,
) -> np.ndarray:
    """Vectorized physical model of total target 3D geometric error [mm].

    Columns of x:
        0: bandwidth_hz_per_px [Hz/px]
        1: air_cavity_delta_chi_ppm [ppm] (effective tissue-air/implant Delta_chi)
        2: gnl_calibration_residual_mm [mm] (residual system GNL error at target)
        3: sct_hu_error_hu [HU] (causes minor sCT-to-MR contour registration shift)
    """
    bw = x[:, 0]
    delta_chi_ppm = x[:, 1]
    gnl_res_mm = x[:, 2]
    sct_hu_err = x[:, 3]

    # Exact susceptibility shift: Delta_B0 = dipole_geom * B0 * Delta_chi
    delta_b0_T = dipole_geometry_factor * b0_T * (delta_chi_ppm * 1e-6)
    delta_f_hz = GAMMA_BAR_HZ_PER_T * delta_b0_T
    disp_b0_mm = (delta_f_hz / bw) * voxel_size_mm

    # Minor registration coupling from sCT contour HU error (~0.001 mm/HU)
    disp_contour_mm = 0.0012 * sct_hu_err

    # Quadrature combination of orthogonal/partially correlated 3D distortion components
    return np.sqrt(disp_b0_mm**2 + gnl_res_mm**2 + disp_contour_mm**2 + 0.35 * disp_b0_mm * gnl_res_mm)


def mr_sct_dosimetric_uncertainty_model(
    x: np.ndarray,
    b0_T: float = 1.5,
    beam_path_bone_mm: float = 45.0,
) -> np.ndarray:
    """Vectorized physical model of MR-only planning target dosimetric uncertainty [%].

    Captures:
    - sCT HU error -> electron density / WEPL error -> 6 MV attenuation error (~0.35%/mm WEPL)
    - Air-cavity susceptibility + ERE coupling (larger cavity -> stronger ERE boundary perturbation)
    - Geometric distortion target shift in steep penumbra (~1.4%/mm gradient)
    """
    delta_chi_ppm = x[:, 1]
    sct_hu_err = x[:, 3]

    # HU error -> relative ED error (~0.00062 per HU) -> WEPL error -> 6 MV dose error
    wepl_err_mm = (0.00062 * sct_hu_err) * beam_path_bone_mm
    dose_err_sct_pct = 0.35 * wepl_err_mm

    # Geometric shift -> penumbra dose perturbation
    geom_err_mm = mr_target_geometric_error_model(x, b0_T=b0_T)
    dose_err_geom_pct = 1.35 * geom_err_mm

    # Air-cavity susceptibility proxy for cavity size -> ERE perturbation at 1.5T
    dose_err_ere_pct = 0.14 * (b0_T / 1.5) * delta_chi_ppm * (1.0 + 0.002 * sct_hu_err)

    return np.sqrt(dose_err_sct_pct**2 + dose_err_geom_pct**2 + dose_err_ere_pct**2)


def run_mr_uncertainty_sobol_suite(
    n_base: int = 1024,
    b0_T: float = 1.5,
    seed: int = 2026,
) -> dict[str, SobolAnalysisReport]:
    """Run paired Saltelli–Jansen Sobol' sensitivity analysis on geometric & dosimetric error."""
    params = default_mr_rt_uncertainty_parameters()
    geom_report = compute_sobol_saltelli_jansen(
        model_fn=lambda arr: mr_target_geometric_error_model(arr, b0_T=b0_T),
        parameters=params,
        output_name="Total Target 3D Geometric Error",
        output_unit="mm",
        n_base=n_base,
        seed=seed,
    )
    dose_report = compute_sobol_saltelli_jansen(
        model_fn=lambda arr: mr_sct_dosimetric_uncertainty_model(arr, b0_T=b0_T),
        parameters=params,
        output_name="sCT & MR-Only Target Dosimetric Uncertainty",
        output_unit="%",
        n_base=n_base,
        seed=seed + 100,
    )
    return {"geometric_error": geom_report, "dosimetric_uncertainty": dose_report}
