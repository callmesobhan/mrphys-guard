"""Generate 8 publication-grade 300-DPI PNG + PDF figures for mrphys-guard.

Visual Identity: High-Field Physics / Magnetic Resonance in Medicine (MRM)
Minimalist Indigo-Mist Editorial — pure-white outer canvas (#FFFFFF),
ultra-soft lavender-indigo plot areas (#F6F5FB), razor-thin deep-indigo
left/bottom spines (#2B2754, 0.8pt), clean horizontal-only guide rules
(#E2DFEF, 0.6pt), asymmetrical left-aligned journal headers with a crisp
vertical violet-indigo accent stripe (#4338CA), STIX mathematical typography,
rounded ultramarine lowercase panel badges (a, b, c, d), and a colorblind-safe
High-Field MR palette (Ultramarine, Magenta-Crimson, Cerulean, Jade, Violet,
Marigold).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from mrphys_guard.dicom_io import load_nrrd_volume
from mrphys_guard.distortion_solver import (
    GAMMA_BAR_HZ_PER_T,
    build_pelvic_prostate_susceptibility_phantom,
    compute_3d_distortion_decomposition,
    compute_distortion_jacobian_and_unwarp,
    fit_spherical_harmonic_gnl,
    generate_synthetic_grid_phantom,
    relative_snr_from_bandwidth,
    unwrap_3d_b0_phase_dct,
)
from mrphys_guard.dvh_margin_certifier import (
    build_clinical_srs_dose_grid,
    build_rectal_wall_oar_mask,
    certify_mronly_dvh_and_anisotropic_margins,
)
from mrphys_guard.harness.agent_orchestrator import (
    MRSequenceCoDesignAgent,
    build_pacemaker_cardiac_brain_scenario,
    build_prostate_srs_hip_gas_scenario,
)
from mrphys_guard.ir import MRSequenceSpec
from mrphys_guard.sct_ere_auditor import (
    audit_synthetic_ct,
    compute_ere_boundary_risk,
    compute_multibeam_ere_cancellation,
    electron_gyroradius_mm,
    inject_sct_failure_modes,
)
from mrphys_guard.sobol_sensitivity import run_mr_uncertainty_sobol_suite

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"
PUBLIC_DATA_DIR = REPO_ROOT / "data" / "public_ctmr_brain"

# ---------------------------------------------------------------------------
# Signature High-Field MRM Minimalist Indigo-Mist Palette (Colorblind-Safe)
# ---------------------------------------------------------------------------
MRM_ULTRAMARINE = "#2E3192"  # 1.5T MR-Linac / Primary
MRM_MAGENTA = "#C2185B"  # 3.0T High-Field / Alert / Violation
MRM_MAGENTA_DEEP = "#7D103D"  # Deep Crimson-Magenta for worst-case curves
MRM_CERULEAN = "#0288D1"  # 0.35T Low-Field / Secondary
MRM_JADE = "#00897B"  # Co-Designed / Safe / Pass
MRM_JADE_DEEP = "#00695C"  # Deep Jade for readable callout text
MRM_VIOLET = "#7E57C2"  # Envelope / Harmonic / Auxiliary
MRM_MARIGOLD = "#E08E00"  # Caution / Threshold
MRM_MARIGOLD_DEEP = "#995F00"  # High-contrast Marigold text on mist surface

# Surface, Typography & Editorial Chrome
MRM_CANVAS = "#FFFFFF"  # Pure-white outer canvas
MRM_MIST = "#F6F5FB"  # Ultra-soft cool lavender-indigo plot area
MRM_SPINE = "#2B2754"  # Razor-thin deep-indigo left/bottom spines
MRM_GRID = "#E2DFEF"  # Clean horizontal-only guide rules
MRM_ACCENT = "#4338CA"  # Crisp vertical violet-indigo header accent stripe
MRM_BADGE = "#3730A3"  # Ultramarine/violet-indigo panel label badge
MRM_INK = "#1B183B"  # Primary editorial text
MRM_MUTED = "#5C577A"  # Secondary subtitle & provenance footer text
MRM_BORDER = "#D4CFE8"  # Subtle legend / callout border
MRM_DARK_FIELD = "#121024"  # Deep midnight-indigo masked exterior for 2D maps
MRM_LAVENDER_MASK = "#ECEAF6"  # Soft cool lavender mask for light sequential maps

# Custom Mako-style cool-indigo-to-cyan sequential colormap (GNL distortion)
MRM_MAKO_CMAP = LinearSegmentedColormap.from_list(
    "mrm_mako",
    [
        "#0E0D26",
        "#1E225D",
        "#2E3E8E",
        "#1D74B0",
        "#1EA8A8",
        "#7BE0CE",
        "#E6FBF6",
    ],
)

# Custom high-contrast violet-magenta-rose-gold colormap for ERE overlays on grayscale CT
MRM_ERE_CMAP = LinearSegmentedColormap.from_list(
    "mrm_ere_rdpu",
    [
        "#3B1D74",
        "#7E3AA6",
        "#C2185B",
        "#F04D75",
        "#FF9E7D",
        "#FFE599",
    ],
)


def configure_journal_style() -> None:
    """Configure global Matplotlib rcParams for MRM Minimalist Indigo-Mist."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stix",
            "font.size": 9.3,
            "axes.titlesize": 9.8,
            "axes.titleweight": "bold",
            "axes.titlecolor": MRM_INK,
            "axes.labelsize": 9.1,
            "axes.labelweight": "bold",
            "axes.labelcolor": MRM_INK,
            "axes.edgecolor": MRM_SPINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "xtick.color": MRM_SPINE,
            "ytick.color": MRM_SPINE,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "grid.color": MRM_GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
            "legend.fontsize": 7.8,
            "legend.facecolor": MRM_CANVAS,
            "legend.edgecolor": MRM_BORDER,
            "legend.framealpha": 0.94,
            "legend.fancybox": True,
            "figure.facecolor": MRM_CANVAS,
            "axes.facecolor": MRM_MIST,
            "savefig.facecolor": MRM_CANVAS,
            "savefig.dpi": 300,
        }
    )


def _style_1d_axes(ax: plt.Axes) -> None:
    """Apply lavender-mist surface, razor-thin indigo spines, and horizontal-only rules."""
    ax.set_facecolor(MRM_MIST)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MRM_SPINE)
    ax.spines["bottom"].set_color(MRM_SPINE)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_axisbelow(True)
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", color=MRM_GRID, linewidth=0.6, alpha=1.0)


def _style_image_axes(ax: plt.Axes, bg_color: str = MRM_DARK_FIELD) -> None:
    """Apply razor-thin left/bottom indigo spines to 2D spatial map axes."""
    ax.set_facecolor(bg_color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MRM_SPINE)
    ax.spines["bottom"].set_color(MRM_SPINE)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(False)


def _style_colorbar(cbar: plt.Colorbar, label: str) -> None:
    """Style colorbar with razor-thin deep-indigo frame and crisp STIX label."""
    cbar.outline.set_edgecolor(MRM_SPINE)
    cbar.outline.set_linewidth(0.65)
    cbar.ax.tick_params(labelsize=8.0, colors=MRM_SPINE, width=0.65, length=3.0)
    cbar.set_label(label, fontweight="bold", fontsize=8.8, color=MRM_INK, labelpad=6)


def _add_mrm_header(
    fig: plt.Figure,
    title: str,
    subtitle: str,
    left: float = 0.065,
    top_y: float = 0.968,
) -> None:
    """Render an asymmetrical left-aligned MRM journal header with violet-indigo accent stripe."""
    stripe_x = max(0.015, left - 0.016)
    stripe_h = 0.044 if fig.get_figheight() > 7.5 else 0.072
    stripe_y0 = top_y - stripe_h + 0.005

    stripe = mpatches.FancyBboxPatch(
        (stripe_x, stripe_y0),
        0.0042,
        stripe_h,
        boxstyle="round,pad=0.0006,rounding_size=0.0018",
        transform=fig.transFigure,
        facecolor=MRM_ACCENT,
        edgecolor="none",
        clip_on=False,
        zorder=30,
    )
    fig.add_artist(stripe)

    fig.text(
        left,
        top_y,
        title,
        ha="left",
        va="top",
        fontsize=12.0,
        fontweight="bold",
        color=MRM_INK,
    )
    sub_offset = 0.025 if fig.get_figheight() > 7.5 else 0.043
    fig.text(
        left,
        top_y - sub_offset,
        subtitle,
        ha="left",
        va="top",
        fontsize=8.2,
        color=MRM_MUTED,
    )


def _set_panel_header(
    ax: plt.Axes,
    letter: str,
    title: str,
    subtitle: str | None = None,
) -> None:
    """Place a clean rounded ultramarine lowercase badge (a, b, c, d) beside a left-aligned title."""
    fig = ax.figure
    y_main = 1.095 if subtitle else 1.038

    ax.text(
        0.0,
        y_main,
        f" {letter} ",
        transform=ax.transAxes,
        fontsize=8.8,
        fontweight="bold",
        color=MRM_CANVAS,
        ha="left",
        va="bottom",
        zorder=25,
        clip_on=False,
        bbox={
            "boxstyle": "round,pad=0.18,rounding_size=0.28",
            "facecolor": MRM_BADGE,
            "edgecolor": "none",
        },
    )

    title_trans = mtransforms.offset_copy(
        ax.transAxes, fig=fig, x=22.0, y=0.0, units="points"
    )
    ax.text(
        0.0,
        y_main,
        title,
        transform=title_trans,
        fontsize=9.6,
        fontweight="bold",
        color=MRM_INK,
        ha="left",
        va="bottom",
        zorder=25,
        clip_on=False,
    )

    if subtitle:
        ax.text(
            0.0,
            1.028,
            subtitle,
            transform=ax.transAxes,
            fontsize=7.9,
            color=MRM_MUTED,
            ha="left",
            va="bottom",
            zorder=25,
            clip_on=False,
        )


def _add_scalebar(
    ax: plt.Axes,
    length_mm: float,
    x_right_mm: float,
    y_bottom_mm: float,
    color: str = "white",
    light_bg: bool = False,
) -> None:
    x0 = x_right_mm - length_mm
    ax.plot(
        [x0, x_right_mm],
        [y_bottom_mm, y_bottom_mm],
        color=color,
        lw=2.6,
        solid_capstyle="butt",
        zorder=15,
    )
    bg = (0.98, 0.98, 1.0, 0.88) if light_bg else (0.07, 0.06, 0.14, 0.65)
    ax.text(
        0.5 * (x0 + x_right_mm),
        y_bottom_mm + 4.0,
        f"{int(length_mm)} mm",
        color=color,
        fontsize=7.8,
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=15,
        bbox={"boxstyle": "round,pad=0.14", "facecolor": bg, "edgecolor": "none"},
    )


def _save_fig(fig: plt.Figure, stem: str, footer: str, left: float = 0.065) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rule_y = 0.030 if fig.get_figheight() > 7.5 else 0.044
    rule = mlines.Line2D(
        [left, 0.96],
        [rule_y, rule_y],
        transform=fig.transFigure,
        color=MRM_GRID,
        linewidth=0.75,
        clip_on=False,
    )
    fig.add_artist(rule)
    fig.text(
        left,
        0.010,
        footer,
        ha="left",
        va="bottom",
        fontsize=7.1,
        color=MRM_MUTED,
        style="italic",
    )
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    print(f"  Saved {stem}.png and {stem}.pdf")


# ---------------------------------------------------------------------------
# Figure 1: 3D Geometric Distortion Decomposition
# ---------------------------------------------------------------------------


def make_fig1_3d_distortion() -> None:
    shape = (80, 80, 48)
    vox = (2.0, 2.0, 2.5)
    phantom = build_pelvic_prostate_susceptibility_phantom(
        shape=shape, voxel_size_mm=vox, include_rectal_gas=True, include_bilateral_hip_prostheses=True
    )
    true_pts, meas_pts, _ = generate_synthetic_grid_phantom(
        extent_mm=180.0, spacing_mm=30.0, max_edge_distortion_mm=2.8, noise_sigma_mm=0.0
    )
    gnl_model = fit_spherical_harmonic_gnl(true_pts, meas_pts, lmax=5)
    # High-bandwidth 1.5T MR-only SRS pelvic sequence (500 Hz/px)
    seq = MRSequenceSpec(
        name="Pelvis_1p5T_BW500",
        b0_T=1.5,
        bandwidth_hz_per_px=500.0,
        voxel_size_mm=vox,
        tr_ms=15.0,
        te_ms=4.5,
    )
    decomp, fields = compute_3d_distortion_decomposition(
        chi_si=phantom["chi_si"],
        sequence=seq,
        gnl_model=gnl_model,
        target_roi_mask=phantom["prostate_mask"],
        body_mask=phantom["body_mask"],
    )

    kz = shape[2] // 2
    half_x = shape[0] * vox[0] / 2.0
    half_y = shape[1] * vox[1] / 2.0
    extent = [-half_x, half_x, -half_y, half_y]

    mri_sl = phantom["mri_t1"][:, :, kz].T
    gnl_sl = fields["gnl_mag_mm"][:, :, kz].T
    ppm_sl = fields["delta_b0_ppm"][:, :, kz].T
    tot_sl = fields["total_mag_mm"][:, :, kz].T
    prostate_sl = phantom["prostate_mask"][:, :, kz].T
    body_sl = phantom["body_mask"][:, :, kz].T

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 9.8), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.93, top=0.875, bottom=0.085, wspace=0.30, hspace=0.33)
    _add_mrm_header(
        fig,
        r"3D MRI Geometric Distortion Decomposition — 1.5T MR-Only Prostate SRS",
        r"High-Field Physics Decomposition  ·  $\mathrm{BW}_{\mathrm{px}} = 500\;\mathrm{Hz/px}$  ·  "
        r"Exact Lorentz-Sphere $k$-Space Dipole + $\ell \leq 5$ Maxwell Harmonics",
        left=0.075,
        top_y=0.972,
    )

    # Panel a: Axial T1w MRI + ROIs (crisp grayscale anatomy)
    ax = axes[0, 0]
    _style_image_axes(ax, bg_color="#000000")
    ax.imshow(mri_sl, origin="lower", extent=extent, cmap="gray", vmin=0.0, vmax=0.95)
    ax.contour(prostate_sl, levels=[0.5], colors=["#00C4A7"], linewidths=2.0, extent=extent)
    ax.contour(
        phantom["rectal_gas_mask"][:, :, kz].T,
        levels=[0.5],
        colors=["#38BDF8"],
        linewidths=1.8,
        linestyles="--",
        extent=extent,
    )
    ax.contour(
        phantom["metal_mask"][:, :, kz].T,
        levels=[0.5],
        colors=[MRM_MARIGOLD],
        linewidths=1.9,
        extent=extent,
    )
    _set_panel_header(ax, "a", r"Axial $T_1$-w MRI Anatomy & Landmarks")
    ax.set_xlabel(r"Left–Right $x\;\mathrm{[mm]}$")
    ax.set_ylabel(r"Anterior–Posterior $y\;\mathrm{[mm]}$")
    _add_scalebar(ax, 20.0, half_x - 8.0, -half_y + 10.0, color="white")
    handles = [
        mpatches.Patch(edgecolor="#00C4A7", facecolor="none", lw=2.0, label="Prostate CTV"),
        mpatches.Patch(edgecolor="#38BDF8", facecolor="none", lw=1.8, ls="--", label="Rectal Gas"),
        mpatches.Patch(edgecolor=MRM_MARIGOLD, facecolor="none", lw=1.9, label="Ti Hip Prostheses"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=7.6, framealpha=0.92)

    # Panel b: System GNL map (mako-style cool-indigo-to-cyan)
    ax = axes[0, 1]
    _style_image_axes(ax, bg_color=MRM_DARK_FIELD)
    gnl_vmax = max(0.35, float(np.max(gnl_sl)) * 1.05)
    im_b = ax.imshow(gnl_sl, origin="lower", extent=extent, cmap=MRM_MAKO_CMAP, vmin=0.0, vmax=gnl_vmax)
    ax.contour(body_sl, levels=[0.5], colors="#E6FBF6", linewidths=1.2, linestyles=":", extent=extent)
    _set_panel_header(
        ax,
        "b",
        rf"System GNL $|\Delta\mathbf{{r}}_{{\mathrm{{GNL}}}}|$ ($\ell \leq 5$, Max {decomp.gnl_max_mm:.2f} mm)",
    )
    ax.set_xlabel(r"Left–Right $x\;\mathrm{[mm]}$")
    ax.set_ylabel(r"Anterior–Posterior $y\;\mathrm{[mm]}$")
    cbar = fig.colorbar(im_b, ax=ax, fraction=0.046, pad=0.04)
    _style_colorbar(cbar, r"$|\Delta\mathbf{r}_{\mathrm{GNL}}|\;\mathrm{[mm]}$")
    _add_scalebar(ax, 20.0, half_x - 8.0, -half_y + 10.0, color="white")

    # Panel c: Susceptibility Delta B0 (ppm) with twilight_shifted
    ax = axes[1, 0]
    _style_image_axes(ax, bg_color=MRM_DARK_FIELD)
    ppm_masked = np.where(body_sl, ppm_sl, np.nan)
    im_c = ax.imshow(ppm_masked, origin="lower", extent=extent, cmap="twilight_shifted", vmin=-8.0, vmax=8.0)
    ax.contour(prostate_sl, levels=[0.5], colors=["#00D4B4"], linewidths=1.9, extent=extent)
    _set_panel_header(ax, "c", r"Patient Susceptibility $\Delta B_0$ ($k$-Space Dipole)")
    ax.set_xlabel(r"Left–Right $x\;\mathrm{[mm]}$")
    ax.set_ylabel(r"Anterior–Posterior $y\;\mathrm{[mm]}$")
    cbar = fig.colorbar(im_c, ax=ax, fraction=0.046, pad=0.04)
    _style_colorbar(cbar, r"$\Delta B_0\;\mathrm{[ppm]}$")
    _add_scalebar(ax, 20.0, half_x - 8.0, -half_y + 10.0, color="white")

    # Panel d: Total 3D displacement |Delta r| with RdPu colormap
    ax = axes[1, 1]
    _style_image_axes(ax, bg_color=MRM_LAVENDER_MASK)
    tot_masked = np.where(body_sl, tot_sl, np.nan)
    im_d = ax.imshow(tot_masked, origin="lower", extent=extent, cmap="RdPu", vmin=0.0, vmax=3.5)
    ax.contour(tot_sl * body_sl, levels=[0.5], colors=[MRM_ULTRAMARINE], linewidths=1.7, extent=extent)
    ax.contour(
        tot_sl * body_sl,
        levels=[1.0],
        colors=[MRM_MARIGOLD],
        linewidths=1.8,
        linestyles="--",
        extent=extent,
    )
    ax.contour(prostate_sl, levels=[0.5], colors=[MRM_JADE], linewidths=2.0, extent=extent)
    _set_panel_header(
        ax,
        "d",
        rf"Total $|\Delta\mathbf{{r}}|$ Overlay (CTV P95 = {decomp.target_roi_p95_mm:.2f} mm)",
    )
    ax.set_xlabel(r"Left–Right $x\;\mathrm{[mm]}$")
    ax.set_ylabel(r"Anterior–Posterior $y\;\mathrm{[mm]}$")
    cbar = fig.colorbar(im_d, ax=ax, fraction=0.046, pad=0.04)
    _style_colorbar(cbar, r"Total $|\Delta\mathbf{r}|\;\mathrm{[mm]}$")
    _add_scalebar(ax, 20.0, half_x - 8.0, -half_y + 10.0, color=MRM_SPINE, light_bg=True)
    h_d = [
        mpatches.Patch(edgecolor=MRM_ULTRAMARINE, facecolor="none", lw=1.7, label="0.5 mm SRS Limit"),
        mpatches.Patch(edgecolor=MRM_MARIGOLD, facecolor="none", lw=1.8, ls="--", label="1.0 mm Contour"),
        mpatches.Patch(edgecolor=MRM_JADE, facecolor="none", lw=2.0, label="Prostate CTV"),
    ]
    ax.legend(handles=h_d, loc="lower left", fontsize=7.6, framealpha=0.94)

    _save_fig(
        fig,
        "fig1_3d_distortion_decomposition",
        "mrphys-guard v0.3.0  |  Synthetic 1.5T Pelvic SRS Phantom (SYNTHETIC_PHYSICS_FIXTURE)  |  "
        "Exact Lorentz-sphere k-space dipole + l<=5 Maxwell harmonics  |  Research Software — NOT A MEDICAL DEVICE",
        left=0.075,
    )


# ---------------------------------------------------------------------------
# Figure 2: Receiver Bandwidth vs. Susceptibility Distortion & Relative SNR Pareto
# ---------------------------------------------------------------------------


def make_fig2_bandwidth_pareto() -> None:
    bw = np.linspace(100.0, 900.0, 400)
    peak_ppm = 3.1
    dx_ro = 1.0
    gnl_res = 0.16

    def total_disp(b0: float, bw_arr: np.ndarray) -> np.ndarray:
        df = GAMMA_BAR_HZ_PER_T * (peak_ppm * 1e-6) * b0
        return np.hypot((df / bw_arr) * dx_ro, gnl_res)

    d_1p5 = total_disp(1.5, bw)
    d_3p0 = total_disp(3.0, bw)
    d_0p35 = total_disp(0.35, bw)
    snr = np.array([relative_snr_from_bandwidth(float(b)) for b in bw])

    feasible = (d_1p5 <= 0.50) & (snr >= 0.60)
    bw_feas_min = float(bw[feasible][0])
    bw_feas_max = float(bw[feasible][-1])

    agent_res = MRSequenceCoDesignAgent().run(build_prostate_srs_hip_gas_scenario())
    base_bw = build_prostate_srs_hip_gas_scenario().initial_sequences[0].bandwidth_hz_per_px
    opt_bw = agent_res.optimized_sequences[0].bandwidth_hz_per_px
    base_d = float(total_disp(1.5, np.array([base_bw]))[0])
    opt_d = float(total_disp(1.5, np.array([opt_bw]))[0])
    base_snr = relative_snr_from_bandwidth(base_bw)
    opt_snr = relative_snr_from_bandwidth(opt_bw)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.4), constrained_layout=False)
    fig.subplots_adjust(left=0.068, right=0.955, top=0.79, bottom=0.16, wspace=0.35)
    _add_mrm_header(
        fig,
        r"Receiver Bandwidth ($\mathrm{BW}_{\mathrm{px}}$) Tradeoff: Susceptibility Distortion vs. Relative SNR",
        r"Multi-Field Physics Scaling  ·  $|\Delta\mathbf{r}_{B_0}| \propto B_0 / \mathrm{BW}_{\mathrm{px}}$  ·  "
        r"$\mathrm{SNR} \propto 1/\sqrt{\mathrm{BW}_{\mathrm{px}}}$  ·  0.35T / 1.5T / 3.0T",
        left=0.068,
        top_y=0.965,
    )

    # Panel a
    _style_1d_axes(ax1)
    ax1.axvspan(bw_feas_min, bw_feas_max, color=MRM_JADE, alpha=0.13)
    (l1,) = ax1.plot(bw, d_1p5, color=MRM_ULTRAMARINE, lw=2.4, label=r"1.5T Target $|\Delta\mathbf{r}|$ ($\propto 1/\mathrm{BW}$)")
    (l2,) = ax1.plot(bw, d_3p0, color=MRM_MAGENTA, lw=2.1, ls="--", label=r"3.0T Target $|\Delta\mathbf{r}|$ ($\propto 1/\mathrm{BW}$)")
    ax1.axhline(0.50, color=MRM_SPINE, lw=1.2, ls=":")
    ax1.scatter([base_bw], [base_d], color=MRM_MAGENTA, s=74, zorder=6, edgecolor=MRM_CANVAS, linewidth=1.1)
    ax1.scatter([opt_bw], [opt_d], color=MRM_JADE, s=82, marker="D", zorder=6, edgecolor=MRM_CANVAS, linewidth=1.1)
    ax1.annotate(
        f"Baseline\n({base_bw:.0f} Hz/px, {base_d:.2f} mm)",
        xy=(base_bw, base_d),
        xytext=(base_bw + 95, base_d + 0.62),
        fontsize=7.8,
        fontweight="bold",
        color=MRM_MAGENTA,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": MRM_CANVAS, "edgecolor": MRM_MAGENTA, "lw": 0.9, "alpha": 0.95},
        arrowprops={"arrowstyle": "->", "color": MRM_MAGENTA, "lw": 1.3},
    )
    ax1.annotate(
        f"Co-Designed\n({opt_bw:.0f} Hz/px, {opt_d:.2f} mm)",
        xy=(opt_bw, opt_d),
        xytext=(opt_bw + 85, opt_d + 0.64),
        fontsize=7.8,
        fontweight="bold",
        color=MRM_JADE_DEEP,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": MRM_CANVAS, "edgecolor": MRM_JADE, "lw": 0.9, "alpha": 0.95},
        arrowprops={"arrowstyle": "->", "color": MRM_JADE, "lw": 1.3},
    )
    ax1.set_xlabel(r"Receiver Bandwidth $\mathrm{BW}_{\mathrm{px}}\;\mathrm{[Hz/px]}$")
    ax1.set_ylabel(r"Target 3D Distortion $|\Delta\mathbf{r}|\;\mathrm{[mm]}$", color=MRM_ULTRAMARINE)
    ax1.set_ylim(0.0, 2.8)
    ax1.set_xlim(100, 900)

    ax1_r = ax1.twinx()
    ax1_r.spines["top"].set_visible(False)
    ax1_r.spines["left"].set_visible(False)
    ax1_r.spines["right"].set_color(MRM_MARIGOLD)
    ax1_r.spines["right"].set_linewidth(0.85)
    ax1_r.tick_params(colors=MRM_MARIGOLD_DEEP, labelsize=8.3, width=0.75)
    (l3,) = ax1_r.plot(bw, snr, color=MRM_MARIGOLD, lw=2.2, ls="-.", label=r"Relative SNR ($\propto 1/\sqrt{\mathrm{BW}}$)")
    ax1_r.axhline(0.60, color=MRM_MARIGOLD, lw=1.1, ls=":", alpha=0.85)
    ax1_r.set_ylabel("Relative SNR (Ref = 250 Hz/px)", color=MRM_MARIGOLD_DEEP)
    ax1_r.set_ylim(0.4, 1.75)
    ax1_r.grid(False)

    ax1.legend(
        handles=[
            l1,
            l2,
            l3,
            mpatches.Patch(color=MRM_JADE, alpha=0.22, label=f"Feasible SRS ({bw_feas_min:.0f}–{bw_feas_max:.0f} Hz/px)"),
        ],
        loc="upper right",
        fontsize=7.4,
    )
    _set_panel_header(ax1, "a", "Bandwidth Scaling & Feasible SRS Window")

    # Panel b
    _style_1d_axes(ax2)
    ax2.fill_between([0.0, 0.50], [0.60, 0.60], [1.7, 1.7], color=MRM_JADE, alpha=0.13, label="SRS Feasible Region")
    ax2.plot(d_0p35, snr, color=MRM_CERULEAN, lw=2.3, label="0.35T MR-Linac")
    ax2.plot(d_1p5, snr, color=MRM_ULTRAMARINE, lw=2.5, label="1.5T MR-Linac")
    ax2.plot(d_3p0, snr, color=MRM_MAGENTA, lw=2.2, ls="--", label="3.0T MR-Simulator")
    ax2.axvline(0.50, color=MRM_SPINE, lw=1.2, ls=":", label="0.50 mm SRS Limit")
    ax2.axhline(0.60, color=MRM_MARIGOLD, lw=1.2, ls=":", label="SNR Floor (0.60)")
    ax2.scatter([base_d], [base_snr], color=MRM_MAGENTA, s=74, zorder=6, edgecolor=MRM_CANVAS, linewidth=1.1)
    ax2.scatter([opt_d], [opt_snr], color=MRM_JADE, s=82, marker="D", zorder=6, edgecolor=MRM_CANVAS, linewidth=1.1)
    ax2.annotate(
        "",
        xy=(opt_d + 0.02, opt_snr + 0.03),
        xytext=(base_d - 0.03, base_snr - 0.01),
        arrowprops={"arrowstyle": "->", "color": MRM_INK, "lw": 1.5, "connectionstyle": "arc3,rad=0.22"},
    )
    ax2.text(
        0.76,
        1.14,
        "Agent Co-Design",
        fontsize=7.9,
        fontweight="bold",
        color=MRM_INK,
        ha="center",
        bbox={"boxstyle": "round,pad=0.20", "facecolor": MRM_CANVAS, "edgecolor": MRM_BORDER, "alpha": 0.95},
    )
    ax2.set_xlim(0.1, 2.2)
    ax2.set_ylim(0.45, 1.65)
    ax2.set_xlabel(r"Target 3D Geometric Distortion $|\Delta\mathbf{r}|\;\mathrm{[mm]}$")
    ax2.set_ylabel("Relative SNR (Ref = 250 Hz/px)")
    ax2.legend(loc="upper right", fontsize=7.4)
    _set_panel_header(ax2, "b", "Multi-Field Distortion–SNR Pareto Frontier")

    _save_fig(
        fig,
        "fig2_bandwidth_snr_distortion_pareto",
        "mrphys-guard v0.3.0  |  Peak ΔB0 = 3.1 ppm, residual GNL = 0.16 mm, 1.0 mm readout voxel  |  "
        "Exact 1/BW & 1/√BW physics laws  |  Research Software — NOT A MEDICAL DEVICE",
        left=0.068,
    )


# ---------------------------------------------------------------------------
# Figure 3: sCT Physics Plausibility Audit & 1.5T vs. 0.35T ERE Boundary Risk
# ---------------------------------------------------------------------------


def make_fig3_sct_and_ere() -> None:
    shape = (80, 80, 48)
    vox = (2.0, 2.0, 2.5)
    phantom = build_pelvic_prostate_susceptibility_phantom(shape=shape, voxel_size_mm=vox)
    sct_hu = inject_sct_failure_modes(
        ct_hu=phantom["ct_hu"],
        bone_mask=phantom["bone_mask"],
        metal_mask=phantom["metal_mask"],
        body_mask=phantom["body_mask"],
        swap_bone_fraction=0.35,
        drop_metal_to_tissue=True,
        truncate_body_border_voxels=2,
        hu_noise_sigma=15.0,
    )
    sct_report, sct_maps = audit_synthetic_ct(
        mri_volume=phantom["mri_t1"],
        sct_hu=sct_hu,
        voxel_size_mm=vox,
        ground_truth_ct_hu=phantom["ct_hu"],
        body_mask=phantom["body_mask"],
    )

    ere_1p5, maps_1p5 = compute_ere_boundary_risk(
        ct_or_sct_hu=phantom["ct_hu"], voxel_size_mm=vox, b0_T=1.5, body_mask=phantom["body_mask"]
    )
    ere_0p35, maps_0p35 = compute_ere_boundary_risk(
        ct_or_sct_hu=phantom["ct_hu"], voxel_size_mm=vox, b0_T=0.35, body_mask=phantom["body_mask"]
    )

    b0_grid = [0.0, 0.35, 1.5, 3.0]
    pelvis_peaks = []
    for b0 in b0_grid:
        r_p, _ = compute_ere_boundary_risk(
            ct_or_sct_hu=phantom["ct_hu"], voxel_size_mm=vox, b0_T=b0, body_mask=phantom["body_mask"]
        )
        pelvis_peaks.append(r_p.peak_dose_enhancement_pct)

    brain_peaks = []
    ct_path = PUBLIC_DATA_DIR / "CT-brain.nrrd"
    has_brain = ct_path.exists()
    if has_brain:
        ct_vol = load_nrrd_volume(ct_path, modality="CT")
        ct_ds = ct_vol.array[::2, ::2, ::2]
        vox_ds = tuple(v * 2.0 for v in ct_vol.voxel_size_mm)
        for b0 in b0_grid:
            r_b, _ = compute_ere_boundary_risk(ct_or_sct_hu=ct_ds, voxel_size_mm=vox_ds, b0_T=b0)
            brain_peaks.append(r_b.peak_dose_enhancement_pct)

    kz = shape[2] // 2
    half_x = shape[0] * vox[0] / 2.0
    half_y = shape[1] * vox[1] / 2.0
    extent = [-half_x, half_x, -half_y, half_y]

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.9), constrained_layout=False)
    fig.subplots_adjust(left=0.072, right=0.935, top=0.875, bottom=0.088, wspace=0.30, hspace=0.34)
    _add_mrm_header(
        fig,
        r"Synthetic-CT Physics Plausibility Audit & Lorentz-Force ERE Boundary Hotspots",
        r"MR-Linac Interface Dosimetry  ·  Bone/Metal/Contour Failure Detection & Relativistic Electron Gyroradius $r_g$ Scaling (0T–3.0T)",
        left=0.072,
        top_y=0.972,
    )

    # Panel a: sCT HU Error Map with PuOr diverging colormap
    ax = axes[0, 0]
    _style_image_axes(ax, bg_color=MRM_MIST)
    err_sl = sct_maps["hu_error_map"][:, :, kz].T
    im_a = ax.imshow(err_sl, origin="lower", extent=extent, cmap="PuOr", vmin=-1800, vmax=1800)
    ax.contour(sct_maps["bone_to_air_mask"][:, :, kz].T, levels=[0.5], colors=[MRM_MAGENTA], linewidths=1.8, extent=extent)
    ax.contour(sct_maps["missing_metal_mask"][:, :, kz].T, levels=[0.5], colors=[MRM_MARIGOLD], linewidths=1.8, extent=extent)
    ax.contour(sct_maps["truncation_mask"][:, :, kz].T, levels=[0.5], colors=[MRM_JADE], linewidths=1.5, extent=extent)
    _set_panel_header(
        ax,
        "a",
        rf"sCT $-$ GT CT $\Delta\mathrm{{HU}}$ (Max WEPL Err = {sct_report.max_wepl_error_mm:.1f} mm)",
    )
    ax.set_xlabel(r"Left–Right $x\;\mathrm{[mm]}$")
    ax.set_ylabel(r"Anterior–Posterior $y\;\mathrm{[mm]}$")
    cbar = fig.colorbar(im_a, ax=ax, fraction=0.046, pad=0.04)
    _style_colorbar(cbar, r"$\Delta\mathrm{HU}$ (sCT $-$ GT)")
    _add_scalebar(ax, 20.0, half_x - 8.0, -half_y + 10.0, color=MRM_SPINE, light_bg=True)
    h_a = [
        mpatches.Patch(edgecolor=MRM_MAGENTA, facecolor="none", lw=1.8, label=f"Bone→Air Swap ({sct_report.bone_to_air_swap_voxels} vox)"),
        mpatches.Patch(edgecolor=MRM_MARIGOLD, facecolor="none", lw=1.8, label=f"Missing Metal ({sct_report.missing_metal_voxels} vox)"),
        mpatches.Patch(edgecolor=MRM_JADE, facecolor="none", lw=1.5, label=f"Truncated Rim ({sct_report.contour_truncation_voxels} vox)"),
    ]
    ax.legend(handles=h_a, loc="lower left", fontsize=7.1, framealpha=0.94)

    # Panel b: 1.5T ERE Hotspot Map on grayscale CT anatomy
    ax = axes[0, 1]
    _style_image_axes(ax, bg_color="#000000")
    ct_sl = phantom["ct_hu"][:, :, kz].T
    ax.imshow(ct_sl, origin="lower", extent=extent, cmap="gray", vmin=-1000, vmax=1000)
    ere15_sl = np.ma.masked_where(
        maps_1p5["ere_enhancement_pct"][:, :, kz].T < 2.0,
        maps_1p5["ere_enhancement_pct"][:, :, kz].T,
    )
    im_b = ax.imshow(ere15_sl, origin="lower", extent=extent, cmap=MRM_ERE_CMAP, vmin=0.0, vmax=42.0, alpha=0.92)
    _set_panel_header(
        ax,
        "b",
        rf"1.5T ERE Hotspots (Peak +{ere_1p5.peak_dose_enhancement_pct:.1f}%, $r_g$ = {ere_1p5.gyroradius_mm:.1f} mm)",
    )
    ax.set_xlabel(r"Left–Right $x\;\mathrm{[mm]}$")
    ax.set_ylabel(r"Anterior–Posterior $y\;\mathrm{[mm]}$")
    cbar = fig.colorbar(im_b, ax=ax, fraction=0.046, pad=0.04)
    _style_colorbar(cbar, "ERE Dose Spike [%]")
    _add_scalebar(ax, 20.0, half_x - 8.0, -half_y + 10.0, color="white")

    # Panel c: 0.35T ERE Hotspot Map on grayscale CT anatomy
    ax = axes[1, 0]
    _style_image_axes(ax, bg_color="#000000")
    ax.imshow(ct_sl, origin="lower", extent=extent, cmap="gray", vmin=-1000, vmax=1000)
    ere035_sl = np.ma.masked_where(
        maps_0p35["ere_enhancement_pct"][:, :, kz].T < 2.0,
        maps_0p35["ere_enhancement_pct"][:, :, kz].T,
    )
    im_c = ax.imshow(ere035_sl, origin="lower", extent=extent, cmap=MRM_ERE_CMAP, vmin=0.0, vmax=42.0, alpha=0.92)
    _set_panel_header(
        ax,
        "c",
        rf"0.35T ERE Hotspots (Peak +{ere_0p35.peak_dose_enhancement_pct:.1f}%, $r_g$ = {ere_0p35.gyroradius_mm:.1f} mm)",
    )
    ax.set_xlabel(r"Left–Right $x\;\mathrm{[mm]}$")
    ax.set_ylabel(r"Anterior–Posterior $y\;\mathrm{[mm]}$")
    cbar = fig.colorbar(im_c, ax=ax, fraction=0.046, pad=0.04)
    _style_colorbar(cbar, "ERE Dose Spike [%]")
    _add_scalebar(ax, 20.0, half_x - 8.0, -half_y + 10.0, color="white")

    # Panel d: Field-Strength ERE Comparison
    ax = axes[1, 1]
    _style_1d_axes(ax)
    x_pos = np.arange(len(b0_grid))
    width = 0.35
    bars1 = ax.bar(
        x_pos - width / 2 - 0.02,
        pelvis_peaks,
        width=width,
        color=MRM_MAGENTA,
        edgecolor=MRM_SPINE,
        lw=0.75,
        label="Pelvic Prostate Phantom",
    )
    if has_brain:
        bars2 = ax.bar(
            x_pos + width / 2 + 0.02,
            brain_peaks,
            width=width,
            color=MRM_ULTRAMARINE,
            edgecolor=MRM_SPINE,
            lw=0.75,
            label="Real Brain CT (3D Slicer)",
        )
    else:
        bars2 = []

    ax.axhline(20.0, color=MRM_MARIGOLD, ls="--", lw=1.35, label="High-Risk ERE (+20%)")
    for idx_b, b in enumerate(bars1):
        h = b.get_height()
        lbl_txt = "0%" if idx_b == 0 else f"+{h:.1f}%"
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h + 1.1,
            lbl_txt,
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold",
            color=MRM_MAGENTA if h > 0 else MRM_MUTED,
        )
    for idx_b, b in enumerate(bars2):
        h = b.get_height()
        lbl_txt = "0%" if idx_b == 0 else f"+{h:.1f}%"
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h + 1.1,
            lbl_txt,
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold",
            color=MRM_ULTRAMARINE if h > 0 else MRM_MUTED,
        )

    xtick_lbls = []
    for b0 in b0_grid:
        rg = electron_gyroradius_mm(1.5, b0, 90.0)
        rg_str = r"$\infty$" if rg is None else f"{rg:.1f} mm"
        xtick_lbls.append(f"{b0:.2f} T\n" + r"($r_g$=" + f"{rg_str})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xtick_lbls, fontsize=8.0)
    ax.set_ylabel("Peak Exit-Interface ERE Dose Spike [%]")
    ax.set_ylim(0, 63)
    ax.legend(loc="upper left", fontsize=7.5)
    _set_panel_header(ax, "d", "ERE Field Dependence (1.5T > 0.35T > 0T)")

    _save_fig(
        fig,
        "fig3_sct_audit_and_ere_risk",
        "mrphys-guard v0.3.0  |  Pelvic Phantom (DERIVED_PERTURBATION) & 3D Slicer CT-brain.nrrd (PUBLIC_OPEN_CLINICAL)  |  "
        "Raaijmakers et al. 2005/2008 ERE physics  |  Research Software — NOT A MEDICAL DEVICE",
        left=0.072,
    )


# ---------------------------------------------------------------------------
# Figure 4: Formal IEC 60601-2-33 Implant Safety CSP Envelope (Before vs After)
# ---------------------------------------------------------------------------


def make_fig4_implant_csp() -> None:
    agent = MRSequenceCoDesignAgent()
    res_pm = agent.run(build_pacemaker_cardiac_brain_scenario())
    implant = build_pacemaker_cardiac_brain_scenario().implant
    assert res_pm.optimized_csp is not None

    metrics = [
        r"$B_{1+\mathrm{rms}}$" + "\n" + r"($\leq 2.0\;\mu\mathrm{T}$)",
        "WB SAR\n" + r"($\leq 2.0\;\mathrm{W/kg}$)",
        "Slew Rate\n" + r"($\leq 200\;\mathrm{T/m/s}$)",
        r"$\mathrm{d}B/\mathrm{d}t$" + "\n" + r"($\leq 60\;\mathrm{T/s}$)",
        "Active Time\n" + r"($\leq 30\;\mathrm{min}$)",
        r"$|\mathrm{d}B/\mathrm{d}x|$" + "\n" + r"($\leq 20\;\mathrm{T/m}$)",
    ]
    before_ratios = [
        res_pm.baseline_csp.max_b1_rms_uT / float(implant.max_b1_rms_uT or 2.0),
        res_pm.baseline_csp.max_wb_sar_W_per_kg / float(implant.max_wb_sar_W_per_kg or 2.0),
        res_pm.baseline_csp.max_slew_rate_T_per_m_s / float(implant.max_slew_rate_T_per_m_s or 200.0),
        max(e.estimated_db_dt_T_per_s for e in res_pm.baseline_csp.sequence_estimates)
        / float(implant.max_db_dt_T_per_s or 60.0),
        res_pm.baseline_csp.total_active_scan_time_min / float(implant.max_active_scan_time_min or 30.0),
        17.0 / float(implant.max_spatial_gradient_T_per_m or 20.0),
    ]
    after_ratios = [
        res_pm.optimized_csp.max_b1_rms_uT / float(implant.max_b1_rms_uT or 2.0),
        res_pm.optimized_csp.max_wb_sar_W_per_kg / float(implant.max_wb_sar_W_per_kg or 2.0),
        res_pm.optimized_csp.max_slew_rate_T_per_m_s / float(implant.max_slew_rate_T_per_m_s or 200.0),
        max(e.estimated_db_dt_T_per_s for e in res_pm.optimized_csp.sequence_estimates)
        / float(implant.max_db_dt_T_per_s or 60.0),
        res_pm.optimized_csp.total_active_scan_time_min / float(implant.max_active_scan_time_min or 30.0),
        17.0 / float(implant.max_spatial_gradient_T_per_m or 20.0),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.5), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.968, top=0.79, bottom=0.19, wspace=0.25)
    _add_mrm_header(
        fig,
        r"Formal IEC 60601-2-33 / ASTM F2503 Implant Safety CSP: Pacemaker Protocol Co-Design",
        r"1.5T MR-Conditional Reference Envelope  ·  $B_{1+\mathrm{rms}}$, WB SAR, $\mathrm{d}B/\mathrm{d}t$, & $|\mathrm{d}B/\mathrm{d}x|$ Constraint Satisfaction",
        left=0.065,
        top_y=0.965,
    )

    # Panel a: Normalized constraint ratios
    _style_1d_axes(ax1)
    x = np.arange(len(metrics))
    w = 0.34
    b1 = ax1.bar(
        x - w / 2 - 0.02,
        before_ratios,
        width=w,
        color=MRM_MAGENTA,
        edgecolor=MRM_SPINE,
        lw=0.75,
        label="Baseline Protocol (VIOLATION)",
    )
    b2 = ax1.bar(
        x + w / 2 + 0.02,
        after_ratios,
        width=w,
        color=MRM_JADE,
        edgecolor=MRM_SPINE,
        lw=0.75,
        label="Co-Designed Protocol (PASS)",
    )
    ax1.axhline(1.0, color=MRM_SPINE, lw=1.4, ls="--", label="Hard IEC / Implant Limit (1.0×)")
    ax1.axhspan(1.0, 2.38, color=MRM_MAGENTA, alpha=0.06)

    for bar in b1:
        h = bar.get_height()
        y_pos = h + 0.05 if abs(h - 1.0) > 0.08 else h + 0.06
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            f"{h:.2f}×",
            ha="center",
            va="bottom",
            fontsize=6.9,
            fontweight="bold",
            color=MRM_MAGENTA if h > 1.0 else MRM_INK,
        )
    for bar in b2:
        h = bar.get_height()
        if 0.80 <= h <= 1.02:
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                h - 0.10,
                f"{h:.2f}×",
                ha="center",
                va="top",
                fontsize=6.9,
                fontweight="bold",
                color=MRM_CANVAS,
            )
        else:
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.05,
                f"{h:.2f}×",
                ha="center",
                va="bottom",
                fontsize=6.9,
                fontweight="bold",
                color=MRM_JADE_DEEP,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=7.8)
    ax1.set_ylabel("Normalized Exposure Ratio (Observed / Limit)")
    ax1.set_ylim(0, 2.38)
    ax1.legend(loc="upper right", fontsize=7.5)
    _set_panel_header(ax1, "a", "Protocol-Wide IEC 60601-2-33 Constraint Margins")

    # Panel b: RF Thermal Envelope (B1+rms vs WB SAR)
    _style_1d_axes(ax2)
    b1_curve = np.linspace(0.0, 4.1, 200)
    sar_1p5 = 0.0868 * (1.5**2) * (b1_curve**2)
    ax2.plot(
        b1_curve,
        sar_1p5,
        color=MRM_VIOLET,
        lw=1.7,
        ls=":",
        label=r"1.5T Physics Manifold ($\mathrm{SAR} \propto B_0^2 \cdot B_{1+\mathrm{rms}}^2$)",
    )
    ax2.add_patch(
        mpatches.Rectangle(
            (0, 0),
            2.0,
            2.0,
            facecolor=MRM_JADE,
            alpha=0.14,
            edgecolor=MRM_JADE,
            lw=1.4,
            label=r"Pacemaker Safe Box ($B_{1+\mathrm{rms}} \leq 2.0\;\mu\mathrm{T}$)",
        )
    )
    ax2.add_patch(
        mpatches.Rectangle(
            (0, 0),
            1.1,
            0.5,
            facecolor=MRM_CERULEAN,
            alpha=0.20,
            edgecolor=MRM_CERULEAN,
            lw=1.3,
            ls="--",
            label=r"DBS Strict Box ($B_{1+\mathrm{rms}} \leq 1.1\;\mu\mathrm{T}$)",
        )
    )

    label_positions = [
        (3.72, 2.98, "right", "Cardiac Cine bSSFP"),
        (3.02, 1.56, "left", "Brain T2 TSE"),
        (0.56, 0.24, "left", "Brain 3D MPRAGE"),
    ]
    for idx, (e_old, e_new) in enumerate(
        zip(res_pm.baseline_csp.sequence_estimates, res_pm.optimized_csp.sequence_estimates, strict=True)
    ):
        ax2.scatter([e_old.b1_rms_uT], [e_old.wb_sar_W_per_kg], color=MRM_MAGENTA, s=74, edgecolor=MRM_CANVAS, linewidth=1.0, zorder=6)
        ax2.scatter([e_new.b1_rms_uT], [e_new.wb_sar_W_per_kg], color=MRM_JADE, s=80, marker="D", edgecolor=MRM_CANVAS, linewidth=1.0, zorder=7)
        if abs(e_old.b1_rms_uT - e_new.b1_rms_uT) > 0.08:
            ax2.annotate(
                "",
                xy=(e_new.b1_rms_uT + 0.04, e_new.wb_sar_W_per_kg + 0.04),
                xytext=(e_old.b1_rms_uT - 0.05, e_old.wb_sar_W_per_kg - 0.04),
                arrowprops={"arrowstyle": "->", "color": MRM_SPINE, "lw": 1.4},
            )
        tx, ty, ha_val, lbl_str = label_positions[idx]
        ax2.text(
            tx,
            ty,
            lbl_str,
            fontsize=7.7,
            fontweight="bold",
            color=MRM_INK,
            ha=ha_val,
            va="center",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": MRM_CANVAS, "edgecolor": MRM_BORDER, "alpha": 0.94},
        )

    ax2.set_xlim(0.0, 4.25)
    ax2.set_ylim(0.0, 3.32)
    ax2.set_xlabel(r"Sequence $B_{1+\mathrm{rms}}\;\mathrm{[\mu T]}$")
    ax2.set_ylabel("Whole-Body Averaged SAR [W/kg]")
    ax2.legend(loc="upper left", fontsize=7.4)
    _set_panel_header(ax2, "b", "Per-Sequence RF Thermal Trajectory into Safe Envelope")

    _save_fig(
        fig,
        "fig4_implant_safety_csp_envelope",
        "mrphys-guard v0.3.0  |  Generic 1.5T MR-Conditional Pacemaker Reference Profile (NOT Manufacturer IFU)  |  "
        "Independent Critic-Verifier: APPROVED  |  Research Software — NOT A MEDICAL DEVICE",
        left=0.065,
    )


# ---------------------------------------------------------------------------
# Figure 5: 3D Grid-Phantom Spherical Harmonics QA (AAPM TG-284)
# ---------------------------------------------------------------------------


def make_fig5_phantom_qa() -> None:
    true_pts, meas_pts, _ = generate_synthetic_grid_phantom(
        extent_mm=180.0, spacing_mm=30.0, max_edge_distortion_mm=2.8, noise_sigma_mm=0.02, seed=42
    )
    gnl_l3 = fit_spherical_harmonic_gnl(true_pts, meas_pts, lmax=3)
    gnl_l5 = fit_spherical_harmonic_gnl(true_pts, meas_pts, lmax=5)

    raw_disp = meas_pts - true_pts
    res_l3 = raw_disp - gnl_l3.predict(true_pts)
    res_l5 = raw_disp - gnl_l5.predict(true_pts)

    radii = np.linalg.norm(true_pts, axis=1)
    mag_raw = np.linalg.norm(raw_disp, axis=1)
    mag_l3 = np.linalg.norm(res_l3, axis=1)
    mag_l5 = np.linalg.norm(res_l5, axis=1)

    mid_mask = np.abs(true_pts[:, 2]) < 1e-3
    p_mid = true_pts[mid_mask]
    d_raw_mid = raw_disp[mid_mask]
    d_l5_mid = res_l5[mid_mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.6), constrained_layout=False)
    fig.subplots_adjust(left=0.068, right=0.962, top=0.795, bottom=0.16, wspace=0.25)
    _add_mrm_header(
        fig,
        r"AAPM TG-284 3D Grid-Phantom QA: Maxwell Solid Spherical-Harmonic GNL Correction",
        r"3D Gradient Nonlinearity Tensor Fit  ·  $\ell \leq 3$ vs. $\ell \leq 5$ Solid Harmonics  ·  200 mm & 400 mm DSV Tolerance Audit",
        left=0.068,
        top_y=0.965,
    )

    # Panel a: Mid-plane quiver (12x magnified vectors)
    _style_1d_axes(ax1)
    ax1.axhline(0.0, color=MRM_BORDER, lw=0.7, ls="-", zorder=1)
    ax1.axvline(0.0, color=MRM_BORDER, lw=0.7, ls="-", zorder=1)
    scale_vis = 12.0
    ax1.scatter(p_mid[:, 0], p_mid[:, 1], s=14, color="#9D96BA", alpha=0.65, label="True Phantom Grid", zorder=2)
    ax1.quiver(
        p_mid[:, 0],
        p_mid[:, 1],
        d_raw_mid[:, 0] * scale_vis,
        d_raw_mid[:, 1] * scale_vis,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=MRM_MAGENTA,
        width=0.0055,
        zorder=4,
        label=f"Uncorrected GNL (12× vis, max {np.max(mag_raw):.2f} mm)",
    )
    ax1.quiver(
        p_mid[:, 0],
        p_mid[:, 1],
        d_l5_mid[:, 0] * scale_vis,
        d_l5_mid[:, 1] * scale_vis,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=MRM_JADE,
        width=0.0055,
        zorder=5,
        label=rf"Post $\ell \leq 5$ Residual (RMS {gnl_l5.fit_residual_rms_mm:.3f} mm)",
    )
    circle_200 = mpatches.Circle(
        (0, 0),
        100.0,
        fill=False,
        edgecolor=MRM_ULTRAMARINE,
        ls="--",
        lw=1.35,
        zorder=3,
        label=r"200 mm DSV ($r = 100\;\mathrm{mm}$)",
    )
    ax1.add_patch(circle_200)
    ax1.set_aspect("equal")
    ax1.set_xlim(-215, 215)
    ax1.set_ylim(-245, 215)
    ax1.set_xlabel(r"$x\;\mathrm{[mm]}$")
    ax1.set_ylabel(r"$y\;\mathrm{[mm]}$")
    ax1.legend(loc="lower left", fontsize=7.2)
    _set_panel_header(ax1, "a", r"Mid-Plane ($z = 0$) Fiducial Displacement Vectors")
    _add_scalebar(ax1, 50.0, 200.0, -230.0, color=MRM_SPINE, light_bg=True)

    # Panel b: Radial distortion curve vs distance from isocenter
    _style_1d_axes(ax2)
    bins = np.linspace(0, 210, 11)
    centers = 0.5 * (bins[:-1] + bins[1:])
    raw_p95, l3_p95, l5_p95 = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (radii >= lo) & (radii < hi)
        if np.any(m):
            raw_p95.append(float(np.percentile(mag_raw[m], 95)))
            l3_p95.append(float(np.percentile(mag_l3[m], 95)))
            l5_p95.append(float(np.percentile(mag_l5[m], 95)))
        else:
            raw_p95.append(np.nan)
            l3_p95.append(np.nan)
            l5_p95.append(np.nan)

    ax2.scatter(radii, mag_raw, s=8, alpha=0.14, color=MRM_MAGENTA)
    ax2.plot(centers, raw_p95, "o-", color=MRM_MAGENTA, lw=2.3, ms=5.0, label="Uncorrected GNL (95th pct)")
    ax2.plot(
        centers,
        l3_p95,
        "s--",
        color=MRM_VIOLET,
        lw=2.1,
        ms=4.8,
        label=rf"Post $\ell \leq 3$ Harmonic Fit (RMS {gnl_l3.fit_residual_rms_mm:.3f} mm)",
    )
    ax2.plot(
        centers,
        l5_p95,
        "D-",
        color=MRM_JADE,
        lw=2.3,
        ms=4.8,
        label=rf"Post $\ell \leq 5$ Harmonic Fit (RMS {gnl_l5.fit_residual_rms_mm:.3f} mm)",
    )
    ax2.axhline(1.0, color=MRM_CERULEAN, ls=":", lw=1.4, label=r"AAPM TG-284 ($\leq 1.0\;\mathrm{mm}$ @ 200 mm DSV)")
    ax2.axhline(2.0, color=MRM_SPINE, ls="--", lw=1.2, label=r"AAPM TG-284 ($\leq 2.0\;\mathrm{mm}$ @ 400 mm DSV)")
    ax2.set_xlim(0, 215)
    ax2.set_ylim(-0.08, 3.18)
    ax2.set_xlabel(r"Radial Distance from Isocenter $r\;\mathrm{[mm]}$")
    ax2.set_ylabel(r"3D Geometric Displacement $|\Delta\mathbf{r}|\;\mathrm{[mm]}$")
    ax2.legend(loc="upper left", fontsize=7.3)
    _set_panel_header(ax2, "b", "Radial GNL Distortion & Harmonic Correction Residual")

    _save_fig(
        fig,
        "fig5_phantom_spherical_harmonics_qa",
        f"mrphys-guard v0.3.0  |  3D Maxwell-Harmonic Grid Phantom (30 mm pitch, σ=0.02 mm noise, N={len(true_pts)} fiducials)  |  "
        "AAPM TG-284 (Glide-Hurst et al. 2021)  |  Research Software — NOT A MEDICAL DEVICE",
        left=0.068,
    )


# ---------------------------------------------------------------------------
# Figure 6: Saltelli–Jansen QMC Global Sobol' Sensitivity
# ---------------------------------------------------------------------------


def make_fig6_sobol() -> None:
    reports = run_mr_uncertainty_sobol_suite(n_base=1024, b0_T=1.5, seed=2026)
    pretty_names = {
        "bandwidth_hz_per_px": "Receiver BW\n(150–650 Hz/px)",
        "air_cavity_delta_chi_ppm": r"Air-Cavity $\Delta\chi$" + "\n(4.5–11.0 ppm)",
        "gnl_calibration_residual_mm": "GNL Residual\n(0.10–0.75 mm)",
        "sct_hu_error_hu": "sCT HU Error\n(15–160 HU)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), constrained_layout=False)
    fig.subplots_adjust(left=0.068, right=0.962, top=0.76, bottom=0.18, wspace=0.24)
    _add_mrm_header(
        fig,
        r"Saltelli–Jansen QMC Global Sobol' Sensitivity Analysis ($S_1$ vs. $S_T$)",
        r"Variance-Based MR-Only Uncertainty Attribution  ·  $N = 1{,}024$ Base Samples (6,144 Evaluations / Output)  ·  1.5T MR-Linac",
        left=0.068,
        top_y=0.965,
    )

    panel_specs = [
        (axes[0], "a", reports["geometric_error"], "Total Target 3D Geometric Error [mm]"),
        (axes[1], "b", reports["dosimetric_uncertainty"], "sCT & MR-Only Dosimetric Uncertainty [%]"),
    ]

    for ax, lbl, rep, title in panel_specs:
        _style_1d_axes(ax)
        names = [pretty_names.get(idx.parameter, idx.parameter) for idx in rep.indices]
        s1 = np.array([idx.s1 for idx in rep.indices])
        st = np.array([idx.st for idx in rep.indices])
        s1_hi = np.array([idx.s1_ci95[1] for idx in rep.indices])
        st_hi = np.array([idx.st_ci95[1] for idx in rep.indices])
        s1_err = np.array(
            [
                [max(0.0, idx.s1 - idx.s1_ci95[0]) for idx in rep.indices],
                [max(0.0, idx.s1_ci95[1] - idx.s1) for idx in rep.indices],
            ]
        )
        st_err = np.array(
            [
                [max(0.0, idx.st - idx.st_ci95[0]) for idx in rep.indices],
                [max(0.0, idx.st_ci95[1] - idx.st) for idx in rep.indices],
            ]
        )

        x = np.arange(len(names))
        w = 0.35
        b1 = ax.bar(
            x - w / 2,
            s1,
            width=w,
            yerr=s1_err,
            capsize=3.2,
            error_kw={"ecolor": MRM_SPINE, "elinewidth": 1.0, "capthick": 0.9},
            color=MRM_ULTRAMARINE,
            edgecolor=MRM_SPINE,
            lw=0.75,
            label=r"First-Order $S_1$ ($\pm 95\%$ CI)",
        )
        b2 = ax.bar(
            x + w / 2,
            st,
            width=w,
            yerr=st_err,
            capsize=3.2,
            error_kw={"ecolor": MRM_SPINE, "elinewidth": 1.0, "capthick": 0.9},
            color=MRM_MARIGOLD,
            edgecolor=MRM_SPINE,
            lw=0.75,
            label=r"Total-Effect $S_T$ ($\pm 95\%$ CI)",
        )

        for bar, val, hi in zip(b1, s1, s1_hi, strict=True):
            y_txt = max(val, hi, 0.0) + 0.026
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_txt,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.6,
                fontweight="bold",
                color=MRM_ULTRAMARINE,
            )
        for bar, val, hi in zip(b2, st, st_hi, strict=True):
            y_txt = max(val, hi, 0.0) + 0.026
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_txt,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.6,
                fontweight="bold",
                color=MRM_MARIGOLD_DEEP,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8.0)
        ax.set_ylabel(r"Sobol' Variance Fraction ($S_1$, $S_T$)")
        ax.set_ylim(0.0, 1.08)
        ax.legend(loc="upper right", fontsize=7.8)
        sub_stats = (
            rf"Mean = {rep.output_mean:.2f} $\pm$ {rep.output_std:.2f} {rep.output_unit}  ·  "
            rf"$\Sigma S_1 = {rep.sum_s1:.2f}$, $\Sigma S_T = {rep.sum_st:.2f}$"
        )
        _set_panel_header(ax, lbl, title, subtitle=sub_stats)

    _save_fig(
        fig,
        "fig6_sobol_mr_uncertainty",
        "mrphys-guard v0.3.0  |  Scrambled Sobol' QMC (Saltelli et al. 2010, DOI: 10.1016/j.cpc.2009.09.018)  |  "
        "1.5T MR-Only Simulation  |  Research Software — NOT A MEDICAL DEVICE",
        left=0.068,
    )


# ---------------------------------------------------------------------------
# Figure 7: Multi-Beam IMRT ERE Cancellation & 3D DCT Phase Unwrapping
# ---------------------------------------------------------------------------


def make_fig7_multibeam_ere_and_phase_unwrap() -> None:
    """Figure 7: Multi-Beam IMRT ERE Cancellation + 3D DCT Phase Unwrapping & Jacobian Audit."""
    shape = (64, 64, 40)
    vox = (2.5, 2.5, 2.5)
    phantom = build_pelvic_prostate_susceptibility_phantom(shape=shape, voxel_size_mm=vox)

    plans = [
        ("1-Beam\n(0°)", [0.0]),
        ("Opposing\n(0°, 180°)", [0.0, 180.0]),
        ("5-Beam\nIMRT", [0.0, 72.0, 144.0, 216.0, 288.0]),
        ("7-Beam\nIMRT", [0.0, 51.0, 103.0, 154.0, 206.0, 257.0, 309.0]),
    ]
    peaks, p95s, reductions = [], [], []
    for _, angs in plans:
        rep, _ = compute_multibeam_ere_cancellation(
            phantom["ct_hu"], vox, gantry_angles_deg=angs, b0_T=1.5, body_mask=phantom["body_mask"]
        )
        peaks.append(rep.multibeam_net_peak_pct)
        p95s.append(rep.multibeam_p95_pct)
        reductions.append(rep.cancellation_reduction_pct)

    # 3D DCT Dual-Echo Phase Unwrapping 1D central profile
    ux, uy, uz = np.meshgrid(
        np.linspace(-1.0, 1.0, 48),
        np.linspace(-1.0, 1.0, 48),
        np.linspace(-1.0, 1.0, 24),
        indexing="ij",
    )
    true_phi = 8.8 * np.exp(-(ux**2 + uy**2 + uz**2) / 0.42) + 4.2 * ux
    wrapped_phi = np.arctan2(np.sin(true_phi), np.cos(true_phi))
    _, unwrap_fields = unwrap_3d_b0_phase_dct(wrapped_phi, delta_te_ms=2.4, b0_T=1.5)
    unwrapped_phi = unwrap_fields["unwrapped_phase_rad"]
    unwrapped_aligned = unwrapped_phi - float(np.median(unwrapped_phi - true_phi))

    # Jacobian folding diagnostic on pelvic phantom at BW=220 Hz/px
    seq_lo = MRSequenceSpec(
        name="Pelvis_BW220", b0_T=1.5, bandwidth_hz_per_px=220.0, voxel_size_mm=vox, tr_ms=15.0, te_ms=4.5
    )
    _, fields_lo = compute_3d_distortion_decomposition(
        phantom["chi_si"], seq_lo, target_roi_mask=phantom["prostate_mask"], body_mask=phantom["body_mask"]
    )
    jac_rep, _ = compute_distortion_jacobian_and_unwarp(
        phantom["mri_t1"], fields_lo["total_disp_vec_mm"], vox, body_mask=phantom["body_mask"]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.4), constrained_layout=False)
    fig.subplots_adjust(left=0.068, right=0.962, top=0.79, bottom=0.17, wspace=0.25)
    _add_mrm_header(
        fig,
        r"Multi-Beam IMRT ERE Cancellation & 3D DCT Dual-Echo $\Delta B_0$ Phase Unwrapping",
        r"v0.2.0 Advanced MR-Linac Physics  ·  Opposing-Beam Lorentz Cancellation, Multi-$2\pi$ Poisson Recovery & Jacobian Singularity Audit",
        left=0.068,
        top_y=0.965,
    )

    # Panel a: Multi-beam ERE cancellation bar comparison
    _style_1d_axes(ax1)
    x = np.arange(len(plans))
    w = 0.35
    b1 = ax1.bar(
        x - w / 2,
        peaks,
        width=w,
        color=MRM_MAGENTA,
        edgecolor=MRM_SPINE,
        lw=0.75,
        label="Peak Exit ERE Spike [%]",
    )
    b2 = ax1.bar(
        x + w / 2,
        p95s,
        width=w,
        color=MRM_ULTRAMARINE,
        edgecolor=MRM_SPINE,
        lw=0.75,
        label="95th-Pct Interface ERE [%]",
    )
    ax1.axhline(20.0, color=MRM_MARIGOLD, ls="--", lw=1.35, label="High-Risk ERE Threshold (+20%)")
    for idx_p, bar in enumerate(b1):
        h = bar.get_height()
        red_txt = f"\n(−{reductions[idx_p]:.0f}%)" if idx_p > 0 else ""
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.85,
            f"+{h:.1f}%{red_txt}",
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold",
            color=MRM_MAGENTA,
        )
    for bar in b2:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.85,
            f"+{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold",
            color=MRM_ULTRAMARINE,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels([p[0] for p in plans], fontsize=8.2)
    ax1.set_ylabel("Net Tissue–Air ERE Dose Enhancement [%]")
    ax1.set_ylim(0, 44)
    ax1.legend(loc="upper right", fontsize=7.6)
    _set_panel_header(ax1, "a", "1.5T Opposing & IMRT Multi-Beam ERE Cancellation")

    # Panel b: 3D DCT Phase Unwrapping profile
    _style_1d_axes(ax2)
    mid_y, mid_z = 24, 12
    x_mm = np.linspace(-60.0, 60.0, 48)
    ax2.plot(x_mm, true_phi[:, mid_y, mid_z], color=MRM_SPINE, lw=3.2, alpha=0.28, label=r"True Unwrapped Phase $\varphi(x)$")
    ax2.plot(
        x_mm,
        wrapped_phi[:, mid_y, mid_z],
        color=MRM_VIOLET,
        lw=1.7,
        ls="--",
        label=r"Measured Wrapped Phase $\psi \in [-\pi, \pi)$",
    )
    ax2.plot(
        x_mm,
        unwrapped_aligned[:, mid_y, mid_z],
        "o",
        ms=3.4,
        color=MRM_JADE,
        markeredgecolor=MRM_CANVAS,
        markeredgewidth=0.45,
        label=r"3D DCT Poisson Unwrap (Err $< 10^{-12}\;\mathrm{rad}$)",
    )
    ax2.axhline(np.pi, color="#9D96BA", ls=":", lw=1.0)
    ax2.axhline(-np.pi, color="#9D96BA", ls=":", lw=1.0)
    ax2.text(
        0.97,
        0.05,
        f"Low-BW (220 Hz/px) Jacobian Audit:\n"
        rf"$\min J = {jac_rep.min_jacobian:.2f}$  |  Folding vox = {jac_rep.folding_voxel_count}",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.6,
        color=MRM_INK,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#FFFBF2", "edgecolor": MRM_MARIGOLD, "lw": 0.95, "alpha": 0.96},
    )
    ax2.set_xlabel(r"Spatial Position Along Readout $x\;\mathrm{[mm]}$")
    ax2.set_ylabel(r"Dual-Echo $\Delta B_0$ Phase [rad]")
    ax2.set_ylim(-5.0, 14.5)
    ax2.legend(loc="upper left", fontsize=7.4)
    _set_panel_header(ax2, "b", r"3D DCT Phase Unwrapping (Multi-$2\pi$) & Jacobian Audit")

    _save_fig(
        fig,
        "fig7_multibeam_ere_and_phase_unwrap",
        "mrphys-guard v0.3.0  |  Raaijmakers 2005/2008 opposing-beam ERE cancellation & Ghiglia–Romero 1994 3D DCT phase unwrapping  |  "
        "Research Software — NOT A MEDICAL DEVICE",
        left=0.068,
    )


# ---------------------------------------------------------------------------
# Figure 8: MR-Only Ambiguity-to-DVH & Anisotropic van Herk PTV Margin Certifier
# ---------------------------------------------------------------------------


def make_fig8_ambiguity_dvh_and_margins() -> None:
    """Figure 8: MR-Only Physics-Ambiguity to DVH Curves & 6-Direction Readout/Margin Optimizer."""
    shape = (64, 64, 48)
    vox = (2.5, 2.5, 2.5)
    phantom = build_pelvic_prostate_susceptibility_phantom(
        shape=shape, voxel_size_mm=vox, include_rectal_gas=True, include_bilateral_hip_prostheses=True
    )
    true_pts, meas_pts, _ = generate_synthetic_grid_phantom(
        extent_mm=180.0, spacing_mm=30.0, max_edge_distortion_mm=2.8, noise_sigma_mm=0.0
    )
    gnl_model = fit_spherical_harmonic_gnl(true_pts, meas_pts, lmax=5)
    oar_wall = build_rectal_wall_oar_mask(
        phantom["rectal_gas_mask"], phantom["body_mask"], phantom["prostate_mask"]
    )
    dose_gy = build_clinical_srs_dose_grid(
        phantom["prostate_mask"], oar_wall, vox, prescription_gy=36.25
    )
    seq = MRSequenceSpec(
        name="Pelvis_SBRT_BW240_AP",
        b0_T=1.5,
        bandwidth_hz_per_px=240.0,
        voxel_size_mm=vox,
        readout_axis=1,
        phase_encode_axis=0,
        tr_ms=15.0,
        te_ms=4.5,
    )
    cert, art = certify_mronly_dvh_and_anisotropic_margins(
        chi_si=phantom["chi_si"],
        mri_volume=phantom["mri_t1"],
        ct_or_sct_hu=phantom["ct_hu"],
        dose_gy=dose_gy,
        target_mask=phantom["prostate_mask"],
        oar_mask=oar_wall,
        body_mask=phantom["body_mask"],
        sequence=seq,
        gnl_model=gnl_model,
        prescription_gy=36.25,
    )

    bins = art["dose_bins_gy"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.6), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.965, top=0.79, bottom=0.18, wspace=0.25)
    _add_mrm_header(
        fig,
        r"MR-Only Ambiguity-to-DVH & Anisotropic van Herk PTV Margin Certifier",
        r"v0.3.0 Clinical Physics Certification  ·  1.5T Prostate SBRT (36.25 Gy / 5 fx)  ·  Dark-MRI $\Omega_{\mathrm{ambig}}$ Bound & 6-Direction Readout Selection",
        left=0.065,
        top_y=0.965,
    )

    # Panel a: Planned vs Distorted vs Worst-Case Dark-MRI sCT+ERE DVH bands
    _style_1d_axes(ax1)
    ax1.fill_between(
        bins,
        art["dvh_ctv_worst_case"],
        art["dvh_ctv_planned"],
        color=MRM_ULTRAMARINE,
        alpha=0.13,
        label="CTV Dark-MRI sCT+Distortion Band",
    )
    ax1.plot(bins, art["dvh_ctv_planned"], color=MRM_ULTRAMARINE, lw=2.3, label="Prostate CTV (Planned)")
    ax1.plot(
        bins,
        art["dvh_ctv_distorted"],
        color=MRM_CERULEAN,
        lw=1.9,
        ls="--",
        label=f"CTV Distorted ({cert.baseline_readout_label})",
    )
    ax1.plot(bins, art["dvh_ctv_worst_case"], color=MRM_VIOLET, lw=1.8, ls=":", label="CTV Worst-Case (Dark-MRI sCT)")

    ax1.fill_between(
        bins,
        art["dvh_oar_planned"],
        art["dvh_oar_worst_case"],
        color=MRM_MAGENTA,
        alpha=0.13,
        label="Rectal Wall sCT+ERE Blind-Spot Band",
    )
    ax1.plot(bins, art["dvh_oar_planned"], color=MRM_MAGENTA, lw=2.3, label="Rectal Wall (Planned)")
    ax1.plot(bins, art["dvh_oar_worst_case"], color=MRM_MAGENTA_DEEP, lw=1.8, ls="-.", label="Rectal Wall Worst-Case (sCT+ERE)")
    ax1.axvline(36.25, color=MRM_SPINE, ls=":", lw=1.15)

    ctv_imp = cert.organ_impacts[0]
    oar_imp = cert.organ_impacts[1]
    ax1.text(
        0.025,
        0.05,
        rf"CTV $D_{{98\%}}$: {ctv_imp.planned_d98_gy:.1f} → {ctv_imp.distorted_d98_gy:.1f} → {ctv_imp.worst_case_d98_gy:.1f} Gy (Worst)" + "\n"
        + rf"Rectum $D_{{0.1\mathrm{{cc}}}}$: {oar_imp.planned_d0p1cc_gy:.1f} → {oar_imp.distorted_d0p1cc_gy:.1f} → {oar_imp.worst_case_d0p1cc_gy:.1f} Gy" + "\n"
        + rf"Dark-MRI $\Omega_{{\mathrm{{ambig}}}} = {cert.dark_mri_certificate.ambiguous_volume_cc:.1f}\;\mathrm{{cc}}$  |  Cert: {cert.overall_certificate}",
        transform=ax1.transAxes,
        fontsize=7.2,
        color=MRM_INK,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.26", "facecolor": MRM_CANVAS, "edgecolor": MRM_BORDER, "lw": 0.9, "alpha": 0.95},
    )
    ax1.set_xlim(0.0, 42.5)
    ax1.set_ylim(-2.0, 114.0)
    ax1.set_xlabel("Dose [Gy]")
    ax1.set_ylabel("Cumulative Volume [%]")
    ax1.legend(loc="center left", bbox_to_anchor=(0.01, 0.56), fontsize=6.9)
    _set_panel_header(ax1, "a", "Distortion-DVH & Worst-Case Dark-MRI sCT+ERE Envelope")

    # Panel b: 6-Direction Readout Axis & Polarity Optimization + Anisotropic van Herk Margins
    _style_1d_axes(ax2)
    cands = cert.readout_candidates
    labels = [c.label for c in cands]
    d98_loss = [max(0.0, -c.target_delta_d98_gy) for c in cands]
    oar_inc = [max(0.0, c.oar_delta_d0p1cc_gy) for c in cands]
    best_idx = next(i for i, c in enumerate(cands) if c.label == cert.recommended_readout_label)

    x = np.arange(len(cands))
    w = 0.34
    b1 = ax2.bar(
        x - w / 2 - 0.02,
        d98_loss,
        width=w,
        color=MRM_ULTRAMARINE,
        edgecolor=MRM_SPINE,
        lw=0.75,
        label=r"Target CTV $D_{98\%}$ Loss [Gy]",
    )
    b2 = ax2.bar(
        x + w / 2 + 0.02,
        oar_inc,
        width=w,
        color=MRM_MAGENTA,
        edgecolor=MRM_SPINE,
        lw=0.75,
        label=r"Rectal Wall $D_{0.1\mathrm{cc}}$ Overdose [Gy]",
    )

    ax2.axvspan(
        best_idx - 0.48,
        best_idx + 0.48,
        color=MRM_JADE,
        alpha=0.14,
        label=f"Optimal Readout: {cert.recommended_readout_label}",
    )

    ymax = max(max(d98_loss + oar_inc, default=1.0) * 1.65, 2.5)
    for bar in b1:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + ymax * 0.018,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color=MRM_ULTRAMARINE,
        )
    for bar in b2:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + ymax * 0.018,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color=MRM_MAGENTA,
        )

    mx, my, mz = cert.recommended_anisotropic_margin_mm
    iso_m = next(c.naive_isotropic_margin_mm for c in cands if c.label == cert.recommended_readout_label)
    ax2.text(
        0.98,
        0.96,
        f"Optimal Anisotropic van Herk PTV Margin:\n({mx:.2f}, {my:.2f}, {mz:.2f}) mm vs. Isotropic {iso_m:.2f} mm",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=7.3,
        fontweight="bold",
        color=MRM_JADE_DEEP,
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "#EFFAF8", "edgecolor": MRM_JADE, "lw": 0.95, "alpha": 0.96},
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7.8)
    ax2.set_ylabel("Adverse Dosimetric Shift [Gy]")
    ax2.set_ylim(0.0, ymax)
    ax2.legend(loc="upper left", fontsize=7.2)
    _set_panel_header(ax2, "b", "Readout Axis/Polarity Selection & Anisotropic Margins")

    _save_fig(
        fig,
        "fig8_ambiguity_dvh_and_margins" if False else "fig8_ambiguity_dvh_and_anisotropic_margins",
        "mrphys-guard v0.3.0  |  Non-rigid 3D MRI distortion → DVH + Dark-MRI sCT+ERE blind-spot certifier + "
        "van Herk (2000) anisotropic margins  |  Research Software — NOT A MEDICAL DEVICE",
        left=0.065,
    )


def main() -> None:
    configure_journal_style()
    print("Generating 8 High-Field MRM Indigo-Mist 300-DPI PNG + PDF figures...")
    make_fig1_3d_distortion()
    make_fig2_bandwidth_pareto()
    make_fig3_sct_and_ere()
    make_fig4_implant_csp()
    make_fig5_phantom_qa()
    make_fig6_sobol()
    make_fig7_multibeam_ere_and_phase_unwrap()
    make_fig8_ambiguity_dvh_and_margins()
    print("All 8 figures generated cleanly in figures/.")


if __name__ == "__main__":
    main()
