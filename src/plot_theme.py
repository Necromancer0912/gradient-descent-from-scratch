"""
Self-Contained GitHub Dark-Mode Scientific Visualization Engine (`src/plot_theme.py`).

Part of the standalone `gradient-descent-from-scratch` repository. Provides
publication-grade dark-mode styling (`#0D1117` canvas, `#161B22` axes face),
custom neon/pastel colormaps, 2D decision boundary meshgrid rendering, 3D loss
surface + 2D contour trajectory helpers, and standardized 300-DPI figure saving.

Depends only on `numpy` and `matplotlib`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np

# =============================================================================
# 1. SIGNATURE GITHUB DARK-MODE PALETTE & CONSTANTS
# =============================================================================

BG_CANVAS: str = "#0D1117"      # GitHub Dark root canvas background
BG_AXES: str = "#161B22"        # Elevated card / axes face background
GRID_COLOR: str = "#21262D"     # Subtle structural grid & border tone
SPINE_COLOR: str = "#30363D"    # Visible left/bottom axis spine tone
TEXT_PRIMARY: str = "#C9D1D9"   # Primary axis labels, titles, and tick text
TEXT_MUTED: str = "#8B949E"     # Annotations, subtitles, and reference text

# Signature Neon/Pastel Accent Colors
CYAN: str = "#00F5D4"           # Primary / Trajectory / Positive Class / Model 1
CORAL: str = "#FF5E5B"          # Secondary / Target / Negative Class / Error
AMBER: str = "#FFBE0B"          # Highlight / Threshold / Split / Optimal Point
PURPLE: str = "#9B5DE5"         # Tertiary / Regularized Path / Model 2
SLATE_BLUE: str = "#4EA8DE"     # Baseline / Reference / Analytical Curve
EMERALD: str = "#3FB950"        # Convergence / Support Vector / Success Accent
MAGENTA: str = "#F72585"        # High-Degree / Divergence / Outlier Accent

PALETTE: Dict[str, str] = {
    "bg_canvas": BG_CANVAS,
    "bg_axes": BG_AXES,
    "grid": GRID_COLOR,
    "spine": SPINE_COLOR,
    "text": TEXT_PRIMARY,
    "text_muted": TEXT_MUTED,
    "cyan": CYAN,
    "coral": CORAL,
    "amber": AMBER,
    "purple": PURPLE,
    "slate_blue": SLATE_BLUE,
    "emerald": EMERALD,
    "magenta": MAGENTA,
}

CLASS_COLORS: Tuple[str, ...] = (CYAN, CORAL, AMBER, PURPLE, SLATE_BLUE, EMERALD)

# Custom Diverging Colormap for Binary Classification Decision Boundaries (Coral <-> Dark <-> Cyan)
CMAP_DECISION = LinearSegmentedColormap.from_list(
    "github_dark_decision",
    [CORAL, "#3B1D26", BG_AXES, "#0E3A38", CYAN],
    N=256,
)

# Custom Sequential/Surface Colormap for 3D Loss Landscapes & Heatmaps
CMAP_SURFACE = LinearSegmentedColormap.from_list(
    "github_dark_surface",
    ["#121824", "#1E2D4D", SLATE_BLUE, PURPLE, CORAL, AMBER, CYAN],
    N=256,
)

# Discrete Multiclass Colormap
CMAP_MULTICLASS = ListedColormap([CYAN, CORAL, AMBER, PURPLE, SLATE_BLUE, EMERALD], name="github_dark_multiclass")


# =============================================================================
# 2. GLOBAL MATPLOTLIB RC-PARAMS CONFIGURATION
# =============================================================================

def apply_github_dark_style() -> Dict[str, str]:
    """
    Configure global Matplotlib rcParams for the unified GitHub Dark-Mode aesthetic.

    Returns
    -------
    Dict[str, str]
        Dictionary of the active color palette for convenient access in scripts.
    """
    plt.rcParams.update({
        # Figure & Axes Backgrounds
        "figure.facecolor": BG_CANVAS,
        "figure.edgecolor": BG_CANVAS,
        "savefig.facecolor": BG_CANVAS,
        "savefig.edgecolor": BG_CANVAS,
        "axes.facecolor": BG_AXES,
        "axes.edgecolor": SPINE_COLOR,
        "axes.linewidth": 0.9,

        # Spines (hide top and right by default)
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,

        # Grid
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.6,
        "axes.axisbelow": True,

        # Typography & Colors
        "text.color": TEXT_PRIMARY,
        "axes.labelcolor": TEXT_PRIMARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "xtick.color": TEXT_PRIMARY,
        "ytick.color": TEXT_PRIMARY,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "stixsans",

        # Line & Scatter Defaults
        "lines.linewidth": 2.2,
        "lines.markersize": 6.0,
        "axes.prop_cycle": mpl.cycler(
            color=[CYAN, CORAL, AMBER, PURPLE, SLATE_BLUE, EMERALD, MAGENTA]
        ),

        # Legend
        "legend.facecolor": "#0D1117",
        "legend.edgecolor": SPINE_COLOR,
        "legend.framealpha": 0.88,
        "legend.fontsize": 9.2,
        "legend.labelcolor": TEXT_PRIMARY,
    })
    return PALETTE


def style_3d_axis(ax: mpl.axes.Axes) -> None:
    """Apply dark-mode pane and grid styling to a Matplotlib 3D Axes (`Axes3D`)."""
    ax.set_facecolor(BG_CANVAS)
    pane_rgba = (22 / 255.0, 27 / 255.0, 34 / 255.0, 0.92)
    grid_rgba = (48 / 255.0, 54 / 255.0, 61 / 255.0, 0.55)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(pane_rgba)
        axis._axinfo["grid"]["color"] = grid_rgba
        axis._axinfo["grid"]["linestyle"] = "--"
        axis._axinfo["grid"]["linewidth"] = 0.6
    ax.tick_params(colors=TEXT_PRIMARY, labelsize=8.5)
    ax.xaxis.label.set_color(TEXT_PRIMARY)
    ax.yaxis.label.set_color(TEXT_PRIMARY)
    ax.zaxis.label.set_color(TEXT_PRIMARY)


# =============================================================================
# 3. REUSABLE DIAGNOSTIC PLOTTING HELPERS
# =============================================================================

def plot_decision_boundary(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    ax: Optional[mpl.axes.Axes] = None,
    title: str = "Decision Boundary & Probability Field",
    xlabel: str = "$x_1$",
    ylabel: str = "$x_2$",
    resolution: int = 220,
    padding: float = 0.6,
    is_probability: bool = True,
    plot_contour_lines: bool = True,
    alpha_surface: float = 0.38,
) -> mpl.axes.Axes:
    """
    Render a 2D decision boundary and probability/class field over feature space.

    Parameters
    ----------
    predict_fn : Callable[[np.ndarray], np.ndarray]
        Function mapping query coordinates `(N, 2)` to probabilities in `[0, 1]`
        or discrete class indices `{0, 1, ..., K-1}`.
    X : np.ndarray of shape (m, 2)
        2D input feature matrix.
    y : np.ndarray of shape (m,)
        Target labels.
    ax : Optional[mpl.axes.Axes]
        Target Matplotlib Axes. Created if None.
    """
    assert X.ndim == 2 and X.shape[1] == 2, f"Expected X of shape (m, 2), got {X.shape}"
    y_flat = np.asarray(y).ravel()

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.2))

    x1_min, x1_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    x2_min, x2_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, resolution),
        np.linspace(x2_min, x2_max, resolution),
    )
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    Z = np.asarray(predict_fn(grid))
    if Z.ndim == 2 and Z.shape[1] > 1:
        Z = np.argmax(Z, axis=1)
        is_probability = False
    Z = Z.reshape(xx1.shape)

    classes = np.unique(y_flat)
    if is_probability and len(classes) <= 2:
        cf = ax.contourf(xx1, xx2, Z, levels=25, cmap=CMAP_DECISION, vmin=0.0, vmax=1.0, alpha=alpha_surface)
        if plot_contour_lines:
            ax.contour(
                xx1, xx2, Z,
                levels=[0.5],
                colors=[AMBER],
                linewidths=2.4,
                linestyles="-",
            )
            ax.contour(
                xx1, xx2, Z,
                levels=[0.25, 0.75],
                colors=[TEXT_MUTED],
                linewidths=0.9,
                linestyles="--",
                alpha=0.7,
            )
    else:
        n_cls = max(len(classes), int(np.max(Z)) + 1)
        cmap_discrete = ListedColormap(CLASS_COLORS[:n_cls])
        levels = np.arange(-0.5, n_cls + 0.5, 1.0)
        cf = ax.contourf(xx1, xx2, Z, levels=levels, cmap=cmap_discrete, alpha=0.28)
        if plot_contour_lines:
            ax.contour(xx1, xx2, Z, levels=levels, colors=[AMBER], linewidths=1.3, alpha=0.85)

    for idx, cls_val in enumerate(classes):
        mask = y_flat == cls_val
        color = CLASS_COLORS[idx % len(CLASS_COLORS)]
        marker = "o" if idx % 2 == 0 else "^"
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            c=color,
            marker=marker,
            s=42,
            edgecolors=BG_CANVAS,
            linewidths=0.8,
            alpha=0.92,
            label=f"Class {int(cls_val)}",
            zorder=4,
        )

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    return ax


def plot_loss_landscape(
    W1: np.ndarray,
    W2: np.ndarray,
    Loss_grid: np.ndarray,
    trajectory: Optional[np.ndarray] = None,
    grad_fn: Optional[Callable[[float, float], Tuple[float, float]]] = None,
    title_3d: str = "3D Loss Surface $J(w, b)$",
    title_2d: str = "2D Contour & Gradient Descent Trajectory",
    param_names: Tuple[str, str] = ("$w$", "$b$"),
    figsize: Tuple[float, float] = (14.0, 5.8),
) -> Tuple[mpl.figure.Figure, Tuple[mpl.axes.Axes, mpl.axes.Axes]]:
    """
    Create a side-by-side 3D Loss Surface and 2D Contour Trajectory visualization.

    Parameters
    ----------
    W1, W2 : np.ndarray of shape (G, G)
        2D parameter meshgrid arrays.
    Loss_grid : np.ndarray of shape (G, G)
        Evaluated cost values over `(W1, W2)`.
    trajectory : Optional[np.ndarray] of shape (T, 2) or (T, 3)
        Optimization path `[(w1_t, w2_t[, J_t]), ...]`.
    grad_fn : Optional[Callable]
        Optional gradient function `(w1, w2) -> (dw1, dw2)` for quiver arrows.
    """
    apply_github_dark_style()
    fig = plt.figure(figsize=figsize)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    style_3d_axis(ax3d)
    ax3d.plot_surface(
        W1, W2, Loss_grid,
        cmap=CMAP_SURFACE,
        alpha=0.78,
        edgecolor="none",
        antialiased=True,
        rstride=2,
        cstride=2,
    )
    ax3d.set_xlabel(param_names[0], labelpad=8)
    ax3d.set_ylabel(param_names[1], labelpad=8)
    ax3d.set_zlabel("Cost $J$", labelpad=6)
    ax3d.set_title(title_3d, pad=12)
    ax3d.view_init(elev=28, azim=-52)

    levels = np.geomspace(
        max(float(np.min(Loss_grid)) + 1e-3, 1e-3),
        float(np.max(Loss_grid)),
        24,
    )
    ax2d.contourf(W1, W2, Loss_grid, levels=levels, cmap=CMAP_SURFACE, alpha=0.35)
    ax2d.contour(W1, W2, Loss_grid, levels=levels, cmap=CMAP_SURFACE, linewidths=0.9, alpha=0.85)

    if grad_fn is not None:
        step = max(1, W1.shape[0] // 10)
        q_w1 = W1[::step, ::step]
        q_w2 = W2[::step, ::step]
        u = np.zeros_like(q_w1)
        v = np.zeros_like(q_w2)
        for i in range(q_w1.shape[0]):
            for j in range(q_w1.shape[1]):
                gw1, gw2 = grad_fn(float(q_w1[i, j]), float(q_w2[i, j]))
                norm = np.hypot(gw1, gw2) + 1e-12
                u[i, j] = -gw1 / norm
                v[i, j] = -gw2 / norm
        ax2d.quiver(
            q_w1, q_w2, u, v,
            color=TEXT_MUTED,
            alpha=0.45,
            scale=24,
            width=0.003,
            label=r"Steepest Descent $-\nabla J$",
        )

    if trajectory is not None and len(trajectory) > 0:
        traj = np.asarray(trajectory, dtype=np.float64)
        ax2d.plot(
            traj[:, 0], traj[:, 1],
            color=CYAN,
            linewidth=2.3,
            marker="o",
            markersize=4.2,
            label="GD Trajectory",
            zorder=5,
        )
        ax2d.scatter(
            traj[0, 0], traj[0, 1],
            color=CORAL, s=90, marker="D", edgecolors=BG_CANVAS, zorder=6, label="Init $\\theta^{(0)}$"
        )
        ax2d.scatter(
            traj[-1, 0], traj[-1, 1],
            color=AMBER, s=140, marker="*", edgecolors=BG_CANVAS, zorder=7, label="Optimum $\\theta^*$"
        )
        if traj.shape[1] >= 3:
            ax3d.plot(
                traj[:, 0], traj[:, 1], traj[:, 2],
                color=CYAN, linewidth=2.4, marker="o", markersize=3.8, zorder=10
            )
            ax3d.scatter(
                [traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]],
                color=AMBER, s=110, marker="*", zorder=11
            )

    ax2d.set_xlabel(param_names[0])
    ax2d.set_ylabel(param_names[1])
    ax2d.set_title(title_2d)
    ax2d.legend(loc="upper right")
    fig.tight_layout()
    return fig, (ax3d, ax2d)


def save_fig(
    fig: mpl.figure.Figure,
    path: Union[str, Path],
    dpi: int = 300,
    close: bool = False,
) -> Path:
    """
    Save a Matplotlib figure at publication-grade 300 DPI with `#0D1117` dark background.

    Parameters
    ----------
    fig : mpl.figure.Figure
        Figure instance to save.
    path : Union[str, Path]
        Destination file path (e.g. `"assets/plot_name.png"`). Parent directories are
        created automatically if missing.
    dpi : int, default=300
        Output resolution in dots per inch.
    close : bool, default=False
        Whether to close the figure after saving.

    Returns
    -------
    Path
        Resolved path to the saved PNG file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=BG_CANVAS,
        edgecolor="none",
    )
    if close:
        plt.close(fig)
    return out_path.resolve()


if __name__ == "__main__":
    # Self-test of the dark-mode plotting theme and finite-difference check on a test paraboloid
    palette = apply_github_dark_style()
    assert palette["bg_canvas"] == "#0D1117"
    assert palette["bg_axes"] == "#161B22"
    assert palette["cyan"] == "#00F5D4"

    # Quick analytical vs. numerical gradient check on the quadratic bowl used in plot_loss_landscape
    def quad_cost(theta: np.ndarray) -> float:
        return float(0.5 * (2.0 * theta[0] ** 2 + 0.5 * theta[1] ** 2))

    def quad_grad(theta: np.ndarray) -> np.ndarray:
        return np.array([2.0 * theta[0], 0.5 * theta[1]], dtype=np.float64)

    theta_test = np.array([1.75, -2.40], dtype=np.float64)
    eps = 1e-6
    num_grad = np.zeros_like(theta_test)
    for idx in range(theta_test.size):
        step = np.zeros_like(theta_test)
        step[idx] = eps
        num_grad[idx] = (quad_cost(theta_test + step) - quad_cost(theta_test - step)) / (2.0 * eps)

    ana_grad = quad_grad(theta_test)
    rel_err = np.linalg.norm(ana_grad - num_grad) / (np.linalg.norm(ana_grad) + np.linalg.norm(num_grad))
    print(f"[plot_theme.py] Style & Numerical Gradient Check Relative Error: {rel_err:.3e}")
    assert rel_err < 1e-7, f"Gradient check failed with relative error {rel_err}"
    print("[SUCCESS] GitHub Dark-Mode Plotting Engine initialized and verified.")

