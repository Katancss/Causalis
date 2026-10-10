"""Love plots for baseline-versus-treatment covariate balance."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from causalis.data_contracts.multicausal_estimate import MultiCausalEstimate
from causalis.data_contracts.multicausaldata import MultiCausalData
from causalis.scenarios.unconfoundedness.refutation.unconfoundedness.love_plot import (
    _coerce_plot_values,
)

from .unconfoundedness_validation import run_unconfoundedness_diagnostics


def love_plot(
    data: MultiCausalData,
    estimate: MultiCausalEstimate,
    *,
    threshold: float = 0.10,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 220,
    font_scale: float = 1.10,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
) -> plt.Figure:
    """Plot absolute SMDs before and after weighting, one panel per contrast.

    Parameters
    ----------
    data : MultiCausalData
        Dataset used to fit the estimator; treatment index 0 is the baseline.
    estimate : MultiCausalEstimate
        ATE or ATTE estimate produced with ``diagnostic_data=True``.
    threshold : float, default 0.10
        Reference threshold for absolute standardized mean differences.
    figsize : tuple, optional
        Figure size in inches. Defaults to a grid of up to three columns with
        height scaled to the number of confounders and panel rows.
    dpi : int, default 220
        Figure resolution.
    font_scale : float, default 1.10
        Font scaling factor.
    save : str, optional
        Output path for saving the figure.
    save_dpi : int, optional
        Save resolution; defaults to 300 for raster formats, otherwise ``dpi``.
    transparent : bool, default False
        Whether to save with a transparent background.

    Returns
    -------
    matplotlib.figure.Figure
        Figure closed in pyplot, but still available for display or saving.

    Notes
    -----
    Each panel compares one active treatment with the baseline using existing
    multi-treatment balance diagnostics. ATE uses inverse propensity weights;
    ATTE reweights the baseline to the corresponding treated population.
    Features share a stable ordering by their largest SMD across all panels,
    before or after weighting. Infinite values are clipped and annotated;
    NaN markers are omitted. Diagnostics are computed once, without refitting.

    Examples
    --------
    >>> from causalis.scenarios.multi_unconfoundedness.refutation import love_plot
    >>> estimate = model.fit().estimate(diagnostic_data=True)  # doctest: +SKIP
    >>> fig = love_plot(data, estimate, save="balance.png")  # doctest: +SKIP
    """
    report = run_unconfoundedness_diagnostics(
        data=data, estimate=estimate, threshold=float(threshold), return_summary=False
    )
    before = report["balance"]["smd_unweighted"]
    after = report["balance"]["smd"]
    worst = np.fmax(before.max(axis=1), after.max(axis=1))
    order = np.argsort(-worst, kind="stable")
    names = [str(before.index[i]) for i in order]
    before_values = before.to_numpy(dtype=float)[order]
    after_values = after.to_numpy(dtype=float)[order]
    before_plot, after_plot, x_limit, _ = _coerce_plot_values(
        before_values, after_values, threshold=float(threshold)
    )
    n_panels = before.shape[1]
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    panel_height = max(4.8, 1.8 + 0.28 * max(len(names), 1))
    default_size = (10.0 if n_cols == 1 else 6.0 * n_cols, panel_height * n_rows)
    y = np.arange(len(names), dtype=float)
    rc = {
        "font.size": 11 * font_scale,
        "axes.titlesize": 13 * font_scale,
        "axes.labelsize": 11 * font_scale,
        "legend.fontsize": 9.5 * font_scale,
        "xtick.labelsize": 10 * font_scale,
        "ytick.labelsize": 9.5 * font_scale,
    }
    with mpl.rc_context(rc):
        fig = plt.figure(figsize=figsize or default_size, dpi=dpi)
        try:
            axes = fig.subplots(n_rows, n_cols, sharex=True, squeeze=False).ravel()
            for idx, comparison in enumerate(before.columns):
                ax = axes[idx]
                b, a = before_plot[:, idx], after_plot[:, idx]
                paired = np.isfinite(b) & np.isfinite(a)
                segments = np.stack(
                    (np.column_stack((b[paired], y[paired])),
                     np.column_stack((a[paired], y[paired]))), axis=1
                )
                ax.add_collection(LineCollection(segments, colors="0.80", linewidths=1, zorder=1))
                for values, size, marker, color, label, zorder in (
                    (b, 42, "o", "C1", "Before (unweighted)", 3),
                    (a, 48, "D", "C0", "After (weighted)", 4),
                ):
                    finite = np.isfinite(values)
                    ax.scatter(
                        values[finite], y[finite], s=size, marker=marker, color=color,
                        edgecolors="white", linewidths=0.6, label=label, zorder=zorder,
                    )
                ax.axvline(
                    float(threshold), color="0.35", linestyle="--", linewidth=1.2,
                    label=f"Threshold |SMD| = {float(threshold):.2f}", zorder=2,
                )
                ax.set_xlim(0, x_limit)
                ax.tick_params(axis="x", labelbottom=True)
                ax.set_yticks(y)
                ax.set_yticklabels(names)
                ax.invert_yaxis()
                ax.set_xlabel("|SMD|")
                ax.set_ylabel("Confounders")
                ax.set_title(f"{comparison}\nLove Plot ({report['params']['score']})")
                ax.grid(True, axis="x", linewidth=0.5, alpha=0.45)
                ax.set_axisbelow(True)
                ax.spines[["top", "right"]].set_visible(False)
                ax.legend(frameon=False, loc="best")
                if np.isinf(before_values[:, idx]).any() or np.isinf(after_values[:, idx]).any():
                    ax.text(
                        0.995, 0.01, "Infinite SMDs clipped at right edge",
                        transform=ax.transAxes, ha="right", va="bottom",
                        color="0.40", fontsize=9 * font_scale,
                    )
            for ax in axes[n_panels:]:
                fig.delaxes(ax)
            fig.tight_layout()
            if save is not None:
                ext = str(save).lower().split(".")[-1]
                save_dpi_eff = save_dpi or (
                    300 if ext in {"png", "jpg", "jpeg", "tif", "tiff"} else dpi
                )
                fig.savefig(
                    save, dpi=save_dpi_eff, bbox_inches="tight", pad_inches=0.1,
                    transparent=transparent, facecolor="none" if transparent else "white",
                )
        finally:
            plt.close(fig)
    return fig


__all__ = ["love_plot"]
