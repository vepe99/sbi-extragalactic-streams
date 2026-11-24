"""Corner plot of the posterior of one track, with its posterior predictive check in the upper right corner.

Shown are the 15 parameters with the halo orientation vector replaced by what it stands for (q, cos i, psi):

  blue     the network's posterior: 68 and 95 % regions of all its draws
  orange   the simulated draws with their importance weights: 68 and 95 % contours and marginals, smoothed
           for the effective sample size, when that reaches `ESS_MIN`. Below it the draws are shown as points
           (marker area proportional to the weight): too few carry weight for a contour
  grey     the prior, on the diagonal
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from data import orientation
from predictive import ORANGE, plt

LABELS = [r"$\log_{10} M$", r"$R_s$ [kpc]", r"$q$", r"$\cos i$", r"$\psi$ [rad]", r"$\log_{10} m$", r"$r_s$ [kpc]", r"$x_0$ [kpc]",
          r"$z_0$ [kpc]", r"$v_{x0}$", r"$v_{y0}$", r"$v_{z0}$", r"$t$ [Gyr]", r"$\sigma_{\rm sys}$ [kpc]", "prog"]  # fmt: skip
BLUE, GREY = "#08306b", "#7b8794"
ESS_MIN = 10.0


def physical15(par) -> np.ndarray:
    """(..., 15) parameters -> the same with (q, cos i, psi) in place of (dirx, diry, dirz)."""
    par = np.asarray(par, np.float64)
    return np.concatenate([par[..., :2], orientation(par[..., 2:5]), par[..., 5:]], axis=-1)


def _levels(H) -> list:
    """Density levels that enclose 95 and 68 % of `H`, and one above its maximum."""
    h = np.sort(H.ravel())[::-1]
    c = np.cumsum(h) / h.sum()
    lv = [h[min(np.searchsorted(c, f), len(h) - 1)] for f in (0.95, 0.68)]
    return [lv[0], max(lv[1], lv[0] * (1 + 1e-9)), max(H.max(), lv[1] * (1 + 1e-9)) * (1 + 1e-9)]


def _regions(x, y, xlim, ylim, bins: int = 36):
    """Smoothed 2-d histogram and its `_levels`."""
    H = gaussian_filter(np.histogram2d(x, y, bins=bins, range=[xlim, ylim])[0].T, 1.0)
    return H, _levels(H)


def _bandwidth(v, w, lim, bins: int, ess: float, dim: int) -> float:
    """Scott's bandwidth for `ess` weighted draws in `dim` dimensions, in bins of the panel; at least one bin."""
    return max(bins * float(np.sqrt(np.cov(v, aweights=w))) / (lim[1] - lim[0]) * ess ** (-1 / (dim + 4)), 1.0)


def _weighted_regions(x, y, w, xlim, ylim, ess: float, bins: int = 36):
    """`_regions` of weighted draws: the smoothing is a kernel density estimate for `ess` draws, reflected at
    the edges of the panel (a parameter that fills its prior range is not pulled down at the bounds)."""
    H = np.histogram2d(x, y, bins=bins, range=[xlim, ylim], weights=w)[0].T
    H = gaussian_filter(H, (_bandwidth(y, w, ylim, bins, ess, 2), _bandwidth(x, w, xlim, bins, ess, 2)))
    return H, _levels(H)


def corner(samples, prior_samples, file, title: str, weighted: tuple | None = None, ppc=None,
           blue: str = "network posterior", orange: str = "simulated draws, importance weighted") -> None:
    """samples (n, 15), prior_samples (N, 15): `physical15` of posterior and prior draws. weighted = (x (M, 15),
    w (M,)): the simulated draws and their importance weights. ppc(ax) draws the predictive check. blue, orange:
    what the two are called in the legend (nested.py draws other posteriors with this function)."""
    D = samples.shape[1]
    x_w, w = weighted if weighted is not None else (None, None)
    show_w = w is not None and np.isfinite(w).all() and w.sum() > 0
    ess = float(w.sum() ** 2 / (w**2).sum()) if show_w else 0.0
    contours = show_w and ess >= ESS_MIN
    big = w > 1e-3 * w.max() if show_w else None  # the draws that carry weight
    lim = []
    for k in range(D):
        v = np.concatenate([samples[:, k], x_w[w > 1e-3 * w.max(), k]]) if show_w else samples[:, k]
        lo, hi = np.percentile(samples[:, k], [0.5, 99.5])
        lo, hi = min(lo, v.min()) if show_w else lo, max(hi, v.max()) if show_w else hi
        lim.append((lo - 0.04 * (hi - lo) - 1e-9, hi + 0.04 * (hi - lo) + 1e-9))

    fig, axes = plt.subplots(D, D, figsize=(1.3 * D, 1.3 * D))
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.05, top=0.99, wspace=0.07, hspace=0.07)
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if j > i:
                ax.set_axis_off()
                continue
            ax.tick_params(labelsize=6, length=2)
            if i == j:
                edges = np.linspace(*lim[j], 41)
                width = edges[1] - edges[0]
                ax.stairs(np.histogram(prior_samples[:, j], edges)[0] / (len(prior_samples) * width), edges, color=GREY, lw=0.9)
                ax.stairs(np.histogram(samples[:, j], edges)[0] / (len(samples) * width), edges, color=BLUE, lw=1.1)
                if contours:
                    h = np.histogram(x_w[:, j], edges, weights=w)[0] / (w.sum() * width)
                    ax.plot(0.5 * (edges[1:] + edges[:-1]), gaussian_filter1d(h, _bandwidth(x_w[:, j], w, lim[j], 40, ess, 1)), color=ORANGE, lw=1.1)
                ax.set_yticks([])
                ax.set_title(LABELS[j], fontsize=8, pad=3)
            else:
                H, lv = _regions(samples[:, j], samples[:, i], lim[j], lim[i])
                ax.contourf(H, levels=lv, extent=[*lim[j], *lim[i]], colors=["#c6dbef", "#6baed6"])
                if contours:
                    Hw, lw_ = _weighted_regions(x_w[:, j], x_w[:, i], w, lim[j], lim[i], ess)
                    ax.contour(Hw, levels=lw_[:2], extent=[*lim[j], *lim[i]], colors=ORANGE, linewidths=[0.6, 1.2], zorder=3)
                elif show_w:
                    ax.scatter(x_w[big, j], x_w[big, i], s=14 * w[big] / w.max(), color=ORANGE, alpha=0.6, lw=0, zorder=3)
                ax.set_ylim(*lim[i])
            ax.set_xlim(*lim[j])
            if i < D - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(LABELS[j], fontsize=8)
                ax.tick_params(axis="x", labelrotation=45)
            if j > 0 or i == 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(LABELS[i], fontsize=8)

    lines = [title, f"blue: {blue} ({len(samples)} draws), 68 and 95 %", "grey: prior"]
    if show_w:
        lines.append(f"orange: {len(w)} {orange} (effective sample size {ess:.1f})\n   "
                     + ("68 % (thick) and 95 % (thin) contours" if contours else f"shown as points: contours need at least {ESS_MIN:.0f}"))  # fmt: skip
    p0 = axes[0, 1].get_position()
    fig.text(p0.x0 + 0.01, p0.y1, "\n".join(lines), va="top", ha="left", fontsize=10, linespacing=1.5)
    if ppc is not None:  # the upper right block of the empty triangle
        rows, col = (D - 1) // 2, D - (D - 1) // 2
        a, b = axes[0, col].get_position(), axes[rows - 1, D - 1].get_position()
        ppc(fig.add_axes([a.x0 + 0.035, b.y0 + 0.035, b.x1 - a.x0 - 0.035, a.y1 - b.y0 - 0.06]))
    fig.savefig(file, dpi=90)
    plt.close(fig)
