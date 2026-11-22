"""Prior inspection before training an amortized posterior estimator on the C26 prior-predictive set.

    uv run python analysis/inspect_prior.py                 # production set
    uv run python analysis/inspect_prior.py --subdir pilot  # the 1,000-stream pilot

Reads data/analysis.h5 and the shard tracks (written by scripts/extragalactic/generate.py), writes figures to
plots/prior_inspection/ and the numbers behind them to data/prior_inspection.json. Nothing is simulated here.

Questions asked of the prior:
  1. funnel       how many of the prior draws are usable training examples
  2. acceptance   which parameters the selection cuts act on (the prior a network trained on the cut set sees)
  3. maps         where in parameter space the usable streams come from
  4. integrator   where the 99-step leapfrog does not resolve the progenitor orbit
  5. observable   what the network would receive: bins per track, coverage in angle, radial signal against noise
  6. strrings     the prior predictive against the 35 STRRINGS streams (arXiv:2508.02154, table 4)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "extragalactic"))
import plot as P  # noqa: E402  palette, matplotlib style and the data loader of the figure script

plt = P.plt
C_ALL, C_CUT, C_CLEAN = P.C_ALL, P.C_BOUND, P.C_PASS  # all draws | passing the C26 cuts | cuts + resolved orbit
DE_MAX = 1e-2  # relative energy error above which the progenitor orbit counts as unresolved (report.md)
NOISE = 0.02  # C26 likelihood: 2 % radial noise per bin

# STRRINGS, arXiv:2508.02154 table 4: (name, distance [Mpc], track length [kpc], width [kpc], angular extent [deg]).
# Transcribed from the arXiv HTML of the paper; check against the published table before quoting a single row.
STRRINGS = [
    ("ESO079-003_GROUP", 38.7, 142.4, 6.1, 110.2), ("ESO298-016", 76.6, 106.2, 10.2, 110.4),
    ("ESO356-012", 75.2, 55.4, 4.3, 147.1), ("IC0120", 70.9, 165.3, 5.9, 352.7), ("IC1124", 77.3, 45.7, 5.0, 81.2),
    ("IC1657_GROUP", 52.6, 210.3, 4.5, 274.2), ("NGC0804", 77.5, 468.3, 6.9, 396.6),
    ("NGC1084_GROUP", 21.0, 293.9, 2.4, 477.7), ("NGC1121", 38.2, 1199.3, 5.4, 713.6),
    ("NGC1162", 58.0, 248.7, 7.9, 352.7), ("NGC2769", 70.9, 125.8, 10.2, 102.8), ("NGC3041", 20.7, 20.0, 3.4, 51.2),
    ("NGC4256_GROUP", 37.3, 241.0, 8.8, 299.9), ("NGC4388_GROUP", 37.1, 66.1, 7.2, 109.6),
    ("NGC4632", 25.4, 33.8, 2.8, 117.2), ("NGC4686", 73.9, 134.7, 8.7, 95.6), ("NGC5055_GROUP", 7.5, 57.8, 2.6, 127.6),
    ("NGC5263", 71.1, 214.5, 5.4, 286.3), ("NGC5387", 76.8, 24.5, 2.1, 93.0), ("NGC5513_GROUP", 73.8, 106.4, 5.9, 152.2),
    ("NGC5907", 9.9, 71.7, 4.6, 212.9), ("PGC000902", 79.1, 1151.5, 7.6, 567.5), ("PGC006791", 79.2, 41.6, 4.5, 58.6),
    ("PGC021008", 83.8, 65.2, 2.3, 331.7), ("PGC039258", 85.1, 82.7, 3.4, 238.6), ("PGC1001085", 80.0, 407.4, 5.4, 426.5),
    ("PGC1092512", 76.3, 128.8, 5.4, 227.7), ("PGC3092153", 87.8, 44.2, 2.2, 141.9), ("PGC430221", 78.3, 45.9, 4.0, 162.2),
    ("PGC938075", 1.8, 4.3, 0.1, 404.0), ("UGC01245", 74.8, 86.0, 11.1, 161.1), ("UGC01424", 75.8, 98.0, 2.4, 205.7),
    ("UGC01970", 28.2, 138.3, 5.8, 227.8), ("UGC08717", 68.6, 333.7, 5.1, 496.8), ("UGC09239", 81.9, 234.4, 8.7, 227.9),
]  # fmt: skip


def binned_fraction(x, mask, edges):
    """Fraction of `mask` in bins of x, with its binomial standard error; NaN where a bin has < 20 draws."""
    n, _ = np.histogram(x, bins=edges)
    k, _ = np.histogram(x[mask], bins=edges)
    with np.errstate(all="ignore"):
        f = np.where(n >= 20, k / n, np.nan)
        e = np.where(n >= 20, np.sqrt(f * (1 - f) / n), np.nan)
    return f, e


def ks_distance(x, mask):
    """Two-sample Kolmogorov-Smirnov distance between x[mask] and x (the largest gap between the two CDFs)."""
    grid = np.sort(x)
    cdf_all = np.arange(1, len(grid) + 1) / len(grid)
    cdf_sel = np.searchsorted(np.sort(x[mask]), grid, side="right") / max(int(mask.sum()), 1)
    return float(np.max(np.abs(cdf_all - cdf_sel)))


def load(d: P.Data) -> dict:
    a = d.a
    th = a["theta"].astype(np.float64)
    x0 = np.stack([th[:, 7], np.zeros(len(th)), th[:, 8]], 1)
    v0 = th[:, 9:12]
    L = np.cross(x0, v0)
    with np.errstate(all="ignore"):
        cos_nL = np.abs(np.sum(a["axis"] * L, 1) / np.linalg.norm(L, axis=1))
        vr = np.sum(x0 * v0, 1) / np.linalg.norm(x0, axis=1)
    sel = a["sel_all"].astype(bool)
    resolved = a["dE_rel"] < DE_MAX
    m = dict(all=np.ones(len(th), bool), has_bin=a["n_valid"] > 0, sel_ang=a["sel_ang"].astype(bool),
             sel_arc=a["sel_arc"].astype(bool), sel_rad=a["sel_rad"].astype(bool), cut=sel, clean=sel & resolved,
             clean_bound=sel & resolved & a["bound"], clean_contig=sel & resolved & a["contiguous"].astype(bool),
             resolved=resolved, bound=a["bound"])
    derived = dict(r0=np.linalg.norm(x0, axis=1), speed=np.linalg.norm(v0, axis=1), cos_nL=cos_nL,
                   vr_frac=vr / np.linalg.norm(v0, axis=1))
    return dict(a=a, th=th, m=m, derived=derived)


# ------------------------------------------------------------------------------------------------------
def fig_funnel(d, s, out) -> None:
    m, n = s["m"], len(s["th"])
    rows = [("all prior draws", "all"), ("at least one valid bin", "has_bin"), ("angular length > 90 deg", "sel_ang"),
            ("arc length > 100 kpc", "sel_arc"), ("valid radii in [10, 500] kpc", "sel_rad"),
            ("all three C26 cuts", "cut"), ("cuts + orbit resolved (energy error < 1 %)", "clean"),
            ("cuts + resolved + bound", "clean_bound"), ("cuts + resolved + contiguous valid bins", "clean_contig")]
    fr = [m[k].mean() for _, k in rows]
    cols = [C_ALL, C_ALL, C_ALL, C_ALL, C_ALL, C_CUT, C_CLEAN, C_CLEAN, C_CLEAN]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, fr, height=0.62, color=cols)
    for yi, f, (_, k) in zip(y, fr, rows):
        ax.text(f + 0.012, yi, f"{100 * f:.1f} %   ({int(m[k].sum()):,})", va="center", ha="left", color=P.INK, fontsize=8.5)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 1.22)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25 %", "50 %", "75 %", "100 %"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"fraction of the {n:,} prior draws (green rows are subsets of the row above them; blue rows are independent)")
    ax.set_title("How much of the prior is usable as training data")
    fig.tight_layout()
    P.save(fig, d, "prior_inspection/funnel")
    out["funnel"] = {k: dict(n=int(m[k].sum()), frac=float(m[k].mean())) for _, k in rows}


def fig_acceptance(d, s, out) -> None:
    th, m, dv, c = s["th"], s["m"], s["derived"], d.cfg
    labels = [r"$\log_{10} M_{\rm halo}\ [M_\odot]$", r"$R_s$ [kpc]", r"$q$", r"$\cos i$", r"$\psi$ [rad]",
              r"$\log_{10} m_{\rm prog}\ [M_\odot]$", r"$r_s$ [kpc]", r"$x_0$ [kpc]", r"$z_0$ [kpc]",
              r"$v_{x0}$ [km/s]", r"$v_{y0}$ [km/s]", r"$v_{z0}$ [km/s]", r"$t_{\rm int}$ [Gyr]"]
    cols = [th[:, i] for i in range(13)] + [dv["r0"], dv["speed"], dv["cos_nL"]]
    labels += [r"3D distance today $|x_0|$ [kpc]", r"speed today $|v_0|$ [km/s]",
               r"$|\cos|$ of the angle between halo axis and orbital pole"]
    names = d.names + ["r0", "speed", "cos_nL"]
    fig, axes = plt.subplots(4, 4, figsize=(14, 11), sharey=True)
    ks = {}
    for ax, x, lab, name in zip(axes.ravel(), cols, labels, names):
        lo, hi = np.percentile(x, [0.5, 99.5]) if name in ("x0", "z0", "vx0", "vy0", "vz0", "r0", "speed") else (x.min(), x.max())
        edges = np.linspace(lo, hi, 21)
        ctr = 0.5 * (edges[:-1] + edges[1:])
        for key, col, lbl in (("cut", C_CUT, "passing the three C26 cuts"), ("clean", C_CLEAN, "cuts + resolved orbit")):
            f, e = binned_fraction(x, m[key], edges)
            ax.fill_between(ctr, f - e, f + e, color=col, alpha=0.22, lw=0)
            ax.plot(ctr, f, color=col, lw=2.0, label=lbl)
            ax.axhline(m[key].mean(), color=col, lw=0.9, ls=":")
        ks[name] = dict(cut=ks_distance(x, m["cut"]), clean=ks_distance(x, m["clean"]))
        ax.set_xlabel(lab)
        ax.text(0.97, 0.95, f"KS {ks[name]['clean']:.2f}", transform=ax.transAxes, ha="right", va="top", color=P.INK2, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction accepted")
    axes[0, 0].set_ylim(0, None)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Acceptance against each parameter: a flat curve means the cut leaves that prior untouched "
                 "(dotted: mean; KS: distance between accepted and full prior, cuts + resolved)", y=1.0, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    P.save(fig, d, "prior_inspection/acceptance")
    out["ks_selected_vs_prior"] = ks
    q = th[:, 2]
    out["q_under_selection"] = {
        k: dict(mean=float(q[m[k]].mean()), std=float(q[m[k]].std()),
                frac_q_below_1=float((q[m[k]] < 1).mean()),
                acceptance_q_0p5_0p7=float(m[k][(q < 0.7)].mean()), acceptance_q_1p3_1p5=float(m[k][(q > 1.3)].mean()))
        for k in ("all", "cut", "clean")}


def fig_bound(d, s, out) -> None:
    """The prior with and without the unbound draws, on one scale: each curve is the share of ALL draws per
    bin, so the gap between the two is what removing the unbound progenitors takes away."""
    a, th, m, dv = s["a"], s["th"], s["m"], s["derived"]
    labels = [r"$\log_{10} M_{\rm halo}\ [M_\odot]$", r"$R_s$ [kpc]", r"$q$", r"$\cos i$", r"$\psi$ [rad]",
              r"$\log_{10} m_{\rm prog}\ [M_\odot]$", r"$r_s$ [kpc]", r"$x_0$ [kpc]", r"$z_0$ [kpc]",
              r"$v_{x0}$ [km/s]", r"$v_{y0}$ [km/s]", r"$v_{z0}$ [km/s]", r"$t_{\rm int}$ [Gyr]",
              r"3D distance today $|x_0|$ [kpc]", r"speed today $|v_0|$ [km/s]",
              r"speed / escape speed today (bound: below 1)"]
    with np.errstate(all="ignore"):
        v_esc_ratio = np.sqrt(np.maximum(1.0 + a["E"] / np.abs(a["Phi0"]), 0.0))
    cols = [th[:, i] for i in range(13)] + [dv["r0"], dv["speed"], v_esc_ratio]
    names = d.names + ["r0", "speed", "v_over_vesc"]
    bound, n = m["bound"], len(th)
    fig, axes = plt.subplots(4, 4, figsize=(14, 11))
    ks = {}
    for ax, x, lab, name in zip(axes.ravel(), cols, labels, names):
        wide = name in ("x0", "z0", "vx0", "vy0", "vz0", "r0", "speed", "v_over_vesc")
        lo, hi = np.nanpercentile(x, [0.5, 99.5] if name != "v_over_vesc" else [0, 97]) if wide else (x.min(), x.max())
        edges = np.linspace(lo, hi, 41)
        h_all, _ = np.histogram(x, bins=edges)
        h_b, _ = np.histogram(x[bound], bins=edges)
        ax.stairs(h_b / n, edges, color=C_CUT, fill=True, alpha=0.30, lw=0)
        ax.stairs(h_b / n, edges, color=C_CUT, lw=1.8, label=f"bound only ({int(bound.sum()):,})")
        ax.stairs(h_all / n, edges, color=C_ALL, lw=1.8, label=f"all prior draws ({n:,})")
        ks[name] = ks_distance(x[np.isfinite(x)], bound[np.isfinite(x)])
        ax.set_xlabel(lab)
        ax.set_ylim(0, 1.2 * h_all.max() / n)  # headroom for the KS label
        ax.set_yticks([])
        ax.grid(False)
        ax.text(0.97, 0.95, f"KS {ks[name]:.2f}", transform=ax.transAxes, ha="right", va="top", color=P.INK2, fontsize=8)
    h, lab_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[::-1], lab_[::-1], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("The prior with and without the unbound draws (same scale: the gap between the curves is the unbound "
                 "share; KS: distance between bound-only and full prior)", y=1.0, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    P.save(fig, d, "prior_inspection/prior_bound_vs_all")
    out["bound_vs_all"] = dict(bound_frac=float(bound.mean()), ks=ks,
                               median_log10_M_halo=dict(all=float(np.median(th[:, 0])), bound=float(np.median(th[bound, 0]))),
                               median_speed_kms=dict(all=float(np.median(dv["speed"])), bound=float(np.median(dv["speed"][bound]))),
                               median_r0_kpc=dict(all=float(np.median(dv["r0"])), bound=float(np.median(dv["r0"][bound]))))


def fig_maps(d, s, out) -> None:
    th, m, dv = s["th"], s["m"], s["derived"]
    pairs = [(th[:, 0], dv["speed"], r"$\log_{10} M_{\rm halo}\ [M_\odot]$", r"speed today $|v_0|$ [km/s]", (11, 14), (0, 900)),
             (th[:, 0], dv["r0"], r"$\log_{10} M_{\rm halo}\ [M_\odot]$", r"3D distance today $|x_0|$ [kpc]", (11, 14), (0, 500)),
             (th[:, 7], th[:, 8], r"$x_0$ [kpc] (projected distance today)", r"$z_0$ [kpc] (line of sight)", (0, 450), (0, 450)),
             (th[:, 2], th[:, 3], r"$q$", r"$\cos i$", (0.5, 1.5), (0, 1))]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    vmax = 0.0
    grids = []
    for x, y, *_rest, rx, ry in pairs:
        ex, ey = np.linspace(*rx, 16), np.linspace(*ry, 16)
        n, _, _ = np.histogram2d(x, y, bins=[ex, ey])
        k, _, _ = np.histogram2d(x[m["clean"]], y[m["clean"]], bins=[ex, ey])
        with np.errstate(all="ignore"):
            f = np.where(n >= 30, k / n, np.nan)
        grids.append((ex, ey, f))
        vmax = max(vmax, float(np.nanmax(f)))
    for ax, (x, y, lx, ly, rx, ry), (ex, ey, f) in zip(axes, pairs, grids):
        im = ax.pcolormesh(ex, ey, f.T, cmap=P.SEQ, vmin=0, vmax=vmax, edgecolors=P.SURFACE, linewidth=0.5)
        ax.grid(False)
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
    fig.colorbar(im, ax=axes, label="fraction passing the cuts with a resolved orbit", pad=0.012, fraction=0.025)
    fig.suptitle("Where the usable streams come from (cells with fewer than 30 draws left blank)", fontsize=10.5, x=0.45)
    P.save(fig, d, "prior_inspection/acceptance_maps")


def fig_integrator(d, s, out) -> None:
    a, th, m = s["a"], s["th"], s["m"]
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.5))
    ax = axes[0]
    bins = np.linspace(-10, 2, 49)
    x = np.log10(np.clip(a["dE_rel"], 1e-10, 1e2))
    for key, col, lbl in (("all", C_ALL, "all draws"), ("cut", C_CUT, "passing the three cuts")):
        h, e = np.histogram(x[m[key]], bins=bins, weights=np.full(int(m[key].sum()), 1.0 / max(int(m[key].sum()), 1)))
        ax.stairs(h, e, color=col, lw=2.0, label=lbl)
    ax.axvline(np.log10(DE_MAX), color=P.INK2, lw=1.0, ls=":")
    ax.text(np.log10(DE_MAX), ax.get_ylim()[1] * 0.97, " 1 %", color=P.INK2, va="top", fontsize=8)
    ax.set_xlabel(r"$\log_{10}$ relative energy error of the progenitor orbit, 99 steps")
    ax.set_ylabel("fraction of the group")
    ax.legend(loc="upper left")
    ax = axes[1]
    with np.errstate(all="ignore"):
        lr = np.log10(a["r_min"])
    ex, ey = np.linspace(11, 14, 16), np.linspace(-0.5, 2.7, 17)
    sel = m["cut"] & np.isfinite(lr)
    n, _, _ = np.histogram2d(th[sel, 0], lr[sel], bins=[ex, ey])
    k, _, _ = np.histogram2d(th[sel & ~m["resolved"], 0], lr[sel & ~m["resolved"]], bins=[ex, ey])
    with np.errstate(all="ignore"):
        f = np.where(n >= 20, k / n, np.nan)
    im = ax.pcolormesh(ex, ey, f.T, cmap=P.SEQ, vmin=0, vmax=1, edgecolors=P.SURFACE, linewidth=0.5)
    ax.grid(False)
    ax.set_xlabel(r"$\log_{10} M_{\rm halo}\ [M_\odot]$")
    ax.set_ylabel(r"$\log_{10}$ pericentre within $t_{\rm int}$ [kpc]")
    fig.colorbar(im, ax=ax, label="fraction unresolved, among streams passing the cuts", pad=0.02)
    ax = axes[2]
    npr = np.minimum(a["n_peri"], 6)
    xs = np.arange(7)
    for key, col, lbl in (("all", C_ALL, "all draws"), ("cut", C_CUT, "passing the three cuts")):
        f = [float((~m["resolved"])[m[key] & (npr == i)].mean()) if (m[key] & (npr == i)).sum() >= 20 else np.nan for i in xs]
        ax.plot(xs, f, "o-", color=col, lw=2.0, ms=6, mfc=P.SURFACE, mew=1.6, label=lbl)
    ax.set_xticks(xs, [str(i) for i in xs[:-1]] + ["6+"])
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"pericentre passages within $t_{\rm int}$")
    ax.set_ylabel("fraction with energy error > 1 %")
    ax.legend(loc="lower right")
    ax = axes[3]
    edges = np.arange(0, 541, 30)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    by_ang = {}
    for key, col, lbl in (("has_bin", C_ALL, "all draws"), ("cut", C_CUT, "passing the three cuts")):
        f, e = binned_fraction(a["ang_len_deg"][m[key]], (~m["resolved"])[m[key]], edges)
        ax.fill_between(ctr, f - e, f + e, color=col, alpha=0.22, lw=0)
        ax.plot(ctr, f, color=col, lw=2.0, label=lbl)
        by_ang[key] = [None if not np.isfinite(v) else float(v) for v in f]
    o_ang = np.array([row[4] for row in STRRINGS])
    ax.plot(np.clip(o_ang, 0, 535), np.full(len(o_ang), 0.03), "|", ms=14, mew=1.6, color=P.INK, label="STRRINGS angular extents")
    ax.set_ylim(0, 1)
    ax.set_xlabel("angular length of the track [deg]  (observed values above 540 drawn at the edge)")
    ax.set_ylabel("fraction with energy error > 1 %")
    ax.legend(loc="upper left")
    fig.suptitle("Where the 99-step leapfrog does not resolve the progenitor orbit", fontsize=10.5)
    fig.tight_layout()
    P.save(fig, d, "prior_inspection/integrator")
    long = a["ang_len_deg"] > 270
    out["unresolved"] = dict(
        by_ang_len=dict(edges_deg=edges.tolist(), **by_ang),
        frac_cut_longer_than_270deg=float(long[m["cut"]].mean()),
        unresolved_among_cut_longer_than_270deg=float((~m["resolved"])[m["cut"] & long].mean()) if (m["cut"] & long).any() else None,
        all=float((~m["resolved"]).mean()), cut=float((~m["resolved"])[m["cut"]].mean()),
        by_n_peri_cut={str(i): (float((~m["resolved"])[m["cut"] & (npr == i)].mean()) if (m["cut"] & (npr == i)).any() else None) for i in xs},
        n_peri_share_cut={str(i): float((npr[m["cut"]] == i).mean()) for i in xs})


def fig_observable(d, s, out) -> None:
    a, m, t, c = s["a"], s["m"], d.tracks, d.cfg
    valid = t["count"] >= c["n_valid_min"]
    phi = d.phi
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    groups = (("cut", C_CUT, "passing the three cuts"), ("clean", C_CLEAN, "cuts + resolved orbit"))
    ax = axes[0]
    bins = np.arange(9.5, 80.5, 2)
    for key, col, lbl in groups:
        x = np.clip(a["n_valid"][m[key]], 0, 79)
        h, e = np.histogram(x, bins=bins, weights=np.full(len(x), 1.0 / max(len(x), 1)))
        ax.stairs(h, e, color=col, lw=2.0, label=lbl)
    ax.set_xlabel("valid 10-degree bins per track (last bin: 78 or more)")
    ax.set_ylabel("fraction of the group")
    ax.legend(loc="upper right")
    ax = axes[1]
    for key, col, lbl in groups:
        ax.plot(phi, valid[m[key]].mean(0), color=col, lw=2.0, label=lbl)
    ax.set_xlim(-540, 540)
    ax.set_xlabel(r"unwrapped angle $\phi$ [deg]  (trailing < 0 < leading)")
    ax.set_ylabel("fraction of tracks with a valid bin there")
    ax = axes[2]
    r = np.where(valid, t["r_mean"], np.nan).astype(np.float64)
    with np.errstate(all="ignore"):
        contrast = (np.nanmax(r, 1) - np.nanmin(r, 1)) / np.nanmean(r, 1)
    bins = np.logspace(-2.5, 1, 57)
    for key, col, lbl in groups:
        x = np.clip(contrast[m[key]], bins[0], bins[-1])
        h, e = np.histogram(x, bins=bins, weights=np.full(len(x), 1.0 / max(len(x), 1)))
        ax.stairs(h, e, color=col, lw=2.0)
    ax.axvline(NOISE, color=P.INK2, lw=1.0, ls=":")
    ax.text(NOISE, ax.get_ylim()[1] * 0.97, " 2 % noise per bin", color=P.INK2, va="top", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(r"radial contrast of the track, $(r_{\max} - r_{\min}) / \bar r$")
    ax.set_ylabel("fraction of the group")
    ax = axes[3]
    with np.errstate(all="ignore"):
        rel = np.where(valid, t["r_std"] / t["r_mean"], np.nan)
        relw = np.nanmedian(rel, 1)
    bins = np.logspace(-3, 0.3, 57)
    for key, col, lbl in groups:
        x = np.clip(relw[m[key]], bins[0], bins[-1])
        h, e = np.histogram(x, bins=bins, weights=np.full(len(x), 1.0 / max(len(x), 1)))
        ax.stairs(h, e, color=col, lw=2.0)
    ax.axvline(NOISE, color=P.INK2, lw=1.0, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"median relative radial width of the stream in a bin, $r_{\rm std} / r$")
    ax.set_ylabel("fraction of the group")
    fig.suptitle("What the network would receive", fontsize=10.5)
    fig.tight_layout()
    P.save(fig, d, "prior_inspection/observable")
    cl = m["clean"]
    out["observable"] = dict(
        n_valid_percentiles_clean=dict(zip(("p5", "p50", "p95"), [float(v) for v in np.percentile(a["n_valid"][cl], [5, 50, 95])])),
        contiguous_frac_cut=float(a["contiguous"][m["cut"]].mean()), contiguous_frac_clean=float(a["contiguous"][cl].mean()),
        both_arms_frac_clean=float((valid[cl][:, phi < 0].any(1) & valid[cl][:, phi > 0].any(1)).mean()),
        radial_contrast_percentiles_clean=dict(zip(("p5", "p50", "p95"), [float(v) for v in np.nanpercentile(contrast[cl], [5, 50, 95])])),
        frac_contrast_below_noise_clean=float((contrast[cl] < NOISE).mean()),
        frac_contrast_below_3x_noise_clean=float((contrast[cl] < 3 * NOISE).mean()),
        nonmonotonic_prog_angle_frac_clean=float(((a["status"][cl] & 32) > 0).mean()),
        grid_overflow_frac_clean=float(((a["status"][cl] & 8) > 0).mean()),
        unbound_frac_clean=float((~a["bound"][cl]).mean()))


def fig_strrings(d, s, out) -> None:
    a, m = s["a"], s["m"]
    obs = np.array([row[1:] for row in STRRINGS])
    o_len, o_ang = obs[:, 1], obs[:, 3]
    o_rad = o_len / np.deg2rad(o_ang)  # radius of the circular arc with that length and angle
    has = m["has_bin"]
    sim_rad = a["arc_len_kpc"] / np.deg2rad(np.maximum(a["ang_len_deg"], 1e-6))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ax = axes[0]
    ex, ey = np.linspace(0, 720, 37), np.logspace(0, 3.6, 37)
    h, _, _ = np.histogram2d(a["ang_len_deg"][has], np.clip(a["arc_len_kpc"][has], 1, None), bins=[ex, ey])
    im = ax.pcolormesh(ex, ey, np.where(h > 0, h, np.nan).T, cmap=P.SEQ, norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=h.max()))
    ax.plot(o_ang, o_len, "o", ms=6.5, color=P.INK, mfc="#eda100", mew=1.0, ls="none", label="STRRINGS (35 streams)")
    ax.axvline(d.cfg["sel_ang_deg"], color=P.INK2, lw=1.0, ls=":")
    ax.axhline(d.cfg["sel_arc_kpc"], color=P.INK2, lw=1.0, ls=":")
    ax.set_yscale("log")
    ax.grid(False)
    ax.set_xlabel("angular length [deg]  (simulation: valid bins x 10 deg)")
    ax.set_ylabel("track length [kpc]")
    ax.legend(loc="lower right")
    fig.colorbar(im, ax=ax, label="prior draws with a valid bin", pad=0.02)
    ax.set_title("Length against angle (dotted: the C26 cuts)")
    ax = axes[1]
    bins = np.logspace(0, 3.2, 49)
    for key, col, lbl in (("has_bin", C_ALL, "all draws with a valid bin"), ("cut", C_CUT, "passing the three cuts"),
                          ("clean", C_CLEAN, "cuts + resolved orbit")):
        x = np.clip(sim_rad[m[key]], bins[0], bins[-1])
        hh, e = np.histogram(x, bins=bins, weights=np.full(len(x), 1.0 / max(len(x), 1)))
        ax.stairs(hh, e, color=col, lw=2.0, label=lbl)
    ymax = ax.get_ylim()[1]
    ax.plot(o_rad, np.full(len(o_rad), 0.04 * ymax), "|", ms=16, mew=1.6, color=P.INK, label="STRRINGS (35 streams)")
    ax.set_xscale("log")
    ax.set_xlabel("characteristic radius = track length / angle [kpc]")
    ax.set_ylabel("fraction of the group")
    ax.legend(loc="upper left")
    ax.set_title("Size of the streams")
    ax = axes[2]
    # simulations in the neighbourhood of each observed stream (0.2 dex in length, 30 deg in angle): among every
    # draw with a resolved orbit (no cut), and among those that also pass the three C26 cuts
    cl, res = m["clean"], m["resolved"] & has
    la, aa = np.log10(np.clip(a["arc_len_kpc"], 1e-3, None)), a["ang_len_deg"]
    box = [(np.abs(la - np.log10(L)) < 0.2) & (np.abs(aa - A) < 30) for L, A in zip(o_len, o_ang)]
    near = np.array([int((b & cl).sum()) for b in box])
    near_res = np.array([int((b & res).sum()) for b in box])
    fails = ~((o_ang > d.cfg["sel_ang_deg"]) & (o_len > d.cfg["sel_arc_kpc"]))
    order = np.lexsort((near, near_res))
    yy = np.arange(len(near))
    ax.barh(yy, np.maximum(near_res[order], 0.5), height=0.7, color=C_ALL, label="resolved orbit, no cut")
    ax.plot(np.maximum(near[order], 0.5), yy, "o", ms=5, color=C_CLEAN, mec=P.SURFACE, mew=0.8, ls="none",
            label="resolved orbit + the three cuts")
    ax.set_yticks(yy, [STRRINGS[i][0] + (" *" if fails[i] else "") for i in order], fontsize=6.5)
    ax.set_xscale("log")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_xlabel("simulations within 0.2 dex in length and 30 deg in angle (0 drawn at 0.5)\n"
                  "* observed stream below the C26 length or angle cut")
    ax.set_title(f"Training examples near each observed stream (of {len(la):,} draws)")
    fig.suptitle("Prior predictive against the STRRINGS sample (arXiv:2508.02154, table 4); lengths are measured "
                 "differently in data and simulation, so this is an order-of-magnitude comparison", fontsize=10.5)
    fig.tight_layout()
    P.save(fig, d, "prior_inspection/strrings")
    pct = lambda x: dict(zip(("p5", "p16", "p50", "p84", "p95"), [float(v) for v in np.nanpercentile(x, [5, 16, 50, 84, 95])]))
    out["strrings"] = dict(
        n=len(STRRINGS),
        obs_pass_ang_cut=int((o_ang > d.cfg["sel_ang_deg"]).sum()), obs_pass_arc_cut=int((o_len > d.cfg["sel_arc_kpc"]).sum()),
        obs_pass_both=int(((o_ang > d.cfg["sel_ang_deg"]) & (o_len > d.cfg["sel_arc_kpc"])).sum()),
        obs_char_radius_kpc=pct(o_rad), sim_char_radius_kpc_cut=pct(sim_rad[m["cut"]]), sim_char_radius_kpc_clean=pct(sim_rad[cl]),
        sim_r_track_mean_kpc_clean=pct(a["r_track_mean"][cl]),
        frac_clean_char_radius_below_60=float((sim_rad[cl] < 60).mean()),
        frac_hasbin_char_radius_below_60=float((sim_rad[has] < 60).mean()),
        neighbours_per_observed_stream={r[0]: dict(resolved_no_cut=int(v0), resolved_and_cuts=int(v1), fails_c26_cuts=bool(f))
                                        for r, v0, v1, f in zip(STRRINGS, near_res, near, fails)},
        neighbours_median=dict(resolved_no_cut=float(np.median(near_res)), resolved_and_cuts=float(np.median(near))),
        n_observed_with_fewer_than_100_neighbours=dict(resolved_no_cut=int((near_res < 100).sum()), resolved_and_cuts=int((near < 100).sum())),
        n_observed_with_no_neighbour=dict(resolved_no_cut=int((near_res == 0).sum()), resolved_and_cuts=int((near == 0).sum())))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="outputs/extragalactic/C26_test")
    ap.add_argument("--subdir", default="")
    arg = ap.parse_args()
    d = P.Data(REPO / arg.out, arg.subdir)
    (d.pdir / "prior_inspection").mkdir(parents=True, exist_ok=True)
    s = load(d)
    out: dict = dict(n_streams=len(s["th"]), dE_max=DE_MAX)
    for f in (fig_funnel, fig_bound, fig_acceptance, fig_maps, fig_integrator, fig_observable, fig_strrings):
        f(d, s, out)
    p = d.ddir / "prior_inspection.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
