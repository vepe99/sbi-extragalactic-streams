"""Prior inspection before training an amortized posterior estimator on the C26 prior-predictive set.

    uv run python scripts/extragalactic/inspect_prior.py                 # production set
    uv run python scripts/extragalactic/inspect_prior.py --subdir pilot  # the 1,000-stream pilot

Reads data/analysis.h5 and the shard tracks (written by scripts/extragalactic/generate.py), writes figures to
plots/prior_inspection/ and the numbers behind them to data/prior_inspection.json. Nothing is simulated here.

Questions asked of the prior:
  1. funnel       how many of the prior draws are usable training examples
  2. acceptance   which parameters the selection cuts act on (the prior a network trained on the cut set sees)
  3. maps         where in parameter space the usable streams come from
  4. integrator   where the 99-step leapfrog does not resolve the progenitor orbit
  5. observable   what the network would receive: bins per track, coverage in angle, radial signal against noise
  6. strrings     the prior predictive against the observed STRRINGS tracks (arXiv:2508.02154), binned as the
                  simulations are; the tracks are the ones released with StreaMAX-fit (fetched on first use)
  7. corner       the prior with and without the unbound draws, pair by pair
  8. neighbours   the simulated tracks closest to each STRRINGS stream, and the parameters they come from
  9. boundness    bound against unbound streams among the usable ones: gallery and track summaries

`--only funnel,corner,...` runs a subset (names: funnel, bound, acceptance, maps, integrator, observable, strrings,
corner, neighbours, boundness).
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "extragalactic"))
import plot as P  # noqa: E402  palette, matplotlib style and the data loader of the figure script

plt = P.plt
C_ALL, C_CUT, C_CLEAN = P.C_ALL, P.C_BOUND, P.C_PASS  # all draws | passing the C26 cuts | cuts + resolved orbit
DE_MAX = 1e-2  # relative energy error above which the progenitor orbit counts as unresolved (report.md)
NOISE = 0.02  # C26 likelihood: 2 % radial noise per bin

# Observed tracks: the 32 STRRINGS streams fitted in Chemaly et al. 2026b, as released with StreaMAX-fit
# (data/strrings/<name>.npz: theta [rad, unwrapped], r [kpc], r_err [kpc] in 10 deg bins around the host centre).
STRRINGS_REPO = "https://github.com/David-Chemaly/StreaMAX-fit.git"
STRRINGS_COMMIT = "2e6a7085c3348cd6963387ab06e2bcbe540b2453"
# Decision of 2026-11-22 (Giuseppe): NGC1084, NGC1121 and PGC021008 are excluded from every comparison and from
# any later inference. They are the three of the 35 streams of arXiv:2508.02154 that StreaMAX-fit does not
# release (its export script drops the tracks flagged as measurement failures), so the 32 files are the sample.
STRRINGS_EXCLUDED = ("NGC1084_GROUP", "NGC1121", "PGC021008")
# Stream widths [kpc] from table 4 of arXiv:2508.02154, transcribed from the arXiv HTML and not checked against
# the journal table. Used only for one rug in the bound / unbound comparison.
STRRINGS_WIDTH_KPC = {
    "ESO079-003_GROUP": 6.1, "ESO298-016": 10.2, "ESO356-012": 4.3, "IC0120": 5.9, "IC1124": 5.0, "IC1657_GROUP": 4.5,
    "NGC0804": 6.9, "NGC1162": 7.9, "NGC2769": 10.2, "NGC3041": 3.4, "NGC4256_GROUP": 8.8, "NGC4388_GROUP": 7.2,
    "NGC4632": 2.8, "NGC4686": 8.7, "NGC5055_GROUP": 2.6, "NGC5263": 5.4, "NGC5387": 2.1, "NGC5513_GROUP": 5.9,
    "NGC5907": 4.6, "PGC000902": 7.6, "PGC006791": 4.5, "PGC039258": 3.4, "PGC1001085": 5.4, "PGC1092512": 5.4,
    "PGC3092153": 2.2, "PGC430221": 4.0, "PGC938075": 0.1, "UGC01245": 11.1, "UGC01424": 2.4, "UGC01970": 5.8,
    "UGC08717": 5.1, "UGC09239": 8.7,
}  # fmt: skip
C_OBS = "#eda100"
N_COVER = 50  # particles a simulated bin needs to be compared with an observed one (coverage rule, data README),
# for 10,000 particles per stream; `main` scales it with the particle number of the set


def fetch_strrings(dst: Path) -> None:
    """Copy the released tracks from StreaMAX-fit at the pinned commit into data/strrings/ (not tracked by git)."""
    with tempfile.TemporaryDirectory() as tmp:
        git = lambda *a: subprocess.run(["git", "-C", tmp, *a], check=True, capture_output=True)
        git("init", "-q")
        git("fetch", "-q", "--depth", "1", STRRINGS_REPO, STRRINGS_COMMIT)
        git("checkout", "-q", "FETCH_HEAD")
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(Path(tmp, "data", "strrings").glob("*.npz")):
            shutil.copy(f, dst / f.name)
        shutil.copy(Path(tmp, "data", "strrings_fit_settings.csv"), dst / "strrings_fit_settings.csv")
        shutil.copy(Path(tmp, "LICENSE"), dst / "LICENSE")
    (dst / "SOURCE.txt").write_text(f"{STRRINGS_REPO} @ {STRRINGS_COMMIT}, data/strrings/\n")
    print(f"fetched the STRRINGS tracks into {dst}")


def load_strrings(obs_dir: Path, cfg: dict) -> list[dict]:
    """The observed tracks on the grid of the simulated ones.

    Convention of the StreaMAX-fit likelihood: the track is centred on its median angle, which is where the
    model puts the progenitor (phi = 0). The points are then moved by at most half a bin so that they fall on
    the centres of the simulation's 10 deg bins. Length, angle and radii are computed exactly as for the
    simulated tracks (`track_summaries` in generate.py), so the three C26 cuts apply to them unchanged."""
    if not list(obs_dir.glob("*.npz")):
        fetch_strrings(obs_dir)
    bw, pmax = cfg["bin_deg"], cfg["phi_max_deg"]
    nb = int(round(2 * pmax / bw))
    obs = []
    for f in sorted(obs_dir.glob("*.npz")):
        with np.load(f, allow_pickle=False) as z:
            th, r, e = np.rad2deg(z["theta"]), z["r"].astype(float), z["r_err"].astype(float)
        name = f.stem
        assert name not in STRRINGS_EXCLUDED, name
        thc = th - np.median(th)
        res = (thc[0] - bw / 2) % bw  # distance of the points from the bin centres
        shift = -res if res <= bw / 2 else bw - res
        thc = thc + shift
        bins = np.floor((thc + pmax) / bw + 1e-6).astype(int)
        assert np.allclose((thc + pmax) / bw - bins, 0.5, atol=0.02), f"{name}: points are not on a {bw:g} deg grid"
        assert len(np.unique(bins)) == len(bins) and bins.min() >= 0 and bins.max() < nb, name
        adj = np.diff(bins) == 1
        chord = np.sqrt(np.maximum(r[:-1] ** 2 + r[1:] ** 2 - 2 * r[:-1] * r[1:] * np.cos(np.deg2rad(bw)), 0.0))
        ang, arc = len(bins) * bw, float(chord[adj].sum())
        obs.append(dict(
            name=name, phi=thc, r=r, r_err=e, bins=bins, shift_deg=float(shift), ang_len=float(ang), arc_len=arc,
            ang_span=float(thc.max() - thc.min() + bw), r_mean=float(r.mean()), r_min=float(r.min()), r_max=float(r.max()),
            contrast=float((r.max() - r.min()) / r.mean()), frac_err=float(np.median(e / r)),
            contiguous=bool(adj.all()), width=STRRINGS_WIDTH_KPC.get(name, np.nan),
            sel_ang=bool(ang > cfg["sel_ang_deg"]), sel_arc=bool(arc > cfg["sel_arc_kpc"]),
            sel_rad=bool((r.min() >= cfg["sel_r_min"]) & (r.max() <= cfg["sel_r_max"]))))
    for o in obs:
        o["sel_all"] = o["sel_ang"] and o["sel_arc"] and o["sel_rad"]
    assert len(obs) == 32, f"expected the 32 released tracks, found {len(obs)}"
    return obs


def load_sim_tracks(d: P.Data) -> dict:
    """count, median radius and 16-84 half-width per bin (the statistics the StreaMAX-fit likelihood uses)."""
    import h5py

    cols: dict[str, list] = {}
    for f in sorted(d.ddir.glob("shard_*.h5")):
        with h5py.File(f, "r") as h:
            cols.setdefault("index", []).append(h["index"][:])
            for k in ("count", "r_med", "r_hw"):
                cols.setdefault(k, []).append(h["track"][k][:])
    t = {k: np.concatenate(v) for k, v in cols.items()}
    o = np.argsort(t["index"])
    return {k: v[o] for k, v in t.items()}


def track_distance(sim: dict, o: dict, pool: np.ndarray, nb: int):
    """Root-mean-square fractional difference between each simulated track and the observed one over the
    observed bins, for the simulation as it is and for its mirror image (phi -> -phi: the sense of rotation of
    an observed stream is not known). inf where a bin has fewer than N_COVER particles."""
    out = []
    for bins in (o["bins"], nb - 1 - o["bins"]):
        ok = pool & (sim["count"][:, bins] >= N_COVER).all(1)
        dist = np.full(len(pool), np.inf)
        dist[ok] = np.sqrt(np.mean((sim["r_med"][ok][:, bins].astype(np.float64) / o["r"] - 1.0) ** 2, axis=1))
        out.append(dist)
    mirrored = out[1] < out[0]
    return np.minimum(out[0], out[1]), mirrored


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
    return dict(a=a, th=th, m=m, derived=derived, obs=load_strrings(d.obs_dir, d.cfg))


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
    ax.set_xlabel(r"$\log_{10}$ relative energy error of the progenitor orbit")
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
    o_ang = np.array([o["ang_len"] for o in s["obs"]])
    ax.plot(np.clip(o_ang, 0, 535), np.full(len(o_ang), 0.03), "|", ms=14, mew=1.6, color=P.INK,
            label=f"STRRINGS angular lengths ({len(o_ang)})")
    ax.set_ylim(0, 1)
    ax.set_xlabel("angular length of the track [deg]")
    ax.set_ylabel("fraction with energy error > 1 %")
    ax.legend(loc="upper left")
    how = (f"the {d.cfg['n_steps']}-step leapfrog" if d.cfg.get("integrator", "leapfrog") == "leapfrog"
           else f"galax's adaptive integrator (at most {d.cfg['max_steps']:,} steps)")
    fig.suptitle(f"Where {how} does not resolve the progenitor orbit", fontsize=10.5)
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
    a, m, obs = s["a"], s["m"], s["obs"]
    sim = s.setdefault("sim", load_sim_tracks(d))
    nb = sim["count"].shape[1]
    o_len, o_ang = np.array([o["arc_len"] for o in obs]), np.array([o["ang_len"] for o in obs])
    o_rad = np.array([o["r_mean"] for o in obs])
    o_pass = np.array([o["sel_all"] for o in obs])
    has = m["has_bin"]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6), gridspec_kw=dict(width_ratios=[1.1, 1, 1.15]))
    ax = axes[0]
    ex, ey = np.linspace(0, 540, 28), np.logspace(0, 3.6, 37)
    h, _, _ = np.histogram2d(a["ang_len_deg"][has], np.clip(a["arc_len_kpc"][has], 1, None), bins=[ex, ey])
    im = ax.pcolormesh(ex, ey, np.where(h > 0, h, np.nan).T, cmap=P.SEQ, norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=h.max()))
    ax.plot(o_ang[o_pass], o_len[o_pass], "o", ms=6.5, color=P.INK, mfc=C_OBS, mew=1.0, ls="none",
            label=f"STRRINGS, passes the three cuts ({int(o_pass.sum())})")
    ax.plot(o_ang[~o_pass], o_len[~o_pass], "o", ms=6.5, color=P.INK, mfc=P.SURFACE, mew=1.0, ls="none",
            label=f"STRRINGS, fails a cut ({int((~o_pass).sum())})")
    ax.axvline(d.cfg["sel_ang_deg"], color=P.INK2, lw=1.0, ls=":")
    ax.axhline(d.cfg["sel_arc_kpc"], color=P.INK2, lw=1.0, ls=":")
    ax.set_yscale("log")
    ax.grid(False)
    ax.set_xlabel("angular length [deg]  (bins x 10 deg)")
    ax.set_ylabel("track length [kpc]")
    ax.legend(loc="lower right")
    fig.colorbar(im, ax=ax, label="prior draws with a valid bin", pad=0.02)
    ax.set_title("Length against angle (dotted: the C26 cuts)")
    ax = axes[1]
    bins = np.logspace(-0.5, 3.2, 53)
    for key, col, lbl in (("has_bin", C_ALL, "all draws with a valid bin"), ("cut", C_CUT, "passing the three cuts"),
                          ("clean", C_CLEAN, "cuts + resolved orbit")):
        x = np.clip(a["r_track_mean"][m[key]], bins[0], bins[-1])
        hh, e = np.histogram(x, bins=bins, weights=np.full(len(x), 1.0 / max(len(x), 1)))
        ax.stairs(hh, e, color=col, lw=2.0, label=lbl)
    ax.plot(o_rad, np.full(len(o_rad), 0.04 * ax.get_ylim()[1]), "|", ms=16, mew=1.6, color=P.INK, label=f"STRRINGS ({len(obs)})")
    ax.set_xscale("log")
    ax.set_xlabel("mean radius of the track [kpc]")
    ax.set_ylabel("fraction of the group")
    ax.legend(loc="upper left")
    ax.set_title("Size of the streams")
    # --- how many simulations can be compared with each observed track, and how many resemble it
    ax = axes[2]
    pool = m["has_bin"] & m["resolved"]
    rows = []
    for o in obs:
        dist, _ = track_distance(sim, o, pool, nb)
        rows.append((o["name"], int(np.isfinite(dist).sum()), int((dist < 0.2).sum()), int((dist < 0.1).sum()), float(dist.min())))
    order = sorted(range(len(rows)), key=lambda i: (rows[i][3], rows[i][2], rows[i][1]))
    yy = np.arange(len(rows))
    z = lambda v: np.maximum(np.array(v, float), 0.5)
    ax.barh(yy, z([rows[i][1] for i in order]), height=0.72, color=C_ALL, alpha=0.35, label="cover every observed bin")
    ax.barh(yy, z([rows[i][2] for i in order]), height=0.72, color=C_ALL, label="and within 20 % rms in radius")
    ax.plot(z([rows[i][3] for i in order]), yy, "o", ms=5, color=C_CLEAN, mec=P.SURFACE, mew=0.8, ls="none", label="within 10 % rms")
    ax.set_yticks(yy, [rows[i][0] + (" *" if not obs[i]["sel_all"] else "") for i in order], fontsize=6.5)
    ax.set_xscale("log")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_xlabel(f"simulations with a resolved orbit, of {int(pool.sum()):,} (0 drawn at 0.5)\n* observed track fails a C26 cut")
    ax.set_title("Simulated tracks comparable with each observed one")
    fig.suptitle(f"Prior predictive against the {len(obs)} observed STRRINGS tracks (StreaMAX-fit release), measured on the same "
                 "10 deg bins as the simulations", fontsize=10.5)
    fig.tight_layout()
    P.save(fig, d, "prior_inspection/strrings")
    pct = lambda x: dict(zip(("p5", "p16", "p50", "p84", "p95"), [float(v) for v in np.nanpercentile(x, [5, 16, 50, 84, 95])]))
    cl = m["clean"]
    out["strrings"] = dict(
        n=len(obs), excluded=list(STRRINGS_EXCLUDED), source=f"{STRRINGS_REPO} @ {STRRINGS_COMMIT[:7]}",
        obs_pass=dict(ang=int(sum(o["sel_ang"] for o in obs)), arc=int(sum(o["sel_arc"] for o in obs)),
                      rad=int(sum(o["sel_rad"] for o in obs)), all=int(o_pass.sum())),
        obs_fail_names=dict(ang=[o["name"] for o in obs if not o["sel_ang"]], arc=[o["name"] for o in obs if not o["sel_arc"]],
                            rad=[o["name"] for o in obs if not o["sel_rad"]]),
        obs_longer_than_270deg=int((o_ang > 270).sum()), obs_contiguous=int(sum(o["contiguous"] for o in obs)),
        obs_ang_len_deg=pct(o_ang), obs_arc_len_kpc=pct(o_len), obs_r_mean_kpc=pct(o_rad),
        obs_radial_contrast=pct([o["contrast"] for o in obs]), obs_frac_err=pct([o["frac_err"] for o in obs]),
        sim_r_track_mean_kpc=dict(all=pct(a["r_track_mean"][has]), cut=pct(a["r_track_mean"][m["cut"]]), clean=pct(a["r_track_mean"][cl])),
        frac_sim_r_mean_below_obs_p95=dict(all=float((a["r_track_mean"][has] < np.percentile(o_rad, 95)).mean()),
                                           clean=float((a["r_track_mean"][cl] < np.percentile(o_rad, 95)).mean())),
        frac_sim_cut_longer_than_270deg=float((a["ang_len_deg"][m["cut"]] > 270).mean()),
        per_stream={r_[0]: dict(n_bins=len(o["bins"]), ang_len_deg=o["ang_len"], arc_len_kpc=o["arc_len"], r_mean_kpc=o["r_mean"],
                                passes_c26_cuts=o["sel_all"], n_covering=r_[1], n_within_20pct=r_[2], n_within_10pct=r_[3],
                                best_rms=r_[4]) for r_, o in zip(rows, obs)},
        n_streams_without_sim_within_10pct=int(sum(r_[3] == 0 for r_ in rows)),
        n_streams_with_fewer_than_100_within_20pct=int(sum(r_[2] < 100 for r_ in rows)))


# ------------------------------------------------------------------------------------------------------
# 7. corner: the prior with and without the unbound draws
# ------------------------------------------------------------------------------------------------------
PAR_LABELS = [r"$\log_{10} M_{\rm halo}$", r"$R_s$ [kpc]", r"$q$", r"$\cos i$", r"$\psi$ [rad]",
              r"$\log_{10} m_{\rm prog}$", r"$r_s$ [kpc]", r"$x_0$ [kpc]", r"$z_0$ [kpc]",
              r"$v_{x0}$ [km/s]", r"$v_{y0}$ [km/s]", r"$v_{z0}$ [km/s]", r"$t_{\rm int}$ [Gyr]"]


def corner_fraction(d, cols, labels, mask, name, title, nb=18, size=1.75) -> None:
    """Corner layout. Diagonal: the two 1D distributions on one scale (share of all draws per bin). Below the
    diagonal: the fraction of draws kept by `mask` in each 2D cell. A flat prior has no meaningful density
    contour, so the pairs show the kept fraction instead of two sets of contours."""
    n = len(cols)
    rng = []
    for x in cols:
        lo, hi = np.nanpercentile(x, [0.5, 99.5])
        flat = (x.max() - x.min()) < 1.25 * (hi - lo)  # bounded (uniform) parameter: use its full range
        rng.append((float(x.min()), float(x.max())) if flat else (float(lo), float(hi)))
    fig, axes = plt.subplots(n, n, figsize=(size * n + 0.8, size * n))
    im = None
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            ax.grid(False)
            if i == j:
                e = np.linspace(*rng[i], 31)
                ha, _ = np.histogram(cols[i], bins=e)
                hb, _ = np.histogram(cols[i][mask], bins=e)
                ax.stairs(hb / len(mask), e, color=C_CUT, fill=True, alpha=0.30, lw=0)
                ax.stairs(hb / len(mask), e, color=C_CUT, lw=1.5)
                ax.stairs(ha / len(mask), e, color=C_ALL, lw=1.5)
                ax.set_ylim(0, 1.1 * ha.max() / len(mask))
                ax.set_yticks([])
                ax.set_xlim(*rng[i])
            else:
                ex, ey = np.linspace(*rng[j], nb + 1), np.linspace(*rng[i], nb + 1)
                na, _, _ = np.histogram2d(cols[j], cols[i], bins=[ex, ey])
                nk, _, _ = np.histogram2d(cols[j][mask], cols[i][mask], bins=[ex, ey])
                with np.errstate(all="ignore"):
                    f = np.where(na >= 20, nk / na, np.nan)
                im = ax.pcolormesh(ex, ey, f.T, cmap=P.SEQ, vmin=0, vmax=1, rasterized=True)
            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=8.5)
                ax.tick_params(axis="x", labelsize=6.5, labelrotation=45)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=8.5)
                ax.tick_params(axis="y", labelsize=6.5)
            elif i != j:
                ax.set_yticklabels([])
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.07, top=0.95, wspace=0.07, hspace=0.07)
    cax = fig.add_axes([0.945, 0.07, 0.012, 0.5])
    fig.colorbar(im, cax=cax, label="fraction of the draws in the cell that are bound (cells with < 20 draws blank)")
    h = [plt.Line2D([], [], color=C_ALL, lw=1.5, label=f"all prior draws ({len(mask):,})"),
         plt.Line2D([], [], color=C_CUT, lw=1.5, label=f"bound only ({int(mask.sum()):,})")]
    fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.93, 0.93), fontsize=10 if n > 6 else 9,
               title="diagonal (same scale)")
    fig.suptitle(title, fontsize=11 if n > 6 else 10, x=0.5, y=0.985)
    P.save(fig, d, f"prior_inspection/{name}")


def fig_corner(d, s, out) -> None:
    th, m, dv = s["th"], s["m"], s["derived"]
    corner_fraction(d, [th[:, i] for i in range(13)], PAR_LABELS, m["bound"], "corner_bound_vs_all",
                    "The prior with and without the unbound draws: all 13 sampled parameters")
    cols = [th[:, 0], th[:, 1], th[:, 2], dv["r0"], dv["speed"]]
    labels = [PAR_LABELS[0], PAR_LABELS[1], PAR_LABELS[2], r"3D distance today $|x_0|$ [kpc]", r"speed today $|v_0|$ [km/s]"]
    corner_fraction(d, cols, labels, m["bound"], "corner_bound_vs_all_focus",
                    "The prior with and without the unbound draws: the quantities that decide it", nb=24, size=2.5)


# ------------------------------------------------------------------------------------------------------
# 8. nearest simulated neighbours of the STRRINGS streams
# ------------------------------------------------------------------------------------------------------
def draw_sim_track(ax, d, sim, row, col, mirrored: bool) -> None:
    """A simulated track on the sky: median radius of the bins with at least N_COVER particles, with the 16-84
    half-width as a band; drawn mirrored (y -> -y) where the mirror image is the better match."""
    ok = sim["count"][row] >= N_COVER
    phi = np.deg2rad(d.phi[ok]) * (-1.0 if mirrored else 1.0)
    r, w = sim["r_med"][row][ok].astype(float), sim["r_hw"][row][ok].astype(float)
    brk = np.nonzero(np.diff(np.nonzero(ok)[0]) > 1)[0] + 1  # do not join across gaps in the track
    for seg in np.split(np.arange(len(r)), brk):
        if len(seg) == 0:
            continue
        lo, hi, ph = np.maximum(r[seg] - w[seg], 0.0), r[seg] + w[seg], phi[seg]
        ax.fill(np.concatenate([hi * np.cos(ph), (lo * np.cos(ph))[::-1]]),
                np.concatenate([hi * np.sin(ph), (lo * np.sin(ph))[::-1]]), color=col, alpha=0.25, lw=0)
        ax.plot(r[seg] * np.cos(ph), r[seg] * np.sin(ph), "-", color=col, lw=1.5)


def fig_neighbours(d, s, out, k: int = 5, per_page: int = 8) -> None:
    a, th, m, obs_all = s["a"], s["th"], s["m"], s["obs"]
    sim = s.setdefault("sim", load_sim_tracks(d))
    nb = sim["count"].shape[1]
    pool = m["has_bin"] & m["resolved"]
    obs = sorted(obs_all, key=lambda o: o["ang_len"])  # short arcs first, loops last
    info, pick = {}, []
    pages = [obs[i : i + per_page] for i in range(0, len(obs), per_page)]
    for ip, page in enumerate(pages, 1):
        fig, axes = plt.subplots(len(page), k + 1, figsize=(2.5 * (k + 1) + 1.0, 2.65 * len(page)), squeeze=False,
                                 gridspec_kw=dict(width_ratios=[1.45] + [1] * k))
        for i, o in enumerate(page):
            dist, mirrored = track_distance(sim, o, pool, nb)
            order = np.argsort(dist)[:k]
            order = order[np.isfinite(dist[order])]
            near = np.argsort(dist)[:100]
            pick.append(near[dist[near] < 0.2])
            info[o["name"]] = dict(
                n_bins=len(o["bins"]), n_covering=int(np.isfinite(dist).sum()), n_within_20pct=int((dist < 0.2).sum()),
                n_within_10pct=int((dist < 0.1).sum()), neighbours=[int(a["index"][r_]) for r_ in order],
                rms=[float(dist[r_]) for r_ in order], mirrored=[bool(mirrored[r_]) for r_ in order],
                bound=[bool(m["bound"][r_]) for r_ in order])
            # --- the radial bins: observed points with their errors, and the neighbours on the same bins
            ax = axes[i, 0]
            for r_ in order:
                bins = (nb - 1 - o["bins"]) if mirrored[r_] else o["bins"]
                ax.plot(o["phi"], sim["r_med"][r_][bins], "-", color=C_CUT if m["bound"][r_] else C_ALL, lw=1.3, alpha=0.85)
            ax.errorbar(o["phi"], o["r"], yerr=o["r_err"], fmt="o", ms=3.2, color=P.INK, mfc=P.SURFACE, mew=0.9, elinewidth=0.9, zorder=5)
            ax.set_ylim(0, 1.25 * max(o["r_max"], max((float(sim["r_med"][r_][o["bins"]].max()) for r_ in order), default=0.0)))
            ax.set_xlabel(r"$\phi$ [deg], 0 at the median angle", fontsize=7)
            ax.set_ylabel("r [kpc]", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_title(f"{o['name']}{'' if o['sel_all'] else ' *'}: {len(o['bins'])} bins, median error {100 * o['frac_err']:.1f} %\n"
                         f"{info[o['name']]['n_covering']:,} sims cover it, {info[o['name']]['n_within_10pct']:,} within 10 % rms",
                         fontsize=6.8)
            # --- the same on the sky
            lim = 1.35 * o["r_max"]
            ox, oy = o["r"] * np.cos(np.deg2rad(o["phi"])), o["r"] * np.sin(np.deg2rad(o["phi"]))
            for jn in range(1, k + 1):
                ax = axes[i, jn]
                if jn > len(order):
                    ax.axis("off")
                    continue
                r_ = order[jn - 1]
                draw_sim_track(ax, d, sim, r_, C_CUT if m["bound"][r_] else C_ALL, bool(mirrored[r_]))
                ax.plot(ox, oy, "o", ms=2.6, color=P.INK, mfc=P.INK, ls="none", zorder=5)
                ax.plot([0], [0], marker="+", ms=6, color=P.INK2)
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_aspect("equal")
                ax.tick_params(labelsize=6)
                ax.set_title(f"#{int(a['index'][r_])}  rms {100 * dist[r_]:.1f} %  {'bound' if m['bound'][r_] else 'unbound'}"
                             f"{', mirrored' if mirrored[r_] else ''}\nq={th[r_, 2]:.2f}, log M={th[r_, 0]:.1f}, "
                             f"t={th[r_, 12]:.1f} Gyr", fontsize=6.8)
        h = [plt.Line2D([], [], marker="o", ms=4, color=P.INK, mfc=P.SURFACE, ls="none", label="observed radial bins (with errors in the first column)"),
             plt.Line2D([], [], color=C_CUT, lw=1.6, label="simulated track, bound progenitor (band: 16-84 half-width)"),
             plt.Line2D([], [], color=C_ALL, lw=1.6, label="simulated track, unbound progenitor")]
        fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.995), fontsize=8)
        fig.suptitle(f"The {k} simulated tracks closest to each observed STRRINGS track on its own radial bins; sky panels in kpc "
                     f"(page {ip} of {len(pages)}; * fails a C26 cut)", y=1.012, fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        P.save(fig, d, f"prior_inspection/strrings_neighbours_p{ip}")
    out["strrings_neighbours"] = dict(
        pool="resolved orbit; every observed bin has at least 50 particles; simulation or its mirror image",
        pool_size=int(pool.sum()), metric="rms of r_sim / r_obs - 1 over the observed bins, track centred on its median angle",
        per_stream=info)

    # --- where in the prior the neighbours come from: up to 100 per observed stream, all within 20 % rms
    pick = np.concatenate(pick)
    dv = s["derived"]
    cols = [th[:, i] for i in range(13)] + [dv["r0"], dv["speed"], a["n_peri"].astype(float)]
    labels = PAR_LABELS + [r"3D distance today $|x_0|$ [kpc]", r"speed today $|v_0|$ [km/s]", r"pericentres within $t_{\rm int}$"]
    names = d.names + ["r0", "speed", "n_peri"]
    fig, axes = plt.subplots(4, 4, figsize=(14, 11))
    ksn = {}
    for ax, x, lab, name in zip(axes.ravel(), cols, labels, names):
        lo, hi = np.nanpercentile(x, [0.5, 99.5])
        if (x.max() - x.min()) < 1.25 * (hi - lo):
            lo, hi = x.min(), x.max()
        e = np.arange(-0.5, 8.5) if name == "n_peri" else np.linspace(lo, hi, 31)
        for xx, col, lbl in ((x, C_ALL, f"all prior draws ({len(x):,})"),
                             (x[pick], C_CLEAN, f"neighbours of the {len(obs)} STRRINGS tracks ({len(pick):,})")):
            hh, _ = np.histogram(np.clip(xx, e[0], e[-1]), bins=e, density=True)
            ax.stairs(hh, e, color=col, lw=1.9, label=lbl)
        sel = np.zeros(len(x), bool)
        sel[np.unique(pick)] = True
        ksn[name] = ks_distance(x, sel)
        ax.text(0.97, 0.95, f"KS {ksn[name]:.2f}", transform=ax.transAxes, ha="right", va="top", color=P.INK2, fontsize=8)
        ax.set_ylim(0, 1.25 * ax.get_ylim()[1])
        ax.set_xlabel(lab)
        ax.set_yticks([])
        ax.grid(False)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Where in the prior the STRRINGS-like tracks come from (up to 100 nearest per observed track, within 20 % rms; "
                 "KS against the full prior)", y=1.0, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    P.save(fig, d, "prior_inspection/strrings_neighbour_parameters")
    out["strrings_neighbours"]["pooled"] = dict(
        n=int(len(pick)), n_unique=int(len(np.unique(pick))), bound_frac=float(m["bound"][pick].mean()),
        pass_c26_cuts_frac=float(m["cut"][pick].mean()), ks_vs_prior=ksn,
        median=dict(log10_M_halo=float(np.median(th[pick, 0])), q=float(np.median(th[pick, 2])),
                    r0_kpc=float(np.median(dv["r0"][pick])), speed_kms=float(np.median(dv["speed"][pick])),
                    x0_kpc=float(np.median(th[pick, 7])), t_int=float(np.median(th[pick, 12]))),
        prior_median=dict(log10_M_halo=float(np.median(th[:, 0])), r0_kpc=float(np.median(dv["r0"])),
                          x0_kpc=float(np.median(th[:, 7])), t_int=float(np.median(th[:, 12]))))


# ------------------------------------------------------------------------------------------------------
# 9. bound against unbound among the usable streams
# ------------------------------------------------------------------------------------------------------
def fig_boundness(d, s, out, n_show: int = 12, seed: int = 1) -> None:
    a, m, t = s["a"], s["m"], d.tracks
    usable = m["clean"]
    groups = (("bound", usable & m["bound"], C_CUT), ("unbound", usable & ~m["bound"], C_ALL))
    # --- galleries from the streams with stored particles (the first 5,000), drawn with the figure script
    have = np.isin(a["index"], d.particle_indices())
    rng = np.random.default_rng(seed)
    for name, mask, _ in groups:
        idx = a["index"][mask & have]
        ks = np.sort(rng.choice(idx, size=min(n_show, len(idx)), replace=False))
        fig, axes = plt.subplots(4, 6, figsize=(25, 16.5))
        for j, k_ in enumerate(ks):
            P.draw_stream(d, axes[j // 3, 2 * (j % 3)], axes[j // 3, 2 * (j % 3) + 1], int(k_), tag=f"{name} ")
        for j in range(len(ks), 12):
            axes[j // 3, 2 * (j % 3)].axis("off")
            axes[j // 3, 2 * (j % 3) + 1].axis("off")
        P._legend_handles(fig)
        fig.suptitle(f"{name.capitalize()} progenitors among the usable streams (three C26 cuts + resolved orbit): "
                     f"{len(ks)} drawn at random from the {len(idx)} with stored particles", y=1.012, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        fig.savefig(d.pdir / f"prior_inspection/gallery_{name}.png", dpi=110, bbox_inches="tight")
        fig.savefig(d.pdir / f"prior_inspection/gallery_{name}.pdf", dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {d.pdir / 'prior_inspection' / ('gallery_' + name)}.pdf / .png")
    # --- track summaries of the two groups
    valid = t["count"] >= d.cfg["n_valid_min"]
    r = np.where(valid, t["r_mean"], np.nan).astype(np.float64)
    with np.errstate(all="ignore"):
        contrast = (np.nanmax(r, 1) - np.nanmin(r, 1)) / np.nanmean(r, 1)
        relw = np.nanmedian(np.where(valid, t["r_std"] / t["r_mean"], np.nan), 1)
        char_r = a["r_track_mean"]
    ob = s["obs"]
    og = lambda key: np.array([o[key] for o in ob])
    specs = [("angular length [deg]", a["ang_len_deg"], np.arange(60, 500, 10), False, og("ang_len")),
             ("track length [kpc]", a["arc_len_kpc"], np.logspace(0.3, 3.3, 49), True, og("arc_len")),
             ("mean radius of the track [kpc]", char_r, np.logspace(-0.5, 2.8, 49), True, og("r_mean")),
             (r"relative width, median $r_{\rm std}/r$  (observed: table width / mean radius)", relw, np.logspace(-2, 0.2, 43), True,
              og("width") / og("r_mean")),
             (r"radial contrast $(r_{\max}-r_{\min})/\bar r$", contrast, np.logspace(-1, 0.8, 43), True, og("contrast")),
             (r"pericentres within $t_{\rm int}$", a["n_peri"].astype(float), np.arange(-0.5, 9.5), False, None)]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8))
    stats = {}
    for ax, (lab, x, bins, logx, rug) in zip(axes.ravel(), specs):
        for name, mask, col in groups:
            xx = np.clip(x[mask], bins[0], bins[-1])
            hh, e = np.histogram(xx, bins=bins, weights=np.full(len(xx), 1.0 / max(len(xx), 1)))
            ax.stairs(hh, e, color=col, lw=2.0, label=f"{name} ({int(mask.sum()):,})")
            stats.setdefault(lab, {})[name] = [float(v) for v in np.nanpercentile(x[mask], [16, 50, 84])]
        if rug is not None:
            ax.plot(np.clip(rug, bins[0], bins[-1]), np.full(len(rug), 0.035 * ax.get_ylim()[1]), "|", ms=13, mew=1.5,
                    color=P.INK, label=f"STRRINGS ({len(ob)})")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(lab + "   (range clipped)")
        ax.set_ylabel("fraction of the group")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Bound against unbound progenitors among the usable streams (three C26 cuts + resolved orbit)", fontsize=10.5)
    fig.tight_layout()
    P.save(fig, d, "prior_inspection/bound_vs_unbound_tracks")
    q = s["th"][:, 2]
    out["bound_vs_unbound_usable"] = dict(
        n=dict(bound=int(groups[0][1].sum()), unbound=int(groups[1][1].sum())), p16_p50_p84=stats,
        contiguous_frac={n_: float(a["contiguous"][mk].mean()) for n_, mk, _ in groups},
        frac_longer_than_180deg={n_: float((a["ang_len_deg"][mk] > 180).mean()) for n_, mk, _ in groups},
        frac_mean_radius_below_60kpc={n_: float((char_r[mk] < 60).mean()) for n_, mk, _ in groups},
        no_pericentre_frac={n_: float((a["n_peri"][mk] == 0).mean()) for n_, mk, _ in groups},
        q_mean={n_: float(q[mk].mean()) for n_, mk, _ in groups})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="outputs/extragalactic/C26_test")
    ap.add_argument("--subdir", default="")
    ap.add_argument("--only", default="", help="comma list of figure groups (see the module docstring)")
    arg = ap.parse_args()
    d = P.Data(REPO / arg.out, arg.subdir)
    (d.pdir / "prior_inspection").mkdir(parents=True, exist_ok=True)
    d.a  # reads analysis.h5 and the configuration
    d.obs_dir = REPO / arg.out / "data" / "strrings"
    global N_COVER
    N_COVER = int(round(50 * d.cfg["n_particles"] / 10_000))
    s = load(d)
    figs = dict(funnel=fig_funnel, bound=fig_bound, acceptance=fig_acceptance, maps=fig_maps, integrator=fig_integrator,
                observable=fig_observable, strrings=fig_strrings, corner=fig_corner, neighbours=fig_neighbours,
                boundness=fig_boundness)
    p = d.ddir / "prior_inspection.json"
    out: dict = json.loads(p.read_text()) if (arg.only and p.exists()) else {}  # a subset updates the file
    out.update(n_streams=len(s["th"]), dE_max=DE_MAX)
    for name in [x for x in arg.only.split(",") if x] or list(figs):
        figs[name](d, s, out)
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
