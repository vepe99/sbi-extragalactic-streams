"""Long streams of a simulation set: how many, what they are, whether their frame is right, what training sees.

    uv run python scripts/extragalactic/long_streams.py --config outputs/extragalactic/C26_test/npe_v2.yaml

Reads the shards and the particle files of `subdir` (and `n_min`, `var_ratio`, `span_max`, `ordering` of the
file), writes four figures and `long_streams.json` to <out>/plots/<subdir>/long_streams/. Nothing is simulated
and no GPU is used; the scan of 10,000,000 streams takes 20 s on 32 cores.

LENGTH of a stream = the unbroken run of 10 deg bins with at least `n_min` particles around the progenitor, in
the C26 ordering: the bins a mock observation can be cut from (guard 1 of 2026b). Long = more than 360 deg.

CLASSES. resolved = relative energy error of the progenitor orbit below 1 % with the simulator's steps
(`diag/dE_rel`, the criterion of report.md). monotonic = the projected angle of the progenitor never turns back
inside t_int (status bit 32 not set); where it does, the unwrapped angle folds and one bin holds two pieces of
the stream. thin = the 16-84 half-width of the radii in a bin is at most 25 % of the median radius (median over
the run); wide ones are clouds of particles around the host, not a line on the sky.

Figures:
  1_census    lengths by class, share of each class against length, long streams against radial periods
  2_checks    rotation of the stored frame against the energy error, C26 against Gibbons length, bin width
  3_sky       picked long streams: sky in the simulation frame, unwrapped track, a mock observation of
              `span_max` bins at most in the data frame (made by mock.py) on the particles rotated with it
  4_training  streams that can hold a track of S bins and the share of each class among the examples of S bins
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts" / "extragalactic"), str(REPO / "scripts" / "extragalactic" / "sbi")]
import plot as P  # noqa: E402  palette and matplotlib style of the figure script
from config import Config, pick_device  # noqa: E402
from data import Grid, load_observed, run_extents  # noqa: E402

plt = P.plt
DE_MAX = 1e-2  # relative energy error above which the progenitor orbit counts as unresolved (report.md)
LONG, SIG_GUARD = 360.0, 1.0  # deg; flat data error [kpc] at which guard 2 is evaluated for the whole set
WIDE = 0.25  # bin half-width over radius above which a stream counts as wide
# classes: (key, label, colour); the fourth colour is the next categorical slot of the palette in plot.py
CLASSES = (("thin", "resolved orbit, angle monotonic, thin", P.C_PASS), ("wide", "resolved orbit, angle monotonic, wide (a cloud)", "#eda100"),
           ("fold", "resolved orbit, angle folds back", P.C_ALL), ("unres", "unresolved orbit (energy error > 1 %)", P.C_BOUND))  # fmt: skip
# train_v2, among the 1,000 streams with particles: one per kind of long stream (see the titles in 3_sky)
PICKS = (629, 328, 990, 181, 731, 987)


# ------------------------------------------------------------------------------------------------------
# scan: one row per stream
# ------------------------------------------------------------------------------------------------------
def scan_shard(args) -> dict:
    path, n_min, var_ratio = args
    with h5py.File(path, "r") as f:
        grid = Grid.from_sims(json.loads(f.attrs["config"]))
        group = {f[g].attrs["ordering"]: g for g in ("track", "track_alt") if g in f}
        c, hw, rm = (f[f"{group['c26']}/{k}"][:] for k in ("count", "r_hw", "r_med"))
        o = dict(index=f["index"][:], theta=f["theta"][:], status=f["status"][:], dissolved=f["prog/dissolved"][:],
                 unbound=f["flags/unbound"][:], dE=f["diag/dE_rel"][:], r_min=f["diag/r_min"][:], r_max=f["diag/r_max"][:],
                 n_orb=np.maximum(f["diag/n_peri"][:], f["diag/n_apo"][:]), n_overflow=f["diag/n_overflow"][:])  # fmt: skip
        ca = f[f"{group['gibbons14']}/count"][:] if "gibbons14" in group else None
    valid = c >= n_min
    o["a"], o["b"] = (x.astype(np.int16) for x in run_extents(valid, grid.i0))
    if ca is not None:
        o["a_g14"], o["b_g14"] = (x.astype(np.int16) for x in run_extents(ca >= n_min, grid.i0))
    # guard 2 at a flat data error: model variance of the bin median <= data variance / var_ratio
    quiet = valid & (hw.astype(np.float64) ** 2 / np.maximum(c, 1) * var_ratio <= SIG_GUARD**2)
    o["a_q"], o["b_q"] = (x.astype(np.int16) for x in run_extents(quiet, grid.i0))
    j = np.arange(c.shape[1])[None]
    run = (j >= grid.i0 - o["a"][:, None]) & (j < grid.i0 + o["b"][:, None])
    with np.errstate(all="ignore"):  # relative 16-84 half-width of the radii in a bin, median over the run
        o["width"] = np.nanmedian(np.where(run, hw / rm, np.nan), axis=1).astype(np.float32)
    return o


def scan(cfg: Config, workers: int) -> dict:
    files = sorted(cfg.data_dir().glob("shard_*.h5"))[: cfg.max_shards or None]
    if not files:
        raise SystemExit(f"no shards in {cfg.data_dir()}")
    with Pool(workers) as pool:
        parts = pool.map(scan_shard, [(p, cfg.n_min, cfg.var_ratio) for p in files], chunksize=8)
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    with h5py.File(files[0], "r") as f:
        d["sim"] = json.loads(f.attrs["config"])
    bin_deg = d["sim"]["bin_deg"]
    d["L"] = (d["a"].astype(int) + d["b"]) * bin_deg
    if "a_g14" in d:
        d["L_g14"] = (d["a_g14"].astype(int) + d["b_g14"]) * bin_deg
    res, mono = d["dE"] < DE_MAX, (d["status"] & 32) == 0
    wide = d["width"] > WIDE
    d["cls"] = dict(thin=res & mono & ~wide, wide=res & mono & wide, fold=res & ~mono, unres=~res)
    return d


# ------------------------------------------------------------------------------------------------------
# particles: is the stored frame the one the files say it is?
# ------------------------------------------------------------------------------------------------------
def particle_checks(cfg: Config, d: dict) -> dict:
    """For every stream with stored particles: the rotation between the unwrapped angle and the angle on the sky,
    the stored track against one rebuilt from the particles, the progenitor against `prog/xy`, the side of the
    progenitor each arm lies on."""
    sc, grid = d["sim"], Grid.from_sims(d["sim"])
    n_per = sc["n_particles"] // (sc["n_steps"] + 1)  # particles per release epoch; the last epoch is today
    out = {k: [] for k in ("index", "rot", "rot_spread", "count_diff", "rmed_diff", "prog_rot", "prog_dr", "lead", "trail")}
    for p in sorted(cfg.data_dir().glob("particles_*.h5")):
        with h5py.File(p, "r") as f:
            idx, xy, phi = f["index"][:], f["xy"][:].astype(np.float64), f["phi"][:].astype(np.float64)
        shard = cfg.data_dir() / f"shard_{int(idx[0]) // sc['shard_size']:05d}.h5"
        with h5py.File(shard, "r") as f:
            row = np.searchsorted(f["index"][:], idx)
            group = {f[g].attrs["ordering"]: g for g in ("track", "track_alt") if g in f}["c26"]
            count, r_med, x0 = f[f"{group}/count"][:][row], f[f"{group}/r_med"][:][row], f["theta"][:][row, 7]
        r = np.hypot(xy[..., 0], xy[..., 1])
        res = (phi - np.rad2deg(np.arctan2(xy[..., 1], xy[..., 0])) + 180.0) % 360.0 - 180.0
        out["index"].append(idx)
        out["rot"].append(np.median(res, 1))
        out["rot_spread"].append(np.ptp(res, 1))
        b = np.floor((phi + sc["phi_max_deg"]) / sc["bin_deg"]).astype(int)
        cd, rd = np.zeros(len(idx), int), np.zeros(len(idx))
        for i in range(len(idx)):
            m = (b[i] >= 0) & (b[i] < grid.n_bins)
            cd[i] = np.abs(np.bincount(b[i][m], minlength=grid.n_bins) - count[i]).sum()
            o = np.argsort(b[i][m], kind="stable")
            bs, rs = b[i][m][o], r[i][m][o]
            for j, s, e in zip(*np.unique(bs, return_index=True)[:2], np.r_[np.unique(bs, return_index=True)[1][1:], len(bs)]):
                if e - s >= cfg.n_min:
                    rd[i] = max(rd[i], abs(np.median(rs[s:e]) / r_med[i, j] - 1.0))
        out["count_diff"].append(cd)
        out["rmed_diff"].append(rd)
        last = xy[:, -n_per:].mean(1)  # released today at +- the tidal radius: their mean is the progenitor
        out["prog_rot"].append(np.rad2deg(np.arctan2(last[:, 1], last[:, 0])))
        out["prog_dr"].append(np.hypot(last[:, 0], last[:, 1]) / x0 - 1.0)
        out["lead"].append((phi[:, 1::2] > 0).mean(1))  # odd particles: inner Lagrange point, the leading arm
        out["trail"].append((phi[:, 0::2] < 0).mean(1))
    return {k: np.concatenate(v) for k, v in out.items()}


def mock_windows(cfg: Config, d: dict, picks, n_keys: int = 512, seed: int = 0) -> dict:
    """A mock observation of each picked stream, made by the training code (`make_mock`): of `n_keys` draws the
    one with the longest window. -> per pick: S, l, the rotation of the data frame against the simulation frame
    (the median observed angle, which mock.py subtracts), the mock itself."""
    import jax
    import jax.random as jr

    from data import load_simulations
    from mock import make_mock

    sc = d["sim"]
    sims = load_simulations(dataclasses.replace(cfg, max_shards=max(picks) // sc["shard_size"] + 1))
    grid, fn = sims["grid"], None
    fn = jax.jit(jax.vmap(make_mock(cfg, grid, sims["design"]), in_axes=(0, None, None, None, None, None)))
    out = {}
    for k in picks:
        th = d["theta"][d["index"] == k][0].astype(np.float64)
        row = np.flatnonzero((sims["theta"] == th).all(1))
        if not len(row):  # dropped by load_simulations: no window fits
            continue
        s = {key: sims["data"][key][row[0]] for key in ("count", "r_med", "r_hw", "dissolved", "z_strata")}
        o = {key: np.asarray(v) for key, v in fn(jr.split(jr.key(seed + k), n_keys), *s.values()).items()}
        i = int(np.argmax(o["S"]))
        S, l = int(o["S"][i]), int(o["l"][i])
        rot = (0.5 - l) * grid.bin_deg - float(o["phi"][i][0])
        out[k] = dict(S=S, l=l, rot=rot, phi=o["phi"][i][:S], r=o["r"][i][:S], m=o["m"][i][:S], obs=o["mask"][i][:S],
                      err=np.sqrt(o["sig"][i][:S] ** 2 + o["sig_sys"][i] ** 2), sig_sys=float(o["sig_sys"][i]),
                      a=int(o["a"][i]), b=int(o["b"][i]))  # fmt: skip
    return out


def load_pick(cfg: Config, d: dict, k: int) -> dict:
    sc = d["sim"]
    with h5py.File(cfg.data_dir() / f"shard_{k // sc['shard_size']:05d}.h5", "r") as f:
        row = int(np.searchsorted(f["index"][:], k))
        group = {f[g].attrs["ordering"]: g for g in ("track", "track_alt") if g in f}
        t = {f"{o}_{q}": f[f"{g}/{q}"][row] for o, g in group.items() for q in ("count", "r_med", "r_hw")}
    for p in sorted(cfg.data_dir().glob("particles_*.h5")):
        with h5py.File(p, "r") as f:
            j = np.flatnonzero(f["index"][:] == k)
            if len(j):
                return t | dict(xy=f["xy"][j[0]].astype(np.float64), phi=f["phi"][j[0]].astype(np.float64))
    raise SystemExit(f"stream {k} has no stored particles")


# ------------------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------------------
def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}.png")


def fig_census(d: dict, cfg: Config, obs_span: np.ndarray, out: Path) -> None:
    L, bin_deg, top = d["L"], d["sim"]["bin_deg"], 2 * d["sim"]["phi_max_deg"]
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 4.6))
    e = np.arange(0, top + 2 * bin_deg, bin_deg) - bin_deg / 2
    ax.stairs(np.histogram(L, e)[0], e, color=P.INK2, lw=1.2, label="all streams")
    for key, lbl, col in CLASSES:
        ax.stairs(np.histogram(L[d["cls"][key]], e)[0], e, color=col, lw=1.6, label=lbl)
    ax.set_yscale("log")
    for x, t, ha in ((LONG, "one turn ", "right"), (cfg.span_max * bin_deg, f" span_max ({cfg.span_max} bins)", "left")):
        ax.axvline(x, color=P.INK, lw=0.8, ls=":")
        ax.text(x, 0.12, t, transform=ax.get_xaxis_transform(), va="center", ha=ha, fontsize=8, color=P.INK2)
    ax.plot(obs_span, np.full(len(obs_span), 1.6), "|", color=P.INK, ms=9, mew=1.0, label="the 32 observed tracks (span)")
    ax.set_xlim(0, top)
    ax.set_xlabel(f"length: unbroken run of bins with $\\geq${cfg.n_min} particles around the progenitor [deg]")
    ax.set_ylabel(f"streams per {bin_deg:.0f} deg")
    ax.set_title(f"Lengths of the {len(L):,} streams")
    ax.legend(loc="upper right", fontsize=8)

    e2 = np.arange(0, 1801, 60.0)
    n = np.histogram(L, e2)[0]
    for key, lbl, col in CLASSES:
        with np.errstate(all="ignore"):
            ax2.plot(0.5 * (e2[:-1] + e2[1:]), np.where(n >= 200, np.histogram(L[d["cls"][key]], e2)[0] / n, np.nan), color=col, label=lbl)
    ax2.axvline(LONG, color=P.INK, lw=0.8, ls=":")
    ax2.axvline(cfg.span_max * bin_deg, color=P.INK, lw=0.8, ls=":")
    ax2.set(xlim=(0, 1800), ylim=(0, 1), xlabel="length [deg]", ylabel="share of the streams of that length",
            title="The longer the stream, the less often its orbit is resolved")  # fmt: skip
    ax2.legend(loc="center right", fontsize=8)

    k = np.arange(0, 21)
    long, res = L > LONG, d["dE"] < DE_MAX
    cnt = np.array([(d["n_orb"] == i).sum() for i in k])
    with np.errstate(all="ignore"):
        ax3.plot(k, [long[d["n_orb"] == i].mean() for i in k], color=P.INK, marker="o", ms=4, label=f"longer than {LONG:.0f} deg")
        ax3.plot(k, [(long & res)[d["n_orb"] == i].mean() for i in k], color=P.C_PASS, marker="o", ms=4,
                 label=f"longer than {LONG:.0f} deg and resolved")  # fmt: skip
        ax3.plot(k, [res[d["n_orb"] == i].mean() for i in k], color=P.INK2, lw=1.2, ls="--", label="orbit resolved")
    ax3.set(xlim=(-0.5, 20.5), ylim=(0, 1), xticks=k[::2], xlabel="radial periods of the progenitor inside $t_{\\rm int}$",
            ylabel="share of the streams with that many periods",
            title=f"Long streams need several orbits; {sc_steps(d)} steps do not resolve them")  # fmt: skip
    ax3.legend(loc="center right", fontsize=8)
    ax3.text(0.99, 0.03, f"{100 * cnt[:2].sum() / len(L):.0f} % of the set has 0 or 1 period", transform=ax3.transAxes,
             ha="right", fontsize=8, color=P.INK2)  # fmt: skip
    fig.tight_layout()
    save(fig, out, "1_census")


def sc_steps(d: dict) -> str:
    return f"{d['sim']['n_steps']} leapfrog" if d["sim"]["integrator"] == "leapfrog" else d["sim"]["integrator"]


def fig_checks(d: dict, pc: dict, cfg: Config, out: Path) -> None:
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 4.8))
    sel = np.isin(d["index"], pc["index"])
    dE, floor = d["dE"][sel], 1e-5
    for key, lbl, col in CLASSES:
        m = d["cls"][key][sel]
        ax.scatter(np.maximum(dE[m], 1e-9), np.maximum(np.abs(pc["rot"][m]), floor), s=9, color=col, alpha=0.7, lw=0, label=lbl)
    ax.axvline(DE_MAX, color=P.INK, lw=0.8, ls=":")
    ax.set(xscale="log", yscale="log", ylim=(floor / 2, 360), xlabel="relative energy error of the progenitor orbit",
           ylabel="rotation of the track frame away from the progenitor on +x [deg]",
           title=f"Stored frame of the {len(pc['index']):,} streams with particles")  # fmt: skip
    nbad = int((np.abs(pc["rot"]) > 0.005).sum())
    ax.text(0.03, 0.97, f"unwrapped angle = sky angle (mod 360) to 0.005 deg\nin {len(pc['index']) - nbad} streams; "
            f"the other {nbad} are rotated as a whole\n(largest spread inside a stream {pc['rot_spread'].max():.4f} deg)\n"
            f"points at {floor:g}: that or less", transform=ax.transAxes, va="top", fontsize=8, color=P.INK2)  # fmt: skip
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.74), fontsize=8)

    if "L_g14" in d:
        top = 2 * d["sim"]["phi_max_deg"]
        e = np.arange(0, top + 41, 40.0)
        h = np.histogram2d(d["L"], d["L_g14"], bins=[e, e])[0]
        im = ax2.pcolormesh(e, e, np.where(h > 0, h, np.nan).T, cmap=P.SEQ, norm=plt.matplotlib.colors.LogNorm(1, h.max()))
        ax2.plot([0, top], [0, top], color=P.INK, lw=0.8, ls=":")
        ax2.grid(False)
        ax2.set(xlabel="length in the C26 ordering [deg]", ylabel="length in the Gibbons ordering [deg]",
                title="The two stored orderings agree for short streams only")  # fmt: skip
        fig.colorbar(im, ax=ax2, label="streams", pad=0.02)

    ex, ey = np.arange(0, 1801, 30.0), np.logspace(-2.5, 0.7, 60)
    h = np.histogram2d(d["L"], d["width"], bins=[ex, ey])[0]
    im = ax3.pcolormesh(ex, ey, np.where(h > 0, h, np.nan).T, cmap=P.SEQ, norm=plt.matplotlib.colors.LogNorm(1, h.max()))
    ax3.grid(False)
    ax3.axvline(LONG, color=P.INK, lw=0.8, ls=":")
    ax3.axhline(WIDE, color=P.INK, lw=0.8, ls=":")
    wide_long = 100 * (d["width"][d["L"] > LONG] > WIDE).mean()
    ax3.text(0.98, 0.03, f"above the line at {WIDE:g}: {wide_long:.0f} % of the\nstreams longer than {LONG:.0f} deg",
             transform=ax3.transAxes, ha="right", fontsize=8, color=P.INK)  # fmt: skip
    ax3.set(yscale="log", xlabel="length [deg]", ylabel="half-width of the radii in a bin / median radius (median over the run)",
            title="Bins of long streams are wide")  # fmt: skip
    fig.colorbar(im, ax=ax3, label="streams", pad=0.02)
    fig.tight_layout()
    save(fig, out, "2_checks")


def fig_sky(d: dict, cfg: Config, picks, mocks: dict, out: Path) -> dict:
    sc, grid = d["sim"], Grid.from_sims(d["sim"])
    bw, ctr = grid.bin_deg, (np.arange(grid.n_bins) - grid.i0 + 0.5) * grid.bin_deg
    fig, axes = plt.subplots(len(picks), 3, figsize=(18, 4.6 * len(picks)), gridspec_kw=dict(width_ratios=[1, 2, 1]), squeeze=False)
    info = {}
    for (ax, ax2, ax3), k in zip(axes, picks):
        i = int(np.flatnonzero(d["index"] == k)[0])
        t, th = load_pick(cfg, d, k), d["theta"][i]
        a, b, phi, xy = int(d["a"][i]), int(d["b"][i]), t["phi"], t["xy"]
        r = np.hypot(xy[:, 0], xy[:, 1])
        inrun = (phi >= -a * bw) & (phi < b * bw)
        vmax = max(a, b, 1) * bw
        kw = dict(c=phi[inrun], cmap=P.DIV, vmin=-vmax, vmax=vmax, s=1.2, lw=0, alpha=0.6, rasterized=True)
        run = slice(grid.i0 - a, grid.i0 + b)
        rm, hw, pc = t["c26_r_med"][run], t["c26_r_hw"][run], np.deg2rad(ctr[run])
        lim = 1.08 * np.percentile(r[inrun], 99.5)
        # sky, simulation frame
        ax.scatter(xy[~inrun, 0], xy[~inrun, 1], s=1.0, color=P.AXIS, lw=0, alpha=0.5, rasterized=True)
        sca = ax.scatter(xy[inrun, 0], xy[inrun, 1], **kw)
        ax.plot(rm * np.cos(pc), rm * np.sin(pc), color=P.INK, lw=0.9)
        ax.plot(0, 0, "+", color=P.INK, ms=9, mew=1.4)
        ax.plot(th[7], 0, "o", mfc="none", mec=P.INK, ms=9, mew=1.4)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xlabel="x [kpc]", ylabel="y [kpc]")
        fig.colorbar(sca, ax=ax, pad=0.02, fraction=0.046, label="unwrapped angle from the progenitor [deg]").solids.set_alpha(1)
        # unwrapped
        ax2.scatter(phi[inrun], r[inrun], **kw)
        ax2.scatter(phi[~inrun], r[~inrun], s=1.0, color=P.AXIS, lw=0, alpha=0.5, rasterized=True)
        ax2.fill_between(ctr[run], rm - hw, rm + hw, color=P.INK, alpha=0.13, lw=0)
        ax2.plot(ctr[run], rm, color=P.INK, lw=1.3, label="C26 track: median, 16-84 half-width")
        if "gibbons14_count" in t:
            g = np.where(t["gibbons14_count"] >= cfg.n_min, t["gibbons14_r_med"], np.nan)
            ax2.plot(ctr, g, color=P.C_BOUND, lw=1.1, ls="--", label="Gibbons track (same particles)")
        for x in np.arange(-8, 9) * 360.0:
            ax2.axvline(x, color=P.INK2, lw=0.6, ls=":")
        pad = 0.04 * (a + b) * bw
        ax2.set(xlim=(-a * bw - pad, b * bw + pad), ylim=(0, 1.15 * np.percentile(r[inrun], 99.5)),
                xlabel="unwrapped angle from the progenitor [deg]   (dotted: full turns)", ylabel="projected radius [kpc]")  # fmt: skip
        cls = next(lbl for key, lbl, _ in CLASSES if d["cls"][key][i])
        ax2.set_title(f"stream {k}: {cls}; progenitor {'dissolved' if d['dissolved'][i] else 'surviving'}\n"
                      f"{d['L'][i]:.0f} deg = {d['L'][i] / 360:.1f} turns ({a} bins trailing, {b} leading);  orbit "
                      f"{d['r_min'][i]:.0f}-{d['r_max'][i]:.0f} kpc, {d['n_orb'][i]} radial periods, energy error {d['dE'][i]:.0e}\n"
                      f"$\\log M$ {th[0]:.2f}, $\\log m$ {th[5]:.1f}, $q$ {th[2]:.2f}, $\\cos i$ {th[3]:.2f}, "
                      f"$t_{{\\rm int}}$ {th[12]:.1f} Gyr", fontsize=9)  # fmt: skip
        ax.set_title("sky, simulation frame (o progenitor, + host)", fontsize=9)
        # the mock observation in the data frame, on the particles rotated with it
        info[k] = dict(length_deg=float(d["L"][i]), cls=cls)
        if k in mocks:
            m = mocks[k]
            S, l, rot = m["S"], m["l"], m["rot"]
            ax2.axvspan(-l * bw, (S - l) * bw, color=P.C_ALL, alpha=0.10, lw=0, label=f"window of the mock: {S} bins")
            win = (phi >= -l * bw) & (phi < (S - l) * bw)
            ang = np.deg2rad(np.rad2deg(np.arctan2(xy[win, 1], xy[win, 0])) - rot)
            ax3.scatter(r[win] * np.cos(ang), r[win] * np.sin(ang), s=1.2, color=P.AXIS, lw=0, alpha=0.7, rasterized=True)
            o, pm = m["obs"], np.deg2rad(m["phi"])
            for sgn in (-1, 1):
                rr = np.stack([m["r"], m["r"] + sgn * m["err"]])[:, o]
                ax3.plot(rr * np.cos(pm[o]), rr * np.sin(pm[o]), color=P.INK, lw=0.8)
            ax3.scatter((m["r"] * np.cos(pm))[o], (m["r"] * np.sin(pm))[o], c=(m["phi"] + rot)[o], cmap=P.DIV, vmin=-vmax, vmax=vmax,
                        s=26, edgecolors=P.INK, linewidths=0.6, zorder=3)  # fmt: skip
            ax3.plot(0, 0, "+", color=P.INK, ms=9, mew=1.4)
            pr = th[7] * np.array([np.cos(np.deg2rad(-rot)), np.sin(np.deg2rad(-rot))])
            ax3.plot(*pr, "o", mfc="none", mec=P.INK, ms=9, mew=1.4)
            lim3 = 1.08 * max(np.percentile(r[win], 99.5), (m["r"] + m["err"])[o].max())
            ax3.set(xlim=(-lim3, lim3), ylim=(-lim3, lim3), aspect="equal", xlabel="x [kpc]", ylabel="y [kpc]")
            # the mock's model radii against the particles of the same bins, binned in the data frame
            jb = np.floor((phi[win] - rot - (m["phi"][0] - bw / 2)) / bw).astype(int)
            dev = max(abs(np.median(r[win][jb == j]) / m["m"][j] - 1) for j in range(S) if (jb == j).sum() >= cfg.n_min)
            over = max(0, S - int(round(360 / bw)))
            ax3.set_title(f"mock of {S} bins in the data frame (sky turned by {-rot:+.0f} deg)\n"
                          + (f"{over} bins lie over earlier ones" if over else "no bin over another")
                          + f";  $\\sigma_{{\\rm sys}}$ {m['sig_sys']:.1f} kpc;  colours as on the left", fontsize=9)  # fmt: skip
            info[k] |= dict(mock_bins=S, mock_left=l, rotation_deg=float(-rot), model_vs_particles_max_rel=float(dev))
        else:
            ax3.set_axis_off()
        ax2.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    save(fig, out, "3_sky")
    return info


def window_shares(d: dict, cfg: Config) -> dict:
    """Per track length S (bins): streams that hold at least one window, and the share of each class among the
    examples of that length (a stream is drawn in proportion to the windows it holds, as `window_design` does),
    with guard 1 alone and with guard 2 at a flat error."""
    S = np.arange(cfg.span_min, cfg.span_max + 1)
    out = dict(S=S)
    for tag, a, b in (("g1", d["a"], d["b"]), ("g2", d["a_q"], d["b_q"])):
        a, b = a.astype(np.int32), b.astype(np.int32)
        host, share = {k: [] for k, *_ in CLASSES}, {k: [] for k, *_ in CLASSES}
        for s in S:
            npos = np.clip(np.minimum(s, a) - np.maximum(0, s - b) + 1, 0, None)
            for k, *_ in CLASSES:
                host[k].append(int((npos[d["cls"][k]] > 0).sum()))
                share[k].append(float(npos[d["cls"][k]].sum() / max(npos.sum(), 1)))
        out[tag] = dict(host={k: np.array(v) for k, v in host.items()}, share={k: np.array(v) for k, v in share.items()})
    return out


def fig_training(d: dict, cfg: Config, ws: dict, obs_bins: np.ndarray, out: Path) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    S = ws["S"]
    for key, lbl, col in CLASSES:
        ax.plot(S, ws["g2"]["host"][key], color=col, label=lbl)
        ax.plot(S, ws["g1"]["host"][key], color=col, lw=1.1, ls="--")
        ax2.plot(S, ws["g2"]["share"][key], color=col, label=lbl)
        ax2.plot(S, ws["g1"]["share"][key], color=col, lw=1.1, ls="--")
    for a in (ax, ax2):
        a.plot([], [], color=P.INK2, label=f"both guards, flat data error {SIG_GUARD:g} kpc, var_ratio {cfg.var_ratio:g}")
        a.plot([], [], color=P.INK2, lw=1.1, ls="--", label=f"guard 1 alone ($\\geq${cfg.n_min} particles per bin)")
        a.plot(obs_bins, np.full(len(obs_bins), 0.02), "|", color=P.INK, ms=9, mew=1.0, transform=a.get_xaxis_transform(),
               label="the 32 observed tracks")  # fmt: skip
        a.set_xlabel("track length S [bins of 10 deg]")
    ax.set(yscale="log", ylabel="streams that hold a track of S bins", title=f"Simulations behind a track of S bins ({len(d['L']):,} in the set)")
    ax2.set(ylim=(0, 1), ylabel="share of the mock tracks of S bins", title="Where the examples of each length come from")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8)
    save(fig, out, "4_training")


# ------------------------------------------------------------------------------------------------------
def numbers(d: dict, pc: dict, ws: dict, cfg: Config) -> dict:
    L, n, bw = d["L"], len(d["L"]), d["sim"]["bin_deg"]
    res, mono, cls = d["dE"] < DE_MAX, (d["status"] & 32) == 0, d["cls"]
    edges = (0, 90, 180, 360, cfg.span_max * bw, 720, 1440, 2 * d["sim"]["phi_max_deg"] + bw)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (L > lo) & (L <= hi)
        rows.append(dict(length_deg=[lo, hi], n=int(m.sum()), share=float(m.mean()), resolved=float(res[m].mean()),
                         folds_back=float((~mono)[m].mean()), classes={c: float(cls[c][m].mean()) for c in cls}, unbound=float(d["unbound"][m].mean()),
                         dissolved=float(d["dissolved"][m].mean()), radial_periods_median=float(np.median(d["n_orb"][m])),
                         steps_per_radial_period_median=float(np.median(d["sim"]["n_steps"] / np.maximum(d["n_orb"][m], 1))),
                         width_median=float(np.nanmedian(d["width"][m])), overflow_1pct=float((d["n_overflow"][m] > 0.01 * d["sim"]["n_particles"]).mean()),
                         same_length_gibbons=float((np.abs(L - d["L_g14"])[m] <= 3 * bw).mean()) if "L_g14" in d else None,
                         log10_M_halo_median=float(np.median(d["theta"][m, 0])), r_peri_median=float(np.median(d["r_min"][m]))))  # fmt: skip
    long = L > LONG
    sel = np.isin(d["index"], pc["index"])
    rot = np.abs(pc["rot"])
    lc = long[sel] & cls["thin"][sel]
    ok, lw = rot <= 0.005, long & cls["wide"]
    circle = (L > LONG) & (L <= LONG + bw)  # one bin more than a full circle: particles all around the host, no turn
    k = lambda s: int(np.flatnonzero(ws["S"] == s)[0])
    out = dict(
        n_streams=n, long_deg=LONG, n_min=cfg.n_min, by_length=rows,
        longer_than=dict({f"{x:.0f}": float((L > x).mean()) for x in (LONG, cfg.span_max * bw, 720, 1440)}),
        long=dict(n=int(long.sum()), resolved=float(res[long].mean()), classes={c: float(cls[c][long].mean()) for c in cls},
                  n_thin_two_periods_or_more=int((long & cls["thin"] & (d["n_orb"] >= 2)).sum()),
                  n_thin_longer_than_span_max=int((cls["thin"] & (L > cfg.span_max * bw)).sum()),
                  wide=float((d["width"][long] > WIDE).mean()), under_two_radial_periods=float((d["n_orb"][long] < 2).mean()),
                  wide_class=dict(under_two_radial_periods=float((d["n_orb"] < 2)[lw].mean()), unbound=float(d["unbound"][lw].mean()),
                                  log10_M_halo_median=float(np.median(d["theta"][lw, 0])), x0_median=float(np.median(d["theta"][lw, 7]))),
                  one_circle=dict(n=int(circle.sum()), classes={c: float(cls[c][circle].mean()) for c in cls}),
                  folds_back=float((~mono)[long].mean()), unbound=float(d["unbound"][long].mean()),
                  trailing_bins_mean=float(d["a"][long].mean()), leading_bins_mean=float(d["b"][long].mean())),
        fills_the_grid=dict(trailing=int((d["a"] == d["a"].max()).sum()), leading=int((d["b"] == d["b"].max()).sum())),
        particles=dict(
            n=len(pc["index"]), rotation_above_0p005_deg=int((rot > 0.005).sum()), rotation_above_1_deg=int((rot > 1).sum()),
            rotated=[dict(index=int(i), rotation_deg=float(r), energy_error=float(e), progenitor_rotation_deg=float(p), progenitor_dr_rel=float(q))
                     for i, r, e, p, q in zip(pc["index"][rot > 0.005], pc["rot"][rot > 0.005], d["dE"][sel][rot > 0.005],
                                              pc["prog_rot"][rot > 0.005], pc["prog_dr"][rot > 0.005])],
            rotation_spread_max_deg=float(pc["rot_spread"].max()),
            # mean of the particles released today against (x0, 0): median and 99th percentile over the other streams
            progenitor_rotation_deg_others=np.percentile(np.abs(pc["prog_rot"][ok]), [50, 99]).tolist(),
            progenitor_dr_rel_others=np.percentile(np.abs(pc["prog_dr"][ok]), [50, 99]).tolist(),
            track_rebuilt=dict(streams_with_other_counts=int((pc["count_diff"] > 0).sum()), particles_in_another_bin=int(pc["count_diff"].sum() // 2),
                               r_med_max_rel_diff=float(pc["rmed_diff"].max())),
            leading_arm_ahead=dict(all=float(np.median(pc["lead"])), long_thin=float(np.median(pc["lead"][lc])), n_long_thin=int(lc.sum())),
            trailing_arm_behind=dict(all=float(np.median(pc["trail"])), long_thin=float(np.median(pc["trail"][lc]))),
        ),
        windows={f"{s}_bins": {tag: dict(streams={c: int(ws[tag]["host"][c][k(s)]) for c in cls},
                                         share={c: float(ws[tag]["share"][c][k(s)]) for c in cls}) for tag in ("g1", "g2")}
                 for s in (12, 24, 36, cfg.span_max) if s in ws["S"]},
        guard2=dict(sigma_kpc=SIG_GUARD, var_ratio=cfg.var_ratio),
    )  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="outputs/extragalactic/C26_test/npe_v2.yaml")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--picks", default="", help="stream indices for 3_sky, comma separated (they need stored particles)")
    a = ap.parse_args()
    cfg = Config.load(REPO / a.config)[0]
    pick_device(cpu=True)  # figures and one jitted mock on a handful of streams: stay off the GPUs
    out = REPO / cfg.out / "plots" / cfg.subdir / "long_streams"
    out.mkdir(parents=True, exist_ok=True)
    d = scan(cfg, a.workers)  # before jax is imported: the pool forks
    pc = particle_checks(cfg, d)
    picks = [int(x) for x in a.picks.split(",")] if a.picks else [k for k in PICKS if k in pc["index"]]
    obs = load_observed(cfg)
    obs_span = np.array([np.ptp(o["theta_deg"]) + d["sim"]["bin_deg"] for o in obs])
    ws = window_shares(d, cfg)
    fig_census(d, cfg, obs_span, out)
    fig_checks(d, pc, cfg, out)
    fig_training(d, cfg, ws, obs_span / d["sim"]["bin_deg"], out)
    num = numbers(d, pc, ws, cfg)
    num["picks"] = fig_sky(d, cfg, picks, mock_windows(cfg, d, picks), out)
    (out / "long_streams.json").write_text(json.dumps(num, indent=1))
    print(f"wrote {out / 'long_streams.json'}")
    print(json.dumps({k: num[k] for k in ("longer_than", "long", "particles", "windows", "picks")}, indent=1))


if __name__ == "__main__":
    main()
