"""Can the particles of a stream be stored in a few kB with a truncated SVD, so that the track can be made
again at training time with other bins? A test on the streams of a set that have their particles stored.

    uv run python scripts/extragalactic/particle_svd.py --config <yaml> [--n 1000]

A stream is 30,000 particles (angle along the stream, projected radius): 240 kB as float32. Three ways to keep
less are compared at equal storage:

  ordered SVD   particles sorted by angle and cut into chunks of `chunk` neighbours. Kept: the angle of every
                `knot`-th particle (the angle distribution, a monotone curve), the mean radius of every chunk
                (the run of the radius along the stream, joined by straight lines), and the truncated SVD of
                the matrix chunk x rank of the residuals (each row is the sorted scatter of a chunk around that
                run: the local distribution of the radius, which changes slowly along the stream and is smooth
                in the rank). Which residual belongs to which particle inside a chunk is lost.
  naive SVD     the matrices release epoch x particle of angle and radius as the simulator orders them. The
                scatter around the mean of an epoch is independent noise, so no low rank is expected.
  subsample     a random subset of the particles with the same number of stored values.

Each is judged by the track it gives back: bins of `bin_deg` (the set's, with the edge on the progenitor and
shifted by half a bin, which is the augmentation the stored tracks cannot do) and of half that width. The
error of the median radius of a bin is in units of the particle noise of that bin, 1.25 hw / sqrt(n): 1 means
the compression costs as much as the 30,000 particles themselves do.

Written to <out>/plots/<subdir>/particle_svd/: particle_svd.json, particle_svd.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts" / "extragalactic"), str(REPO / "scripts" / "extragalactic" / "sbi")]
import long_streams as LS  # noqa: E402
from config import Config, pick_device  # noqa: E402

N_MIN = 10
INTERP = "stored track, neighbouring bins averaged"


def track(phi, r, bin_deg: float, shift: float, phi_max: float):
    """count, median radius, 16-84 half-width per bin of one stream; bin edges at shift + k bin_deg."""
    b = np.floor((phi - shift + phi_max) / bin_deg).astype(int)
    nb = int(round(2 * phi_max / bin_deg))
    ok = (b >= 0) & (b < nb)
    order = np.lexsort((r[ok], b[ok]))
    bs, rs = b[ok][order], r[ok][order]
    count = np.bincount(bs, minlength=nb)
    start = np.cumsum(count) - count
    med, hw = np.full(nb, np.nan), np.full(nb, np.nan)
    for j in np.flatnonzero(count >= N_MIN):
        q16, q50, q84 = np.quantile(rs[start[j] : start[j] + count[j]], [0.16, 0.5, 0.84])
        med[j], hw[j] = q50, 0.5 * (q84 - q16)
    return count, med, hw


def errors(ref, new) -> dict:
    """Track `new` against `ref`, over the bins both hold with >= N_MIN particles."""
    (c0, m0, h0), (c1, m1, h1) = ref, new
    ok = (c0 >= N_MIN) & (c1 >= N_MIN)
    if not ok.any():
        return dict(z=np.array([]), dhw=np.array([]), dcount=np.array([]), lost=float((c0 >= N_MIN).sum()))
    noise = 1.2533 * h0[ok] / np.sqrt(c0[ok])
    return dict(z=np.abs(m1 - m0)[ok] / np.maximum(noise, 1e-9), dhw=np.abs(h1 - h0)[ok] / np.maximum(h0[ok], 1e-9),
                dcount=np.abs(c1 - c0)[ok] / np.sqrt(c0[ok]), lost=float(((c0 >= N_MIN) & ~(c1 >= N_MIN)).sum()))  # fmt: skip


def ordered_svd(phi, r, k: int, chunk: int, knot: int, rng) -> tuple:
    """-> (angles, radii) rebuilt from the kept values, and how many values were kept."""
    order = np.argsort(phi, kind="stable")
    ps, rs = phi[order], r[order]
    n = len(ps) // chunk * chunk
    idx = np.unique(np.r_[np.arange(0, n, knot), n - 1])
    p_hat = np.interp(np.arange(n), idx, ps[idx])
    R = rs[:n].reshape(-1, chunk)
    # the run of the radius along the stream: the mean of every chunk, joined by straight lines. In the thin
    # outskirts a chunk spans several bins, and without this the radii of a chunk would be dealt out at random
    centre = (np.arange(R.shape[0]) + 0.5) * chunk - 0.5
    trend = np.interp(np.arange(n), centre, R.mean(1)).reshape(R.shape)
    E = np.sort(R - trend, axis=1)  # scatter around it, sorted: the local distribution, smooth in the rank
    U, s, Vt = np.linalg.svd(E, full_matrices=False)
    E_hat = (U[:, :k] * s[:k]) @ Vt[:k]
    r_hat = (trend + rng.permuted(E_hat, axis=1)).ravel()  # which residual goes with which particle of a chunk is not kept
    return p_hat, r_hat, len(idx) + R.shape[0] + k * (R.shape[0] + R.shape[1])


def naive_svd(phi, r, k: int, n_epochs: int) -> tuple:
    out = []
    for v in (phi, r):
        A = v.reshape(n_epochs, -1)
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        out.append(((U[:, :k] * s[:k]) @ Vt[:k]).ravel())
    return out[0], out[1], 2 * k * (n_epochs + len(phi) // n_epochs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="outputs/extragalactic/C26_test/npe_v2.yaml")
    ap.add_argument("--subdir", default="", help="simulation set, if not that of the file")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--knot", type=int, default=20)
    a = ap.parse_args()
    cfg = Config.load(REPO / a.config)[0]
    cfg.subdir = a.subdir or cfg.subdir
    pick_device(cpu=True)
    out = REPO / cfg.out / "plots" / cfg.subdir / "particle_svd"
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(sorted(cfg.data_dir().glob("shard_*.h5"))[0], "r") as f:
        sc = json.loads(f.attrs["config"])
    xy, phi = [], []
    for p in sorted(cfg.data_dir().glob("particles_*.h5")):
        with h5py.File(p, "r") as f:
            xy.append(f["xy"][:]), phi.append(f["phi"][:])
        if sum(len(x) for x in xy) >= a.n:
            break
    xy, phi = np.concatenate(xy)[: a.n].astype(np.float64), np.concatenate(phi)[: a.n].astype(np.float64)
    r = np.hypot(xy[..., 0], xy[..., 1])
    good = np.isfinite(phi).all(1) & np.isfinite(r).all(1)  # streams with every particle finite
    bw, pm, ne = sc["bin_deg"], sc["phi_max_deg"], sc["n_steps"] + 1
    rng = np.random.default_rng(0)
    views = {"bins of the set": (bw, 0.0), "shifted by half a bin": (bw, bw / 2), "bins of half the width": (bw / 2, 0.0)}
    methods = {f"ordered SVD, rank {k}": ("ordered", k) for k in (1, 2, 4, 8, 16)} | {f"naive SVD, rank {k}": ("naive", k) for k in (4, 16, 64)}
    acc = {m: {v: dict(z=[], dhw=[], dcount=[], lost=0.0, bins=0) for v in views} for m in list(methods) + ["subsample, as rank 4", "subsample, as rank 16"]}
    acc[INTERP] = {v: dict(z=[], dhw=[], dcount=[], lost=0.0, bins=0) for v in views}
    stored, example = {INTERP: 0}, None
    streams = np.flatnonzero(good)
    for i in streams:
        ref = {v: track(phi[i], r[i], *views[v], pm) for v in views}
        rebuilt = {}
        for m, (kind, k) in methods.items():
            rebuilt[m] = ordered_svd(phi[i], r[i], k, a.chunk, a.knot, rng) if kind == "ordered" else naive_svd(phi[i], r[i], k, ne)
            stored[m] = rebuilt[m][2]
        for m, like in (("subsample, as rank 4", "ordered SVD, rank 4"), ("subsample, as rank 16", "ordered SVD, rank 16")):
            pick = rng.choice(phi.shape[1], stored[like] // 2, replace=False)  # two values per particle
            rebuilt[m], stored[m] = (phi[i][pick], r[i][pick], stored[like]), stored[like]
        # no particles at all: the stored track, two neighbouring bins averaged, as the track shifted by half a bin
        c0, m0, h0 = ref["bins of the set"]
        w0, w1 = c0[:-1].astype(float), c0[1:].astype(float)
        with np.errstate(all="ignore"):
            both = (c0[:-1] >= N_MIN) & (c0[1:] >= N_MIN)
            mi = np.where(both, (w0 * m0[:-1] + w1 * m0[1:]) / (w0 + w1), np.nan)
            hi = np.where(both, (w0 * h0[:-1] + w1 * h0[1:]) / (w0 + w1), np.nan)
        interp = (np.r_[np.where(both, (c0[:-1] + c0[1:]) // 2, 0), 0], np.r_[mi, np.nan], np.r_[hi, np.nan])
        e = errors(ref["shifted by half a bin"], interp)
        for q in ("z", "dhw", "dcount"):
            acc[INTERP]["shifted by half a bin"][q].append(e[q])
        acc[INTERP]["shifted by half a bin"]["lost"] += e["lost"]
        acc[INTERP]["shifted by half a bin"]["bins"] += int((ref["shifted by half a bin"][0] >= N_MIN).sum())
        for m, (p_hat, r_hat, _) in rebuilt.items():
            for v in views:
                e = errors(ref[v], track(p_hat, r_hat, *views[v], pm))
                for q in ("z", "dhw", "dcount"):
                    acc[m][v][q].append(e[q])
                acc[m][v]["lost"] += e["lost"]
                acc[m][v]["bins"] += int((ref[v][0] >= N_MIN).sum())
        c_, m_, h_ = ref["bins of the set"]
        if example is None and (c_ >= N_MIN).sum() > 25 and np.nanmax(m_) < 300 and np.nanmedian(h_ / m_) < 0.1:  # a long, thin stream
            example = (i, ref, {m: track(*rebuilt[m][:2], *views["shifted by half a bin"], pm) for m in ("ordered SVD, rank 4", "naive SVD, rank 16", "subsample, as rank 4")})
    raw = 2 * phi.shape[1]
    num = dict(subdir=cfg.subdir, n_streams=int(len(streams)), particles=int(phi.shape[1]), values_raw=raw, kB_raw_float32=raw * 4 / 1000, chunk=a.chunk, knot=a.knot, methods={})
    for m in acc:
        num["methods"][m] = dict(values=int(stored[m]), kB_float32=stored[m] * 4 / 1000, compression=raw / max(stored[m], 1), views={})
        for v in views:
            if not acc[m][v]["z"]:
                continue
            z, dhw, dc = (np.concatenate(acc[m][v][q]) for q in ("z", "dhw", "dcount"))
            num["methods"][m]["views"][v] = dict(median_error_in_particle_noise=dict(median=float(np.median(z)), p90=float(np.percentile(z, 90)), rms=float(np.sqrt(np.mean(np.minimum(z, 100) ** 2)))),
                                                 width_relative_error_median=float(np.median(dhw)), count_error_in_poisson_sd_median=float(np.median(dc)),
                                                 bins_lost=float(acc[m][v]["lost"] / max(acc[m][v]["bins"], 1)))  # fmt: skip
    (out / "particle_svd.json").write_text(json.dumps(num, indent=1))
    print(f"{len(streams)} streams, {phi.shape[1]} particles each ({raw * 4 / 1000:.0f} kB)")
    for m, d in num["methods"].items():
        print(f"{m:24s} {d['kB_float32']:6.1f} kB  x{d['compression']:5.0f} | " + " | ".join(
            f"{v}: med {q['median_error_in_particle_noise']['median']:5.2f} rms {q['median_error_in_particle_noise']['rms']:5.2f} p90 {q['median_error_in_particle_noise']['p90']:5.2f} width {q['width_relative_error_median']:.3f} lost {q['bins_lost']:.3f}"
            for v, q in d["views"].items()))  # fmt: skip

    P, plt = LS.P, LS.plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    v = "shifted by half a bin"  # the augmentation the stored tracks cannot do
    for kind, col, lab in (("ordered", P.C_PASS, "ordered SVD (sorted by angle, chunks of neighbours)"), ("naive", P.C_BOUND, "naive SVD (release epoch x particle)"), ("subsample", P.C_ALL, "random subset of the particles")):
        ms = sorted((m for m in num["methods"] if m.startswith(kind)), key=lambda m: num["methods"][m]["kB_float32"])
        kb = [num["methods"][m]["kB_float32"] for m in ms]
        for q, ls in (("median", "-"), ("p90", "--")):
            ax.plot(kb, [num["methods"][m]["views"][v]["median_error_in_particle_noise"][q] for m in ms], color=col, ls=ls, marker="o", ms=5, label=lab if q == "median" else None)
    e = num["methods"][INTERP]["views"][v]["median_error_in_particle_noise"]
    ax.axhline(e["median"], color=P.INK2, lw=1.2, label="no particles: stored track, neighbouring bins averaged")
    ax.axhline(e["p90"], color=P.INK2, lw=1.2, ls="--")
    ax.axhline(1, color=P.INK, lw=0.8, ls=":")
    ax.set(xscale="log", yscale="log", xlabel=f"stored per stream [kB as float32]; all particles: {raw * 4 / 1000:.0f} kB",
           ylabel="error of a bin median / particle noise of the bin", title="Track in bins shifted by half a bin, from the stored values (solid: median bin, dashed: 90th percentile)")  # fmt: skip
    ax.title.set_fontsize(9.5)
    ax.legend(fontsize=7.5, loc="upper right")
    if example is not None:
        i, ref, alt = example
        c, m, h = ref["shifted by half a bin"]
        x = (np.arange(len(c)) + 0.5) * bw - pm + bw / 2
        ok = c >= N_MIN
        ax2.fill_between(x[ok], (m - h)[ok], (m + h)[ok], color=P.INK, alpha=0.12, lw=0)
        ax2.plot(x[ok], m[ok], color=P.INK, lw=2.2, label="all 30,000 particles")
        for (name, (c1, m1, _)), col in zip(alt.items(), (P.C_PASS, P.C_BOUND, P.C_ALL)):
            o1 = c1 >= N_MIN
            ax2.plot(x[o1], m1[o1], color=col, lw=1.1, label=f"{name} ({num['methods'][name]['kB_float32']:.1f} kB)")
        ax2.set(xlim=(x[ok].min() - bw, x[ok].max() + bw), ylim=(0, 1.3 * np.nanmax((m + h)[ok])), xlabel="angle from the progenitor [deg], bins shifted by half a bin",
                ylabel="median radius of the bin [kpc]", title=f"One stream (index {i}) in bins the stored track does not have")  # fmt: skip
        ax2.legend(fontsize=8)
    fig.tight_layout()
    LS.save(fig, out, "particle_svd")


if __name__ == "__main__":
    main()
