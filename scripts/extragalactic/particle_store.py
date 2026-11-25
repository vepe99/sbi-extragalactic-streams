"""A simulation set that keeps its particles in compressed form, and the track made from them at training time.

    uv run python scripts/extragalactic/particle_store.py build --subdir particles_pilot
    uv run --extra gpu python scripts/extragalactic/particle_store.py bench --subdir particles_pilot [--gpu 3]

build   reads the particle files of <out>/data/<subdir>/ (generate.py with --n-save-particles) and writes
        svd_particles.h5 there: per stream the "ordered SVD" of particle_svd.py (particles sorted by angle,
        chunks of `chunk` neighbours; the angle of every `knot`-th particle, the mean radius of every chunk,
        and the rank-`rank` SVD of the sorted residuals of the chunks), as float16 or float32.
bench   makes tracks from the store in JAX (`track_from_store`): the particles are rebuilt, binned with the
        bin edges shifted by a random part of a bin, and the count, median and 16-84 half-width of every bin
        come back, as generate.py computes them. Reports how many streams per second that is, and how the
        track at shift 0 compares with the track generate.py stored for the same stream.

Written to <out>/plots/<subdir>/particle_store/particle_store.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts" / "extragalactic"), str(REPO / "scripts" / "extragalactic" / "sbi")]
from config import pick_device  # noqa: E402


def compress(phi, r, rank: int, chunk: int, knot: int) -> dict:
    """One stream (all particles finite) -> the kept arrays."""
    order = np.argsort(phi, kind="stable")
    ps, rs = phi[order], r[order]
    n = len(ps)
    idx = np.unique(np.r_[np.arange(0, n, knot), n - 1])
    R = rs.reshape(-1, chunk)
    centre = (np.arange(R.shape[0]) + 0.5) * chunk - 0.5
    mean = R.mean(1)
    E = np.sort(R - np.interp(np.arange(n), centre, mean).reshape(R.shape), axis=1)
    U, s, Vt = np.linalg.svd(E, full_matrices=False)
    return dict(knots=ps[idx], mean=mean, us=U[:, :rank] * s[:rank], vt=Vt[:rank])


def build(a) -> None:
    ddir = REPO / a.out / "data" / a.subdir
    files = sorted(ddir.glob("particles_*.h5"))
    if not files:
        raise SystemExit(f"no particle files in {ddir}")
    parts, index, skipped, t0, raw_bytes = [], [], 0, time.time(), 0
    for p in files:
        with h5py.File(p, "r") as f:
            idx, xy, phi = f["index"][:], f["xy"][:].astype(np.float64), f["phi"][:].astype(np.float64)
        raw_bytes += p.stat().st_size
        r = np.hypot(xy[..., 0], xy[..., 1])
        good = np.isfinite(phi).all(1) & np.isfinite(r).all(1)
        skipped += int((~good).sum())
        for i in np.flatnonzero(good):
            parts.append(compress(phi[i], r[i], a.rank, a.chunk, a.knot))
            index.append(idx[i])
        print(f"{p.name}: {int(good.sum())} of {len(good)} streams", flush=True)
    out = ddir / "svd_particles.h5"
    with h5py.File(out, "w") as f:
        f.create_dataset("index", data=np.array(index, np.int64))
        for k in ("knots", "mean", "us", "vt"):
            f.create_dataset(k, data=np.stack([q[k] for q in parts]).astype(a.dtype))
        f.attrs.update(rank=a.rank, chunk=a.chunk, knot=a.knot, n_particles=phi.shape[1])
    per = out.stat().st_size / len(index)
    print(f"wrote {out}: {len(index)} streams ({skipped} with a non-finite particle left out), {per / 1000:.1f} kB each "
          f"against {raw_bytes / (len(index) + skipped) / 1000:.0f} kB of particles, in {time.time() - t0:.0f} s")  # fmt: skip


def make_track_fn(n: int, chunk: int, knot: int, bin_deg: float, phi_max: float):
    """-> f(key, knots, mean, us, vt, shift) for ONE stream: count, r_med, r_hw on the grid of generate.py with
    its bin edges moved by `shift` degrees. vmap it over a batch."""
    import jax.numpy as jnp
    import jax.random as jr

    idx = jnp.asarray(np.unique(np.r_[np.arange(0, n, knot), n - 1]), jnp.float32)
    centre = (jnp.arange(n // chunk) + 0.5) * chunk - 0.5
    pos, nb = jnp.arange(n, dtype=jnp.float32), int(round(2 * phi_max / bin_deg))

    def f(key, knots, mean, us, vt, shift):
        phi = jnp.interp(pos, idx, knots.astype(jnp.float32))
        E = us.astype(jnp.float32) @ vt.astype(jnp.float32)  # (chunks, chunk): sorted residuals
        E = jnp.take_along_axis(E, jnp.argsort(jr.uniform(key, E.shape), axis=1), axis=1)  # dealt out at random in a chunk
        r = jnp.interp(pos, centre, mean.astype(jnp.float32)) + E.ravel()
        b = jnp.floor((phi - shift + phi_max) / bin_deg).astype(jnp.int32)
        seg = jnp.where((b >= 0) & (b < nb), b, nb)
        count = jnp.zeros(nb + 1, jnp.int32).at[seg].add(1)
        r_sorted = r[jnp.lexsort((r, seg))]
        start = jnp.cumsum(count) - count

        def quantile(q):  # as bin_track of generate.py
            p = start + q * (count - 1)
            lo = jnp.floor(p)
            w = p - lo
            lo = jnp.clip(lo.astype(jnp.int32), 0, n - 1)
            hi = jnp.clip(jnp.minimum(lo + 1, start + count - 1), 0, n - 1)
            return r_sorted[lo] * (1.0 - w) + r_sorted[hi] * w

        return count[:nb], quantile(0.5)[:nb], 0.5 * (quantile(0.84) - quantile(0.16))[:nb]

    return f


def bench(a) -> None:
    print(f"device: {pick_device(a.cpu, a.gpu)}", flush=True)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    ddir = REPO / a.out / "data" / a.subdir
    with h5py.File(ddir / "svd_particles.h5", "r") as f:
        store = {k: f[k][:] for k in ("index", "knots", "mean", "us", "vt")}
        at = dict(f.attrs)
    shards = sorted(ddir.glob("shard_*.h5"))
    with h5py.File(shards[0], "r") as f:
        sc = json.loads(f.attrs["config"])
    cols = {k: [] for k in ("index", "track/count", "track/r_med", "track/r_hw")}
    for p in shards:
        with h5py.File(p, "r") as f:
            for k in cols:
                cols[k].append(f[k][:])
    cols = {k: np.concatenate(v) for k, v in cols.items()}
    row = np.searchsorted(cols["index"], store["index"])
    fn = jax.jit(jax.vmap(make_track_fn(int(at["n_particles"]), int(at["chunk"]), int(at["knot"]), sc["bin_deg"], sc["phi_max_deg"])))
    B, N = a.batch, len(store["index"])
    dev = {k: jnp.asarray(store[k]) for k in ("knots", "mean", "us", "vt")}
    call = lambda s, shift: fn(jr.split(jr.key(s), B), *(dev[k][s : s + B] for k in ("knots", "mean", "us", "vt")), shift)
    jax.block_until_ready(call(0, jnp.zeros(B)))  # compile
    t0, n_calls = time.time(), 0
    for s in range(0, N - B + 1, B):
        jax.block_until_ready(call(s, jr.uniform(jr.key(s + 1), (B,)) * sc["bin_deg"]))
        n_calls += 1
    rate = n_calls * B / (time.time() - t0)
    # shift 0 against the track generate.py stored
    z, dc, lost, n_bins = [], [], 0, 0
    for s in range(0, N - B + 1, B):
        c1, m1, h1 = (np.asarray(v) for v in call(s, jnp.zeros(B)))
        r_ = row[s : s + B]
        c0, m0, h0 = cols["track/count"][r_].astype(int), cols["track/r_med"][r_], cols["track/r_hw"][r_]
        ok = (c0 >= 10) & (c1 >= 10)
        z.append(np.abs(m1 - m0)[ok] / np.maximum(1.2533 * h0[ok] / np.sqrt(c0[ok]), 1e-9))
        dc.append(np.abs(c1 - c0)[ok])
        lost += int(((c0 >= 10) & ~(c1 >= 10)).sum())
        n_bins += int((c0 >= 10).sum())
    z, dc = np.concatenate(z), np.concatenate(dc)
    size = (ddir / "svd_particles.h5").stat().st_size
    num = dict(subdir=a.subdir, n_streams=int(N), store=dict(rank=int(at["rank"]), chunk=int(at["chunk"]), knot=int(at["knot"]), dtype=str(store["us"].dtype),
                                                             kB_per_stream=size / N / 1000, GB_for_10_million=size / N * 1e7 / 1e9),
               bench=dict(device=jax.devices()[0].platform, batch=B, streams_per_second=rate, ms_per_batch=1000 * B / rate),
               against_stored_track=dict(bins=int(n_bins), median_error_in_particle_noise=dict(median=float(np.median(z)), p90=float(np.percentile(z, 90)),
                                                                                             p99=float(np.percentile(z, 99))),
                                         count_differs=float((dc > 0).mean()), bins_lost=lost / max(n_bins, 1)))  # fmt: skip
    out = REPO / a.out / "plots" / a.subdir / "particle_store"
    out.mkdir(parents=True, exist_ok=True)
    (out / "particle_store.json").write_text(json.dumps(num, indent=1))
    print(json.dumps(num, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "bench"])
    ap.add_argument("--out", default="outputs/extragalactic/C26_test")
    ap.add_argument("--subdir", required=True)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--knot", type=int, default=20)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--gpu", type=int, default=-1)
    a = ap.parse_args()
    build(a) if a.command == "build" else bench(a)


if __name__ == "__main__":
    main()
