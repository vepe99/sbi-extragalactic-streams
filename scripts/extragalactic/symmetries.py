"""Scaling symmetries of the stream model that the training set does not use, tested on the simulator.

    uv run --extra gpu python scripts/extragalactic/symmetries.py --config <run>/config.yaml [--gpu 3]

Newtonian gravity has no scale of its own: with lengths times L and masses times U, times go as sqrt(L^3 / U)
and velocities as sqrt(U / L). The model has only lengths (R_s, r_s, x0, z0), masses (M_halo, m_prog),
velocities and t_int, and the leapfrog takes a fixed NUMBER of steps, so a simulation with scaled parameters
should be the same stream, its track multiplied by L:

    mass     L = 1, U = 2        masses x 2, velocities x sqrt 2, t_int / sqrt 2      track unchanged
    length   L = 1.3, U = 1      lengths x 1.3, velocities / sqrt 1.3, t_int x 1.3^1.5  track x 1.3
    both     L = 1.3, U = 1.3^3  lengths and velocities x 1.3, masses x 2.197        track x 1.3, t_int unchanged

The test simulates the streams of track_noise.py (run A, `sims/A`) again with scaled parameters and the same
random key, and compares the tracks bin by bin. A dissolved progenitor loses its mass towards a FIXED 1 Msun
(generate.py), which is a mass scale: the scalings that change the masses are then not exact.

Written to <out>/plots/<subdir>/symmetries/: symmetries.json, symmetries.png; the simulations in sims/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts" / "extragalactic"), str(REPO / "scripts" / "extragalactic" / "sbi")]
import long_streams as LS  # noqa: E402
import track_noise as TN  # noqa: E402
from config import Config, pick_device  # noqa: E402

SCALINGS = dict(mass=(1.0, 2.0), length=(1.3, 1.0), both=(1.3, 1.3**3))  # name: (L, U)
I_LOGM, I_RS, I_LOGMP, I_RSP, I_POS, I_VEL, I_T = 0, 1, 5, 6, slice(7, 9), slice(9, 12), 12  # columns of generate.py's theta


def scaled(theta: np.ndarray, L: float, U: float) -> np.ndarray:
    t = theta.astype(np.float64).copy()
    t[:, I_LOGM] += np.log10(U)
    t[:, I_LOGMP] += np.log10(U)
    t[:, I_RS] *= L
    t[:, I_RSP] *= L
    t[:, I_POS] *= L
    t[:, I_VEL] *= np.sqrt(U / L)
    t[:, I_T] *= np.sqrt(L**3 / U)
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    cfg = Config.load(REPO / a.config)[0]
    base = REPO / cfg.out / "plots" / cfg.subdir / "track_noise" / "sims" / "A"
    out = REPO / cfg.out / "plots" / cfg.subdir / "symmetries"
    out.mkdir(parents=True, exist_ok=True)
    pick_device(False, a.gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    with h5py.File(sorted(cfg.data_dir().glob("shard_*.h5"))[0], "r") as f:
        sc = json.loads(f.attrs["config"])
    src = np.load(base / "theta.npz")
    theta, dis = src["theta"][: a.n], src["dissolved"][: a.n]
    runs = {"same": TN.simulate(theta, dis, sc, cfg, out / "sims/same", 11, batch=128)}
    for name, (L, U) in SCALINGS.items():
        runs[name] = TN.simulate(scaled(theta, L, U), dis, sc, cfg, out / f"sims/{name}", 11, batch=128)
    ref = runs["same"]
    resolved = ref["dE"] <= LS.DE_MAX
    groups = {"progenitor kept, orbit resolved": ~dis & resolved, "progenitor kept, orbit unresolved": ~dis & ~resolved,
              "progenitor dissolved, orbit resolved": dis & resolved, "progenitor dissolved, orbit unresolved": dis & ~resolved}
    num, dev = dict(n_streams=int(a.n), scalings={k: dict(length=v[0], mass=v[1]) for k, v in SCALINGS.items()}, groups={k: int(v.sum()) for k, v in groups.items()}), {}
    for name, (L, _) in SCALINGS.items():
        r = runs[name]
        ok = (ref["count"] >= cfg.n_min) & (r["count"] >= cfg.n_min)
        with np.errstate(all="ignore"):
            rel = np.where(ok, np.abs(r["r_med"] / (L * ref["r_med"]) - 1.0), np.nan)
        worst = np.nanmax(np.where(ok.any(1, keepdims=True), rel, 0.0), axis=1)  # largest relative difference of a bin, per stream
        same_counts = (ref["count"] == r["count"]).all(1)
        dev[name] = worst
        num[name] = {g: dict(identical_counts=float(same_counts[s].mean()), worst_bin_median=float(np.median(worst[s])), worst_bin_p90=float(np.percentile(worst[s], 90)),
                             within_1e3=float((worst[s] < 1e-3).mean()), dE_ratio_median=float(np.median(r["dE"][s] / np.maximum(ref["dE"][s], 1e-30))))
                     for g, s in groups.items() if s.any()}  # fmt: skip
    (out / "symmetries.json").write_text(json.dumps(num, indent=1))
    print(json.dumps(num, indent=1))

    P, plt = LS.P, LS.plt
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), sharey=True)
    cols = (P.C_PASS, P.C_BOUND, P.C_ALL, "#eda100")
    e = np.logspace(-9, 1, 61)
    titles = dict(mass="masses x 2, velocities x $\\sqrt{2}$, $t_{\\rm int}$ / $\\sqrt{2}$: track unchanged",
                  length="lengths x 1.3, velocities / $\\sqrt{1.3}$, $t_{\\rm int}$ x 1.3$^{1.5}$: track x 1.3",
                  both="lengths and velocities x 1.3, masses x 1.3$^3$: track x 1.3")
    for ax, name in zip(axes, SCALINGS):
        for (g, s), col in zip(groups.items(), cols):
            if s.any():
                ax.stairs(np.histogram(np.clip(dev[name][s], e[0], e[-1]), e)[0] / s.sum(), e, color=col, lw=1.5, label=f"{g} ({s.sum()})")
        ax.set(xscale="log", xlabel="largest relative difference of a bin:\nsimulation of the scaled parameters against the scaled track", title=titles[name])
        ax.title.set_fontsize(9.5)
    axes[0].set_ylabel("share of the streams")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    LS.save(fig, out, "symmetries")


if __name__ == "__main__":
    main()
