"""How reproducible are the model tracks behind the training examples? A check of the simulation set, by kind of
stream and by distance from the progenitor.

    uv run --extra gpu python scripts/extragalactic/track_noise.py --config <run>/config.yaml

A mock observation is the stored model track of a stream plus noise. Two things can make that track a poor
"model at these parameters", and neither is in the likelihood:

  key noise     the track of the same parameters with another random key for the release of the particles;
  roughness     the track of parameters shifted by a tiny step (same key). For a smooth simulator the change
                shrinks with the step; where the orbit of the progenitor is not resolved by the leapfrog it
                need not.

For `n_sim` mock observations of held-out streams, drawn as the training draws them, the stream is simulated
again: run A and run B with two keys, and runs C with the key of A and the parameters shifted by `steps`
(standard deviations of the prior, a random direction in the 13 latent coordinates). Every pair of runs is
compared in the window of the mock, in units of its errors. With a trained run (`name` of the file) the
loss of every one of the `n` mock observations is stored too.

Classes of streams are those of long_streams.py: thin | wide | fold (angle of the progenitor not monotonic) |
unres (energy error of the progenitor orbit above 1 %).

Written to <out>/plots/<subdir>/track_noise/: track_noise.npz (one row per mock), track_noise.json (the numbers
by class, track length, energy error and distance from the progenitor) and track_noise.png. The simulations stay
in sims/ there and are reused; `--report-only` remakes the json and the figure from the npz.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts" / "extragalactic"), str(REPO / "scripts" / "extragalactic" / "sbi")]
import long_streams as LS  # noqa: E402
from config import Config, pick_device  # noqa: E402

CLASS_KEYS = [k for k, *_ in LS.CLASSES]


def n_bad_of(path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["diag/n_bad"][:]


def stream_table(cfg: Config, workers: int) -> dict:
    """One row per usable stream (the rows of `load_simulations`): class, energy error, radial periods, width."""
    d = LS.scan(cfg, workers)
    with Pool(workers) as pool:
        n_bad = np.concatenate(pool.map(n_bad_of, sorted(cfg.data_dir().glob("shard_*.h5"))[: cfg.max_shards or None], chunksize=8))
    failed, lossy, unbound = (d["status"] & 16) > 0, n_bad > 0.01 * d["sim"]["n_particles"], (d["status"] & 1) > 0
    keep = ~failed & ~lossy & (d["a"].astype(int) + d["b"] >= cfg.span_min) & (cfg.include_unbound | ~unbound)
    keep &= ~((d["dE"] > cfg.de_max) & (cfg.de_max > 0))  # as load_simulations
    cls = np.select([d["cls"][k] for k in CLASS_KEYS], np.arange(len(CLASS_KEYS)))
    return dict(theta=d["theta"][keep], cls=cls[keep], dE=d["dE"][keep], n_orb=d["n_orb"][keep], width=d["width"][keep],
                L=d["L"][keep], unbound=unbound[keep], nonmono=((d["status"] & 32) > 0)[keep], row=np.flatnonzero(keep))  # fmt: skip


def simulate(theta, dissolved, sc: dict, cfg: Config, out: Path, seed: int, batch: int = 0, cpu_workers: int = 0, cpu_cores: str = "", gpus: str = "", alt_ordering: bool = False) -> dict:
    """generate.py on given parameters with the master key `seed` (stream k uses fold_in(key(seed), k)), on the
    card this process holds. As `predictive.simulate`, plus the seed. batch: streams at once (0 = generate.py's
    memory probe, which assumes the card is its own). cpu_workers > 0: on that many CPU processes instead; give them
    `cpu_cores` (e.g. "32-63,160-191") to share out, or every process starts a thread for every core of the machine.
    gpus: cards (comma-separated UUIDs or indices, each checked to be free by the caller) for one generate.py worker
    each, instead of the single card this process holds. alt_ordering: also store the track in the other ordering
    (track_alt/), as train_v2 does."""
    from predictive import read_tracks

    out.mkdir(parents=True, exist_ok=True)
    same = (out / "theta.npz").exists() and np.array_equal(np.load(out / "theta.npz")["theta"], theta)
    if not (same and (out / "data/manifest.json").exists()):
        if not same:  # a run that was stopped is taken up again: its shards stay, and so must the file they were made from
            np.savez(out / "theta.npz", theta=theta.astype(np.float64), dissolved=dissolved.astype(bool))
        physics = [f for k in ("integrator", "solver", "rtol", "max_steps", "n_particles", "n_steps", "halo_flattening")
                   for f in (f"--{k.replace('_', '-')}", str(sc[k]))]  # fmt: skip
        where = (["--cpu", "--cpu-workers", str(cpu_workers), "--shard-size", str(max(1, -(-len(theta) // (4 * cpu_workers)))), *(["--cpu-cores", cpu_cores] if cpu_cores else [])] if cpu_workers
                 else ["--gpus", gpus or os.environ["CUDA_VISIBLE_DEVICES"], "--shard-size", "1024"])
        cmd = ["uv", "run", "--script", str(REPO / "scripts/extragalactic/generate.py"), "run", *([] if same else ["--overwrite"]), *where,
               "--theta-file", str(out / "theta.npz"), "--n-streams", str(len(theta)), "--n-save-particles", "0",
               "--out", str(out.relative_to(REPO)), "--track-ordering", cfg.ordering, "--progenitor-mass-loss", "mixed", "--seed", str(seed), *physics, *(["--alt-ordering"] if alt_ordering else []), *(["--batch-size", str(batch)] if batch else [])]  # fmt: skip
        print(f"simulating {len(theta)} streams -> {out.name}", flush=True)
        with open(out / "generate.log", "w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    return read_tracks(out)


def jumps(d, x: str, y: str, sel) -> dict:
    """Runs x and y compared on the simulated mocks `sel`: among those where x passes the guards, the share where
    y fails them, the share where the log-likelihood of the mock differs by more than 2, their sum (`jump`), and
    the shift of the track per bin in data errors (rms over the observed bins; median and 90th percentile)."""
    M = len(d[f"ll_{x}"])
    obs, sig = d["mask"][:M], d["sig"][:M]
    fx, fy = np.isfinite(d[f"ll_{x}"]) & sel, np.isfinite(d[f"ll_{y}"])
    both = fx & fy
    with np.errstate(invalid="ignore"):
        dll = np.abs(d[f"ll_{x}"] - d[f"ll_{y}"])[both]
    z = np.sqrt(np.where(obs, (d[f"m_{x}"] - d[f"m_{y}"]) ** 2 / sig**2, 0).sum(1) / obs.sum(1))[both]
    n = max(int(fx.sum()), 1)
    fails, big = float((fx & ~fy).sum() / n), float((dll > 2).sum() / n)
    return dict(n=int(fx.sum()), fails_guards=fails, dlogl_above_2=big, jump=fails + big, dlogl_p90=float(np.percentile(dll, 90)),
                shift_median=float(np.median(z)), shift_p90=float(np.percentile(z, 90)))  # fmt: skip


def report(out: Path, n_min: int) -> None:
    """track_noise.npz -> track_noise.json and track_noise.png."""
    P, plt = LS.P, LS.plt
    d = np.load(out / "track_noise.npz", allow_pickle=True)
    M, steps = len(d["ll_A"]), [float(s) for s in d["steps"]]
    cls, S, dE = d["cls"], d["S"], d["dE"]
    lengths = {"short (6-12 bins)": S <= 12, "medium (13-25)": (S > 12) & (S <= 25), "long (26-49)": S > 25, "all": S > 0}
    pairs = [("another key", "A", "B")] + [(f"parameters shifted by {s:g} prior sd", "A", f"C{s:g}") for s in steps]
    num = dict(n_mocks=len(S), n_simulated=M, classes={k: lbl for k, lbl, _ in LS.CLASSES},
               share={lab: {k: float((cls[sel] == i).mean()) for i, k in enumerate(CLASS_KEYS)} for lab, sel in lengths.items()},
               by_class={lab: {k: jumps(d, x, y, cls[:M] == i) for i, k in enumerate(CLASS_KEYS)} for lab, x, y in pairs},
               by_length={lab: {ln: jumps(d, x, y, sel[:M]) for ln, sel in lengths.items()} for lab, x, y in pairs})  # fmt: skip
    if "nll" in d.files:
        num["loss_nats"] = {lab: {k: float(d["nll"][sel & (cls == i)].mean()) for i, k in enumerate(CLASS_KEYS)} | dict(all=float(d["nll"][sel].mean()))
                            for lab, sel in lengths.items()}  # fmt: skip
    e_dE = [0, 1e-4, 1e-3, 1e-2, 1e-1, np.inf]
    num["by_energy_error"] = [dict(dE=[a, b if np.isfinite(b) else None], share=float(((dE >= a) & (dE < b)).mean()),
                                   **{f"shift_{s:g}": jumps(d, "A", f"C{s:g}", (dE[:M] >= a) & (dE[:M] < b)) for s in steps})
                              for a, b in zip(e_dE[:-1], e_dE[1:])]  # fmt: skip
    # key noise per bin against the distance from the progenitor
    obs, sig, l = d["mask"][:M], d["sig"][:M], d["l"][:M]
    j = np.arange(obs.shape[1])[None]
    dist = np.where(j >= l[:, None], j - l[:, None], l[:, None] - 1 - j)
    ok = obs & (d["n_A"] >= n_min) & (d["n_B"] >= n_min)
    dz = np.abs(d["m_A"] - d["m_B"]) / sig
    expected = np.sqrt(d["hw_A"] ** 2 / np.maximum(d["n_A"], 1) + d["hw_B"] ** 2 / np.maximum(d["n_B"], 1))  # what guard 2 takes as noise
    e_d = [0, 3, 6, 10, 15, 20, 30, obs.shape[1]]
    num["key_noise_by_distance"] = []
    for a, b in zip(e_d[:-1], e_d[1:]):
        s = ok & (dist >= a) & (dist < b)
        num["key_noise_by_distance"].append(dict(bins=[a, b - 1], n=int(s.sum()), median=float(np.median(dz[s])), p90=float(np.percentile(dz[s], 90)),
                                                 p99=float(np.percentile(dz[s], 99)), particles_median=float(np.median(d["n_A"][s])),
                                                 realised_over_expected_rms=float(np.sqrt(np.mean(np.minimum((np.abs(d["m_A"] - d["m_B"]) / np.maximum(expected, 1e-9))[s], 50) ** 2)))))  # fmt: skip
    (out / "track_noise.json").write_text(json.dumps(num, indent=1))

    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 4.7))
    for i, (k, lbl, col) in enumerate(LS.CLASSES):
        y = [num["by_class"][f"parameters shifted by {s:g} prior sd"][k]["jump"] for s in steps]
        ax.plot(steps, y, color=col, marker="o", ms=5, label=f"{lbl} ({num['share']['all'][k]:.0%} of the examples)")
        ax.plot([steps[-1] * 30], [num["by_class"]["another key"][k]["jump"]], color=col, marker="s", ms=6, ls="none")
    ax.set_xscale("log")
    ax.set_xticks(steps + [steps[-1] * 30], [f"{s:g}" for s in steps] + ["another key,\nsame parameters"])
    ax.set(ylim=(0, None), xlabel="shift of the parameters [standard deviations of the prior], same key",
           ylabel="share of the mock tracks whose likelihood jumps", title="A tiny shift of the parameters moves the track of an unresolved orbit")  # fmt: skip
    ax.text(0.02, 0.97, "jump: the stream simulated again fails a guard (likelihood 0)\nor log L of the mock changes by more than 2",
            transform=ax.transAxes, va="top", fontsize=8, color=P.INK2)  # fmt: skip
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.84), fontsize=7.5)

    x = np.arange(len(e_dE) - 1)
    for s, col in zip(steps[1:], (P.C_ALL, P.C_BOUND)):
        ax2.plot(x, [r[f"shift_{s:g}"]["jump"] for r in num["by_energy_error"]], color=col, marker="o", ms=5, label=f"parameters shifted by {s:g} prior sd")
    ax2.axvline(2.5, color=P.INK, lw=0.8, ls=":")
    ax2.text(2.55, 0.97, " 1 %: unresolved", transform=ax2.get_xaxis_transform(), va="top", fontsize=8, color=P.INK2)
    ax2.set_xticks(x, [f"{t}\n{r['share']:.0%}" for t, r in zip(["< 1e-4", "1e-4 - 1e-3", "1e-3 - 1e-2", "1e-2 - 0.1", "> 0.1"], num["by_energy_error"])])
    ax2.set(ylim=(0, None), xlabel="relative energy error of the progenitor orbit (99 leapfrog steps); second line: share of the examples",
            ylabel="share of the mock tracks whose likelihood jumps", title="The jumps start at an energy error of 1 %")  # fmt: skip
    ax2.legend(loc="upper left", fontsize=8)

    xd = np.arange(len(e_d) - 1)
    ax3.plot(xd, [r["median"] for r in num["key_noise_by_distance"]], color=P.C_ALL, marker="o", ms=5, label="median")
    ax3.plot(xd, [r["p90"] for r in num["key_noise_by_distance"]], color=P.C_ALL, marker="o", ms=5, ls="--", label="90th percentile")
    ax3.plot(xd, [r["p99"] for r in num["key_noise_by_distance"]], color=P.C_ALL, marker="o", ms=5, ls=":", label="99th percentile")
    ax3.set_xticks(xd, [f"{r['bins'][0]}-{r['bins'][1]}" for r in num["key_noise_by_distance"]])
    ax3.set(ylim=(0, None), xlabel="bins of 10 deg between the bin and the progenitor",
            ylabel="|track with key A - track with key B| / data error of the bin", title="Particle noise grows slowly along the stream and stays below the errors")  # fmt: skip
    ax3.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    LS.save(fig, out, "track_noise")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--report-only", action="store_true", help="track_noise.json and the figure from an existing track_noise.npz")
    ap.add_argument("--n", type=int, default=40000, help="mock observations whose loss is stored")
    ap.add_argument("--n-sim", type=int, default=6000, help="the first this-many are simulated again")
    ap.add_argument("--steps", default="1e-6,1e-4,1e-2", help="parameter shifts, in prior standard deviations")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    cfg = Config.load(REPO / a.config)[0]
    steps = [float(x) for x in a.steps.split(",")]
    out = REPO / cfg.out / "plots" / cfg.subdir / "track_noise"
    out.mkdir(parents=True, exist_ok=True)
    if a.report_only:
        return report(out, cfg.n_min)
    tab = stream_table(cfg, a.workers)  # before jax is imported: the pool forks
    cfg.device = pick_device(cpu=False)  # a free GPU
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # the simulator shares the card

    import keras

    import networks  # noqa: F401  registers the custom layers
    from data import load_simulations, split, to_simulator
    from likelihood import log_likelihood, model_at
    from mock import fixed_observations

    sims = load_simulations(cfg)
    assert np.array_equal(tab["theta"].astype(np.float64), sims["theta"]), "stream table and load_simulations disagree"
    prior, grid, sc = sims["prior"], sims["grid"], sims["sim_config"]
    held = split(sims, cfg.val_frac)[1]
    cut = len(sims["theta"]) - len(held["zeta13"])
    test = fixed_observations(held, cfg, grid, sims["design"], cfg.seed + 7, a.n, unweighted=True)
    stream = cut + test["pick"]
    res = dict(stream=stream, S=test["S"], l=test["l"], n_obs=test["mask"].sum(1), sig_sys=test["sig_sys"], mask=test["mask"], sig=test["sig"],
               sig_med=np.array([np.median(s[m]) for s, m in zip(test["sig"], test["mask"])]), zeta=test["zeta"],
               **{k: tab[k][stream] for k in ("cls", "dE", "n_orb", "width", "L", "unbound", "nonmono")})  # fmt: skip

    # --- loss of the trained run on every mock
    if (cfg.run_dir() / "model.keras").exists():
        est = keras.saving.load_model(str(cfg.run_dir() / "model.keras"))
        fixed = np.ones_like(test["zeta"])
        lp = [np.asarray(est.log_prob(dict(zeta=test["zeta"][s], tokens=test["tokens"][s], mask=test["mask"][s], cond=test["cond"][s],
                                           fixed_target_mask=fixed[s], fixed_target_value=0 * fixed[s])))
              for s in (slice(i, i + 4096) for i in range(0, a.n, 4096))]  # fmt: skip
        res["nll"] = -np.concatenate(lp)
        print(f"loss of {cfg.name} on {a.n} mocks: {res['nll'].mean():.3f} nats", flush=True)
        del est

    # --- the streams of the first n_sim mocks again
    M, T = a.n_sim, cfg.span_max
    th, dis = sims["theta"][stream[:M]], held["dissolved"][test["pick"][:M]] > 0.5
    z13 = prior.to_latent(sims["phys13"][stream[:M]], cols=range(13), eps=1e-12)
    eps = np.random.default_rng(cfg.seed).standard_normal(z13.shape)
    eps /= np.linalg.norm(eps, axis=1, keepdims=True)
    runs = dict(A=simulate(th, dis, sc, cfg, out / "sims/A", 11), B=simulate(th, dis, sc, cfg, out / "sims/B", 12))
    for s in steps:
        shifted = to_simulator(np.concatenate([prior.from_latent(z13 + s * eps, cols=range(13)), np.zeros((M, 1))], axis=1))
        runs[f"C{s:g}"] = simulate(shifted, dis, sc, cfg, out / f"sims/C{s:g}", 11)
    obs, sig, r, l = test["mask"][:M], test["sig"][:M], test["r"][:M], test["l"][:M]
    var = sig**2 + test["sig_sys"][:M, None] ** 2
    at = {k: tuple(np.asarray(x) for x in model_at(v, grid.i0, l, T)) for k, v in runs.items()}
    at["0"] = (test["m"][:M], test["hw"][:M], test["n"][:M])  # the stored track, as the mock used it
    for k, (m, hw, n) in at.items():
        res[f"m_{k}"], res[f"hw_{k}"], res[f"n_{k}"] = m, hw, n
        res[f"ll_{k}"] = np.asarray(log_likelihood(r, sig, obs, m, hw, n, test["sig_sys"][:M, None], cfg.n_min, cfg.var_ratio))
        res[f"g1_{k}"] = np.all(~obs | (n >= cfg.n_min), axis=1)  # guard 1 alone
    res["r"], res["var"] = r, var
    np.savez(out / "track_noise.npz", **res, steps=steps, classes=CLASS_KEYS)
    print(f"wrote {out / 'track_noise.npz'}")
    report(out, cfg.n_min)


if __name__ == "__main__":
    main()
