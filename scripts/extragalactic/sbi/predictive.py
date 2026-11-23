"""Posterior draws -> simulations -> posterior predictive check and importance weights.

Shared by evaluation.py (mock tracks) and evaluation_real.py (observed tracks). For every track:

  1. draw from the posterior and keep the draws (draws.npz), so a second call reuses them and their simulations;
  2. simulate the first `ppc_draws` of them again: generate.py with `--theta-file` and the physics of the
     training set, on `sim_workers` CPU processes;
  3. replicated data: the simulated model track plus noise drawn from the likelihood,
     r_rep ~ N(m, sigma_data^2 + sigma_sys^2), in the bins of the observed track. A draw "covers" the track if
     its model reaches every observed bin (the first guard of the paper);
  4. predictive p-value: the share of covering draws whose replicated data fit their own model worse (larger
     chi-square) than the observed data do. Near 0 = the data are not typical of what the posterior predicts;
  5. importance weights of the same draws towards prior x likelihood of the paper (likelihood.py), if the
     inference network gives log q. Their efficiency (effective sample size / draws) is 1 for an exact posterior.
"""

from __future__ import annotations

import subprocess

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import REPO, Config
from data import I_PROG, I_SIG, to_simulator
from likelihood import importance_weights, log_likelihood, model_at

INPUTS = ("tokens", "mask", "cond", "fixed_target_mask", "fixed_target_value")


def sample(est, obs: dict, n: int, seed: int) -> np.ndarray:
    """Posterior draws in latent coordinates, (tracks, n, 15). A fixed parameter comes back at its given value."""
    return est.sample(num_samples=n, conditions={k: obs[k] for k in INPUTS}, batch_size=64, seed=seed)["zeta"]


def draws(est, tracks: list[dict], n: int, cfg: Config, out) -> np.ndarray:
    """Posterior draws for a list of tracks, read back from `out`/draws.npz if they were made before."""
    f = out / "draws.npz"
    names = np.array([t["title"] for t in tracks])
    if f.exists() and np.array_equal(np.load(f)["names"], names) and np.load(f)["zeta"].shape[1] == n:
        return np.load(f)["zeta"]
    zeta = sample(est, {k: np.stack([t[k] for t in tracks]) for k in INPUTS}, n, cfg.seed)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(f, names=names, zeta=zeta)
    return zeta


def simulate(theta: np.ndarray, dissolved: np.ndarray, sc: dict, cfg: Config, out) -> dict:
    """Run generate.py on given parameters and return the binned tracks. Skipped if `out` already holds them."""
    out.mkdir(parents=True, exist_ok=True)
    n, w = len(theta), max(1, min(cfg.sim_workers, len(theta)))
    if not ((out / "theta.npz").exists() and np.array_equal(np.load(out / "theta.npz")["theta"], theta) and (out / "data/manifest.json").exists()):
        np.savez(out / "theta.npz", theta=theta.astype(np.float64), dissolved=dissolved.astype(bool))
        physics = [f for k in ("integrator", "solver", "rtol", "max_steps", "n_particles", "n_steps", "halo_flattening")
                   for f in (f"--{k.replace('_', '-')}", str(sc[k]))]  # fmt: skip
        cmd = ["uv", "run", "--script", str(REPO / "scripts/extragalactic/generate.py"), "run", "--cpu", "--batch-size", "1", "--overwrite",
               "--cpu-workers", str(w), "--theta-file", str(out / "theta.npz"), "--n-streams", str(n), "--shard-size", str(-(-n // w)),
               "--n-save-particles", "0", "--out", str(out.relative_to(REPO)), "--track-ordering", cfg.ordering,
               "--progenitor-mass-loss", "mixed", *physics] + (["--cpu-cores", cfg.sim_cores] if cfg.sim_cores else [])  # fmt: skip
        with open(out / "generate.log", "w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    cols = {k: [] for k in ("index", "status", "track/count", "track/r_med", "track/r_hw")}
    for p in sorted((out / "data").glob("shard_*.h5")):
        with h5py.File(p, "r") as f:
            for k in cols:
                cols[k].append(f[k][:])
    order = np.argsort(np.concatenate(cols["index"]))
    get = lambda k: np.nan_to_num(np.concatenate(cols[k])[order]).astype(np.float32)
    return dict(count=get("track/count"), r_med=get("track/r_med"), r_hw=get("track/r_hw"), failed=(get("status").astype(int) & 16) > 0)


def predictive(est, tracks: list[dict], zeta: np.ndarray, prior, grid, sc: dict, cfg: Config, out) -> list[dict]:
    """Steps 2-5 for `tracks` (network inputs, r, sig, mask, phi, S, title; mock tracks also carry the true model
    track `m`) and their posterior draws `zeta` (tracks, >= ppc_draws, 15). Writes ppc_p*.png and weights.npz
    into `out`; returns one row of numbers per track."""
    M, T, K = cfg.ppc_draws, cfg.span_max, len(tracks)
    zeta = zeta[:, :M]
    par = prior.from_latent(zeta)
    sim = simulate(to_simulator(par[..., :14]).reshape(-1, 13), np.repeat([t["cond"][0] > 0.5 for t in tracks], M), sc, cfg, out / "simulations")
    l = np.floor(par[..., I_PROG] * (np.array([t["S"] for t in tracks])[:, None] + 1)).astype(int)  # bins left of the progenitor
    m, hw, n = (np.asarray(x).reshape(K, M, T) for x in model_at(sim, grid.i0, l.reshape(-1), T))
    r, sig, mask = (np.stack([t[k] for t in tracks])[:, None] for k in ("r", "sig", "mask"))
    var = sig**2 + par[..., I_SIG, None] ** 2
    cover = np.all(~mask | (n >= cfg.n_min), axis=-1) & ~sim["failed"].reshape(K, M)
    rep = m + np.sqrt(var) * np.random.default_rng(cfg.seed).standard_normal(m.shape)
    chi2 = lambda d: np.sum(np.where(mask, (d - m) ** 2 / var, 0.0), axis=-1)
    worse = chi2(rep) >= chi2(r)

    rows, weights = [], np.full((K, M), np.nan)
    for i, t in enumerate(tracks):
        c = cover[i]
        row = dict(track=t["title"], covering=float(c.mean()), p_value=float(worse[i][c].mean()) if c.any() else float("nan"))
        if cfg.importance and c.any():
            try:  # log q: exact for a coupling flow, an ODE solve for diffusion / flow matching
                log_q = est.log_prob(dict(zeta=zeta[i]) | {k: np.repeat(t[k][None], M, axis=0) for k in INPUTS})
                ll = np.asarray(log_likelihood(t["r"], t["sig"], t["mask"], m[i], hw[i], n[i], par[i, :, I_SIG, None], cfg.n_min, cfg.var_ratio))
                iw = importance_weights(zeta[i][:, t["fixed_target_mask"] > 0], log_q, np.where(c, ll, -np.inf))  # prior of the inferred ones only
                weights[i] = iw["weights"]
                row |= dict(importance_efficiency=iw["efficiency"], effective_samples=iw["ess"], log_evidence=iw["log_evidence"])
            except Exception as e:  # noqa: BLE001
                row["importance_error"] = f"{type(e).__name__}: {e}"[:200]
        rows.append(row)
    np.savez(out / "weights.npz", names=[t["title"] for t in tracks], zeta=zeta, params=par, weights=weights, covering=cover)

    for page in range(-(-K // 8)):  # 8 tracks per page
        part = range(8 * page, min(K, 8 * page + 8))
        fig, axes = plt.subplots(2, 4, figsize=(17, 7), constrained_layout=True)
        for ax, i in zip(axes.ravel(), part):
            t, o, c, row = tracks[i], tracks[i]["mask"], cover[i], rows[i]
            if c.any():
                lo2, lo1, hi1, hi2 = np.percentile(rep[i][c][:, o], [2.5, 16, 84, 97.5], axis=0)
                ax.fill_between(t["phi"][o], lo2, hi2, color="#9ecae1", lw=0, label="replicated data, 95 %")
                ax.fill_between(t["phi"][o], lo1, hi1, color="#4292c6", lw=0, label="68 %")
                ax.plot(t["phi"][o], np.median(m[i][c][:, o], axis=0), color="#08306b", lw=1.2, label="median model track")
            if "m" in t:
                ax.plot(t["phi"][o], t["m"][o], color="#2f855a", lw=1.2, ls="--", label="true model track")
            ax.errorbar(t["phi"][o], t["r"][o], yerr=t["sig"][o], fmt="o", ms=3, lw=0.8, color="#c05621", label="data")
            eff = f", IS efficiency {row['importance_efficiency']:.2f}" if "importance_efficiency" in row else ""
            ax.set_title(f"{t['title']}\ncovered {c.mean():.0%}, predictive p = {row['p_value']:.2f}{eff}", fontsize=8, loc="left")
            ax.set_xlabel("angle minus median angle [deg]")
            ax.set_ylabel("projected radius [kpc]")
        for ax in axes.ravel()[len(part) :]:
            ax.set_axis_off()
        axes.ravel()[0].legend(fontsize=6, frameon=False)
        fig.savefig(out / f"ppc_p{page + 1}.png", dpi=120)
        plt.close(fig)
    return rows
