"""Posterior draws -> simulations -> posterior predictive check and importance weights.

Shared by evaluation.py (mock tracks) and evaluation_real.py (observed tracks). For every track:

  1. draw from the posterior and keep the draws (draws.npz), so a second call reuses them and their simulations;
  2. simulate the first `ppc_draws` of them again: generate.py with `--theta-file` and the physics of the
     training set (or another integrator, `sim_integrator`), on GPUs (`sim_devices`) or on `sim_workers` CPU
     processes;
  3. replicated data: the simulated model track plus noise drawn from the likelihood,
     r_rep ~ N(m, sigma_data^2 + sigma_sys^2), in the bins of the observed track. A draw "covers" the track if
     its model reaches every observed bin (the first guard of the paper). Drawn against the angle (ppc_p*.png)
     and, for tracks that carry their observed angle `angle0`, in the plane of the sky (sky_p*.png);
  4. predictive p-value: the share of covering draws whose replicated data fit their own model worse (larger
     chi-square) than the observed data do. Near 0 = the data are not typical of what the posterior predicts;
  5. importance weights of the same draws towards prior x likelihood of the paper (likelihood.py), if the
     inference network gives log q. Their efficiency (effective sample size / draws) is 1 for an exact posterior.
     The figures show the predictive of the reweighted posterior next to the network's (`reweighted`).
"""

from __future__ import annotations

import os
import subprocess

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import REPO, Config, free_gpus
from data import I_PROG, I_SIG, I_X0, to_simulator
from likelihood import importance_weights, log_likelihood, model_at

INPUTS = ("tokens", "mask", "cond", "fixed_target_mask", "fixed_target_value")
ORANGE = "#c05621"  # the importance-weighted posterior, in every figure


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


def sim_devices(cfg: Config) -> list[str]:
    """GPUs for the simulations, [] = CPU. The card this evaluation holds comes first: the network needs little
    of it once the evaluation does not preallocate (XLA_PYTHON_CLIENT_PREALLOCATE=false, set by the evaluation
    scripts), so the simulator shares it. Then the cards that are free right now, at most `sim_workers` in all.
    Under Slurm only the allocation is used."""
    own = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]
    if cfg.sim_device != "gpu" or not own:
        return []
    more = [] if os.environ.get("SLURM_JOB_ID") else [uuid for _, uuid in free_gpus() if uuid not in own]
    return (own + more)[: max(1, cfg.sim_workers)]


def simulate(theta: np.ndarray, dissolved: np.ndarray, sc: dict, cfg: Config, out) -> dict:
    """Run generate.py on given parameters and return the binned tracks. Skipped if `out` already holds them."""
    out.mkdir(parents=True, exist_ok=True)
    gpus = sim_devices(cfg)
    n, w = len(theta), len(gpus) or max(1, min(cfg.sim_workers, len(theta)))
    if not ((out / "theta.npz").exists() and np.array_equal(np.load(out / "theta.npz")["theta"], theta) and (out / "data/manifest.json").exists()):
        np.savez(out / "theta.npz", theta=theta.astype(np.float64), dissolved=dissolved.astype(bool))
        sc = sc | (dict(integrator=cfg.sim_integrator) if cfg.sim_integrator else {})
        physics = [f for k in ("integrator", "solver", "rtol", "max_steps", "n_particles", "n_steps", "halo_flattening")
                   for f in (f"--{k.replace('_', '-')}", str(sc[k]))]  # fmt: skip
        where = (["--gpus", ",".join(gpus)] + (["--batch-size", str(cfg.sim_batch)] if cfg.sim_batch else []) if gpus else
                 ["--cpu", "--batch-size", "1", "--cpu-workers", str(w)] + (["--cpu-cores", cfg.sim_cores] if cfg.sim_cores else []))  # fmt: skip
        shard = min(-(-n // w), 1024) if gpus else -(-n // w)  # one shard per CPU process; shards of at most 1024 on a GPU
        cmd = ["uv", "run", "--script", str(REPO / "scripts/extragalactic/generate.py"), "run", "--overwrite", *where,
               "--theta-file", str(out / "theta.npz"), "--n-streams", str(n), "--shard-size", str(shard),
               "--n-save-particles", "0", "--out", str(out.relative_to(REPO)), "--track-ordering", cfg.ordering,
               "--progenitor-mass-loss", "mixed", *physics]  # fmt: skip
        print(f"simulating {n} posterior draws ({sc['integrator']}) on {'GPU ' + ', '.join(gpus) if gpus else f'{w} CPU processes'}", flush=True)
        with open(out / "generate.log", "w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    return read_tracks(out)


def read_tracks(out) -> dict:
    """The binned tracks generate.py left in `out`, in the order of its theta.npz. Simulates nothing."""
    cols = {k: [] for k in ("index", "status", "track/count", "track/r_med", "track/r_hw")}
    for p in sorted((out / "data").glob("shard_*.h5")):
        with h5py.File(p, "r") as f:
            for k in cols:
                cols[k].append(f[k][:])
    order = np.argsort(np.concatenate(cols["index"]))
    get = lambda k: np.nan_to_num(np.concatenate(cols[k])[order]).astype(np.float32)
    return dict(count=get("track/count"), r_med=get("track/r_med"), r_hw=get("track/r_hw"), failed=(get("status").astype(int) & 16) > 0)


def reweighted(m, var, w, n: int = 4000):
    """Predictive of the importance-weighted posterior of one track: draws resampled by their weights `w` (M,),
    each with new noise. m, var (M, bins). Returns (replicated data, model tracks), both (n, bins), or None
    without usable weights."""
    if w is None or var is None or not np.isfinite(w).all() or w.sum() <= 0:
        return None
    rng = np.random.default_rng(0)
    k = rng.choice(len(w), n, p=w / w.sum())
    return m[k] + np.sqrt(var[k]) * rng.standard_normal((n, m.shape[1])), m[k]


def ppc_title(t: dict, c, row: dict) -> str:
    eff = f"\nIS efficiency {row['importance_efficiency']:.3f}, effective sample size {row['effective_samples']:.1f} of {len(c)}" if "importance_efficiency" in row else ""
    return f"{t['title']}\ncovered {c.mean():.0%}, predictive p = {row['p_value']:.2f}{eff}"


def ppc_panel(ax, t: dict, rep, m, c, row: dict, legend: bool = False, fontsize: float = 8, w=None, var=None, name: str = "network posterior") -> None:
    """The predictive check of one track on `ax`. Blue: replicated data of the covering draws of the network's
    posterior (or of the posterior called `name`) and their median model track. Orange lines, if importance weights `w` and the likelihood variance
    `var` are given: the same for the reweighted posterior (`reweighted`). Black: the data (green: the true
    model track of a mock)."""
    o = t["mask"]
    phi = t["phi"][o]
    if c.any():
        lo2, lo1, hi1, hi2 = np.percentile(rep[c][:, o], [2.5, 16, 84, 97.5], axis=0)
        ax.fill_between(phi, lo2, hi2, color="#9ecae1", lw=0, label=f"{name}: replicated data, 95 %")
        ax.fill_between(phi, lo1, hi1, color="#4292c6", lw=0, label="68 %")
        ax.plot(phi, np.median(m[c][:, o], axis=0), color="#08306b", lw=1.2, label="median model track")
    rw = reweighted(m, var, w)
    if rw is not None:
        lo2, lo1, hi1, hi2 = np.percentile(rw[0][:, o], [2.5, 16, 84, 97.5], axis=0)
        for lo, hi, ls, lab in ((lo2, hi2, ":", "reweighted posterior: replicated data, 95 %"), (lo1, hi1, "-", "68 %")):
            ax.plot(phi, lo, color=ORANGE, lw=1.0, ls=ls, label=lab)
            ax.plot(phi, hi, color=ORANGE, lw=1.0, ls=ls)
        ax.plot(phi, np.median(rw[1][:, o], axis=0), color=ORANGE, lw=1.6, ls="--", label="median model track")
    if "m" in t:
        ax.plot(phi, t["m"][o], color="#2f855a", lw=1.2, ls="--", label="true model track")
    ax.errorbar(phi, t["r"][o], yerr=t["sig"][o], fmt="o", ms=3, lw=0.8, color="k", zorder=5, label="data")
    ax.set_title(ppc_title(t, c, row), fontsize=fontsize, loc="left")
    ax.set_xlabel("angle minus median angle [deg]", fontsize=fontsize)
    ax.set_ylabel("projected radius [kpc]", fontsize=fontsize)
    if legend:
        ax.legend(fontsize=fontsize - 2, frameon=False)


def replicate(sim: dict, par: np.ndarray, tracks: list[dict], grid, cfg: Config) -> dict:
    """Step 3 for the simulated draws `par` (tracks, draws, 15) of `tracks`: the model track of every draw in
    the bins of its observed track (m, hw, n), the bins `l` left of its progenitor, the variance of the
    likelihood, the covering draws and the replicated data. All (tracks, draws, ...)."""
    K, M, T = *par.shape[:2], cfg.span_max
    l = np.floor(par[..., I_PROG] * (np.array([t["S"] for t in tracks])[:, None] + 1)).astype(int)  # bins left of the progenitor
    m, hw, n = (np.asarray(x).reshape(K, M, T) for x in model_at(sim, grid.i0, l.reshape(-1), T))
    sig, mask = (np.stack([t[k] for t in tracks])[:, None] for k in ("sig", "mask"))
    var = sig**2 + par[..., I_SIG, None] ** 2
    cover = np.all(~mask | (n >= cfg.n_min), axis=-1) & ~sim["failed"].reshape(K, M)
    rep = m + np.sqrt(var) * np.random.default_rng(cfg.seed).standard_normal(m.shape)
    return dict(l=l, m=m, hw=hw, n=n, var=var, cover=cover, rep=rep)


def sky_panel(ax, t: dict, z: dict, i: int, x0, bin_deg: float, row: dict, legend: bool = False, fontsize: float = 8) -> None:
    """`ppc_panel` in the plane of the track files (x, y in kpc, host at the origin): every bin is drawn at its
    observed angle `angle0` + phi, so bands and error bars are radial. `z` = `replicate` (with the importance
    `weights`, if any, for the orange lines of the reweighted posterior), i = the track in it,
    x0 = projected radius of the progenitor per draw. A draw's progenitor is on the left edge of its bin l.
    The panel is framed on the host, the data and the 68 % band; a wide 95 % band runs off it. Replicated radii
    below zero are drawn at zero."""
    o, c = t["mask"], z["cover"][i]
    r, e = t["r"][o], t["sig"][o]
    reach = [r + e]
    ang = np.deg2rad(t["angle0"] + t["phi"][o])
    cs, sn = np.cos(ang), np.sin(ang)
    if c.any():
        lo2, lo1, hi1, hi2 = np.clip(np.percentile(z["rep"][i][c][:, o], [2.5, 16, 84, 97.5], axis=0), 0.0, None)
        reach.append(hi1)
        for lo, hi, col, lab in ((lo2, hi2, "#9ecae1", "network posterior: replicated data, 95 %"), (lo1, hi1, "#4292c6", "68 %")):
            for k in range(len(ang) - 1):  # one piece per pair of bins: a track may wind around the host more than once
                s = slice(k, k + 2)
                ax.fill(np.r_[lo[s] * cs[s], (hi[s] * cs[s])[::-1]], np.r_[lo[s] * sn[s], (hi[s] * sn[s])[::-1]], fc=col, ec=col, lw=0.4, label=lab if k == 0 else None)
        mm = np.median(z["m"][i][c][:, o], axis=0)
        ax.plot(mm * cs, mm * sn, color="#08306b", lw=1.2, label="median model track")
    rw = reweighted(z["m"][i], z["var"][i], z["weights"][i] if "weights" in z else None)
    if rw is not None:  # the reweighted posterior, as in `ppc_panel`
        lo2, lo1, hi1, hi2 = np.clip(np.percentile(rw[0][:, o], [2.5, 16, 84, 97.5], axis=0), 0.0, None)
        reach.append(hi1)
        for lo, hi, ls, lab in ((lo2, hi2, ":", "reweighted posterior: replicated data, 95 %"), (lo1, hi1, "-", "68 %")):
            ax.plot(lo * cs, lo * sn, color=ORANGE, lw=1.0, ls=ls, zorder=3, label=lab)
            ax.plot(hi * cs, hi * sn, color=ORANGE, lw=1.0, ls=ls, zorder=3)
        mm = np.median(rw[1][:, o], axis=0)
        ax.plot(mm * cs, mm * sn, color=ORANGE, lw=1.6, ls="--", zorder=3, label="median model track")
    if c.any():
        ap = np.deg2rad(t["angle0"] + t["phi"][0] + (z["l"][i][c] - 0.5) * bin_deg)
        ax.scatter(x0[c] * np.cos(ap), x0[c] * np.sin(ap), s=5, color="#2f855a", alpha=0.35, lw=0, zorder=4, label="progenitor of each draw")
    ax.plot([(r - e) * cs, (r + e) * cs], [(r - e) * sn, (r + e) * sn], color="k", lw=0.8, zorder=5)
    ax.plot(r * cs, r * sn, "o", ms=3, color="k", zorder=5, label="data")
    ax.plot(r[0] * cs[0], r[0] * sn[0], "o", ms=7, mfc="none", color="k", zorder=5, label="first bin")
    ax.plot(0, 0, "+", ms=9, color="#7b8794", zorder=5, label="host")
    px, py = np.r_[np.concatenate([q * cs for q in reach]), 0.0], np.r_[np.concatenate([q * sn for q in reach]), 0.0]
    half = 0.58 * max(np.ptp(px), np.ptp(py))
    ax.set_xlim(0.5 * (px.min() + px.max()) - half, 0.5 * (px.min() + px.max()) + half)
    ax.set_ylim(0.5 * (py.min() + py.max()) - half, 0.5 * (py.min() + py.max()) + half)
    ax.set_aspect("equal")
    ax.set_title(ppc_title(t, c, row), fontsize=fontsize, loc="left")
    ax.set_xlabel("x [kpc]", fontsize=fontsize)
    ax.set_ylabel("y [kpc]", fontsize=fontsize)
    if legend:
        ax.legend(fontsize=fontsize - 2, frameon=False)


def sky_pages(tracks: list[dict], z: dict, par: np.ndarray, rows: list[dict], grid, out) -> None:
    """sky_p*.png in `out`: `sky_panel` of every track, 8 per page."""
    for page in range(-(-len(tracks) // 8)):
        part = range(8 * page, min(len(tracks), 8 * page + 8))
        fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
        for k, (ax, i) in enumerate(zip(axes.ravel(), part)):
            sky_panel(ax, tracks[i], z, i, par[i, :, I_X0], grid.bin_deg, rows[i], legend=k == 0)
        for ax in axes.ravel()[len(part) :]:
            ax.set_axis_off()
        fig.savefig(out / f"sky_p{page + 1}.png", dpi=120)
        plt.close(fig)


def predictive(est, tracks: list[dict], zeta: np.ndarray, prior, grid, sc: dict, cfg: Config, out) -> tuple[list[dict], dict]:
    """Steps 2-5 for `tracks` (network inputs, r, sig, mask, phi, S, title; mock tracks also carry the true model
    track `m`) and their posterior draws `zeta` (tracks, >= ppc_draws, 15). Writes ppc_p*.png and weights.npz
    into `out`; returns one row of numbers per track, and the arrays behind them (replicated data `rep`, model
    tracks `m`, covering draws `cover`, importance `weights`, and the rest of `replicate`; all (tracks, ppc_draws,
    ...)) for other figures."""
    M, T, K = cfg.ppc_draws, cfg.span_max, len(tracks)
    zeta = zeta[:, :M]
    par = prior.from_latent(zeta)
    sim = simulate(to_simulator(par[..., :14]).reshape(-1, 13), np.repeat([t["cond"][0] > 0.5 for t in tracks], M), sc, cfg, out / "simulations")
    z = replicate(sim, par, tracks, grid, cfg)
    m, hw, n, var, cover, rep = (z[k] for k in ("m", "hw", "n", "var", "cover", "rep"))
    r, mask = (np.stack([t[k] for t in tracks])[:, None] for k in ("r", "mask"))
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
        fig, axes = plt.subplots(2, 4, figsize=(17, 7.6), constrained_layout=True)
        for k, (ax, i) in enumerate(zip(axes.ravel(), part)):
            ppc_panel(ax, tracks[i], rep[i], m[i], cover[i], rows[i], legend=k == 0, w=weights[i], var=var[i])
        for ax in axes.ravel()[len(part) :]:
            ax.set_axis_off()
        fig.savefig(out / f"ppc_p{page + 1}.png", dpi=120)
        plt.close(fig)
    return rows, z | dict(weights=weights)
