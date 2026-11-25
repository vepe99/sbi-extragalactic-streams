"""Posterior draws -> simulations -> posterior predictive check and importance weights.

Shared by evaluation.py (mock tracks) and evaluation_real.py (observed tracks). For every track:

  1. draw from the posterior and keep the draws (draws.npz), so a second call reuses them and their simulations;
  2. simulate the first `ppc_draws` of them again: generate.py with `--theta-file` and the physics of the
     training set (or another integrator, `sim_integrator`), on GPUs (`sim_devices`) or on `sim_workers` CPU
     processes. For a run trained with `replace_tracks` the draws whose stream is one that set would have
     replaced are then simulated once more with its particle number, as the training set was made
     (`more_particles`), and guard 1 asks those for the same fraction of the particles;
  3. replicated data: the simulated model track plus noise drawn from the likelihood,
     r_rep ~ N(m, sigma_data^2 + sigma_sys^2), in the bins of the observed track. A draw "covers" the track if
     its model reaches every observed bin (the first guard of the paper). Drawn against the angle
     (ppc/ppc_p*.png) and, for tracks that carry their observed angle `angle0`, in the plane of the sky
     (sky/sky_p*.png);
  4. predictive p-value: the share of covering draws whose replicated data fit their own model worse (larger
     chi-square) than the observed data do. Near 0 = the data are not typical of what the posterior predicts;
  5. importance weights of the same draws towards prior x likelihood of the paper (likelihood.py), if the
     inference network gives log q. Their efficiency (effective sample size / draws) is 1 for an exact posterior.
     The figures show the predictive of the reweighted posterior next to the network's (`reweighted`);
  6. the acceleration of the host at the bins of the track (`accelerations`): every simulated draw gives the
     place of its stream in three dimensions (generate.py stores the mean line-of-sight coordinate of the
     particles of each bin), and the potential of the draw gives the acceleration there. Its projection on the
     sky is drawn as arrows in the sky panel of the corner plots.
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
from data import (I_PROG, I_SIG, I_X0, KPC_GYR2_TO_KMS_MYR, chosen_for_replacement, host_acceleration, replacement_rule, resimulated_ok,
                  to_simulator)  # fmt: skip
from likelihood import importance_weights, log_likelihood, model_at

INPUTS = ("tokens", "mask", "cond", "fixed_target_mask", "fixed_target_value")
ORANGE = "#c05621"  # the importance-weighted posterior, in every figure
MORE, MORE_SEED = "simulations_more_particles", 90  # second simulation of the long draws (`more_particles`): folder, master key
TRUTH = "#9b2c8a"  # the truth of a mock track: its model track and progenitor on the sky, its parameters in a corner plot


def figures(out, name: str):
    """The subfolder `name` of an evaluation directory, made if missing. The figures of an evaluation go to
    subfolders, one per kind; the files other code reads (npz, metrics.json, simulations/) stay at the top."""
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def sample(est, obs: dict, n: int, seed: int) -> np.ndarray:
    """Posterior draws in latent coordinates, (tracks, n, 15). A fixed parameter comes back at its given value."""
    return est.sample(num_samples=n, conditions={k: obs[k] for k in INPUTS}, batch_size=64, seed=seed)["zeta"]


def model_saved(cfg: Config) -> int:
    """When the model.keras of the run was written (ns): which checkpoint an evaluation looked at. A run that
    is still training, or is taken up again, rewrites the file whenever the held-out loss improves."""
    return (cfg.run_dir() / "model.keras").stat().st_mtime_ns


def draws(est, tracks: list[dict], n: int, cfg: Config, out) -> np.ndarray:
    """Posterior draws for a list of tracks, read back from `out`/draws.npz if they were made before, for the
    same tracks, with the model.keras there is now (draws kept without that record are taken as they are)."""
    f = out / "draws.npz"
    names, stamp = np.array([t["title"] for t in tracks]), model_saved(cfg)
    if f.exists():
        old = np.load(f)
        if np.array_equal(old["names"], names) and old["zeta"].shape[1] == n and ("model_saved" not in old.files or int(old["model_saved"]) == stamp):
            return old["zeta"]
    zeta = sample(est, {k: np.stack([t[k] for t in tracks]) for k in INPUTS}, n, cfg.seed)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(f, names=names, zeta=zeta, model_saved=stamp)
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


def simulate(theta: np.ndarray, dissolved: np.ndarray, sc: dict, cfg: Config, out, seed: int | None = None) -> dict:
    """Run generate.py on given parameters and return the binned tracks. Skipped if `out` already holds them.
    The particle number is that of `sc`. seed: master key of the particle release (stream k uses
    fold_in(key(seed), k)); None = generate.py's default."""
    out.mkdir(parents=True, exist_ok=True)
    gpus = sim_devices(cfg)
    n, w = len(theta), len(gpus) or max(1, min(cfg.sim_workers, len(theta)))
    # simulations kept from before generate.py stored the line-of-sight coordinate (track/los_mean) are made again
    if not ((out / "theta.npz").exists() and np.array_equal(np.load(out / "theta.npz")["theta"], theta) and (out / "data/manifest.json").exists()
            and has_depth(out)):  # fmt: skip
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
               "--progenitor-mass-loss", "mixed", *physics, *(["--seed", str(seed)] if seed is not None else [])]  # fmt: skip
        print(f"simulating {n} posterior draws ({sc['integrator']}, {sc['n_particles']} particles) on {'GPU ' + ', '.join(gpus) if gpus else f'{w} CPU processes'}", flush=True)
        with open(out / "generate.log", "w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    return read_tracks(out)


def has_depth(out) -> bool:
    """Whether the simulations in `out` hold the mean line-of-sight coordinate of every bin (track/los_mean)."""
    files = sorted((out / "data").glob("shard_*.h5"))
    if not files:
        return False
    with h5py.File(files[0], "r") as f:
        return "track/los_mean" in f


def read_tracks(out) -> dict:
    """The binned tracks generate.py left in `out`, in the order of its theta.npz. Simulates nothing. `los`
    (the mean line-of-sight coordinate per bin) is there if the simulations hold it."""
    depth = has_depth(out)
    cols = {k: [] for k in ("index", "status", "diag/dE_rel", "diag/n_bad", "track/count", "track/r_med", "track/r_hw") + (("track/los_mean",) if depth else ())}
    for p in sorted((out / "data").glob("shard_*.h5")):
        with h5py.File(p, "r") as f:
            for k in cols:
                cols[k].append(f[k][:])
    order = np.argsort(np.concatenate(cols["index"]))
    get = lambda k: np.nan_to_num(np.concatenate(cols[k])[order]).astype(np.float32)
    status = np.concatenate(cols["status"])[order].astype(int)
    return dict(count=get("track/count"), r_med=get("track/r_med"), r_hw=get("track/r_hw"), failed=(status & 16) > 0, status=status,
                n_bad=np.concatenate(cols["diag/n_bad"])[order], dE=np.concatenate(cols["diag/dE_rel"])[order],
                **(dict(los=get("track/los_mean")) if depth else {}))  # fmt: skip


def merged(sim: dict, new: dict, rows: np.ndarray, ok: np.ndarray, n_min: float) -> dict:
    """`sim` with the tracks `new` of its streams `rows` in place of theirs where `ok`, and per stream the
    particles a bin needs (`n_min`) and whether it holds the new track (`more`)."""
    sim, use = {k: np.array(v) for k, v in sim.items()}, rows[ok]
    for k in new:
        if k in sim:
            sim[k][use] = new[k][ok]
    sim["n_min"][use], sim["more"][use] = n_min, True
    return sim


def more_particles(sim: dict, theta: np.ndarray, dissolved: np.ndarray, sc: dict, cfg: Config, i0: int, out) -> dict:
    """The simulations `sim` of the draws as the training set of the run holds such streams. Adds `n_min` (the
    particles a bin needs, per draw) and `more` (the draw holds a track with more particles).

    With `replace_tracks` the training set holds, for every stream that the rule of that set picks from its
    track at the particle number of `subdir` (data.chosen_for_replacement: a long stream with a resolved orbit),
    a second simulation with more particles (train_v2_long90k: 90,000). The particle number of a stream is so a
    function of its parameters and of its first realisation, and the likelihood of a draw has to be made the same
    way: the draws the rule picks are simulated again, with another key, into `out`/simulations_more_particles/,
    and use that track with guard 1 at the same fraction of the particles, unless it fails the loader's cuts
    (data.resimulated_ok; such a draw keeps its first track, as such a stream does in the training set).
    used.npz in that folder records which draws were simulated again and which use the new track."""
    sim = sim | dict(n_min=np.full(len(sim["count"]), cfg.n_min, np.float32), more=np.zeros(len(sim["count"]), bool))
    used = out / MORE / "used.npz"
    used.unlink(missing_ok=True)
    if not cfg.replace_tracks:
        return sim
    rule = replacement_rule(cfg)
    N = rule["n_particles"]
    n_min = cfg.n_min * N / sc["n_particles"]
    rows = np.flatnonzero(chosen_for_replacement(cfg, rule, sim, i0, sc["n_particles"]))
    if not len(rows):
        return sim
    new = simulate(theta[rows], dissolved[rows], sc | dict(n_particles=N), cfg, out / MORE, seed=MORE_SEED)
    ok = resimulated_ok(cfg, rule, new, i0, N, n_min)
    np.savez(used, rows=rows, ok=ok, n_min=n_min, n_particles=N)
    return merged(sim, new, rows, ok, n_min)


def read_simulations(out, cfg: Config) -> dict:
    """The simulations of the draws that a finished `predictive` left in the evaluation directory `out`, with
    the second simulations of `more_particles` in place where they were used. Simulates nothing."""
    sim = read_tracks(out / "simulations")
    sim |= dict(n_min=np.full(len(sim["count"]), cfg.n_min, np.float32), more=np.zeros(len(sim["count"]), bool))
    if (out / MORE / "used.npz").exists():
        u = np.load(out / MORE / "used.npz")
        sim = merged(sim, read_tracks(out / MORE), u["rows"], u["ok"], float(u["n_min"]))
    return sim


def reweighted(m, var, w, n: int = 4000):
    """Predictive of the importance-weighted posterior of one track: draws resampled by their weights `w` (M,),
    each with new noise. m, var (M, bins). Returns (replicated data, model tracks, the draw behind each), the
    first two (n, bins), or None without usable weights."""
    if w is None or var is None or not np.isfinite(w).all() or w.sum() <= 0:
        return None
    rng = np.random.default_rng(0)
    k = rng.choice(len(w), n, p=w / w.sum())
    return m[k] + np.sqrt(var[k]) * rng.standard_normal((n, m.shape[1])), m[k], k


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
    likelihood, the covering draws and the replicated data, and `los`, the mean line-of-sight coordinate of the
    stream in those bins, if the simulations hold it. All (tracks, draws, ...). With `de_max` > 0 a draw whose
    progenitor orbit is not resolved does not cover the track either. `sim` may carry the particles a bin needs
    per draw (`n_min`, from `more_particles`); otherwise it is cfg.n_min for all. Returned as `n_min` (tracks,
    draws), with `more` (the draw holds a track with more particles)."""
    K, M, T = *par.shape[:2], cfg.span_max
    l = np.floor(par[..., I_PROG] * (np.array([t["S"] for t in tracks])[:, None] + 1)).astype(int)  # bins left of the progenitor
    m, hw, n = (np.asarray(x).reshape(K, M, T) for x in model_at(sim, grid.i0, l.reshape(-1), T))
    sig, mask = (np.stack([t[k] for t in tracks])[:, None] for k in ("sig", "mask"))
    var = sig**2 + par[..., I_SIG, None] ** 2
    n_min = np.asarray(sim["n_min"], np.float32).reshape(K, M) if "n_min" in sim else np.full((K, M), cfg.n_min, np.float32)
    more = np.asarray(sim["more"]).reshape(K, M) if "more" in sim else np.zeros((K, M), bool)
    cover = np.all(~mask | (n >= n_min[..., None]), axis=-1) & ~sim["failed"].reshape(K, M)
    if cfg.de_max > 0:  # the prior of the run holds resolved orbits only: an unresolved draw is outside it
        cover &= ~(sim["dE"].reshape(K, M) > cfg.de_max)
    rep = m + np.sqrt(var) * np.random.default_rng(cfg.seed).standard_normal(m.shape)
    los = dict(los=np.asarray(model_at(sim, grid.i0, l.reshape(-1), T, keys=("los",))[0]).reshape(K, M, T)) if "los" in sim else {}
    return dict(l=l, m=m, hw=hw, n=n, var=var, cover=cover, rep=rep, n_min=n_min, more=more, **los)


def sky_acceleration(par, m, los, l, t: dict, bin_deg: float, flatten: str) -> np.ndarray:
    """Acceleration of the host of `par` (..., 15) at the bins of its stream, projected on the sky, in the frame
    the track `t` is drawn in (`sky_panel`) [km / s / Myr]; (..., bins, 2).

    m, los (..., bins): projected radius and line-of-sight coordinate of the stream in the bins of the window,
    l (...): bins left of the progenitor. In the frame of the simulation the progenitor is on the +x axis and
    bin k of the window is centred on the angle (k - l + 0.5) bin_deg; the figure shows that bin at
    angle0 + phi[k], so the two frames differ by a rotation about the line of sight."""
    m, los, l = np.asarray(m, np.float64), np.asarray(los, np.float64), np.asarray(l, np.float64)[..., None]
    s = np.deg2rad((np.arange(m.shape[-1]) - l + 0.5) * bin_deg)
    a = host_acceleration(par, np.stack([m * np.cos(s), m * np.sin(s), los], axis=-1), flatten) * KPC_GYR2_TO_KMS_MYR
    rot = np.deg2rad(t["angle0"] + float(t["phi"][0]) + (l - 0.5) * bin_deg)
    return np.stack([a[..., 0] * np.cos(rot) - a[..., 1] * np.sin(rot), a[..., 0] * np.sin(rot) + a[..., 1] * np.cos(rot)], axis=-1).astype(np.float32)


def accelerations(z: dict, par: np.ndarray, tracks: list[dict], grid, flatten: str, sim_true: dict | None = None, truth=None) -> dict:
    """`sky_acceleration` of every simulated draw: `acc` (tracks, draws, bins, 2), to add to `z` = `replicate`.
    Nothing if the simulations do not hold the line-of-sight coordinate, or the tracks have no angle on the sky.
    Mock tracks: with the simulations `sim_true` of their true parameters `truth` (tracks, 15), also `acc_true`
    (tracks, bins, 2), at the true model track; NaN in a bin the simulation of the truth left empty."""
    if "los" not in z or any("angle0" not in t for t in tracks):
        return {}
    out = dict(acc=np.stack([sky_acceleration(par[i], z["m"][i], z["los"][i], z["l"][i], t, grid.bin_deg, flatten) for i, t in enumerate(tracks)]))
    if sim_true is not None and "los" in sim_true and all("m" in t and "l" in t for t in tracks):
        l = np.array([int(t["l"]) for t in tracks])
        los, n = (np.asarray(x) for x in model_at(sim_true, grid.i0, l, z["m"].shape[-1], keys=("los", "count")))
        acc = np.stack([sky_acceleration(truth[i], t["m"], los[i], l[i], t, grid.bin_deg, flatten) for i, t in enumerate(tracks)])
        out["acc_true"] = np.where((n > 0)[..., None], acc, np.nan)
    return out


def acceleration_arrows(ax, sets: list, half: float, fontsize: float) -> None:
    """Arrows of the sky-projected acceleration on a sky panel. sets: (x, y, a (bins, 2), colour) per posterior
    (and for the truth of a mock track). One scale for all: the largest acceleration is 0.2 of the half-width
    of the panel, and the key in the lower right corner gives it in km / s / Myr. At most 24 arrows per set."""
    sets = [(x, y, a, col) for x, y, a, col in sets if len(x) and np.isfinite(a).any()]
    if not sets:
        return
    top = max(float(np.nanmax(np.hypot(a[:, 0], a[:, 1]))) for _, _, a, _ in sets)
    for x, y, a, col in sets:
        k = slice(None, None, -(-len(x) // 24))
        q = ax.quiver(x[k], y[k], a[k, 0], a[k, 1], color=col, angles="xy", scale_units="xy", scale=top / (0.2 * half), width=0.0035,
                      headwidth=4.0, headlength=5.0, zorder=7, ec="white", lw=0.3)  # fmt: skip
    key = float(f"{top:.1g}")
    ax.quiverkey(q, 0.86, 0.04, key, f"{key:g} km/s/Myr", coordinates="axes", labelpos="W", color="#52606d", fontproperties=dict(size=fontsize - 2))


def sky_panel(ax, t: dict, z: dict, i: int, x0, bin_deg: float, row: dict, legend: bool = False, fontsize: float = 8, accel: bool = False,
              name: str = "network posterior", other: tuple | None = None) -> None:
    """`ppc_panel` in the plane of the track files (x, y in kpc, host at the origin): every bin is drawn at its
    observed angle `angle0` + phi, so bands and error bars are radial. `z` = `replicate` (with the importance
    `weights`, if any, for the orange lines of the reweighted posterior), i = the track in it,
    x0 = projected radius of the progenitor per draw. A draw's progenitor is on the left edge of its bin l.
    The panel is framed on the host, the data and the 68 % band; a wide 95 % band runs off it. Replicated radii
    below zero are drawn at zero. A mock track (it carries its true model track `m`) also gets that track and,
    with `l` and `x0_true`, its true progenitor.

    accel: also the acceleration of the host at the bins, projected on the sky, as arrows (`accelerations`
    must be in `z`): the median over the covering draws of the network's posterior at their median model track
    (blue), the same for the reweighted posterior (orange), and for a mock track the truth (`acc_true`).

    name: what the posterior behind `z` is called in the legend. other = (z, i) of another `replicate`: the
    reweighted posterior is then that of its draws and `weights` (nested.py: blue is nested sampling, orange the
    importance-weighted draws of the network)."""
    arrows = []  # (x, y, acceleration, colour): drawn last, when the size of the panel is known
    accel = accel and "acc" in z
    o, c = t["mask"], z["cover"][i]
    r, e = t["r"][o], t["sig"][o]
    reach = [r + e]
    ang = np.deg2rad(t["angle0"] + t["phi"][o])
    cs, sn = np.cos(ang), np.sin(ang)
    if c.any():
        lo2, lo1, hi1, hi2 = np.clip(np.percentile(z["rep"][i][c][:, o], [2.5, 16, 84, 97.5], axis=0), 0.0, None)
        reach.append(hi1)
        for lo, hi, col, lab in ((lo2, hi2, "#9ecae1", f"{name}: replicated data, 95 %"), (lo1, hi1, "#4292c6", "68 %")):
            for k in range(len(ang) - 1):  # one piece per pair of bins: a track may wind around the host more than once
                s = slice(k, k + 2)
                ax.fill(np.r_[lo[s] * cs[s], (hi[s] * cs[s])[::-1]], np.r_[lo[s] * sn[s], (hi[s] * sn[s])[::-1]], fc=col, ec=col, lw=0.4, label=lab if k == 0 else None)
        mm = np.median(z["m"][i][c][:, o], axis=0)
        ax.plot(mm * cs, mm * sn, color="#08306b", lw=1.2, label="median model track")
        if accel:
            arrows.append((mm * cs, mm * sn, np.median(z["acc"][i][c][:, o], axis=0), "#08306b"))
    zw, iw = other or (z, i)
    rw = reweighted(zw["m"][iw], zw["var"][iw], zw["weights"][iw] if "weights" in zw else None)
    if rw is not None:  # the reweighted posterior, as in `ppc_panel`
        lo2, lo1, hi1, hi2 = np.clip(np.percentile(rw[0][:, o], [2.5, 16, 84, 97.5], axis=0), 0.0, None)
        reach.append(hi1)
        for lo, hi, ls, lab in ((lo2, hi2, ":", "reweighted posterior: replicated data, 95 %"), (lo1, hi1, "-", "68 %")):
            ax.plot(lo * cs, lo * sn, color=ORANGE, lw=1.0, ls=ls, zorder=3, label=lab)
            ax.plot(hi * cs, hi * sn, color=ORANGE, lw=1.0, ls=ls, zorder=3)
        mm = np.median(rw[1][:, o], axis=0)
        ax.plot(mm * cs, mm * sn, color=ORANGE, lw=1.6, ls="--", zorder=3, label="median model track")
        if accel and "acc" in zw:
            arrows.append((mm * cs, mm * sn, np.median(zw["acc"][iw][rw[2]][:, o], axis=0), ORANGE))
    if c.any():
        ap = np.deg2rad(t["angle0"] + t["phi"][0] + (z["l"][i][c] - 0.5) * bin_deg)
        ax.scatter(x0[c] * np.cos(ap), x0[c] * np.sin(ap), s=5, color="#2f855a", alpha=0.35, lw=0, zorder=4, label="progenitor of each draw")
    extra_x, extra_y = [0.0], [0.0]  # the host, and the true progenitor of a mock: always inside the frame
    if "m" in t:
        ax.plot(t["m"][o] * cs, t["m"][o] * sn, color=TRUTH, lw=1.4, ls="--", zorder=4, label="true model track")
        if accel and "acc_true" in z:
            arrows.append((t["m"][o] * cs, t["m"][o] * sn, z["acc_true"][i][o], TRUTH))
        if "x0_true" in t and "l" in t:
            ap = np.deg2rad(t["angle0"] + t["phi"][0] + (t["l"] - 0.5) * bin_deg)
            extra_x, extra_y = [0.0, t["x0_true"] * np.cos(ap)], [0.0, t["x0_true"] * np.sin(ap)]
            ax.plot(extra_x[1], extra_y[1], "*", ms=13, color=TRUTH, mec="white", mew=0.8, zorder=6, label="true progenitor")
    ax.plot([(r - e) * cs, (r + e) * cs], [(r - e) * sn, (r + e) * sn], color="k", lw=0.8, zorder=5)
    ax.plot(r * cs, r * sn, "o", ms=3, color="k", zorder=5, label="data")
    ax.plot(r[0] * cs[0], r[0] * sn[0], "o", ms=7, mfc="none", color="k", zorder=5, label="first bin")
    ax.plot(0, 0, "+", ms=9, color="#7b8794", zorder=5, label="host")
    px, py = np.r_[np.concatenate([q * cs for q in reach]), extra_x], np.r_[np.concatenate([q * sn for q in reach]), extra_y]
    half = 0.58 * max(np.ptp(px), np.ptp(py))
    ax.set_xlim(0.5 * (px.min() + px.max()) - half, 0.5 * (px.min() + px.max()) + half)
    ax.set_ylim(0.5 * (py.min() + py.max()) - half, 0.5 * (py.min() + py.max()) + half)
    ax.set_aspect("equal")
    if arrows:
        acceleration_arrows(ax, arrows, half, fontsize)
        ax.plot([], [], ls="none", marker=r"$\rightarrow$", ms=11, color="#52606d", label="acceleration of the host at the bins, on the sky\n(median; colours as the model tracks)")
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


def importance_table(rows: list[dict], n: int, file, title: str) -> None:
    """The importance-sampling numbers of `rows` (those of `predictive`, one per track) as a Markdown table in
    `file`: the n draws of the network's posterior that were weighted, how many of them cover the track, the
    effective sample size n_eff = (sum w)^2 / sum w^2 and the sample efficiency n_eff / n, in per cent."""
    done = [r for r in rows if "importance_efficiency" in r]
    eff, ess = (np.array([r[k] for r in done]) for k in ("importance_efficiency", "effective_samples"))
    lines = [f"# Importance sampling: {title}", "",
             "Every track has n draws theta_i ~ q(theta | d) from the network (the proposal), each with the weight",
             "w_i = p(d | theta_i) p(theta_i) / q(theta_i | d): the likelihood and the prior of the paper over the network's",
             "density.", "",
             "- n_eff = (sum_i w_i)^2 / sum_i w_i^2, the effective number of samples",
             "- efficiency = 100 n_eff / n, in per cent; 100 % for a proposal equal to the posterior",
             "- covering: draws whose simulated stream reaches every observed bin. The others have zero likelihood",
             "  (the first guard of the paper), so w_i = 0, and they count in n", "",
             "| track | observed bins | n | covering | n_eff | efficiency [%] |", "|---|---:|---:|---:|---:|---:|"]  # fmt: skip
    for r in rows:
        bins = r.get("points", "")  # the observed tracks carry it; the rows of mock tracks do not
        if "importance_efficiency" in r:
            e, s = r["importance_efficiency"], r["effective_samples"]
            lines.append(f"| {r['track']} | {bins} | {n} | {round(r['covering'] * n)} | {s:.1f} | {100 * e:.2f} |")
        else:  # no draw covers the track, or log q failed (`importance_error`)
            lines.append(f"| {r['track']} | {bins} | {n} | {round(r['covering'] * n)} | - | - |")
    if done:
        lines += [f"| **median of {len(done)} tracks** | | | | {np.median(ess):.1f} | {100 * np.median(eff):.2f} |", "",
                  f"Tracks with n_eff of at least 10: {int((ess >= 10).sum())} of {len(rows)}; of at least 100: {int((ess >= 100).sum())}. "
                  f"Lowest n_eff {ess.min():.1f} ({done[int(ess.argmin())]['track']}), highest {ess.max():.1f} ({done[int(ess.argmax())]['track']})."]  # fmt: skip
    file.write_text("\n".join(lines) + "\n")


def predictive(est, tracks: list[dict], zeta: np.ndarray, prior, grid, sc: dict, cfg: Config, out, angle_pages: bool = True, truth=None) -> tuple[list[dict], dict]:
    """Steps 2-5 for `tracks` (network inputs, r, sig, mask, phi, S, title; mock tracks also carry the true model
    track `m`) and their posterior draws `zeta` (tracks, >= ppc_draws, 15). Writes weights.npz and, unless
    `angle_pages` is off, ppc/ppc_p*.png (radius against angle) into `out`; returns one row of numbers per track, and the arrays behind them (replicated data `rep`, model
    tracks `m`, covering draws `cover`, importance `weights`, and the rest of `replicate`; all (tracks, ppc_draws,
    ...)) for other figures, with the `accelerations` at the bins. truth (tracks, 15): the true parameters of
    mock tracks; they are simulated with the draws, for the true acceleration."""
    M, T, K = cfg.ppc_draws, cfg.span_max, len(tracks)
    zeta = zeta[:, :M]
    par = prior.from_latent(zeta)
    theta, dissolved = to_simulator(par[..., :14]).reshape(-1, 13), np.repeat([t["cond"][0] > 0.5 for t in tracks], M)
    if truth is not None:  # after the draws, one simulation per track
        theta, dissolved = np.concatenate([theta, to_simulator(truth[:, :14])]), np.concatenate([dissolved, [t["cond"][0] > 0.5 for t in tracks]])
    sim = more_particles(simulate(theta, dissolved, sc, cfg, out / "simulations"), theta, dissolved, sc, cfg, grid.i0, out)
    sim, sim_true = {k: v[: K * M] for k, v in sim.items()}, ({k: v[K * M :] for k, v in sim.items()} if truth is not None else None)
    z = replicate(sim, par, tracks, grid, cfg)
    z |= accelerations(z, par, tracks, grid, sc["halo_flattening"], sim_true, truth)
    m, hw, n, var, cover, rep = (z[k] for k in ("m", "hw", "n", "var", "cover", "rep"))
    r, mask = (np.stack([t[k] for t in tracks])[:, None] for k in ("r", "mask"))
    chi2 = lambda d: np.sum(np.where(mask, (d - m) ** 2 / var, 0.0), axis=-1)
    worse = chi2(rep) >= chi2(r)

    rows, weights = [], np.full((K, M), np.nan)
    for i, t in enumerate(tracks):
        c = cover[i]
        row = dict(track=t["title"], covering=float(c.mean()), p_value=float(worse[i][c].mean()) if c.any() else float("nan"))
        if cfg.replace_tracks:  # share of the draws whose track is the one with more particles
            row["more_particles"] = float(z["more"][i].mean())
        if cfg.importance and c.any():
            try:  # log q: exact for a coupling flow, an ODE solve for diffusion / flow matching
                log_q = est.log_prob(dict(zeta=zeta[i]) | {k: np.repeat(t[k][None], M, axis=0) for k in INPUTS})
                ll = np.asarray(log_likelihood(t["r"], t["sig"], t["mask"], m[i], hw[i], n[i], par[i, :, I_SIG, None], z["n_min"][i][:, None], cfg.var_ratio))
                iw = importance_weights(zeta[i][:, t["fixed_target_mask"] > 0], log_q, np.where(c, ll, -np.inf))  # prior of the inferred ones only
                weights[i] = iw["weights"]
                row |= dict(importance_efficiency=iw["efficiency"], effective_samples=iw["ess"], log_evidence=iw["log_evidence"])
            except Exception as e:  # noqa: BLE001
                row["importance_error"] = f"{type(e).__name__}: {e}"[:200]
        rows.append(row)
    np.savez(out / "weights.npz", names=[t["title"] for t in tracks], zeta=zeta, params=par, weights=weights, covering=cover)

    for page in range(-(-K // 8) if angle_pages else 0):  # 8 tracks per page
        part = range(8 * page, min(K, 8 * page + 8))
        fig, axes = plt.subplots(2, 4, figsize=(17, 7.6), constrained_layout=True)
        for k, (ax, i) in enumerate(zip(axes.ravel(), part)):
            ppc_panel(ax, tracks[i], rep[i], m[i], cover[i], rows[i], legend=k == 0, w=weights[i], var=var[i])
        for ax in axes.ravel()[len(part) :]:
            ax.set_axis_off()
        fig.savefig(figures(out, "ppc") / f"ppc_p{page + 1}.png", dpi=120)
        plt.close(fig)
    return rows, z | dict(weights=weights)
