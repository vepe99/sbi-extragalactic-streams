"""Evaluate a trained run. Everything is written to <run>/evaluation/.

    uv run python scripts/extragalactic/sbi/evaluation.py --name npe_v1
    uv run python scripts/extragalactic/sbi/evaluation.py --name npe_v1 --what ppc --ppc-workers 16 --ppc-cores 0-15

  loss         training and held-out loss per epoch, the saved epoch marked            loss.png
  recovery     posterior median against the truth on mock tracks of held-out streams   recovery.png
  calibration  simulation-based calibration: rank of the truth among the samples       calibration_ecdf.png
  contraction  posterior z-score against posterior contraction                         z_score_contraction.png
  ppc          posterior predictive check: posterior draws are SIMULATED again with    ppc.png
               generate.py and their tracks, with noise drawn from the likelihood,
               compared with the track they were inferred from; for a coupling flow
               the same simulations give the importance-sampling efficiency
  observed     posteriors of the 32 STRRINGS tracks                                    posterior_observed.npz, observed_q.png

Numbers go to metrics.json. `--prog given` (default) passes the progenitor position to the network, as the
paper does (the true one for mock tracks, the centre of the track for observed ones); `--prog free` infers it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from config import REPO, Config, pick_device

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--out", default=Config.out)
ap.add_argument("--what", default="loss,recovery,calibration,contraction,ppc,observed")
ap.add_argument("--prog", default="given", choices=["given", "free"])
ap.add_argument("--n-test", type=int, default=1000)  # mock tracks of held-out streams
ap.add_argument("--n-samples", type=int, default=500)  # posterior draws per track
ap.add_argument("--ppc-tracks", type=int, default=6)  # mock tracks in the predictive check (plus --ppc-observed)
ap.add_argument("--ppc-observed", default="NGC5907,NGC1162")
ap.add_argument("--ppc-draws", type=int, default=48)  # posterior draws simulated per track
ap.add_argument("--ppc-workers", type=int, default=8)
ap.add_argument("--ppc-cores", default="")  # e.g. 0-15; "" = no pinning
ap.add_argument("--cpu", action="store_true")
args = ap.parse_args()
cfg, stored = Config.load(REPO / args.out / "models" / args.name / "config.yaml")
pick_device(args.cpu)  # before jax / keras are imported

import bayesflow as bf  # noqa: E402
import h5py  # noqa: E402
import keras  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import networks  # noqa: E402,F401  registers the custom summary network for loading
from data import I_PROG, I_SIG, PARAM_NAMES, load_observed, load_simulations, orientation, split, to_simulator  # noqa: E402
from likelihood import importance_weights, log_likelihood, model_at  # noqa: E402
from mock import encode_observation, fixed_observations  # noqa: E402

RUN = cfg.run_dir()
OUT = RUN / "evaluation"
INPUTS = ("tokens", "mask", "cond", "fixed_target_mask", "fixed_target_value")
GIVEN = args.prog == "given"


def sample(est, obs: dict, n: int) -> np.ndarray:
    """Posterior draws in latent coordinates, (tracks, n, 15). A fixed parameter comes back at its given value."""
    return est.sample(num_samples=n, conditions={k: obs[k] for k in INPUTS}, batch_size=64, seed=cfg.seed)["zeta"]


def with_progenitor(obs: dict) -> dict:
    """Mark the progenitor position of mock tracks as given (at its true value) or as inferred."""
    fixed = np.ones_like(obs["zeta"])
    fixed[:, I_PROG] = 0.0 if GIVEN else 1.0
    return obs | dict(fixed_target_mask=fixed, fixed_target_value=obs["zeta"] * (1 - fixed))


def physical(zeta, prior) -> tuple[np.ndarray, list]:
    """Latent -> the parameters with q appended; the progenitor position is dropped when it was given."""
    p = prior.from_latent(zeta)
    p = np.concatenate([p, orientation(p[..., 2:5])[..., :1]], axis=-1)
    names = PARAM_NAMES + ["q"]
    keep = [i for i, n in enumerate(names) if not (GIVEN and n == "prog")]
    return p[..., keep], [names[i] for i in keep]


def loss_curve(metrics: dict) -> None:
    """history.csv -> loss.png. For a coupling flow the loss is in nats and the dashed line is a network that
    learnt nothing (the entropy of the prior over the inferred parameters, between 19.9 and 21.3)."""
    h = np.atleast_2d(np.genfromtxt(RUN / "history.csv", delimiter=",", skip_header=1))
    best = int(np.argmin(h[:, 2]))
    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    ax.plot(h[:, 0], h[:, 1], color="#08306b", lw=1.5, label="training")
    ax.plot(h[:, 0], h[:, 2], color="#c05621", lw=1.5, label="held out")
    ax.plot(h[best, 0], h[best, 2], "o", color="#c05621", ms=6, label=f"saved model (epoch {int(h[best, 0])})")
    if cfg.inference == "flow":
        ax.axhline(0.5 * (1 + np.log(2 * np.pi)) * (15 - cfg.p_known), color="#7b8794", lw=0.8, ls="--", label="nothing learnt")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.savefig(OUT / "loss.png", dpi=130)
    plt.close(fig)
    metrics["loss"] = dict(epochs=len(h), best_epoch=int(h[best, 0]), best_val_loss=float(h[best, 2]), train_loss_there=float(h[best, 1]),
                           seconds_per_epoch=float(np.median(h[:, 3])))  # fmt: skip


def classical(est, test: dict, prior, metrics: dict) -> None:
    """Recovery, calibration and contraction on mock tracks of held-out streams."""
    E, names = physical(sample(est, test, args.n_samples), prior)
    T, _ = physical(test["zeta"], prior)
    plots = dict(recovery=bf.diagnostics.plots.recovery, calibration=bf.diagnostics.plots.calibration_ecdf,
                 contraction=bf.diagnostics.plots.z_score_contraction)  # fmt: skip
    files = dict(recovery="recovery", calibration="calibration_ecdf", contraction="z_score_contraction")
    for key in [k for k in plots if k in args.what]:
        fig = plots[key](E, T, variable_names=names, **(dict(difference=True) if key == "calibration" else {}))
        fig.savefig(OUT / f"{files[key]}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    m = bf.diagnostics.metrics
    table = {f.__name__: f(E, T)["values"] for f in (m.root_mean_squared_error, m.calibration_error, m.posterior_contraction)}
    metrics["classical"] = dict(n_tracks=len(T), n_samples=args.n_samples, progenitor=args.prog,
                                per_parameter={n: {k: float(v[i]) for k, v in table.items()} for i, n in enumerate(names)})  # fmt: skip


def simulate(theta: np.ndarray, dissolved: np.ndarray, sc: dict) -> dict:
    """Run generate.py on given parameters (same physics as the training set) and return the binned tracks."""
    d = OUT / "ppc"
    d.mkdir(parents=True, exist_ok=True)
    n, w = len(theta), max(1, min(args.ppc_workers, len(theta)))
    done = (d / "theta.npz").exists() and np.array_equal(np.load(d / "theta.npz")["theta"], theta) and (d / "data" / "manifest.json").exists()
    physics = [f for k in ("integrator", "solver", "rtol", "max_steps", "n_particles", "n_steps", "halo_flattening")
               for f in (f"--{k.replace('_', '-')}", str(sc[k]))]  # fmt: skip
    cmd = ["uv", "run", "--script", str(REPO / "scripts/extragalactic/generate.py"), "run", "--cpu", "--batch-size", "1", "--overwrite",
           "--cpu-workers", str(w), "--theta-file", str(d / "theta.npz"), "--n-streams", str(n), "--shard-size", str(-(-n // w)),
           "--n-save-particles", "0", "--out", str(d.relative_to(REPO)), "--track-ordering", cfg.ordering,
           "--progenitor-mass-loss", "mixed", *physics] + (["--cpu-cores", args.ppc_cores] if args.ppc_cores else [])  # fmt: skip
    if not done:  # the same draws are not simulated twice
        np.savez(d / "theta.npz", theta=theta.astype(np.float64), dissolved=dissolved.astype(bool))
        with open(d / "generate.log", "w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    cols = {k: [] for k in ("index", "status", "track/count", "track/r_med", "track/r_hw")}
    for p in sorted((d / "data").glob("shard_*.h5")):
        with h5py.File(p, "r") as f:
            for k in cols:
                cols[k].append(f[k][:])
    order = np.argsort(np.concatenate(cols["index"]))
    get = lambda k: np.nan_to_num(np.concatenate(cols[k])[order]).astype(np.float32)
    return dict(count=get("track/count"), r_med=get("track/r_med"), r_hw=get("track/r_hw"), failed=(get("status").astype(int) & 16) > 0)


def ppc(est, tracks: list[dict], prior, grid, sc: dict, metrics: dict) -> None:
    """Posterior predictive check. Each track is a dict with the network inputs, r, sig, mask, S, a title and,
    for mock tracks, the true model track `m`."""
    M, T = args.ppc_draws, cfg.span_max
    obs = {k: np.stack([t[k] for t in tracks]) for k in INPUTS}
    zeta = sample(est, obs, M)  # (tracks, M, 15)
    par = prior.from_latent(zeta)
    sim = simulate(to_simulator(par[..., :14]).reshape(-1, 13), np.repeat([t["cond"][0] > 0.5 for t in tracks], M), sc)
    l = np.floor(par[..., I_PROG] * (np.array([t["S"] for t in tracks])[:, None] + 1)).astype(int)  # bins left of the progenitor
    m, hw, n = (np.asarray(x).reshape(len(tracks), M, T) for x in model_at(sim, grid.i0, l.reshape(-1), T))
    r, sig, mask = (np.stack([t[k] for t in tracks])[:, None] for k in ("r", "sig", "mask"))
    var = sig**2 + par[..., I_SIG, None] ** 2
    cover = np.all(~mask | (n >= cfg.n_min), axis=-1) & ~sim["failed"].reshape(len(tracks), M)  # model reaches every observed bin
    rng = np.random.default_rng(cfg.seed)
    rep = m + np.sqrt(var) * rng.standard_normal(m.shape)  # replicated data: a draw from the likelihood
    chi2 = lambda d: np.sum(np.where(mask, (d - m) ** 2 / var, 0.0), axis=-1)
    p_val = [float(np.mean(chi2(rep)[i][c] >= chi2(r)[i][c])) if c.any() else float("nan") for i, c in enumerate(cover)]

    rows = []
    fig, axes = plt.subplots(2, -(-len(tracks) // 2), figsize=(4.4 * -(-len(tracks) // 2), 6.6), constrained_layout=True, squeeze=False)
    for i, (ax, t) in enumerate(zip(axes.ravel(), tracks)):
        o, c = t["mask"], cover[i]
        row = dict(track=t["title"], covering=float(c.mean()), p_value=p_val[i])
        if c.any():
            lo2, lo1, hi1, hi2 = np.percentile(rep[i][c][:, o], [2.5, 16, 84, 97.5], axis=0)
            ax.fill_between(t["phi"][o], lo2, hi2, color="#9ecae1", lw=0, label="replicated data, 95 %")
            ax.fill_between(t["phi"][o], lo1, hi1, color="#4292c6", lw=0, label="68 %")
            ax.plot(t["phi"][o], np.median(m[i][c][:, o], axis=0), color="#08306b", lw=1.2, label="median model track")
        if "m" in t:
            ax.plot(t["phi"][o], t["m"][o], color="#2f855a", lw=1.2, ls="--", label="true model track")
        ax.errorbar(t["phi"][o], t["r"][o], yerr=t["sig"][o], fmt="o", ms=3, lw=0.8, color="#c05621", label="data")
        if cfg.inference == "flow" and c.any():  # importance weights towards prior x likelihood of the paper
            rep_in = lambda x: np.repeat(x[None], M, axis=0)
            log_q = est.log_prob(dict(zeta=zeta[i]) | {k: rep_in(t[k]) for k in INPUTS})
            ll = np.asarray(log_likelihood(t["r"], t["sig"], t["mask"], m[i], hw[i], n[i], par[i, :, I_SIG, None], cfg.n_min, cfg.var_ratio))
            inferred = t["fixed_target_mask"] > 0  # the prior of a fixed parameter is not part of q
            row["importance_efficiency"] = importance_weights(zeta[i][:, inferred], log_q, np.where(c, ll, -np.inf))["efficiency"]
        ax.set_title(f"{t['title']}\ncovered {c.mean():.0%}, predictive p = {p_val[i]:.2f}", fontsize=8, loc="left")
        ax.set_xlabel("angle minus median angle [deg]")
        ax.set_ylabel("projected radius [kpc]")
        rows.append(row)
    axes.ravel()[0].legend(fontsize=6, frameon=False)
    fig.savefig(OUT / "ppc.png", dpi=130)
    plt.close(fig)
    metrics["ppc"] = dict(draws_per_track=M, progenitor=args.prog, tracks=rows)


def encode_all(observed: list[dict], grid) -> list[dict]:
    """Observed tracks -> network inputs; progenitor at the centre (the paper) or inferred; no visible progenitor."""
    enc = [encode_observation(o["theta_deg"], o["r"], o["r_err"], cfg, grid, dissolved=True, prog=0.5 if GIVEN else None) for o in observed]
    return [e | dict(title=o["name"]) for e, o in zip(enc, observed)]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    est = keras.saving.load_model(str(RUN / "model.keras"))
    sims = load_simulations(cfg)
    prior, grid = sims["prior"], sims["grid"]
    held_out = split(sims, cfg.val_frac)[1]
    test = with_progenitor(fixed_observations(held_out, cfg, grid, sims["design"], cfg.seed + 2, args.n_test, unweighted=True))
    mfile = OUT / "metrics.json"
    metrics = json.loads(mfile.read_text()) if mfile.exists() else {}
    observed = encode_all(load_observed(cfg), grid)

    if "loss" in args.what:
        loss_curve(metrics)
    if any(k in args.what for k in ("recovery", "calibration", "contraction")):
        classical(est, test, prior, metrics)
    if "ppc" in args.what:
        mocks = [{k: v[i] for k, v in test.items()} | dict(title=f"mock track {i}") for i in range(args.ppc_tracks)]
        real = [e for e in observed if e["title"].split("_")[0] in args.ppc_observed.split(",")]
        ppc(est, mocks + real, prior, grid, sims["sim_config"], metrics)
    if "observed" in args.what:
        zeta = sample(est, {k: np.stack([e[k] for e in observed]) for k in INPUTS}, args.n_samples)
        par = prior.from_latent(zeta)
        q = orientation(par[..., 2:5])[..., 0]
        np.savez(RUN / "posterior_observed.npz", names=[e["title"] for e in observed], param_names=PARAM_NAMES, zeta=zeta.astype(np.float32),
                 params=par.astype(np.float32), q=q.astype(np.float32), progenitor=args.prog)  # fmt: skip
        lo, mid, hi = np.percentile(q, [16, 50, 84], axis=1)
        fig, ax = plt.subplots(figsize=(5, 8), constrained_layout=True)
        ax.errorbar(mid, np.arange(len(mid)), xerr=[mid - lo, hi - mid], fmt="o", ms=3, lw=1, color="#08306b")
        ax.axvline(1.0, color="#7b8794", lw=0.8)
        ax.set_yticks(np.arange(len(mid)), [e["title"] for e in observed], fontsize=7)
        ax.set_xlabel("halo flattening q (median, 16-84 %)")
        fig.savefig(OUT / "observed_q.png", dpi=130)
        plt.close(fig)
        metrics["observed"] = dict(progenitor=args.prog, q={e["title"]: [float(a), float(b), float(c)] for e, a, b, c in zip(observed, lo, mid, hi)})
    mfile.write_text(json.dumps(metrics, indent=1))
    print(f"wrote {OUT}: " + ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    sys.exit(main())
