"""Posteriors, posterior predictive checks and importance weights for the observed STRRINGS tracks.
Written to <run>/evaluation_real/.

    uv run python scripts/extragalactic/sbi/evaluation_real.py --config outputs/extragalactic/C26_test/npe_v2.yaml
    ... --real-tracks NGC5907,NGC1162 --ppc-draws 256        # two tracks, more draws for the importance weights
    ... --real-tracks NGC1162 --n-samples 20000 --ppc-draws 20000 --eval-tag NGC1162_20000
                                                             # one track, many draws, in <run>/evaluation_real_NGC1162_20000/

Figures:

  q.png            halo flattening per track: median and 16-84 % of the network's posterior and, if importance
                   weights were computed, of the weighted draws
  ppc/             ppc_p*.png: posterior predictive check per track, 8 per page (predictive.py): `ppc_draws`
                   posterior draws are simulated again with generate.py (on GPUs unless `sim_device: cpu`) and
                   compared with the observed track, radius against angle. Blue: the network's posterior;
                   orange: the same draws importance weighted
  sky/             sky_p*.png: the same check in the plane of the track files (x, y in kpc, host at the origin),
                   with the progenitor position of every covering draw. sky.py redraws it from the files of a
                   finished evaluation, without simulating
  corner/          corner_<track>.png: corner plot of the posterior (corner.py): regions of the network's draws
                   and of the importance-weighted simulated draws, and the prior, with the predictive check in
                   the plane of the sky (as in sky_p*.png) in the upper right corner. `ppc_draws` = `n_samples`
                   weights every draw. The arrows on that check are the acceleration of the host at the bins
                   of the track, projected on the sky: the median over the draws, each at the place of its own
                   simulated stream (blue: network, orange: reweighted). sky.py redraws these too

Files, at the top of the directory:

  posterior.npz    n_samples posterior draws per track: `zeta` (latent), `params` (the 15 parameters), `q`,
                   `q_cosi_psi`, track `names`, `param_names`
  weights.npz      the simulated draws, their importance weights and which of them cover the track
  importance.md    table of the importance sampling per track: draws n, effective sample size
                   n_eff = (sum w)^2 / sum w^2 and sample efficiency n_eff / n in per cent
  metrics.json     per track: q percentiles, share of covering draws, predictive p-value, importance-sampling
                   efficiency, effective sample size and log evidence
  draws.npz, simulations/   the posterior draws and their simulations (predictive.py); kept and reused by a
                   second call

The importance weights use the likelihood of the paper (likelihood.py: eq. 4 with both guards) and its prior:
w = prior x likelihood / q. They need log q, which a coupling flow gives exactly. A track is treated as the paper
treats a stream without a visible progenitor: mass ramped to zero, and the progenitor position inferred
(`prog_given: true` fixes it at the centre of the track instead, for a run trained with p_known > 0).
"""

from __future__ import annotations

import dataclasses
import json
import os
import time

from config import parse, trained

cfg = parse()  # selects the device: must come before jax / keras are imported
RUN = trained(cfg)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # leave room on the card for the simulations

import keras  # noqa: E402
import numpy as np  # noqa: E402

import networks  # noqa: E402,F401  registers the custom summary network for loading
from data import PARAM_NAMES, load_observed, load_simulations, orientation  # noqa: E402
from mock import encode_observation  # noqa: E402
from corner import physical18, track_corners  # noqa: E402
from predictive import draws, figures, importance_table, model_saved, plt, predictive, sky_pages  # noqa: E402

OUT = RUN / ("evaluation_real" + (f"_{cfg.sim_integrator}" if cfg.sim_integrator else "") + (f"_{cfg.eval_tag}" if cfg.eval_tag else ""))


def weighted_percentiles(x, w, q=(16, 50, 84)) -> list:
    order = np.argsort(x)
    return np.interp(np.array(q) / 100, np.cumsum(w[order]) / w.sum(), x[order]).tolist()


def main() -> None:
    if cfg.prog_given and cfg.p_known == 0:
        raise SystemExit("prog_given needs a run trained with p_known > 0: this network never saw a given position")
    if not cfg.prog_given and cfg.p_known >= 1:
        raise SystemExit("this run was trained with p_known = 1: set prog_given: true, the network never inferred the position")
    if cfg.noise_scale != 1:
        raise SystemExit(f"this run was trained with noise_scale {cfg.noise_scale}: its posteriors are not those of tracks with the observed errors")
    OUT.mkdir(exist_ok=True)
    saved = model_saved(cfg)  # before loading: a run that is still training may rewrite the file
    est = keras.saving.load_model(str(RUN / "model.keras"))
    # prior, grid and the configuration of the simulator are those stored in the first shard: the streams
    # themselves (85 GB of host memory for train_v2) are not needed here
    sims = load_simulations(dataclasses.replace(cfg, max_shards=1))
    prior, grid = sims["prior"], sims["grid"]
    wanted = [s for s in cfg.real_tracks.split(",") if s]
    observed = [o for o in load_observed(cfg) if not wanted or o["name"] in wanted or o["name"].split("_")[0] in wanted]
    if not observed:
        raise SystemExit(f"no observed track matches {wanted}")
    tracks = [encode_observation(o["theta_deg"], o["r"], o["r_err"], cfg, grid, dissolved=True, prog=0.5 if cfg.prog_given else None)
              | dict(title=o["name"], angle0=float(np.median(o["theta_deg"]))) for o in observed]  # fmt: skip

    zeta = draws(est, tracks, cfg.n_samples, cfg, OUT)
    par = prior.from_latent(zeta)
    o3 = orientation(par[..., 2:5])
    np.savez(OUT / "posterior.npz", names=[t["title"] for t in tracks], param_names=PARAM_NAMES, zeta=zeta.astype(np.float32),
             params=par.astype(np.float32), q=o3[..., 0].astype(np.float32), q_cosi_psi=o3.astype(np.float32))  # fmt: skip
    rows = [dict(track=t["title"], points=int(t["mask"].sum()), q=np.percentile(q, [16, 50, 84]).tolist()) for t, q in zip(tracks, o3[..., 0])]

    sim = None
    if cfg.ppc_draws > 0:
        extra, sim = predictive(est, tracks, zeta, prior, grid, sims["sim_config"], cfg, OUT)
        for row, e in zip(rows, extra):
            row |= e
        sky_pages(tracks, sim, par[:, : cfg.ppc_draws], rows, grid, figures(OUT, "sky"))
        for row, wi, q in zip(rows, sim["weights"], o3[:, : cfg.ppc_draws, 0]):
            if np.isfinite(wi).all() and wi.sum() > 0:
                row["q_weighted"] = weighted_percentiles(q, wi)

    prior_phys = physical18(prior.from_latent(np.random.default_rng(cfg.seed).standard_normal((20000, len(PARAM_NAMES)))))
    track_corners(tracks, rows, physical18(par), prior_phys, sim, par[:, : cfg.ppc_draws] if sim else None, grid.bin_deg, figures(OUT, "corner"))

    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(5.5, 1.5 + 0.22 * len(rows)), constrained_layout=True)
    lo, mid, hi = np.array([r["q"] for r in rows]).T
    ax.errorbar(mid, y, xerr=[mid - lo, hi - mid], fmt="o", ms=3, lw=1, color="#08306b", label="network posterior")
    have = [i for i, r in enumerate(rows) if "q_weighted" in r]
    if have:
        lo, mid, hi = np.array([rows[i]["q_weighted"] for i in have]).T
        ax.errorbar(mid, y[have] + 0.3, xerr=[mid - lo, hi - mid], fmt="s", ms=3, lw=1, color="#c05621", label="importance weighted")
    ax.axvline(1.0, color="#7b8794", lw=0.8)
    ax.set_yticks(y, [r["track"] for r in rows], fontsize=7)
    ax.set_xlabel("halo flattening q (median, 16-84 %)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "q.png", dpi=130)
    plt.close(fig)
    (OUT / "metrics.json").write_text(json.dumps(dict(model_saved=time.strftime("%F %T", time.localtime(saved / 1e9)), progenitor_given=cfg.prog_given, n_samples=cfg.n_samples, ppc_draws=cfg.ppc_draws, tracks=rows), indent=1))
    if cfg.ppc_draws > 0 and cfg.importance:
        importance_table(rows, cfg.ppc_draws, OUT / "importance.md", f"{cfg.name}, observed tracks")
    print(f"wrote {OUT}: " + ", ".join(sorted(p.name + ("/" if p.is_dir() else "") for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
