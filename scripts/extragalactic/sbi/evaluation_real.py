"""Posteriors, posterior predictive checks and importance weights for the observed STRRINGS tracks.
Written to <run>/evaluation_real/.

    uv run python scripts/extragalactic/sbi/evaluation_real.py --config outputs/extragalactic/C26_test/npe_v2.yaml
    ... --real-tracks NGC5907,NGC1162 --ppc-draws 256        # two tracks, more draws for the importance weights

  posterior.npz    n_samples posterior draws per track: `zeta` (latent), `params` (the 15 parameters), `q`,
                   `q_cosi_psi`, track `names`, `param_names`
  q.png            halo flattening per track: median and 16-84 % of the network's posterior and, if importance
                   weights were computed, of the weighted draws
  ppc_p*.png       posterior predictive check per track (predictive.py): `ppc_draws` posterior draws are
                   simulated again with generate.py and compared with the observed track
  weights.npz      the simulated draws, their importance weights and which of them cover the track
  metrics.json     per track: q percentiles, share of covering draws, predictive p-value, importance-sampling
                   efficiency, effective sample size and log evidence

The importance weights use the likelihood of the paper (likelihood.py: eq. 4 with both guards) and its prior:
w = prior x likelihood / q. They need log q, which a coupling flow gives exactly. A track is treated as the paper
treats a stream without a visible progenitor: mass ramped to zero, and the progenitor position inferred
(`prog_given: true` fixes it at the centre of the track instead, for a run trained with p_known > 0).
"""

from __future__ import annotations

import json

from config import parse, trained

cfg = parse()  # selects the device: must come before jax / keras are imported
RUN = trained(cfg)

import keras  # noqa: E402
import numpy as np  # noqa: E402

import networks  # noqa: E402,F401  registers the custom summary network for loading
from data import PARAM_NAMES, load_observed, load_simulations, orientation  # noqa: E402
from mock import encode_observation  # noqa: E402
from predictive import draws, plt, predictive  # noqa: E402

OUT = RUN / "evaluation_real"


def weighted_percentiles(x, w, q=(16, 50, 84)) -> list:
    order = np.argsort(x)
    return np.interp(np.array(q) / 100, np.cumsum(w[order]) / w.sum(), x[order]).tolist()


def main() -> None:
    if cfg.prog_given and cfg.p_known == 0:
        raise SystemExit("prog_given needs a run trained with p_known > 0: this network never saw a given position")
    if not cfg.prog_given and cfg.p_known >= 1:
        raise SystemExit("this run was trained with p_known = 1: set prog_given: true, the network never inferred the position")
    OUT.mkdir(exist_ok=True)
    est = keras.saving.load_model(str(RUN / "model.keras"))
    sims = load_simulations(cfg)
    prior, grid = sims["prior"], sims["grid"]
    wanted = [s for s in cfg.real_tracks.split(",") if s]
    observed = [o for o in load_observed(cfg) if not wanted or o["name"] in wanted or o["name"].split("_")[0] in wanted]
    if not observed:
        raise SystemExit(f"no observed track matches {wanted}")
    tracks = [encode_observation(o["theta_deg"], o["r"], o["r_err"], cfg, grid, dissolved=True, prog=0.5 if cfg.prog_given else None)
              | dict(title=o["name"]) for o in observed]  # fmt: skip

    zeta = draws(est, tracks, cfg.n_samples, cfg, OUT)
    par = prior.from_latent(zeta)
    o3 = orientation(par[..., 2:5])
    np.savez(OUT / "posterior.npz", names=[t["title"] for t in tracks], param_names=PARAM_NAMES, zeta=zeta.astype(np.float32),
             params=par.astype(np.float32), q=o3[..., 0].astype(np.float32), q_cosi_psi=o3.astype(np.float32))  # fmt: skip
    rows = [dict(track=t["title"], points=int(t["mask"].sum()), q=np.percentile(q, [16, 50, 84]).tolist()) for t, q in zip(tracks, o3[..., 0])]

    if cfg.ppc_draws > 0:
        for row, extra in zip(rows, predictive(est, tracks, zeta, prior, grid, sims["sim_config"], cfg, OUT)):
            row |= extra
        w = np.load(OUT / "weights.npz")["weights"]
        for row, wi, q in zip(rows, w, o3[:, : cfg.ppc_draws, 0]):
            if np.isfinite(wi).all() and wi.sum() > 0:
                row["q_weighted"] = weighted_percentiles(q, wi)

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
    (OUT / "metrics.json").write_text(json.dumps(dict(progenitor_given=cfg.prog_given, n_samples=cfg.n_samples, ppc_draws=cfg.ppc_draws, tracks=rows), indent=1))
    print(f"wrote {OUT}: " + ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
