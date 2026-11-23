"""Evaluate a trained run on mock tracks of held-out streams. Written to <run>/evaluation/.

    uv run python scripts/extragalactic/sbi/evaluation.py --config outputs/extragalactic/C26_test/npe_v2.yaml

  loss.png                 training and held-out loss per epoch, the saved epoch marked
  recovery.png             posterior median against the truth
  calibration_ecdf.png     simulation-based calibration: rank of the truth among the posterior draws
  z_score_contraction.png  posterior z-score against posterior contraction
  ppc_p*.png               posterior predictive check on `ppc_tracks` mock tracks, with importance-sampling
                           efficiency (predictive.py); posterior draws are simulated again with generate.py
  metrics.json             the numbers behind the figures

Settings (config.py): n_test, n_samples, ppc_tracks, ppc_draws, importance, sim_workers, sim_cores, prog_given.
"""

from __future__ import annotations

import json

from config import parse, trained

cfg = parse()  # selects the device: must come before jax / keras are imported
RUN = trained(cfg)

import bayesflow as bf  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402

import networks  # noqa: E402,F401  registers the custom summary network for loading
from data import I_PROG, PARAM_NAMES, load_simulations, orientation, split  # noqa: E402
from mock import fixed_observations  # noqa: E402
from predictive import draws, plt, predictive  # noqa: E402

OUT = RUN / "evaluation"


def loss_curve() -> dict:
    """history.csv -> loss.png. For a coupling flow the loss is in nats and the dashed line is a network that
    learnt nothing (the entropy of the prior over the inferred parameters)."""
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
    return dict(epochs=len(h), best_epoch=int(h[best, 0]), best_val_loss=float(h[best, 2]), train_loss_there=float(h[best, 1]),
                seconds_per_epoch=float(np.median(h[:, 3])))  # fmt: skip


def with_progenitor(obs: dict) -> dict:
    """Mark the progenitor position of mock tracks as given (at its true value) or as inferred."""
    fixed, cond = np.ones_like(obs["zeta"]), obs["cond"].copy()
    fixed[:, I_PROG], cond[:, 1] = (0.0, 1.0) if cfg.prog_given else (1.0, 0.0)  # the mask and the "position given" flag
    return obs | dict(cond=cond, fixed_target_mask=fixed, fixed_target_value=obs["zeta"] * (1 - fixed))


def physical(zeta, prior) -> tuple[np.ndarray, list]:
    """Latent -> the parameters with q appended; the progenitor position is dropped when it was given."""
    p = prior.from_latent(zeta)
    p = np.concatenate([p, orientation(p[..., 2:5])[..., :1]], axis=-1)
    keep = [i for i, n in enumerate(PARAM_NAMES + ["q"]) if not (cfg.prog_given and n == "prog")]
    return p[..., keep], [(PARAM_NAMES + ["q"])[i] for i in keep]


def classical(zeta, test: dict, prior) -> dict:
    """Recovery, calibration and contraction, with BayesFlow's diagnostics."""
    E, names = physical(zeta, prior)
    T, _ = physical(test["zeta"], prior)
    d = bf.diagnostics
    for f, file, kw in ((d.plots.recovery, "recovery", {}), (d.plots.calibration_ecdf, "calibration_ecdf", dict(difference=True)),
                        (d.plots.z_score_contraction, "z_score_contraction", {})):  # fmt: skip
        fig = f(E, T, variable_names=names, **kw)
        fig.savefig(OUT / f"{file}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    table = {f.__name__: f(E, T)["values"] for f in (d.metrics.root_mean_squared_error, d.metrics.calibration_error, d.metrics.posterior_contraction)}
    return dict(n_tracks=len(T), n_samples=zeta.shape[1], progenitor_given=cfg.prog_given,
                per_parameter={n: {k: float(v[i]) for k, v in table.items()} for i, n in enumerate(names)})  # fmt: skip


def main() -> None:
    if cfg.prog_given and cfg.p_known == 0:
        raise SystemExit("prog_given needs a run trained with p_known > 0: this network never saw a given position")
    if not cfg.prog_given and cfg.p_known >= 1:
        raise SystemExit("this run was trained with p_known = 1: set prog_given: true, the network never inferred the position")
    OUT.mkdir(exist_ok=True)
    est = keras.saving.load_model(str(RUN / "model.keras"))
    sims = load_simulations(cfg)
    prior, grid = sims["prior"], sims["grid"]
    held_out = split(sims, cfg.val_frac)[1]
    test = with_progenitor(fixed_observations(held_out, cfg, grid, sims["design"], cfg.seed + 2, cfg.n_test, unweighted=True))
    tracks = [{k: v[i] for k, v in test.items()} | dict(title=f"mock track {i}") for i in range(cfg.n_test)]
    zeta = draws(est, tracks, cfg.n_samples, cfg, OUT)
    metrics = dict(loss=loss_curve(), classical=classical(zeta, test, prior))
    if cfg.ppc_tracks > 0:
        k = cfg.ppc_tracks
        metrics["ppc"] = predictive(est, tracks[:k], zeta[:k], prior, grid, sims["sim_config"], cfg, OUT)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"wrote {OUT}: " + ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
