"""Evaluate a trained run on mock tracks of held-out streams. Written to <run>/evaluation/.

    uv run python scripts/extragalactic/sbi/evaluation.py --config outputs/extragalactic/C26_test/npe_v2.yaml

Figures, one subfolder per kind:

  diagnostics/   all `n_test` mock tracks
    loss.png                 training and held-out loss per epoch, the saved epoch marked
    recovery.png             posterior median against the truth
    calibration_ecdf.png     simulation-based calibration: rank of the truth among the posterior draws
    z_score_contraction.png  posterior z-score against posterior contraction
  by_length/     the same split by track length
    by_length.png            recovery, contraction and calibration per parameter, the three length classes side by side
    recovery_<class>.png, calibration_ecdf_<class>.png, z_score_contraction_<class>.png
                             the three diagnostics for the mock tracks of each class: short, medium, long
  sky/           sky_p*.png: posterior predictive check on `ppc_tracks` mock tracks, 8 per page, with
                 importance-sampling efficiency (predictive.py); posterior draws are simulated again with
                 generate.py. Drawn in the plane of the sky, x and y in kpc with the host at the origin, in the
                 frame of the simulation: the true progenitor is on the +x axis. With the true model track and
                 the true progenitor
  corner/        corner_mock_track_<i>.png: corner plot of the posterior of each of those tracks (corner.py),
                 true values marked, that predictive check in its upper right corner. The arrows on it are
                 the acceleration of the host at the bins of the track, projected on the sky: posterior
                 median (blue), reweighted posterior (orange) and the truth (purple)

Files, at the top of the directory:

  metrics.json     the numbers behind the figures
  test_tracks.npz  truth, length and number of observed bins of every mock track
  draws.npz, weights.npz, simulations/   posterior draws, their importance weights and their simulations
                   (predictive.py); kept and reused by a second call

Settings (config.py): n_test, n_samples, length_edges, ppc_tracks, ppc_draws, importance, sim_device, sim_integrator, sim_batch,
sim_workers, sim_cores, prog_given.
"""

from __future__ import annotations

import json
import os
import time

from config import parse, trained

cfg = parse()  # selects the device: must come before jax / keras are imported
RUN = trained(cfg)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # leave room on the card for the simulations

import bayesflow as bf  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402

import networks  # noqa: E402,F401  registers the custom summary network for loading
from corner import physical18, track_corners  # noqa: E402
from data import I_PROG, I_X0, PARAM_NAMES, load_simulations, orientation, split  # noqa: E402
from mock import fixed_observations  # noqa: E402
from predictive import draws, figures, model_saved, plt, predictive, sky_pages  # noqa: E402

OUT = RUN / ("evaluation" + (f"_{cfg.sim_integrator}" if cfg.sim_integrator else ""))


def loss_curve() -> dict:
    """history.csv -> diagnostics/loss.png. For a coupling flow the loss is in nats and the dashed line is a network that
    learnt nothing (the entropy of the prior over the inferred parameters)."""
    h = np.atleast_2d(np.genfromtxt(RUN / "history.csv", delimiter=",", skip_header=1))
    best = int(np.flatnonzero(h[:, 4])[-1])  # the last epoch that was saved
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
    fig.savefig(figures(OUT, "diagnostics") / "loss.png", dpi=130)
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


def classical(zeta, truth, prior, folder: str = "diagnostics", tag: str = "") -> dict:
    """Recovery, calibration and contraction, with BayesFlow's diagnostics. The figures go to the subfolder
    `folder`, with `tag` in their file names."""
    E, names = physical(zeta, prior)
    T, _ = physical(truth, prior)
    d = bf.diagnostics
    for f, file, kw in ((d.plots.recovery, "recovery", {}), (d.plots.calibration_ecdf, "calibration_ecdf", dict(difference=True)),
                        (d.plots.z_score_contraction, "z_score_contraction", {})):  # fmt: skip
        fig = f(E, T, variable_names=names, **kw)
        fig.savefig(figures(OUT, folder) / f"{file}{tag}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    table = {f.__name__: f(E, T)["values"] for f in (d.metrics.root_mean_squared_error, d.metrics.calibration_error, d.metrics.posterior_contraction)}
    med = np.median(E, axis=1)  # the correlation the recovery figure quotes: posterior median against the truth
    table["recovery_correlation"] = [float(np.corrcoef(med[:, i], T[:, i])[0, 1]) for i in range(len(names))]
    return dict(n_tracks=len(T), n_samples=zeta.shape[1], progenitor_given=cfg.prog_given,
                per_parameter={n: {k: float(v[i]) for k, v in table.items()} for i, n in enumerate(names)})  # fmt: skip


LENGTH_COLOURS = {"short": "#6baed6", "medium": "#2171b5", "long": "#08306b"}  # one hue, light to dark with the length
LENGTH_MARKERS = {"short": "o", "medium": "s", "long": "D"}


def length_classes(S) -> dict:
    """Track length (bins from the first to the last, gaps included) -> a selection per class."""
    a, b = (int(x) for x in cfg.length_edges)
    return {"short": (S <= a, f"short: {cfg.span_min}-{a} bins"), "medium": ((S > a) & (S <= b), f"medium: {a + 1}-{b} bins"),
            "long": (S > b, f"long: {b + 1}-{cfg.span_max} bins")}  # fmt: skip


def by_length(zeta, test: dict, prior, overall: dict) -> dict:
    """The classical diagnostics for short, medium and long tracks separately, and by_length.png: per parameter,
    the three classes side by side. All in the subfolder by_length/. A class with fewer than 30 tracks is counted but not evaluated."""
    out = {}
    for key, (sel, label) in length_classes(test["S"]).items():
        out[key] = dict(label=label, n_tracks=int(sel.sum()), mean_observed_bins=float(test["mask"][sel].sum(1).mean()) if sel.any() else 0.0)
        if sel.sum() >= 30:
            out[key] |= classical(zeta[sel], test["zeta"][sel], prior, folder="by_length", tag=f"_{key}")
    done = [k for k in out if "per_parameter" in out[k]]
    names = list(overall["per_parameter"])
    rows = (("recovery_correlation", "correlation of the posterior median with the truth", (-0.1, 1.02)),
            ("posterior_contraction", "posterior contraction (1 - posterior variance / prior variance)", (-0.05, 1.02)),
            ("calibration_error", "calibration error", None))  # fmt: skip
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    x = np.arange(len(names))
    for ax, (metric, title, ylim) in zip(axes, rows):
        ax.plot(x, [overall["per_parameter"][n][metric] for n in names], ls="none", marker="_", ms=22, mew=1.5, color="#7b8794", label=f"all {overall['n_tracks']} tracks")
        for j, key in enumerate(done):
            v = np.array([out[key]["per_parameter"][n][metric] for n in names])
            xo = x + (j - (len(done) - 1) / 2) * 0.22
            if metric == "recovery_correlation":  # 68 % interval of a correlation from n tracks (Fisher)
                z, se = np.arctanh(np.clip(v, -0.999, 0.999)), 1 / np.sqrt(out[key]["n_tracks"] - 3)
                ax.vlines(xo, np.tanh(z - se), np.tanh(z + se), color=LENGTH_COLOURS[key], lw=1.2)
            ax.plot(xo, v, ls="none", marker=LENGTH_MARKERS[key], ms=7, mec="white", mew=0.8, color=LENGTH_COLOURS[key],
                    label=f"{out[key]['label']} ({out[key]['n_tracks']} tracks)")  # fmt: skip
        ax.set_title(title, loc="left", fontsize=10, color="#1f2933")
        ax.set_ylim(ylim) if ylim else ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e4e7eb", lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors="#52606d", labelsize=9)
        for xv in x[:-1] + 0.5:
            ax.axvline(xv, color="#f0f2f5", lw=0.6)
    axes[0].legend(frameon=False, fontsize=9, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.12), borderaxespad=0)
    axes[-1].set_xticks(x, names, fontsize=10)
    axes[-1].set_xlim(-0.6, len(names) - 0.4)
    fig.savefig(figures(OUT, "by_length") / "by_length.png", dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    if cfg.prog_given and cfg.p_known == 0:
        raise SystemExit("prog_given needs a run trained with p_known > 0: this network never saw a given position")
    if not cfg.prog_given and cfg.p_known >= 1:
        raise SystemExit("this run was trained with p_known = 1: set prog_given: true, the network never inferred the position")
    OUT.mkdir(exist_ok=True)
    saved = model_saved(cfg)  # before loading: a run that is still training may rewrite the file
    est = keras.saving.load_model(str(RUN / "model.keras"))
    sims = load_simulations(cfg)
    prior, grid = sims["prior"], sims["grid"]
    held_out = split(sims, cfg.val_frac)[1]
    test = with_progenitor(fixed_observations(held_out, cfg, grid, sims["design"], cfg.seed + 2, cfg.n_test, unweighted=True))
    tracks = [{k: v[i] for k, v in test.items()} | dict(title=f"mock track {i}") for i in range(cfg.n_test)]
    zeta = draws(est, tracks, cfg.n_samples, cfg, OUT)
    np.savez(OUT / "test_tracks.npz", zeta=test["zeta"], S=test["S"], n_observed=test["mask"].sum(1))
    metrics = dict(model_saved=time.strftime("%F %T", time.localtime(saved / 1e9)), loss=loss_curve(), classical=classical(zeta, test["zeta"], prior))
    metrics["by_length"] = by_length(zeta, test, prior, metrics["classical"])
    if cfg.ppc_tracks > 0:
        k = cfg.ppc_tracks
        true = prior.from_latent(test["zeta"][:k])
        # a mock track has no orientation on the sky: it is shown in the frame of its simulation, true progenitor on the +x axis
        shown = [t | dict(angle0=-(float(t["phi"][0]) + (int(t["l"]) - 0.5) * grid.bin_deg), x0_true=float(p[I_X0])) for t, p in zip(tracks[:k], true)]
        metrics["ppc"], sim = predictive(est, shown, zeta[:k], prior, grid, sims["sim_config"], cfg, OUT, angle_pages=False, truth=true)
        par = prior.from_latent(zeta[:k, : cfg.ppc_draws])
        sky_pages(shown, sim, par, metrics["ppc"], grid, figures(OUT, "sky"))
        prior_phys = physical18(prior.from_latent(np.random.default_rng(cfg.seed).standard_normal((20000, len(PARAM_NAMES)))))
        track_corners(shown, metrics["ppc"], physical18(prior.from_latent(zeta[:k])), prior_phys, sim, par, grid.bin_deg, figures(OUT, "corner"), truth=physical18(true))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"wrote {OUT}: " + ", ".join(sorted(p.name + ("/" if p.is_dir() else "") for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
