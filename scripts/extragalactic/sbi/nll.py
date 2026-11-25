"""Held-out loss in nats of trained runs, on the same mock tracks for all of them: the number that compares a
coupling flow with a diffusion or flow-matching model, whose training losses are not in nats.

    uv run --extra gpu python scripts/extragalactic/sbi/nll.py --runs npe_a npe_b [--n 20000] [--gpu 3]

For `n` mock tracks of the held-out streams, drawn as the training draws them (the mock settings and the
`de_max` of the first run; the others must have the same mock settings), the loss of a run is -log q(true parameters | track), from `log_prob`:
exact for a coupling flow, the probability-flow ODE for diffusion and flow matching. For a coupling flow its
mean is the `val_loss` of train.log up to the sampling of the mocks. The mocks are those of
scripts/extragalactic/track_noise.py (same seed), which stores the kind of stream behind each.

The mocks come from the held-out streams of the FIRST run's set (its `subdir`, `max_shards`, `de_max`): put first the
run whose training set holds all the others', so that no run has trained on a stream behind a mock.

Written to <run>/heldout_nll.npz (`nll`, and `S`, `n_obs`, `stream`, `dE` of every mock; for the first `n_post`
mocks also the mean and sd of `draws` posterior draws and the truth, in latent coordinates) and printed: mean
loss, its standard error, and the mean per length class.

`--train` adds the same number of mock tracks cut from the TRAINING streams (`train_nll`, `train_S`, `train_stream`,
`train_dE` in the file) and prints, per track length, the loss on training streams, on held-out streams and the
gap between them: how much a run has learnt its own streams. The mocks are never scaled (`scale_aug` is a
training augmentation), so a run trained with it is asked about its streams at the scale they were simulated at.
"""

from __future__ import annotations

import argparse
import dataclasses
import os

import numpy as np

from config import REPO, Config, pick_device

MOCK_FIELDS = ("out", "subdir", "replace_tracks", "ordering", "val_frac", "include_unbound", "n_min", "var_ratio", "span_min", "span_max", "span_prior",
               "err_strata", "err_level", "err_scatter", "err_clip", "gap_prob", "gap_rate_max", "sig_sys_max", "noise_scale", "p_known", "seed")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run names under <out>/models/")
    ap.add_argument("--out", default="outputs/extragalactic/C26_test")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--n-ode", type=int, default=0, help="diffusion / flow matching: density of the first this-many mocks only (0 = all); an ODE solve each")
    ap.add_argument("--n-post", type=int, default=2000, help="mocks for which the posterior is also sampled (0 = none)")
    ap.add_argument("--draws", type=int, default=256, help="posterior draws for each of them")
    ap.add_argument("--train", action="store_true", help="also mocks of the training streams, and the gap between the two losses by track length")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--gpu", type=int, default=-1, help="a card to share with our own processes (config.py)")
    a = ap.parse_args()
    cfgs = [Config.load(REPO / a.out / "models" / r / "config.yaml")[0] for r in a.runs]
    cfg = cfgs[0]
    for c in cfgs[1:]:
        diff = {k: (getattr(cfg, k), getattr(c, k)) for k in MOCK_FIELDS if getattr(cfg, k) != getattr(c, k)}
        if diff:
            raise SystemExit(f"{c.name} has other mock settings than {cfg.name}: {diff}")
    print(f"device: {pick_device(a.cpu, a.gpu)}", flush=True)  # before jax / keras are imported
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import keras

    import networks  # noqa: F401  registers the custom layers
    from data import load_simulations, split
    from mock import fixed_observations

    keras.ops.zeros(1)  # a process on the card now: loading takes minutes, and until then the card looks free to others
    sims = load_simulations(cfg)
    seen, held = split(sims, cfg.val_frac)
    test = fixed_observations(held, cfg, sims["grid"], sims["design"], cfg.seed + 7, a.n, unweighted=True)
    own = fixed_observations(seen, cfg, sims["grid"], sims["design"], cfg.seed + 7, a.n, unweighted=True) if a.train else None
    bands = {"short 6-12": (6, 12), "medium 13-25": (13, 25), "long 26-35": (26, 35), "long 36-49": (36, 49), "long 26-49": (26, 49), "all": (0, 10**6)}
    stream = len(sims["theta"]) - len(held["zeta13"]) + test["pick"]
    fixed = np.ones_like(test["zeta"])
    edges = [int(x) for x in dataclasses.asdict(cfg)["length_edges"]]
    classes = {"short": test["S"] <= edges[0], "medium": (test["S"] > edges[0]) & (test["S"] <= edges[1]), "long": test["S"] > edges[1]}
    for c in cfgs:
        est = keras.saving.load_model(str(c.run_dir() / "model.keras"))
        n_dens = a.n_ode if (a.n_ode and c.inference != "flow") else a.n
        lp = [np.asarray(est.log_prob(dict(zeta=test["zeta"][s], tokens=test["tokens"][s], mask=test["mask"][s], cond=test["cond"][s],
                                           fixed_target_mask=fixed[s], fixed_target_value=0 * fixed[s])))
              for s in (slice(i, min(i + a.batch, n_dens)) for i in range(0, n_dens, a.batch))]  # fmt: skip
        nll = np.full(a.n, np.nan)
        nll[:n_dens] = -np.concatenate(lp)
        post = {}
        if a.n_post:  # posterior mean and sd of the latent parameters: recovery, on the same mocks for every run
            k = slice(0, a.n_post)
            z = est.sample(num_samples=a.draws, conditions=dict(tokens=test["tokens"][k], mask=test["mask"][k], cond=test["cond"][k],
                                                                fixed_target_mask=fixed[k], fixed_target_value=0 * fixed[k]), batch_size=256, seed=cfg.seed)["zeta"]  # fmt: skip
            post = dict(post_mean=z.mean(1), post_sd=z.std(1), truth=test["zeta"][k])
        if a.train and n_dens == a.n:  # the same for mock tracks cut from the streams the run trained on
            lp = [np.asarray(est.log_prob(dict(zeta=own["zeta"][s], tokens=own["tokens"][s], mask=own["mask"][s], cond=own["cond"][s],
                                               fixed_target_mask=fixed[s], fixed_target_value=0 * fixed[s])))
                  for s in (slice(i, min(i + a.batch, a.n)) for i in range(0, a.n, a.batch))]  # fmt: skip
            post |= dict(train_nll=-np.concatenate(lp), train_S=own["S"], train_stream=own["pick"], train_dE=sims["dE"][own["pick"]])
        if cfg.replace_tracks:
            post |= dict(replaced=sims["replaced"][stream]) | (dict(train_replaced=sims["replaced"][own["pick"]]) if "train_nll" in post else {})
        np.savez(c.run_dir() / "heldout_nll.npz", nll=nll, S=test["S"], n_obs=test["mask"].sum(1), stream=stream, dE=sims["dE"][stream], mocks_of=cfg.name, **post)
        ok = np.isfinite(nll)
        print(f"{c.name}: {nll[ok].mean():.3f} +- {nll[ok].std() / np.sqrt(ok.sum()):.3f} nats ({n_dens} mocks, {int((~ok[:n_dens]).sum())} not finite); "
              + ", ".join(f"{k} {nll[ok & s].mean():.3f}" for k, s in classes.items()), flush=True)  # fmt: skip
        if "train_nll" in post:
            print("  track length    training streams | held-out streams | gap (nats)")
            for lab, (lo, hi) in bands.items():
                x, y = post["train_nll"][(own["S"] >= lo) & (own["S"] <= hi)], nll[(test["S"] >= lo) & (test["S"] <= hi)]
                x, y = x[np.isfinite(x)], y[np.isfinite(y)]
                print(f"  {lab:14s} {x.mean():7.3f} | {y.mean():7.3f} | {y.mean() - x.mean():+.3f} +- {np.sqrt(x.var() / len(x) + y.var() / len(y)):.3f}   ({len(x)}, {len(y)} mocks)", flush=True)
        del est


if __name__ == "__main__":
    main()
