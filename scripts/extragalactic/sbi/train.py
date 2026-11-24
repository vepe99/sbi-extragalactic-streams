"""Train an amortised posterior estimator for extragalactic stream tracks (BayesFlow, JAX backend).

    uv run python scripts/extragalactic/sbi/train.py --config outputs/extragalactic/C26_test/npe_v2.yaml
    uv run python scripts/extragalactic/sbi/train.py --name dm_v1 --inference diffusion --backbone transformer

Settings come from the YAML file and/or options (config.py). Without --cpu a free GPU is taken with autocvd.
Written to <out>/models/<name>/: config.yaml (all settings), train.log (one line per epoch), history.csv,
model.keras (the epoch with the lowest held-out loss, rewritten whenever it improves).

For the coupling flow the loss is in nats per example: a network that learnt nothing scores the entropy of the
prior over the inferred parameters, 1.419 nats each (21.3 for all 15, 19.9 when the progenitor position is
given), also when the parameters are standardised (`Journal` adds the Jacobian back). Diffusion and flow matching
minimise a regression loss; their values compare only within a run.
"""

from __future__ import annotations

import dataclasses
import math
import os
import subprocess
import time

from config import REPO, parse

cfg = parse()  # selects the device: must come before jax / keras are imported
if cfg.data_on_device:  # the set and the network share the card, which is ours alone: JAX may take 95 % of it, not its usual 75 %
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import jax  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402

from data import I_PROG, PARAM_NAMES, load_simulations, split  # noqa: E402
from mock import NETWORK_KEYS, DeviceDataset, MockObservations, SelectionDataset, fixed_observations  # noqa: E402
from networks import build_workflow, network_names, standardized  # noqa: E402


class Journal(keras.callbacks.Callback):
    """train.log and history.csv. The checkpoint itself is written by the workflow.

    With `standardize` on for the parameters BayesFlow computes the loss on the standardised parameters, which
    differs from the loss in nats by the sum of the log sd of the inferred ones. That sum is added back here,
    so losses compare across runs and with the "nothing learnt" level. The checkpoint goes by BayesFlow's own
    number, and so does the `best` flag."""

    def __init__(self, run, epochs: int):
        super().__init__()
        self.run, self.epochs, self.t0, self.best, self.best_raw = run, epochs, time.time(), math.inf, math.inf

    def to_nats(self) -> float:
        layer = (getattr(self.model.standardizer, "standardize_layers", None) or {}).get("inference_variables")
        if layer is None or cfg.inference != "flow":
            return 0.0
        log_sd = np.log(keras.ops.convert_to_numpy(layer.moving_std(0)))
        return float(log_sd.sum() - cfg.p_known * log_sd[I_PROG])  # a given progenitor position is not in the density

    def say(self, msg: str) -> None:
        line = f"{time.strftime('%F %T')}  {msg}"
        with open(self.run / "train.log", "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def on_epoch_begin(self, epoch, logs=None):
        self.t_epoch = time.time()

    def on_epoch_end(self, epoch, logs=None):
        raw, shift = float(logs["val_loss"]), self.to_nats()  # the sd are those at the end of the epoch
        loss, val = float(logs["loss"]) + shift, raw + shift
        best = raw < self.best_raw
        self.best_raw, self.best = min(self.best_raw, raw), val if best else self.best
        new = not (self.run / "history.csv").exists()
        with open(self.run / "history.csv", "a") as f:
            f.write("epoch,loss,val_loss,seconds,best\n" * new + f"{epoch + 1},{loss:.6f},{val:.6f},{time.time() - self.t_epoch:.1f},{int(best)}\n")
        left = (time.time() - self.t0) / (epoch + 1) * (self.epochs - epoch - 1)
        self.say(f"epoch {epoch + 1}/{self.epochs} done  loss {loss:.4f}  val_loss {val:.4f}  best {self.best:.4f}"
                 f"{'  [saved model.keras]' if best else ''}  {time.time() - self.t_epoch:.0f} s  remaining {left / 3600:.2f} h")  # fmt: skip


def main() -> None:
    run = cfg.run_dir()
    if (run / "history.csv").exists():
        raise SystemExit(f"{run} already holds a run; choose another --name")
    run.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(cfg.seed)

    sims = load_simulations(cfg)
    train, held_out = split(sims, cfg.val_frac)
    grid, design = sims["grid"], sims["design"]
    # held-out set: fixed mock observations, the same at every epoch. Their mean loss weight normalises the
    # weights of the training examples too, so that the loss stays in nats (the weight is 1 with the guard off)
    val = fixed_observations(held_out, cfg, grid, design, cfg.seed + 1, cfg.n_val)
    w_mean = float(val["weight"].mean())
    val = {k: val[k] for k in NETWORK_KEYS} | dict(weight=val["weight"] / w_mean)

    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    cfg.save(dict(parameters=PARAM_NAMES, prior=dataclasses.asdict(sims["prior"]), grid=dataclasses.asdict(grid), data=sims["stats"],
                  window_prior=dict(S=design["S"].tolist(), p0=design["pS"].tolist()), commit=commit, device=cfg.device))  # fmt: skip

    workflow = build_workflow(cfg)
    if cfg.data_on_device:  # the same batches either way; on the device they skip the trip through the host
        dataset = DeviceDataset(train, cfg, grid, design, seed=cfg.seed, batch_size=cfg.batch_size, names=network_names(cfg), weight_norm=w_mean)
    else:
        dataset = SelectionDataset(train, seed=cfg.seed, batch_size=cfg.batch_size, adapter=workflow.adapter, drop_last=True,
                                   augmentations=[MockObservations(cfg, grid, design, cfg.seed, weight_norm=w_mean)])  # fmt: skip
    journal = Journal(run, cfg.epochs)
    journal.say(f"run {cfg.name}  device {cfg.device}  jax {jax.devices()}  commit {commit}")
    journal.say(f"data {cfg.subdir}: {sims['stats']['n_streams']} streams, usable {sims['stats']['n_usable']}, "
                f"train {len(train['zeta13'])}, held out {len(held_out['zeta13'])} -> {cfg.n_val} fixed mock tracks; "
                f"training set {f'on the device ({dataset.nbytes / 1e9:.1f} GB)' if cfg.data_on_device else 'on the host'}")  # fmt: skip
    journal.say(f"{cfg.inference} ({cfg.backbone if cfg.inference != 'flow' else 'mlp'}) + {cfg.summary} summary; "
                f"{len(dataset)} steps per epoch at batch {cfg.batch_size}; progenitor given in {cfg.p_known:.0%} of examples; "
                f"standardised: {', '.join(standardized(cfg)) or 'nothing'}")  # fmt: skip
    workflow._fit(dataset, cfg.epochs, strategy="offline", keep_optimizer=False, validation_data=val, callbacks=[journal], verbose=0)
    journal.say(f"finished  best val_loss {journal.best:.4f}  parameters {workflow.estimator.count_params()}")


if __name__ == "__main__":
    main()
