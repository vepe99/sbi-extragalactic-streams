"""Train an amortised posterior estimator for extragalactic stream tracks (BayesFlow, JAX backend).

    uv run python scripts/extragalactic/sbi/train.py --config outputs/extragalactic/C26_test/npe_v2.yaml
    uv run python scripts/extragalactic/sbi/train.py --name dm_v1 --inference diffusion --backbone transformer

Settings come from the YAML file and/or options (config.py). Without --cpu a free GPU is taken with autocvd.
Written to <out>/models/<name>/: config.yaml (all settings), train.log (one line per epoch), history.csv,
model.keras (the epoch with the lowest held-out loss, rewritten whenever it improves). A run whose loss stops
being finite ends at that epoch, with exit code 1; model.keras is then the best epoch before it.

    uv run python scripts/extragalactic/sbi/train.py --config <out>/models/<name>/config.yaml --resume

takes a run up again from its model.keras (after a NaN, a kill, a reboot): weights, standardisation and the
optimizer with its step count are read from it, so the learning-rate schedule goes on where it was, and the
epochs after the one in model.keras are trained again. The settings must be those of the run, hence its own
config.yaml. Stop the first process before: two would write to the same directory.

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

from config import REPO, parse, trained

cfg = parse()  # selects the device: must come before jax / keras are imported
if cfg.data_on_device:  # the set and the network share the card, which is ours alone: JAX may take 95 % of it, not its usual 75 %
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import jax  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402

from data import I_PROG, PARAM_NAMES, load_simulations, split  # noqa: E402
from mock import NETWORK_KEYS, DeviceDataset, MockObservations, SelectionDataset, accepted_inputs, fixed_observations  # noqa: E402
from networks import build_workflow, network_names, standardized  # noqa: E402


class Journal(keras.callbacks.Callback):
    """train.log and history.csv. The checkpoint itself is written by the workflow.

    With `standardize` on for the parameters BayesFlow computes the loss on the standardised parameters, which
    differs from the loss in nats by the sum of the log sd of the inferred ones. That sum is added back here,
    so losses compare across runs and with the "nothing learnt" level. The checkpoint goes by BayesFlow's own
    number, and so does the `best` flag."""

    def __init__(self, run, epochs: int, start: int = 0, best: float = math.inf):
        super().__init__()
        self.run, self.epochs, self.t0, self.best, self.best_raw = run, epochs, time.time(), best, math.inf
        self.start = start  # epochs done before this process: those of a resumed run, with `best` the loss of the last
        self.callbacks = [self]  # the list handed to the fit, to which the workflow adds its checkpoint
        self.diverged = 0  # the epoch at which the loss stopped being finite, 0 = never

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

    def on_train_begin(self, logs=None):
        if self.start:  # resumed: the epoch in model.keras is the one to beat, here and for the checkpoint
            self.best_raw = self.best - self.to_nats()  # the model is as it was at the end of that epoch
            for cb in self.callbacks:
                if isinstance(cb, keras.callbacks.ModelCheckpoint):
                    cb.best = self.best_raw

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
        left = (time.time() - self.t0) / (epoch + 1 - self.start) * (self.epochs - epoch - 1)
        self.say(f"epoch {epoch + 1}/{self.epochs} done  loss {loss:.4f}  val_loss {val:.4f}  best {self.best:.4f}"
                 f"{'  [saved model.keras]' if best else ''}  {time.time() - self.t_epoch:.0f} s  remaining {left / 3600:.2f} h")  # fmt: skip
        if not (math.isfinite(loss) and math.isfinite(val)):  # NaN weights never recover: end the run here
            self.diverged, self.model.stop_training = epoch + 1, True


def resume_point(run, steps_done: int, steps_per_epoch: int) -> tuple[int, float]:
    """-> (the epoch model.keras holds, its held-out loss in nats), from the optimizer's step count and
    history.csv, whose later rows are dropped: those epochs are trained again (train.log keeps their lines)."""
    lines = (run / "history.csv").read_text().splitlines()
    start, rest = divmod(steps_done, steps_per_epoch)
    row = next((i for i, line in enumerate(lines) if line.split(",")[0] == str(start) and line.split(",")[4] == "1"), None)
    if rest or row is None:
        raise SystemExit(f"{run}/model.keras is {steps_done} steps into training, at {steps_per_epoch} steps per epoch: "
                         "not an epoch that history.csv flags as best, so not a model of this run with these settings")  # fmt: skip
    if start >= cfg.epochs:
        raise SystemExit(f"{run}/model.keras is epoch {start} of {cfg.epochs}: nothing left to train")
    (run / "history.csv").write_text("\n".join(lines[: row + 1]) + "\n")
    return start, float(lines[row].split(",")[2])


def main() -> None:
    run = cfg.run_dir()
    if cfg.resume:
        trained(cfg)  # there is a model.keras, and the settings in hand are those it was trained with
    elif (run / "history.csv").exists():
        raise SystemExit(f"{run} already holds a run; choose another --name, or --resume to take it up from its model.keras")
    run.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(cfg.seed)
    # Claim the card now (JAX takes its share of the memory at its first allocation): loading takes minutes, and
    # until then a card that only CUDA_VISIBLE_DEVICES points at looks free to every other job that waits for one.
    jax.numpy.zeros(1).block_until_ready()

    sims = load_simulations(cfg)
    train, held_out = split(sims, cfg.val_frac)
    grid, design = sims["grid"], sims["design"]
    # held-out set: fixed mock observations, the same at every epoch. Their mean loss weight normalises the
    # weights of the training examples too, so that the loss stays in nats (the weight is 1 with the guard off)
    val = fixed_observations(held_out, cfg, grid, design, cfg.seed + 1, cfg.n_val)
    w_mean = float(val["weight"].mean())
    val = accepted_inputs({k: val[k] for k in NETWORK_KEYS} | dict(weight=val["weight"] / w_mean))  # as the training batches

    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    workflow, steps_done = build_workflow(cfg), 0
    if cfg.resume:  # the estimator as model.keras holds it; config.yaml stays that of the first start
        workflow.estimator = keras.saving.load_model(str(run / "model.keras"))
        workflow.optimizer, workflow._needs_compile = workflow.estimator.optimizer, False  # the fit then keeps this optimizer
        steps_done = int(workflow.optimizer.iterations)
    else:
        cfg.save(dict(parameters=PARAM_NAMES, prior=dataclasses.asdict(sims["prior"]), grid=dataclasses.asdict(grid), data=sims["stats"],
                      window_prior=dict(S=design["S"].tolist(), p0=design["pS"].tolist()), commit=commit, device=cfg.device))  # fmt: skip
    # a resumed run draws new streams and new mock observations, not those of the first epochs again (seed + 1 is the held-out set's)
    seed = cfg.seed + (2 + steps_done if cfg.resume else 0)
    if cfg.data_on_device:  # the same batches either way; on the device they skip the trip through the host
        dataset = DeviceDataset(train, cfg, grid, design, seed=seed, batch_size=cfg.batch_size, names=network_names(cfg), weight_norm=w_mean, prior=sims["prior"])
    else:
        dataset = SelectionDataset(train, seed=seed, batch_size=cfg.batch_size, adapter=workflow.adapter, drop_last=True,
                                   augmentations=[MockObservations(cfg, grid, design, seed, weight_norm=w_mean, prior=sims["prior"])])  # fmt: skip
    start, best = resume_point(run, steps_done, len(dataset)) if cfg.resume else (0, math.inf)
    journal = Journal(run, cfg.epochs, start, best)
    journal.say(f"run {cfg.name}  device {cfg.device}  jax {jax.devices()}  commit {commit}")
    journal.say(f"data {cfg.subdir}: {sims['stats']['n_streams']} streams, usable {sims['stats']['n_usable']}, "
                f"train {len(train['zeta13'])}, held out {len(held_out['zeta13'])} -> {cfg.n_val} fixed mock tracks; "
                f"training set {f'on the device ({dataset.nbytes / 1e9:.1f} GB)' if cfg.data_on_device else 'on the host'}")  # fmt: skip
    if cfg.replace_tracks:
        st = sims["stats"]
        journal.say(f"tracks of {cfg.replace_tracks} in place of the stored ones for {st['replaced']} of its {st['replaced_of']} streams "
                    f"({st['replaced_n_particles']} particles, n_min {st['replaced_n_min']:g}); {int(sims['replaced'][: len(train['zeta13'])].sum())} of them train")  # fmt: skip
    journal.say(f"scale augmentation {'on: ' + str(cfg.scale_points) + ' points per example' if cfg.scale_aug else 'off'}")
    journal.say(f"{cfg.inference} ({cfg.backbone if cfg.inference != 'flow' else f'{cfg.transform}, mlp'}) + {cfg.summary} summary; "
                f"{len(dataset)} steps per epoch at batch {cfg.batch_size}; progenitor given in {cfg.p_known:.0%} of examples; "
                f"standardised: {', '.join(standardized(cfg)) or 'nothing'}")  # fmt: skip
    if cfg.resume:
        journal.say(f"resumed from model.keras: epoch {start}, val_loss {best:.4f}, learning rate {float(workflow.optimizer.learning_rate):.2e}; "
                    f"the epochs after {start} are trained again")  # fmt: skip
    workflow._fit(dataset, cfg.epochs, strategy="offline", keep_optimizer=False, validation_data=val, callbacks=journal.callbacks,
                  verbose=0, initial_epoch=start)  # fmt: skip
    if journal.diverged:
        kept = f"model.keras is the epoch with the lowest held-out loss before it ({journal.best:.4f})" if math.isfinite(journal.best) else "no usable model.keras"
        journal.say(f"stopped at epoch {journal.diverged}: the loss is not finite; {kept}")
        raise SystemExit(1)
    journal.say(f"finished  best val_loss {journal.best:.4f}  parameters {workflow.estimator.count_params()}")


if __name__ == "__main__":
    main()
