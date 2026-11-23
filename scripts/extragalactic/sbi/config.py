"""Settings of an SBI run. One dataclass; every field is a command-line option and the whole of it is written
to `config.yaml` in the run directory, which is what `evaluation.py` (and anyone reproducing the run) reads.

    uv run python scripts/extragalactic/sbi/train.py --name dm_v1 --inference diffusion --backbone transformer
    uv run python scripts/extragalactic/sbi/train.py --config some/config.yaml --epochs 200   # file, then overrides

Everything a run produces goes to one directory, <out>/models/<name>/: config.yaml, train.log, history.csv,
model.keras (the epoch with the lowest held-out loss), evaluation/ (figures, metrics.json, posterior predictive
simulations) and posterior_observed.npz.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]


@dataclass
class Config:
    # --- data
    out: str = "outputs/extragalactic/C26_test"
    subdir: str = "train_v1"  # simulation set: <out>/data/<subdir>/shard_*.h5
    ordering: str = "c26"  # track ordering: c26 | gibbons14 (the one of 2026b, stored as track_alt/)
    max_shards: int = 0  # 0 = all
    val_frac: float = 0.05  # last fraction of the usable streams, held out for validation and evaluation
    n_val: int = 20000  # fixed mock observations of the held-out streams used as validation set
    include_unbound: bool = True  # the paper keeps the raw prior
    obs_dir: str = "outputs/extragalactic/C26_test/data/strrings"
    # --- mock observation (mock.py): the two guards of 2026b sec. 3.3 ...
    n_min: int = 10  # model particles per bin (3 of 10,000 in the paper = 9 of the 30,000 simulated)
    var_ratio: float = 9.0  # model variance <= data variance / var_ratio; 0 = off
    # ... the observed window ...
    span_min: int = 6  # bins; the shortest observed track has 7 points
    span_max: int = 64  # bins = tokens; the longest observed track spans 49
    span_prior: str = "loguniform"  # distribution of the track length of the examples: loguniform | uniform
    err_strata: int = 8  # strata of the error level used to draw streams when the guard is on
    # ... errors, gaps, systematic term, progenitor
    err_level: tuple = (0.1, 5.0)  # kpc; track level of sigma_data, log-uniform (observed medians 0.2-2.9)
    err_scatter: tuple = (0.2, 1.0)  # log-normal scatter along a track (observed 0.4-0.95)
    err_clip: tuple = (0.03, 20.0)  # kpc
    gap_prob: float = 0.3  # share of mock tracks with missing interior bins
    gap_rate_max: float = 0.25  # each interior bin of such a track is missing with a rate U(0, this)
    sig_sys_max: float = 25.0  # kpc; sigma_sys ~ U(0, sig_sys_max), the prior of the paper
    p_known: float = 0.5  # share of examples in which the progenitor position is given (fixed), not inferred
    # --- networks (networks.py)
    inference: str = "flow"  # flow (coupling, spline) | diffusion | flow_matching
    backbone: str = "mlp"  # subnet of diffusion / flow_matching: mlp | transformer (flow always uses an MLP)
    width: int = 256  # width of the inference subnet
    depth: int = 6  # coupling layers (flow) or subnet layers (diffusion, flow_matching)
    flow_layers: int = 6  # hidden layers of the MLP inside each coupling layer of the flow
    summary: str = "transformer"  # transformer (angle = time axis, padding mask) | timeseries (conv + GRU) | mlp
    summary_dim: int = 32
    embed_dim: int = 128
    n_blocks: int = 3
    n_heads: int = 4
    dropout: float = 0.05
    # --- training (train.py)
    name: str = "npe_v2"
    epochs: int = 100
    batch_size: int = 512
    lr: float = 5e-4
    seed: int = 2607
    cpu: bool = False  # stay off the GPUs

    def data_dir(self) -> Path:
        return REPO / self.out / "data" / self.subdir

    def run_dir(self) -> Path:
        return REPO / self.out / "models" / self.name

    def save(self, extra: dict | None = None) -> None:
        d = {k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(self).items()}
        (self.run_dir() / "config.yaml").write_text(yaml.safe_dump(dict(config=d, **(extra or {})), sort_keys=False))

    @classmethod
    def load(cls, path: Path) -> tuple["Config", dict]:
        """-> (settings, everything else stored with them: prior, grid, data statistics, commit)."""
        doc = yaml.safe_load(Path(path).read_text())
        names = {f.name for f in dataclasses.fields(cls)}  # a file written before a field was removed still loads
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in doc.pop("config").items() if k in names}), doc


def parse(extra_args=()) -> tuple[Config, argparse.Namespace]:
    """Command line -> (Config, the extra arguments of the calling script). `--config file.yaml` is read first,
    explicit options override it. Also selects the device, so call this before importing jax or keras."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="")
    for f in dataclasses.fields(Config):
        flag, d = "--" + f.name.replace("_", "-"), f.default
        if isinstance(d, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        else:
            p.add_argument(flag, default=None, type=(lambda s: tuple(float(x) for x in s.split(","))) if isinstance(d, tuple) else type(d))
    names = []
    for flag, kw in extra_args:
        names.append(p.add_argument(flag, **kw).dest)
    a = vars(p.parse_args())
    path = a.pop("config")
    cfg = Config.load(path)[0] if path else Config()
    extras = argparse.Namespace(**{n: a.pop(n) for n in names})
    cfg = dataclasses.replace(cfg, **{k: v for k, v in a.items() if v is not None})
    cfg.device = pick_device(cfg.cpu)
    return cfg, extras


def pick_device(cpu: bool) -> str:
    """CPU if asked or if the machine has no NVIDIA driver; otherwise one FREE GPU, chosen by autocvd (it waits
    for one and sets CUDA_VISIBLE_DEVICES). Under Slurm the allocation is used as given."""
    os.environ.setdefault("KERAS_BACKEND", "jax")
    if cpu or shutil.which("nvidia-smi") is None:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu" + ("" if cpu else " (no NVIDIA driver)")
    if os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES"):
        return f"gpu {os.environ['CUDA_VISIBLE_DEVICES']} (Slurm)"
    from autocvd import autocvd

    return f"gpu {autocvd(num_gpus=1)} (autocvd)"
