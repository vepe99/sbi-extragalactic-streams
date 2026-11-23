"""Settings of an SBI run. One dataclass serves training and both evaluations; every field is a command-line
option and can be set in a YAML file:

    uv run python scripts/extragalactic/sbi/train.py           --config outputs/extragalactic/C26_test/npe_v2.yaml
    uv run python scripts/extragalactic/sbi/evaluation.py      --config outputs/extragalactic/C26_test/npe_v2.yaml
    uv run python scripts/extragalactic/sbi/evaluation_real.py --config outputs/extragalactic/C26_test/npe_v2.yaml

The file is a flat list of `field: value`; fields it leaves out keep their defaults; explicit options
(`--epochs 200`) override the file. Everything a run produces goes to one directory, <out>/models/<name>/:
config.yaml (the settings as used, with the prior, the data statistics and the commit), train.log, history.csv,
model.keras (the epoch with the lowest held-out loss), evaluation/ and evaluation_real/.
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
    # share of examples in which the progenitor position is given to the network instead of inferred. 0 (decided
    # 2026-11-23): never given; the released STRRINGS tracks carry no progenitor position, so it is a nuisance
    # parameter the posterior marginalises over. 0.5 trains both queries (see mock.py)
    p_known: float = 0.0
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
    # --- evaluation (evaluation.py: mock tracks of held-out streams; evaluation_real.py: observed tracks)
    n_test: int = 1000  # mock tracks for recovery, calibration and contraction
    n_samples: int = 1000  # posterior draws per track
    prog_given: bool = False  # pass the progenitor position to the network (needs p_known > 0 at training)
    ppc_tracks: int = 6  # mock tracks in the posterior predictive check
    ppc_draws: int = 64  # posterior draws per track that are simulated again (predictive check, importance weights)
    real_tracks: str = ""  # observed tracks to evaluate, comma separated (e.g. NGC5907,NGC1162); "" = all 32
    importance: bool = True  # importance weights of the simulated draws towards prior x likelihood of the paper
    sim_workers: int = 16  # CPU worker processes generate.py uses for those simulations (about 10 core-seconds each)
    sim_cores: str = ""  # cores to pin them to, e.g. 0-15; "" = no pinning

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
        doc = yaml.safe_load(Path(path).read_text()) or {}
        nested = "config" in doc  # the file a run writes nests the settings under `config`; a hand-written one is flat
        settings = doc.pop("config") if nested else doc
        names = {f.name for f in dataclasses.fields(cls)}
        if not nested and set(settings) - names:  # a typo in a hand-written file must not pass silently
            raise SystemExit(f"{path}: unknown setting(s) {sorted(set(settings) - names)}")
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in settings.items() if k in names}), (doc if nested else {})


EVALUATION_FIELDS = ("n_test", "n_samples", "prog_given", "ppc_tracks", "ppc_draws", "real_tracks", "importance",
                     "sim_workers", "sim_cores", "cpu")  # fmt: skip


def parse() -> Config:
    """Command line -> Config. `--config file.yaml` is read first, explicit options override it. Also selects
    the device, so call this before importing jax or keras."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="")
    for f in dataclasses.fields(Config):
        flag, d = "--" + f.name.replace("_", "-"), f.default
        if isinstance(d, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        else:
            p.add_argument(flag, default=None, type=(lambda s: tuple(float(x) for x in s.split(","))) if isinstance(d, tuple) else type(d))
    a = vars(p.parse_args())
    path = a.pop("config")
    cfg = dataclasses.replace(Config.load(path)[0] if path else Config(), **{k: v for k, v in a.items() if v is not None})
    cfg.device = pick_device(cfg.cpu)
    return cfg


def trained(cfg: Config) -> Path:
    """The run directory of a trained model, after checking that the settings in hand are the ones it was
    trained with (the evaluation settings may differ)."""
    run = cfg.run_dir()
    if not (run / "model.keras").exists():
        raise SystemExit(f"no trained model in {run}: run train.py with the same settings first")
    then = dataclasses.asdict(Config.load(run / "config.yaml")[0])
    diff = {k: (then[k], v) for k, v in dataclasses.asdict(cfg).items() if k not in EVALUATION_FIELDS and then[k] != v}
    if diff:
        raise SystemExit(f"settings differ from those {run.name} was trained with (trained, now): {diff}")
    return run


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
