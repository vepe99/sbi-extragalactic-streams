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
import re
import shutil
import subprocess
import sys
import time
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
    # drop the streams whose progenitor orbit the simulator's steps do not resolve: relative energy error
    # (`diag/dE_rel`) above this. 0 = keep all, the paper's prior. 0.01 is the criterion of report.md; above it the
    # track jumps when the parameters move by 1e-4 of the prior width (scripts/extragalactic/track_noise.py). The
    # prior is then the paper's restricted to resolved orbits, and the importance weights of the evaluations give
    # an unresolved draw the weight 0. nested.py does not know this setting yet
    de_max: float = 0.0
    # put the tracks of a set of streams of `subdir` that were simulated again with more particles in place of the
    # stored ones (scripts/extragalactic/long90k.py; its name under <out>/data/, e.g. train_v2_long90k). "" = off.
    # Replaced, not added: a second copy of a stream would change the prior. Guard 1 is a fraction of the particles,
    # so a replaced stream needs n_min x N / N_set particles in a bin (30 at 90,000); guard 2 is as it is. A stream
    # whose new simulation fails the loader's cuts keeps its stored track
    replace_tracks: str = ""
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
    # factor on the scatter of the mock radii around the model track. 1 = a draw from the likelihood. 0 = the
    # model track itself (a test of what the track holds without observational noise): errors, guards, windows
    # and gaps are drawn as before and the tokens still carry sigma, but sigma_sys then leaves no trace in the
    # track (its posterior is its prior) and the observed tracks and the paper's likelihood do not apply
    noise_scale: float = 1.0
    # share of examples in which the progenitor position is given to the network instead of inferred. 0 (decided
    # 2026-11-23): never given; the released STRRINGS tracks carry no progenitor position, so it is a nuisance
    # parameter the posterior marginalises over. 0.5 trains both queries (see mock.py)
    p_known: float = 0.0
    # scale augmentation of the training examples (mock.py, `draw_scale`): the simulator has no scale of its own,
    # so a stream with lengths x L and masses x U (velocities x sqrt(U / L), t_int x sqrt(L^3 / U)) is the same
    # stream with its track x L. Every training example then gets its own (L, U), drawn so that the scaled
    # parameters still follow the prior; a dissolved progenitor only L (its mass is ramped to a fixed 1 Msun).
    # The fixed mock tracks of the held-out streams are never scaled
    scale_aug: bool = False
    scale_points: int = 256  # points of the plane (ln L, ln U) the draw chooses among, per example
    # --- networks (networks.py)
    inference: str = "flow"  # flow (coupling flow) | diffusion | flow_matching
    # the coupling flow's transform: spline (rational-quadratic, the more flexible) | affine (scale and shift,
    # BayesFlow's default, with its soft clamp of the scale)
    transform: str = "spline"
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
    # BayesFlow's standardisation (running mean and sd per feature, learnt during training and stored in the
    # model): none | all | a comma-separated list of inference_variables (the parameters), summary_variables
    # (the tokens; moments over the observed bins only), inference_conditions
    standardize: str = "none"
    # --- training (train.py)
    name: str = "npe_v2"
    epochs: int = 100
    batch_size: int = 512
    lr: float = 5e-4
    seed: int = 2607
    resume: bool = False  # train.py: take the run <name> up again from its model.keras (see train.py); the other settings must be the run's
    cpu: bool = False  # stay off the GPUs
    # nvidia-smi index of a card to take as it is, without waiting for a free one: only a card whose processes
    # are all our own (checked; another user's process on it ends the run). -1 = a free card, the normal way
    gpu: int = -1
    # hold the training set on the device that trains (1.0 GB per million usable streams at span_max 49) and
    # make the batches there; train.py then lets JAX take 95 % of the card. false = on the host: the same
    # batches, 17 ms a step slower at batch 4096. Use false when the run stops with RESOURCE_EXHAUSTED: the set
    # and the training step do not fit together (the step of embed_dim 256, n_blocks 6 takes 26.9 GiB at batch
    # 4096, so a 40 GB card holds it with a set of up to about 10 GiB)
    data_on_device: bool = True
    # --- evaluation (evaluation.py: mock tracks of held-out streams; evaluation_real.py: observed tracks)
    n_test: int = 1000  # mock tracks for recovery, calibration and contraction
    n_samples: int = 1000  # posterior draws per track
    prog_given: bool = False  # pass the progenitor position to the network (needs p_known > 0 at training)
    ppc_tracks: int = 6  # mock tracks in the posterior predictive check
    ppc_draws: int = 64  # posterior draws per track that are simulated again (predictive check, importance weights)
    # evaluation.py repeats recovery, calibration and contraction for short, medium and long mock tracks. The two
    # edges between the classes, in bins of track length (first to last bin): 12, 25 gives short 6-12, medium
    # 13-25, long 26-49, three classes of about equal size under the log-uniform length prior
    length_edges: tuple = (12.0, 25.0)
    real_tracks: str = ""  # observed tracks to evaluate, comma separated (e.g. NGC5907,NGC1162); "" = all 32
    # evaluation_real.py writes to evaluation_real[_<integrator>]_<tag>/, next to the evaluation without a tag
    # (e.g. one track with many more draws); nested.py then compares with that directory
    eval_tag: str = ""
    importance: bool = True  # importance weights of the simulated draws towards prior x likelihood of the paper
    # where generate.py simulates them: gpu = the card this evaluation holds plus the cards free at that moment
    # (at most sim_workers of them) | cpu = sim_workers processes. A CPU evaluation (`cpu: true`, or no NVIDIA
    # driver) always simulates on the CPU
    sim_device: str = "gpu"
    # integrator of those simulations: "" = that of the training set | leapfrog | galax. The adaptive galax
    # integrator takes about 10 s per stream on a CPU core and is unusable on a GPU (no stream finished in
    # 280 s on a 2080 Ti); the leapfrog (StreaMAX's own, the paper's) runs tens of streams per second on a GPU.
    # With an integrator named here the results go to evaluation*_<integrator>/, next to the default ones
    sim_integrator: str = ""
    sim_batch: int = 0  # gpu: streams per batch; 0 = the simulator's choice (1 for galax, a memory probe for leapfrog)
    sim_workers: int = 16
    sim_cores: str = ""  # cpu: cores to pin the processes to, e.g. 0-15; "" = no pinning
    # --- nested.py: dynamic nested sampling (dynesty) of the observed tracks in `real_tracks`, the simulator on
    # GPUs. Sampler defaults = the "paper" preset of StreaMAX-fit: 2000 live points, random-slice sampling,
    # 10,000 effective samples
    ns_gpus: int = 4  # free GPUs to use, one process each (a CPU run uses one CPU process)
    ns_batch: int = 128  # streams simulated at once on a GPU; dynesty proposes ns_gpus x ns_batch points at a time
    ns_nlive: int = 2000  # live points of the baseline run and of every batch added to it
    ns_sample: str = "rslice"  # dynesty's sampler inside the bounds: rslice | slice | rwalk | unif | auto
    ns_neff: int = 10000  # stop when the posterior has this many effective samples
    ns_dlogz: float = 0.01  # the baseline run stops when the evidence left in the live points is below this
    ns_maxcall: int = 0  # stop after this many likelihood calls: for a test, such a run is not converged. 0 = no limit
    ns_tag: str = ""  # results go to <out>/nested/<track>[_<tag>]/

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


EVALUATION_FIELDS = ("resume", "gpu", "n_test", "n_samples", "prog_given", "ppc_tracks", "ppc_draws", "length_edges", "real_tracks", "eval_tag", "importance",
                     "sim_device", "sim_integrator", "sim_batch", "sim_workers", "sim_cores", "cpu", "data_on_device",
                     "ns_gpus", "ns_batch", "ns_nlive", "ns_sample", "ns_neff", "ns_dlogz", "ns_maxcall", "ns_tag")  # fmt: skip


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
    cfg.device = pick_device(cfg.cpu, cfg.gpu)
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


def pick_device(cpu: bool, gpu: int = -1) -> str:
    """CPU if asked or if the machine has no NVIDIA driver; otherwise one FREE GPU (`free_gpu` waits for one and
    sets CUDA_VISIBLE_DEVICES). Under Slurm the allocation is used as given. gpu >= 0 names a card to share with
    our own processes (`own_gpu`)."""
    os.environ.setdefault("KERAS_BACKEND", "jax")
    if cpu or shutil.which("nvidia-smi") is None:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu" + ("" if cpu else " (no NVIDIA driver)")
    if os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES"):
        return f"gpu {os.environ['CUDA_VISIBLE_DEVICES']} (Slurm)"
    if gpu >= 0:
        return f"gpu {own_gpu(gpu)} (shared with our own processes)"
    return f"gpu {free_gpu()} (free)"


def own_gpu(index: int) -> str:
    """Point CUDA_VISIBLE_DEVICES at the card with this nvidia-smi index, which need not be free, after checking
    that every compute process on it belongs to the user who runs this. The memory left on the card is the
    caller's concern (XLA_PYTHON_CLIENT_MEM_FRACTION)."""
    query = lambda what, fields: subprocess.run(["nvidia-smi", f"--query-{what}={fields}", "--format=csv,noheader"],
                                                capture_output=True, text=True).stdout.splitlines()  # noqa: E731  # fmt: skip
    uuid = {i.strip(): u.strip() for i, u in (ln.split(",") for ln in query("gpu", "index,uuid") if "," in ln)}.get(str(index))
    if uuid is None:
        raise SystemExit(f"gpu {index}: nvidia-smi lists no such card")
    for pid, card in (ln.split(",") for ln in query("compute-apps", "pid,gpu_uuid") if "," in ln):
        try:
            mine = os.stat(f"/proc/{pid.strip()}").st_uid == os.getuid()
        except FileNotFoundError:  # ended since the query
            continue
        if card.strip() == uuid and not mine:
            raise SystemExit(f"gpu {index} holds a process of another user (pid {pid.strip()}): not used")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    return str(index)


def free_gpus() -> list[tuple[str, str]]:
    """(nvidia-smi index, UUID) of every GPU that is free right now, by autocvd's own test `gpu_is_free` (= no
    compute process on the card).

    `autocvd(num_gpus=1)` cannot be used on a machine with a broken card (compgpu6 has one): it counts the cards
    with `nvidia-smi -L`, which then exits with an error. So the cards are listed with a query that survives
    the broken one. Same method as `free_gpus` in scripts/extragalactic/generate.py."""
    from autocvd.nvidia_smi_calls import gpu_is_free

    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout  # fmt: skip
    rows = [m.groups() for m in (re.match(r"\s*(\d+),\s*(GPU-\S+)", ln) for ln in out.splitlines()) if m]
    if not rows:
        raise SystemExit("nvidia-smi lists no usable GPU; pass --cpu for a CPU run")
    free = []
    for idx, uuid in rows:
        try:
            if gpu_is_free(int(idx)):
                free.append((idx, uuid))
        except subprocess.CalledProcessError:  # a card nvidia-smi cannot query is not usable
            pass
    return free


def free_gpu(interval: int = 30) -> str:
    """Wait for a free GPU, point CUDA_VISIBLE_DEVICES at it and return its nvidia-smi index: the free card with
    the highest index, as autocvd does. The card is addressed by UUID because a broken card shifts the CUDA
    indices relative to nvidia-smi's."""
    start = time.time()
    while True:
        free = free_gpus()
        if free:
            idx, uuid = free[-1]
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = uuid
            return idx
        print(f"no free GPU, waiting ({int(time.time() - start)} s)", file=sys.stderr, end="\r")
        time.sleep(interval)
