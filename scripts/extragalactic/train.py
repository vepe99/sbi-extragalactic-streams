"""Amortised posterior estimation for extragalactic stream tracks (BayesFlow, JAX backend).

Run inside the project environment (pyproject.toml):

    uv run python scripts/extragalactic/train.py check     # tests of the transforms, the mock observations, the likelihood
    uv run python scripts/extragalactic/train.py train     # train; writes train.log, history.csv, best.keras, last.keras
    uv run python scripts/extragalactic/train.py sample    # posterior samples + log q for the 32 STRRINGS tracks

Every field of `Config` is a command-line option (``--summary mlp --epochs 50`` ...). Add ``--cpu`` to stay off
the GPUs; without it a free GPU is taken with autocvd when the machine has one.

WHAT IS INFERRED
----------------
The 14 parameters of Chemaly et al. 2026b (arXiv:2607.05510, table A1), in the order of StreaMAX-fit:

    logM  Rs  dirx  diry  dirz  logm  rs  x0  z0  vx0  vy0  vz0  time  sig

13 of them come from the simulation set (`theta` and `c26_vec` of the shards; velocities in km/s as in the
table), `sig` is the systematic term sigma_sys ~ U(0, 25) kpc, which is drawn in the augmentation because it
only enters the noise. The flow does not work on these directly but on

    zeta_k = Phi^-1( F_k(parameter_k) )          F_k = prior cdf of parameter k, Phi = normal cdf

so the prior is exactly N(0, I_14) in zeta, and F_k(parameter_k) is the unit-cube coordinate dynesty samples
in the paper. Importance weights can then be written in zeta with no Jacobian (see `importance_weights`).

FRAME (why there is no rotation or mirror augmentation)
-------------------------------------------------------
The prior of the paper keeps one copy of every set of equivalent systems: progenitor on the +x axis (y0 = 0,
x0 > 0) removes rotations about the line of sight, vy0 > 0 the left-right mirror, z0 > 0 the front-back mirror
and dirz >= 0 the sign of the halo axis. A rotated or mirrored example would either be identical or lie outside
that prior, so none is generated; the only randomness added to a simulation is the observation itself (which
bins are seen, with which errors, and the noise drawn from the likelihood).

  * the track is given in the data frame of StreaMAX-fit: angle minus the median angle of the track;
  * the parameters are given in the frame of the paper, in which the progenitor sits at angle 0.

The paper places the progenitor at the centre of the observed track (or at the visible progenitor), and so do
the mock tracks by default (`prog_position = centre`). With `prog_position = random` the progenitor is put on
a random bin edge of the track instead and its angle is given to the network for a fraction `p_known` of the
examples; that is the route to inferring or conditioning on the progenitor position later, not a rotation of
the parameters. `to_data_frame` rotates samples into a frame in which the progenitor is elsewhere, for plots.

MOCK OBSERVATION MODEL (the augmentation, `mock_observation`)
------------------------------------------------------------
A simulation gives the model track on a fixed grid: median radius m_b, 16-84 half-width h_b and particle
count n_b in 10 deg bins of the unwrapped angle, the progenitor on the edge between bins 143 and 144. From one
simulation every call draws a new observation:

  1. data errors sigma_b on the whole grid, independent of the parameters;
  2. the bins where model and data can be compared, by the two guards of the paper: n_b >= n_min and
     h_b^2 / n_b <= sigma_b^2 / var_ratio (the noise of the model track at most a third of the data error);
  3. the run of such bins around the progenitor: a bins to its left, b to its right;
  4. the observed window: S bins, l of them to the left of the progenitor, S ~ p0(S) and l = S / 2 (or
     l ~ U{0..S} with prog_position = random), restricted to windows inside the run. Z = the probability that an unrestricted draw falls inside the run. An epoch
     draws simulation i with probability proportional to Z_i (`SelectionDataset`), which together with the
     restricted draw (and a weight in the loss that corrects for the scatter of the errors along the track,
     exactly 1 with the second guard off) is rejection sampling of p0: the training distribution is
     prior(theta) p0(S, l) 1[window inside the run of theta]  and carries no hidden preference for streams
     whose run is short or long. The indicator is the one the paper's likelihood has ("the modelled stream
     must cover the angular range of the data"). p0(S) is set so that the track lengths S of the training
     examples follow `span_prior` (log-uniform: 41 % of 24 bins or more, as for the observed tracks); S is
     observed, so this choice does not enter the posterior;
  5. gaps: interior bins dropped at random (10 of the 32 observed tracks have missing bins);
  6. sigma_sys from its prior, and the radii  r_b ~ N(m_b, sigma_b^2 + sigma_sys^2)   (eq. 5 of the paper);
  7. the data frame: angles centred on the median of the observed angles.

LIKELIHOOD (written out for importance sampling, `log_likelihood`)
-----------------------------------------------------------------
For data {r_i, sigma_i} in N angular bins and the model track m_i(theta) in the same bins (eq. 4):

    log L = -1/2 sum_i [ (m_i - r_i)^2 / (sigma_i^2 + sigma_sys^2) + log(2 pi (sigma_i^2 + sigma_sys^2)) ]

with L = 0 if a bin has fewer than n_min model particles or h_i^2 / n_i > sigma_i^2 / var_ratio.
With prog_position = random and the position unknown, the likelihood is the average over its S + 1 positions
(`log_likelihood_marginal`). Self-normalised importance weights of posterior samples are

    w  propto  N(zeta; 0, I) L(data | zeta) / q(zeta | data).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


# ======================================================================================================
# Device: must run before jax / keras are imported
# ======================================================================================================
def pick_device(cpu: bool) -> str:
    """CPU if asked or if the machine has no NVIDIA driver; otherwise one free GPU chosen by autocvd."""
    os.environ.setdefault("KERAS_BACKEND", "jax")
    if cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu (--cpu)"
    if os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES"):
        return f"gpu {os.environ['CUDA_VISIBLE_DEVICES']} (Slurm allocation)"
    if shutil.which("nvidia-smi") is None:
        os.environ["JAX_PLATFORMS"] = "cpu"
        return "cpu (no NVIDIA driver on this machine)"
    from autocvd import autocvd

    ids = autocvd(num_gpus=1)  # waits for a free card and sets CUDA_VISIBLE_DEVICES
    return f"gpu {ids} (autocvd)"


DEVICE = pick_device("--cpu" in sys.argv)

import h5py  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
from jax.scipy.special import logsumexp, ndtri  # noqa: E402
from scipy import special  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PARAM_NAMES = ["logM", "Rs", "dirx", "diry", "dirz", "logm", "rs", "x0", "z0", "vx0", "vy0", "vz0", "time", "sig"]
N_PAR = len(PARAM_NAMES)
I_SIG = PARAM_NAMES.index("sig")
N_TOKEN_FEATURES = 5  # angle, observed flag, r / rbar - 1, sigma / rbar, log10 sigma
N_COND = 5  # dissolved, position known, position, log10 rbar, number of points
KMS_TO_KPCGYR = 1.0227121650537077


# ======================================================================================================
# Settings
# ======================================================================================================
@dataclass
class Config:
    # --- data
    out: str = "outputs/extragalactic/C26_test"
    subdir: str = "train_v1"  # simulation set: <out>/data/<subdir>/shard_*.h5
    ordering: str = "c26"  # track ordering to train on: c26 (track/) | gibbons14 (the ordering of 2026b)
    max_shards: int = 0  # 0 = all shards on disk
    val_frac: float = 0.05  # last fraction of the usable streams, held out
    n_val_draws: int = 2  # mock observations per held-out stream (fixed for the whole run)
    include_unbound: bool = True  # the paper keeps the raw prior, so unbound progenitors stay in
    obs_dir: str = "outputs/extragalactic/C26_test/data/strrings"
    # --- mock observation: comparable bins (the two guards of 2026b sec. 3.3)
    n_min: int = 10  # model particles per bin; 3 of 10,000 in the paper = 9 of the 30,000 simulated here
    var_ratio: float = 9.0  # the paper's guard: model variance <= data variance / 9; 0 = guard off
    # --- mock observation: window
    span_min: int = 6  # bins; the shortest observed track has 7 points
    span_max: int = 64  # bins = tokens; the longest observed track has 48 points
    span_prior: str = "loguniform"  # distribution of the track length S of the training examples: loguniform | uniform
    span_balance: float = 1.0  # 1 = S follows span_prior exactly; 0 = p0(S) = span_prior, long tracks then rare
    selection_weights: bool = True  # draw simulations in proportion to Z (no hidden selection on the run length)
    # --- mock observation: errors, gaps, systematic term, progenitor position
    err_level: tuple = (0.1, 5.0)  # kpc; track level of sigma_data, log-uniform (observed medians 0.2-2.9)
    err_scatter: tuple = (0.2, 1.0)  # log-normal scatter of sigma_data along a track (observed 0.4-0.95)
    err_clip: tuple = (0.03, 20.0)  # kpc
    err_strata: int = 8  # strata of the track error level used to draw streams when the model-noise guard is on
    gap_prob: float = 0.3  # fraction of mock tracks with missing interior bins
    gap_rate_max: float = 0.25  # each interior bin of such a track is missing with a rate U(0, this)
    sig_sys_max: float = 25.0  # kpc; sigma_sys ~ U(0, sig_sys_max)
    sig_sys_power: float = 1.0  # training draws sigma_sys = max * u^power; 1 = the prior of the paper
    prog_position: str = "centre"  # progenitor along the mock track: centre (the middle of the window, the
    # assumption of 2026b; nothing is drawn) | random (any of the S + 1 bin edges, for a later step in which
    # the position is inferred or passed as a condition)
    p_known: float = 0.5  # fraction of examples in which the progenitor position is given to the network
    # --- networks
    summary: str = "transformer"  # transformer | timeseries | mlp
    summary_dim: int = 32
    embed_dim: int = 128  # transformer width / GRU width / MLP width
    n_blocks: int = 3  # transformer blocks / MLP layers
    n_heads: int = 4
    dropout: float = 0.05
    flow_transform: str = "spline"  # spline | affine
    flow_depth: int = 6
    flow_width: int = 256
    flow_layers: int = 2
    # --- training
    name: str = "npe_v1"  # run directory: <out>/models/<name>/
    epochs: int = 100
    batch_size: int = 512
    lr: float = 5e-4
    weight_decay: float = 1e-4
    warmup_epochs: float = 1.0
    seed: int = 2607
    log_every: int = 200  # steps between progress lines in train.log
    resume: bool = False
    cpu: bool = False
    # --- sampling
    n_samples: int = 5000
    prog_mode: str = "centre"  # centre (2026b: progenitor at the middle of the track) | free (marginalised)
    dissolved: bool = True  # 2026b for streams with no visible progenitor: mass ramped to zero

    def data_dir(self) -> Path:
        return REPO / self.out / "data" / self.subdir

    def run_dir(self) -> Path:
        return REPO / self.out / "models" / self.name


def parse_args(argv=None) -> tuple[str, Config]:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=["check", "train", "sample"])
    for f in dataclasses.fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        elif isinstance(f.default, tuple):
            p.add_argument(flag, type=lambda s: tuple(float(x) for x in s.split(",")), default=f.default)
        else:
            p.add_argument(flag, type=type(f.default), default=f.default)
    a = vars(p.parse_args(argv))
    return a.pop("command"), Config(**a)


# ======================================================================================================
# Parameters <-> latent coordinates in which the prior is a standard normal
# ======================================================================================================
@dataclass(frozen=True)
class Prior:
    """Table A1 of 2026b, read from the configuration stored with the simulations (never retyped)."""

    kind: tuple  # per parameter: "U" uniform(a, b) | "N" normal(0, a) | "H" half-normal(0, a), > 0
    a: tuple
    b: tuple

    @classmethod
    def from_sim_config(cls, sc: dict, sig_sys_max: float) -> "Prior":
        U = lambda ab: ("U", float(ab[0]), float(ab[1]))
        rows = [U(sc["log10_M_halo"]), U(sc["R_s"]), ("N", 1.0, 0.0), ("N", 1.0, 0.0), ("H", 1.0, 0.0),
                U(sc["log10_m_prog"]), U(sc["r_s"]), ("H", sc["sigma_pos"], 0.0), ("H", sc["sigma_pos"], 0.0),
                ("N", sc["sigma_vel"], 0.0), ("H", sc["sigma_vel"], 0.0), ("N", sc["sigma_vel"], 0.0),
                U(sc["t_int"]), ("U", 0.0, sig_sys_max)]  # fmt: skip
        return cls(*zip(*rows))

    def to_latent(self, x: np.ndarray, cols=None, eps: float = 1e-7) -> np.ndarray:
        """zeta = Phi^-1(F(x)) column by column (float64). `cols` selects a subset of the 14 parameters."""
        cols = range(N_PAR) if cols is None else cols
        x = np.asarray(x, np.float64)
        out = np.empty_like(x)
        for j, k in enumerate(cols):
            kind, a, b = self.kind[k], self.a[k], self.b[k]
            if kind == "N":
                out[..., j] = x[..., j] / a
            else:
                u = (x[..., j] - a) / (b - a) if kind == "U" else special.erf(x[..., j] / (a * math.sqrt(2.0)))
                out[..., j] = special.ndtri(np.clip(u, eps, 1.0 - eps))
        return out

    def from_latent(self, z: np.ndarray, cols=None) -> np.ndarray:
        cols = range(N_PAR) if cols is None else cols
        z = np.asarray(z, np.float64)
        out = np.empty_like(z)
        for j, k in enumerate(cols):
            kind, a, b = self.kind[k], self.a[k], self.b[k]
            if kind == "N":
                out[..., j] = z[..., j] * a
            elif kind == "U":
                out[..., j] = a + (b - a) * special.ndtr(z[..., j])
            else:  # half-normal: F(x) = erf(x / (a sqrt 2)) = 2 Phi(x / a) - 1
                out[..., j] = a * special.ndtri(0.5 * (1.0 + special.ndtr(z[..., j])))
        return out


def flattening(vec: np.ndarray, q_lim=(0.5, 1.5)) -> np.ndarray:
    """q from the orientation vector: its length r maps to q through the Maxwell cdf (StreaMAX-fit `get_q`)."""
    r = np.linalg.norm(np.asarray(vec, np.float64), axis=-1)
    u = special.erf(r / math.sqrt(2.0)) - math.sqrt(2.0 / math.pi) * r * np.exp(-0.5 * r**2)
    return q_lim[0] + (q_lim[1] - q_lim[0]) * u


def to_streamax_fit(params: np.ndarray) -> np.ndarray:
    """Same 14 parameters with velocities in kpc/Gyr, the unit StreaMAX-fit passes to the simulator."""
    out = np.array(params, np.float64)
    out[..., 9:12] *= KMS_TO_KPCGYR
    return out


def to_data_frame(params: np.ndarray, phi_prog_deg) -> np.ndarray:
    """Rotate samples from the progenitor frame (y0 = 0) into the data frame, where the progenitor sits at the
    angle `phi_prog_deg`. Returns (..., 15): the 14 parameters with y0 inserted after x0."""
    p = np.asarray(params, np.float64)
    c, s = np.cos(np.deg2rad(phi_prog_deg)), np.sin(np.deg2rad(phi_prog_deg))
    rot = lambda x, y: (c * x - s * y, s * x + c * y)
    dx, dy = rot(p[..., 2], p[..., 3])
    x0, y0 = rot(p[..., 7], 0.0)
    vx, vy = rot(p[..., 9], p[..., 10])
    return np.stack([p[..., 0], p[..., 1], dx, dy, p[..., 4], p[..., 5], p[..., 6], x0, y0, p[..., 8], vx, vy,
                     p[..., 11], p[..., 12], p[..., 13]], axis=-1)  # fmt: skip


# ======================================================================================================
# Simulation set
# ======================================================================================================
@dataclass
class Grid:
    n_bins: int
    bin_deg: float
    i0: int  # the progenitor sits on the left edge of bin i0

    def centres(self) -> np.ndarray:
        return (np.arange(self.n_bins) - self.i0 + 0.5) * self.bin_deg


def run_extents(valid: np.ndarray, i0: int) -> tuple[np.ndarray, np.ndarray]:
    """Lengths (a, b) of the unbroken run of valid bins to the left / right of the progenitor (numpy)."""
    left, right = valid[:, :i0][:, ::-1], valid[:, i0:]
    a = np.where(left.all(1), left.shape[1], np.argmin(left, 1))
    b = np.where(right.all(1), right.shape[1], np.argmin(right, 1))
    return a, b


def load_simulations(cfg: Config) -> dict:
    """Read the shards and keep the streams a mock observation can be drawn from.

    Dropped: integration failures (status bit 16), streams that lost more than 1 % of their particles at the
    step cap, and streams whose run of bins with >= n_min particles around the progenitor is shorter than
    `span_min`. Unbound progenitors are kept unless --no-include-unbound.
    """
    files = sorted(cfg.data_dir().glob("shard_*.h5"))
    if not files:
        raise SystemExit(f"no shards in {cfg.data_dir()}")
    files = files[: cfg.max_shards] if cfg.max_shards else files
    with h5py.File(files[0], "r") as f:
        sc = json.loads(f.attrs["config"])
        groups = {f[g].attrs["ordering"]: g for g in ("track", "track_alt") if g in f}
        names = json.loads(f.attrs["theta_names"])
    if cfg.ordering not in groups:
        raise SystemExit(f"ordering {cfg.ordering!r} is not stored in {files[0].name} (has {list(groups)})")
    assert names == ["log10_M_halo", "R_s", "q", "cos_i", "psi", "log10_m_prog", "r_s", "x0", "z0", "vx0", "vy0",
                     "vz0", "t_int"], names  # fmt: skip
    g = groups[cfg.ordering]
    grid = Grid(n_bins=int(round(2 * sc["phi_max_deg"] / sc["bin_deg"])), bin_deg=float(sc["bin_deg"]),
                i0=int(round(sc["phi_max_deg"] / sc["bin_deg"])))  # fmt: skip
    cols = dict(index="index", theta="theta", vec="c26_vec", status="status", n_bad="diag/n_bad",
                dissolved="prog/dissolved", count=f"{g}/count", r_med=f"{g}/r_med", r_hw=f"{g}/r_hw")  # fmt: skip
    raw = {k: [] for k in cols}
    for p in files:
        with h5py.File(p, "r") as f:
            for k, path in cols.items():
                raw[k].append(f[path][:])
    raw = {k: np.concatenate(v) for k, v in raw.items()}
    n_all = len(raw["index"])

    failed = (raw["status"] & 16) > 0
    lossy = raw["n_bad"] > 0.01 * sc["n_particles"]
    unbound = (raw["status"] & 1) > 0
    count = raw["count"].astype(np.float32)
    r_med = np.nan_to_num(raw["r_med"], nan=0.0).astype(np.float32)
    r_hw = np.nan_to_num(raw["r_hw"], nan=0.0).astype(np.float32)
    a, b = run_extents(count >= cfg.n_min, grid.i0)
    keep = ~failed & ~lossy & (a + b >= cfg.span_min) & (cfg.include_unbound | ~unbound)

    prior = Prior.from_sim_config(sc, cfg.sig_sys_max)
    th = raw["theta"].astype(np.float64)
    phys13 = np.concatenate([th[:, :2], raw["vec"].astype(np.float64), th[:, 5:]], axis=1)  # the 13 simulated
    zeta13 = prior.to_latent(phys13, cols=range(13)).astype(np.float32)
    stats = dict(n_streams=n_all, n_shards=len(files), failed=float(failed.mean()), lossy=float(lossy.mean()),
                 unbound=float(unbound.mean()), short_run=float(((a + b < cfg.span_min) & ~failed & ~lossy).mean()),
                 usable=float(keep.mean()), n_usable=int(keep.sum()), n_particles=int(sc["n_particles"]),
                 ordering=cfg.ordering, physics_hash=sc.get("physics_hash", ""))  # fmt: skip
    design = window_design(cfg, count[keep], r_hw[keep], grid.i0)
    data = dict(zeta13=zeta13[keep], count=count[keep], r_med=r_med[keep], r_hw=r_hw[keep],
                dissolved=raw["dissolved"][keep].astype(np.float32), z_strata=design["Z"].astype(np.float32))  # fmt: skip
    return dict(data=data, index=raw["index"][keep], phys13=phys13[keep], q=th[keep, 2], zeta13_all=zeta13,
                prior=prior, grid=grid, stats=stats, sim_config=sc, design=design)  # fmt: skip


def split_train_val(sims: dict, val_frac: float) -> tuple[dict, dict]:
    n = len(sims["index"])
    n_val = max(1, int(round(val_frac * n)))
    take = lambda sl: {k: v[sl] for k, v in sims["data"].items()}
    return take(slice(0, n - n_val)), take(slice(n - n_val, n))


# ======================================================================================================
# Mock observations (JAX; runs on the GPU when there is one)
# ======================================================================================================
def window_design(cfg: Config, count: np.ndarray, r_hw: np.ndarray, i0: int) -> dict:
    """The window prior p0(S, l) and the acceptance probabilities Z of every simulation.

    A stream with a comparable bins left and b right of its progenitor covers a window (S, l) if l <= a and
    S - l <= b. Which bins are comparable depends on the data errors when the model-noise guard is on, so the
    track level of sigma_data is split into `err_strata` log-uniform strata and everything is computed per
    stratum k (at its mid level, errors taken flat along the track):

        frac[k, i, S]  share of the S + 1 progenitor positions stream i covers
        acc[S]         its mean over streams and strata
        p0(S)          span_prior(S) / acc(S)^span_balance   (span_balance = 1: accepted lengths follow span_prior)
        Z[i, k]        sum_S p0(S) frac[k, i, S], the acceptance probability of stream i in stratum k

    Z is floored at 5 % of its value without the guard: the scatter of sigma_data along a track can make bins
    comparable that the flat level does not, and those draws must stay possible (their weight corrects for it).
    logp[S, l] = log p0(S) - log(S + 1) on S = span_min..span_max, l = 0..span_max (-inf where l > S). With
    prog_position = centre only the edges next to the middle of the window are allowed (l = S/2).
    """
    S = np.arange(cfg.span_min, cfg.span_max + 1)
    L = np.arange(0, cfg.span_max + 1)
    K = cfg.err_strata if cfg.var_ratio > 0 else 1
    edges = np.logspace(math.log10(cfg.err_level[0]), math.log10(cfg.err_level[1]), K + 1)

    centre = cfg.prog_position == "centre"
    allowed = (L[None] == S[:, None] // 2) | (L[None] == (S[:, None] + 1) // 2) if centre else L[None] <= S[:, None]

    def frac_of(valid):
        a, b = run_extents(valid, i0)
        if centre:  # the one or two edges next to the middle of the window
            cov = lambda l: (l[None] <= a[:, None]) & (S[None] - l[None] <= b[:, None])
            return (0.5 * (cov(S // 2).astype(np.float32) + cov((S + 1) // 2)))
        n_pos = np.minimum(S[None], a[:, None]) - np.maximum(0, S[None] - b[:, None]) + 1
        return (np.clip(n_pos, 0, None) / (S[None] + 1.0)).astype(np.float32)

    base = count >= cfg.n_min
    noise_var = r_hw**2 / np.maximum(count, 1.0) * cfg.var_ratio
    fracs = [frac_of(base & (noise_var <= lv**2)) for lv in np.sqrt(edges[:-1] * edges[1:])] if cfg.var_ratio > 0 else [frac_of(base)]
    acc = np.mean([f.mean(0, dtype=np.float64) for f in fracs], axis=0)
    target = 1.0 / S if cfg.span_prior == "loguniform" else np.ones(len(S))
    pS = target / np.maximum(acc, 1e-4) ** cfg.span_balance
    pS = pS / pS.sum()
    logp = np.log(pS)[:, None] - np.log(allowed.sum(1))[:, None] + np.where(allowed, 0.0, -np.inf)
    Z = np.stack([f.astype(np.float64) @ pS for f in fracs], axis=1)
    Z = np.maximum(Z, 0.05 * (frac_of(base).astype(np.float64) @ pS)[:, None])
    return dict(S=S, L=L, logp=logp.astype(np.float32), pS=pS, acc=acc, Z=Z, realised=pS * acc / np.sum(pS * acc))


def masked_median(x, m):
    """Median of x where m, as np.median (mean of the two middle values for an even count)."""
    xs = jnp.sort(jnp.where(m, x, jnp.inf))
    n = jnp.sum(m)
    return 0.5 * (xs[(n - 1) // 2] + xs[n // 2])


def featurise(phi, r, sig, obs, dissolved, known, phi_prog):
    """Network inputs from one track on the token grid. Used for mock and for real observations alike.

    phi [deg, data frame], r, sig [kpc], obs (bool): arrays of length n_tokens; unobserved tokens are zeroed.
    tokens: (angle / 90 deg, observed, r / rbar - 1, sigma / rbar, log10 sigma); rbar = mean observed radius.
    cond:   (dissolved, position known, phi_prog / 90 deg if known, log10(rbar / 30 kpc), points / 32).
    """
    w = obs.astype(jnp.float32)
    n = jnp.maximum(jnp.sum(w), 1.0)
    rbar = jnp.clip(jnp.sum(w * r) / n, 3.0, 1000.0)
    sig = jnp.maximum(sig, 1e-3)
    tok = jnp.stack([phi / 90.0, w, w * jnp.clip(r / rbar - 1.0, -20.0, 20.0), w * jnp.minimum(sig / rbar, 20.0),
                     w * jnp.log10(sig)], axis=-1)  # fmt: skip
    k = jnp.asarray(known, jnp.float32)
    cond = jnp.stack([jnp.asarray(dissolved, jnp.float32), k, k * phi_prog / 90.0, jnp.log10(rbar / 30.0), n / 32.0])
    return tok.astype(jnp.float32), cond.astype(jnp.float32)


def make_mock_observation(cfg: Config, grid: Grid, design: dict):
    """Returns f(key, count, r_med, r_hw, dissolved, z_strata) -> dict for ONE simulation (vmap it over a batch)."""
    S_grid, L_grid, logp0 = (jnp.asarray(design[k]) for k in ("S", "L", "logp"))
    T, NB, I0, BIN = cfg.span_max, grid.n_bins, grid.i0, grid.bin_deg
    nL = len(L_grid)
    j = jnp.arange(T)

    lo, hi = math.log10(cfg.err_level[0]), math.log10(cfg.err_level[1])

    def mock(key, count, r_med, r_hw, dissolved, z_strata):
        k = jr.split(key, 11)
        # 1. data errors on the whole grid. The track level is log-uniform; its stratum is drawn in proportion
        #    to the acceptance probability of this stream there, which step 4 divides out again.
        K = z_strata.shape[0]
        kk = jr.categorical(k[0], jnp.log(z_strata))
        level = 10.0 ** (lo + (kk + jr.uniform(k[10])) * (hi - lo) / K)
        tau = jr.uniform(k[1], (), minval=cfg.err_scatter[0], maxval=cfg.err_scatter[1])
        sig_grid = jnp.clip(level * jnp.exp(tau * jr.normal(k[2], (NB,))), cfg.err_clip[0], cfg.err_clip[1])
        # 2. bins where the model can be compared with data
        valid = count >= cfg.n_min
        if cfg.var_ratio > 0:
            valid &= r_hw**2 / jnp.maximum(count, 1.0) * cfg.var_ratio <= sig_grid**2
        # 3. unbroken run around the progenitor
        left, right = valid[:I0][::-1], valid[I0:]
        a = jnp.where(jnp.all(left), I0, jnp.argmin(left))
        b = jnp.where(jnp.all(right), NB - I0, jnp.argmin(right))
        # 4. window (S, l) ~ p0 restricted to windows inside the run; Z = P0(window inside the run).
        #    The stream and its error stratum were drawn with probability proportional to z_strata (Z for flat
        #    errors at the stratum level), so the weight left for the loss is Z / z_strata: exactly 1 with the
        #    model-noise guard off, close to 1 with it on.
        inside = (L_grid[None, :] <= a) & (S_grid[:, None] - L_grid[None, :] <= b)
        logits = jnp.where(inside, logp0, -jnp.inf).ravel()
        Z = jnp.sum(jnp.exp(logits))
        pick = jr.categorical(k[3], logits)
        S, l = S_grid[pick // nL], L_grid[pick % nL]
        start = I0 - l
        # 5. gaps: interior bins missing at random
        rate = jnp.where(jr.uniform(k[4]) < cfg.gap_prob, jr.uniform(k[5]) * cfg.gap_rate_max, 0.0)
        obs = (j < S) & ~((jr.uniform(k[6], (T,)) < rate) & (j > 0) & (j < S - 1))
        # 6. systematic term and noisy radii:  r ~ N(m, sigma_data^2 + sigma_sys^2)
        bins = jnp.clip(start + j, 0, NB - 1)
        m, sig = r_med[bins], sig_grid[bins]
        u = jr.uniform(k[7], (), minval=1e-6, maxval=1.0 - 1e-6)
        sig_sys = cfg.sig_sys_max * u**cfg.sig_sys_power
        r = m + jnp.sqrt(sig**2 + sig_sys**2) * jr.normal(k[8], (T,))
        # 7. data frame: angle minus the median observed angle (the progenitor is at 0 in the simulation)
        phi_sim = (start + j - I0 + 0.5) * BIN
        med = masked_median(phi_sim, obs)
        phi, phi_prog = phi_sim - med, -med
        known = jr.bernoulli(k[9], cfg.p_known)
        tokens, cond = featurise(phi, r, sig, obs, dissolved, known, phi_prog)
        zeta_sig = ndtri(jnp.clip(sig_sys / cfg.sig_sys_max, 1e-6, 1.0 - 1e-6))  # latent of sig under U(0, max)
        weight = Z / z_strata[kk] if cfg.selection_weights else (Z > 0).astype(jnp.float32)
        return dict(tokens=tokens, mask=obs, cond=cond, weight=weight, zeta_sig=zeta_sig,
                    # everything below is for the likelihood, the tests and the plots
                    r=r, sig=sig, phi=phi, phi_prog=phi_prog, known=known, sig_sys=sig_sys, S=S, l=l, a=a, b=b,
                    m=m, hw=r_hw[bins], n=count[bins])  # fmt: skip

    return mock


class MockObservations:
    """The augmentation handed to `bf.OfflineDataset`: simulations in, one fresh mock observation each out.

    The batch of numpy arrays is moved to the JAX device, transformed by one jitted call and moved back, so
    with a GPU the whole augmentation runs there. Each call uses its own key (seed, call number).
    """

    def __init__(self, cfg: Config, grid: Grid, design: dict, seed: int, extras: bool = False, weight_norm: float = 1.0):
        self.cfg, self.extras, self.weight_norm = cfg, extras, weight_norm  # weights are divided by weight_norm
        self.fn = jax.jit(jax.vmap(make_mock_observation(cfg, grid, design)))
        self.key = jr.key(seed)
        self.calls = 0

    def __call__(self, batch: dict) -> dict:
        n = len(batch["zeta13"])
        keys = jr.split(jr.fold_in(self.key, self.calls), n)
        self.calls += 1
        o = self.fn(keys, jnp.asarray(batch["count"]), jnp.asarray(batch["r_med"]), jnp.asarray(batch["r_hw"]),
                    jnp.asarray(batch["dissolved"]), jnp.asarray(batch["z_strata"]))  # fmt: skip
        o = {k: np.asarray(v) for k, v in o.items()}
        out = dict(zeta=np.concatenate([batch["zeta13"], o["zeta_sig"][:, None]], axis=1).astype(np.float32),
                   tokens=o["tokens"], mask=o["mask"], cond=o["cond"],
                   weight=(o["weight"] / self.weight_norm).astype(np.float32))  # fmt: skip
        if self.extras:
            out["extras"] = o
        return out


def encode_observation(theta_deg, r, r_err, cfg: Config, grid: Grid, *, dissolved: bool, phi_prog_deg=None) -> dict:
    """A real track -> the network inputs, with exactly the conventions of the mock observations.

    theta_deg must be increasing with steps that are multiples of the bin width (missing bins are allowed).
    phi_prog_deg: progenitor angle in the data frame (angle minus median angle); None = not given.
    """
    th = np.asarray(theta_deg, np.float64)
    phi = th - np.median(th)
    kk = np.rint((phi - phi[0]) / grid.bin_deg).astype(int)
    if np.abs(phi - phi[0] - kk * grid.bin_deg).max() > 1e-3 * grid.bin_deg or np.any(np.diff(kk) <= 0):
        raise ValueError("track angles are not on a regular grid of the bin width")
    S, T = int(kk[-1]) + 1, cfg.span_max
    if S > T:
        raise ValueError(f"track spans {S} bins, more than span_max = {T}")
    rg, sg, og = np.zeros(T, np.float32), np.ones(T, np.float32), np.zeros(T, bool)
    rg[kk], sg[kk], og[kk] = r, r_err, True
    phig = (phi[0] + grid.bin_deg * np.arange(T)).astype(np.float32)
    known = phi_prog_deg is not None
    tok, cond = featurise(jnp.asarray(phig), jnp.asarray(rg), jnp.asarray(sg), jnp.asarray(og), float(dissolved),
                          known, float(phi_prog_deg or 0.0))  # fmt: skip
    return dict(tokens=np.asarray(tok), mask=og, cond=np.asarray(cond), r=rg, sig=sg, phi=phig, S=S,
                in_support=bool(cfg.span_min <= S and cfg.err_clip[0] <= np.min(r_err) and np.max(r_err) <= cfg.err_clip[1]))  # fmt: skip


# ======================================================================================================
# Likelihood and importance weights
# ======================================================================================================
def log_likelihood(r, sig, obs, m, hw, n, sig_sys, n_min: float = 3.0, var_ratio: float = 9.0):
    """Eq. 4 of 2026b for one track, with the two guards of its sec. 3.3.

    r, sig, obs : data radii, their errors [kpc] and the observed flag on the token grid
    m, hw, n    : model track in the same bins (median radius, 16-84 half-width, particle count)
    sig_sys     : systematic term [kpc]
    Returns -inf where a guard fails (the paper returns a large negative number to help the sampler; for
    importance weights the two are the same). var_ratio = 0 switches the second guard off.
    """
    var = sig**2 + sig_sys**2
    ll = -0.5 * jnp.sum(jnp.where(obs, (m - r) ** 2 / var + jnp.log(2.0 * jnp.pi * var), 0.0))
    ok = jnp.all(~obs | (n >= n_min))
    ok &= jnp.all(~obs | (hw**2 / jnp.maximum(n, 1.0) * var_ratio <= sig**2))
    return jnp.where(ok, ll, -jnp.inf)


def log_likelihood_offsets(r, sig, obs, S, r_med, r_hw, count, sig_sys, i0: int, n_min=3.0, var_ratio=9.0):
    """log L for every position of the progenitor: entry l puts l bins of the window to its left.

    r_med, r_hw, count: model track on the full simulation grid. Entries l > S are -inf.
    """
    T, NB = r.shape[0], r_med.shape[0]
    j = jnp.arange(T)

    def one(l):
        bins = jnp.clip(i0 - l + j, 0, NB - 1)
        inside = jnp.all(~obs | ((i0 - l + j >= 0) & (i0 - l + j < NB)))
        ll = log_likelihood(r, sig, obs, r_med[bins], r_hw[bins], count[bins], sig_sys, n_min, var_ratio)
        return jnp.where(inside & (l <= S), ll, -jnp.inf)

    return jax.vmap(one)(jnp.arange(T + 1))


def log_likelihood_marginal(r, sig, obs, S, r_med, r_hw, count, sig_sys, i0: int, n_min=3.0, var_ratio=9.0):
    """Progenitor position unknown: average of L over its S + 1 positions, the prior p0(l | S) of the mocks."""
    ll = log_likelihood_offsets(r, sig, obs, S, r_med, r_hw, count, sig_sys, i0, n_min, var_ratio)
    return logsumexp(ll) - jnp.log(S + 1.0)


def log_prior_latent(zeta):
    """The prior in latent coordinates: N(0, I)."""
    return -0.5 * jnp.sum(zeta**2, axis=-1) - 0.5 * zeta.shape[-1] * math.log(2.0 * math.pi)


def importance_weights(zeta, log_q, log_like) -> dict:
    """Self-normalised importance weights of samples zeta ~ q(. | data) towards prior x likelihood.

    Returns the normalised weights, the effective sample size and the log evidence estimate.
    """
    lw = np.asarray(log_prior_latent(jnp.asarray(zeta))) + np.asarray(log_like) - np.asarray(log_q)
    lw = np.where(np.isfinite(lw), lw, -np.inf)
    top = lw.max()
    if not np.isfinite(top):
        return dict(weights=np.zeros_like(lw), ess=0.0, efficiency=0.0, log_evidence=-np.inf)
    w = np.exp(lw - top)
    return dict(weights=w / w.sum(), ess=float(w.sum() ** 2 / (w**2).sum()),
                efficiency=float(w.sum() ** 2 / (w**2).sum() / len(w)), log_evidence=float(top + np.log(w.mean())))  # fmt: skip


# ======================================================================================================
# Networks
# ======================================================================================================
import keras  # noqa: E402

import bayesflow as bf  # noqa: E402


@keras.saving.register_keras_serializable(package="streams")
class FlatMLP(bf.networks.SummaryNetwork):
    """Baseline summary network: the token grid flattened into one vector, then a residual MLP."""

    def __init__(self, summary_dim: int = 32, widths=(256, 256, 256), dropout: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.summary_dim, self.widths, self.dropout = summary_dim, tuple(widths), dropout
        self.mlp = bf.networks.MLP(widths=self.widths, dropout=dropout)
        self.head = keras.layers.Dense(summary_dim)

    def call(self, x, training: bool = False, **kwargs):
        x = keras.ops.reshape(x, (keras.ops.shape(x)[0], -1))
        return self.head(self.mlp(x, training=training))

    def get_config(self):
        return super().get_config() | dict(summary_dim=self.summary_dim, widths=self.widths, dropout=self.dropout)


USES_MASK = {"transformer": True, "timeseries": False, "mlp": False}


def build_summary(cfg: Config):
    if cfg.summary == "transformer":  # attention over the bins; the angle is the explicit time axis
        return bf.networks.TimeSeriesTransformer(
            summary_dim=cfg.summary_dim, embed_dims=(cfg.embed_dim,) * cfg.n_blocks,
            num_heads=(cfg.n_heads,) * cfg.n_blocks, dropout=cfg.dropout, time_embedding="time2vec",
            time_embed_dim=16, time_axis=0)  # fmt: skip
    if cfg.summary == "timeseries":  # convolutions + bidirectional GRU along the angle
        return bf.networks.TimeSeriesNetwork(summary_dim=cfg.summary_dim, filters=(32, 64), kernel_sizes=(3, 3),
                                             recurrent_dim=cfg.embed_dim, bidirectional=True, dropout=cfg.dropout)  # fmt: skip
    if cfg.summary == "mlp":
        return FlatMLP(summary_dim=cfg.summary_dim, widths=(2 * cfg.embed_dim,) * cfg.n_blocks, dropout=cfg.dropout)
    raise SystemExit(f"unknown summary network {cfg.summary!r}")


def build_adapter(cfg: Config):
    ad = bf.Adapter().convert_dtype("float64", "float32")
    ad = ad.rename("zeta", "inference_variables").rename("cond", "inference_conditions")
    ad = ad.rename("tokens", "summary_variables").rename("weight", "sample_weight")
    keep = ["inference_variables", "inference_conditions", "summary_variables", "sample_weight"]
    if USES_MASK[cfg.summary]:
        ad = ad.rename("mask", "summary_mask")
        keep.append("summary_mask")
    return ad.keep(keep)


def build_estimator(cfg: Config):
    """Normalizing flow (coupling flow) on the 14 latent parameters, conditioned on the track summary and on
    (dissolved, progenitor position known, position, radius scale, number of points)."""
    tk = dict(default_domain=(-5.0, 5.0, -5.0, 5.0)) if cfg.flow_transform == "spline" else None
    flow = bf.networks.CouplingFlow(depth=cfg.flow_depth, transform=cfg.flow_transform, transform_kwargs=tk,
                                    subnet_kwargs=dict(widths=(cfg.flow_width,) * cfg.flow_layers, dropout=cfg.dropout))  # fmt: skip
    return bf.BasicEstimator(inference_network=flow, summary_network=build_summary(cfg), adapter=build_adapter(cfg),
                             standardize=None)  # fmt: skip


# ======================================================================================================
# Training
# ======================================================================================================
class Journal(keras.callbacks.Callback):
    """train.log (one line every `log_every` steps and one per epoch), history.csv, last.keras after every
    epoch and best.keras whenever the held-out loss improves."""

    def __init__(self, run: Path, epochs: int, steps: int, log_every: int, best: float = math.inf):
        super().__init__()
        self.run, self.epochs, self.steps, self.log_every, self.best = run, epochs, steps, log_every, best
        self.t0 = self.t_epoch = time.time()
        self.first_epoch = None

    def say(self, msg: str) -> None:
        line = f"{time.strftime('%F %T')}  {msg}"
        with open(self.run / "train.log", "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def on_epoch_begin(self, epoch, logs=None):
        self.t_epoch = time.time()
        self.epoch = epoch
        self.first_epoch = epoch if self.first_epoch is None else self.first_epoch

    def on_train_batch_end(self, batch, logs=None):
        if (batch + 1) % self.log_every == 0:
            rate = (batch + 1) / (time.time() - self.t_epoch)
            self.say(f"epoch {self.epoch + 1}/{self.epochs}  step {batch + 1}/{self.steps}  "
                     f"loss {float(logs['loss']):.4f}  {rate:.1f} steps/s")  # fmt: skip

    def on_epoch_end(self, epoch, logs=None):
        loss, val = float(logs["loss"]), float(logs.get("val_loss", math.nan))
        lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        self.model.save(str(self.run / "last.keras"))
        improved = val < self.best
        if improved:
            self.best = val
            self.model.save(str(self.run / "best.keras"))
            (self.run / "best.json").write_text(json.dumps(dict(epoch=epoch + 1, val_loss=val, loss=loss)))
        dt, done = time.time() - self.t_epoch, epoch + 1 - self.first_epoch
        eta = (time.time() - self.t0) / done * (self.epochs - epoch - 1)
        new = not (self.run / "history.csv").exists()
        with open(self.run / "history.csv", "a") as f:
            if new:
                f.write("epoch,loss,val_loss,lr,seconds,best\n")
            f.write(f"{epoch + 1},{loss:.6f},{val:.6f},{lr:.3e},{dt:.1f},{int(improved)}\n")
        self.say(f"epoch {epoch + 1}/{self.epochs} done  loss {loss:.4f}  val_loss {val:.4f}  best {self.best:.4f}"
                 f"{'  [saved best.keras]' if improved else ''}  lr {lr:.2e}  {dt:.0f} s  "
                 f"remaining {eta / 3600:.2f} h")  # fmt: skip


class SelectionDataset(bf.OfflineDataset):
    """`bf.OfflineDataset` whose epochs draw stream i with probability proportional to p[i] (with replacement)
    instead of visiting every stream once. With p = Z this is the rejection step of the window prior."""

    def __init__(self, data, *, p: np.ndarray, seed: int, **kwargs):
        self._p = np.asarray(p, np.float64)
        self._p = self._p / self._p.sum()
        self._rng = np.random.default_rng(seed)
        super().__init__(data, **kwargs)

    def shuffle(self) -> None:
        self.indices = self._rng.choice(self.num_samples, size=self.num_samples, p=self._p)


def selection_probability(data: dict, cfg: Config) -> np.ndarray:
    return data["z_strata"].mean(1) if cfg.selection_weights else np.ones(len(data["z_strata"]))


def fixed_observations(data: dict, cfg: Config, grid: Grid, design: dict, seed: int, n: int, extras: bool = False) -> dict:
    """n mock observations with a fixed key (streams drawn in proportion to Z): the held-out set, identical at
    every epoch."""
    p = selection_probability(data, cfg).astype(np.float64)
    pick = np.random.default_rng(seed).choice(len(p), size=n, p=p / p.sum())
    aug = MockObservations(cfg, grid, design, seed, extras=extras)
    parts = [aug({k: v[pick[s : s + 4096]] for k, v in data.items()}) for s in range(0, n, 4096)]
    cat = lambda xs: {k: (cat([x[k] for x in xs]) if isinstance(xs[0][k], dict) else np.concatenate([x[k] for x in xs]))
                      for k in xs[0]}  # fmt: skip
    out = cat(parts)
    out["pick"] = pick
    return out


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip()  # fmt: skip
    except Exception:
        return ""


def cmd_train(cfg: Config) -> None:
    run = cfg.run_dir()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "history.csv").exists() and not cfg.resume:
        raise SystemExit(f"{run} already holds a run; pass --resume to continue it or choose another --name")
    keras.utils.set_random_seed(cfg.seed)
    sims = load_simulations(cfg)
    train, val = split_train_val(sims, cfg.val_frac)
    grid = sims["grid"]
    adapter = build_adapter(cfg)
    design = sims["design"]
    val_fixed = fixed_observations(val, cfg, grid, design, cfg.seed + 1, cfg.n_val_draws * len(val["zeta13"]))
    val_fixed.pop("pick")
    # one constant puts the mean loss weight at 1 (it is exactly 1 already with the model-noise guard off), so
    # the loss stays in nats per example: 19.9 is the entropy of the prior, the value of a network that learnt nothing
    w_mean = float(val_fixed["weight"].mean())
    val_fixed["weight"] = val_fixed["weight"] / w_mean
    train_ds = SelectionDataset(train, p=selection_probability(train, cfg), seed=cfg.seed, batch_size=cfg.batch_size,
                                adapter=adapter, drop_last=True,
                                augmentations=[MockObservations(cfg, grid, design, cfg.seed, weight_norm=w_mean)])  # fmt: skip
    val_ds = bf.OfflineDataset(val_fixed, batch_size=cfg.batch_size, adapter=adapter, shuffle=False)
    steps = len(train_ds)

    done, best = 0, math.inf
    if cfg.resume and (run / "last.keras").exists():
        est = keras.saving.load_model(str(run / "last.keras"))
        hist = np.atleast_2d(np.genfromtxt(run / "history.csv", delimiter=",", skip_header=1))
        done, best = int(hist[-1, 0]), float(np.nanmin(hist[:, 2]))
    else:
        est = build_estimator(cfg)
        warm = max(1, int(cfg.warmup_epochs * steps))
        sched = keras.optimizers.schedules.CosineDecay(initial_learning_rate=cfg.lr * 0.01, warmup_target=cfg.lr,
                                                       warmup_steps=warm, alpha=0.01,
                                                       decay_steps=max(1, cfg.epochs * steps - warm))  # fmt: skip
        est.compile(optimizer=keras.optimizers.AdamW(learning_rate=sched, weight_decay=cfg.weight_decay, clipnorm=1.0))
        (run / "config.json").write_text(json.dumps(dict(config=dataclasses.asdict(cfg), data=sims["stats"],
                                                         prior=dataclasses.asdict(sims["prior"]), params=PARAM_NAMES,
                                                         grid=dataclasses.asdict(grid), commit=git_commit(),
                                                         window_prior=dict(S=design["S"].tolist(), p0=design["pS"].tolist(),
                                                                           coverage=design["acc"].tolist())), indent=1))  # fmt: skip

    journal = Journal(run, cfg.epochs, steps, cfg.log_every, best)
    journal.say(f"{'resume' if done else 'start'}  run {cfg.name}  device {DEVICE}  jax devices {jax.devices()}")
    journal.say(f"data {cfg.subdir}: {sims['stats']['n_streams']} streams in {sims['stats']['n_shards']} shards, "
                f"usable {sims['stats']['n_usable']} ({100 * sims['stats']['usable']:.1f} %), "
                f"train {len(train['zeta13'])}, held out {len(val['zeta13'])} x {cfg.n_val_draws} draws")  # fmt: skip
    journal.say(f"mean loss weight before normalisation {w_mean:.3f}; a network that learnt nothing has loss "
                f"{0.5 * N_PAR * (1 + math.log(2 * math.pi)):.2f}")  # fmt: skip
    journal.say(f"summary {cfg.summary}, flow {cfg.flow_transform} depth {cfg.flow_depth}; "
                f"{steps} steps per epoch at batch {cfg.batch_size}; epochs {done} -> {cfg.epochs}")  # fmt: skip
    est.fit(dataset=train_ds, validation_data=val_ds, epochs=cfg.epochs, initial_epoch=done, callbacks=[journal],
            verbose=0)  # fmt: skip
    journal.say(f"finished  best val_loss {journal.best:.4f}  parameters {est.count_params()}")


# ======================================================================================================
# Posterior samples for the observed tracks
# ======================================================================================================
def load_observed(cfg: Config) -> list[dict]:
    out = []
    for p in sorted((REPO / cfg.obs_dir).glob("*.npz")):
        d = np.load(p)
        scale = float(d["r_scale"]) if "r_scale" in d.files and np.isfinite(d["r_scale"]) else 1.0
        out.append(dict(name=p.stem, theta_deg=np.rad2deg(d["theta"]), r=d["r"] * scale, r_err=d["r_err"] * scale,
                        r_scale=scale))  # fmt: skip
    return out


def cmd_sample(cfg: Config) -> None:
    """Posterior samples and their log q for every observed track, written to <run>/posterior_<mode>.npz."""
    run = cfg.run_dir()
    meta = json.loads((run / "config.json").read_text())
    tc = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in meta["config"].items()})
    grid, prior = Grid(**meta["grid"]), Prior(**{k: tuple(v) for k, v in meta["prior"].items()})
    est = keras.saving.load_model(str(run / "best.keras"))
    obs = load_observed(cfg)
    enc = [encode_observation(o["theta_deg"], o["r"], o["r_err"], tc, grid, dissolved=cfg.dissolved,
                              phi_prog_deg=0.0 if cfg.prog_mode == "centre" else None) for o in obs]  # fmt: skip
    cond = dict(tokens=np.stack([e["tokens"] for e in enc]), mask=np.stack([e["mask"] for e in enc]),
                cond=np.stack([e["cond"] for e in enc]))  # fmt: skip
    zeta = est.sample(num_samples=cfg.n_samples, conditions=cond, seed=cfg.seed)["zeta"]
    rep = lambda x: np.repeat(x, cfg.n_samples, axis=0)
    log_q = est.log_prob(dict(zeta=zeta.reshape(-1, N_PAR), **{k: rep(v) for k, v in cond.items()}))
    params = prior.from_latent(zeta)
    np.savez(run / f"posterior_{cfg.prog_mode}.npz", names=np.array([o["name"] for o in obs]), param_names=PARAM_NAMES,
             zeta=zeta.astype(np.float32), params=params.astype(np.float32), q=flattening(params[..., 2:5]).astype(np.float32),
             log_q=log_q.reshape(len(obs), cfg.n_samples).astype(np.float32),
             in_support=np.array([e["in_support"] for e in enc]), r_scale=np.array([o["r_scale"] for o in obs]))  # fmt: skip
    for o, e, p in zip(obs, enc, params):
        q = flattening(p[:, 2:5])
        lo, mid, hi = np.percentile(q, [16, 50, 84])
        print(f"{o['name']:18s} points {int(e['mask'].sum()):2d}  span {e['S']:2d}  q = {mid:.2f} -{mid - lo:.2f} +{hi - mid:.2f}"
              f"  sig_sys = {np.median(p[:, I_SIG]):5.2f} kpc{'' if e['in_support'] else '   (outside the training range)'}")  # fmt: skip


# ======================================================================================================
# Checks
# ======================================================================================================
def cmd_check(cfg: Config) -> None:
    """Tests of the pieces the posterior rests on. Prints one line per test; exits non-zero on a failure."""
    from scipy import stats

    fails = []

    def check(name: str, ok: bool, **info) -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  " + "  ".join(f"{k}={v}" for k, v in info.items()), flush=True)
        if not ok:
            fails.append(name)

    sims = load_simulations(cfg)
    prior, grid, data, st, design = sims["prior"], sims["grid"], sims["data"], sims["stats"], sims["design"]
    print(f"data {cfg.subdir}: " + json.dumps(st))

    # --- transforms
    back = prior.from_latent(prior.to_latent(sims["phys13"], cols=range(13)), cols=range(13))
    rel = np.abs(back - sims["phys13"]) / (np.abs(sims["phys13"]) + 1e-3)
    check("latent transform round trip", rel.max() < 1e-4, max_rel_err=f"{rel.max():.1e}")
    pv = [stats.kstest(sims["zeta13_all"][:, k], "norm").pvalue for k in range(13)]
    check("prior is N(0,1) in latent coordinates (all streams, KS)", min(pv) > 1e-3, min_p=f"{min(pv):.3f}")
    sig = np.linspace(0.01, cfg.sig_sys_max - 0.01, 7)[:, None]
    check("sigma_sys transform round trip",
          np.allclose(prior.from_latent(prior.to_latent(sig, cols=[I_SIG]), cols=[I_SIG]), sig, rtol=1e-6))
    dq = np.abs(flattening(sims["phys13"][:, 2:5], sims["sim_config"]["q"]) - sims["q"]).max()
    check("flattening recovered from the orientation vector", dq < 2e-3, max_abs_err=f"{dq:.1e}")

    # --- mock observations
    n = 20000
    t0 = time.time()
    mo = fixed_observations(data, cfg, grid, design, seed=1, n=n, extras=True)
    t_first = time.time() - t0
    e = mo["extras"]
    sub = {k: v[mo["pick"]] for k, v in data.items()}
    obs, S, l = e["mask"], e["S"], e["l"]
    live = mo["weight"] > 0  # with the model-noise guard on, a draw can leave no window: weight 0, never used
    check("window lies inside the run of comparable bins", bool(np.all(((l <= e["a"]) & (S - l <= e["b"]))[live])),
          spans=f"{S[live].min()}-{S[live].max()}")  # fmt: skip
    j = np.arange(cfg.span_max)
    check("every observed bin has at least n_min model particles", bool(np.all(e["n"][live][obs[live]] >= cfg.n_min)))
    if cfg.var_ratio > 0:
        quiet_ok = (e["hw"] ** 2 / np.maximum(e["n"], 1.0) * cfg.var_ratio <= e["sig"] ** 2 * (1 + 1e-5))[live][obs[live]]
        check("every observed bin passes the model-noise guard", bool(np.all(quiet_ok)))
    check("observed bins are inside the window, both ends kept",
          bool(np.all(~obs | (j[None] < S[:, None])) and np.all(obs[:, 0]) and np.all(obs[np.arange(n), S - 1])))
    med = np.array([np.median(p[m]) for p, m in zip(e["phi"], obs)])
    check("angles are centred on the median observed angle", np.abs(med).max() < 1e-3, max_abs=f"{np.abs(med).max():.1e}")
    edge = e["phi"][:, 0] - 0.5 * grid.bin_deg + grid.bin_deg * l  # progenitor edge in the data frame
    check("progenitor angle matches the window offset", np.abs(edge - e["phi_prog"]).max() < 1e-3)
    w = mo["weight"]
    check("loss weights are finite, and exactly 1 with the model-noise guard off",
          bool(np.all(np.isfinite(w)) and (cfg.var_ratio > 0 or np.abs(w - 1).max() < 1e-3)),
          efficiency=f"{w.sum() ** 2 / (w**2).sum() / n:.2f}", zero_weight=f"{(w == 0).mean():.2f}", largest=f"{w.max():.1f}")  # fmt: skip
    print(f"info  share of the weighted examples with 24 bins or more: {w[S >= 24].sum() / w.sum():.3f}, 40 or more: "
          f"{w[S >= 40].sum() / w.sum():.3f}  (span_prior: {design['realised'][design['S'] >= 24].sum():.3f}, "
          f"{design['realised'][design['S'] >= 40].sum():.3f})")  # fmt: skip
    if cfg.selection_weights and cfg.span_balance == 1.0 and cfg.var_ratio == 0:
        want = float(design["realised"][design["S"] >= 24].sum())
        got = float((S >= 24).mean())
        check("track lengths of the examples follow span_prior", abs(got - want) < 0.02,
              share_of_24_bins_or_more=f"{got:.3f}", expected=f"{want:.3f}")  # fmt: skip
    z = data["z_strata"].astype(np.float64).mean(1)
    print(f"info  streams drawn in proportion to Z: the 1 % with the largest Z give {100 * np.sort(z)[-len(z) // 100:].sum() / z.sum():.0f} %"
          f" of the examples; a stream of the top 1 % is drawn {np.sort(z)[-len(z) // 100:].mean() / z.mean():.0f} times per epoch")  # fmt: skip
    check("network inputs are finite", bool(np.isfinite(mo["tokens"]).all() and np.isfinite(mo["cond"]).all()
                                             and np.isfinite(mo["zeta"]).all()),
          token_abs_max=f"{np.abs(mo['tokens']).max():.1f}")  # fmt: skip
    chi2 = np.sum(np.where(obs, (e["r"] - e["m"]) ** 2 / (e["sig"] ** 2 + e["sig_sys"][:, None] ** 2), 0.0), axis=1)
    red = chi2.sum() / obs.sum()
    check("mock radii scatter as N(m, sigma_data^2 + sigma_sys^2)", abs(red - 1.0) < 0.02, chi2_per_point=f"{red:.4f}")
    check("sigma_sys draws follow the training distribution",
          stats.kstest((e["sig_sys"] / cfg.sig_sys_max) ** (1.0 / cfg.sig_sys_power), "uniform").pvalue > 1e-3)
    known = e["known"].astype(bool)
    check("progenitor position is hidden when not known", bool(np.all(mo["cond"][~known, 2] == 0.0)),
          known_fraction=f"{known.mean():.2f}")  # fmt: skip
    noise = e["hw"][live] / np.sqrt(np.maximum(e["n"][live], 1.0)) / e["sig"][live]
    worst = np.where(obs[live], noise, 0.0).max(1)
    print(f"info  largest ratio of model noise to data error along a mock track: median {np.median(worst):.2f}, "
          f"90th percentile {np.percentile(worst, 90):.2f}  (the paper's guard allows 0.33)")  # fmt: skip

    # --- the real-data path gives the same inputs as the mock path
    worst = 0.0
    for i in range(200):
        th = e["phi"][i][obs[i]] + 123.0  # any rotation of the sky
        known_i = bool(e["known"][i])
        enc = encode_observation(th, e["r"][i][obs[i]], e["sig"][i][obs[i]], cfg, grid, dissolved=bool(sub["dissolved"][i]),
                                 phi_prog_deg=float(e["phi_prog"][i]) if known_i else None)  # fmt: skip
        worst = max(worst, np.abs(enc["tokens"][obs[i]] - mo["tokens"][i][obs[i]]).max(),
                    np.abs(enc["cond"] - mo["cond"][i]).max())  # fmt: skip
    check("real-track encoding reproduces the mock inputs under a sky rotation", worst < 1e-4, max_abs_diff=f"{worst:.1e}")

    # --- likelihood
    first = np.flatnonzero(live)[:2000]  # examples that have a window
    e, obs, S, l, sub = ({k: v[first] for k, v in e.items()}, obs[first], S[first], l[first],
                         {k: v[first] for k, v in sub.items()})  # fmt: skip
    m_ = len(first)
    ll_off = jax.jit(jax.vmap(lambda r, s, o, S_, rm, rh, c, ss: log_likelihood_offsets(
        r, s, o, S_, rm, rh, c, ss, grid.i0, n_min=cfg.n_min, var_ratio=cfg.var_ratio)))  # fmt: skip
    L = np.asarray(ll_off(e["r"][:m_], e["sig"][:m_], obs[:m_], S[:m_], sub["r_med"][:m_], sub["r_hw"][:m_],
                          sub["count"][:m_], e["sig_sys"][:m_]))  # fmt: skip
    var = e["sig"][:m_] ** 2 + e["sig_sys"][:m_, None] ** 2
    direct = -0.5 * np.sum(np.where(obs[:m_], (e["m"][:m_] - e["r"][:m_]) ** 2 / var + np.log(2 * np.pi * var), 0.0), 1)
    at_true = L[np.arange(m_), l[:m_]]
    check("likelihood at the true offset equals eq. 4 written by hand", np.allclose(at_true, direct, rtol=1e-4, atol=1e-2),
          max_abs_diff=f"{np.abs(at_true - direct).max():.1e}")  # fmt: skip
    check("likelihood is zero for offsets outside the window",
          bool(np.all(np.isneginf(L[np.arange(L.shape[1])[None] > S[:m_, None]]))))
    quiet = (e["sig_sys"][:m_] < 2.0) & (S[:m_] >= 12)
    hit = (np.argmax(L, 1) == l[:m_])[quiet].mean() if quiet.any() else math.nan
    print(f"info  low-noise tracks of >= 12 bins whose most likely progenitor offset is the true one: {100 * hit:.0f} %"
          f"  ({int(quiet.sum())} tracks)")  # fmt: skip

    # --- the training dataset: streams drawn in proportion to Z, augmented, adapted
    ds = SelectionDataset(data, p=selection_probability(data, cfg), seed=3, batch_size=64, adapter=build_adapter(cfg),
                          drop_last=True, augmentations=[MockObservations(cfg, grid, design, 3)])  # fmt: skip
    bt = ds[0]
    want = dict(inference_variables=(64, N_PAR), inference_conditions=(64, N_COND),
                summary_variables=(64, cfg.span_max, N_TOKEN_FEATURES), sample_weight=(64,))  # fmt: skip
    check("training batches have the shapes the estimator expects",
          all(bt[k].shape == v for k, v in want.items()) and (("summary_mask" in bt) == USES_MASK[cfg.summary]),
          keys=sorted(bt))  # fmt: skip
    picks = np.bincount(ds.indices, minlength=len(z))
    check("streams are drawn in proportion to Z", abs(np.corrcoef(picks, z)[0, 1]) > 0.5 or z.std() / z.mean() < 0.05,
          correlation=f"{np.corrcoef(picks, z)[0, 1]:.2f}")  # fmt: skip

    # --- importance weights: prior samples re-weighted by the likelihood reproduce a known posterior
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200000, 2))
    ll = -0.5 * (z[:, 0] - 1.0) ** 2 / 0.25  # data constrains the first coordinate to N(1, 0.5^2)
    iw = importance_weights(z, log_prior_latent(jnp.asarray(z)), ll)
    mean = float(np.sum(iw["weights"] * z[:, 0]))
    check("importance weights recover an analytic posterior mean", abs(mean - 0.8) < 0.02, mean=f"{mean:.3f}", expected=0.8,
          efficiency=f"{iw['efficiency']:.2f}")  # fmt: skip

    # --- observed tracks against the training range
    obs_tracks = load_observed(cfg)
    inside = []
    for o in obs_tracks:
        try:
            inside.append(encode_observation(o["theta_deg"], o["r"], o["r_err"], cfg, grid, dissolved=True, phi_prog_deg=0.0))
        except ValueError as err:
            print(f"info  {o['name']}: {err}")
    n_in = sum(x["in_support"] for x in inside)
    spans = [x["S"] for x in inside]
    check("observed tracks encode on the token grid", len(inside) == len(obs_tracks), encoded=f"{len(inside)}/{len(obs_tracks)}",
          in_training_range=n_in, spans=f"{min(spans)}-{max(spans)}")  # fmt: skip
    print("info  share of (stream, progenitor position) pairs that cover a track of S bins: "
          + "  ".join(f"S={s}: {100 * design['acc'][s - cfg.span_min]:.1f} %" for s in (8, 16, 24, 32, 48, 64)))  # fmt: skip

    # --- speed of the augmentation
    aug = MockObservations(cfg, grid, design, 7)
    batch = {k: v[: cfg.batch_size] for k, v in data.items()}
    aug(batch)
    t0 = time.time()
    for _ in range(20):
        aug(batch)
    print(f"info  augmentation: {1e3 * (time.time() - t0) / 20:.1f} ms per batch of {cfg.batch_size} on {DEVICE} "
          f"(first call with compilation {t_first:.1f} s)")  # fmt: skip
    if fails:
        raise SystemExit(f"{len(fails)} check(s) failed: {fails}")
    print("all checks passed")


if __name__ == "__main__":
    command, config = parse_args()
    dict(check=cmd_check, train=cmd_train, sample=cmd_sample)[command](config)
