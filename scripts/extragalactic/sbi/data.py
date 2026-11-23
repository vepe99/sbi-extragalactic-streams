"""The simulation set, the prior, and the observed tracks.

PARAMETERS. The 14 parameters of Chemaly et al. 2026b (arXiv:2607.05510, table A1) in the order of StreaMAX-fit,
plus the position of the progenitor along the observed track:

    logM Rs dirx diry dirz logm rs x0 z0 vx0 vy0 vz0 time sig | prog

13 come from the shards: `theta` holds (q, cos_i, psi) and `c26_vec` the equivalent orientation vector
(dirx, diry, dirz) = r(q) (-n_x, -n_y, n_z) that the paper samples; the vector is used because psi is periodic.
`sig` (sigma_sys) and `prog` are drawn in the mock observation. `prog` is the fraction of the way along the
track at which the progenitor sits (0 = first bin edge, 1 = last), uniform a priori.

LATENT COORDINATES. Networks work on zeta_k = Phi^-1(F_k(parameter_k)), F_k the prior cdf. The prior is then
exactly N(0, I), and F_k(parameter_k) is the unit cube dynesty samples in the paper, so importance weights need
no Jacobian.

FRAME. The prior keeps one copy of every set of equivalent systems (y0 = 0 and x0 > 0, vy0 > 0, z0 > 0,
dirz >= 0), so there is no rotation or mirror augmentation: a transformed example would be identical or
outside the prior.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import h5py
import numpy as np
from scipy import special

from config import REPO, Config

PARAM_NAMES = ["logM", "Rs", "dirx", "diry", "dirz", "logm", "rs", "x0", "z0", "vx0", "vy0", "vz0", "time", "sig", "prog"]
N_PAR, I_SIG, I_PROG = len(PARAM_NAMES), 13, 14
KMS_TO_KPCGYR = 1.0227121650537077


@dataclass(frozen=True)
class Prior:
    """Per parameter: "U" uniform(a, b), "N" normal(0, a), "H" half-normal(0, a) > 0. Table A1 is read from the
    configuration stored with the simulations, never retyped."""

    kind: tuple
    a: tuple
    b: tuple

    @classmethod
    def from_sims(cls, sc: dict, sig_sys_max: float) -> "Prior":
        U, N, H = (lambda ab: ("U", *map(float, ab))), (lambda s: ("N", float(s), 0.0)), (lambda s: ("H", float(s), 0.0))
        rows = [U(sc["log10_M_halo"]), U(sc["R_s"]), N(1), N(1), H(1), U(sc["log10_m_prog"]), U(sc["r_s"]),
                H(sc["sigma_pos"]), H(sc["sigma_pos"]), N(sc["sigma_vel"]), H(sc["sigma_vel"]), N(sc["sigma_vel"]),
                U(sc["t_int"]), U((0.0, sig_sys_max)), U((0.0, 1.0))]  # fmt: skip
        return cls(*zip(*rows))

    def _u(self, x, k):  # prior cdf
        return (x - self.a[k]) / (self.b[k] - self.a[k]) if self.kind[k] == "U" else special.erf(x / (self.a[k] * math.sqrt(2)))

    def to_latent(self, x, cols=None, eps=1e-7) -> np.ndarray:
        x = np.asarray(x, np.float64)
        f = lambda j, k: x[..., j] / self.a[k] if self.kind[k] == "N" else special.ndtri(np.clip(self._u(x[..., j], k), eps, 1 - eps))
        return np.stack([f(j, k) for j, k in enumerate(range(N_PAR) if cols is None else cols)], axis=-1)

    def from_latent(self, z, cols=None) -> np.ndarray:
        z = np.asarray(z, np.float64)

        def f(j, k):
            kind, a, b = self.kind[k], self.a[k], self.b[k]
            if kind == "N":
                return z[..., j] * a
            u = special.ndtr(z[..., j])
            return a + (b - a) * u if kind == "U" else a * special.ndtri(0.5 * (1 + u))  # H: F(x) = 2 Phi(x/a) - 1

        return np.stack([f(j, k) for j, k in enumerate(range(N_PAR) if cols is None else cols)], axis=-1)


def orientation(vec, q_lim=(0.5, 1.5)) -> np.ndarray:
    """(dirx, diry, dirz) -> (q, cos_i, psi): the length maps to q through the Maxwell cdf (StreaMAX-fit `get_q`),
    the direction is the halo axis with its sky-plane components mirrored (`c26_vector` in generate.py)."""
    v = np.asarray(vec, np.float64)
    r = np.linalg.norm(v, axis=-1)
    q = q_lim[0] + (q_lim[1] - q_lim[0]) * (special.erf(r / math.sqrt(2)) - math.sqrt(2 / math.pi) * r * np.exp(-0.5 * r**2))
    n = v / r[..., None] * np.array([-1.0, -1.0, 1.0])
    return np.stack([q, n[..., 2], np.mod(np.arctan2(n[..., 1], n[..., 0]), 2 * np.pi)], axis=-1)


def to_simulator(params) -> np.ndarray:
    """The 14 parameters -> the 13 columns of `theta` generate.py simulates (q, cos_i, psi instead of the vector)."""
    p = np.asarray(params, np.float64)
    return np.concatenate([p[..., :2], orientation(p[..., 2:5]), p[..., 5:13]], axis=-1)


@dataclass
class Grid:
    n_bins: int
    bin_deg: float
    i0: int  # the progenitor sits on the left edge of bin i0


def run_extents(valid: np.ndarray, i0: int):
    """Lengths (a, b) of the unbroken run of valid bins to the left / right of the progenitor."""
    left, right = valid[:, :i0][:, ::-1], valid[:, i0:]
    return (np.where(left.all(1), left.shape[1], np.argmin(left, 1)), np.where(right.all(1), right.shape[1], np.argmin(right, 1)))


def window_design(cfg: Config, count, r_hw, i0) -> dict:
    """The window prior p0(S) and the acceptance probability Z of every simulation.

    A mock track is a window of S bins with l of them left of the progenitor, S ~ p0(S), l ~ U{0..S}. A stream
    whose comparable bins extend a to the left and b to the right of its progenitor covers it if l <= a and
    S - l <= b. Drawing every stream once per epoch and then a window it covers would favour streams barely
    longer than the data; instead streams are drawn in proportion to Z_i = P0(a random window is covered),
    which is rejection sampling of p0 and leaves  prior x p0 x 1[model covers the data]  (the paper's guards).
    Comparable bins depend on the data errors when the model-noise guard is on, so Z is computed per stratum of
    the error level (flat errors at the stratum's mid level) and the loss weight in mock.py corrects the rest.
    p0(S) = span_prior(S) / coverage(S), so the accepted track lengths follow span_prior.
    """
    S = np.arange(cfg.span_min, cfg.span_max + 1)
    K = cfg.err_strata if cfg.var_ratio > 0 else 1
    edges = np.logspace(*np.log10(cfg.err_level), K + 1)

    def frac(valid):  # share of the S + 1 progenitor positions each stream covers, for every S
        a, b = run_extents(valid, i0)
        n = np.minimum(S[None], a[:, None]) - np.maximum(0, S[None] - b[:, None]) + 1
        return (np.clip(n, 0, None) / (S[None] + 1.0)).astype(np.float32)

    base = count >= cfg.n_min
    noise = r_hw**2 / np.maximum(count, 1.0) * cfg.var_ratio
    fr = [frac(base & (noise <= lv**2)) for lv in np.sqrt(edges[:-1] * edges[1:])] if cfg.var_ratio > 0 else [frac(base)]
    cover = np.mean([f.mean(0, dtype=np.float64) for f in fr], axis=0)
    pS = (1.0 / S if cfg.span_prior == "loguniform" else np.ones(len(S))) / np.maximum(cover, 1e-4)
    pS = pS / pS.sum()
    # Z is floored at 5 % of its guard-free value: the scatter of the errors along a track can make bins
    # comparable that the flat level does not, and those draws must stay possible
    Z = np.maximum(np.stack([f.astype(np.float64) @ pS for f in fr], 1), 0.05 * (frac(base).astype(np.float64) @ pS)[:, None])
    return dict(S=S, pS=pS, cover=cover, Z=Z.astype(np.float32))


def load_simulations(cfg: Config) -> dict:
    """Shards -> arrays for the streams a mock observation can be drawn from. Dropped: failed integrations
    (status bit 16), streams that lost more than 1 % of their particles, and streams with fewer than `span_min`
    bins of >= n_min particles around the progenitor."""
    files = sorted(cfg.data_dir().glob("shard_*.h5"))[: cfg.max_shards or None]
    if not files:
        raise SystemExit(f"no shards in {cfg.data_dir()}")
    with h5py.File(files[0], "r") as f:
        sc = json.loads(f.attrs["config"])
        group = {f[g].attrs["ordering"]: g for g in ("track", "track_alt") if g in f}[cfg.ordering]
    grid = Grid(int(round(2 * sc["phi_max_deg"] / sc["bin_deg"])), float(sc["bin_deg"]), int(round(sc["phi_max_deg"] / sc["bin_deg"])))
    cols = dict(theta="theta", vec="c26_vec", status="status", n_bad="diag/n_bad", dissolved="prog/dissolved",
                count=f"{group}/count", r_med=f"{group}/r_med", r_hw=f"{group}/r_hw")  # fmt: skip
    raw = {k: [] for k in cols}
    for p in files:
        with h5py.File(p, "r") as f:
            for k, path in cols.items():
                raw[k].append(f[path][:])
    raw = {k: np.concatenate(v) for k, v in raw.items()}
    count = raw["count"].astype(np.float32)
    r_med, r_hw = (np.nan_to_num(raw[k]).astype(np.float32) for k in ("r_med", "r_hw"))
    failed, lossy, unbound = (raw["status"] & 16) > 0, raw["n_bad"] > 0.01 * sc["n_particles"], (raw["status"] & 1) > 0
    a, b = run_extents(count >= cfg.n_min, grid.i0)
    keep = ~failed & ~lossy & (a + b >= cfg.span_min) & (cfg.include_unbound | ~unbound)

    prior = Prior.from_sims(sc, cfg.sig_sys_max)
    th = raw["theta"].astype(np.float64)
    phys13 = np.concatenate([th[:, :2], raw["vec"], th[:, 5:]], axis=1)
    design = window_design(cfg, count[keep], r_hw[keep], grid.i0)
    data = dict(zeta13=prior.to_latent(phys13[keep], cols=range(13)).astype(np.float32), count=count[keep], r_med=r_med[keep],
                r_hw=r_hw[keep], dissolved=raw["dissolved"][keep].astype(np.float32), z_strata=design["Z"])  # fmt: skip
    stats = dict(n_streams=len(keep), n_shards=len(files), n_usable=int(keep.sum()), failed=float(failed.mean()),
                 lossy=float(lossy.mean()), unbound=float(unbound.mean()), n_particles=int(sc["n_particles"]))  # fmt: skip
    return dict(data=data, phys13=phys13[keep], phys13_all=phys13, theta=th[keep], prior=prior, grid=grid, design=design, stats=stats, sim_config=sc)


def split(sims: dict, val_frac: float) -> tuple[dict, dict]:
    """(training streams, held-out streams): the held-out ones are the last `val_frac` in stream order."""
    n = len(sims["phys13"])
    cut = n - max(1, int(round(val_frac * n)))
    return ({k: v[:cut] for k, v in sims["data"].items()}, {k: v[cut:] for k, v in sims["data"].items()})


def load_observed(cfg: Config) -> list[dict]:
    """The 32 STRRINGS tracks released with StreaMAX-fit. `r_scale` is the paper's rescaling of the radii
    (100 for PGC938075, whose radii are below 1 kpc; 1 otherwise) and is applied here as it is in its fits."""
    out = []
    for p in sorted((REPO / cfg.obs_dir).glob("*.npz")):
        d = np.load(p)
        s = float(d["r_scale"]) if "r_scale" in d.files and np.isfinite(d["r_scale"]) else 1.0
        out.append(dict(name=p.stem, theta_deg=np.rad2deg(d["theta"]), r=d["r"] * s, r_err=d["r_err"] * s, r_scale=s))
    return out
