"""Mock observations: the only augmentation. Written in JAX, so it runs on the GPU when there is one.

A simulation gives the model track on a fixed grid of 10 deg bins of the unwrapped angle: median radius m_b,
16-84 half-width h_b and particle count n_b, the progenitor on the left edge of bin i0. Every call turns one
simulation into a new observation:

  1. data errors sigma_b on the whole grid, independent of the parameters;
  2. the bins where model and data can be compared, by the two guards of the paper (2026b sec. 3.3):
     n_b >= n_min and h_b^2 / n_b <= sigma_b^2 / var_ratio;
  3. the unbroken run of such bins around the progenitor: a to its left, b to its right;
  4. the observed window: S bins, l of them left of the progenitor, (S, l) ~ p0(S) U{0..S} restricted to the
     run (`window_design` in data.py explains why streams are drawn in proportion to Z and what the weight is);
  5. gaps: interior bins dropped at random (10 of the 32 observed tracks have missing bins);
  6. sigma_sys ~ U(0, 25 kpc) and the radii  r_b ~ N(m_b, sigma_b^2 + sigma_sys^2): eq. 5 of the paper, i.e.
     a draw from its likelihood;
  7. the data frame of StreaMAX-fit: angle minus the median observed angle.

PROGENITOR POSITION. The paper fixes one sky coordinate of the progenitor (at the visible progenitor, else at
the centre of the track). Here its position along the track is the 15th parameter, `prog` = (l + u) / (S + 1)
with u ~ U(0, 1), uniform a priori. In a share `p_known` of the examples it is marked as fixed
(`fixed_target_mask` = 0): the network is then given its value and infers the other 14, as in the paper. In the
rest it is inferred with them. At inference either is available: fix it at 0.5 (the centre), at a visible
progenitor, or leave it free.
"""

from __future__ import annotations

import math

import bayesflow as bf
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.scipy.special import ndtri

from config import Config
from data import I_PROG, N_PAR, Grid


def featurise(phi, r, sig, obs, dissolved):
    """One track on the token grid -> network inputs; the same function serves mock and real tracks.

    tokens (T, 5): angle / 90 deg, observed flag, r / rbar - 1, sigma / rbar, log10 sigma (zero where unobserved)
    cond (3,):     progenitor dissolved (mass ramped to zero) or not, log10(rbar / 30 kpc), points / 32
    """
    w = obs.astype(jnp.float32)
    n = jnp.maximum(w.sum(), 1.0)
    rbar = jnp.clip((w * r).sum() / n, 3.0, 1000.0)  # mean observed radius
    sig = jnp.maximum(sig, 1e-3)
    tokens = jnp.stack([phi / 90.0, w, w * jnp.clip(r / rbar - 1, -20, 20), w * jnp.minimum(sig / rbar, 20), w * jnp.log10(sig)], -1)
    return tokens, jnp.stack([jnp.asarray(dissolved, jnp.float32), jnp.log10(rbar / 30.0), n / 32.0])


def make_mock(cfg: Config, grid: Grid, design: dict):
    """-> f(key, count, r_med, r_hw, dissolved, z_strata) for ONE simulation; vmap it over a batch."""
    S_grid, L = jnp.asarray(design["S"]), jnp.arange(cfg.span_max + 1)
    logp0 = jnp.asarray(np.log(design["pS"])[:, None] - np.log(design["S"] + 1.0)[:, None], jnp.float32)
    T, NB, I0, BIN = cfg.span_max, grid.n_bins, grid.i0, grid.bin_deg
    lo, hi = (math.log10(x) for x in cfg.err_level)
    j = jnp.arange(T)

    def mock(key, count, r_med, r_hw, dissolved, z_strata):
        k = jr.split(key, 12)
        # 1. errors: the stratum of the track level is drawn in proportion to this stream's acceptance there
        kk = jr.categorical(k[0], jnp.log(z_strata))
        level = 10.0 ** (lo + (kk + jr.uniform(k[1])) * (hi - lo) / z_strata.shape[0])
        tau = jr.uniform(k[2], minval=cfg.err_scatter[0], maxval=cfg.err_scatter[1])
        sig_grid = jnp.clip(level * jnp.exp(tau * jr.normal(k[3], (NB,))), *cfg.err_clip)
        # 2.-3. comparable bins and their run around the progenitor
        valid = count >= cfg.n_min
        if cfg.var_ratio > 0:
            valid &= r_hw**2 / jnp.maximum(count, 1.0) * cfg.var_ratio <= sig_grid**2
        left, right = valid[:I0][::-1], valid[I0:]
        a = jnp.where(left.all(), I0, jnp.argmin(left))
        b = jnp.where(right.all(), NB - I0, jnp.argmin(right))
        # 4. window inside the run; Z = P0(a random window is inside). The stream and stratum were drawn with
        #    probability z_strata, so Z / z_strata is what is left for the loss (1 with the second guard off)
        ok = (L[None] <= S_grid[:, None]) & (L[None] <= a) & (S_grid[:, None] - L[None] <= b)
        logits = jnp.where(ok, logp0, -jnp.inf).ravel()
        pick = jr.categorical(k[4], logits)
        S, l = S_grid[pick // L.shape[0]], L[pick % L.shape[0]]
        weight = jnp.exp(logits).sum() / z_strata[kk]
        # 5. gaps
        rate = jnp.where(jr.uniform(k[5]) < cfg.gap_prob, jr.uniform(k[6]) * cfg.gap_rate_max, 0.0)
        obs = (j < S) & ~((jr.uniform(k[7], (T,)) < rate) & (j > 0) & (j < S - 1))
        # 6. a draw from the likelihood
        bins = jnp.clip(I0 - l + j, 0, NB - 1)
        m, sig = r_med[bins], sig_grid[bins]
        sig_sys = cfg.sig_sys_max * jr.uniform(k[8], minval=1e-6, maxval=1 - 1e-6)
        r = m + jnp.sqrt(sig**2 + sig_sys**2) * jr.normal(k[9], (T,))
        # 7. data frame
        phi_sim = (j - l + 0.5) * BIN  # the progenitor is at 0
        xs = jnp.sort(jnp.where(obs, phi_sim, jnp.inf))
        n = obs.sum()
        phi = phi_sim - 0.5 * (xs[(n - 1) // 2] + xs[n // 2])  # minus the median, as np.median
        tokens, cond = featurise(phi, r, sig, obs, dissolved)
        # the two parameters that exist only in the observation, in latent coordinates
        prog = (l + jr.uniform(k[10], minval=1e-4, maxval=1 - 1e-4)) / (S + 1.0)
        zeta2 = ndtri(jnp.stack([jnp.clip(sig_sys / cfg.sig_sys_max, 1e-6, 1 - 1e-6), prog]))
        fixed = jnp.ones(N_PAR).at[I_PROG].set(1.0 - jr.bernoulli(k[11], cfg.p_known))  # 0 = given, 1 = inferred
        return dict(tokens=tokens, mask=obs, cond=cond, weight=weight, zeta2=zeta2, fixed_target_mask=fixed,
                    r=r, sig=sig, phi=phi, sig_sys=sig_sys, S=S, l=l, a=a, b=b, m=m, hw=r_hw[bins], n=count[bins])  # fmt: skip

    return mock


NETWORK_KEYS = ("zeta", "tokens", "mask", "cond", "weight", "fixed_target_mask")


class MockObservations:
    """The augmentation handed to the BayesFlow dataset: a batch of simulations in, one fresh mock observation
    each out. One jitted call on the JAX device; each call has its own key (seed, call number)."""

    def __init__(self, cfg: Config, grid: Grid, design: dict, seed: int, weight_norm: float = 1.0, extras: bool = False):
        self.fn, self.key, self.calls = jax.jit(jax.vmap(make_mock(cfg, grid, design))), jr.key(seed), 0
        self.weight_norm, self.extras = weight_norm, extras

    def __call__(self, batch: dict) -> dict:
        keys = jr.split(jr.fold_in(self.key, self.calls), len(batch["zeta13"]))
        self.calls += 1
        o = self.fn(keys, *(jnp.asarray(batch[k]) for k in ("count", "r_med", "r_hw", "dissolved", "z_strata")))
        o = {k: np.asarray(v) for k, v in o.items()}
        o["zeta"] = np.concatenate([batch["zeta13"], o.pop("zeta2")], axis=1)
        o["weight"] = o["weight"] / self.weight_norm
        return o if self.extras else {k: o[k] for k in NETWORK_KEYS}


class SelectionDataset(bf.OfflineDataset):
    """`bf.OfflineDataset` whose epochs draw stream i with probability proportional to Z_i (with replacement)
    instead of visiting every stream once: the rejection step of the window prior."""

    def __init__(self, data, *, seed: int, **kwargs):
        self._p = data["z_strata"].mean(1).astype(np.float64)
        self._p, self._rng = self._p / self._p.sum(), np.random.default_rng(seed)
        super().__init__(data, **kwargs)

    def shuffle(self) -> None:
        self.indices = self._rng.choice(self.num_samples, size=self.num_samples, p=self._p)


def fixed_observations(data: dict, cfg: Config, grid: Grid, design: dict, seed: int, n: int, unweighted: bool = False) -> dict:
    """n mock observations with a fixed key, with everything the tests and the evaluation need (`pick` = the
    stream behind each). unweighted=True resamples them by their loss weight, so that plain averages over the
    set are averages over the training distribution."""
    p = data["z_strata"].mean(1).astype(np.float64)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(p), size=n, p=p / p.sum())
    aug = MockObservations(cfg, grid, design, seed, extras=True)
    parts = [aug({k: v[pick[s : s + 4096]] for k, v in data.items()}) for s in range(0, n, 4096)]
    out = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]} | dict(pick=pick)
    if unweighted:
        keep = rng.choice(n, size=n, p=out["weight"] / out["weight"].sum())
        out = {k: v[keep] for k, v in out.items()} | dict(weight=np.ones(n, np.float32))
    return out


def prog_fraction(phi_prog_deg: float, phi_first_deg: float, S: int, bin_deg: float) -> float:
    """`prog` of a progenitor at angle phi_prog in the data frame, for a track whose first bin is centred on
    phi_first and spans S bins. A progenitor at the centre of the track gives 0.5."""
    return float(np.clip(((phi_prog_deg - phi_first_deg) / bin_deg + 1.0) / (S + 1.0), 1e-4, 1 - 1e-4))


def encode_observation(theta_deg, r, r_err, cfg: Config, grid: Grid, *, dissolved: bool, prog: float | None = 0.5) -> dict:
    """A real track -> the network inputs, with exactly the conventions of the mock observations.

    theta_deg: increasing, steps that are multiples of the bin width (missing bins allowed).
    prog: position of the progenitor along the track (see `prog_fraction`), given to the network as a fixed
          parameter; 0.5 is the paper's assumption for a stream without a visible progenitor. None = infer it.
    """
    phi = np.asarray(theta_deg, np.float64) - np.median(theta_deg)
    kk = np.rint((phi - phi[0]) / grid.bin_deg).astype(int)
    S, T = int(kk[-1]) + 1, cfg.span_max
    if np.abs(phi - phi[0] - kk * grid.bin_deg).max() > 1e-3 * grid.bin_deg or np.any(np.diff(kk) <= 0) or S > T:
        raise ValueError(f"track is not on a regular grid of the bin width, or spans more than {T} bins")
    rg, sg, og = np.zeros(T, np.float32), np.ones(T, np.float32), np.zeros(T, bool)
    rg[kk], sg[kk], og[kk] = r, r_err, True
    phig = (phi[0] + grid.bin_deg * np.arange(T)).astype(np.float32)
    tokens, cond = featurise(*(jnp.asarray(x) for x in (phig, rg, sg, og)), float(dissolved))
    fixed, value = np.ones(N_PAR, np.float32), np.zeros(N_PAR, np.float32)
    if prog is not None:
        fixed[I_PROG], value[I_PROG] = 0.0, float(ndtri(prog))
    return dict(tokens=np.asarray(tokens), mask=og, cond=np.asarray(cond), fixed_target_mask=fixed, fixed_target_value=value,
                r=rg, sig=sg, phi=phig, S=S)  # fmt: skip
