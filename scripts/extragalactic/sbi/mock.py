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
     a draw from its likelihood (`noise_scale` multiplies the scatter; 0 gives r_b = m_b, a test without noise);
  7. the data frame of StreaMAX-fit: angle minus the median observed angle.

With `replace_tracks` the streams of the set have different particle numbers, and guard 1, a fraction of the
particles, is per stream (`n_min` of the data).

SCALE AUGMENTATION (`scale_aug`, training batches only). Gravity has no scale of its own and the leapfrog takes a
fixed number of steps, so with lengths x L and masses x U (velocities x sqrt(U / L), t_int x sqrt(L^3 / U)) the
simulator returns the same stream, its track x L (scripts/extragalactic/symmetries.py). Before step 1 every
example therefore moves along this family: (ln L, ln U) is drawn from the prior restricted to the family
(`draw_scale`), the parameters are scaled and their latent coordinates computed again, and m_b and h_b are
multiplied by L, so that guards, errors and sigma_sys act on the scaled track. A dissolved progenitor moves in L
only: its mass is ramped to a fixed 1 Msun, which breaks the mass scaling.

PROGENITOR POSITION. The paper fixes one sky coordinate of the progenitor (at the visible progenitor, else at
the centre of the track). Here its position along the track is the 15th parameter, `prog` = (l + u) / (S + 1)
with u ~ U(0, 1), uniform a priori. In a share `p_known` of the examples it is marked as fixed
(`fixed_target_mask` = 0): the network is then given its value and infers the other 14, as in the paper. In the
rest it is inferred with them. At inference either is available: fix it at 0.5 (the centre), at a visible
progenitor, or leave it free. In BayesFlow's terms this is a fixed target, p(theta_14 | prog = value, track): the
position is a parameter of the model whose value is known, not a piece of data that may be missing
(`observed_condition_mask`) nor a parameter to integrate out (`infer_target_mask`; that marginal is the joint
posterior with the column dropped). The mask only clamps the coordinate; a coupling flow or an MLP subnet is not
told which case it is in, so the conditions also carry a "position given" flag.
"""

from __future__ import annotations

import dataclasses
import math

import bayesflow as bf
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.scipy.special import erf, erfc, ndtri

from config import Config
from data import I_PROG, N_PAR, Grid, Prior

LN10 = math.log(10.0)
# columns of the 13 simulated parameters (data.PARAM_NAMES): logM Rs dirx diry dirz logm rs x0 z0 vx0 vy0 vz0 time
I_LOGM, I_RS, I_LOGMP, I_RSP, I_X0, I_Z0, I_T = 0, 1, 5, 6, 7, 8, 12
I_LEN, I_VEL, I_DIR = (1, 6, 7, 8), (9, 10, 11), (2, 3, 4)


def scale_params(phys13, a, b):
    """The 13 parameters of the same stream at lengths x L = e^a and masses x U = e^b: log M and log m + log10 U;
    R_s, r_s, x0, z0 x L; velocities x sqrt(U / L); t_int x sqrt(L^3 / U); the halo orientation (and q) as it is."""
    f = jnp.ones(13).at[jnp.array(I_LEN)].set(jnp.exp(a)).at[jnp.array(I_VEL)].set(jnp.exp(0.5 * (b - a))).at[I_T].set(jnp.exp(0.5 * (3 * a - b)))
    return phys13 * f + jnp.zeros(13).at[jnp.array([I_LOGM, I_LOGMP])].set(b / LN10)


def latent13(phys13, prior: Prior, eps: float = 1e-7):
    """`Prior.to_latent` for the 13 simulated parameters, in JAX and single precision: the upper half of a cdf is
    taken from the other end (1 - u for a uniform, erfc for a half-normal), where float32 still resolves it."""
    cols = []
    for k in range(13):
        x, kind, lo, hi = phys13[..., k], prior.kind[k], prior.a[k], prior.b[k]
        if kind == "N":
            cols.append(x / lo)
            continue
        if kind == "U":
            u, v = (x - lo) / (hi - lo), (hi - x) / (hi - lo)
        else:  # half-normal: F(x) = erf(x / (a sqrt 2))
            u, v = erf(x / (lo * math.sqrt(2))), erfc(x / (lo * math.sqrt(2)))
        cols.append(jnp.where(u < 0.5, ndtri(jnp.clip(u, eps, 0.5)), -ndtri(jnp.clip(v, eps, 0.5))))
    return jnp.stack(cols, -1)


def draw_scale(key, phys13, dissolved, prior: Prior, n_points: int):
    """(a, b) = (ln L, ln U) for ONE stream, such that the scaled parameters still follow the prior.

    The scalings form a group acting on the parameters, and along the orbit of one stream the prior has, per unit
    of da db, the density  prior(scaled parameters) x |Jacobian|  with the Jacobian L^4 (R_s, r_s, x0, z0)
    x (U / L)^(3/2) (velocities) x (L^3 / U)^(1/2) (t_int) = L^4 U:

        w(a, b) = 1[R_s, r_s, log M, log m, t_int inside their ranges]
                  x exp(-L^2 (x0^2 + z0^2) / (2 sigma_pos^2)) x exp(-(U / L) |v|^2 / (2 sigma_vel^2)) x L^4 U.

    The draw: the point the stream is at, (0, 0), and n_points - 1 points uniform in the box that bounds the
    allowed region (the same box from every point of the orbit), one of them picked with probability
    proportional to w. With the stream's own point among the candidates this leaves the prior exactly invariant
    for any n_points; more points only bring the draw closer to an independent one from w. A dissolved progenitor
    has b = 0 (one dimension, the same formula)."""
    ln = jnp.log
    a_lo = jnp.maximum(ln(prior.a[I_RS] / phys13[I_RS]), ln(prior.a[I_RSP] / phys13[I_RSP]))
    a_hi = jnp.minimum(ln(prior.b[I_RS] / phys13[I_RS]), ln(prior.b[I_RSP] / phys13[I_RSP]))
    b_lo = LN10 * jnp.maximum(prior.a[I_LOGM] - phys13[I_LOGM], prior.a[I_LOGMP] - phys13[I_LOGMP])
    b_hi = LN10 * jnp.minimum(prior.b[I_LOGM] - phys13[I_LOGM], prior.b[I_LOGMP] - phys13[I_LOGMP])
    c_lo, c_hi = ln(prior.a[I_T] / phys13[I_T]), ln(prior.b[I_T] / phys13[I_T])  # of c = (3 a - b) / 2, the log of the factor on t_int
    b_lo, b_hi = jnp.where(dissolved > 0.5, 0.0, b_lo), jnp.where(dissolved > 0.5, 0.0, b_hi)
    # the box around the part of [a_lo, a_hi] x [b_lo, b_hi] inside the band of c; (0, 0) is in it
    a0, a1 = jnp.minimum(jnp.maximum(a_lo, (b_lo + 2 * c_lo) / 3), 0.0), jnp.maximum(jnp.minimum(a_hi, (b_hi + 2 * c_hi) / 3), 0.0)
    b0, b1 = jnp.minimum(jnp.maximum(b_lo, 3 * a_lo - 2 * c_hi), 0.0), jnp.maximum(jnp.minimum(b_hi, 3 * a_hi - 2 * c_lo), 0.0)
    k1, k2 = jr.split(key)
    u = jr.uniform(k1, (n_points - 1, 2))
    a = jnp.concatenate([jnp.zeros(1), a0 + u[:, 0] * (a1 - a0)])
    b = jnp.concatenate([jnp.zeros(1), b0 + u[:, 1] * (b1 - b0)])
    c = 0.5 * (3 * a - b)
    inside = ((c >= c_lo) & (c <= c_hi)).at[0].set(True)
    rho2 = phys13[I_X0] ** 2 + phys13[I_Z0] ** 2
    v2 = jnp.sum(phys13[jnp.array(I_VEL)] ** 2)
    logw = 4 * a + b - jnp.exp(2 * a) * rho2 / (2 * prior.a[I_X0] ** 2) - jnp.exp(b - a) * v2 / (2 * prior.a[I_VEL[0]] ** 2)
    j = jr.categorical(k2, jnp.where(inside, logw, -jnp.inf))
    return a[j], b[j]


def scale_stream(phys13, zeta13, r_med, r_hw, a, b, prior: Prior):
    """ONE stream at lengths x e^a and masses x e^b -> (latent coordinates, r_med, r_hw). The counts and the
    angles of the track stay; the latent coordinates of the halo orientation are the stored ones."""
    z = latent13(scale_params(phys13, a, b), prior)
    L = jnp.exp(a)
    return z.at[jnp.array(I_DIR)].set(zeta13[jnp.array(I_DIR)]), r_med * L, r_hw * L


def featurise(phi, r, sig, obs, dissolved, known):
    """One track on the token grid -> network inputs; the same function serves mock and real tracks.

    tokens (T, 5): angle / 90 deg, observed flag, r / rbar - 1, sigma / rbar, log10 sigma (zero where unobserved)
    cond (4,):     progenitor dissolved (mass ramped to zero) or not, progenitor position given or not,
                   log10(rbar / 30 kpc), points / 32
    """
    w = obs.astype(jnp.float32)
    n = jnp.maximum(w.sum(), 1.0)
    rbar = jnp.clip((w * r).sum() / n, 3.0, 1000.0)  # mean observed radius
    sig = jnp.maximum(sig, 1e-3)
    tokens = jnp.stack([phi / 90.0, w, w * jnp.clip(r / rbar - 1, -20, 20), w * jnp.minimum(sig / rbar, 20), w * jnp.log10(sig)], -1)
    return tokens, jnp.stack([jnp.asarray(dissolved, jnp.float32), jnp.asarray(known, jnp.float32), jnp.log10(rbar / 30.0), n / 32.0])


def make_mock(cfg: Config, grid: Grid, design: dict, bins: tuple[int, int] | None = None, prior: Prior | None = None):
    """-> f(key, count, r_med, r_hw, dissolved, z_strata[, n_min[, phys13, zeta13]]) for ONE simulation; vmap it
    over a batch. n_min: particles a bin of this stream needs (default cfg.n_min). With `scale_aug` the 13
    parameters and their latent coordinates are arguments too, `prior` is needed, and the result holds the
    latent coordinates of the scaled stream (`zeta13`) and the scaling (`scale` = ln L, ln U).

    bins = (lo, hi): the track arrays hold only bins lo .. hi - 1 of the grid. A window reaches at most span_max
    bins to either side of the progenitor, so with those bins in the arrays the result is the same as with all
    of them (the errors are still drawn for the whole grid, to keep the random numbers)."""
    S_grid, L = jnp.asarray(design["S"]), jnp.arange(cfg.span_max + 1)
    logp0 = jnp.asarray(np.log(design["pS"])[:, None] - np.log(design["S"] + 1.0)[:, None], jnp.float32)
    LO, HI = bins or (0, grid.n_bins)
    if LO > grid.i0 - cfg.span_max or HI < grid.i0 + cfg.span_max:
        raise ValueError(f"bins {LO}..{HI} do not hold the {cfg.span_max} bins either side of the progenitor")
    T, NB, I0, BIN = cfg.span_max, HI - LO, grid.i0 - LO, grid.bin_deg
    lo, hi = (math.log10(x) for x in cfg.err_level)
    j = jnp.arange(T)
    aug = cfg.scale_aug
    if aug and prior is None:
        raise ValueError("scale_aug needs the prior")

    def mock(key, count, r_med, r_hw, dissolved, z_strata, n_min=cfg.n_min, phys13=None, zeta13=None):
        k = jr.split(key, 12)
        extra = {}
        if aug:
            # 0. the same stream at another scale. Its acceptance per stratum of the error level is that of the
            #    stored track at the level / L (guard 2 compares h_b^2 / n_b with sigma_b^2), i.e. the stored
            #    z_strata shifted by ln L in strata; the stream itself was drawn with the stored one, hence `norm`
            a_s, b_s = draw_scale(jr.fold_in(key, 0x5CA1E), phys13, dissolved, prior, cfg.scale_points)
            zeta13, r_med, r_hw = scale_stream(phys13, zeta13, r_med, r_hw, a_s, b_s, prior)
            extra = dict(zeta13=zeta13, scale=jnp.stack([a_s, b_s]))
            K = z_strata.shape[0]
            if K > 1:
                z_stored = z_strata
                at = jnp.clip(jnp.arange(K) - a_s * K / ((hi - lo) * LN10), 0, K - 1)
                z_strata = jnp.exp(jnp.interp(at, jnp.arange(K), jnp.log(z_stored)))
                norm = z_strata.sum() / z_stored.sum()
        # 1. errors: the stratum of the track level is drawn in proportion to this stream's acceptance there
        kk = jr.categorical(k[0], jnp.log(z_strata))
        level = 10.0 ** (lo + (kk + jr.uniform(k[1])) * (hi - lo) / z_strata.shape[0])
        tau = jr.uniform(k[2], minval=cfg.err_scatter[0], maxval=cfg.err_scatter[1])
        sig_grid = jnp.clip(level * jnp.exp(tau * jr.normal(k[3], (grid.n_bins,))[LO:HI]), *cfg.err_clip)
        # 2.-3. comparable bins and their run around the progenitor
        valid = count >= n_min
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
        if aug and z_strata.shape[0] > 1:
            weight = weight * norm
        # 5. gaps
        rate = jnp.where(jr.uniform(k[5]) < cfg.gap_prob, jr.uniform(k[6]) * cfg.gap_rate_max, 0.0)
        obs = (j < S) & ~((jr.uniform(k[7], (T,)) < rate) & (j > 0) & (j < S - 1))
        # 6. a draw from the likelihood
        bins = jnp.clip(I0 - l + j, 0, NB - 1)
        m, sig = r_med[bins], sig_grid[bins]
        sig_sys = cfg.sig_sys_max * jr.uniform(k[8], minval=1e-6, maxval=1 - 1e-6)
        r = m + cfg.noise_scale * jnp.sqrt(sig**2 + sig_sys**2) * jr.normal(k[9], (T,))  # noise_scale 0: the model track itself
        # 7. data frame
        phi_sim = (j - l + 0.5) * BIN  # the progenitor is at 0
        xs = jnp.sort(jnp.where(obs, phi_sim, jnp.inf))
        n = obs.sum()
        phi = phi_sim - 0.5 * (xs[(n - 1) // 2] + xs[n // 2])  # minus the median, as np.median
        known = jr.bernoulli(k[11], cfg.p_known)  # is the progenitor position given to the network?
        tokens, cond = featurise(phi, r, sig, obs, dissolved, known)
        # the two parameters that exist only in the observation, in latent coordinates
        prog = (l + jr.uniform(k[10], minval=1e-4, maxval=1 - 1e-4)) / (S + 1.0)
        zeta2 = ndtri(jnp.stack([jnp.clip(sig_sys / cfg.sig_sys_max, 1e-6, 1 - 1e-6), prog]))
        fixed = jnp.ones(N_PAR).at[I_PROG].set(1.0 - known)  # 0 = given, 1 = inferred
        return dict(tokens=tokens, mask=obs, cond=cond, weight=weight, zeta2=zeta2, fixed_target_mask=fixed,
                    r=r, sig=sig, phi=phi, sig_sys=sig_sys, S=S, l=l, a=a, b=b, m=m, hw=r_hw[bins], n=count[bins],
                    n_min=jnp.asarray(n_min, jnp.float32), **extra)  # fmt: skip

    return mock


NETWORK_KEYS = ("zeta", "tokens", "mask", "cond", "weight", "fixed_target_mask")


def accepted_inputs(o: dict, xp=np) -> dict:
    """A batch in which no rejected draw reaches the network. A draw without a window that passes the guards has
    loss weight 0, but its track is whatever lies outside the stream's comparable bins (radii of thousands of
    kpc), and a network can turn that into a gradient that overflows: 0 x inf = NaN in every weight, at an
    ordinary loss (seen twice with the MLP summary, 2026-11-24, about once in 10^7 such draws). So a rejected row
    keeps its weight 0 and takes the network inputs of an accepted row of the batch, a different one for each
    while they last. The loss is unchanged. xp: numpy or jax.numpy, as the arrays are."""
    live = o["weight"] > 0
    n = live.shape[0]
    order = xp.argsort(~live, stable=True)  # the accepted rows first
    src = xp.where(live, xp.arange(n), order[(xp.cumsum(~live) - 1) % xp.maximum(live.sum(), 1)])
    return o | {k: o[k][src] for k in NETWORK_KEYS if k != "weight"}


class MockObservations:
    """The augmentation handed to the BayesFlow dataset: a batch of simulations in, one fresh mock observation
    each out. One jitted call on the JAX device; each call has its own key (seed, call number). A training batch
    (extras off) goes through `accepted_inputs`; with extras on every draw comes back as it was made.
    prior: needed with `scale_aug`."""

    def __init__(self, cfg: Config, grid: Grid, design: dict, seed: int, weight_norm: float = 1.0, extras: bool = False, prior: Prior | None = None):
        self.fn, self.key, self.calls = jax.jit(jax.vmap(make_mock(cfg, grid, design, prior=prior))), jr.key(seed), 0
        self.weight_norm, self.extras, self.aug, self.n_min = weight_norm, extras, cfg.scale_aug, float(cfg.n_min)

    def __call__(self, batch: dict) -> dict:
        keys = jr.split(jr.fold_in(self.key, self.calls), len(batch["zeta13"]))
        self.calls += 1
        n_min = batch["n_min"] if "n_min" in batch else np.full(len(batch["zeta13"]), self.n_min, np.float32)
        o = self.fn(keys, *(jnp.asarray(batch[k]) for k in ("count", "r_med", "r_hw", "dissolved", "z_strata")), jnp.asarray(n_min),
                    *((jnp.asarray(batch["phys13"]), jnp.asarray(batch["zeta13"])) if self.aug else ()))  # fmt: skip
        o = {k: np.asarray(v) for k, v in o.items()}
        o["zeta"] = np.concatenate([o.pop("zeta13") if self.aug else batch["zeta13"], o.pop("zeta2")], axis=1)
        o["weight"] = o["weight"] / self.weight_norm
        return o if self.extras else accepted_inputs({k: o[k] for k in NETWORK_KEYS})


class SelectionDataset(bf.OfflineDataset):
    """`bf.OfflineDataset` whose epochs draw stream i with probability proportional to Z_i (with replacement)
    instead of visiting every stream once: the rejection step of the window prior."""

    def __init__(self, data, *, seed: int, **kwargs):
        self._p = data["z_strata"].mean(1).astype(np.float64)
        self._p, self._rng = self._p / self._p.sum(), np.random.default_rng(seed)
        super().__init__(data, **kwargs)

    def shuffle(self) -> None:
        self.indices = self._rng.choice(self.num_samples, size=self.num_samples, p=self._p)


class DeviceDataset(SelectionDataset):
    """The training set held on the JAX device. The streams of a batch are gathered, turned into mock
    observations and named as the adapter names them in one jitted call, so a batch never visits the host;
    `SelectionDataset` with `MockObservations` and the adapter copies it there and back, 17 ms a batch at 4096.
    Same streams, same keys and same numbers as that route (check.py compares the two).

    Only what a window can reach is put on the device: the span_max bins either side of the progenitor, the
    counts in 16 bits. That is 1.0 GB per million streams at span_max 49 (5.5 GB for train_v2_1715).

    Counts up to 32,767 (a set of 30,000 particles) are held as int16. With streams of up to 131,071 particles in
    the set (`replace_tracks`: 90,000) a count needs 17 bits, but only one bin of a stream can reach 65,536, so the
    counts are held modulo 65,536 as uint16 with, per stream, the index of that bin (`count_big`, 255 = none):
    exact, and 1.2 GB smaller for train_v2 than 32-bit counts, which leave too little of an 11 GB card for the
    training step. With `scale_aug` the 13 parameters are held too, and of their stored latent coordinates only
    the three of the halo orientation, which the scaling leaves alone."""

    def __init__(self, data: dict, cfg: Config, grid: Grid, design: dict, *, seed: int, batch_size: int, names: dict, weight_norm: float = 1.0,
                 prior: Prior | None = None):  # fmt: skip
        super().__init__(dict(z_strata=data["z_strata"]), seed=seed, batch_size=batch_size, adapter=None, drop_last=True)
        lo, hi = grid.i0 - cfg.span_max, grid.i0 + cfg.span_max
        aug, dirs = cfg.scale_aug, np.array(I_DIR)
        self.dev = {k: jnp.asarray(data[k]) for k in ("dissolved", "z_strata", "n_min") + (("phys13",) if aug else ("zeta13",))}
        if aug:
            self.dev["zeta_dir"] = jnp.asarray(data["zeta13"][:, dirs])
        self.dev |= {k: jnp.asarray(data[k][:, lo:hi]) for k in ("r_med", "r_hw")}
        count = data["count"][:, lo:hi]
        top = count.max()
        if top < 2**15:  # int16 holds the counts of a set of up to 32,767 particles
            self.dev["count"] = jnp.asarray(count.astype(np.int16))
        elif top < 2**17 and hi - lo < 255 and ((count >= 2**16).sum(1) <= 1).all():
            big = count >= 2**16
            self.dev["count"] = jnp.asarray((count.astype(np.int32) & 0xFFFF).astype(np.uint16))
            self.dev["count_big"] = jnp.asarray(np.where(big.any(1), big.argmax(1), 255).astype(np.uint8))
        else:
            self.dev["count"] = jnp.asarray(count.astype(np.int32))
        self.nbytes = sum(v.nbytes for v in self.dev.values())
        mock = jax.vmap(make_mock(cfg, grid, design, (lo, hi), prior))

        def batch(key, idx, dev):  # `dev` is an argument: closed over, it would be compiled in as a constant
            b = {k: v[idx] for k, v in dev.items()}
            count = b["count"]
            if "count_big" in b:  # the 17th bit of the one bin that has it
                count = count.astype(jnp.int32) + 2**16 * (jnp.arange(hi - lo, dtype=jnp.int32)[None] == b["count_big"].astype(jnp.int32)[:, None])
            if aug:
                b["zeta13"] = jnp.zeros((idx.shape[0], 13), jnp.float32).at[:, dirs].set(b["zeta_dir"])
            o = mock(jr.split(key, idx.shape[0]), count.astype(jnp.float32), b["r_med"], b["r_hw"], b["dissolved"], b["z_strata"], b["n_min"],
                     *((b["phys13"], b["zeta13"]) if aug else ()))  # fmt: skip
            o |= dict(zeta=jnp.concatenate([o["zeta13"] if aug else b["zeta13"], o["zeta2"]], axis=1), weight=o["weight"] / weight_norm)
            o = accepted_inputs(o, jnp)
            return {new: o[old].astype(jnp.float32) for old, new in names.items()}  # as the adapter: all float32, the mask too

        self.fn, self.key, self.calls = jax.jit(batch), jr.key(seed), 0

    def get_batch_by_sample_indices(self, indices) -> dict:
        self.calls += 1
        return self.fn(jr.fold_in(self.key, self.calls - 1), jnp.asarray(indices, jnp.int32), self.dev)


def fixed_observations(data: dict, cfg: Config, grid: Grid, design: dict, seed: int, n: int, unweighted: bool = False) -> dict:
    """n mock observations with a fixed key, with everything the tests and the evaluation need (`pick` = the
    stream behind each). unweighted=True resamples them by their loss weight, so that plain averages over the
    set are averages over the training distribution. Never scaled (`scale_aug` is for training batches)."""
    cfg = dataclasses.replace(cfg, scale_aug=False)
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
    tokens, cond = featurise(*(jnp.asarray(x) for x in (phig, rg, sg, og)), float(dissolved), float(prog is not None))
    fixed, value = np.ones(N_PAR, np.float32), np.zeros(N_PAR, np.float32)
    if prog is not None:
        fixed[I_PROG], value[I_PROG] = 0.0, float(ndtri(prog))
    return dict(tokens=np.asarray(tokens), mask=og, cond=np.asarray(cond), fixed_target_mask=fixed, fixed_target_value=value,
                r=rg, sig=sg, phi=phig, S=S)  # fmt: skip
