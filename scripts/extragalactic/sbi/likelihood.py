"""The likelihood of Chemaly et al. 2026b, written out for importance sampling.

For data {r_i, sigma_i} in N angular bins and the model track m_i(theta) in the same bins (their eq. 4):

    log L = -1/2 sum_i [ (m_i - r_i)^2 / (sigma_i^2 + sigma_sys^2) + log(2 pi (sigma_i^2 + sigma_sys^2)) ]

and L = 0 when the model cannot be compared with the data (their sec. 3.3): a bin with fewer than n_min model
particles, or a model track noisier than a third of the data error, h_i^2 / n_i > sigma_i^2 / var_ratio. The
paper returns a large negative number there to help its sampler; for weights the two are the same.

Posterior samples zeta ~ q(. | data) get self-normalised importance weights

    w  propto  N(zeta; 0, I) L(data | zeta) / q(zeta | data)

with no Jacobian, because the prior is a standard normal in the latent coordinates (data.py). The model track
of a sample comes from generate.py (`--theta-file`); evaluation.py does that for the predictive checks.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np


def log_likelihood(r, sig, obs, m, hw, n, sig_sys, n_min: float = 3.0, var_ratio: float = 9.0):
    """r, sig, obs: data radii, errors [kpc] and observed flag per bin; m, hw, n: model median radius, 16-84
    half-width and particle count in the same bins; sig_sys [kpc]. -inf where a guard fails."""
    var = sig**2 + sig_sys**2
    ll = -0.5 * jnp.sum(jnp.where(obs, (m - r) ** 2 / var + jnp.log(2 * jnp.pi * var), 0.0), axis=-1)
    ok = jnp.all(~obs | ((n >= n_min) & (hw**2 / jnp.maximum(n, 1.0) * var_ratio <= sig**2)), axis=-1)
    return jnp.where(ok, ll, -jnp.inf)


FLOOR, STEP_NOISY, STEP_FEW = -1e9, 1e10, 1e12


def log_likelihood_steps(r, sig, obs, m, hw, n, sig_sys, n_min: float = 3.0, var_ratio: float = 9.0):
    """`log_likelihood` for a sampler (nested.py): equal to it where both guards hold, and where one fails a
    finite number that rises towards the region where they hold, instead of -inf. The paper's code does the
    same for dynesty (StreaMAX-fit `_track_loglike`: -1e15 per bin with too few particles, else -1e12 per bin
    that is too noisy). Here a bin with too few particles costs 1e12 and a noisy bin 1e10, and eq. 4 over the
    bins that can be compared is added, so that two points on the same step are not tied (dynesty stops if
    all its live points are). Eq. 4 is floored at -1e9, far below any value of a model that passes the guards
    (-5e8 is 28 bins each 1000 kpc off with errors of 0.17 kpc), so a step is below all of them. Posterior and
    evidence are unchanged: exp(-1e10) is zero."""
    var = sig**2 + sig_sys**2
    few = obs & (n < n_min)
    noisy = obs & ~few & (hw**2 / jnp.maximum(n, 1.0) * var_ratio > sig**2)
    ll = -0.5 * jnp.sum(jnp.where(obs & ~few & ~noisy, (m - r) ** 2 / var + jnp.log(2 * jnp.pi * var), 0.0), axis=-1)
    return jnp.maximum(ll, FLOOR) - STEP_FEW * jnp.sum(few, axis=-1) - STEP_NOISY * jnp.sum(noisy, axis=-1)


def model_at(track: dict, i0: int, l, T: int) -> tuple:
    """Model track (m, hw, n) in the T bins of a window that has l bins left of the progenitor.
    track: r_med, r_hw, count on the simulation grid (..., n_bins); l: integer array broadcastable to it."""
    bins = jnp.clip(i0 - jnp.asarray(l)[..., None] + jnp.arange(T), 0, track["count"].shape[-1] - 1)
    return tuple(jnp.take_along_axis(jnp.asarray(track[k], jnp.float32), bins, axis=-1) for k in ("r_med", "r_hw", "count"))


def log_prior_latent(zeta):
    return -0.5 * jnp.sum(jnp.asarray(zeta) ** 2, axis=-1) - 0.5 * np.shape(zeta)[-1] * math.log(2 * math.pi)


def importance_weights(zeta, log_q, log_like) -> dict:
    """-> normalised weights, effective sample size, its share of the samples, and the log evidence estimate."""
    lw = np.asarray(log_prior_latent(zeta)) + np.asarray(log_like) - np.asarray(log_q)
    lw = np.where(np.isfinite(lw), lw, -np.inf)
    if not np.isfinite(lw.max()):
        return dict(weights=np.zeros_like(lw), ess=0.0, efficiency=0.0, log_evidence=-np.inf)
    w = np.exp(lw - lw.max())
    ess = float(w.sum() ** 2 / (w**2).sum())
    return dict(weights=w / w.sum(), ess=ess, efficiency=ess / len(w), log_evidence=float(lw.max() + np.log(w.mean())))
