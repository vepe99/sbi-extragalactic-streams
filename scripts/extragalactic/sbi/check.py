"""Tests of the pieces a posterior rests on: transforms, mock observations, likelihood, the dataset.

    uv run python scripts/extragalactic/sbi/check.py --cpu --max-shards 200

One line per test; exits non-zero if one fails. Takes the same options as train.py.
"""

from __future__ import annotations

from config import parse

cfg, _ = parse()

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from data import I_PROG, I_SIG, N_PAR, load_observed, load_simulations, orientation  # noqa: E402
from likelihood import importance_weights, log_likelihood, log_prior_latent, model_at  # noqa: E402
from mock import MockObservations, SelectionDataset, encode_observation, fixed_observations, prog_fraction  # noqa: E402
from networks import USES_MASK, build_adapter  # noqa: E402

fails = []


def check(name: str, ok, **info) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  " + "  ".join(f"{k}={v}" for k, v in info.items()), flush=True)
    fails.append(name) if not ok else None


sims = load_simulations(cfg)
prior, grid, data, design = sims["prior"], sims["grid"], sims["data"], sims["design"]
print("data:", sims["stats"])

# --- parameters and latent coordinates
back = prior.from_latent(prior.to_latent(sims["phys13"], cols=range(13)), cols=range(13))
check("latent transform round trip", np.allclose(back, sims["phys13"], rtol=1e-6, atol=1e-6))
zeta_all = prior.to_latent(sims["phys13_all"], cols=range(13))  # every prior draw, before any selection
pv = [stats.kstest(zeta_all[:, k], "norm").pvalue for k in range(13)]
check("the prior is N(0,1) in every latent coordinate (all streams, KS)", min(pv) > 1e-3, min_p=f"{min(pv):.3f}")
o3 = orientation(sims["phys13"][:, 2:5], sims["sim_config"]["q"])
err = max(np.abs(o3[:, 0] - sims["theta"][:, 2]).max(), np.abs(o3[:, 1] - sims["theta"][:, 3]).max(),
          np.abs(np.angle(np.exp(1j * (o3[:, 2] - sims["theta"][:, 4])))).max())  # fmt: skip
check("q, cos_i, psi recovered from the orientation vector", err < 1e-4, max_abs_err=f"{err:.1e}")

# --- mock observations
n = 20000
mo = fixed_observations(data, cfg, grid, design, seed=1, n=n)
live, obs, S, l = mo["weight"] > 0, mo["mask"], mo["S"], mo["l"]  # with the model-noise guard a draw can have no window
check("window lies inside the run of comparable bins", np.all(((l <= mo["a"]) & (S - l <= mo["b"]))[live]), spans=f"{S[live].min()}-{S[live].max()}")
guard = (mo["n"] >= cfg.n_min) & (mo["hw"] ** 2 / np.maximum(mo["n"], 1) * cfg.var_ratio <= mo["sig"] ** 2 * (1 + 1e-5))
check("every observed bin passes both guards of the paper", np.all(guard[live][obs[live]]))
med = np.array([np.median(p[m]) for p, m in zip(mo["phi"], obs)])
check("angles are centred on the median observed angle", np.abs(med).max() < 1e-3)
chi2 = np.sum(np.where(obs, (mo["r"] - mo["m"]) ** 2 / (mo["sig"] ** 2 + mo["sig_sys"][:, None] ** 2), 0.0)) / obs.sum()
check("mock radii scatter as N(m, sigma_data^2 + sigma_sys^2)", abs(chi2 - 1) < 0.02, chi2_per_point=f"{chi2:.4f}")
par2 = prior.from_latent(mo["zeta"][:, [I_SIG, I_PROG]], cols=[I_SIG, I_PROG])
check("sigma_sys and the progenitor position map back from their latent values",
      np.allclose(par2[:, 0], mo["sig_sys"], rtol=1e-3, atol=1e-3) and np.all(np.floor(par2[:, 1] * (S + 1)) == l))
known = mo["fixed_target_mask"][:, I_PROG] == 0
check("only the progenitor position is ever marked as given", np.all(mo["fixed_target_mask"][:, :I_PROG] == 1), given=f"{known.mean():.2f}")
w = mo["weight"]
check("loss weights are finite (exactly 1 with the model-noise guard off)", np.all(np.isfinite(w)) and (cfg.var_ratio > 0 or np.abs(w - 1).max() < 1e-3),
      efficiency=f"{w.sum() ** 2 / (w**2).sum() / n:.2f}", zero=f"{(w == 0).mean():.2f}")  # fmt: skip
print(f"info  weighted share of examples with >= 24 bins: {w[S >= 24].sum() / w.sum():.2f}, >= 40 bins: {w[S >= 40].sum() / w.sum():.2f}")

# --- a real track goes through the same door: encode a mock as if it were observed, at any sky rotation
worst = 0.0
for i in np.flatnonzero(live)[:200]:
    e = encode_observation(mo["phi"][i][obs[i]] + 123.0, mo["r"][i][obs[i]], mo["sig"][i][obs[i]], cfg, grid, dissolved=bool(data["dissolved"][mo["pick"][i]]),
                           prog=prog_fraction(mo["phi"][i][0] + (l[i] - 0.5) * grid.bin_deg, mo["phi"][i][0], int(S[i]), grid.bin_deg))  # fmt: skip
    worst = max(worst, np.abs(e["tokens"][obs[i]] - mo["tokens"][i][obs[i]]).max(), np.abs(e["cond"] - mo["cond"][i]).max(),
                abs(prior.from_latent(e["fixed_target_value"][[I_PROG]], cols=[I_PROG])[0] * (S[i] + 1) - (l[i] + 0.5)))  # fmt: skip
check("real-track encoding reproduces the mock inputs and the progenitor position", worst < 1e-3, max_abs_diff=f"{worst:.1e}")
enc = [encode_observation(o["theta_deg"], o["r"], o["r_err"], cfg, grid, dissolved=True) for o in load_observed(cfg)]
check("the observed tracks fit on the token grid", len(enc) == 32, spans=f"{min(e['S'] for e in enc)}-{max(e['S'] for e in enc)}")

# --- likelihood and importance weights
idx = np.flatnonzero(live)[:2000]
track = {k: data[k][mo["pick"][idx]] for k in ("r_med", "r_hw", "count")}
m, hw, cnt = model_at(track, grid.i0, l[idx], cfg.span_max)
ll = np.asarray(log_likelihood(mo["r"][idx], mo["sig"][idx], obs[idx], m, hw, cnt, mo["sig_sys"][idx, None], cfg.n_min, cfg.var_ratio))
var = mo["sig"][idx] ** 2 + mo["sig_sys"][idx, None] ** 2
hand = -0.5 * np.sum(np.where(obs[idx], (mo["m"][idx] - mo["r"][idx]) ** 2 / var + np.log(2 * np.pi * var), 0.0), axis=1)
check("likelihood of a mock track at its own model equals eq. 4 written by hand", np.allclose(ll, hand, rtol=1e-4, atol=1e-2))
z = np.random.default_rng(0).standard_normal((200000, 2))
iw = importance_weights(z, log_prior_latent(z), -0.5 * (z[:, 0] - 1) ** 2 / 0.25)  # prior N(0,1), data N(1, 0.5^2) -> mean 0.8
check("importance weights recover an analytic posterior mean", abs(np.sum(iw["weights"] * z[:, 0]) - 0.8) < 0.02, efficiency=f"{iw['efficiency']:.2f}")

# --- the training dataset
ds = SelectionDataset(data, seed=3, batch_size=64, adapter=build_adapter(cfg), drop_last=True, augmentations=[MockObservations(cfg, grid, design, 3)])
b = ds[0]
want = dict(inference_variables=(64, N_PAR), inference_conditions=(64, 3), summary_variables=(64, cfg.span_max, 5), sample_weight=(64,), fixed_target_mask=(64, N_PAR))
check("training batches have the shapes the estimator expects", all(b[k].shape == v for k, v in want.items()) and (("summary_mask" in b) == USES_MASK[cfg.summary]))
zbar = data["z_strata"].mean(1)
check("streams are drawn in proportion to Z", np.corrcoef(np.bincount(ds.indices, minlength=len(zbar)), zbar)[0, 1] > 0.5)

raise SystemExit(f"{len(fails)} check(s) failed: {fails}" if fails else print("all checks passed"))
