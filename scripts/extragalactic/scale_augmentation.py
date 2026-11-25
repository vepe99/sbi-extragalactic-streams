"""Tests of the scale augmentation of the training examples (`scale_aug`, sbi/mock.py), before a run uses it.

    uv run --extra gpu python scripts/extragalactic/scale_augmentation.py --config <run file> [--gpu 3] [--routes]

The simulator has no scale of its own: lengths x L and masses x U give the same stream with its track x L
(symmetries.py). The augmentation moves every training example along this family with (ln L, ln U) drawn from
the prior restricted to it (`mock.draw_scale`). Tested here:

  cuts        the loader's cuts (failed integration, lost particles, bins with enough particles, energy error)
              do not depend on the scale: the streams of symmetries.py (`plots/<subdir>/symmetries/sims/`, the
              same key at (L, U) = (1, 1), (1, 2), (1.3, 1), (1.3, 1.3^3)) pass or fail them together;
  end_to_end  a stored track and its parameters scaled by `mock.scale_stream` against those simulations: latent
              coordinates and the track bin by bin;
  invariance  the kernel on every usable stream of the set: the scaled parameters have the marginals of the
              stored ones, dissolved and surviving progenitors separately (two-sample KS between the scaled
              parameters of one random half of the streams and the stored ones of the other half, a new split
              for every sweep; `ks_p_split_only` is the same test with stored parameters on both sides, i.e.
              what the split alone gives), stay inside the prior ranges, and keep the correlations;
  mock        mock observations drawn as in training with the augmentation off and on: the same weighted
              distribution of track length, track radius and parameters, and what the augmentation costs in
              efficiency of the loss weights;
  routes      (--routes; loads `--max-shards` shards with load_simulations) batches made on the device equal those
              made through the host with the augmentation on, and the validation mocks are not scaled.

Written to <out>/plots/<subdir>/scale_augmentation/scale_augmentation.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts" / "extragalactic"), str(REPO / "scripts" / "extragalactic" / "sbi")]
from config import Config, pick_device  # noqa: E402
from data import PARAM_NAMES, Prior, run_extents  # noqa: E402
from prior_weights import lean_load  # noqa: E402

SCALINGS = dict(mass=(1.0, 2.0), length=(1.3, 1.0), both=(1.3, 1.3**3))  # symmetries.py: name -> (L, U)
DE_MAX = 0.01


def read_sim(d: Path) -> dict:
    cols = dict(index="index", theta="theta", vec="c26_vec", status="status", n_bad="diag/n_bad", dE="diag/dE_rel", dissolved="prog/dissolved",
                count="track/count", r_med="track/r_med", r_hw="track/r_hw")  # fmt: skip
    raw = {k: [] for k in cols}
    files = sorted((d / "data").glob("shard_*.h5"))
    for p in files:
        with h5py.File(p, "r") as f:
            for k, path in cols.items():
                raw[k].append(f[path][:])
    with h5py.File(files[0], "r") as f:
        sc = json.loads(f.attrs["config"])
    order = np.argsort(np.concatenate(raw["index"]))
    out = {k: np.concatenate(v)[order] for k, v in raw.items()}
    out["theta64"] = np.load(d / "theta.npz")["theta"]  # the parameters as they were simulated
    return out | dict(r_med=np.nan_to_num(out["r_med"]), r_hw=np.nan_to_num(out["r_hw"]), sc=sc)


def wstats(x, w) -> tuple[float, float]:
    """Weighted mean and its standard error."""
    m = np.sum(w * x) / w.sum()
    return float(m), float(np.sqrt(np.sum(w**2 * (x - m) ** 2)) / w.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--n-mock", type=int, default=2_000_000, help="mock observations of the `mock` test, with the augmentation off and on")
    ap.add_argument("--sweeps", type=int, default=3, help="the kernel applied this many times in a row in the invariance test")
    ap.add_argument("--routes", action="store_true", help="only the `routes` test")
    ap.add_argument("--max-shards", type=int, default=40, help="shards loaded for the `routes` test")
    a = ap.parse_args()
    cfg = dataclasses.replace(Config.load(REPO / a.config)[0], scale_aug=True)
    out = REPO / cfg.out / "plots" / cfg.subdir / "scale_augmentation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "scale_augmentation.json"
    num = json.loads(path.read_text()) if path.exists() else {}
    sims = None if a.routes else lean_load(cfg, a.workers)  # before jax is imported: the pool forks
    print(f"device: {pick_device(a.cpu, a.gpu)}", flush=True)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    import jax.random as jr

    import mock

    if a.routes:
        from data import load_simulations, split
        from networks import build_adapter, network_names

        c = dataclasses.replace(cfg, max_shards=a.max_shards)
        s = load_simulations(c)
        data, grid, design, prior = s["data"], s["grid"], s["design"], s["prior"]
        dd = mock.DeviceDataset(data, c, grid, design, seed=3, batch_size=256, names=network_names(c), prior=prior)
        ds = mock.SelectionDataset(data, seed=3, batch_size=256, adapter=build_adapter(c), drop_last=True, augmentations=[mock.MockObservations(c, grid, design, 3, prior=prior)])
        pairs = [tuple({k: np.asarray(v) for k, v in q[i].items()} for q in (ds, dd)) for i in range(8)]
        diff = max(float(np.abs(h[k] - d[k]).max()) for h, d in pairs for k in h)
        same_keys = all(set(h) == set(d) and all(h[k].dtype == d[k].dtype for k in h) for h, d in pairs)
        raw = mock.MockObservations(c, grid, design, 3, extras=True, prior=prior)({k: v[ds.indices[:256]] for k, v in data.items()})
        moved = float((np.abs(raw["zeta"][:, :13] - data["zeta13"][ds.indices[:256]]).max(1) > 1e-3).mean())
        held = split(s, c.val_frac)[1]
        val = mock.fixed_observations(held, c, grid, design, c.seed + 1, 4096)
        val_off = mock.fixed_observations(held, dataclasses.replace(c, scale_aug=False), grid, design, c.seed + 1, 4096)
        num["routes"] = dict(shards=a.max_shards, streams=int(len(data["zeta13"])), replaced=int(s["replaced"].sum()), batches=8, batch=256, same_keys_and_dtypes=bool(same_keys),
                             max_abs_diff_device_vs_host=diff, share_of_examples_scaled=moved,
                             validation_unscaled=bool(np.array_equal(val["zeta"][:, :13], held["zeta13"][val["pick"]])),
                             validation_same_as_without_augmentation=bool(all(np.array_equal(val[k], val_off[k]) for k in val_off)))  # fmt: skip
        num["routes"]["pass"] = bool(same_keys and diff < 1e-4 and num["routes"]["validation_unscaled"] and num["routes"]["validation_same_as_without_augmentation"])
        path.write_text(json.dumps(num, indent=1))
        print(json.dumps(num["routes"], indent=1))
        return

    prior, grid, d, T = sims["prior"], sims["grid"], sims["data"], cfg.span_max
    sc = sims["sim_config"]

    # --- cuts and end to end, on the simulations of symmetries.py
    base = REPO / cfg.out / "plots" / cfg.subdir / "symmetries" / "sims"
    ref = read_sim(base / "same")
    N0, i0 = ref["sc"]["n_particles"], grid.i0
    dis, res = ref["dissolved"].astype(bool), ref["dE"] <= DE_MAX
    groups = {"surviving, resolved": ~dis & res, "dissolved, resolved": dis & res, "surviving, unresolved": ~dis & ~res, "dissolved, unresolved": dis & ~res}
    phys = np.concatenate([ref["theta64"][:, :2], ref["vec"], ref["theta64"][:, 5:]], axis=1)
    z_ref = prior.to_latent(phys, cols=range(13)).astype(np.float32)
    lo, hi = np.array(prior.a[:13]), np.array(prior.b[:13])
    uni = np.array([k == "U" for k in prior.kind[:13]])
    scale = jax.jit(jax.vmap(lambda p, z, m, h, sa, sb: mock.scale_stream(p, z, m, h, sa, sb, prior)))
    cut = lambda r: dict(failed=(r["status"] & 16) > 0, lossy=r["n_bad"] > 0.01 * N0, short=np.add(*run_extents(r["count"] >= cfg.n_min, i0)) < cfg.span_min,
                         unresolved=r["dE"] > DE_MAX)  # noqa: E731  # fmt: skip
    c0 = cut(ref)
    num["cuts"], num["end_to_end"] = dict(n_streams=int(len(dis)), groups={g: int(s.sum()) for g, s in groups.items()}), {}
    for name, (L, U) in SCALINGS.items():
        r = read_sim(base / name)
        c1 = cut(r)
        num["cuts"][name] = dict(L=L, U=U, **{g: {k: dict(share=float(c0[k][s].mean()), changed=int((c0[k] != c1[k])[s].sum())) for k in c0}
                                               | dict(identical_counts=float((ref["count"] == r["count"]).all(1)[s].mean())) for g, s in groups.items() if s.any()})  # fmt: skip
        n = len(dis)
        z, m, h = (np.asarray(x) for x in scale(jnp.asarray(phys, jnp.float32), jnp.asarray(z_ref), jnp.asarray(ref["r_med"]), jnp.asarray(ref["r_hw"]),
                                                jnp.full(n, np.log(L), jnp.float32), jnp.full(n, np.log(U), jnp.float32)))  # fmt: skip
        phys_new = np.concatenate([r["theta64"][:, :2], ref["vec"], r["theta64"][:, 5:]], axis=1)  # the parameters the scaled run was simulated with
        z_want = prior.to_latent(phys_new, cols=range(13))
        inside = np.all(~uni | ((phys_new >= lo) & (phys_new <= hi)), axis=1)  # outside the ranges both clip
        ok = (ref["count"] >= cfg.n_min) & (r["count"] >= cfg.n_min)
        with np.errstate(all="ignore"):
            rel_m = np.where(ok, np.abs(m / r["r_med"] - 1.0), 0.0).max(1)
            rel_h = np.where(ok & (r["r_hw"] > 0), np.abs(h / r["r_hw"] - 1.0), 0.0).max(1)
        row = dict(L=L, U=U, latent_max_abs_diff=float(np.abs(z - z_want)[inside].max()), latent_max_abs_diff_all=float(np.abs(z - z_want).max()),
                   inside_prior_ranges=float(inside.mean()))  # fmt: skip
        for g, s in groups.items():
            if s.any():
                row[g] = dict(r_med_worst_bin_median=float(np.median(rel_m[s])), r_med_worst_bin_p90=float(np.percentile(rel_m[s], 90)), r_med_within_1e3=float((rel_m[s] < 1e-3).mean()),
                              r_hw_worst_bin_median=float(np.median(rel_h[s])), r_hw_within_1e3=float((rel_h[s] < 1e-3).mean()))  # fmt: skip
        num["end_to_end"][name] = row
    e = num["end_to_end"]
    num["end_to_end"]["pass"] = bool(max(e[k]["latent_max_abs_diff"] for k in SCALINGS) < 2e-3 and min(e[k]["surviving, resolved"]["r_med_within_1e3"] for k in SCALINGS) > 0.97
                                     and e["length"]["dissolved, resolved"]["r_med_within_1e3"] > 0.97)  # fmt: skip
    num["cuts"]["pass"] = bool(all(v["changed"] == 0 for name in SCALINGS for g in ("surviving, resolved",) + (("dissolved, resolved",) if name == "length" else ())
                                   for k, v in num["cuts"][name][g].items() if isinstance(v, dict) and k != "unresolved"))  # fmt: skip
    print(json.dumps(dict(cuts=num["cuts"], end_to_end=num["end_to_end"]), indent=1), flush=True)

    # --- invariance of the kernel on every usable stream
    def one(key, p, ds):
        sa, sb = mock.draw_scale(key, p, ds, prior, cfg.scale_points)
        p2 = mock.scale_params(p, sa, sb)
        return p2, mock.latent13(p2, prior), sa, sb

    kern = jax.jit(jax.vmap(one))
    N, B = len(d["phys13"]), 200_000
    z0, dissolved = d["zeta13"], d["dissolved"] > 0.5
    p, sweeps = d["phys13"], []
    for sweep in range(a.sweeps):
        parts = [kern(jr.split(jr.fold_in(jr.key(cfg.seed + sweep), s), min(B, N - s)), jnp.asarray(p[s : s + B]), jnp.asarray(d["dissolved"][s : s + B])) for s in range(0, N, B)]
        p, z, sa, sb = (np.concatenate([np.asarray(q[k]) for q in parts]) for k in range(4))
        sweeps.append(dict(z=z, a=sa, b=sb, p=p))
    z_plain = np.asarray(jax.jit(jax.vmap(lambda q: mock.latent13(q, prior)))(jnp.asarray(d["phys13"][:B])))
    inv = dict(n_streams=N, points=cfg.scale_points, latent_single_precision_max_abs_diff=float(np.abs(z_plain - z0[:B]).max()),
               scaled_columns=[PARAM_NAMES[k] for k in (0, 1, 5, 6, 7, 8, 9, 10, 11, 12)])  # fmt: skip
    worst_p = 1.0
    for g, s in (("surviving", ~dissolved), ("dissolved", dissolved)):
        row = dict(n=int(s.sum()))
        for i, sw in enumerate(sweeps):
            half = np.random.default_rng(cfg.seed + i).random(N) < 0.5
            ks = [stats.ks_2samp(sw["z"][s & half, k], z0[s & ~half, k]) for k in range(13)]
            ks0 = [stats.ks_2samp(z0[s & half, k], z0[s & ~half, k]) for k in range(13)]
            pin = sw["p"][s]
            tol = 1e-5 * np.maximum(np.abs(lo), np.abs(hi))
            c_new, c_old = np.corrcoef(sw["z"][s][::5].T), np.corrcoef(z0[s][::5].T)
            row[f"sweep_{i + 1}"] = dict(ks_D={PARAM_NAMES[k]: float(ks[k].statistic) for k in range(13)}, ks_p={PARAM_NAMES[k]: float(ks[k].pvalue) for k in range(13)},
                                         ks_p_split_only={PARAM_NAMES[k]: float(ks0[k].pvalue) for k in range(13)},
                                         min_p=float(min(q.pvalue for q in ks)), outside_ranges=int((uni & ((pin < lo - tol) | (pin > hi + tol))).any(1).sum()),
                                         correlation_max_abs_change=float(np.abs(c_new - c_old).max()))  # fmt: skip
            worst_p = min(worst_p, row[f"sweep_{i + 1}"]["min_p"])
        sw = sweeps[0]
        row |= dict(stayed=float(((sw["a"] == 0) & (sw["b"] == 0))[s].mean()), lnL_sd=float(sw["a"][s].std()), lnU_sd=float(sw["b"][s].std()),
                    L_p01_p99=[float(x) for x in np.exp(np.percentile(sw["a"][s], [1, 99]))], U_p01_p99=[float(x) for x in np.exp(np.percentile(sw["b"][s], [1, 99]))],
                    latent_shift_sd={PARAM_NAMES[k]: float((sw["z"][s, k] - z0[s, k]).std()) for k in range(13)})  # fmt: skip
        inv[g] = row
    inv["pass"] = bool(worst_p > 1e-3 and all(inv[g][f"sweep_{i + 1}"]["outside_ranges"] == 0 for g in ("surviving", "dissolved") for i in range(a.sweeps)))
    num["invariance"] = inv
    print(json.dumps(inv, indent=1), flush=True)
    del sweeps

    # --- the mock observations with the augmentation off and on
    lo_b, hi_b = grid.i0 - T, grid.i0 + T
    assert d["count"].shape[1] == 2 * T
    zbar = d["z_strata"].mean(1).astype(np.float64)
    pick = np.random.default_rng(cfg.seed + 3).choice(N, a.n_mock, p=zbar / zbar.sum())
    res = {}
    for tag, c in (("off", dataclasses.replace(cfg, scale_aug=False)), ("on", cfg)):
        fn = jax.jit(jax.vmap(mock.make_mock(c, grid, sims["design"], (lo_b, hi_b), prior)))
        parts = []
        for i, s in enumerate(range(0, a.n_mock, 16384)):
            ix = pick[s : s + 16384]
            args = [jnp.asarray(d[k][ix]) for k in ("count", "r_med", "r_hw", "dissolved", "z_strata", "n_min")] + ([jnp.asarray(d["phys13"][ix]), jnp.asarray(d["zeta13"][ix])] if c.scale_aug else [])
            o = fn(jr.split(jr.fold_in(jr.key(cfg.seed + 5), i), len(ix)), *args)
            parts.append({k: np.asarray(o[k]) for k in ("weight", "S", "cond") + (("zeta13", "scale") if c.scale_aug else ())})
        res[tag] = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    res["off"]["zeta13"] = d["zeta13"][pick]
    mk = dict(n=a.n_mock)
    for tag, r in res.items():
        w = r["weight"].astype(np.float64)
        mk[tag] = dict(mean_weight=wstats(w, np.ones_like(w)), efficiency=float(w.sum() ** 2 / (w**2).sum() / len(w)), zero=float((w == 0).mean()), max_over_mean=float(w.max() / w.mean()))
        for name, x in (("S>=13", r["S"] >= 13), ("S>=26", r["S"] >= 26), ("S>=36", r["S"] >= 36), ("dissolved", r["cond"][:, 0]), ("log10(rbar/30)", r["cond"][:, 2]),
                        ("log10(rbar/30)^2", r["cond"][:, 2] ** 2)) + tuple((f"zeta {PARAM_NAMES[k]}", r["zeta13"][:, k]) for k in range(13)) + tuple((f"zeta^2 {PARAM_NAMES[k]}", r["zeta13"][:, k] ** 2) for k in range(13)):  # fmt: skip
            mk[tag][name] = wstats(x.astype(np.float64), w)
        for cls, s in (("long", r["S"] >= 26), ("short", r["S"] <= 12)):
            mk[tag][f"efficiency_{cls}"] = float(w[s].sum() ** 2 / (w[s] ** 2).sum() / max(int((s & (w > 0)).sum()), 1))
    zs = {k: (mk["on"][k][0] - mk["off"][k][0]) / np.hypot(mk["on"][k][1], mk["off"][k][1]) for k in mk["on"] if isinstance(mk["on"][k], tuple) or isinstance(mk["on"][k], list)}
    mk["z_scores_on_minus_off"] = {k: float(v) for k, v in zs.items()}
    mk["largest_abs_z"] = float(max(abs(v) for v in zs.values()))
    mk["pass"] = bool(mk["largest_abs_z"] < 4.0)
    num["mock"] = mk
    path.write_text(json.dumps(num, indent=1))
    print(json.dumps(mk, indent=1))
    print("pass:", {k: num[k]["pass"] for k in ("cuts", "end_to_end", "invariance", "mock")})


if __name__ == "__main__":
    main()
