"""What the simulation set itself says about a track, without a network: every simulated stream is a draw from
the prior, so its likelihood for a track is its posterior weight.

    uv run --extra gpu python scripts/extragalactic/sbi/prior_weights.py --config <run>/config.yaml [--gpu 3]

For a track d and every usable stream i of the set (parameters theta_i, one prior draw each):

    w_i = 1 / (S + 1) sum_l  int d sigma_sys / sig_sys_max  L(d | theta_i, progenitor l bins from the first, sigma_sys)

with the likelihood and the two guards of likelihood.py; the progenitor position and sigma_sys are integrated
over their priors (sigma_sys with 8 Gauss-Legendre nodes). Then

    n_eff = (sum w)^2 / sum w^2     how many simulations of the set carry the posterior of this track
    log Z = log(sum w / N)          the evidence, N = all simulated streams with the progenitor state of the track
    sum w_i f(theta_i) / sum w_i    posterior means without a network (trustworthy where n_eff is large)

Tracks: the 32 observed ones, each as given and with its angle reversed (the model has the progenitor moving
anticlockwise, v_y0 > 0, while the sense of an observed stream is unknown), and the first `n_mock` mock tracks
of evaluation.py, weighted with the training streams only. Compared with the network of the run: its
posterior draws (evaluation/draws.npz, evaluation_real/posterior.npz) and its evidence (metrics.json).

The set is read with only the 2 x span_max bins a window can reach: 10 GB instead of the 85 GB
`load_simulations` takes at its peak, with the same mock tracks (checked against evaluation/test_tracks.npz).

Written to <run>/prior_weights/: prior_weights.npz, prior_weights.json, gallery_p*.png (every observed track
with the simulated tracks that carry its posterior), summary.png.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from multiprocessing import Pool

import h5py
import numpy as np

from config import REPO, Config, pick_device

TOP = 4000  # streams kept per track, by weight


def read_shard(args) -> dict:
    path, T, group = args
    with h5py.File(path, "r") as f:
        i0 = int(round(json.loads(f.attrs["config"])["phi_max_deg"] / json.loads(f.attrs["config"])["bin_deg"]))
        s = slice(i0 - T, i0 + T)
        return dict(theta=f["theta"][:], vec=f["c26_vec"][:], status=f["status"][:], n_bad=f["diag/n_bad"][:], dE=f["diag/dE_rel"][:],
                    dissolved=f["prog/dissolved"][:], count=f[f"{group}/count"][:, s],
                    r_med=np.nan_to_num(f[f"{group}/r_med"][:, s]).astype(np.float32), r_hw=np.nan_to_num(f[f"{group}/r_hw"][:, s]).astype(np.float32))  # fmt: skip


def lean_load(cfg: Config, workers: int) -> dict:
    """`load_simulations` with the bins a window can reach only (span_max either side of the progenitor)."""
    from data import Grid, Prior, replacement, run_extents, window_design

    files = sorted(cfg.data_dir().glob("shard_*.h5"))[: cfg.max_shards or None]
    with h5py.File(files[0], "r") as f:
        sc = json.loads(f.attrs["config"])
        group = {f[g].attrs["ordering"]: g for g in ("track", "track_alt") if g in f}[cfg.ordering]
    T = cfg.span_max
    with Pool(workers) as pool:
        parts = pool.map(read_shard, [(p, T, group) for p in files], chunksize=8)
    raw = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    failed, lossy, unbound = (raw["status"] & 16) > 0, raw["n_bad"] > 0.01 * sc["n_particles"], (raw["status"] & 1) > 0
    a, b = run_extents(raw["count"] >= cfg.n_min, T)
    keep = ~failed & ~lossy & (a + b >= cfg.span_min) & (cfg.include_unbound | ~unbound) & ~((raw["dE"] > cfg.de_max) & (cfg.de_max > 0))
    grid = Grid.from_sims(sc)
    n_min, replaced = np.full(len(keep), cfg.n_min, np.float32), np.zeros(len(keep), bool)
    if cfg.replace_tracks:  # as load_simulations
        rep, win = replacement(cfg, sc, grid.i0, raw["theta"], keep), slice(grid.i0 - T, grid.i0 + T)
        if rep["count"].max() >= 2**15:
            raw["count"] = raw["count"].astype(np.int32)
        for k in ("count", "r_med", "r_hw"):
            raw[k][rep["row"]] = rep[k][:, win]
        n_min[rep["row"]], replaced[rep["row"]] = rep["n_min"], True
    prior = Prior.from_sims(sc, cfg.sig_sys_max)
    th = raw["theta"].astype(np.float64)
    phys13 = np.concatenate([th[:, :2], raw["vec"], th[:, 5:]], axis=1)[keep]
    count = raw["count"][keep].astype(np.float32)
    design = window_design(cfg, count, raw["r_hw"][keep], T, n_min[keep])
    data = dict(zeta13=prior.to_latent(phys13, cols=range(13)).astype(np.float32), count=count, r_med=raw["r_med"][keep], r_hw=raw["r_hw"][keep],
                dissolved=raw["dissolved"][keep].astype(np.float32), z_strata=design["Z"], n_min=n_min[keep])  # fmt: skip
    if cfg.scale_aug:
        data["phys13"] = phys13.astype(np.float32)
    return dict(data=data, phys13=phys13, theta=th[keep], prior=prior, grid=grid, design=design, sim_config=sc, dE=raw["dE"][keep],
                nonmono=((raw["status"] & 32) > 0)[keep], row=np.flatnonzero(keep), dissolved_all=raw["dissolved"].astype(bool), replaced=replaced[keep])  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--n-mock", type=int, default=48)
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--var-ratio", type=float, default=None, help="another model-noise guard than the run's (0 = off): what more particles would let in")
    ap.add_argument("--tag", default="", help="results go to <run>/prior_weights<tag>/")
    a = ap.parse_args()
    cfg = Config.load(REPO / a.config)[0]
    out = cfg.run_dir() / f"prior_weights{a.tag}"
    if a.var_ratio is not None:
        cfg.var_ratio = a.var_ratio
    out.mkdir(exist_ok=True)
    sims = lean_load(cfg, a.workers)  # before jax is imported: the pool forks
    print(f"device: {pick_device(False, a.gpu)}; {len(sims['row'])} usable streams", flush=True)
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.12")

    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    import mock
    from data import PARAM_NAMES, load_observed

    grid, T, data = sims["grid"], cfg.span_max, sims["data"]
    N = len(data["zeta13"])
    cut = N - max(1, int(round(cfg.val_frac * N)))  # as data.split
    make = mock.make_mock
    mock.make_mock = lambda c, g, d, bins=None: make(c, g, d, (g.i0 - T, g.i0 + T))  # the arrays hold those bins only

    # --- tracks: observed (as given, reversed) and mock
    tracks = []
    for o in load_observed(cfg):
        for tag, th, r, e in (("", o["theta_deg"], o["r"], o["r_err"]), (" reversed", -o["theta_deg"][::-1], o["r"][::-1], o["r_err"][::-1])):
            t = mock.encode_observation(th, r, e, cfg, grid, dissolved=True, prog=None)
            tracks.append(dict(name=o["name"] + tag, kind="observed" + tag, r=t["r"], sig=t["sig"], mask=t["mask"], phi=t["phi"], S=t["S"], dissolved=True, limit=N))
    held = {k: v[cut:] for k, v in data.items()}
    ev = cfg.run_dir() / "evaluation"
    test = mock.fixed_observations(held, cfg, grid, sims["design"], cfg.seed + 2, cfg.n_test, unweighted=True) if a.n_mock else None
    if test is not None and (ev / "test_tracks.npz").exists():
        saved = np.load(ev / "test_tracks.npz")
        same = saved["zeta"].shape == test["zeta"].shape and np.allclose(saved["zeta"], test["zeta"], atol=1e-5) and np.array_equal(saved["S"], test["S"])
        print(f"mock tracks {'equal' if same else 'DIFFER FROM'} those of evaluation.py", flush=True)
    else:
        same = False
    for i in range(a.n_mock):
        tracks.append(dict(name=f"mock track {i}", kind="mock", r=test["r"][i], sig=test["sig"][i], mask=test["mask"][i], phi=test["phi"][i], S=int(test["S"][i]),
                           dissolved=bool(test["cond"][i, 0] > 0.5), limit=cut, truth=test["zeta"][i], m=test["m"][i], stream=cut + int(test["pick"][i])))  # fmt: skip

    # --- weights of every stream for every track, on the device
    dev = {k: jnp.asarray(data[k]) for k in ("r_med", "r_hw")} | dict(count=jnp.asarray(data["count"].astype(np.int16 if data["count"].max() < 2**15 else np.int32)))
    x, wq = np.polynomial.legendre.leggauss(8)
    u = 0.5 * (x + 1)  # sigma_sys = sig_sys_max u^2: nodes dense near 0, where the likelihood of a tight track lives
    s_nodes, logq = jnp.asarray(cfg.sig_sys_max * u**2, jnp.float32), jnp.asarray(np.log(0.5 * wq * 2 * u), jnp.float32)
    l, j = jnp.arange(T + 1), jnp.arange(T)
    idx = jnp.clip(T - l[:, None] + j[None], 0, 2 * T - 1)

    @jax.jit
    def weigh(dev, lo, r, sig, obs, S):
        sl = lambda v: jax.lax.dynamic_slice_in_dim(v, lo, a.batch, 0)[:, idx]  # (B, L, T)
        m, hw, n = sl(dev["r_med"]), sl(dev["r_hw"]), sl(dev["count"]).astype(jnp.float32)
        ok = jnp.all(~obs | ((n >= cfg.n_min) & (hw**2 / jnp.maximum(n, 1.0) * cfg.var_ratio <= sig**2)), axis=-1) & (l <= S)
        var = sig[:, None] ** 2 + s_nodes[None] ** 2  # (T, G)
        chi = jnp.where(obs, (m - r) ** 2, 0.0) @ jnp.where(obs[:, None], 1.0 / var, 0.0)  # (B, L, G)
        ll = -0.5 * (chi + jnp.sum(jnp.where(obs[:, None], jnp.log(2 * jnp.pi * var), 0.0), axis=0)) + logq
        L = jnp.where(ok, logsumexp(ll, axis=-1), -jnp.inf)
        return logsumexp(L, axis=-1) - jnp.log(S + 1.0), jnp.argmax(L, axis=-1), ok.any(-1)

    starts = [min(s, N - a.batch) for s in range(0, N, a.batch)]
    flag = data["dissolved"] > 0.5
    n_all = {True: int(sims["dissolved_all"].sum()), False: int((~sims["dissolved_all"]).sum())}
    n_all_train = {f: int((sims["dissolved_all"][: sims["row"][cut]] == f).sum()) for f in (True, False)}
    res, tops = [], []
    for k, t in enumerate(tracks):
        logw, best, covers = np.full(N, -np.inf, np.float32), np.zeros(N, np.int16), np.zeros(N, bool)
        args = tuple(jnp.asarray(t[q]) for q in ("r", "sig", "mask")) + (t["S"],)
        for s in starts:
            lw, bl, ok = weigh(dev, s, *args)
            logw[s : s + a.batch], best[s : s + a.batch], covers[s : s + a.batch] = np.asarray(lw), np.asarray(bl), np.asarray(ok)
        use = (flag == t["dissolved"]) & (np.arange(N) < t["limit"])
        logw = np.where(use, logw, -np.inf)
        n_prior = (n_all if t["limit"] == N else n_all_train)[t["dissolved"]]
        mx = float(logw.max())
        if not np.isfinite(mx):
            res.append(dict(name=t["name"], kind=t["kind"], S=int(t["S"]), points=int(t["mask"].sum()), n_covering=0, n_eff=0.0, log_evidence=None))
            tops.append(None)
            continue
        w = np.exp(logw - mx, dtype=np.float64)
        ess = float(w.sum() ** 2 / (w**2).sum())
        top = np.argsort(-logw)[:TOP]
        wn = w / w.sum()
        mean = wn @ data["zeta13"].astype(np.float64)
        sd = np.sqrt(np.maximum(wn @ data["zeta13"].astype(np.float64) ** 2 - mean**2, 0))
        res.append(dict(name=t["name"], kind=t["kind"], S=int(t["S"]), points=int(t["mask"].sum()), n_covering=int((covers & use).sum()), n_eff=ess,
                        log_evidence=mx + math.log(w.sum() / n_prior), unresolved_weight=float(wn[sims["dE"] > 0.01].sum()),
                        largest_weight=float(wn.max()), mean=mean.tolist(), sd=sd.tolist()))  # fmt: skip
        tops.append(dict(stream=top, logw=logw[top], l=best[top]))
        print(f"{t['name']:28s} S {t['S']:2d}  covering {res[-1]['n_covering']:8d}  n_eff {ess:9.1f}  log Z {res[-1]['log_evidence']:9.2f}  "
              f"unresolved weight {res[-1]['unresolved_weight']:.2f}", flush=True)  # fmt: skip

    # --- the network of the run, where it was evaluated
    def network(draws_file, names_wanted):
        if not draws_file.exists():
            return {}
        d = np.load(draws_file, allow_pickle=True)
        at = {str(n): i for i, n in enumerate(d["names"])}
        return {n: d["zeta"][at[n]][:, :13] for n in names_wanted if n in at}

    npe = network(cfg.run_dir() / "evaluation_real" / "posterior.npz", [t["name"] for t in tracks if t["kind"] == "observed"])
    if same:
        npe |= network(ev / "draws.npz", [t["name"] for t in tracks if t["kind"] == "mock"])
    for r in res:
        if r["name"] in npe:
            z = npe[r["name"]].astype(np.float64)
            r["network_mean"], r["network_sd"] = z.mean(0).tolist(), z.std(0).tolist()
    for file, key in ((cfg.run_dir() / "evaluation_real" / "metrics.json", "tracks"), (ev / "metrics.json", "ppc")):
        if file.exists() and (key == "tracks" or same):
            for row in json.loads(file.read_text())[key]:
                for r in res:
                    if r["name"] == row["track"] and "log_evidence" in row:
                        r["network_log_evidence"], r["network_n_eff"] = row["log_evidence"], row.get("effective_samples")
    (out / "prior_weights.json").write_text(json.dumps(dict(run=cfg.name, n_streams=N, n_training=cut, parameters=PARAM_NAMES[:13], tracks=res), indent=1))
    keep = [k for k, tp in enumerate(tops) if tp is not None]
    np.savez(out / "prior_weights.npz", names=[tracks[k]["name"] for k in keep], stream=np.stack([tops[k]["stream"] for k in keep]),
             logw=np.stack([tops[k]["logw"] for k in keep]), l=np.stack([tops[k]["l"] for k in keep]))  # fmt: skip
    figures(out, tracks, res, tops, data, cfg, T)


def figures(out, tracks, res, tops, data, cfg, T) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#0b0b0b"
    obs = [k for k, t in enumerate(tracks) if t["kind"] == "observed"]
    by_name = {r["name"]: r for r in res}
    # gallery: every observed track with the simulated tracks that carry its posterior
    for page in range(-(-len(obs) // 16)):
        part = obs[16 * page : 16 * page + 16]
        fig, axes = plt.subplots(4, 4, figsize=(18, 14), constrained_layout=True)
        for ax, k in zip(axes.ravel(), part):
            t, r, rr = tracks[k], res[k], by_name.get(tracks[k]["name"] + " reversed", {})
            o, tp = t["mask"], tops[k]
            if tp is not None:
                w = np.exp(tp["logw"] - tp["logw"][0])
                for s, l, wi in list(zip(tp["stream"], tp["l"], w))[:25][::-1]:
                    ax.plot(t["phi"][o], data["r_med"][s, T - l + np.arange(T)][o], color=BLUE, lw=0.9, alpha=float(0.15 + 0.85 * wi))
            ax.errorbar(t["phi"][o], t["r"][o], yerr=t["sig"][o], fmt="o", ms=3, lw=0.8, color=INK, zorder=5)
            lz = lambda q: f"{q['log_evidence']:.1f}" if q.get("log_evidence") is not None else "none"
            ax.set_title(f"{t['name']}: {r['points']} bins, {r['n_covering']:,} simulations pass the guards\n"
                         f"n_eff {r['n_eff']:.1f} (reversed {rr.get('n_eff', 0):.1f});  log Z {lz(r)} (reversed {lz(rr)})", fontsize=8.5, loc="left")  # fmt: skip
            ax.set_xlabel("angle minus median angle [deg]", fontsize=8)
            ax.set_ylabel("projected radius [kpc]", fontsize=8)
        for ax in axes.ravel()[len(part) :]:
            ax.set_axis_off()
        fig.suptitle("Observed tracks (black) and the 25 simulated tracks of the set with the largest posterior weight (blue, darker = heavier)", fontsize=11)
        fig.savefig(out / f"gallery_p{page + 1}.png", dpi=110)
        plt.close(fig)

    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for kind, col, lab in (("mock", "#7b8794", "mock tracks (training streams only)"), ("observed", BLUE, "observed tracks, as given"), ("observed reversed", ORANGE, "observed tracks, angle reversed")):
        sel = [r for r in res if r["kind"] == kind]
        ax.scatter([r["points"] for r in sel], [max(r["n_eff"], 0.5) for r in sel], s=26, color=col, label=lab, alpha=0.85, lw=0)
    ax.axhline(100, color=INK, lw=0.8, ls=":")
    ax.set(yscale="log", xlabel="observed bins of the track", ylabel="simulations that carry the posterior, n_eff (0.5 = none)",
           title="How many simulations of the set stand behind the posterior of a track")  # fmt: skip
    ax.legend(fontsize=8, frameon=False)
    pairs = [(by_name[t["name"]], by_name.get(t["name"] + " reversed")) for t in tracks if t["kind"] == "observed"]
    pairs = [(p, q) for p, q in pairs if p.get("log_evidence") is not None and q and q.get("log_evidence") is not None]
    d = np.array([q["log_evidence"] - p["log_evidence"] for p, q in pairs])
    order = np.argsort(d)
    ax2.barh(np.arange(len(d)), d[order], color=[ORANGE if v > 0 else BLUE for v in d[order]])
    ax2.set_yticks(np.arange(len(d)), [pairs[i][0]["name"] for i in order], fontsize=6.5)
    ax2.axvline(0, color=INK, lw=0.8)
    ax2.set(xlabel="log Z (angle reversed) - log Z (as given)", title="Which sense of the angle the model prefers")
    done = [r for r in res if "network_mean" in r and r["n_eff"] >= 100]
    for kind, col in (("mock", "#7b8794"), ("observed", BLUE)):
        sel = [r for r in done if r["kind"] == kind]
        if sel:
            ratio = np.array([np.array(r["network_sd"]) / np.maximum(np.array(r["sd"]), 1e-6) for r in sel])
            shift = np.array([(np.array(r["network_mean"]) - np.array(r["mean"])) / np.maximum(np.array(r["sd"]), 1e-6) for r in sel])
            ax3.scatter(shift.ravel(), ratio.ravel(), s=9, color=col, alpha=0.6, lw=0, label=f"{kind}: {len(sel)} tracks x 13 parameters")
    ax3.axhline(1, color=INK, lw=0.8, ls=":")
    ax3.axvline(0, color=INK, lw=0.8, ls=":")
    ax3.set(yscale="log", xlabel="(network mean - weighted-simulation mean) / weighted-simulation sd, latent parameters",
            ylabel="network sd / weighted-simulation sd", title="Network posterior against the simulations themselves (tracks with n_eff of 100 or more)")  # fmt: skip
    ax3.legend(fontsize=8, frameon=False)
    fig.savefig(out / "summary.png", dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
