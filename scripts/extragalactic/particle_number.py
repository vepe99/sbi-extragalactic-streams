"""Do more particles make the long streams better? A small set of good prior samples of a simulation set,
simulated again with more particles.

    uv run --extra gpu python scripts/extragalactic/particle_number.py --config <run>/config.yaml

GOOD PRIOR SAMPLES are taken from the set itself (`subdir` of the file), in two groups:

  long   streams with a resolved progenitor orbit (energy error <= 1 %) whose unbroken run of bins with
         >= n_min particles around the progenitor is at least `long_bins` bins: the streams a long mock track
         can be cut from. `n_long` of them, at random;
  fits   for every observed track of more than `min_points` bins and both senses of its angle, the `top`
         simulations with the largest posterior weight when the particle-noise guard is off
         (sbi/prior_weights.py --var-ratio 0 --tag _guard1): the simulations that fit the long observed
         tracks best, whether or not the guard lets them count.

The same parameters are simulated with every particle number of `particles` (a new key each; the first is the
set's own number, the control: the stored tracks of the `fits` group were picked for fitting, noise included,
so only simulations made after the pick compare fairly). The largest numbers run on the `fits` group and
`n_sub` of the `long` group only. Guard 1 is kept at the same FRACTION of the particles (n_min x N / N_set), as
the paper defines it; guard 2 is unchanged, so it lets in more as the particle noise falls.

Measured, per particle number:
  usable length   bins of the unbroken run that pass both guards at a flat data error of 0.3, 1 and 3 kpc
  observed tracks posterior weight (sbi/prior_weights.py) of the simulated samples for every long observed track:
                  how many pass the guards, the best log-likelihood, the summed weight, the effective sample
  network         loss of the trained run of the file on mock tracks of at least `long_bins` bins cut from the
                  `long` streams, with the model track of each particle number under the same window, errors
                  and noise: does a cleaner model track change what the network infers

Written to <out>/plots/<subdir>/particle_number/: selection.npz, particle_number.json, good_samples.png,
particle_number.png, gallery.png; the simulations in sims/ (reused).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts" / "extragalactic"), str(REPO / "scripts" / "extragalactic" / "sbi")]
import long_streams as LS  # noqa: E402
import track_noise as TN  # noqa: E402
from config import Config, pick_device  # noqa: E402
from prior_weights import lean_load  # noqa: E402

P, plt = LS.P, LS.plt
NAMES13 = ["log M", "R_s", "q", "cos i", "psi", "log m", "r_s", "x0", "z0", "vx0", "vy0", "vz0", "t_int"]  # generate.py's theta


def select(cfg: Config, sims: dict, a) -> dict:
    from data import run_extents

    T, d = cfg.span_max, sims["data"]
    la, lb = run_extents(d["count"] >= cfg.n_min, T)
    run = la + lb
    with np.errstate(all="ignore"):
        j = np.arange(2 * T)[None]
        inrun = (j >= T - la[:, None]) & (j < T + lb[:, None])
        width = np.nanmedian(np.where(inrun, d["r_hw"] / np.maximum(d["r_med"], 1e-3), np.nan), axis=1)
    long = (sims["dE"] <= LS.DE_MAX) & (run >= a.long_bins)
    rng = np.random.default_rng(cfg.seed)
    pick_long = rng.choice(np.flatnonzero(long), min(a.n_long, int(long.sum())), replace=False)
    pw = np.load(cfg.run_dir() / "prior_weights_guard1" / "prior_weights.npz", allow_pickle=True)
    meta = {t["name"]: t for t in json.loads((cfg.run_dir() / "prior_weights_guard1" / "prior_weights.json").read_text())["tracks"]}
    tops = lambda k_: np.unique(np.concatenate([pw["stream"][k][:k_] for k, n in enumerate(pw["names"]) if meta[str(n)]["points"] > a.min_points]))
    fits = tops(a.top)
    stream = np.unique(np.r_[pick_long, fits])
    sub = np.unique(np.r_[tops(a.top_sub), pick_long[: a.n_sub]])
    return dict(stream=stream, is_fit=np.isin(stream, fits), is_long=np.isin(stream, pick_long), in_sub=np.isin(stream, sub), long_all=long, run=run, width=width)


def fig_good_samples(sims: dict, sel: dict, cfg: Config, out: Path) -> dict:
    """Where in the prior the long, resolved streams are: the 13 simulated parameters of the `long` group against
    those of all usable streams of the set."""
    th, long = sims["theta"], sel["long_all"]
    fig, axes = plt.subplots(3, 5, figsize=(17, 9))
    num = dict(n_usable=int(len(th)), n_long=int(long.sum()), share_of_usable=float(long.mean()), median={})
    for ax, k, name in zip(axes.ravel(), range(13), NAMES13):
        lo, hi = np.percentile(th[:, k], [0.2, 99.8])
        e = np.linspace(lo, hi, 50)
        ax.stairs(np.histogram(th[:, k], e, density=True)[0], e, color=P.INK2, lw=1.2, label="all usable streams")
        ax.stairs(np.histogram(th[long, k], e, density=True)[0], e, color=P.C_PASS, lw=1.8, label=f"long and resolved ({long.mean():.1%})")
        ax.set_title(name, fontsize=10)
        ax.set_yticks([])
        num["median"][name] = dict(all=float(np.median(th[:, k])), long=float(np.median(th[long, k])))
    ax = axes.ravel()[13]
    e = np.arange(6, 99, 2)
    ax.stairs(np.histogram(sel["run"], e)[0], e, color=P.INK2, lw=1.2)
    ax.stairs(np.histogram(sel["run"][sims["dE"] <= LS.DE_MAX], e)[0], e, color=P.C_PASS, lw=1.8)
    ax.axvline(cfg.span_max, color=P.INK, lw=0.8, ls=":")
    ax.set(yscale="log", title="run of bins with enough particles (all; resolved)")
    ax.title.set_fontsize(9)
    axes.ravel()[14].set_axis_off()
    axes.ravel()[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle(f"Good prior samples for long tracks: resolved orbit and a run of at least {int(sel['run'][long].min())} bins "
                 f"({long.sum():,} of {len(th):,} usable streams)", fontsize=11)  # fmt: skip
    fig.tight_layout()
    LS.save(fig, out, "good_samples")
    return num


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--particles", default="30000,90000,300000")
    ap.add_argument("--long-bins", type=int, default=26)
    ap.add_argument("--n-long", type=int, default=5000)
    ap.add_argument("--n-sub", type=int, default=600, help="streams of the long group that also get the largest particle numbers")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--top-sub", type=int, default=40, help="best fits per track that also get the largest particle numbers")
    ap.add_argument("--cpu-workers", type=int, default=0, help="simulate and analyse on the CPU with this many simulator processes (no free GPU)")
    ap.add_argument("--cpu-cores", default="", help="cores the simulator processes share out, e.g. 32-63,160-191; give them, or each process uses the whole machine")
    ap.add_argument("--shards-per-worker", type=int, default=4)
    ap.add_argument("--min-points", type=int, default=16)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    cfg = Config.load(REPO / a.config)[0]
    Ns = [int(x) for x in a.particles.split(",")]
    out = REPO / cfg.out / "plots" / cfg.subdir / "particle_number"
    out.mkdir(parents=True, exist_ok=True)
    sims = lean_load(cfg, a.workers)  # before jax is imported: the pool forks
    print(f"device: {pick_device(a.cpu_workers > 0, a.gpu)}", flush=True)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sel = select(cfg, sims, a)
    num = dict(particles=Ns, selection=dict(n=int(len(sel["stream"])), fits=int(sel["is_fit"].sum()), long=int(sel["is_long"].sum()), sub=int(sel["in_sub"].sum())))
    num["good_samples"] = fig_good_samples(sims, sel, cfg, out)
    st, T, grid, sc, N0 = sel["stream"], cfg.span_max, sims["grid"], sims["sim_config"], sims["sim_config"]["n_particles"]
    theta, dis = sims["theta"][st], sims["data"]["dissolved"][st] > 0.5
    np.savez(out / "selection.npz", stream=st, theta=theta, dissolved=dis, is_fit=sel["is_fit"], is_long=sel["is_long"], in_sub=sel["in_sub"])
    print(f"selected {len(st)} streams: {num['selection']}", flush=True)

    # --- the same parameters with every particle number
    win = slice(grid.i0 - T, grid.i0 + T)
    tracks = {"stored": dict(count=sims["data"]["count"][st], r_med=sims["data"]["r_med"][st], r_hw=sims["data"]["r_hw"][st], rows=np.arange(len(st)), N=N0)}
    for N in Ns:
        rows = np.arange(len(st)) if N <= Ns[min(1, len(Ns) - 1)] else np.flatnonzero(sel["in_sub"])
        r = TN.simulate(theta[rows], dis[rows], sc | dict(n_particles=N), cfg, out / f"sims/n{N}", 21,
                        batch=max(1, int(8 * N0 / N)) if a.cpu_workers else max(8, int(128 * N0 / N)), cpu_workers=a.cpu_workers, cpu_cores=a.cpu_cores)
        tracks[f"{N}"] = dict(count=r["count"][:, win], r_med=r["r_med"][:, win], r_hw=r["r_hw"][:, win], rows=rows, N=N, failed=r["failed"], dE=r["dE"])

    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    import mock
    from data import load_observed, run_extents

    # --- usable length at a flat data error
    num["usable_length"] = {}
    for key, t in tracks.items():
        n_min = cfg.n_min * t["N"] / N0
        num["usable_length"][key] = dict(n_min=n_min)
        # "sub": the long streams every particle number has, so that the numbers compare across them
        for scope, grp in (("all", sel["is_long"][t["rows"]]), ("sub", (sel["is_long"] & sel["in_sub"])[t["rows"]])):
            row = dict(n=int(grp.sum()))
            for sig in (0.3, 1.0, 3.0):
                ok = (t["count"] >= n_min) & (t["r_hw"].astype(np.float64) ** 2 / np.maximum(t["count"], 1) * cfg.var_ratio <= sig**2)
                la, lb = run_extents(ok, T)
                L = (la + lb)[grp]
                row[f"sigma_{sig:g}"] = dict(mean_bins=float(L.mean()), at_least_26=float((L >= 26).mean()), at_least_36=float((L >= 36).mean()))
            num["usable_length"][key][scope] = row

    # --- posterior weights for the long observed tracks
    x, wq = np.polynomial.legendre.leggauss(8)
    u = 0.5 * (x + 1)
    s_nodes, logq = jnp.asarray(cfg.sig_sys_max * u**2, jnp.float32), jnp.asarray(np.log(0.5 * wq * 2 * u), jnp.float32)
    l, j = jnp.arange(T + 1), jnp.arange(T)
    idx = jnp.clip(T - l[:, None] + j[None], 0, 2 * T - 1)

    @jax.jit
    def weigh(r_med, r_hw, count, n_min, r, sig, obs, S):
        m, hw, n = r_med[:, idx], r_hw[:, idx], count[:, idx]
        enough = jnp.all(~obs | (n >= n_min), axis=-1) & (l <= S)
        quiet = jnp.all(~obs | (hw**2 / jnp.maximum(n, 1.0) * cfg.var_ratio <= sig**2), axis=-1)
        var = sig[:, None] ** 2 + s_nodes[None] ** 2
        chi = jnp.where(obs, (m - r) ** 2, 0.0) @ jnp.where(obs[:, None], 1.0 / var, 0.0)
        ll = logsumexp(-0.5 * (chi + jnp.sum(jnp.where(obs[:, None], jnp.log(2 * jnp.pi * var), 0.0), axis=0)) + logq, axis=-1)
        both = jnp.where(enough & quiet, ll, -jnp.inf)
        return logsumexp(both, axis=-1) - jnp.log(S + 1.0), jnp.argmax(both, axis=-1), logsumexp(jnp.where(enough, ll, -jnp.inf), axis=-1) - jnp.log(S + 1.0)

    obs_tracks = []
    for o in load_observed(cfg):
        if len(o["r"]) <= a.min_points:
            continue
        for tag, th_, r_, e_ in (("", o["theta_deg"], o["r"], o["r_err"]), (" reversed", -o["theta_deg"][::-1], o["r"][::-1], o["r_err"][::-1])):
            t = mock.encode_observation(th_, r_, e_, cfg, grid, dissolved=True, prog=None)
            obs_tracks.append(dict(name=o["name"] + tag, points=len(o["r"]), r=t["r"], sig=t["sig"], mask=t["mask"], phi=t["phi"], S=t["S"]))
    num["observed"], best = {}, {}
    for t in obs_tracks:
        row = {}
        for key, tr in tracks.items():
            use = dis[tr["rows"]]  # the observed tracks are fitted with a dissolved progenitor
            sub = np.isin(tr["rows"], np.flatnonzero(sel["in_sub"]))  # the streams every particle number has
            lw, bl, lw1 = (np.asarray(v) for v in weigh(jnp.asarray(tr["r_med"]), jnp.asarray(tr["r_hw"]), jnp.asarray(tr["count"], jnp.float32), cfg.n_min * tr["N"] / N0,
                                                       jnp.asarray(t["r"]), jnp.asarray(t["sig"]), jnp.asarray(t["mask"]), t["S"]))  # fmt: skip
            for scope, m in (("all", use), ("sub", use & sub)):
                v, v1 = np.where(m, lw, -np.inf), np.where(m, lw1, -np.inf)
                fin = np.isfinite(v)
                w = np.exp(v[fin] - v[fin].max()) if fin.any() else np.zeros(0)
                row[f"{key}/{scope}"] = dict(n=int(m.sum()), pass_both=int(fin.sum()), pass_guard1=int(np.isfinite(v1).sum()), best=float(v.max()) if fin.any() else None,
                                             best_guard1=float(v1.max()) if np.isfinite(v1).any() else None,
                                             log_sum=float(v[fin].max() + np.log(w.sum())) if fin.any() else None,
                                             n_eff=float(w.sum() ** 2 / (w**2).sum()) if fin.any() else 0.0)  # fmt: skip
            best[(t["name"], key)] = (np.where(use, lw, -np.inf), bl)
        num["observed"][t["name"]] = dict(points=t["points"], S=int(t["S"]), **row)

    # --- the network on mock tracks cut from the long streams, model track of each particle number under the same mock
    model = cfg.run_dir() / "model.keras"
    if model.exists():
        import keras
        import jax.random as jr

        import networks  # noqa: F401
        from data import I_PROG, N_PAR

        est = keras.saving.load_model(str(model))
        rows0 = tracks[f"{Ns[0]}"]["rows"]
        grp = np.flatnonzero(sel["is_long"][rows0])
        base = tracks[f"{Ns[0]}"]
        fn = jax.jit(jax.vmap(mock.make_mock(cfg, grid, sims["design"], (grid.i0 - T, grid.i0 + T))))
        feat = jax.jit(jax.vmap(mock.featurise))
        res = {}
        for rep in range(6):  # several mock tracks per stream; the long ones are kept
            keys = jr.split(jr.key(cfg.seed + 100 + rep), len(grp))
            o = fn(keys, jnp.asarray(base["count"][grp], jnp.float32), jnp.asarray(base["r_med"][grp]), jnp.asarray(base["r_hw"][grp]),
                   jnp.asarray(sims["data"]["dissolved"][st][grp]), jnp.asarray(sims["data"]["z_strata"][st][grp]))  # fmt: skip
            o = {k: np.asarray(v) for k, v in o.items()}
            keep = (o["weight"] > 0) & (o["S"] >= a.long_bins)
            zeta = np.concatenate([sims["data"]["zeta13"][st][grp], o["zeta2"]], axis=1)
            bins = np.clip(T - o["l"][:, None] + np.arange(T)[None], 0, 2 * T - 1)
            for key, tr in tracks.items():
                if key == "stored":
                    continue
                pos = np.searchsorted(tr["rows"], rows0[grp])  # where these streams are in this run
                have = (pos < len(tr["rows"])) & (tr["rows"][np.minimum(pos, len(tr["rows"]) - 1)] == rows0[grp])
                pos = np.minimum(pos, len(tr["rows"]) - 1)
                m_new = np.take_along_axis(tr["r_med"][pos], bins, axis=1)
                n_new = np.take_along_axis(tr["count"][pos], bins, axis=1)
                ok = keep & have & np.all(~o["mask"] | (n_new >= cfg.n_min * tr["N"] / N0), axis=1)
                r_new = o["r"] - o["m"] + m_new  # the same window, errors and noise around the other model track
                tok, cond = feat(jnp.asarray(o["phi"]), jnp.asarray(r_new), jnp.asarray(o["sig"]), jnp.asarray(o["mask"]),
                                 jnp.asarray(sims["data"]["dissolved"][st][grp]), jnp.zeros(len(grp)))  # fmt: skip
                fixed = np.ones((len(grp), N_PAR), np.float32)
                lp = np.concatenate([np.asarray(est.log_prob(dict(zeta=zeta[s], tokens=np.asarray(tok)[s], mask=o["mask"][s], cond=np.asarray(cond)[s],
                                                                  fixed_target_mask=fixed[s], fixed_target_value=0 * fixed[s])))
                                     for s in (slice(i, i + 4096) for i in range(0, len(grp), 4096))])  # fmt: skip
                res.setdefault(key, []).append(dict(nll=-lp, ok=ok, w=o["weight"], stream=grp))
        common = np.concatenate([np.all([res[k][rep]["ok"] for k in res if (res[k][rep]["ok"].any())], axis=0) for rep in range(6)])
        sub_keys = [k for k in res]
        num["network"] = dict(run=cfg.name, note="weighted mean loss in nats on the mock tracks every listed particle number has")
        for scope, keys_ in (("all", sub_keys[:2]), ("sub", sub_keys)):
            c = np.concatenate([np.all([res[k][rep]["ok"] for k in keys_], axis=0) for rep in range(6)])
            w = np.concatenate([res[keys_[0]][rep]["w"] for rep in range(6)]) * c
            nl = {k: np.concatenate([res[k][rep]["nll"] for rep in range(6)]) for k in keys_}
            num["network"][scope] = dict(n_mocks=int(c.sum()), loss={k: float(np.sum(w * np.where(c, nl[k], 0)) / w.sum()) for k in keys_},
                                         paired_diff_from_first={k: [float(np.sum(w * np.where(c, nl[k] - nl[keys_[0]], 0)) / w.sum()),
                                                                     float(np.sqrt(np.sum(w**2 * np.where(c, (nl[k] - nl[keys_[0]]) ** 2, 0))) / w.sum())] for k in keys_[1:]})  # fmt: skip
        del common
    (out / "particle_number.json").write_text(json.dumps(num, indent=1))

    # --- figures
    keys = [f"{N}" for N in Ns]
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.8))
    for sig, col in ((0.3, P.C_BOUND), (1.0, P.C_ALL), (3.0, P.C_PASS)):
        ax.plot(Ns, [num["usable_length"][k]["sub"][f"sigma_{sig:g}"]["at_least_26"] for k in keys], color=col, marker="o", ms=6, label=f"data error {sig:g} kpc: at least 26 bins")
        ax.plot(Ns, [num["usable_length"][k]["sub"][f"sigma_{sig:g}"]["at_least_36"] for k in keys], color=col, marker="o", ms=5, ls="--", label="at least 36 bins")
    ax.set(xscale="log", ylim=(0, 1), xlabel="particles per stream", ylabel=f"share of {num['usable_length'][keys[0]]['sub']['n']} long, resolved streams",
           title="Streams that can still host a long track once both guards apply")  # fmt: skip
    ax.set_xticks(Ns, [f"{N:,}" for N in Ns])
    ax.legend(fontsize=7.5)
    names = [t["name"] for t in obs_tracks if not t["name"].endswith("reversed")]
    order = sorted(names, key=lambda n: num["observed"][n]["points"])
    xs = np.arange(len(order))
    ref = keys[0]
    for k, col in zip(keys[1:], (P.C_ALL, P.C_BOUND)):
        scope = "all" if k == keys[1] else "sub"
        val = lambda n, kk, q: num["observed"][n][f"{kk}/{scope}"][q]
        d = [((val(n, k, "best") or np.nan) - (val(n, ref, "best") or np.nan)) for n in order]
        ax2.plot(xs + (0.12 if k != keys[1] else -0.12), d, ls="none", marker="o", ms=7, color=col, label=f"{int(k):,} particles")
        ratio = [val(n, k, "pass_both") / max(val(n, ref, "pass_both"), 1) for n in order]
        ax3.plot(xs + (0.12 if k != keys[1] else -0.12), ratio, ls="none", marker="o", ms=7, color=col, label=f"{int(k):,} particles")
    ax2.axhline(0, color=P.INK, lw=0.8)
    ax3.axhline(1, color=P.INK, lw=0.8)
    for a_ in (ax2, ax3):
        a_.set_xticks(xs, [f"{n}\n{num['observed'][n]['points']}" for n in order], fontsize=6.5, rotation=60)
        a_.legend(fontsize=8)
    ax2.set(ylabel=f"best log-likelihood minus that at {int(ref):,} particles (same parameters)", title="Best fit to each long observed track among the selected samples")
    ax3.set(yscale="log", ylabel=f"samples that pass both guards / at {int(ref):,} particles", title="Samples the guards let count for each track")
    fig.tight_layout()
    LS.save(fig, out, "particle_number")

    show = sorted(names, key=lambda n: -num["observed"][n]["points"])[:8]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for ax, n in zip(axes.ravel(), show):
        t = next(q for q in obs_tracks if q["name"] == n)
        o = t["mask"]
        for k, col in ((keys[0], P.C_ALL), (keys[-1], P.C_BOUND)):
            lw, bl = best[(n, k)]
            tr = tracks[k]
            top = np.argsort(-lw)[:8]
            for i, s in enumerate(top):
                if np.isfinite(lw[s]):
                    ax.plot(t["phi"][o], tr["r_med"][s, T - bl[s] + np.arange(T)][o], color=col, lw=1.0, alpha=0.35 + 0.65 * float(np.exp(lw[s] - lw[top[0]])), label=f"{int(k):,} particles" if i == 0 else None)
        ax.errorbar(t["phi"][o], t["r"][o], yerr=t["sig"][o], fmt="o", ms=3, lw=0.8, color=P.INK, zorder=5)
        row = num["observed"][n]
        bst = lambda k: row[f"{k}/sub"]["best"]
        ax.set_title(f"{n}: {row['points']} bins;  best log L " + ", ".join(f"{bst(k):.1f}" if bst(k) is not None else "none" for k in keys) + f"  ({', '.join(f'{int(k) // 1000}k' for k in keys)})", fontsize=8.5, loc="left")
        ax.set_xlabel("angle minus median angle [deg]", fontsize=8)
        ax.set_ylabel("projected radius [kpc]", fontsize=8)
        ax.legend(fontsize=7.5)
    fig.suptitle("The eight longest observed tracks with the best-fitting selected samples (those that pass both guards), simulated with the fewest and the most particles", fontsize=10.5)
    fig.savefig(out / "gallery.png", dpi=110)
    plt.close(fig)
    print(json.dumps({k: num[k] for k in ("selection", "usable_length", "network") if k in num}, indent=1))


if __name__ == "__main__":
    main()
