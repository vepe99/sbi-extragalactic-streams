"""How many particles and how many leapfrog steps does a usable training set need?

    uv run python scripts/extragalactic/convergence.py

Reads the test sets data/conv_p10, conv_p30, conv_p50 (99 steps; 10,000 / 30,000 / 50,000 particles),
data/conv_s399, conv_s1999 (about 30,000 particles; 399 / 1,999 steps) and conv_s399b (399 steps, 20,000
particles: a second particle realisation at 399 steps) and, if present, conv_galax (galax's adaptive
integrator, 30,000 particles, at most 1,999 steps per solve), all made with the same seed, so stream k
has the same parameters and the same progenitor state in every set. Writes plots/convergence/ and
data/convergence.json.

Particles: for each observed STRRINGS track, the share of simulations that the StreaMAX-fit likelihood would
accept as sampled well enough (at least 3 particles in every observed bin, and the variance of the model's
bin median at most 1/9 of the data variance), and the Monte Carlo error of the simulated bins.
Steps: how far a stream's binned track moves when the step number changes, against the difference between two
particle realisations at fixed steps (the Monte Carlo floor) and against the 4 % median error of the data.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "extragalactic"))
import inspect_prior as I  # noqa: E402  observed tracks, palette
import plot as P  # noqa: E402

plt = P.plt
SETS = dict(p10="conv_p10", p30="conv_p30", p50="conv_p50", s399="conv_s399", s399b="conv_s399b", s1999="conv_s1999")
FRAC_VALID = 0.005  # a bin is compared if it holds at least this share of the particles (50 of 10,000)
R_DATA_LIKE = 47.0  # kpc: 95th percentile of the mean radius of the observed tracks
DATA_ERR = 0.04  # median fractional radial error of the observed tracks


def load_set(ddir: Path) -> dict:
    cols: dict[str, list] = {}
    cfg = None
    for f in sorted(ddir.glob("shard_*.h5")):
        with h5py.File(f, "r") as h:
            cfg = json.loads(h.attrs["config"])
            for k, path in dict(index="index", theta="theta", dissolved="prog/dissolved", count="track/count",
                                r_med="track/r_med", r_hw="track/r_hw", count_alt="track_alt/count",
                                r_med_alt="track_alt/r_med", dE="diag/dE_rel", n_peri="diag/n_peri",
                                unbound="flags/unbound").items():
                cols.setdefault(k, []).append(h[path][:])
            for k in ("solver_steps_prog", "solver_steps_max", "solver_steps_mean", "n_bad"):
                if k in h["diag"]:
                    cols.setdefault(k, []).append(h["diag"][k][:])
            cols.setdefault("status", []).append(h["status"][:])
    if cfg is None:
        raise SystemExit(f"no shards in {ddir}")
    d = {k: np.concatenate(v) for k, v in cols.items()}
    o = np.argsort(d["index"])
    d = {k: v[o] for k, v in d.items()}
    man = json.loads((ddir / "manifest.json").read_text())
    d.update(cfg=cfg, N=cfg["n_particles"], steps=cfg["n_steps"], wall_s=man["run"]["wall_s"],
             workers=len(man["run"]["devices"]), n=len(d["index"]),
             shard_s_per_stream=np.array([sh["wall_s"] / sh["n"] for sh in man["shards"]]))
    d["valid"] = d["count"] >= FRAC_VALID * d["N"]
    with np.errstate(all="ignore"):
        d["r_mean_track"] = np.nanmean(np.where(d["valid"], d["r_med"], np.nan), axis=1)
        d["mc_err"] = d["r_hw"] / np.sqrt(np.maximum(d["count"], 1)) / d["r_med"]  # error of the bin median
    return d


def track_change(a: dict, b: dict, key: str = "r_med", vkey: str = "count") -> np.ndarray:
    """Per stream: rms of r_a / r_b - 1 over the bins valid in both sets (NaN with fewer than 5 such bins)."""
    n = min(a["n"], b["n"])
    va = a[vkey][:n] >= FRAC_VALID * a["N"]
    vb = b[vkey][:n] >= FRAC_VALID * b["N"]
    both = va & vb
    with np.errstate(all="ignore"):
        d2 = np.where(both, (a[key][:n].astype(np.float64) / b[key][:n] - 1.0) ** 2, 0.0)
        rms = np.sqrt(d2.sum(1) / np.maximum(both.sum(1), 1))
    return np.where(both.sum(1) >= 5, rms, np.nan)


def fit_accepts(sim: dict, o: dict, nb: int, var_ratio: float = 9.0) -> np.ndarray:
    """The two sampling guards of the StreaMAX-fit likelihood, for the simulation or its mirror image."""
    acc = np.zeros(sim["n"], bool)
    for bins in (o["bins"], nb - 1 - o["bins"]):
        c = sim["count"][:, bins].astype(np.float64)
        with np.errstate(all="ignore"):
            ok = (c >= 3).all(1) & (sim["r_hw"][:, bins].astype(np.float64) ** 2 / np.maximum(c, 1)
                                    <= o["r_err"] ** 2 / var_ratio).all(1)
        acc |= ok
    return acc


def frac_above(x, thr):
    x = x[np.isfinite(x)]
    return float((x > thr).mean()) if len(x) else float("nan")


def galax_figure(root, pdir, S, obs, nb, chg, groups, out) -> None:
    """galax's adaptive integrator against the fixed-step leapfrog, on the same prior draws."""
    gx = load_set(root / "data" / "conv_galax")
    n0 = min(gx["n"], S["s1999"]["n"])
    pairs = [("floor399", None, None, "two particle realisations, leapfrog 399 steps", P.INK2, "--"),
             ("99_gx", "p30", gx, "leapfrog 99 steps against galax", P.C_ALL, "-"),
             ("399_gx", "s399", gx, "leapfrog 399 steps against galax", P.C_BOUND, "-"),
             ("1999_gx", "s1999", gx, "leapfrog 1,999 steps against galax", P.C_PASS, "-")]
    c = {name: (chg[name] if a is None else track_change(S[a], b)) for name, a, b, *_ in pairs}
    c_alt = {name: track_change(S[a], b, "r_med_alt", "count_alt") for name, a, b, *_ in pairs if a is not None}
    failed = (gx["status"][:n0] & 16) > 0
    pct = lambda x, q=(50, 90, 99, 100): [float(v) for v in np.nanpercentile(x, q)]
    # per-shard wall times: a worker whose shard is done sits idle, so the run's wall time overstates the cost
    cost = {k: float(np.mean(S[k]["shard_s_per_stream"])) for k in ("p30", "s399", "s1999")}
    cost["galax"] = float(np.mean(gx["shard_s_per_stream"]))
    acc = {k: {o["name"]: float(fit_accepts(d, o, nb).mean()) for o in obs} for k, d in (("p30", S["p30"]), ("s1999", S["s1999"]), ("galax", gx))}
    classes = [("7-12 bins", lambda o: len(o["bins"]) <= 12), ("14-17 bins", lambda o: 13 <= len(o["bins"]) <= 17),
               ("22-48 bins", lambda o: len(o["bins"]) >= 22 and o["name"] != "PGC938075")]
    out["galax"] = dict(
        n_streams=gx["n"], n_particles=gx["N"], max_steps=gx["cfg"]["max_steps"], rtol=gx["cfg"]["rtol"],
        wall_s=gx["wall_s"], workers=gx["workers"], core_s_per_stream=cost,
        shard_core_s_per_stream=pct(gx["shard_s_per_stream"], (0, 50, 90, 100)),
        energy_error_p50_p90_p99_max=pct(gx["dE"]), unresolved_energy=float((gx["dE"] >= 1e-2).mean()),
        failed_streams=int(failed.sum()), streams_with_nonfinite_particles=int((gx["n_bad"] > 0).sum()),
        mean_nonfinite_particle_fraction=float(gx["n_bad"].mean() / gx["N"]),
        solver_steps=dict(progenitor=pct(gx["solver_steps_prog"]), particle_mean=pct(gx["solver_steps_mean"]),
                          particle_max=pct(gx["solver_steps_max"]),
                          streams_with_a_solve_at_the_cap=int((gx["solver_steps_max"] >= gx["cfg"]["max_steps"]).sum())),
        track_change={name: {g: dict(median=float(np.nanmedian(c[name][:n0][m[:n0]])), above_4pct=frac_above(c[name][:n0][m[:n0]], 0.04),
                                     above_10pct=frac_above(c[name][:n0][m[:n0]], 0.10))
                             for g, m in groups.items()} for name in c},
        track_change_c26_ordering={name: dict(median=float(np.nanmedian(v[:n0])), above_4pct=frac_above(v[:n0], 0.04)) for name, v in c_alt.items()},
        fit_acceptance_median={lab: {k: float(np.median([acc[k][o["name"]] for o in obs if sel(o)])) for k in acc} for lab, sel in classes})

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.6))
    for ax, g, title in ((axes[0, 0], "all", "all draws"), (axes[0, 1], "data_like_bound", "bound streams as small as the observed ones")):
        for name, a_, b_, lbl, col, ls in pairs:
            x = c[name][:n0][groups[g][:n0]]
            x = np.sort(x[np.isfinite(x)])
            ax.plot(x, 1.0 - np.arange(1, len(x) + 1) / len(x), color=col, lw=2.0, ls=ls, label=lbl)
        ax.axvline(DATA_ERR, color=P.INK2, lw=1.0, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(2e-3, 3)
        ax.set_ylim(0, 1)
        ax.set_xlabel("difference of the binned tracks, rms of the radius ratio minus 1")
        ax.set_ylabel("fraction of streams differing by more")
        ax.set_title(f"Tracks: {title}", fontsize=9.5)
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].text(DATA_ERR, 0.03, " data error 4 %", color=P.INK2, fontsize=8)
    ax = axes[0, 2]
    bins = np.linspace(-12, 2, 57)
    for k, d, col, lbl in (("p30", S["p30"], P.C_ALL, "leapfrog, 99 steps"), ("s399", S["s399"], P.C_BOUND, "leapfrog, 399 steps"),
                           ("s1999", S["s1999"], P.C_PASS, "leapfrog, 1,999 steps"), ("gx", gx, P.INK, "galax, adaptive")):
        x = np.log10(np.clip(d["dE"][:n0], 1e-12, 1e2))
        h, e = np.histogram(x, bins=bins, weights=np.full(len(x), 1.0 / len(x)))
        ax.stairs(h, e, color=col, lw=2.0, label=lbl)
    ax.axvline(-2, color=P.INK2, lw=1.0, ls=":")
    ax.set_xlabel(r"$\log_{10}$ relative energy error of the progenitor orbit")
    ax.set_ylabel("fraction of streams")
    ax.set_title("Energy conservation")
    ax.legend(loc="upper right")
    ax = axes[1, 0]
    bins = np.logspace(0, np.log10(max(gx["cfg"]["max_steps"], 2000)), 50)
    for key, col, lbl in (("solver_steps_prog", P.C_ALL, "progenitor orbit (back and forth)"),
                          ("solver_steps_mean", P.C_BOUND, "particles, mean of a stream"),
                          ("solver_steps_max", P.C_PASS, "particles, largest of a stream")):
        x = np.clip(gx[key].astype(float), bins[0], bins[-1])
        h, e = np.histogram(x, bins=bins, weights=np.full(len(x), 1.0 / len(x)))
        ax.stairs(h, e, color=col, lw=2.0, label=f"{lbl} (median {np.median(gx[key]):.0f})")
    ax.axvline(gx["cfg"]["max_steps"], color=P.INK2, lw=1.0, ls=":")
    ax.text(gx["cfg"]["max_steps"], ax.get_ylim()[1] * 0.97, "cap ", color=P.INK2, va="top", ha="right", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("steps taken by the adaptive solver (accepted and rejected)")
    ax.set_ylabel("fraction of streams")
    ax.set_title("galax: how many steps the adaptive solver takes")
    ax.legend(loc="upper left")
    ax = axes[1, 1]
    labs = ["leapfrog\n99", "leapfrog\n399", "leapfrog\n1,999", "galax\nadaptive"]
    vals = [cost["p30"], cost["s399"], cost["s1999"], cost["galax"]]
    ax.bar(np.arange(4), vals, width=0.62, color=[P.C_ALL, P.C_BOUND, P.C_PASS, P.INK2])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=P.INK)
    ax.set_xticks(np.arange(4), labs)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("core-seconds per stream (about 30,000 particles)")
    ax.set_title("Cost (not all sets ran under the same machine load)")
    ax = axes[1, 2]
    npr = np.minimum(S["s1999"]["n_peri"][:n0], 6)
    xi = np.arange(7)
    for name, a_, b_, lbl, col, ls in pairs:
        ys = [frac_above(c[name][:n0][npr == i], 0.04) for i in xi]
        ax.plot(xi, ys, "o", ls=ls, color=col, lw=2.0, ms=6, mfc=P.SURFACE, mew=1.6, label=lbl)
    ax.set_xticks(xi, [str(i) for i in xi[:-1]] + ["6+"])
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"pericentre passages within $t_{\rm int}$")
    ax.set_ylabel("fraction of streams whose track differs by more than 4 %")
    ax.set_title("Where the integrators disagree")
    fig.suptitle(f"galax's adaptive integrator (Dopri8, rtol {gx['cfg']['rtol']:g}, at most {gx['cfg']['max_steps']:,} steps) against the "
                 f"fixed-step leapfrog, {n0:,} prior draws, about 30,000 particles", fontsize=10.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(pdir / f"galax_integrator.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdir / 'galax_integrator'}.pdf / .png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="outputs/extragalactic/C26_test")
    arg = ap.parse_args()
    root = REPO / arg.out
    S = {k: load_set(root / "data" / v) for k, v in SETS.items()}
    pdir = root / "plots" / "convergence"
    pdir.mkdir(parents=True, exist_ok=True)
    cfg = S["p30"]["cfg"]
    obs = I.load_strrings(root / "data" / "strrings", cfg)
    nb = S["p30"]["count"].shape[1]
    out: dict = dict(sets={k: dict(n_streams=d["n"], n_particles=d["N"], n_steps=d["steps"], wall_s=d["wall_s"],
                                   workers=d["workers"], streams_per_core_s=d["n"] / d["wall_s"] / d["workers"])
                           for k, d in S.items()})

    # ------------------------------------------------------------------ particles
    classes = [("7-12 bins", lambda o: len(o["bins"]) <= 12), ("14-17 bins", lambda o: 13 <= len(o["bins"]) <= 17),
               ("22-48 bins", lambda o: len(o["bins"]) >= 22 and o["name"] != "PGC938075")]
    acc = {k: {o["name"]: float(fit_accepts(S[k], o, nb).mean()) for o in obs} for k in ("p10", "p30", "p50")}
    out["particles"] = dict(
        fit_acceptance_median={lab: {k: float(np.median([acc[k][o["name"]] for o in obs if sel(o)])) for k in acc}
                               for lab, sel in classes},
        fit_acceptance_per_track=acc,
        bins_with_at_least_50={k: float(np.mean((S[k]["count"] >= 50).sum(1))) for k in ("p10", "p30", "p50")},
        bins_with_mc_error_below_2pct={k: float(np.mean(((S[k]["mc_err"] < 0.02) & (S[k]["count"] >= 3)).sum(1)))
                                       for k in ("p10", "p30", "p50")},
        mc_error_at_fixed_bins={k: float(np.nanmedian(S[k]["mc_err"][S["p10"]["valid"]])) for k in ("p10", "p30", "p50")})

    # ------------------------------------------------------------------ steps
    pairs = [("floor", "p30", "p50", "two particle realisations, 99 steps (Monte Carlo floor)", P.INK2, ":"),
             ("floor399", "s399", "s399b", "two particle realisations, 399 steps", P.INK2, "--"),
             ("99_399", "p30", "s399", "99 against 399 steps", P.C_ALL, "-"),
             ("99_1999", "p30", "s1999", "99 against 1,999 steps", P.C_BOUND, "-"),
             ("399_1999", "s399", "s1999", "399 against 1,999 steps", P.C_PASS, "-")]
    chg = {name: track_change(S[a], S[b]) for name, a, b, *_ in pairs}
    chg_alt = {name: track_change(S[a], S[b], "r_med_alt", "count_alt") for name, a, b, *_ in pairs}
    ref = S["s1999"]
    n0 = ref["n"]
    groups = dict(all=np.ones(n0, bool), bound=~ref["unbound"].astype(bool), unbound=ref["unbound"].astype(bool),
                  data_like=ref["r_mean_track"] < R_DATA_LIKE,
                  data_like_bound=(ref["r_mean_track"] < R_DATA_LIKE) & ~ref["unbound"].astype(bool),
                  dissolved=ref["dissolved"].astype(bool), surviving=~ref["dissolved"].astype(bool),
                  resolved_at_99=S["p30"]["dE"][:n0] < 1e-2, unresolved_at_99=S["p30"]["dE"][:n0] >= 1e-2)
    out["steps"] = dict(
        n_common=n0, group_sizes={g: int(m.sum()) for g, m in groups.items()},
        track_change={name: {g: dict(median=float(np.nanmedian(chg[name][:n0][m])),
                                     above_2pct=frac_above(chg[name][:n0][m], 0.02),
                                     above_4pct=frac_above(chg[name][:n0][m], 0.04),
                                     above_10pct=frac_above(chg[name][:n0][m], 0.10))
                             for g, m in groups.items()} for name in chg},
        track_change_c26_ordering={name: dict(median=float(np.nanmedian(chg_alt[name][:n0])),
                                              above_4pct=frac_above(chg_alt[name][:n0], 0.04)) for name in chg_alt},
        unresolved_energy={k: {g: float((S[k]["dE"][:n0][m] >= 1e-2).mean()) for g, m in groups.items()
                               if g in ("all", "bound", "data_like", "data_like_bound")} for k in ("p30", "s399", "s1999")})
    npr = np.minimum(ref["n_peri"], 6)
    out["steps"]["above_4pct_by_n_peri"] = {
        name: {str(i): frac_above(chg[name][:n0][npr == i], 0.04) for i in range(7)} for name in chg}

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.6))
    ax = axes[0, 0]
    xs = [S[k]["N"] for k in ("p10", "p30", "p50")]
    for (lab, sel), col in zip(classes, (P.C_ALL, P.C_BOUND, P.C_PASS)):
        ys = [100 * out["particles"]["fit_acceptance_median"][lab][k] for k in ("p10", "p30", "p50")]
        ax.plot(xs, ys, "o-", color=col, lw=2.0, ms=6, mfc=P.SURFACE, mew=1.6, label=f"observed tracks of {lab}")
        ax.annotate(f"{ys[-1]:.1f} %", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, 0), va="center", fontsize=8, color=P.INK)
    ax.set_xticks(xs, [f"{x:,}" for x in xs])
    ax.set_xlim(xs[0] - 4000, xs[-1] + 9000)
    ax.set_ylim(0, None)
    ax.set_xlabel("particles per stream (99 steps)")
    ax.set_ylabel("simulations sampled well enough for the fit's likelihood [%]")
    ax.set_title("Particles: share of the prior the StreaMAX-fit guards accept\n(median over the observed tracks of each length)")
    ax.legend(loc="upper left")
    ax = axes[0, 1]
    bins = np.logspace(-4, 0, 70)
    ref_valid = S["p10"]["valid"]
    for k, col in zip(("p10", "p30", "p50"), (P.C_ALL, P.C_BOUND, P.C_PASS)):
        x = S[k]["mc_err"][ref_valid]
        x = np.clip(x[np.isfinite(x)], bins[0], bins[-1])
        h, e = np.histogram(x, bins=bins, weights=np.full(len(x), 1.0 / len(x)))
        ax.stairs(h, e, color=col, lw=2.0, label=f"{S[k]['N']:,} particles (median {100 * np.median(x):.1f} %)")
    ax.axvline(DATA_ERR, color=P.INK2, lw=1.0, ls=":")
    ax.text(DATA_ERR, ax.get_ylim()[1] * 0.97, " data error 4 %", color=P.INK2, va="top", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Monte Carlo error of the bin median, relative to the radius")
    ax.set_ylabel("fraction of bins")
    ax.set_title("Particles: noise of the simulated track\n(the bins that hold at least 50 of 10,000 particles)")
    ax.legend(loc="upper left")
    ax = axes[0, 2]
    stp = [S[k]["steps"] for k in ("p30", "s399", "s1999")]
    for g, col, lbl, dy in (("all", P.C_ALL, "all draws", -7), ("bound", P.C_BOUND, "bound", None),
                            ("data_like_bound", P.C_PASS, "bound, mean radius below 47 kpc", 7)):
        ys = [100 * out["steps"]["unresolved_energy"][k][g] for k in ("p30", "s399", "s1999")]
        ax.plot(stp, ys, "o-", color=col, lw=2.0, ms=6, mfc=P.SURFACE, mew=1.6, label=lbl)
        if dy is not None:
            ax.annotate(f"{ys[-1]:.1f} %", (stp[-1], ys[-1]), textcoords="offset points", xytext=(7, dy), va="center", fontsize=8, color=P.INK)
    ax.set_xscale("log")
    ax.set_xticks(stp, [f"{x:,}" for x in stp])
    ax.set_xlim(stp[0] * 0.8, stp[-1] * 1.6)
    ax.set_ylim(0, None)
    ax.set_xlabel("leapfrog steps")
    ax.set_ylabel("progenitor energy error above 1 % [%]")
    ax.set_title("Steps: orbits the integrator does not resolve")
    ax.legend(loc="upper right")
    xb = np.logspace(-3, 1, 200)
    for ax, g, title in ((axes[1, 0], "all", "all draws"), (axes[1, 1], "data_like", "tracks as small as the observed ones (mean radius below 47 kpc)")):
        for name, a_, b_, lbl, col, ls in pairs:
            x = chg[name][:n0][groups[g]]
            x = np.sort(x[np.isfinite(x)])
            ax.plot(x, 1.0 - np.arange(1, len(x) + 1) / len(x), color=col, lw=2.0, ls=ls, label=lbl)
        ax.axvline(DATA_ERR, color=P.INK2, lw=1.0, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(2e-3, 3)
        ax.set_ylim(0, 1)
        ax.set_xlabel("change of the binned track, rms of the radius ratio minus 1")
        ax.set_ylabel("fraction of streams changing by more")
        ax.set_title(f"Steps: how far the track moves, {title}", fontsize=9)
    axes[1, 0].legend(loc="upper right")
    axes[1, 0].text(DATA_ERR, 0.03, " data error 4 %", color=P.INK2, fontsize=8)
    ax = axes[1, 2]
    xi = np.arange(7)
    for name, a_, b_, lbl, col, ls in pairs:
        ys = [out["steps"]["above_4pct_by_n_peri"][name][str(i)] for i in xi]
        ax.plot(xi, ys, "o", ls=ls, color=col, lw=2.0, ms=6, mfc=P.SURFACE, mew=1.6, label=lbl)
    ax.set_xticks(xi, [str(i) for i in xi[:-1]] + ["6+"])
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"pericentre passages within $t_{\rm int}$")
    ax.set_ylabel("fraction of streams whose track moves by more than 4 %")
    ax.set_title("Steps: where the track depends on the step number")
    fig.suptitle(f"Particle number and step number, {S['p30']['n']:,} prior draws ({n0:,} at 1,999 steps), mixed progenitor state, "
                 "Gibbons ordering", fontsize=10.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(pdir / f"convergence.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    if (root / "data" / "conv_galax" / "manifest.json").exists():
        galax_figure(root, pdir, S, obs, nb, chg, groups, out)
    (root / "data" / "convergence.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {pdir / 'convergence'}.pdf / .png and {root / 'data' / 'convergence.json'}")


if __name__ == "__main__":
    main()
