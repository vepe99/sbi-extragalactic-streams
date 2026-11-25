"""Figures and summary numbers of spline_track.py: reads <out>/spline_test/*.npz and numbers/observed.json,
writes figures/*.png and numbers/summary.json. Called by `spline_track.py figures`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts" / "extragalactic"), str(REPO / "scripts" / "extragalactic" / "sbi")]
import plot as P  # noqa: E402  palette and matplotlib style of the figure script
import spline_track as S  # noqa: E402

plt = P.plt
FIG = S.RES / "figures"
YELLOW, LIGHT = "#eda100", "#9cc3f2"
CLS_COL = dict(thin=P.C_PASS, wide=YELLOW, fold=P.C_ALL, unres=P.C_BOUND)
CLS_LBL = dict(thin="thin, resolved orbit", wide="wide (a cloud)", fold="angle folds back", unres="unresolved orbit")
POOL = "prior"  # the mock pool of the figures: 5,000 prior draws (data/particles_pilot)
NETWORK = "npe_trainv2_1000epoch_vr4_5e4lr"  # the trained run whose sigma_sys is compared with the spline's scatter


def save(fig, name: str) -> None:
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pct(a, q):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if len(a) else float("nan")


# ------------------------------------------------------------------------------------------------------
# observed tracks
# ------------------------------------------------------------------------------------------------------
def fig_observed(obs: list[dict], o) -> None:
    for page in range(2):
        rows = obs[16 * page : 16 * page + 16]
        for kind in ("tracks", "sky"):
            fig, axes = plt.subplots(4, 4, figsize=(15, 12.5 if kind == "sky" else 10.5), constrained_layout=True)
            for ax, r in zip(axes.ravel(), rows):
                n = r["name"]
                x, y, sg, xg, m, pa = (o[f"{n}/{k}"] for k in ("x", "y", "sig", "xg", "r", "p_away"))
                if kind == "tracks":
                    ax.fill_between(xg, o[f"{n}/r_lo"], o[f"{n}/r_hi"], color=P.C_ALL, alpha=0.25, lw=0)
                    ax.plot(xg, m, color=P.C_ALL, lw=1.6)
                    ax.errorbar(x, y, sg, fmt="o", ms=3, color=P.INK, lw=0.9, zorder=3)
                    strip = ax.inset_axes([0, 0, 1, 0.05], sharex=ax)
                    im = strip.imshow(pa[None], extent=(xg[0], xg[-1], 0, 1), aspect="auto", cmap=P.DIV, vmin=0, vmax=1)
                    strip.axis("off")
                    lo, hi = min((y - sg).min(), o[f"{n}/r_lo"].min()), max((y + sg).max(), o[f"{n}/r_hi"].max())
                    ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.05 * (hi - lo))
                    ax.set_title(f"{n}: {r['n']} bins, {r['edf']:.1f} effective, s = {r['s'][1]:.1f} kpc")
                else:
                    ph = np.deg2rad(xg)
                    pts = np.stack([m * np.cos(ph), m * np.sin(ph)], -1)
                    lc = LineCollection(np.stack([pts[:-1], pts[1:]], 1), cmap=P.DIV, array=0.5 * (pa[:-1] + pa[1:]), lw=3.0, capstyle="round")
                    lc.set_clim(0, 1)
                    im = ax.add_collection(lc)
                    ax.plot(y * np.cos(np.deg2rad(x)), y * np.sin(np.deg2rad(x)), "o", ms=2.2, color=P.INK, zorder=3)
                    ax.plot(0, 0, "+", ms=11, mew=1.6, color=P.INK)
                    ax.set_aspect("equal", adjustable="datalim")
                    ax.autoscale_view()
                    ax.set_title(n)
            for ax in axes.ravel()[len(rows) :]:
                ax.axis("off")
            if kind == "tracks":
                fig.supxlabel("unwrapped angle [deg]", fontsize=10, color=P.INK2)
                fig.supylabel("projected radius [kpc]", fontsize=10, color=P.INK2)
                fig.suptitle("Observed tracks (points), spline posterior (line: mean, band: 16-84 %) and, along the bottom of each panel, the sign of its curvature", fontsize=11)
            else:
                fig.supxlabel("x [kpc]", fontsize=10, color=P.INK2)
                fig.supylabel("y [kpc]", fontsize=10, color=P.INK2)
                fig.suptitle("The same splines in the plane of the track files (host at the cross), coloured by the sign of the curvature", fontsize=11)
            cb = fig.colorbar(im, ax=axes, location="right", shrink=0.5, pad=0.01, aspect=30)
            cb.set_label("posterior probability that the track bends away from the host\n(0: bends towards it, as an orbit in a centred halo must)")
            cb.outline.set_visible(False)
            save(fig, f"{1 if kind == 'tracks' else 2}_observed_{kind}_p{page + 1}.png")


def mock_table(obs: list[dict], mk) -> dict:
    """Observed value, mock percentiles and P(mock >= observed) of every statistic, per track and pool."""
    out = {}
    for r in obs:
        out[r["name"]] = {}
        for pool in ("prior", "long"):
            row = {}
            for st in ("edf", "s_med", "f_exp", "p_max", "L_sph", "n_flip"):
                v, m = (r["s"][1] if st == "s_med" else r[st]), mk[f"{pool}/{r['name']}/{st}"]
                row[st] = dict(observed=float(v), mock=[pct(m, q) for q in (2.5, 16, 50, 84, 97.5)], p_ge=float(np.mean(m >= v)), p_le=float(np.mean(m <= v)))
            out[r["name"]][pool] = row
    return out


def fig_mocks(obs: list[dict], tab: dict) -> None:
    order = sorted(obs, key=lambda r: (r["n"], r["name"]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 9), sharey=True, constrained_layout=True)
    panels = (("edf", "effective number of parameters of the spline", "the track needs no more numbers than a mock of its length"),
              ("s_med", "scatter beyond the stated errors, s [kpc] (posterior median)", "orange: more scatter than 97.5 % of the mocks"),
              ("p_max", "largest probability along the track that it bends away from the host", "orange: a bend away that 97.5 % of the mocks do not show"))  # fmt: skip
    for ax, (st, xl, title) in zip(axes, panels):
        for i, r in enumerate(order):
            t = tab[r["name"]][POOL][st]
            ax.plot(t["mock"][::4], [i, i], color=P.AXIS, lw=1.2, solid_capstyle="round", label="mock tracks, central 95 %" if i == 0 else None)
            ax.plot(t["mock"][1:4:2], [i, i], color=LIGHT, lw=5, solid_capstyle="round", label="mock tracks, 16-84 %" if i == 0 else None)
            ax.plot(t["mock"][2], i, "|", color=P.C_ALL, ms=9, mew=1.6, label="mock median" if i == 0 else None)
            out = t["observed"] > t["mock"][4]
            ax.plot(t["observed"], i, "o", ms=6.5, color=P.C_BOUND if out else P.INK, mec=P.SURFACE, mew=1.0, zorder=4)
        ax.plot([], [], "o", ms=6.5, color=P.INK, label="observed track")
        ax.plot([], [], "o", ms=6.5, color=P.C_BOUND, label="observed, above 97.5 % of the mocks")
        ax.set(xlabel=xl, title=title, ylim=(-0.7, len(order) - 0.3))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), [f"{r['name']}  ({r['n']})" for r in order])
    axes[0].legend(loc="lower right")
    fig.suptitle("What the splines show in the observed tracks, against mock tracks with the same bins and errors cut from 5,000 prior simulations (tracks sorted by number of bins, in brackets)", fontsize=11)
    save(fig, "3_observed_vs_mocks.png")


def fig_shape(obs: list[dict], o, mk) -> dict:
    order = sorted(obs, key=lambda r: (r["n"], r["name"]))
    q = o["q_grid"]
    L = np.stack([o[f"{r['name']}/L_q"] for r in order])
    dead = L.max(1) < 5 / 400  # fewer than 5 of the 400 spline draws pass for the best halo: no likelihood to speak of
    rel = np.where(dead[:, None], np.nan, L / np.maximum(L.max(1, keepdims=True), 1e-12))
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.5), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    ax = axes[0]
    cmap = P.SEQ.copy()
    cmap.set_bad(P.GRID)
    im = ax.imshow(rel, aspect="auto", cmap=cmap, vmin=0, vmax=1, extent=(q[0] - 0.025, q[-1] + 0.025, len(order) - 0.5, -0.5))
    ax.set_yticks(range(len(order)), [f"{r['name']}  ({r['n']})" for r in order])
    ax.grid(False)
    ax.set(xlabel="flattening q of the paper's halo in the plane of the sky (1 = round)", title="Paper's likelihood of q per observed track, over its maximum\n(grey: fewer than 5 of 400 spline draws are allowed by any halo)")
    cb = fig.colorbar(im, ax=ax, pad=0.01, aspect=35)
    cb.outline.set_visible(False)
    # stack of the tracks with a likelihood: sum of ln (L / L_max), floored at half a draw
    ax = axes[1]
    stack = np.sum(np.log(np.maximum(L[~dead], 0.5 / 400) / L[~dead].max(1, keepdims=True)), axis=0)
    ax.plot(q, stack - stack.max(), "o-", color=P.C_ALL, ms=4)
    ax.set(xlabel="q", ylabel="summed log-likelihood, from its maximum", title=f"The {int((~dead).sum())} observed tracks with a likelihood, stacked")
    # mocks by the true flattening of their halo
    ax = axes[2]
    Lm, qt = (np.concatenate([mk[f"{POOL}/{r['name']}/{k}"] for r in obs]) for k in ("L_q", "q_true"))
    alive = Lm.max(1) > 0
    out = dict(mock_no_likelihood=float(1 - alive.mean()), mock_round_halo_is_best=float(np.mean(Lm[alive][:, -1] >= Lm[alive].max(1) - 1e-12)))
    for (lo, hi), col in zip(((0.5, 0.7), (0.9, 1.1), (1.3, 1.5)), (P.C_BOUND, P.C_ALL, P.C_PASS)):
        m = alive & (qt >= lo) & (qt < hi)
        ax.plot(q, (Lm[m] / Lm[m].max(1, keepdims=True)).mean(0), "o-", color=col, ms=4, label=f"true 3D flattening {lo}-{hi}  ({int(m.sum()):,} mocks)")
        out[f"mock_rel_like_at_q0.5|true_{lo}-{hi}"] = float((Lm[m] / Lm[m].max(1, keepdims=True)).mean(0)[np.argmin(abs(q - 0.5))])
    ax.set(xlabel="q", ylabel="likelihood over its maximum, mean over mocks", ylim=(0, 1.02), title="Mock tracks by the true flattening of their halo:\na round halo is preferred whatever it is")
    ax.legend(loc="lower right")
    fig.suptitle("The paper's test: which flattened halos are allowed by the sign of the curvature", fontsize=11)
    save(fig, "4_halo_shape.png")
    return out | {"observed_no_likelihood": [r["name"] for r, d in zip(order, dead) if d], "stack_best_q": float(q[np.argmax(stack)]),
                  "stack_dlnL_at_q0.5": float(stack[np.argmin(abs(q - 0.5))] - stack.max())}  # fmt: skip


# ------------------------------------------------------------------------------------------------------
# simulations
# ------------------------------------------------------------------------------------------------------
QB = ((0.5, 0.7), (0.7, 0.9), (0.9, 1.1), (1.1, 1.3), (1.3, 1.5))


def fig_premise() -> dict:
    z = np.load(S.RES / "premise.npz", allow_pickle=True)
    t = z["table"]
    k = t[:, 0].astype(int)
    phi, away, can, car, q, cls = np.abs(t[:, 1]), t[:, 2] > S.P_SIG, t[:, 3], t[:, 4], z["q"][k], z["cls"][k]
    toward = t[:, 2] < 1 - S.P_SIG
    out = dict(streams={c: int((z["cls"][z["used"]] == c).sum()) for c in S.CLS}, bins=int(len(t)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    ax, edges = axes[0], np.array([0, 10, 20, 30, 60, 90, 180, 360, 500])
    for c in S.CLS:
        f = [away[(cls == c) & (phi >= a) & (phi < b)].mean() for a, b in zip(edges[:-1], edges[1:])]
        ax.plot(range(len(f)), f, "o-", color=CLS_COL[c], ms=5, label=CLS_LBL[c])
        out[f"away_by_distance|{c}"] = [float(x) for x in f]
        out[f"away|{c}"] = float(away[cls == c].mean())
        out[f"toward|{c}"] = float(toward[cls == c].mean())
    out["distance_edges_deg"] = edges.tolist()
    ax.set_xticks(range(len(edges) - 1), [f"{a}-{b}" for a, b in zip(edges[:-1], edges[1:])])
    ax.set(xlabel="angle from the progenitor [deg]", ylabel="share of the bins", ylim=(0, None), title="Bins where the track bends away from the host\n(significantly, 300,000 particles)")
    ax.legend()
    thin = cls == "thin"
    ax, xs = axes[1], np.arange(len(QB))
    f_away = [away[thin & (q >= a) & (q < b)].mean() for a, b in QB]
    f_acc = [(can[thin & (q >= a) & (q < b)] < 0).mean() for a, b in QB]
    ax.plot(xs, f_away, "o-", color=P.C_BOUND, ms=6, label="the track bends away from the host")
    ax.plot(xs, f_acc, "s-", color=P.INK2, ms=6, label="the halo's acceleration points to that side\n(what the paper's criterion needs for it)")
    ax.set_xticks(xs, [f"{a}-{b}" for a, b in QB])
    ax.set(xlabel="true flattening q of the halo", ylabel="share of the bins of thin streams", ylim=(-0.004, 0.1), title="The bends away do not follow the halo's shape,\nand the halo never asks for one")
    ax.legend(loc="center right")
    ax = axes[2]
    ang = np.degrees(np.arccos(np.clip(car, -1, 1)))
    for qq, col, lbl in ((50, P.C_ALL, "median"), (90, P.C_BOUND, "90th percentile"), (99, P.INK2, "99th percentile")):
        ax.plot(xs, [np.percentile(ang[thin & (q >= a) & (q < b)], qq) for a, b in QB], "o-", color=col, ms=6, label=lbl)
    ax.set_xticks(xs, [f"{a}-{b}" for a, b in QB])
    ax.set(xlabel="true flattening q of the halo", ylabel="angle [deg]", ylim=(0, None), title="Angle on the sky between the halo's acceleration\nand the direction to the host centre, at the stream")
    ax.legend()
    fig.suptitle("The paper's premise in the forward model: 1,797 simulated streams, track against the true halo", fontsize=11)
    save(fig, "5_premise.png")
    sig = thin & (away | toward)
    out |= dict(q_bins=[list(b) for b in QB], away_by_q_thin=[float(x) for x in f_away], acc_away_side_by_q_thin=[float(x) for x in f_acc],
                angle_p50_p90_p99_by_q_thin=[[float(np.percentile(ang[thin & (q >= a) & (q < b)], p)) for p in (50, 90, 99)] for a, b in QB],
                criterion_violated_of_significant_thin=float(((away & (can > 0)) | (toward & (can < 0)))[sig].mean()),
                thin_streams_with_an_away_bin=float(np.mean([away[k == s].any() for s in np.unique(k[thin])])),
                share_of_away_bins_within_30deg_thin=float((away & thin & (phi < 30)).sum() / (away & thin).sum()))  # fmt: skip
    return out


def fig_smoothing() -> dict:
    z = np.load(S.RES / "smoothing.npz", allow_pickle=True)
    t, cols = z["table"], [str(c) for c in z["columns"]]
    row = {int(s): j for j, s in enumerate(z["stream"])}
    cls = z["cls"][np.array([row[int(s)] for s in t[:, 0]])]
    raw, vr, ref, vref = t[:, 2], t[:, 3], t[:, 4], t[:, 5]
    m = np.isfinite(ref) & (vr > 0) & (cls == "thin")
    names = [c[6:] for c in cols if c.startswith("mean: ")]
    mse = lambda e: float(np.mean(np.clip((e[m] ** 2 - vref[m]) / vr[m], -50, 50)))  # in units of the particle variance, reference noise removed
    out = dict(bins_thin=int(m.sum()), mse={"bins as stored": mse(raw - ref)}, calibration={}, rel_err_p50_p90={"bins as stored": [pct(100 * np.abs((raw - ref) / ref)[m], p) for p in (50, 90)]})
    for i, n in enumerate(names):
        sp, vs = t[:, 6 + 2 * i], t[:, 7 + 2 * i]
        out["mse"][n] = mse(sp - ref)
        out["calibration"][n] = float(np.sqrt(np.mean(np.clip((sp - ref)[m] / np.sqrt(vs + vref)[m], -5, 5) ** 2)))
        out["rel_err_p50_p90"][n] = [pct(100 * np.abs((sp - ref) / ref)[m], p) for p in (50, 90)]
    lg = z["is_long"]
    keys = ["30000"] + [f"30000, {n}" for n in names] + ["90000", "300000"]
    out["long_streams"] = int(lg.sum())
    out["share_26_bins"] = {str(e): {k: float((z[f"len/{k}|{e}"][lg] >= 26).mean()) for k in keys} for e in (0.3, 1.0, 3.0)}
    out["mean_usable_bins"] = {str(e): {k: float(z[f"len/{k}|{e}"][lg].mean()) for k in keys} for e in (0.3, 1.0, 3.0)}

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), constrained_layout=True)
    ax = axes[0]
    lab = list(out["mse"])
    val = [out["mse"][k] for k in lab]
    ax.barh(range(len(lab)), val, height=0.55, color=P.C_ALL)
    for i, v in enumerate(val):
        ax.text(v + 0.06, i, f"{v:.2f}", va="center", fontsize=9)
    for x in (1 / 3, 0.1):
        ax.axvline(x, color=P.C_PASS, lw=1.4)
    ax.text(0.98, 0.42, "green lines: the bins with 90,000 particles (0.33)\nand with 300,000 particles (0.10)", transform=ax.transAxes, ha="right", fontsize=8.5, color=P.INK2)
    ax.set_yticks(range(len(lab)), ["30,000 particles, " + k if i == 0 else "spline: " + k for i, k in enumerate(lab)])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="squared error of the model track against 300,000 particles, in units of the particle noise of 30,000", ylim=(len(lab) - 0.4, -0.6),
           title="A spline of the 30,000-particle track: little is gained, and a stiffer spline is worse than the bins")  # fmt: skip
    ax = axes[1]
    sh = out["share_26_bins"]["1.0"]
    lab2 = ["30,000 particles, bins as stored"] + ["spline: " + n for n in names] + ["90,000 particles, bins", "300,000 particles, bins"]
    col = [P.C_ALL] * (1 + len(names)) + [P.C_PASS] * 2
    ax.barh(range(len(keys)), [100 * sh[k] for k in keys], height=0.55, color=col)
    for i, k in enumerate(keys):
        note = "  (error underestimated)" if k in ("30000, " + names[2], "30000, " + names[3]) else ""
        ax.text(100 * sh[k] + 1, i, f"{100 * sh[k]:.0f} %{note}", va="center", fontsize=9)
    ax.set_yticks(range(len(keys)), lab2)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="share of the long streams that keep 26 usable bins at a data error of 1 kpc [%]", xlim=(0, 100), title=f"What the particle-noise guard lets through ({int(lg.sum())} long streams)")
    fig.suptitle("The spline as the model track: thin streams simulated with 30,000 and with 300,000 particles", fontsize=11)
    save(fig, "6_smoothing.png")
    return out


def fig_phase() -> dict:
    z = np.load(S.RES / "phase.npz", allow_pickle=True)
    t, cols = z["table"], [str(c) for c in z["columns"]]
    idx = {int(i): j for j, i in enumerate(z["index"])}
    cls = z["cls"][np.array([idx[int(s)] for s in t[:, 0]])]
    dist, thin = np.abs(t[:, 1] + 5.0), cls == "thin"  # of the shifted bin from the progenitor
    meth = (("direct", "particles binned at the new phase (the floor)", P.INK2, "--"), ("linear", "mean of the two neighbouring bins", P.C_BOUND, "-"),
            ("cubic", "cubic spline through the bins", YELLOW, "-"), ("spline", "penalised spline, one for the stream", P.C_ALL, "-"),
            ("spline per arm", "penalised spline, one per arm", P.C_PASS, "-"))  # fmt: skip
    edges = [0, 10, 20, 30, 40, 60, 90, 130, 500]
    groups = [dist == 0] + [(dist >= a) & (dist < b) for a, b in zip(edges[1:-1], edges[2:])]
    labels = ["across it"] + [f"{a}-{b}" if b < 500 else f">{a}" for a, b in zip(edges[1:-1], edges[2:])]
    out = dict(streams={c: int((z["cls"][np.unique([idx[int(s)] for s in t[:, 0]])] == c).sum()) for c in S.CLS}, groups=labels, median={}, p90={}, rel_p90={})
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), constrained_layout=True, sharex=True)
    for name, lbl, col, ls in meth:
        e, rs = np.abs(t[:, cols.index(name)]), t[:, 2]
        out["median"][name] = [pct(e[thin & g], 50) for g in groups]
        out["p90"][name] = [pct(e[thin & g], 90) for g in groups]
        out["rel_p90"][name] = [pct(100 * (e * rs)[thin & g], 90) for g in groups]
        for ax, key in zip(axes, ("median", "p90")):
            ax.plot(range(len(groups)), out[key][name], "o", ls=ls, color=col, ms=5, label=lbl)
    for ax, ttl in zip(axes, ("median error", "90th percentile of the error")):
        ax.set_yscale("log")
        ax.set_xticks(range(len(groups)), labels)
        ax.set(xlabel="angle of the shifted bin from the progenitor [deg]", ylabel="error in units of the particle noise", title=ttl)
    axes[0].legend()
    fig.suptitle("The track in bins shifted by half a bin, predicted from the stored bins: thin streams of the 5,000-stream particle set, halves of 15,000 particles", fontsize=11)
    save(fig, "7_bin_phase.png")
    away = thin & (dist >= 20)
    out["thin_20deg_or_more"] = {n: dict(median=pct(np.abs(t[away, cols.index(n)]), 50), p90=pct(np.abs(t[away, cols.index(n)]), 90), p99=pct(np.abs(t[away, cols.index(n)]), 99)) for n, *_ in meth}
    out["same_bins_thin"] = {n: dict(median=pct(np.abs(t[thin, cols.index(n)]), 50), p90=pct(np.abs(t[thin, cols.index(n)]), 90)) for n in ("direct0", "spline0", "spline per arm0")}
    return out


def fig_scatter(obs: list[dict], o) -> dict:
    """The systematic scatter the stream model asks for (sigma_sys of a trained network, and of nested sampling
    where it was run) against the scatter the spline leaves: what the stream model cannot fit of a smooth track."""
    net = S.OUT / "models" / NETWORK / "evaluation_real/posterior.npz"
    if not net.exists():
        return {}
    z = np.load(net)
    sig = {str(n): z["params"][i, :, 13] for i, n in enumerate(z["names"])}
    rng, rows = np.random.default_rng(2615), {}
    fig, ax = plt.subplots(figsize=(7.6, 7.2), constrained_layout=True)
    ax.plot([0, S.S_MAX], [0, S.S_MAX], color=P.AXIS, lw=1.2)
    for r in obs:
        n = r["name"]
        ps = o[f"{n}/post"].sum(1)
        draw = rng.choice(o["s_grid"], 4000, p=ps / ps.sum())
        m = [pct(sig[n], q) for q in (16, 50, 84)]
        rows[n] = dict(spline=r["s"], model=m, p_model_larger=float(np.mean(rng.choice(sig[n], 4000) > draw)), bins=r["n"])
        far = rows[n]["p_model_larger"] > 0.95
        ax.errorbar(r["s"][1], m[1], xerr=[[r["s"][1] - r["s"][0]], [r["s"][2] - r["s"][1]]], yerr=[[m[1] - m[0]], [m[2] - m[1]]], fmt="o", ms=5,
                    color=P.C_BOUND if far else P.C_ALL, ecolor=P.AXIS, elinewidth=0.9, mec=P.SURFACE, mew=0.8, zorder=3)  # fmt: skip
        if far:
            ax.annotate(n.replace("_GROUP", ""), (r["s"][1], m[1]), xytext=(5, 4), textcoords="offset points", fontsize=7.5, color=P.INK2)
    out = dict(network=NETWORK, tracks=rows, model_larger_95=[n for n, v in rows.items() if v["p_model_larger"] > 0.95])
    ns = S.OUT / "nested/NGC1162/result.npz"
    if ns.exists():
        q = [pct(np.load(ns)["posterior"][:, 13], p) for p in (16, 50, 84)]
        ax.plot(rows["NGC1162"]["spline"][1], q[1], "D", ms=7, color=P.INK, mec=P.SURFACE, mew=0.8, zorder=4, label="NGC1162, nested sampling")
        out["NGC1162_nested"] = q
    ax.plot([], [], "o", color=P.C_BOUND, label="stream model needs more scatter than the spline leaves (probability above 95 %)")
    ax.plot([], [], "o", color=P.C_ALL, label="the other tracks")
    ax.set(xlim=(0, S.S_MAX), ylim=(0, S.S_MAX), xlabel="scatter beyond the errors around a smooth spline, s [kpc]", ylabel="scatter beyond the errors in the stream model, $\\sigma_{sys}$ [kpc]",
           title="What the stream model does not fit of a smooth track\n(points: posterior median, bars: 16-84 %; network " + NETWORK + ")")  # fmt: skip
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "8_scatter_model_vs_spline.png")
    return out


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    obs = json.load(open(S.RES / "numbers/observed.json"))
    o, mk = np.load(S.RES / "observed.npz"), np.load(S.RES / "mocks.npz")
    fig_observed(obs, o)
    tab = mock_table(obs, mk)
    fig_mocks(obs, tab)
    rel = np.concatenate([o[f"{r['name']}/sig"] / o[f"{r['name']}/y"] for r in obs])
    summary = dict(
        observed=dict(tracks=len(obs), edf_over_n_median=float(np.median([r["edf"] / r["n"] for r in obs])), edf_range=[min(r["edf"] for r in obs), max(r["edf"] for r in obs)],
                      tracks_edf_below_10=int(sum(r["edf"] < 10 for r in obs)), tracks_with_significant_away=[r["name"] for r in obs if r["f_away"] > 0],
                      L_sph_zero=[r["name"] for r in obs if r["L_sph"] == 0], sigma_over_r_pct_p10_p50_p90=[pct(100 * rel, p) for p in (10, 50, 90)]),
        mocks={st: {pool: dict(above_97_5=[n for n in tab if tab[n][pool][st]["observed"] > tab[n][pool][st]["mock"][4]],
                               below_2_5=[n for n in tab if tab[n][pool][st]["observed"] < tab[n][pool][st]["mock"][0]]) for pool in ("prior", "long")}
               for st in ("edf", "s_med", "p_max", "f_exp")},
        mock_table=tab, shape=fig_shape(obs, o, mk), premise=fig_premise(), smoothing=fig_smoothing(), phase=fig_phase(), scatter=fig_scatter(obs, o))  # fmt: skip
    json.dump(summary, open(S.RES / "numbers/summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "mock_table"}, indent=1))


if __name__ == "__main__":
    main()
