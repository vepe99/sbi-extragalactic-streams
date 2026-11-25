"""Collect the accuracy study of 2026-11-24 in one folder: <out>/accuracy_study/ (figures/, numbers/).

    uv run python scripts/extragalactic/accuracy_study.py [--run npe_trainv2_1000epoch_vr4_5e4lr] [--experiments q_base q_dm ...]

It simulates nothing and trains nothing. It reads what the other scripts left and draws the figures that put
their numbers side by side:

  track_noise.py       plots/<subdir>/track_noise/     key noise and roughness of the model tracks
  symmetries.py        plots/<subdir>/symmetries/      scaling symmetries of the simulator
  particle_svd.py      plots/<subdir>/particle_svd/    particles kept with a truncated SVD
  particle_store.py    plots/<subdir>/particle_store/  the compressed store and the track made from it
  long_streams.py      plots/<subdir>/long_streams/    census of the long streams
  particle_number.py   plots/<subdir>/particle_number/ good prior samples simulated again with more particles
  sbi/prior_weights.py models/<run>/prior_weights/     the simulation set as a posterior sample of each track
  sbi/nll.py           models/<experiment>/heldout_nll.npz   loss and recovery of the experiments on common mocks
  train_vs_heldout     numbers/train_vs_heldout.npz    loss on training streams against held-out streams

Figures made here: 01_loss_by_class, 03_importance_sampling, 04_train_vs_heldout, 07_experiments.
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts" / "extragalactic"), str(REPO / "scripts" / "extragalactic" / "sbi")]
import long_streams as LS  # noqa: E402
from config import pick_device  # noqa: E402

P, plt = LS.P, LS.plt
CLS = [(k, lbl, col) for k, lbl, col in LS.CLASSES]
SHORT = dict(thin="resolved, thin", wide="resolved, wide (a cloud)", fold="resolved, angle folds back", unres="unresolved orbit")
LENGTHS = (("short\n6-12 bins", lambda S: S <= 12), ("medium\n13-25", lambda S: (S > 12) & (S <= 25)), ("long\n26-35", lambda S: (S > 25) & (S <= 35)), ("long\n36-49", lambda S: S > 35))
PARAMS = ["logM", "Rs", "dirx", "diry", "dirz", "logm", "rs", "x0", "z0", "vx0", "vy0", "vz0", "time", "sig", "prog"]


def fig_loss_by_class(noise_npz: Path, out: Path) -> dict:
    d = np.load(noise_npz, allow_pickle=True)
    cls, S, nll = d["cls"], d["S"], d["nll"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    x, num = np.arange(len(LENGTHS)), {}
    bottom = np.zeros(len(LENGTHS))
    for i, (k, _, col) in enumerate(CLS):
        share = np.array([(cls[f(S)] == i).mean() for _, f in LENGTHS])
        ax.bar(x, share, bottom=bottom, color=col, width=0.62, label=SHORT[k], edgecolor="white", lw=0.6)
        for xi, (b, s) in enumerate(zip(bottom, share)):
            if s > 0.06:
                ax.text(xi, b + s / 2, f"{s:.0%}", ha="center", va="center", fontsize=8.5, color="white")
        bottom += share
        loss = [float(nll[f(S) & (cls == i)].mean()) for _, f in LENGTHS]
        ax2.plot(x, loss, color=col, marker="o", ms=6, label=SHORT[k])
        num[k] = dict(share=share.tolist(), loss_nats=loss)
    ax2.plot(x, [nll[f(S)].mean() for _, f in LENGTHS], color=P.INK, marker="o", ms=5, ls="--", lw=1.2, label="all")
    ax2.axhline(0.5 * (1 + np.log(2 * np.pi)) * 15, color=P.INK2, lw=0.8, ls=":")
    ax2.text(0.02, 0.5 * (1 + np.log(2 * np.pi)) * 15 - 0.25, "nothing learnt (the prior)", fontsize=8, color=P.INK2, va="top", transform=ax2.get_yaxis_transform())
    for a in (ax, ax2):
        a.set_xticks(x, [lab for lab, _ in LENGTHS])
        a.set_xlabel("length of the mock track")
    ax.set(ylim=(0, 1), ylabel="share of the training examples", title="Where the training examples come from")
    ax2.set(ylabel="held-out loss of the network [nats]", title="What the network learns from each kind (lower = more)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    LS.save(fig, out, "01_loss_by_class")
    return num


def fig_importance(run: Path, out: Path) -> dict:
    """Importance sampling of the network's posterior: the 32 mock tracks of evaluation.py (efficiency against
    length, and whether the true stream has a resolved orbit) and the observed tracks (the network's effective
    sample against that of the simulation set itself, prior_weights.py)."""
    ev = run / "evaluation"
    w = np.load(ev / "weights.npz", allow_pickle=True)
    K, M = w["weights"].shape
    cols = {k: [] for k in ("index", "diag/dE_rel")}
    for p in sorted(glob.glob(str(ev / "simulations/data/shard_*.h5"))):
        with h5py.File(p, "r") as f:
            for k in cols:
                cols[k].append(f[k][:])
    dE = np.concatenate(cols["diag/dE_rel"])[np.argsort(np.concatenate(cols["index"]))]
    unres_true, unres = dE[K * M :] > LS.DE_MAX, (dE[: K * M] > LS.DE_MAX).reshape(K, M)
    rows = json.loads((ev / "metrics.json").read_text())["ppc"]
    eff = np.array([r.get("importance_efficiency", np.nan) for r in rows])
    S = np.load(ev / "test_tracks.npz")["S"][:K]
    share = np.array([unres[i][w["covering"][i]].mean() for i in range(K)])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.7))
    for sel, col, lab in ((~unres_true, P.C_PASS, "true stream: resolved orbit"), (unres_true, P.C_BOUND, "true stream: unresolved orbit")):
        ax.scatter(S[sel], 100 * eff[sel], s=40, color=col, label=f"{lab} ({sel.sum()})", lw=0, alpha=0.9)
    ax.set(yscale="log", xlabel="length of the mock track [bins]", ylabel="importance-sampling efficiency [%]",
           title="Mock tracks: efficiency of the network's posterior falls with length")  # fmt: skip
    ax.legend(fontsize=8, loc="upper right")
    num = dict(mock=dict(efficiency_median_resolved=float(np.nanmedian(eff[~unres_true])), efficiency_median_unresolved=float(np.nanmedian(eff[unres_true])),
                         by_length={lab.replace("\n", " "): float(np.nanmedian(eff[f(S)])) for lab, f in LENGTHS if f(S).any()},
                         unresolved_share_of_draws_median=float(np.median(share))))  # fmt: skip
    pw = run / "prior_weights" / "prior_weights.json"
    if pw.exists():
        T = [t for t in json.loads(pw.read_text())["tracks"] if t["kind"] == "observed" and t.get("network_n_eff") is not None]
        x, y, b = (np.array([t[k] for t in T], float) for k in ("n_eff", "network_n_eff", "points"))
        sc = ax2.scatter(np.maximum(x, 1), np.maximum(y, 1), c=b, cmap=P.SEQ, s=46, lw=0.5, edgecolors=P.INK)
        fig.colorbar(sc, ax=ax2, label="observed bins of the track", pad=0.02)
        ax2.plot([1, 1000], [1, 1000], color=P.INK2, lw=0.8, ls=":")
        ax2.set(xscale="log", yscale="log", xlabel="effective sample of the 7 million usable simulations of the set", ylabel="effective sample of 1,000 network draws",
                title="Observed tracks: where the set has no simulations, the network has no posterior")  # fmt: skip
        num["observed"] = dict(n=len(T), set_n_eff_median=float(np.median(x)), network_n_eff_median=float(np.median(y)), set_n_eff_below_10=int((x < 10).sum()))
    fig.tight_layout()
    LS.save(fig, out, "03_importance_sampling")
    return num


def fig_train_vs_heldout(npz: Path, out: Path) -> dict:
    d = np.load(npz, allow_pickle=True)
    runs = [k[len("train_nll_") :] for k in d.files if k.startswith("train_nll_")]
    fig, axes = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 4.6), sharey=True, squeeze=False)
    x, num = np.arange(len(LENGTHS)), {}
    for ax, run in zip(axes[0], runs):
        num[run] = {}
        for sel_cls, col, lab in ((None, P.INK, "all examples"), (0, P.C_PASS, "resolved, thin streams")):
            gap, se = [], []
            for _, f in LENGTHS:
                a = d[f"train_nll_{run}"][f(d["train_S"]) & ((d["train_cls"] == sel_cls) if sel_cls is not None else True)]
                b = d[f"heldout_nll_{run}"][f(d["heldout_S"]) & ((d["heldout_cls"] == sel_cls) if sel_cls is not None else True)]
                gap.append(b.mean() - a.mean())
                se.append(np.sqrt(a.var() / len(a) + b.var() / len(b)))
            ax.errorbar(x + (0.06 if sel_cls == 0 else -0.06), gap, yerr=se, color=col, marker="o", ms=6, capsize=3, label=lab)
            num[run][lab] = dict(gap_nats=[float(g) for g in gap], se=[float(s) for s in se])
        ax.axhline(0, color=P.INK2, lw=0.8, ls=":")
        ax.set_xticks(x, [lab for lab, _ in LENGTHS])
        ax.set(xlabel="length of the mock track", title=run, ylabel="")
        ax.title.set_fontsize(9.5)
    axes[0][0].set_ylabel("loss on held-out streams - loss on training streams [nats]")
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle("The network fits tracks cut from its own training streams better: the set is too small for long tracks", fontsize=11)
    fig.tight_layout()
    LS.save(fig, out, "04_train_vs_heldout")
    return num


def fig_experiments(models: Path, names: list[str], labels: dict, out: Path, tag: str = "07_experiments") -> dict:
    """Held-out loss (all, and resolved-orbit mocks per length) and recovery of the experiments on common mocks."""
    runs = {n: np.load(models / n / "heldout_nll.npz", allow_pickle=True) for n in names if (models / n / "heldout_nll.npz").exists()}
    if not runs:
        return {}
    ref = next(iter(runs.values()))
    S, res = ref["S"], ref["dE"] <= LS.DE_MAX
    groups = [("all", S > 0), ("resolved,\nshort", res & (S <= 12)), ("resolved,\nmedium", res & (S > 12) & (S <= 25)), ("resolved,\nlong", res & (S > 25)), ("unresolved", ~res)]
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4.9), gridspec_kw=dict(width_ratios=[1, 1.25, 1.25]))
    cols = [P.INK, P.C_ALL, P.C_BOUND, P.C_PASS, "#eda100", "#8f1f1f", "#7b8794", "#104281"]
    x, num, wd = np.arange(len(groups)), {}, 0.8 / len(runs)
    base = np.array([np.nanmean(np.where(np.isfinite(ref["nll"][g]), ref["nll"][g], np.nan)) for _, g in groups])
    for i, (n, d) in enumerate(runs.items()):
        nll = np.where(np.isfinite(d["nll"]), d["nll"], np.nan)
        v = np.array([np.nanmean(nll[g]) for _, g in groups])
        dd = nll - np.where(np.isfinite(ref["nll"]), ref["nll"], np.nan)  # paired with the first run, on the mocks both have
        diff = np.array([np.nanmean(dd[g]) for _, g in groups])
        se = np.array([np.nanstd(dd[g]) / np.sqrt(max(np.isfinite(dd[g]).sum(), 1)) for _, g in groups])
        if i:
            ax.bar(x + (i - len(runs) / 2) * wd, diff, wd * 0.9, yerr=se, color=cols[i % len(cols)], label=labels.get(n, n), capsize=2)
        num[n] = dict(label=labels.get(n, n), mocks=int(np.isfinite(nll).sum()), loss_nats={lab.replace("\n", " "): float(q) for (lab, _), q in zip(groups, v)},
                      minus_first_run={lab.replace("\n", " "): [float(q), float(e)] for (lab, _), q, e in zip(groups, diff, se)})  # fmt: skip
        if "post_mean" in d.files:
            mean, sd, truth = d["post_mean"], d["post_sd"], d["truth"]
            k = len(truth)
            sel = res[:k] if n != names[0] or True else slice(None)
            corr = [float(np.corrcoef(mean[sel, j], truth[sel, j])[0, 1]) for j in range(mean.shape[1])]
            width = [float(np.mean(sd[sel, j])) for j in range(mean.shape[1])]
            num[n] |= dict(recovery_correlation=dict(zip(PARAMS, corr)), posterior_sd=dict(zip(PARAMS, width)))
            xx = np.arange(len(PARAMS)) + (i - (len(runs) - 1) / 2) * (0.8 / len(runs))
            ax2.plot(xx, corr, ls="none", marker="o", ms=5, color=cols[i % len(cols)], label=labels.get(n, n))
            ax3.plot(xx, width, ls="none", marker="o", ms=5, color=cols[i % len(cols)])
    ax.axhline(0, color=P.INK, lw=0.8)
    ax.set_xticks(x, [lab for lab, _ in groups])
    lo, hi = ax.get_ylim()
    if hi > 1.5:  # one bar far above the rest: cut the axis there and write its value
        ax.set_ylim(lo, 1.5)
        for patch in ax.patches:
            if patch.get_height() > 1.5:
                ax.text(patch.get_x() + patch.get_width() / 2, 1.42, f"{patch.get_height():+.1f}", ha="center", va="top", fontsize=7.5, color="white", rotation=90)
    ax.set(ylabel="held-out loss minus the first run's [nats]", title=f"Loss on the same mock tracks, against '{labels.get(names[0], names[0])}'")
    ax.title.set_fontsize(9)
    ax.legend(fontsize=7.5, loc="upper left")
    for a, yl, tt in ((ax2, "correlation of the posterior mean with the truth", "Recovery, mocks of resolved streams"), (ax3, "posterior sd / prior sd (latent)", "Posterior width, same mocks")):
        a.set_xticks(np.arange(len(PARAMS)), PARAMS, fontsize=8, rotation=45)
        a.set(ylabel=yl, title=tt)
    ax2.legend(fontsize=7.5, loc="center", bbox_to_anchor=(0.36, 0.74))
    fig.tight_layout()
    LS.save(fig, out, tag)
    return num


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/extragalactic/C26_test")
    ap.add_argument("--subdir", default="train_v2")
    ap.add_argument("--run", default="npe_trainv2_1000epoch_vr4_5e4lr")
    ap.add_argument("--experiments", nargs="*", default=["q_base", "q_flowbig", "q_sum128", "q_fm", "q_dm", "q_resolved"])
    ap.add_argument("--data-size", nargs="*", default=["f_base", "q_base", "s_data06"])
    a = ap.parse_args()
    pick_device(cpu=True)
    base = REPO / a.out
    study, plots, models = base / "accuracy_study", base / "plots" / a.subdir, base / "models"
    figs, nums = study / "figures", study / "numbers"
    figs.mkdir(parents=True, exist_ok=True)
    nums.mkdir(exist_ok=True)
    labels = dict(q_base="baseline (transformer 64 x 2, spline flow 8 x 512)", q_flowbig="flow 12 x 1024", q_sum128="transformer 128 x 4", q_fm="flow matching",
                  q_dm="diffusion", q_resolved="trained on resolved orbits only", f_base="all 10 M streams", s_data06="1/16 of the streams (625 k)")  # fmt: skip
    num = {}
    if (plots / "track_noise" / "track_noise.npz").exists():
        num["loss_by_class"] = fig_loss_by_class(plots / "track_noise" / "track_noise.npz", figs)
    if (models / a.run / "evaluation" / "weights.npz").exists():
        num["importance_sampling"] = fig_importance(models / a.run, figs)
    if (nums / "train_vs_heldout.npz").exists():
        num["train_vs_heldout"] = fig_train_vs_heldout(nums / "train_vs_heldout.npz", figs)
    num["experiments"] = fig_experiments(models, a.experiments, labels, figs)
    num["data_size"] = fig_experiments(models, a.data_size, labels | dict(q_base="1/4 of the streams (2.5 M)"), figs, "07b_data_size")
    (nums / "study.json").write_text(json.dumps(num, indent=1))
    copies = [(plots / "track_noise" / "track_noise.png", "02_track_noise.png"), (models / a.run / "prior_weights" / "summary.png", "05_prior_weights_summary.png"),
              (models / a.run / "prior_weights" / "gallery_p1.png", "05_gallery_p1.png"), (models / a.run / "prior_weights" / "gallery_p2.png", "05_gallery_p2.png"),
              (plots / "symmetries" / "symmetries.png", "06_symmetries.png"), (plots / "particle_svd" / "particle_svd.png", "08_particle_svd.png"),
              (plots / "long_streams" / "1_census.png", "long_streams_census.png"), (plots / "long_streams" / "4_training.png", "long_streams_training.png"),
              (plots / "particle_number" / "good_samples.png", "09_good_samples.png"), (plots / "particle_number" / "particle_number.png", "10_particle_number.png"),
              (plots / "particle_number" / "gallery.png", "11_particle_gallery.png")]  # fmt: skip
    for src, name in copies:
        if src.exists():
            shutil.copy2(src, figs / name)
    for src in (plots / "track_noise" / "track_noise.json", plots / "symmetries" / "symmetries.json", plots / "particle_svd" / "particle_svd.json",
                plots / "long_streams" / "long_streams.json", models / a.run / "prior_weights" / "prior_weights.json", plots / "particle_number" / "particle_number.json",
                base / "plots" / "particles_pilot" / "particle_store" / "particle_store.json"):  # fmt: skip
        if src.exists():
            shutil.copy2(src, nums / src.name)
    print(f"wrote {study}: " + ", ".join(sorted(p.name for p in figs.iterdir())))


if __name__ == "__main__":
    main()
