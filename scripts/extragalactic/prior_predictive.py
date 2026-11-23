"""Prior predictive checks of a simulated set against the observed STRRINGS tracks.

    uv run python scripts/extragalactic/prior_predictive.py --subdir train_v1

Works on a finished set or on the shards written so far (they are independent prior draws). Reads the shards
directly, so `generate.py analyze` is not needed. Writes plots/<subdir>/prior_predictive/ and
data/<subdir>/prior_predictive.json.

Each observed track is centred on its median angle (the StreaMAX-fit convention) and compared with every
simulated track that covers all of its bins, in either sense of rotation. Three questions:

  summaries   where each observed track falls in the prior predictive distribution of track summaries (mean
              radius, radial contrast, slope, curvature), among the simulations that cover its bins
  envelope    the observed radii against the prior predictive band of r(phi) on the same bins
  coverage    how many simulations cover each track and how many resemble it, split by progenitor state
  joint       is the observed track inside the JOINT distribution of the simulated radii over its bins, not only
              inside each bin's band: corner plots of the bins, and two rank tests per track (Mahalanobis
              distance, distance to the nearest simulation). The simulations get the observed errors added.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "extragalactic"))
import inspect_prior as I  # noqa: E402  observed tracks
import plot as P  # noqa: E402  palette and style

plt = P.plt
C_SURV, C_DISS = P.C_BOUND, P.C_ALL  # progenitor still there | dissolved (mass loss)


def load_set(ddir: Path, ordering: str) -> dict:
    """Tracks in the requested ordering (whichever of track/ and track_alt/ holds it), parameters and flags."""
    cols: dict[str, list] = {}
    cfg = None
    for f in sorted(ddir.glob("shard_*.h5")):
        with h5py.File(f, "r") as h:
            cfg = json.loads(h.attrs["config"])
            grp = "track" if h["track"].attrs.get("ordering", cfg["track_ordering"]) == ordering else "track_alt"
            if grp not in h:
                raise SystemExit(f"{f} has no track in the {ordering} ordering")
            for k, path in dict(index="index", theta="theta", status="status", dissolved="prog/dissolved",
                                count=f"{grp}/count", r_med=f"{grp}/r_med", r_hw=f"{grp}/r_hw",
                                unbound="flags/unbound", n_bad="diag/n_bad", n_peri="diag/n_peri").items():
                cols.setdefault(k, []).append(h[path][:])
    if cfg is None:
        raise SystemExit(f"no shards in {ddir}")
    d = {k: np.concatenate(v) for k, v in cols.items()}
    d["cfg"], d["n"], d["N"] = cfg, len(d["index"]), cfg["n_particles"]
    d["failed"] = (d["status"] & 16) > 0
    d["lossy"] = d["n_bad"] > 0.01 * d["N"]  # lost more than 1 % of its particles at the step cap
    return d


def window(sim: dict, o: dict, nb: int, n_cover: int, pool: np.ndarray):
    """Simulations covering every bin of the observed track, in the better of the two senses of rotation.
    -> row indices, their radii on the observed bins (n, bins), and the rms fractional distance to the data."""
    best_r, best_d = None, None
    for bins in (o["bins"], nb - 1 - o["bins"]):
        ok = pool & (sim["count"][:, bins] >= n_cover).all(1)
        r = np.full((sim["n"], len(bins)), np.nan)
        r[ok] = sim["r_med"][ok][:, bins]
        dist = np.sqrt(np.nanmean((r / o["r"] - 1.0) ** 2, axis=1))
        if best_d is None:
            best_r, best_d = r, dist
        else:
            use = np.where(np.isnan(best_d), True, dist < best_d) & ~np.isnan(dist)
            best_r[use], best_d[use] = r[use], dist[use]
    rows = np.nonzero(np.isfinite(best_d))[0]
    return rows, best_r[rows], best_d[rows]


def summaries(phi: np.ndarray, r: np.ndarray) -> dict:
    """Track summaries on the observed bins; r is (n, bins) or (bins,). Slope and curvature come from a
    quadratic fit of ln r against phi in radians, so they are independent of the size of the stream."""
    r = np.atleast_2d(r).astype(np.float64)
    x = np.deg2rad(phi - phi.mean())
    A = np.stack([np.ones_like(x), x, x * x], 1)
    coef = np.linalg.lstsq(A, np.log(r).T, rcond=None)[0]  # (3, n)
    return dict(mean_radius=r.mean(1), contrast=(r.max(1) - r.min(1)) / r.mean(1), slope=coef[1], curvature=coef[2])


def joint_tests(o: dict, r: np.ndarray, rng, m_ref: int = 3000):
    """Rank tests of the observed track in the joint distribution of ln r over its bins.

    The simulated radii are first given the noise of the data (Gaussian, the observed fractional error of each
    bin), so that both sides are 'observations'. Each test returns the fraction of simulations that are LESS
    extreme than the observed track: values near 1 mean the data are in the far tail or outside.
      maha        Mahalanobis distance from the simulated mean (covariance shrunk towards its diagonal)
      maha_shape  the same after removing each track's mean ln r: shape only, not size
      nn          distance to the nearest simulation, against each simulation's distance to its own nearest
    -> the results and the noisy ln r of the simulations (for the corner plot)."""
    n, d = r.shape
    X = np.log(np.clip(r * (1.0 + (o["r_err"] / o["r"]) * rng.standard_normal(r.shape)), 1e-3, None))
    x = np.log(o["r"])
    out: dict = dict(n=n, d=d, maha=None, maha_shape=None, nn=None)

    def maha(A, a):
        mu, S = A.mean(0), np.atleast_2d(np.cov(A.T))
        lam = float(np.clip(A.shape[1] / A.shape[0], 0.02, 0.5))
        Si = np.linalg.inv((1 - lam) * S + lam * np.diag(np.diag(S)))
        d2 = np.einsum("ni,ij,nj->n", A - mu, Si, A - mu)
        return float((d2 < (a - mu) @ Si @ (a - mu)).mean())

    if n >= 5 * d and n >= 50:
        out["maha"] = maha(X, x)
        out["maha_shape"] = maha((X - X.mean(1, keepdims=True))[:, :-1], (x - x.mean())[:-1])
    if n >= 50:
        Y = X[rng.choice(n, min(n, m_ref), replace=False)]
        sq = (Y * Y).sum(1)
        D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * Y @ Y.T, 0.0) / d
        np.fill_diagonal(D2, np.inf)
        nn_sim = np.sqrt(D2.min(1))
        nn_obs = float(np.sqrt(((Y - x) ** 2).mean(1).min()))
        out.update(nn=float((nn_sim < nn_obs).mean()), nn_obs_rms=nn_obs, nn_sim_median_rms=float(np.median(nn_sim)))
    return out, X


def corner_bins(o: dict, X: np.ndarray, jt: dict, path: Path, max_bins: int = 8) -> None:
    """Pairwise distributions of the simulated ln r over (up to `max_bins` of) the observed bins, with the
    observed track marked. X already carries the observed noise."""
    d = X.shape[1]
    sel = np.unique(np.round(np.linspace(0, d - 1, min(d, max_bins))).astype(int))
    k = len(sel)
    L10 = X[:, sel] / np.log(10.0)
    xo = np.log10(o["r"][sel])
    lo, hi = np.percentile(L10, 0.5, axis=0), np.percentile(L10, 99.5, axis=0)
    lo, hi = np.minimum(lo, xo - 0.08), np.maximum(hi, xo + 0.08)
    fig, axes = plt.subplots(k, k, figsize=(1.75 * k + 0.6, 1.75 * k + 0.4), squeeze=False)
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            ax.grid(False)
            if i == j:
                h, e = np.histogram(L10[:, i], bins=np.linspace(lo[i], hi[i], 31))
                ax.stairs(h, e, color=P.C_ALL, fill=True, alpha=0.3, lw=0)
                ax.stairs(h, e, color=P.C_ALL, lw=1.4)
                ax.axvline(xo[i], color=P.INK, lw=1.6)
                ax.set_yticks([])
                ax.set_xlim(lo[i], hi[i])
                ax.set_title(f"$\\phi$ = {o['phi'][sel[i]]:+.0f}$^\\circ$", fontsize=8)
            else:
                H, ex, ey = np.histogram2d(L10[:, j], L10[:, i], bins=[np.linspace(lo[j], hi[j], 26), np.linspace(lo[i], hi[i], 26)])
                ax.pcolormesh(ex, ey, np.where(H > 0, H, np.nan).T, cmap=P.SEQ, rasterized=True)
                ax.plot([xo[j]], [xo[i]], marker="o", ms=6.5, color=P.INK, mfc=I.C_OBS, mew=1.0, ls="none", zorder=5)
                ax.set_xlim(lo[j], hi[j])
                ax.set_ylim(lo[i], hi[i])
            if i == k - 1:
                ax.set_xlabel(r"$\log_{10} r$ [kpc]", fontsize=7.5)
                ax.tick_params(axis="x", labelsize=6)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(r"$\log_{10} r$ [kpc]", fontsize=7.5)
                ax.tick_params(axis="y", labelsize=6)
            elif i != j:
                ax.set_yticklabels([])
    f = lambda v: "-" if v is None else f"{v:.2f}"
    fig.suptitle(f"{o['name']}: {jt['n']:,} simulations covering its {jt['d']} bins ({k} shown), observed errors added\n"
                 f"share of simulations less extreme than the data: Mahalanobis {f(jt['maha'])}, shape only "
                 f"{f(jt['maha_shape'])}, nearest neighbour {f(jt['nn'])}", fontsize=9, y=0.995)
    h = [plt.Line2D([], [], marker="o", ms=6.5, color=P.INK, mfc=I.C_OBS, ls="none", label="observed track"),
         plt.Line2D([], [], color=P.C_ALL, lw=1.4, label="simulations (colour: number per cell)")]
    fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.97, 0.9), fontsize=8.5)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.07, top=0.92, wspace=0.06, hspace=0.06)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


SUMMARY_LABELS = dict(mean_radius="mean radius [kpc]", contrast=r"radial contrast $(r_{\max}-r_{\min})/\bar r$",
                      slope=r"slope $d\ln r/d\phi$ [1/rad]", curvature=r"curvature of $\ln r(\phi)$ [1/rad$^2$]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="outputs/extragalactic/C26_test")
    ap.add_argument("--subdir", default="")
    ap.add_argument("--ordering", default="c26", help="c26 or gibbons14")
    ap.add_argument("--corner-min", type=int, default=150, help="corner plot for tracks covered by at least this many simulations")
    arg = ap.parse_args()
    root = REPO / arg.out
    ddir = root / "data" / arg.subdir if arg.subdir else root / "data"
    pdir = (root / "plots" / arg.subdir if arg.subdir else root / "plots") / "prior_predictive"
    pdir.mkdir(parents=True, exist_ok=True)
    sim = load_set(ddir, arg.ordering)
    cfg = sim["cfg"]
    obs = sorted(I.load_strrings(root / "data" / "strrings", cfg), key=lambda o: len(o["bins"]))
    nb = sim["count"].shape[1]
    n_cover = int(round(50 * sim["N"] / 10_000))
    pool = ~sim["failed"] & ~sim["lossy"]
    out: dict = dict(n_streams=sim["n"], n_usable=int(pool.sum()), failed_frac=float(sim["failed"].mean()),
                     lossy_frac=float(sim["lossy"].mean()), dissolved_frac=float(sim["dissolved"].mean()),
                     ordering=arg.ordering, n_cover=n_cover, complete=sim["n"] == cfg["n_streams"], per_track={})
    keys = list(SUMMARY_LABELS)
    res = []
    for o in obs:
        rows, r, dist = window(sim, o, nb, n_cover, pool)
        so = {k: float(v[0]) for k, v in summaries(o["phi"], o["r"]).items()}
        ss = summaries(o["phi"], r) if len(rows) else {k: np.array([]) for k in keys}
        q = {k: (float((ss[k] < so[k]).mean()) if len(rows) >= 20 else None) for k in keys}
        diss = sim["dissolved"][rows].astype(bool)
        band = np.nanpercentile(r, [2.5, 16, 50, 84, 97.5], axis=0) if len(rows) >= 20 else None
        inside = (float(((o["r"] >= band[0]) & (o["r"] <= band[4])).mean()) if band is not None else None)
        near = dist < 0.10
        out["per_track"][o["name"]] = dict(
            n_bins=len(o["bins"]), median_frac_err=o["frac_err"], n_covering=int(len(rows)),
            n_within_20pct=int((dist < 0.20).sum()), n_within_10pct=int(near.sum()),
            n_within_10pct_dissolved=int((near & diss).sum()), n_within_10pct_surviving=int((near & ~diss).sum()),
            best_rms=float(dist.min()) if len(rows) else None, observed=so, quantile=q,
            frac_bins_inside_95pct_band=inside, covering_dissolved_frac=float(diss.mean()) if len(rows) else None,
            covering_bound_frac=float((~sim["unbound"][rows].astype(bool)).mean()) if len(rows) else None)
        res.append((o, rows, r, dist, so, ss, q, band, diss))

    # ---------------------------------------------------------------- 1. quantiles of the observed summaries
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 0.26 * len(obs) + 1.6), sharey=True)
    y = np.arange(len(obs))
    for ax, k in zip(axes, keys):
        qv = np.array([np.nan if t[6][k] is None else t[6][k] for t in res])
        ax.axvspan(0.025, 0.975, color=P.GRID, alpha=0.6, lw=0)
        ax.axvline(0.5, color=P.AXIS, lw=1.0)
        outside = (qv < 0.025) | (qv > 0.975)
        ax.plot(qv[~outside], y[~outside], "o", ms=6, color=P.C_ALL, mec=P.SURFACE, mew=0.8, ls="none")
        ax.plot(qv[outside], y[outside], "o", ms=6, color=P.C_BOUND, mec=P.SURFACE, mew=0.8, ls="none")
        ax.plot(np.full(int(np.isnan(qv).sum()), 0.5), y[np.isnan(qv)], "x", ms=6, color=P.INK2, ls="none")
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("quantile of the observed value")
        ax.set_title(SUMMARY_LABELS[k], fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [f"{t[0]['name']} ({len(t[0]['bins'])} bins, {len(t[1]):,} sims)" for t in res], fontsize=6.8)
    h = [plt.Line2D([], [], marker="o", ms=6, color=P.C_ALL, ls="none", label="inside the central 95 %"),
         plt.Line2D([], [], marker="o", ms=6, color=P.C_BOUND, ls="none", label="outside it"),
         plt.Line2D([], [], marker="x", ms=6, color=P.INK2, ls="none", label="fewer than 20 simulations cover the track")]
    fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    fig.suptitle(f"Where each observed track falls in the prior predictive, among the simulations covering its bins "
                 f"({sim['n']:,} prior draws, {arg.ordering} ordering)", y=1.045, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("pdf", "png"):
        fig.savefig(pdir / f"summary_quantiles.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- 2. envelopes of r(phi)
    ncol = 8
    nrow = -(-len(obs) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.75 * ncol, 2.45 * nrow), squeeze=False)
    for ax, (o, rows, r, dist, so, ss, q, band, diss) in zip(axes.ravel(), res):
        if band is not None:
            ax.fill_between(o["phi"], band[0], band[4], color=P.C_ALL, alpha=0.18, lw=0)
            ax.fill_between(o["phi"], band[1], band[3], color=P.C_ALL, alpha=0.32, lw=0)
            ax.plot(o["phi"], band[2], color=P.C_ALL, lw=1.4)
            for j in np.argsort(dist)[:3]:
                ax.plot(o["phi"], r[j], color=P.C_PASS, lw=1.0, alpha=0.9)
        ax.errorbar(o["phi"], o["r"], yerr=o["r_err"], fmt="o", ms=2.8, color=P.INK, mfc=P.SURFACE, mew=0.8, elinewidth=0.8, zorder=5)
        ax.set_yscale("log")
        ax.tick_params(labelsize=6)
        best = f"{100 * dist.min():.0f} %" if len(rows) else "-"
        ax.set_title(f"{o['name']}\n{len(rows):,} sims, best {best}", fontsize=6.8)
    for ax in axes.ravel()[len(obs):]:
        ax.axis("off")
    h = [plt.Line2D([], [], marker="o", ms=4, color=P.INK, mfc=P.SURFACE, ls="none", label="observed"),
         plt.Line2D([], [], color=P.C_ALL, lw=1.4, label="prior predictive: median, 68 %, 95 %"),
         plt.Line2D([], [], color=P.C_PASS, lw=1.0, label="the three closest simulations")]
    fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    fig.suptitle("Observed radii [kpc] against the prior predictive band on the same bins (x: degrees from the median angle)",
                 y=1.03, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    for ext in ("pdf", "png"):
        fig.savefig(pdir / f"envelopes.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- 3. coverage, by progenitor state
    fig, ax = plt.subplots(figsize=(9.5, 0.27 * len(obs) + 1.4))
    z = lambda v: np.maximum(np.array(v, float), 0.5)
    pt = [out["per_track"][t[0]["name"]] for t in res]
    ax.barh(y, z([p["n_covering"] for p in pt]), height=0.72, color=P.GRID, label="cover every observed bin")
    ax.barh(y, z([p["n_within_20pct"] for p in pt]), height=0.72, color=P.AXIS, label="within 20 % rms")
    ax.plot(z([p["n_within_10pct_surviving"] for p in pt]), y - 0.14, "o", ms=5, color=C_SURV, mec=P.SURFACE, mew=0.7, ls="none",
            label="within 10 % rms, progenitor still there")
    ax.plot(z([p["n_within_10pct_dissolved"] for p in pt]), y + 0.14, "o", ms=5, color=C_DISS, mec=P.SURFACE, mew=0.7, ls="none",
            label="within 10 % rms, progenitor dissolved")
    ax.set_yticks(y, [f"{t[0]['name']} ({len(t[0]['bins'])} bins)" for t in res], fontsize=6.8)
    ax.set_xscale("log")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"simulations, of {int(pool.sum()):,} usable prior draws (0 drawn at 0.5)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Simulations comparable with each observed track", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(pdir / f"coverage.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- 4. joint distribution over the bins
    rng = np.random.default_rng(0)
    jts = []
    for (o, rows, r, dist, so, ss, q, band, diss) in res:
        if len(rows) >= 50:
            jt, X = joint_tests(o, r, rng)
            if len(rows) >= arg.corner_min:
                corner_bins(o, X, jt, pdir / f"corner_{o['name']}.png")
        else:
            jt = dict(n=len(rows), d=len(o["bins"]), maha=None, maha_shape=None, nn=None)
        jts.append(jt)
        out["per_track"][o["name"]]["joint"] = jt
    tests = [("maha", "Mahalanobis distance"), ("maha_shape", "Mahalanobis distance, shape only"), ("nn", "distance to the nearest simulation")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 0.26 * len(obs) + 1.6), sharey=True)
    for ax, (key, lab) in zip(axes, tests):
        v = np.array([np.nan if jt[key] is None else jt[key] for jt in jts])
        ax.axvspan(0.95, 1.03, color=P.GRID, alpha=0.8, lw=0)
        ax.plot(v[v <= 0.95], y[v <= 0.95], "o", ms=6, color=P.C_ALL, mec=P.SURFACE, mew=0.8, ls="none")
        ax.plot(v[v > 0.95], y[v > 0.95], "o", ms=6, color=P.C_BOUND, mec=P.SURFACE, mew=0.8, ls="none")
        ax.plot(np.full(int(np.isnan(v).sum()), 0.5), y[np.isnan(v)], "x", ms=6, color=P.INK2, ls="none")
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("share of simulations less extreme than the data")
        ax.set_title(lab, fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [f"{t[0]['name']} ({len(t[0]['bins'])} bins, {len(t[1]):,} sims)" for t in res], fontsize=6.8)
    h = [plt.Line2D([], [], marker="o", ms=6, color=P.C_ALL, ls="none", label="inside the simulated distribution"),
         plt.Line2D([], [], marker="o", ms=6, color=P.C_BOUND, ls="none", label="more extreme than 95 % of the simulations"),
         plt.Line2D([], [], marker="x", ms=6, color=P.INK2, ls="none", label="too few simulations for the test")]
    fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    fig.suptitle("Is each observed track inside the joint distribution of the simulated radii over its bins? "
                 "(observed errors added to the simulations)", y=1.045, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("pdf", "png"):
        fig.savefig(pdir / f"joint_tests.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)

    qs = np.array([[np.nan if t[6][k] is None else t[6][k] for k in keys] for t in res])
    out["summary"] = dict(
        tracks=len(obs), tracks_with_20_covering=int(sum(len(t[1]) >= 20 for t in res)),
        outside_95pct={k: int(np.nansum((qs[:, i] < 0.025) | (qs[:, i] > 0.975))) for i, k in enumerate(keys)},
        tracks_with_any_summary_outside_95pct=int(np.sum(np.nansum((qs < 0.025) | (qs > 0.975), axis=1) > 0)),
        tracks_without_sim_within_10pct=int(sum(p["n_within_10pct"] == 0 for p in pt)),
        tracks_with_100_within_10pct=int(sum(p["n_within_10pct"] >= 100 for p in pt)),
        median_n_covering=float(np.median([p["n_covering"] for p in pt])),
        joint={key: dict(tested=int(sum(jt[key] is not None for jt in jts)),
                         beyond_95pct=[t[0]["name"] for t, jt in zip(res, jts) if jt[key] is not None and jt[key] > 0.95])
               for key, _ in tests})
    (ddir / "prior_predictive.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {pdir}/summary_quantiles, envelopes, coverage (.pdf, .png) and {ddir / 'prior_predictive.json'}")
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
