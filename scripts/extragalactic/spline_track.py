"""Smooth splines of stream tracks in the unwrapped angle, and the curvature test of arXiv:2606.21774.

    uv run python scripts/extragalactic/spline_track.py all [--workers 16]
    uv run python scripts/extragalactic/spline_track.py observed | mocks | premise | smoothing | phase | figures

Everything runs on the CPU (numpy / scipy); no GPU is taken. Results go to <out>/spline_test/.

THE SPLINE. A track is r(phi), phi the unwrapped angle (the independent variable; the paper uses a normalised
arc length, which needs a curve drawn in the image, and the track files hold the binned track only). The fit is
a penalised cubic B-spline with one knot per bin width and a penalty lambda on the third differences of the
coefficients (changes of r'', the analogue of the paper's cost on changes of curvature; lambda -> infinity is a
parabola in phi). Errors are Gaussian with variance sigma_i^2 + s^2, s the systematic scatter of the paper's
likelihood (Chemaly et al. 2026b, eq. 4). lambda and s are not chosen: the posterior of (lambda, s) on a grid
is the restricted marginal likelihood times a flat prior in log lambda and in s, and curves are drawn from it
(first a cell of the grid, then the Gaussian of the coefficients). This replaces the optimiser + NUTS of the
paper; the model is linear, so the draws are exact.

THE CURVATURE TEST (the paper's eq. 4-9). With x = r cos phi, y = r sin phi the signed curvature is

    kappa = (r^2 + 2 r'^2 - r r'') / (r^2 + r'^2)^(3/2)      (derivatives in phi, radians)

kappa > 0: the track bends towards the side the host is on; kappa < 0: it bends away. For an orbit the
curvature vector and the acceleration projected on the sky lie within 90 deg of each other. A spherical host
centred on the origin therefore allows kappa > 0 only, at any depth along the line of sight. The paper's
likelihood of a halo model is the share of spline draws for which this holds at every one of I points evenly
spaced in arc length; its halo is a logarithmic potential flattened by q in the plane of the sky along a
position angle PA (only the direction of the acceleration enters, so v_c and r_s drop out).

Sub-commands
  observed   the 32 STRRINGS tracks: spline posterior, effective number of parameters, s, sign of the curvature
             along the track, the paper's likelihood on a (q, PA) grid.
  mocks      the same analysis on mock tracks cut from simulations with the bins, gaps and errors of every
             observed track (radii scaled to its median radius, which the simulator's length scaling allows):
             is what the splines show in the data produced by the simulator?
  premise    simulated streams with 300,000 particles: does the track obey the paper's criterion against the
             true halo, and does the sign of the curvature depend on the true flattening?
  smoothing  the spline as the model track: a 30,000-particle track smoothed across bins against the same
             parameters at 300,000 particles.
  phase      the spline as a track at any bin phase: the track in bins shifted by half a bin, predicted from
             the stored bins, against the particles (the 5,000 streams of data/particles_pilot).
  figures    all figures and README numbers from the saved files.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # many small problems: one thread per process, processes in a pool

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import BSpline, CubicSpline
from scipy.linalg import cho_factor, cho_solve, solve_triangular

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts" / "extragalactic"), str(REPO / "scripts" / "extragalactic" / "sbi")]
from config import Config  # noqa: E402
from data import host_acceleration, load_observed  # noqa: E402

OUT = REPO / "outputs/extragalactic/C26_test"
RES = OUT / "spline_test"
PILOT = OUT / "data/particles_pilot"
PN = OUT / "plots/train_v2/particle_number/sims"

DEG, PEN, RAD = 3, 3, 180.0 / np.pi
LAM = 10.0 ** np.linspace(-3.0, 7.0, 31)  # penalty grid, flat prior in log lambda
S_MAX = 5.0  # kpc, sigma_sys ~ U(0, 5) as in the run files
S_GRID = S_MAX * np.linspace(0.0, 1.0, 26) ** 2
Q_GRID, PA_GRID = np.round(np.arange(0.10, 1.0001, 0.05), 2), np.arange(-90.0, 90.0, 5.0)
P_SIG = 0.95  # the sign of the curvature is called significant above this posterior probability
MED = 1.2533  # standard error of a median over that of a mean (normal scatter)
VAR_RATIO, CAP, DE_MAX, WIDE = 4.0, 49, 0.01, 0.25  # guard 2 of the runs, bins per side, class limits (long_streams.py)


# ------------------------------------------------------------------------------------------------------
# penalised spline with the posterior of its smoothing parameter
# ------------------------------------------------------------------------------------------------------
class PSpline:
    """Cubic B-splines on uniform knots of spacing h [deg] over [lo, hi], third-difference penalty."""

    def __init__(self, lo: float, hi: float, h: float = 10.0):
        n = max(1, int(np.ceil((hi - lo) / h - 1e-9)))
        self.t, self.K = lo + h * np.arange(-DEG, n + DEG + 1), n + DEG
        D = np.diff(np.eye(self.K), PEN, axis=0)
        self.P = D.T @ D

    def basis(self, x, nu: int = 0) -> np.ndarray:
        """(len(x), K): the nu-th derivative of every basis function, per RADIAN."""
        return BSpline(self.t, np.eye(self.K), DEG)(np.asarray(x, np.float64), nu) * RAD**nu


def _quantiles(grid, p, qs=(0.16, 0.5, 0.84)):
    c = np.cumsum(p) / p.sum()
    return [float(np.interp(q, np.r_[0.0, c], np.r_[grid[0], grid])) for q in qs]


def fit(sp: PSpline, B, y, sig, s_grid=(0.0,), n_draw: int = 0, rng=None, cov: bool = False) -> dict:
    """Posterior of the coefficients, of lambda (grid LAM) and of s (grid s_grid) for y ~ N(B beta, sig^2 + s^2).

    Returns the grid posterior `p` (s, lambda), the posterior-mean coefficients `beta`, the mean effective
    number of parameters `edf` (trace of the smoother), quantiles of s and log10 lambda, `draws` (n_draw, K)
    and, with `cov`, the posterior covariance of the coefficients."""
    s_grid = np.asarray(s_grid, np.float64)
    ns, nl, K = len(s_grid), len(LAM), sp.K
    logp, edf, beta = np.full((ns, nl), -np.inf), np.zeros((ns, nl)), np.zeros((ns, nl, K))
    GB = []
    for a, s in enumerate(s_grid):
        w = 1.0 / (sig**2 + s**2)
        G, b = B.T @ (w[:, None] * B), B.T @ (w * y)
        GB.append((G, b))
        for c, lam in enumerate(LAM):
            try:
                L = cho_factor(G + lam * sp.P, lower=True)
            except np.linalg.LinAlgError:
                continue
            beta[a, c] = cho_solve(L, b)
            logp[a, c] = (-0.5 * (np.sum(w * y * y) - b @ beta[a, c]) + 0.5 * np.sum(np.log(w)) + 0.5 * (K - PEN) * np.log(lam)
                          - np.sum(np.log(np.diag(L[0]))))  # fmt: skip
            edf[a, c] = np.trace(cho_solve(L, G))
    ws = np.gradient(s_grid) if ns > 1 else np.ones(1)  # flat prior in s on a non-uniform grid
    p = np.exp(logp - logp.max()) * ws[:, None]
    p /= p.sum()
    out = dict(p=p, beta=np.einsum("ac,ack->k", p, beta), edf=float((p * edf).sum()), loglam=_quantiles(np.log10(LAM), p.sum(0)),
               s=_quantiles(s_grid, p.sum(1)) if ns > 1 else [0.0] * 3, s_zero=float(p[0].sum()))  # fmt: skip
    if n_draw or cov:
        cells = rng.choice(ns * nl, n_draw, p=p.ravel()) if n_draw else np.zeros(0, int)  # rng is not used without draws
        use = np.union1d(np.unique(cells), np.flatnonzero(p.ravel() > 1e-5)) if cov else np.unique(cells)
        draws, C = np.zeros((n_draw, K)), np.zeros((K, K))
        for cell in use:
            a, c = divmod(int(cell), nl)
            L = np.linalg.cholesky(GB[a][0] + LAM[c] * sp.P)
            if cov:
                Ai = cho_solve((L, True), np.eye(K))
                C += p[a, c] * (Ai + np.outer(beta[a, c], beta[a, c]))
            m = cells == cell
            if m.any():
                draws[m] = beta[a, c] + solve_triangular(L.T, rng.standard_normal((K, int(m.sum()))), lower=False).T
        out["draws"] = draws
        if cov:
            out["cov"] = C / p.ravel()[use].sum() - np.outer(out["beta"], out["beta"])
    return out


def curve(sp: PSpline, beta, x) -> dict:
    """r, r', r'' (per radian), the numerator of the signed curvature (its sign) and the speed |dx/dphi| at x."""
    r, r1, r2 = (beta @ sp.basis(x, nu).T for nu in (0, 1, 2))
    return dict(r=r, r1=r1, r2=r2, num=r * r + 2 * r1 * r1 - r * r2, speed=np.sqrt(r * r + r1 * r1))


def shape_likelihood(phi_deg, c: dict) -> np.ndarray:
    """The paper's eq. 6-9 on the grid (Q_GRID, PA_GRID): share of the draws whose curvature vector lies within
    90 deg of the acceleration of the flattened logarithmic halo at every point. c: `curve` of the draws at the
    points phi_deg, arrays (draws, points)."""
    ph = np.deg2rad(phi_deg)
    er, ep = np.stack([np.cos(ph), np.sin(ph)], -1), np.stack([-np.sin(ph), np.cos(ph)], -1)
    pos = c["r"][..., None] * er  # (D, I, 2)
    kap = np.sign(c["num"])[..., None] * (c["r1"][..., None] * ep - c["r"][..., None] * er)  # direction of the curvature vector
    pa = np.deg2rad(PA_GRID)
    e1, e2 = np.stack([np.cos(pa), np.sin(pa)], -1), np.stack([-np.sin(pa), np.cos(pa)], -1)  # major and minor axis
    uu = np.einsum("dik,ak->adi", kap, e1) * np.einsum("dik,ak->adi", pos, e1)
    vv = np.einsum("dik,ak->adi", kap, e2) * np.einsum("dik,ak->adi", pos, e2)
    return np.stack([np.all(-(uu + vv / q**2) > 0, axis=-1).mean(-1) for q in Q_GRID])  # (q, PA)


def analyse(x, y, sig, rng, n_draw: int = 400, s_grid=S_GRID, full: bool = False) -> dict:
    """Spline posterior of one track (x [deg] increasing, y, sig [kpc]) and everything derived from it."""
    sp = PSpline(x[0], x[-1])
    f = fit(sp, sp.basis(x), y, sig, s_grid, n_draw, rng)
    xg = np.linspace(x[0], x[-1], 4 * int(round((x[-1] - x[0]) / 10.0)) + 1)
    cm, cd = curve(sp, f["beta"], xg), curve(sp, f["draws"], xg)
    ds = cm["speed"] / cm["speed"].sum()  # arc-length weights of the grid points
    p_away = (cd["num"] < 0).mean(0)
    arc = np.cumsum(cm["speed"]) - 0.5 * cm["speed"]
    xi = np.interp((np.arange(len(x)) + 0.5) / len(x) * cm["speed"].sum(), arc, xg)  # I = N points even in arc length
    ci = curve(sp, f["draws"], xi)
    like = shape_likelihood(xi, ci)
    k, a = np.unravel_index(np.argmax(like), like.shape)
    res = y - f["beta"] @ sp.basis(x).T
    out = dict(n=len(x), span=int(round((x[-1] - x[0]) / 10.0)) + 1, edf=f["edf"], s=f["s"], s_zero=f["s_zero"], loglam=f["loglam"],
               chi2=float(np.sum((res / sig) ** 2)), f_away=float(ds[p_away > P_SIG].sum()), f_toward=float(ds[p_away < 1 - P_SIG].sum()),
               f_exp=float(ds @ p_away), p_max=float(p_away.max()), n_flip=float(np.median((np.diff(np.sign(ci["num"]), axis=1) != 0).sum(1))),
               L_sph=float((ci["num"] > 0).all(1).mean()), L_max=float(like[k, a]), q_best=float(Q_GRID[k]), pa_best=float(PA_GRID[a]),
               L_q=like.mean(1))  # fmt: skip
    if full:
        lo, hi = np.percentile(cd["r"], [16, 84], axis=0)
        out |= dict(xg=xg, r=cm["r"], r_lo=lo, r_hi=hi, p_away=p_away, like=like, xi=xi, knots=sp.t, beta=f["beta"], post=f["p"])
    return out


# ------------------------------------------------------------------------------------------------------
# simulations
# ------------------------------------------------------------------------------------------------------
KEYS = ("index", "theta", "c26_vec", "status", "prog/dissolved", "diag/dE_rel", "diag/n_bad", "track/count", "track/r_med", "track/r_hw", "track/los_mean")


def read_shards(ddir: Path) -> dict:
    """The shards of a simulation set, in the order of `index`; tracks cut to CAP + 15 bins either side of the
    progenitor (`i0` is then the first bin right of it)."""
    cols = {k: [] for k in KEYS}
    files = sorted(Path(ddir).glob("shard_*.h5"))
    with h5py.File(files[0], "r") as f:
        cfg = json.loads(f.attrs["config"])
    i0, half = int(round(cfg["phi_max_deg"] / cfg["bin_deg"])), CAP + 15
    for p in files:
        with h5py.File(p, "r") as f:
            for k in KEYS:
                cols[k].append(f[k][:, i0 - half : i0 + half] if k.startswith("track/") else f[k][:])
    d = {k.split("/")[-1]: np.concatenate(v) for k, v in cols.items()}
    order = np.argsort(d["index"])
    d = {k: v[order] for k, v in d.items()}
    N = int(cfg["n_particles"])
    d["count"] = d["count"].astype(np.int64)
    return d | dict(i0=half, N=N, n_min=max(10, round(10 * N / 30000)), usable=((d["status"] & 16) == 0) & (d["n_bad"] <= 0.01 * N))


def run_of(valid: np.ndarray, i0: int, cap: int = CAP):
    """(a, b): unbroken valid bins left and right of the progenitor, at most `cap` each; valid (n, bins)."""
    left, right = valid[:, :i0][:, ::-1][:, :cap], valid[:, i0:][:, :cap]
    return np.where(left.all(1), left.shape[1], np.argmin(left, 1)), np.where(right.all(1), right.shape[1], np.argmin(right, 1))


def stream_class(d: dict, a, b) -> np.ndarray:
    """thin | wide | fold | unres, the classes of long_streams.py."""
    j = np.arange(d["count"].shape[1])[None]
    run = (j >= d["i0"] - a[:, None]) & (j < d["i0"] + b[:, None])
    with np.errstate(all="ignore"):
        width = np.nanmedian(np.where(run, d["r_hw"] / d["r_med"], np.nan), axis=1)
    return np.where(d["dE_rel"] > DE_MAX, "unres", np.where((d["status"] & 32) > 0, "fold", np.where(width > WIDE, "wide", "thin")))


def ok_bins(d: dict, n_min=None) -> np.ndarray:
    return (d["count"] >= (d["n_min"] if n_min is None else n_min)) & np.isfinite(d["r_med"]) & (d["r_hw"] > 0)


def particle_sigma(hw, n):
    """Standard error of the median radius of a bin."""
    return MED * hw / np.sqrt(np.maximum(n, 1))


_G: dict = {}  # arrays shared with forked workers


# --- observed ------------------------------------------------------------------------------------------
def observed_tracks() -> list[dict]:
    out = []
    for t in load_observed(Config()):
        x = t["theta_deg"]
        out.append(dict(name=t["name"], x=x, y=t["r"], sig=t["r_err"], kk=np.rint((x - x[0]) / 10.0).astype(int)))
    return out


def _observed(i: int) -> dict:
    t = _G["tracks"][i]
    return analyse(t["x"], t["y"], t["sig"], np.random.default_rng([2611, i]), full=True) | dict(name=t["name"])


def cmd_observed(a) -> None:
    _G["tracks"] = observed_tracks()
    with Pool(a.workers) as pool:
        res = pool.map(_observed, range(len(_G["tracks"])))
    arr = {}
    for t, r in zip(_G["tracks"], res):
        for k in ("xg", "r", "r_lo", "r_hi", "p_away", "like", "xi", "knots", "beta", "post", "L_q"):
            arr[f"{t['name']}/{k}"] = r.pop(k)
        for k in ("x", "y", "sig"):
            arr[f"{t['name']}/{k}"] = t[k]
    np.savez(RES / "observed.npz", q_grid=Q_GRID, pa_grid=PA_GRID, lam_grid=LAM, s_grid=S_GRID, **arr)
    (RES / "numbers").mkdir(exist_ok=True)
    json.dump(res, open(RES / "numbers/observed.json", "w"), indent=1)
    for r in res:
        print(f"{r['name']:18s} N {r['n']:2d} edf {r['edf']:5.1f} chi2 {r['chi2']:6.1f} s {r['s'][1]:5.2f} away {r['f_away']:.2f} toward {r['f_toward']:.2f} "
              f"L_sph {r['L_sph']:.3f} L_max {r['L_max']:.3f} q {r['q_best']:.2f}")  # fmt: skip


# --- mocks ---------------------------------------------------------------------------------------------
MOCK_KEYS = ("edf", "chi2", "f_away", "f_toward", "f_exp", "p_max", "n_flip", "L_sph", "L_max", "q_best")


def _mocks(task) -> dict:
    """`n` mock tracks with the bins and errors of observed track i, cut from the streams of pool `name`."""
    i, name, n = task
    t, d = _G["tracks"][i], _G["pools"][name]
    rng = np.random.default_rng([2612, i, len(name)])
    S, kk, i0 = int(t["kk"][-1]) + 1, t["kk"], d["i0"]
    lo, hi = np.maximum(0, S - d["b"]), np.minimum(S, d["a"])  # allowed bins left of the progenitor
    w = np.clip(hi - lo + 1, 0, None) * d["usable"]
    info = dict(name=t["name"], pool=name, streams=int((w > 0).sum()), guard2=True)
    if info["streams"] < 30:
        return info | dict(n=0)
    rows, tries, guard = [], 0, True
    while len(rows) < n:
        tries += 1
        if guard and tries > 300 * n:  # too few streams pass the particle-noise guard at these errors: drop it
            guard, info["guard2"] = False, False
        k = rng.choice(len(w), p=w / w.sum())
        l = rng.integers(lo[k], hi[k] + 1)
        b = i0 - l + kk
        m, hw, cnt = d["r_med"][k, b], d["r_hw"][k, b], d["count"][k, b]
        L = np.median(t["y"]) / np.median(m)
        if guard and np.any((L * hw) ** 2 / cnt * VAR_RATIO > t["sig"] ** 2):
            continue
        r = analyse(t["x"], L * m + t["sig"] * rng.standard_normal(len(m)), t["sig"], rng, n_draw=200)
        rows.append([r[c] for c in MOCK_KEYS] + [r["s"][1], r["s_zero"], d["theta"][k, 2], float(d["dissolved"][k]), L, float(d["cls_i"][k])] + list(r["L_q"]))
    rows = np.array(rows)
    nk = len(MOCK_KEYS)
    return info | dict(n=n, accept=n / tries, **{c: rows[:, j] for j, c in enumerate(MOCK_KEYS)}, s_med=rows[:, nk], s_zero=rows[:, nk + 1],
                       q_true=rows[:, nk + 2], dissolved=rows[:, nk + 3], scale=rows[:, nk + 4], cls=rows[:, nk + 5], L_q=rows[:, nk + 6 :])  # fmt: skip


CLS = ("thin", "wide", "fold", "unres")


def load_pool(ddir: Path) -> dict:
    d = read_shards(ddir)
    d["a"], d["b"] = run_of(ok_bins(d), d["i0"])
    d["cls"] = stream_class(d, d["a"], d["b"])
    d["cls_i"] = np.array([CLS.index(c) for c in d["cls"]])
    return d


def cmd_mocks(a) -> None:
    _G["tracks"] = observed_tracks()
    _G["pools"] = dict(prior=load_pool(PILOT / "."), long=load_pool(PN / "n90000/data"))
    tasks = [(i, name, a.n_mock) for name in _G["pools"] for i in range(len(_G["tracks"]))]
    with Pool(a.workers) as pool:
        res = pool.map(_mocks, tasks, chunksize=1)
    arr, meta = {}, []
    for r in res:
        meta.append({k: v for k, v in r.items() if not isinstance(v, np.ndarray)})
        arr |= {f"{r['pool']}/{r['name']}/{k}": v for k, v in r.items() if isinstance(v, np.ndarray)}
        print(meta[-1])
    np.savez(RES / "mocks.npz", **arr)
    json.dump(meta, open(RES / "numbers/mocks_meta.json", "w"), indent=1)


# --- premise: the paper's criterion against the true halo ------------------------------------------------
def _premise(k: int):
    d = _G["sim"]
    sl = slice(d["i0"] - d["a"][k], d["i0"] + d["b"][k])
    x = (np.arange(sl.start, sl.stop) - d["i0"] + 0.5) * 10.0
    m, los = d["r_med"][k, sl].astype(np.float64), d["los_mean"][k, sl].astype(np.float64)
    sp = PSpline(x[0], x[-1])
    f = fit(sp, sp.basis(x), m, particle_sigma(d["r_hw"][k, sl], d["count"][k, sl]), n_draw=200, rng=np.random.default_rng([2613, k]))
    c = curve(sp, f["draws"], x)
    ph = np.deg2rad(x)
    er, ep = np.stack([np.cos(ph), np.sin(ph)], -1), np.stack([-np.sin(ph), np.cos(ph)], -1)
    acc = host_acceleration(np.r_[d["theta"][k, :2], d["c26_vec"][k]], np.concatenate([m[:, None] * er, los[:, None]], -1), _G["flatten"])[:, :2]
    cm = curve(sp, f["beta"], x)
    nrm = cm["r1"][:, None] * ep - cm["r"][:, None] * er  # left normal of the track (towards the host side)
    an = np.sum(acc * nrm, -1) / np.linalg.norm(acc, axis=-1) / np.linalg.norm(nrm, axis=-1)  # cos(acceleration, normal)
    ar = -np.sum(acc * er, -1) / np.linalg.norm(acc, axis=-1)  # cos(acceleration, direction to the host centre)
    return np.stack([np.full(len(x), k), x, (c["num"] < 0).mean(0), an, ar, cm["num"] / cm["speed"] ** 3 * cm["r"]], -1)


def cmd_premise(a) -> None:
    d = _G["sim"] = read_shards(PN / "n300000/data")
    with h5py.File(sorted((PN / "n300000/data").glob("shard_*.h5"))[0], "r") as f:
        _G["flatten"] = json.loads(f.attrs["config"])["halo_flattening"]
    d["a"], d["b"] = run_of(ok_bins(d), d["i0"])
    cls = stream_class(d, d["a"], d["b"])
    use = np.flatnonzero(d["usable"] & (d["a"] + d["b"] >= 8))
    with Pool(a.workers) as pool:
        tab = np.concatenate(pool.map(_premise, use, chunksize=8))
    np.savez(RES / "premise.npz", table=tab, columns=["stream", "phi_deg", "p_away", "cos_acc_normal", "cos_acc_centre", "kappa_r"],
             q=d["theta"][:, 2], cls=cls, dissolved=d["dissolved"], used=use)  # fmt: skip
    print("premise:", len(use), "streams,", len(tab), "bins")


# --- smoothing: the spline as the model track -------------------------------------------------------------
VARIANTS = (("one spline, knot every bin", 10.0, False), ("one per arm, knot every bin", 10.0, True),
            ("one per arm, knot every 2 bins", 20.0, True), ("one per arm, knot every 3 bins", 30.0, True))  # fmt: skip


def arm_fit(x, y, sig, h: float = 10.0, split: bool = False, xe=None):
    """Spline of a model track with the particle noise `sig` as its errors: (mean, variance) at x, and the mean
    at `xe`. split: a spline for each side of the progenitor (phi = 0), where the two arms meet with a step. A
    side of fewer than 5 bins is left as it is (linear interpolation at xe)."""
    pred, var = y.copy(), sig**2
    pe = None if xe is None else np.full(len(xe), np.nan)
    for side in (-1, 1) if split else (0,):
        m = np.ones(len(x), bool) if side == 0 else x * side > 0
        me = None if xe is None else (np.ones(len(xe), bool) if side == 0 else xe * side > 0) & (xe >= x[m].min(initial=np.inf)) & (xe <= x[m].max(initial=-np.inf))
        if m.sum() < 5:
            if xe is not None and m.sum() > 1:
                pe[me] = np.interp(xe[me], x[m], y[m])
            continue
        sp = PSpline(x[m][0], x[m][-1], h)
        B = sp.basis(x[m])
        f = fit(sp, B, y[m], sig[m], cov=True)
        pred[m], var[m] = f["beta"] @ B.T, np.einsum("ik,kl,il->i", B, f["cov"], B)
        if xe is not None:
            pe[me] = f["beta"] @ sp.basis(xe[me]).T
    return pred, var, pe


def _smooth(k: int):
    d, ref = _G["sim"], _G["ref"]
    sl = slice(d["i0"] - d["a"][k], d["i0"] + d["b"][k])
    x = (np.arange(sl.start, sl.stop) - d["i0"] + 0.5) * 10.0
    m, sg = d["r_med"][k, sl].astype(np.float64), particle_sigma(d["r_hw"][k, sl], d["count"][k, sl]).astype(np.float64)
    r = _G["row"][k]
    ok = ok_bins(ref)[r, sl]
    cols = [np.full(len(x), k), x, m, sg**2, np.where(ok, ref["r_med"][r, sl], np.nan), particle_sigma(ref["r_hw"][r, sl], ref["count"][r, sl]) ** 2]
    for _, h, split in VARIANTS:
        cols += list(arm_fit(x, m, sg, h, split)[:2])
    return np.stack(cols, -1)


def cmd_smoothing(a) -> None:
    sel = np.load(PN.parent / "selection.npz")
    sub = np.flatnonzero(sel["in_sub"])  # the streams simulated with every particle number
    sims = {N: read_shards(PN / f"n{N}/data") for N in (30000, 90000, 300000)}
    d, ref = _G["sim"], _G["ref"] = sims[30000], sims[300000]
    _G["row"] = {int(k): j for j, k in enumerate(sub)}
    d["a"], d["b"] = run_of(ok_bins(d), d["i0"])
    use = sub[d["usable"][sub] & ref["usable"] & ((d["a"] + d["b"])[sub] >= 8)]
    with Pool(a.workers) as pool:
        tab = np.concatenate(pool.map(_smooth, use, chunksize=8))
    # usable length (bins of the unbroken run that pass both guards at a flat data error), as particle_number.py;
    # for a spline the variance of the guard is the posterior variance of the spline (as that of a mean)
    length, rows, bins = {}, tab[:, 0].astype(int), np.rint(tab[:, 1] / 10.0 - 0.5).astype(int) + d["i0"]
    for err in (0.3, 1.0, 3.0):
        for N, s in sims.items():
            al, bl = run_of(ok_bins(s) & (s["r_hw"] ** 2 / np.maximum(s["count"], 1) * VAR_RATIO <= err**2), s["i0"])
            length[f"{N}|{err}"] = (al + bl)[sub if N < 300000 else slice(None)]
        for j, (name, _, _) in enumerate(VARIANTS):
            v = np.full(d["r_med"].shape, np.inf)
            v[rows, bins] = tab[:, 7 + 2 * j] / MED**2
            al, bl = run_of(ok_bins(d) & (v * VAR_RATIO <= err**2), d["i0"])
            length[f"30000, {name}|{err}"] = (al + bl)[sub]
    cls = stream_class(d, d["a"], d["b"])
    np.savez(RES / "smoothing.npz", table=tab, columns=["stream", "phi_deg", "raw", "var_raw", "ref", "var_ref"] + [f"{c}: {n}" for n, _, _ in VARIANTS for c in ("mean", "var")],
             cls=cls[sub], is_long=sel["is_long"][sub], stream=sub, dissolved=d["dissolved"][sub], **{f"len/{k}": v for k, v in length.items()})  # fmt: skip
    print("smoothing:", len(use), "streams,", len(tab), "bins")


# --- phase: the track at another bin phase ------------------------------------------------------------------
def bin_stats(phi, r, width: float, offset: float, lo: float, nb: int):
    """count, median and 16-84 half-width of r in nb bins of `width` starting at lo + offset (generate.py `bin_track`)."""
    idx = np.floor((phi - lo - offset) / width).astype(np.int64)
    m = (idx >= 0) & (idx < nb)
    idx, r = idx[m], r[m]
    o = np.lexsort((r, idx))
    rs, count = r[o], np.bincount(idx, minlength=nb)
    start = np.cumsum(count) - count

    def quantile(f):
        p = start + f * (count - 1)
        l = np.clip(np.floor(p).astype(np.int64), 0, max(len(rs) - 1, 0))
        h = np.clip(np.minimum(l + 1, start + count - 1), 0, max(len(rs) - 1, 0))
        return np.where(count > 0, rs[l] * (1 - (p - np.floor(p))) + rs[h] * (p - np.floor(p)), np.nan) if len(rs) else np.full(nb, np.nan)

    return count, quantile(0.5), 0.5 * (quantile(0.84) - quantile(0.16))


PHASE_METHODS = ("direct", "linear", "cubic", "spline", "spline per arm", "direct0", "spline0", "spline per arm0")


def _phase(task):
    """Particles of every stream split at random in two halves. From the standard bins of half A the track of
    half B is predicted in bins shifted by half a bin (and, with the suffix 0, in the same bins); `direct` is
    half A binned at the phase of the target. Errors in units of the particle noise of the target bin."""
    path, k0, k1 = task
    with h5py.File(path, "r") as f:
        index, phi_all, xy = f["index"][k0:k1], f["phi"][k0:k1], f["xy"][k0:k1]
    nb, lo, out = 2 * CAP, -CAP * 10.0, []
    for index_k, phi, r in zip(index, phi_all, np.hypot(xy[..., 0], xy[..., 1])):
        rng = np.random.default_rng([2614, int(index_k)])
        fin = np.isfinite(phi) & np.isfinite(r)
        phi, r = phi[fin].astype(np.float64), r[fin].astype(np.float64)
        half = rng.random(len(phi)) < 0.5
        A, Bh = (phi[half], r[half]), (phi[~half], r[~half])
        nA, mA, hA = bin_stats(*A, 10.0, 0.0, lo, nb)
        nB, mB, _ = bin_stats(*Bh, 10.0, 0.0, lo, nb)
        nAs, mAs, _ = bin_stats(*A, 10.0, 5.0, lo, nb)
        nBs, mBs, _ = bin_stats(*Bh, 10.0, 5.0, lo, nb)
        _, _, h0 = bin_stats(phi, r, 10.0, 0.0, lo, nb)  # widths from all particles, for the errors
        _, _, hs = bin_stats(phi, r, 10.0, 5.0, lo, nb)
        valid = (nA >= 10) & (nB >= 10) & (hA > 0)
        a, b = run_of(valid[None], CAP)
        a, b = int(a[0]), int(b[0])
        if a + b < 8:
            continue
        sl = slice(CAP - a, CAP + b)
        xc = (np.arange(sl.start, sl.stop) - CAP + 0.5) * 10.0
        xs, tg = xc[:-1] + 5.0, slice(sl.start, sl.stop - 1)  # shifted bins with both neighbours in the run
        sgA = particle_sigma(hA[sl], nA[sl])
        s0, _, ss = arm_fit(xc, mA[sl], sgA, xe=xs)
        a0, _, as_ = arm_fit(xc, mA[sl], sgA, split=True, xe=np.where(xs == 0, np.nan, xs))  # no prediction for the bin across the progenitor
        pred = {"direct": np.where(nAs[tg] >= 10, mAs[tg], np.nan), "linear": 0.5 * (mA[sl][:-1] + mA[sl][1:]), "cubic": CubicSpline(xc, mA[sl])(xs),
                "spline": ss, "spline per arm": as_}  # fmt: skip
        sgs, sg0 = particle_sigma(hs[tg], nBs[tg]), particle_sigma(h0[sl], nB[sl])
        e = {k: np.where(nBs[tg] >= 10, (v - mBs[tg]) / sgs, np.nan) for k, v in pred.items()}
        e |= {"direct0": (mA[sl] - mB[sl]) / sg0, "spline0": (s0 - mB[sl]) / sg0, "spline per arm0": (a0 - mB[sl]) / sg0}
        row = np.full((len(xc), 4 + len(PHASE_METHODS)), np.nan)
        row[:, 0], row[:, 1] = index_k, xc
        row[:-1, 2], row[:, 3] = sgs / mBs[tg], sg0 / mB[sl]  # the particle noise of the target over its radius
        for j, k in enumerate(PHASE_METHODS):
            row[: len(e[k]), 4 + j] = e[k]
        out.append(row)
    return np.concatenate(out) if out else np.zeros((0, 4 + len(PHASE_METHODS)))


def cmd_phase(a) -> None:
    d = read_shards(PILOT)
    d["a"], d["b"] = run_of(ok_bins(d), d["i0"])
    tasks = [(p, k, k + 50) for p in sorted(PILOT.glob("particles_*.h5")) for k in range(0, 1000, 50)]
    with Pool(a.workers) as pool:
        tab = np.concatenate(pool.map(_phase, tasks, chunksize=1))
    np.savez(RES / "phase.npz", table=tab, columns=["stream", "phi_deg", "rel_sigma_shifted", "rel_sigma", *PHASE_METHODS], cls=stream_class(d, d["a"], d["b"]),
             usable=d["usable"], index=d["index"], dissolved=d["dissolved"])  # fmt: skip
    print("phase:", len(np.unique(tab[:, 0])), "streams,", len(tab), "bins")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["observed", "mocks", "premise", "smoothing", "phase", "figures", "all"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n-mock", type=int, default=200, help="mock tracks per observed track and pool")
    a = ap.parse_args()
    (RES / "numbers").mkdir(parents=True, exist_ok=True)
    for c in ("observed", "mocks", "premise", "smoothing", "phase", "figures"):
        if a.command in (c, "all"):
            if c == "figures":
                import spline_track_figures

                spline_track_figures.main()
            else:
                globals()[f"cmd_{c}"](a)


if __name__ == "__main__":
    main()
