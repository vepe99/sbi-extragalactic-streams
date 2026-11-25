"""The simulation set, the prior, and the observed tracks.

PARAMETERS. The 14 parameters of Chemaly et al. 2026b (arXiv:2607.05510, table A1) in the order of StreaMAX-fit,
plus the position of the progenitor along the observed track:

    logM Rs dirx diry dirz logm rs x0 z0 vx0 vy0 vz0 time sig | prog

13 come from the shards: `theta` holds (q, cos_i, psi) and `c26_vec` the equivalent orientation vector
(dirx, diry, dirz) = r(q) (-n_x, -n_y, n_z) that the paper samples; the vector is used because psi is periodic.
`sig` (sigma_sys) and `prog` are drawn in the mock observation. `prog` is the fraction of the way along the
track at which the progenitor sits (0 = first bin edge, 1 = last), uniform a priori.

LATENT COORDINATES. Networks work on zeta_k = Phi^-1(F_k(parameter_k)), F_k the prior cdf. The prior is then
exactly N(0, I), and F_k(parameter_k) is the unit cube dynesty samples in the paper, so importance weights need
no Jacobian.

FRAME. The prior keeps one copy of every set of equivalent systems (y0 = 0 and x0 > 0, vy0 > 0, z0 > 0,
dirz >= 0), so there is no rotation or mirror augmentation: a transformed example would be identical or
outside the prior.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import h5py
import numpy as np
from scipy import special

from config import REPO, Config

PARAM_NAMES = ["logM", "Rs", "dirx", "diry", "dirz", "logm", "rs", "x0", "z0", "vx0", "vy0", "vz0", "time", "sig", "prog"]
N_PAR, I_X0, I_SIG, I_PROG = len(PARAM_NAMES), 7, 13, 14
KMS_TO_KPCGYR = 1.0227121650537077


@dataclass(frozen=True)
class Prior:
    """Per parameter: "U" uniform(a, b), "N" normal(0, a), "H" half-normal(0, a) > 0. Table A1 is read from the
    configuration stored with the simulations, never retyped."""

    kind: tuple
    a: tuple
    b: tuple

    @classmethod
    def from_sims(cls, sc: dict, sig_sys_max: float) -> "Prior":
        U, N, H = (lambda ab: ("U", *map(float, ab))), (lambda s: ("N", float(s), 0.0)), (lambda s: ("H", float(s), 0.0))
        rows = [U(sc["log10_M_halo"]), U(sc["R_s"]), N(1), N(1), H(1), U(sc["log10_m_prog"]), U(sc["r_s"]),
                H(sc["sigma_pos"]), H(sc["sigma_pos"]), N(sc["sigma_vel"]), H(sc["sigma_vel"]), N(sc["sigma_vel"]),
                U(sc["t_int"]), U((0.0, sig_sys_max)), U((0.0, 1.0))]  # fmt: skip
        return cls(*zip(*rows))

    def _u(self, x, k):  # prior cdf
        return (x - self.a[k]) / (self.b[k] - self.a[k]) if self.kind[k] == "U" else special.erf(x / (self.a[k] * math.sqrt(2)))

    def to_latent(self, x, cols=None, eps=1e-7) -> np.ndarray:
        x = np.asarray(x, np.float64)
        f = lambda j, k: x[..., j] / self.a[k] if self.kind[k] == "N" else special.ndtri(np.clip(self._u(x[..., j], k), eps, 1 - eps))
        return np.stack([f(j, k) for j, k in enumerate(range(N_PAR) if cols is None else cols)], axis=-1)

    def from_latent(self, z, cols=None) -> np.ndarray:
        z = np.asarray(z, np.float64)

        def f(j, k):
            kind, a, b = self.kind[k], self.a[k], self.b[k]
            if kind == "N":
                return z[..., j] * a
            u = special.ndtr(z[..., j])
            return a + (b - a) * u if kind == "U" else a * special.ndtri(0.5 * (1 + u))  # H: F(x) = 2 Phi(x/a) - 1

        return np.stack([f(j, k) for j, k in enumerate(range(N_PAR) if cols is None else cols)], axis=-1)


def orientation(vec, q_lim=(0.5, 1.5)) -> np.ndarray:
    """(dirx, diry, dirz) -> (q, cos_i, psi): the length maps to q through the Maxwell cdf (StreaMAX-fit `get_q`),
    the direction is the halo axis with its sky-plane components mirrored (`c26_vector` in generate.py)."""
    v = np.asarray(vec, np.float64)
    r = np.linalg.norm(v, axis=-1)
    q = q_lim[0] + (q_lim[1] - q_lim[0]) * (special.erf(r / math.sqrt(2)) - math.sqrt(2 / math.pi) * r * np.exp(-0.5 * r**2))
    n = v / r[..., None] * np.array([-1.0, -1.0, 1.0])
    return np.stack([q, n[..., 2], np.mod(np.arctan2(n[..., 1], n[..., 0]), 2 * np.pi)], axis=-1)


G_KPC_GYR_MSUN = 4.498502151469553e-06  # kpc^3 / (Msun Gyr^2), the constant of generate.py and StreaMAX
KPC_GYR2_TO_KMS_MYR = 1.0 / (KMS_TO_KPCGYR * 1000.0)  # acceleration: kpc / Gyr^2 -> km / s / Myr
_GL_X, _GL_W = np.polynomial.legendre.leggauss(12)
_GL_V, _GL_W = 0.5 * (_GL_X + 1.0), 0.5 * _GL_W  # the 12 nodes of StreaMAX's flattened NFW


def host_acceleration(params, xyz, flatten: str = "density") -> np.ndarray:
    """Acceleration -grad Phi [kpc / Gyr^2] of the host halo of `params` (..., >= 5: logM, Rs, dirx, diry, dirz)
    at the points `xyz` (..., B, 3) [kpc], in the frame of the simulation (the sky is the XY plane, z the line
    of sight). The numpy copy of `_nfw_grad` in generate.py, which cannot be imported in this environment:
    the NFW flattened by q in the density (or in the potential: `halo_flattening` of the simulation set) along
    the axis n = (-dirx, -diry, dirz) / |dir|."""
    p, xyz = np.asarray(params, np.float64), np.asarray(xyz, np.float64)
    M, Rs, q = 10.0 ** p[..., 0, None, None], p[..., 1, None, None], orientation(p[..., 2:5])[..., 0, None, None]
    n = (p[..., 2:5] / np.linalg.norm(p[..., 2:5], axis=-1, keepdims=True) * np.array([-1.0, -1.0, 1.0]))[..., None, :]
    eps = 1e-12
    Z = np.sum(xyz * n, axis=-1, keepdims=True)
    perp = xyz - Z * n
    R2 = np.sum(perp * perp, axis=-1, keepdims=True)
    if flatten == "potential":
        r = np.sqrt(R2 + (Z / q) ** 2 + eps)
        return -G_KPC_GYR_MSUN * M * (np.log(1.0 + r / Rs) / r**2 - 1.0 / (r * (Rs + r))) * (perp + Z * n / q**2) / r
    q2, R = q * q, np.sqrt(R2 + eps)
    xi, zeta = R / Rs, Z / Rs
    D = q2 + (1.0 - q2) * _GL_V**2
    m = _GL_V * np.sqrt(xi**2 / D + zeta**2 / q2 + eps)
    base = _GL_V**2 / ((1.0 + m) ** 2 * m + eps)
    dI_dxi = -xi * np.sum(_GL_W * base / D**2, axis=-1, keepdims=True)
    dI_dzeta = -(zeta / q2) * np.sum(_GL_W * base / D, axis=-1, keepdims=True)
    pref = -(G_KPC_GYR_MSUN * M * q2 / Rs) / Rs
    return -(pref * dI_dxi * perp / R + pref * dI_dzeta * n)


def to_simulator(params) -> np.ndarray:
    """The 14 parameters -> the 13 columns of `theta` generate.py simulates (q, cos_i, psi instead of the vector)."""
    p = np.asarray(params, np.float64)
    return np.concatenate([p[..., :2], orientation(p[..., 2:5]), p[..., 5:13]], axis=-1)


@dataclass
class Grid:
    n_bins: int
    bin_deg: float
    i0: int  # the progenitor sits on the left edge of bin i0

    @classmethod
    def from_sims(cls, sc: dict) -> "Grid":
        return cls(int(round(2 * sc["phi_max_deg"] / sc["bin_deg"])), float(sc["bin_deg"]), int(round(sc["phi_max_deg"] / sc["bin_deg"])))


def run_extents(valid: np.ndarray, i0: int):
    """Lengths (a, b) of the unbroken run of valid bins to the left / right of the progenitor."""
    left, right = valid[:, :i0][:, ::-1], valid[:, i0:]
    return (np.where(left.all(1), left.shape[1], np.argmin(left, 1)), np.where(right.all(1), right.shape[1], np.argmin(right, 1)))


def window_design(cfg: Config, count, r_hw, i0, n_min=None) -> dict:
    """The window prior p0(S) and the acceptance probability Z of every simulation.

    A mock track is a window of S bins with l of them left of the progenitor, S ~ p0(S), l ~ U{0..S}. A stream
    whose comparable bins extend a to the left and b to the right of its progenitor covers it if l <= a and
    S - l <= b. Drawing every stream once per epoch and then a window it covers would favour streams barely
    longer than the data; instead streams are drawn in proportion to Z_i = P0(a random window is covered),
    which is rejection sampling of p0 and leaves  prior x p0 x 1[model covers the data]  (the paper's guards).
    Comparable bins depend on the data errors when the model-noise guard is on, so Z is computed per stratum of
    the error level (flat errors at the stratum's mid level) and the loss weight in mock.py corrects the rest.
    p0(S) = span_prior(S) / coverage(S), so the accepted track lengths follow span_prior.
    n_min: particles a bin needs, per stream (`replace_tracks`); None = cfg.n_min for all.
    """
    S = np.arange(cfg.span_min, cfg.span_max + 1)
    K = cfg.err_strata if cfg.var_ratio > 0 else 1
    edges = np.logspace(*np.log10(cfg.err_level), K + 1)

    def frac(valid):  # share of the S + 1 progenitor positions each stream covers, for every S
        a, b = run_extents(valid, i0)
        n = np.minimum(S[None], a[:, None]) - np.maximum(0, S[None] - b[:, None]) + 1
        return (np.clip(n, 0, None) / (S[None] + 1.0)).astype(np.float32)

    base = count >= (cfg.n_min if n_min is None else np.asarray(n_min)[:, None])
    noise = r_hw**2 / np.maximum(count, 1.0) * cfg.var_ratio
    fr = [frac(base & (noise <= lv**2)) for lv in np.sqrt(edges[:-1] * edges[1:])] if cfg.var_ratio > 0 else [frac(base)]
    cover = np.mean([f.mean(0, dtype=np.float64) for f in fr], axis=0)
    pS = (1.0 / S if cfg.span_prior == "loguniform" else np.ones(len(S))) / np.maximum(cover, 1e-4)
    pS = pS / pS.sum()
    # Z is floored at 5 % of its guard-free value: the scatter of the errors along a track can make bins
    # comparable that the flat level does not, and those draws must stay possible
    Z = np.maximum(np.stack([f.astype(np.float64) @ pS for f in fr], 1), 0.05 * (frac(base).astype(np.float64) @ pS)[:, None])
    return dict(S=S, pS=pS, cover=cover, Z=Z.astype(np.float32))


def read_resimulated(d, ordering: str) -> dict:
    """A set made by scripts/extragalactic/long90k.py (streams of another set simulated again with another particle
    number) -> its tracks on the whole grid, stream k being row k of its selection.npz."""
    files = sorted((d / "data").glob("shard_*.h5"))
    if not files:
        raise SystemExit(f"no shards in {d / 'data'}")
    with h5py.File(files[0], "r") as f:
        sc = json.loads(f.attrs["config"])
        group = {f[g].attrs["ordering"]: g for g in ("track", "track_alt") if g in f}[ordering]
        dtype = str(f[f"{group}/count"].dtype)
    cols = dict(index="index", theta="theta", status="status", n_bad="diag/n_bad", dE="diag/dE_rel", dissolved="prog/dissolved",
                count=f"{group}/count", r_med=f"{group}/r_med", r_hw=f"{group}/r_hw")  # fmt: skip
    raw = {k: [] for k in cols}
    for p in files:
        with h5py.File(p, "r") as f:
            for k, path in cols.items():
                raw[k].append(f[path][:])
    raw = {k: np.concatenate(v) for k, v in raw.items()}
    order = np.argsort(raw.pop("index"))
    raw = {k: v[order] for k, v in raw.items()}
    return raw | dict(r_med=np.nan_to_num(raw["r_med"]).astype(np.float32), r_hw=np.nan_to_num(raw["r_hw"]).astype(np.float32),
                      n_particles=int(sc["n_particles"]), sim_config=sc, count_dtype=dtype)  # fmt: skip


SAME_PHYSICS = ("n_steps", "integrator", "halo_flattening", "progenitor_mass_loss", "bin_deg", "phi_max_deg", "track_ordering",
                "log10_M_halo", "R_s", "q", "log10_m_prog", "r_s", "sigma_pos", "sigma_vel", "t_int")  # fmt: skip


def replacement_rule(cfg: Config) -> dict:
    """The rule by which the streams of `replace_tracks` were chosen (scripts/extragalactic/long90k.py: n_min,
    span_max, span_min, long_bins, de_max, on the tracks of `subdir`), with the particle number they were
    simulated again with (`n_particles`)."""
    d = REPO / cfg.out / "data" / cfg.replace_tracks
    rule = json.loads(str(np.load(d / "selection.npz", allow_pickle=True)["rule"]))
    with h5py.File(sorted((d / "data").glob("shard_*.h5"))[0], "r") as f:
        return rule | dict(n_particles=int(json.loads(f.attrs["config"])["n_particles"]))


def chosen_for_replacement(cfg: Config, rule: dict, sim: dict, i0: int, n_particles: int) -> np.ndarray:
    """Which streams the rule of `replace_tracks` picks, from their tracks at the particle number of `subdir`
    (sim: count on the whole grid, status, n_bad, dE): streams the loader keeps, with a resolved orbit and an
    unbroken run of at least `long_bins` bins with enough particles around the progenitor, within span_max
    bins on each side. The same test as `select` of long90k.py, for streams simulated anew (predictive.py)."""
    status, T = np.asarray(sim["status"]).astype(int), rule["span_max"]
    a, b = run_extents(np.asarray(sim["count"]) >= rule["n_min"], i0)
    kept = ((status & 16) == 0) & (sim["n_bad"] <= 0.01 * n_particles) & (a + b >= rule["span_min"]) & (cfg.include_unbound | ((status & 1) == 0))
    if cfg.de_max > 0:
        kept &= sim["dE"] <= cfg.de_max
    return kept & (sim["dE"] <= rule["de_max"]) & (np.minimum(a, T) + np.minimum(b, T) >= rule["long_bins"])


def resimulated_ok(cfg: Config, rule: dict, new: dict, i0: int, n_particles: int, n_min: float) -> np.ndarray:
    """Which of the streams simulated again pass the loader's cuts with their new track, and so use it: the
    others keep their first track (see `replacement`)."""
    a, b = run_extents(np.asarray(new["count"]) >= n_min, i0)
    ok = ((np.asarray(new["status"]).astype(int) & 16) == 0) & (new["n_bad"] <= 0.01 * n_particles) & (a + b >= cfg.span_min)
    for de_max in (rule.get("de_max", 0.0), cfg.de_max):
        if de_max > 0:
            ok &= new["dE"] <= de_max
    return ok


def replacement(cfg: Config, sc: dict, i0: int, theta_raw, keep) -> dict:
    """The tracks of `replace_tracks` that take the place of stored ones: `row` (index in the raw set) and the
    tracks on the whole grid. Left out, so that they keep their stored track: streams the loader does not keep,
    and streams whose new simulation fails the loader's cuts (failed integration, more than 1 % of the particles
    lost, fewer than span_min bins with enough particles), and streams whose new progenitor orbit has a relative
    energy error above the `de_max` the streams were selected with, or above the loader's own:
    a poorly resolved orbit differs between two runs with the same parameters, and 0.9 % of train_v2_long90k,
    all selected with dE_rel <= 0.01 in train_v2, came out above it."""
    d = REPO / cfg.out / "data" / cfg.replace_tracks
    sel, new = np.load(d / "selection.npz", allow_pickle=True), read_resimulated(d, cfg.ordering)
    row, N = sel["row"], new["n_particles"]
    if json.loads(str(sel["rule"]))["subdir"] != cfg.subdir or len(row) != len(new["count"]):
        raise SystemExit(f"{d}: made from {json.loads(str(sel['rule']))['subdir']}, not {cfg.subdir}, or not complete ({len(new['count'])} of {len(row)} streams)")
    diff = {k: (sc[k], new["sim_config"][k]) for k in SAME_PHYSICS if sc[k] != new["sim_config"][k]}
    if diff:
        raise SystemExit(f"{d} was simulated with other physics than {cfg.subdir}: {diff}")
    n_min = cfg.n_min * N / sc["n_particles"]  # guard 1 is a fraction of the particles
    inside = row < len(keep)  # with max_shards only the first rows are loaded
    if not np.array_equal(new["theta"][inside], theta_raw[row[inside]]):
        raise SystemExit(f"{d}: the parameters are not those of the same rows of {cfg.subdir}")
    ok = inside & resimulated_ok(cfg, json.loads(str(sel["rule"])), new, i0, N, n_min)
    ok[inside] &= keep[row[inside]]
    return dict(row=row[ok], n_min=n_min, n_particles=N, n_selected=int(inside.sum()), **{k: new[k][ok] for k in ("count", "r_med", "r_hw")})


def load_simulations(cfg: Config) -> dict:
    """Shards -> arrays for the streams a mock observation can be drawn from. Dropped: failed integrations
    (status bit 16), streams that lost more than 1 % of their particles, streams with fewer than `span_min`
    bins of >= n_min particles around the progenitor and, with `de_max` > 0, streams whose progenitor orbit has a
    relative energy error above it. With `replace_tracks` the tracks of that set then take the place of the stored
    ones (`replacement`); which streams are kept, and their order, is decided by the stored tracks alone."""
    files = sorted(cfg.data_dir().glob("shard_*.h5"))[: cfg.max_shards or None]
    if not files:
        raise SystemExit(f"no shards in {cfg.data_dir()}")
    with h5py.File(files[0], "r") as f:
        sc = json.loads(f.attrs["config"])
        group = {f[g].attrs["ordering"]: g for g in ("track", "track_alt") if g in f}[cfg.ordering]
    grid = Grid.from_sims(sc)
    cols = dict(theta="theta", vec="c26_vec", status="status", n_bad="diag/n_bad", dE="diag/dE_rel", dissolved="prog/dissolved",
                count=f"{group}/count", r_med=f"{group}/r_med", r_hw=f"{group}/r_hw")  # fmt: skip
    raw = {k: [] for k in cols}
    for p in files:
        with h5py.File(p, "r") as f:
            for k, path in cols.items():
                raw[k].append(f[path][:])
    raw = {k: np.concatenate(v) for k, v in raw.items()}
    count = raw["count"].astype(np.float32)
    r_med, r_hw = (np.nan_to_num(raw[k]).astype(np.float32) for k in ("r_med", "r_hw"))
    failed, lossy, unbound = (raw["status"] & 16) > 0, raw["n_bad"] > 0.01 * sc["n_particles"], (raw["status"] & 1) > 0
    a, b = run_extents(count >= cfg.n_min, grid.i0)
    unresolved = (raw["dE"] > cfg.de_max) if cfg.de_max > 0 else np.zeros(len(failed), bool)
    keep = ~failed & ~lossy & (a + b >= cfg.span_min) & (cfg.include_unbound | ~unbound) & ~unresolved
    n_min, replaced = np.full(len(keep), cfg.n_min, np.float32), np.zeros(len(keep), bool)  # particles a bin needs, per stream
    if cfg.replace_tracks:
        rep = replacement(cfg, sc, grid.i0, raw["theta"], keep)
        count[rep["row"]], r_med[rep["row"]], r_hw[rep["row"]] = rep["count"], rep["r_med"], rep["r_hw"]
        n_min[rep["row"]], replaced[rep["row"]] = rep["n_min"], True

    prior = Prior.from_sims(sc, cfg.sig_sys_max)
    th = raw["theta"].astype(np.float64)
    phys13 = np.concatenate([th[:, :2], raw["vec"], th[:, 5:]], axis=1)
    design = window_design(cfg, count[keep], r_hw[keep], grid.i0, n_min[keep])
    data = dict(zeta13=prior.to_latent(phys13[keep], cols=range(13)).astype(np.float32), count=count[keep], r_med=r_med[keep],
                r_hw=r_hw[keep], dissolved=raw["dissolved"][keep].astype(np.float32), z_strata=design["Z"], n_min=n_min[keep])  # fmt: skip
    if cfg.scale_aug:  # the scale augmentation works on the parameters themselves (mock.py)
        data["phys13"] = phys13[keep].astype(np.float32)
    stats = dict(n_streams=len(keep), n_shards=len(files), n_usable=int(keep.sum()), failed=float(failed.mean()),
                 lossy=float(lossy.mean()), unbound=float(unbound.mean()), unresolved_dropped=float(unresolved.mean()), n_particles=int(sc["n_particles"]))  # fmt: skip
    if cfg.replace_tracks:
        stats |= dict(replaced=int(replaced.sum()), replaced_of=rep["n_selected"], replaced_n_particles=rep["n_particles"], replaced_n_min=float(rep["n_min"]))
    return dict(data=data, phys13=phys13[keep], phys13_all=phys13, theta=th[keep], prior=prior, grid=grid, design=design, stats=stats, sim_config=sc,
                dE=raw["dE"][keep], replaced=replaced[keep])  # fmt: skip


def split(sims: dict, val_frac: float) -> tuple[dict, dict]:
    """(training streams, held-out streams): the held-out ones are the last `val_frac` in stream order."""
    n = len(sims["phys13"])
    cut = n - max(1, int(round(val_frac * n)))
    return ({k: v[:cut] for k, v in sims["data"].items()}, {k: v[cut:] for k, v in sims["data"].items()})


def load_observed(cfg: Config) -> list[dict]:
    """The 32 STRRINGS tracks released with StreaMAX-fit. `r_scale` is the paper's rescaling of the radii
    (100 for PGC938075, whose radii are below 1 kpc; 1 otherwise) and is applied here as it is in its fits."""
    out = []
    for p in sorted((REPO / cfg.obs_dir).glob("*.npz")):
        d = np.load(p)
        s = float(d["r_scale"]) if "r_scale" in d.files and np.isfinite(d["r_scale"]) else 1.0
        out.append(dict(name=p.stem, theta_deg=np.rad2deg(d["theta"]), r=d["r"] * s, r_err=d["r_err"] * s, r_scale=s))
    return out
