"""The long streams of a simulation set, simulated again with more particles: a set of its own that the loader
can put in place of the stored tracks (`replace_tracks` in sbi/config.py).

    uv run --extra gpu python scripts/extragalactic/long90k.py --config outputs/extragalactic/C26_test/npe_v2_mlp.yaml            # selection and pilot
    uv run --extra gpu python scripts/extragalactic/long90k.py --config outputs/extragalactic/C26_test/npe_v2_mlp.yaml --full     # every long stream
    uv run --extra gpu python scripts/extragalactic/long90k.py --config outputs/extragalactic/C26_test/npe_v2_mlp.yaml --check    # the checks

WHICH STREAMS. The rule of `select()` in particle_number.py, applied to every usable stream of `subdir` (the
streams the loader keeps): a resolved progenitor orbit (`diag/dE_rel` <= `--de-max`; more particles do not repair
an unresolved one) and an unbroken run of at least `--long-bins` bins with >= n_min particles around the
progenitor, counted within +-span_max bins. These are the streams a long mock track can be cut from.

HOW. generate.py on the stored parameters and dissolved flag (`--theta-file`), the physics of the set, a new seed
(`--seed`; stream k of the new set uses fold_in(key(seed), k)), `--n-particles` particles, no saved particles.
A stopped run is taken up again by the same command: finished shards stay.

THE SELECTION IS MADE ON THE STORED TRACKS, so the particle number of a stream becomes a function of its
parameters and of its first realisation. Importance or nested sampling against a set with these tracks in place
must simulate a long stream with `--n-particles` particles (guard 1 at n_min x N / N_set) to match.

Written to <out>/data/<subdir>_long<N/1000>k/ (nothing in <out>/data/<subdir> is touched):
    selection.npz     row (index of each stream in the raw set, increasing), theta, dissolved, the rule
    check_sample.npz  the stored tracks of a random sample of them, for the checks
    theta.npz, data/shard_*.h5, data/manifest.json, generate.log    as generate.py leaves them; stream k is row[k]
    pilot/            the pilot: a random sample, simulated first to measure the rate
    checks.json       --check
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts" / "extragalactic"), str(REPO / "scripts" / "extragalactic" / "sbi")]
import track_noise as TN  # noqa: E402
from config import Config, free_gpus  # noqa: E402
from data import read_resimulated, run_extents  # noqa: E402
from prior_weights import lean_load  # noqa: E402


def select(cfg: Config, a, out: Path) -> dict:
    """selection.npz and check_sample.npz, made once (the set is loaded only then)."""
    rule = dict(subdir=cfg.subdir, ordering=cfg.ordering, n_min=cfg.n_min, span_max=cfg.span_max, span_min=cfg.span_min, long_bins=a.long_bins, de_max=a.de_max)
    if (out / "selection.npz").exists():
        sel = dict(np.load(out / "selection.npz", allow_pickle=True))
        if json.loads(str(sel["rule"])) != rule:
            raise SystemExit(f"{out}/selection.npz was made with another rule: {sel['rule']} (now {rule})")
        return sel
    sims = lean_load(dataclasses.replace(cfg, replace_tracks="", de_max=0.0), a.workers)
    d, T = sims["data"], cfg.span_max
    la, lb = run_extents(d["count"] >= cfg.n_min, T)
    long = (sims["dE"] <= a.de_max) & (la + lb >= a.long_bins)
    k = np.flatnonzero(long)
    sel = dict(row=sims["row"][k], theta=sims["theta"][k], dissolved=d["dissolved"][k] > 0.5, dE=sims["dE"][k], run=(la + lb)[k], n_usable=len(long),
               n_raw=len(sims["dissolved_all"]), n_particles_set=sims["sim_config"]["n_particles"], sim_config=json.dumps(sims["sim_config"]), rule=json.dumps(rule))  # fmt: skip
    np.savez(out / "selection.npz", **sel)
    s = np.sort(np.random.default_rng(cfg.seed).choice(len(k), min(a.n_check, len(k)), replace=False))
    np.savez(out / "check_sample.npz", pos=s, count=d["count"][k[s]], r_med=d["r_med"][k[s]], r_hw=d["r_hw"][k[s]])
    return sel


def rate_of(out: Path) -> dict:
    """Streams per second of a generate.py run, from its worker logs: per worker the mean over its shards."""
    per = {}
    for p in sorted((out / "data" / "logs").glob("worker_*.log")):
        ev = [json.loads(ln) for ln in p.read_text().splitlines() if ln.startswith('{"event": "shard"')]
        if ev:
            per[p.stem] = dict(shards=len(ev), streams=sum(e["n"] for e in ev), seconds=sum(e["wall_s"] for e in ev), batch=ev[-1]["batch"])
    return dict(workers=len(per), rate=sum(v["streams"] / v["seconds"] for v in per.values()), per_worker=per)


def usable(count, r_hw, n_min: float, var_ratio: float, T: int, sig: float) -> np.ndarray:
    """Bins of the unbroken run around the progenitor that pass both guards at a flat data error `sig` [kpc]."""
    ok = (count >= n_min) & (r_hw.astype(np.float64) ** 2 / np.maximum(count, 1) * var_ratio <= sig**2)
    la, lb = run_extents(ok, T)
    return la + lb


def check(cfg: Config, a, out: Path, sel: dict) -> None:
    sc, T = json.loads(str(sel["sim_config"])), cfg.span_max
    i0 = int(round(sc["phi_max_deg"] / sc["bin_deg"]))
    new = read_resimulated(out, cfg.ordering)
    N, N0 = new["n_particles"], int(sel["n_particles_set"])
    n_min = cfg.n_min * N / N0
    assert len(new["count"]) == len(sel["row"]), f"{len(new['count'])} streams on disk, {len(sel['row'])} selected"
    num = dict(n_streams=int(len(sel["row"])), n_particles=N, count_dtype=new["count_dtype"], n_min=n_min)
    num["same_parameters_and_dissolved_flag"] = bool(np.array_equal(new["theta"], sel["theta"].astype(np.float32)) and np.array_equal(new["dissolved"].astype(bool), sel["dissolved"]))
    num["negative_counts"] = int((new["count"] < 0).sum())
    num["largest_count"] = int(new["count"].max())
    num["counts_sum_at_most_n_particles"] = bool((new["count"].sum(1, dtype=np.int64) <= N).all())
    failed, lossy = (new["status"] & 16) > 0, new["n_bad"] > 0.01 * N
    la, lb = run_extents(new["count"] >= n_min, i0)
    num["failed"] = dict(new=float(failed.mean()), stored=0.0, note="the stored streams are usable ones: none failed")
    num["lossy"] = dict(new=float(lossy.mean()), stored=0.0)
    num["run_below_span_min"] = float((la + lb < cfg.span_min).mean())
    num["not_replaceable"] = int((failed | lossy | (la + lb < cfg.span_min) | (new["dE"] > a.de_max)).sum())  # as data.replacement
    dE0, dE1 = sel["dE"].astype(np.float64), new["dE"].astype(np.float64)
    ratio = dE1 / np.maximum(dE0, 1e-30)
    num["dE_rel"] = dict(median_stored=float(np.median(dE0)), median_new=float(np.median(dE1)), ratio_median=float(np.median(ratio)),
                         ratio_p01_p99=[float(x) for x in np.percentile(ratio, [1, 99])], identical=float((dE0.astype(np.float32) == dE1.astype(np.float32)).mean()),
                         new_above_de_max=float((dE1 > a.de_max).mean()))  # fmt: skip
    cs = np.load(out / "check_sample.npz")
    s, win = cs["pos"], slice(i0 - T, i0 + T)
    num["usable_length"] = dict(n=int(len(s)), reference="plots/<subdir>/particle_number/particle_number.json, usable_length")
    for sig in (0.3, 1.0, 3.0):
        row = {}
        for key, cnt, hw, nm in (("stored", cs["count"].astype(np.float32), cs["r_hw"], float(cfg.n_min)), ("new", new["count"][s][:, win].astype(np.float32), new["r_hw"][s][:, win], n_min)):
            L = usable(cnt, hw, nm, cfg.var_ratio, T, sig)
            row[key] = dict(mean_bins=float(L.mean()), at_least_26=float((L >= 26).mean()), at_least_36=float((L >= 36).mean()))
        num["usable_length"][f"sigma_{sig:g}"] = row
    L0, L1 = (usable(c, cs["r_hw"] if k == 0 else new["r_hw"][s][:, win], nm, 0.0, T, 1.0) for k, (c, nm) in
              enumerate(((cs["count"].astype(np.float32), float(cfg.n_min)), (new["count"][s][:, win].astype(np.float32), n_min))))  # fmt: skip
    num["run_guard1_only"] = dict(stored_mean=float(L0.mean()), new_mean=float(L1.mean()), new_at_least_long_bins=float((L1 >= a.long_bins).mean()))
    num["loader"] = check_loader(cfg, a, out)
    (out / "checks.json").write_text(json.dumps(num, indent=1))
    print(json.dumps(num, indent=1))


def check_loader(cfg: Config, a, out: Path) -> dict:
    """The loader with `replace_tracks` on against the loader without: the same streams in the same order, the same
    arrays for the streams that were not replaced, the new tracks and n_min for the others. With lean_load on the
    whole set, and with load_simulations on its first `--loader-shards` shards."""
    from data import load_simulations

    off, on = dataclasses.replace(cfg, replace_tracks="", de_max=0.0), dataclasses.replace(cfg, replace_tracks=out.name, de_max=0.0)
    keys = ("zeta13", "count", "r_med", "r_hw", "dissolved")

    def compare(A, B) -> dict:
        r, dA, dB = B["replaced"], A["data"], B["data"]
        return dict(n_usable=int(len(r)), replaced=int(r.sum()), same_streams_and_order=bool(np.array_equal(A["theta"], B["theta"])),
                    unreplaced_identical={k: bool(np.array_equal(dA[k][~r], dB[k][~r])) for k in keys},
                    replaced_parameters_identical=bool(np.array_equal(dA["zeta13"][r], dB["zeta13"][r]) and np.array_equal(dA["dissolved"][r], dB["dissolved"][r])),
                    replaced_tracks_differ=float((dA["count"][r] != dB["count"][r]).any(1).mean()),
                    n_min_unreplaced=sorted(set(np.unique(dB["n_min"][~r]).tolist())), n_min_replaced=sorted(set(np.unique(dB["n_min"][r]).tolist())),
                    z_strata_unreplaced_max_rel_change=float(np.abs(dB["z_strata"][~r] / dA["z_strata"][~r] - 1).max()),
                    z_all_positive=bool((dB["z_strata"] > 0).all()),
                    window_prior_p0_max_rel_change=float(np.abs(B["design"]["pS"] / A["design"]["pS"] - 1).max()),
                    coverage_of_longest_window=dict(stored=float(A["design"]["cover"][-1]), replaced=float(B["design"]["cover"][-1])))  # fmt: skip

    A, B = lean_load(off, a.workers), lean_load(on, a.workers)
    num = dict(lean_load=compare(A, B))
    del A
    s0, s1 = (load_simulations(dataclasses.replace(c, max_shards=a.loader_shards)) for c in (off, on))
    num["load_simulations"] = dict(shards=a.loader_shards, **compare(s0, s1))
    n, i0, T = len(s1["replaced"]), s1["grid"].i0, cfg.span_max
    win = slice(i0 - T, i0 + T)
    num["load_simulations"]["agrees_with_lean_load"] = bool(all(np.array_equal(s1["data"][k][:, win] if k in ("count", "r_med", "r_hw") else s1["data"][k], B["data"][k][:n])
                                                                for k in keys + ("n_min",)) and np.array_equal(s1["replaced"], B["replaced"][:n]))  # fmt: skip
    return num


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="a run file that names the set (`subdir`) and the guards; `replace_tracks` in it is ignored")
    ap.add_argument("--long-bins", type=int, default=26, help="shortest unbroken run of bins with enough particles that makes a stream long")
    ap.add_argument("--de-max", type=float, default=0.01, help="largest relative energy error of the progenitor orbit (resolved orbits only)")
    ap.add_argument("--n-particles", type=int, default=90000)
    ap.add_argument("--seed", type=int, default=90)
    ap.add_argument("--name", default="", help="name of the new set under <out>/data/; default <subdir>_long<n_particles / 1000>k")
    ap.add_argument("--pilot", type=int, default=6144, help="streams of the pilot (a random sample)")
    ap.add_argument("--full", action="store_true", help="simulate every selected stream")
    ap.add_argument("--check", action="store_true", help="the checks on the finished set")
    ap.add_argument("--max-gpus", type=int, default=8, help="most free GPUs to take, one generate.py worker each")
    ap.add_argument("--batch", type=int, default=32, help="streams at once on a GPU (0 = generate.py's memory probe)")
    ap.add_argument("--cpu-workers", type=int, default=0, help="simulate on this many CPU processes instead (no free GPU)")
    ap.add_argument("--cpu-cores", default="", help="cores the CPU processes share out, e.g. 32-63,160-191; give them, or each process uses the whole machine")
    ap.add_argument("--n-check", type=int, default=20000)
    ap.add_argument("--loader-shards", type=int, default=100, help="--check: shards of the set that load_simulations reads with and without the new tracks")
    ap.add_argument("--workers", type=int, default=24, help="processes that read the shards of the set")
    a = ap.parse_args()
    cfg = Config.load(REPO / a.config)[0]
    out = REPO / cfg.out / "data" / (a.name or f"{cfg.subdir}_long{a.n_particles // 1000}k")
    out.mkdir(parents=True, exist_ok=True)
    sel = select(cfg, a, out)
    n = len(sel["row"])
    print(f"{n} long streams of {int(sel['n_usable'])} usable ones ({n / int(sel['n_usable']):.2%}; {int(sel['n_raw'])} simulated): "
          f"dE_rel <= {a.de_max:g}, run >= {a.long_bins} bins; dissolved {sel['dissolved'].mean():.1%}", flush=True)  # fmt: skip
    if a.check:
        return check(cfg, a, out, sel)
    sc = json.loads(str(sel["sim_config"])) | dict(n_particles=a.n_particles)
    if a.cpu_workers:
        where = dict(cpu_workers=a.cpu_workers, cpu_cores=a.cpu_cores, batch=a.batch and max(1, a.batch // 16))
    else:
        cards = free_gpus()[-a.max_gpus :]  # free right now, by autocvd's test
        if not cards:
            raise SystemExit("no free GPU right now; wait, or pass --cpu-workers with --cpu-cores")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        where = dict(gpus=",".join(u for _, u in cards), batch=a.batch)
        print(f"GPUs {[i for i, _ in cards]}", flush=True)
    theta, dis = sel["theta"], sel["dissolved"]
    if not a.full:
        s = np.sort(np.random.default_rng(cfg.seed + 1).choice(n, min(a.pilot, n), replace=False))
        t0 = time.time()
        r = TN.simulate(theta[s], dis[s], sc, cfg, out / "pilot", a.seed, **where)
        rate = rate_of(out / "pilot")
        print(json.dumps(dict(pilot_streams=len(s), wall_s=round(time.time() - t0, 1), **rate, negative_counts=int((r["count"] < 0).sum()), failed=float(r["failed"].mean()),
                              full_run_hours=round(n / rate["rate"] / 3600, 2)), indent=1))  # fmt: skip
        return
    t0 = time.time()
    TN.simulate(theta, dis, sc, cfg, out, a.seed, **where)
    print(f"done in {(time.time() - t0) / 3600:.2f} h; rate {rate_of(out)['rate']:.1f} streams per second", flush=True)


if __name__ == "__main__":
    main()
