"""Dynamic nested sampling (dynesty) of the posterior of observed STRRINGS tracks, the simulator on several GPUs.
The posterior the network approximates, sampled the way the paper samples it. Written to <out>/nested/<track>/.

    uv run --extra gpu python scripts/extragalactic/sbi/nested.py --config outputs/extragalactic/C26_test/npe_v2.yaml --real-tracks NGC1162
    ... --ns-nlive 500 --ns-sample rwalk --ns-tag quick      # other sampler settings, in <out>/nested/NGC1162_quick/

TARGET. prior x likelihood of evaluation_real.py's importance weights: the 15 parameters (the progenitor position
inferred), the paper's likelihood with both guards (likelihood.py) at the `n_min`, `var_ratio` and `sig_sys_max`
of the configuration, the mass of the progenitor ramped to zero, and the simulator and physics of the training
set (`subdir`; generate.py is imported, not run). No trained network is needed. One difference from the
simulations behind the importance weights: every likelihood call here uses the same random key for the release
of the particles (that of stream 0 of the set), so the likelihood is a function of the parameters, which nested
sampling needs; those simulations use a different key per draw.

SAMPLER. dynesty's DynamicNestedSampler on the unit cube of the prior, with the settings of StreaMAX-fit's
"paper" preset by default: 2000 live points, random-slice sampling, multi-ellipsoid bounds, batches added until
the posterior has 10,000 effective samples (`ns_*` in config.py). Where a guard fails the likelihood is not
-inf but a step that rises towards the allowed region (`log_likelihood_steps`), as in the paper's code.

GPUS. `ns_gpus` free cards, found with autocvd's test (`free_gpus` in config.py; it waits until that many are
free), one worker process each. A worker is this file run with the interpreter of generate.py's own uv
environment (`uv python find --script`): the simulator needs the galax of that environment, not the project's.
dynesty proposes `ns_gpus` x `ns_batch` points at once, each by a chain of likelihood calls, one after the
other. Every chain runs in a thread of this process; a likelihood call parks its thread, and when all chains
are parked their calls go to the cards as one batch each (`BatchPool`). This process stays on the CPU.

  settings.json     what was sampled and how; a second call with other settings refuses to continue the run
  nested.log        a line every 30 s: iteration, likelihood calls, ln L of the worst live point, ln Z
  checkpoint.save   dynesty's checkpoint, every 5 minutes; a second call resumes from it
  result.npz        every sample with its weight (`samples`, `samples_u` in the unit cube, `logl`, `weights`),
                    `posterior` (equal-weight draws), `logz`, `logzerr`
  ppc.npz           the model tracks of `ppc_draws` posterior draws, simulated again
  ppc.png, sky_p1.png   posterior predictive check against the angle and in the plane of the track files
  corner.png        the posterior. If evaluation_real.py was run for the network of the configuration, its
                    draws are the blue regions and nested sampling the orange contours
  metrics.json      ln Z, effective samples, likelihood calls, hours, percentiles of q and of every parameter

Sampling, the simulations of the predictive check and the figures are each skipped if their file exists:
delete result.npz or ppc.npz to redo one.
"""

from __future__ import annotations

import dataclasses
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from multiprocessing.connection import Connection

import numpy as np
from scipy.special import ndtri

from config import REPO, free_gpus, parse, pick_device

GENERATE = REPO / "scripts/extragalactic/generate.py"
CHECKPOINT_S = 300  # seconds between two checkpoints of the sampler


def worker(conn) -> None:
    """One process per device, in the environment of generate.py; the device is set by the environment variables
    it was started with. Receives `spec` (the track, the likelihood settings, the configuration of the
    simulations), compiles simulator + likelihood for the batch sizes in spec["sizes"], then answers
    (tracks, params (n, 15), stream indices or None) with the log-likelihoods, or with the model tracks on the
    simulation grid as predictive.read_tracks gives them. The index sets the key of the particle release,
    None = that of stream 0 for every row."""
    try:
        spec = conn.recv()
        sys.path.insert(0, str(GENERATE.parent))
        import generate as gen  # the simulator of the training set; turns on 64-bit JAX

        import jax
        import jax.numpy as jnp
        from data import I_PROG, I_SIG, to_simulator
        from likelihood import log_likelihood_steps, model_at

        names = {f.name for f in dataclasses.fields(gen.Config)}
        gcfg = gen.Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in spec["sim"].items() if k in names})
        gcfg = dataclasses.replace(gcfg, track_ordering=spec["ordering"], alt_ordering=False, theta_file="")
        sim = gen.make_simulator_galax(gcfg) if gcfg.integrator == "galax" else gen.make_simulator(gcfg)
        r, sig, obs = (jnp.asarray(spec[k]) for k in ("r", "sig", "mask"))
        S, sizes = spec["S"], sorted(spec["sizes"])

        @jax.jit
        def run(inp, keys, l, sig_sys):
            res = sim(inp, keys)
            failed = ~res["prog_finite"] | (res["n_bad"] >= gcfg.n_particles) | ~jnp.isfinite(res["E"])  # status bit 16 of generate.py
            track = dict(r_med=jnp.nan_to_num(res["r_med"]), r_hw=jnp.nan_to_num(res["r_hw"]), count=jnp.where(failed[:, None], 0, res["count"]))
            m, hw, n = model_at(track, spec["i0"], l, spec["T"])
            return log_likelihood_steps(r, sig, obs, m, hw, n, sig_sys[:, None], spec["n_min"], spec["var_ratio"]), track, failed

        def evaluate(tracks: bool, par, idx):
            out = []
            idx = np.zeros(len(par), int) if idx is None else idx
            for s in range(0, len(par), sizes[-1]):
                p, k = par[s : s + sizes[-1]], idx[s : s + sizes[-1]]
                n, B = len(p), next(b for b in sizes if b >= len(p))  # the smallest compiled batch that holds them
                p, k = (np.concatenate([a, np.repeat(a[-1:], B - n, axis=0)]) for a in (p, k))  # filled up with the last row
                inp = gen.theta_to_inputs(to_simulator(p[:, :14])) | dict(ramp=np.full(B, float(spec["dissolved"])))
                l = np.clip(np.floor(p[:, I_PROG] * (S + 1)), 0, S).astype(np.int32)  # bins left of the progenitor
                ll, track, failed = run(inp, gen.stream_keys(gcfg, k)[1], l, p[:, I_SIG])
                out.append({a: np.asarray(v, np.float32)[:n] for a, v in track.items()} | dict(failed=np.asarray(failed)[:n]) if tracks else np.asarray(ll)[:n])
            return {a: np.concatenate([o[a] for o in out]) for a in out[0]} if tracks else np.concatenate(out)

        seconds = {}
        for B in sizes:  # compile, then time one batch
            evaluate(False, spec["warm"][:B], None)
            t0 = time.time()
            evaluate(False, spec["warm"][:B], None)
            seconds[B] = time.time() - t0
        conn.send(("ok", dict(device=str(jax.devices()[0]), seconds=seconds)))
        while (msg := conn.recv()) is not None:
            conn.send(("ok", evaluate(*msg)))
    except (EOFError, KeyboardInterrupt):
        pass
    except BaseException:  # noqa: BLE001  the parent raises it
        try:
            conn.send(("error", traceback.format_exc()))
        except OSError:  # the parent is gone
            pass


class Devices:
    """The worker processes, one per device (a GPU UUID, or "cpu"). `devices(params)` -> the log-likelihood of
    every row, the rows shared out among the workers; with tracks=True the model tracks instead."""

    def __init__(self, devices: list[str], spec: dict):
        uv = lambda *a: subprocess.run(["uv", *a, "--script", str(GENERATE)], capture_output=True, text=True, check=True).stdout.strip()
        uv("sync")  # makes the environment of generate.py if this machine has not run it yet
        python = uv("python", "find")
        self.conns, self.procs, self.small = [], [], min(spec["sizes"])
        for d in devices:
            here, there = socket.socketpair()
            env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
            env |= dict(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="") if d == "cpu" else dict(CUDA_VISIBLE_DEVICES=d, CUDA_DEVICE_ORDER="PCI_BUS_ID")
            self.procs.append(subprocess.Popen([python, __file__, "--worker", str(there.fileno())], pass_fds=[there.fileno()], env=env))
            there.close()
            self.conns.append(Connection(here.detach()))
            self.conns[-1].send(spec)
        self.info = [self._recv(c) for c in self.conns]  # waits for the compilations

    @staticmethod
    def _recv(conn):
        try:
            kind, x = conn.recv()
        except EOFError:
            raise RuntimeError("a worker process died (out of memory on its card?)") from None
        if kind == "error":
            raise RuntimeError("worker failed:\n" + x)
        return x

    def __call__(self, par, idx=None, tracks: bool = False):
        parts = np.array_split(np.arange(len(par)), min(len(self.conns), -(-len(par) // self.small)))
        for c, p in zip(self.conns, parts):
            c.send((tracks, par[p], None if idx is None else np.asarray(idx)[p]))
        out = [self._recv(c) for c, _ in zip(self.conns, parts)]
        return {k: np.concatenate([o[k] for o in out]) for k in out[0]} if tracks else np.concatenate(out)

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(None)
            except OSError:
                pass
        for p in self.procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.terminate()


class BatchPool:
    """The `pool` given to dynesty. Its `map(f, items)` runs `size` items at a time, each in a thread. A
    thread that calls the likelihood (`call`) waits; once every running thread waits or has finished, the
    calls are evaluated together by `evaluate` (rows of parameters -> log-likelihoods) in the calling thread,
    and the threads go on. So `size` chains of likelihood calls advance one batch at a time."""

    def __init__(self, evaluate, size: int):
        self.evaluate, self.size = evaluate, size
        self.cond, self.pending, self.running, self.local = threading.Condition(), [], 0, threading.local()
        self.calls = self.batches = 0
        self.seconds = 0.0  # spent in `evaluate`

    def call(self, v) -> float:
        if not getattr(self.local, "pooled", False):  # called outside `map`: a batch of one
            return float(self._evaluate(np.asarray(v, np.float64)[None])[0])
        box = dict(v=np.array(v, np.float64), done=threading.Event())
        with self.cond:
            self.pending.append(box)
            if len(self.pending) >= self.running:
                self.cond.notify_all()
        box["done"].wait()
        if "error" in box:
            raise box["error"]
        return box["value"]

    def _evaluate(self, par):
        t0 = time.time()
        out = self.evaluate(par)
        self.calls, self.batches, self.seconds = self.calls + len(par), self.batches + 1, self.seconds + time.time() - t0
        return out

    def map(self, func, items) -> list:
        items = list(items)
        out, errors = [None] * len(items), []

        def task(i: int) -> None:
            self.local.pooled = True
            try:
                out[i] = func(items[i])
            except BaseException as e:  # noqa: BLE001  raised below, in the calling thread
                errors.append(e)
            finally:
                with self.cond:
                    self.running -= 1
                    self.cond.notify_all()

        for s in range(0, len(items), self.size):
            threads = [threading.Thread(target=task, args=(i,), daemon=True) for i in range(s, min(s + self.size, len(items)))]
            with self.cond:
                self.running = len(threads)
            for th in threads:
                th.start()
            while True:
                with self.cond:
                    self.cond.wait_for(lambda: self.running == 0 or len(self.pending) >= self.running)
                    if self.running == 0:
                        break
                    batch, self.pending = self.pending, []
                try:
                    for box, x in zip(batch, self._evaluate(np.stack([b["v"] for b in batch]))):
                        box["value"] = float(x)
                except BaseException as e:  # noqa: BLE001  every waiting thread gets it and ends
                    for box in batch:
                        box["error"] = e
                for box in batch:
                    box["done"].set()
            for th in threads:
                th.join()
            if errors:
                raise errors[0]
        return out


class Cube:
    """dynesty's unit cube -> the 15 parameters: `prior.from_latent(ndtri(u))` for one point, without its loop
    over the parameters (this is called once per likelihood call, under Python's lock)."""

    def __init__(self, prior):
        kind = np.array(prior.kind)
        self.U, self.N, self.H, self.a, self.b = kind == "U", kind == "N", kind == "H", np.array(prior.a), np.array(prior.b)

    def __call__(self, u):
        x = np.empty(len(u))
        x[self.U] = self.a[self.U] + (self.b[self.U] - self.a[self.U]) * u[self.U]
        x[self.N] = self.a[self.N] * ndtri(u[self.N])
        x[self.H] = self.a[self.H] * ndtri(0.5 * (1.0 + u[self.H]))
        return x


POOL: BatchPool | None = None


def loglike(v) -> float:
    """What dynesty calls. A module-level function, so that the sampler can be pickled into its checkpoint."""
    return POOL.call(v)


def pick_gpus(n: int, interval: int = 30) -> list[str]:
    """UUIDs of n GPUs that are free now, those with the highest index (as `free_gpu` in config.py); waits until
    there are n. Under Slurm the allocation is used."""
    if os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES"):
        return [g for g in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if g][:n]
    start = time.time()
    while True:
        free = free_gpus()
        if len(free) >= n:
            return [uuid for _, uuid in free[-n:]]
        print(f"{len(free)} of {n} GPUs free, waiting ({int(time.time() - start)} s)", file=sys.stderr, end="\r")
        time.sleep(interval)


class Progress:
    """dynesty's `print_func`: a line in nested.log (and on the terminal) every `every` seconds."""

    def __init__(self, path, pool: BatchPool, every: float = 30.0):
        self.path, self.pool, self.every, self.start, self.last, self.calls = path, pool, every, time.time(), 0.0, pool.calls

    def write(self, line: str) -> None:
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")

    def __call__(self, res, niter, ncall, nbatch=None, dlogz=None, stop_val=None, **_) -> None:
        now = time.time()
        if now - self.last < self.every:
            return
        p, dt = self.pool, now - (self.last or self.start)
        end = f"stop {stop_val:.3f} (ends at 1)" if stop_val is not None else f"dlogz {res.delta_logz:.3g} (ends at {dlogz})"
        self.write(f"{(now - self.start) / 3600:6.2f} h  batch {nbatch}  it {niter}  calls {ncall}  eff {res.eff:.2f} %  logl* {res.loglstar:.6g}  "
                   f"logz {res.logz:.6g}  {end}  |  {(p.calls - self.calls) / dt:.0f} calls/s, {p.calls / max(p.batches, 1):.0f} per batch, "
                   f"{p.seconds / max(now - self.start, 1e-9):.0%} of the time on the devices")  # fmt: skip
        self.last, self.calls = now, p.calls


def main() -> None:
    global POOL
    cfg = parse()
    cpu = cfg.device.startswith("cpu")
    gpus = None if cpu else pick_gpus(cfg.ns_gpus)  # before the next line empties CUDA_VISIBLE_DEVICES (Slurm)
    pick_device(cpu=True)  # this process steers and plots: it stays on the CPU

    import dynesty
    import h5py
    from dynesty.utils import quantile

    from corner import corner, physical15
    from data import PARAM_NAMES, Grid, Prior, load_observed, orientation
    from mock import encode_observation
    from predictive import plt, ppc_panel, replicate, sky_pages

    if cfg.prog_given:
        raise SystemExit("nested.py infers the progenitor position (15 parameters); prog_given is not written")
    wanted = [s for s in cfg.real_tracks.split(",") if s]
    observed = [o for o in load_observed(cfg) if o["name"] in wanted or o["name"].split("_")[0] in wanted]
    if not observed:
        raise SystemExit(f"name the observed tracks to sample with --real-tracks (e.g. NGC1162); none matches {wanted}")
    shards = sorted(cfg.data_dir().glob("shard_*.h5"))
    if not shards:
        raise SystemExit(f"no shards in {cfg.data_dir()}: they hold the prior and the physics of the simulator")
    with h5py.File(shards[0], "r") as f:
        sc = json.loads(f.attrs["config"])
    sc |= dict(integrator=cfg.sim_integrator) if cfg.sim_integrator else {}
    prior, grid, rng = Prior.from_sims(sc, cfg.sig_sys_max), Grid.from_sims(sc), np.random.default_rng(cfg.seed)
    # batch sizes the workers compile: the chains of a round end one by one, so most batches are far from full
    # (75 calls for 512 chains on average with rslice) and a batch costs its compiled size
    cube, sizes = Cube(prior), sorted({s for s in range(8, cfg.ns_batch, 8) if s <= 64 or s % 16 == 0} | {cfg.ns_batch})
    D = len(PARAM_NAMES)

    for o in observed:
        t = encode_observation(o["theta_deg"], o["r"], o["r_err"], cfg, grid, dissolved=True, prog=None) | dict(title=o["name"], angle0=float(np.median(o["theta_deg"])))
        out = REPO / cfg.out / "nested" / (o["name"] + (f"_{cfg.ns_tag}" if cfg.ns_tag else ""))
        out.mkdir(parents=True, exist_ok=True)
        settings = dict(track=o["name"], points=int(t["mask"].sum()), subdir=cfg.subdir, ordering=cfg.ordering, n_min=cfg.n_min, var_ratio=cfg.var_ratio,
                        sig_sys_max=cfg.sig_sys_max, span_max=cfg.span_max, seed=cfg.seed, nlive=cfg.ns_nlive, sample=cfg.ns_sample,
                        queue_size=(1 if cpu else cfg.ns_gpus) * cfg.ns_batch,
                        simulator={k: sc[k] for k in ("integrator", "n_particles", "n_steps", "halo_flattening", "seed")})  # fmt: skip
        if (out / "settings.json").exists() and json.loads((out / "settings.json").read_text()) != settings:
            raise SystemExit(f"{out} holds a run with other settings (settings.json): delete it or give this one a --ns-tag")
        (out / "settings.json").write_text(json.dumps(settings, indent=1))
        spec = dict(sim=sc, ordering=cfg.ordering, n_min=cfg.n_min, var_ratio=cfg.var_ratio, T=cfg.span_max, i0=grid.i0, S=t["S"], dissolved=True,
                    r=t["r"], sig=t["sig"], mask=t["mask"], sizes=sizes, warm=prior.from_latent(rng.standard_normal((max(sizes), D))))  # fmt: skip
        devices, extra = None, {}

        def start() -> Devices:  # the workers, when first needed
            nonlocal devices
            if devices is None:
                t0 = time.time()
                devices = Devices(["cpu"] if cpu else gpus, spec)
                print(f"{o['name']}: {len(devices.procs)} worker(s) ready in {time.time() - t0:.0f} s: " + "; ".join(
                    f"{i['device']} " + ", ".join(f"{b} streams in {s:.2f} s" for b, s in i["seconds"].items()) for i in devices.info), flush=True)  # fmt: skip
            return devices

        # --- sampling
        if not (out / "result.npz").exists():
            POOL = BatchPool(start(), len(devices.procs) * cfg.ns_batch)
            ck, log = out / "checkpoint.save", Progress(out / "nested.log", POOL)
            resume = ck.exists()
            if resume:
                sampler = dynesty.DynamicNestedSampler.restore(str(ck), pool=POOL)
            else:
                sampler = dynesty.DynamicNestedSampler(loglike, cube, D, nlive=cfg.ns_nlive, sample=cfg.ns_sample, pool=POOL,
                                                       queue_size=POOL.size, rstate=np.random.default_rng(cfg.seed))  # fmt: skip
            log.write(f"# {time.strftime('%Y-%m-%d %H:%M')}  {'resumed' if resume else 'started'}: {json.dumps(settings)}")
            sampler.run_nested(n_effective=cfg.ns_neff, dlogz_init=cfg.ns_dlogz, maxcall=cfg.ns_maxcall or None, checkpoint_file=str(ck),
                               checkpoint_every=CHECKPOINT_S, resume=resume, print_func=log)  # fmt: skip
            res = sampler.results
            w = np.exp(res.logwt - res.logwt.max())
            w /= w.sum()
            hours = (time.time() - log.start) / 3600
            log.write(f"# done in {hours:.2f} h (this call): {res.niter} iterations, {int(np.sum(res.ncall))} calls, logz {res.logz[-1]:.3f} +- {res.logzerr[-1]:.3f}")
            np.savez(out / "result.npz", param_names=PARAM_NAMES, samples=res.samples, samples_u=res.samples_u, logl=res.logl, weights=w,
                     posterior=res.samples[rng.choice(len(w), size=cfg.ns_neff, p=w)], logz=res.logz[-1], logzerr=res.logzerr[-1],
                     niter=res.niter, ncall=int(np.sum(res.ncall)), hours=hours, calls_per_s=POOL.calls / (3600 * hours))  # fmt: skip
        z = np.load(out / "result.npz")
        w, post = z["weights"], z["posterior"]

        # --- posterior predictive check: `ppc_draws` posterior draws simulated again, with the key of the sampling
        M = min(cfg.ppc_draws, len(post))
        if M > 0 and not ((out / "ppc.npz").exists() and np.array_equal(np.load(out / "ppc.npz")["params"], post[:M])):
            np.savez(out / "ppc.npz", params=post[:M], **start()(post[:M], tracks=True))
        if devices is not None:
            devices.close()
        row = dict(track=o["name"], points=settings["points"])
        if M > 0:
            sim = dict(np.load(out / "ppc.npz"))
            par = sim.pop("params")
            rep = replicate(sim, par[None], [t], grid, cfg)
            m, var, cover = rep["m"][0], rep["var"][0], rep["cover"][0]
            chi2 = lambda d: np.sum(np.where(t["mask"], (d - m) ** 2 / var, 0.0), axis=-1)
            row |= dict(covering=float(cover.mean()), p_value=float((chi2(rep["rep"][0]) >= chi2(t["r"]))[cover].mean()) if cover.any() else float("nan"))
            panel = lambda ax, **kw: ppc_panel(ax, t, rep["rep"][0], m, cover, row, name="nested sampling", **kw)
            fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
            panel(ax, legend=True)
            fig.savefig(out / "ppc.png", dpi=130)
            plt.close(fig)
            sky_pages([t], rep, par[None], [row], grid, out)
            extra = dict(ppc=lambda ax: panel(ax, legend=True, fontsize=10))

        # --- corner plot, next to the network's posterior if evaluation_real.py left one for this track
        net = cfg.run_dir() / ("evaluation_real" + (f"_{cfg.sim_integrator}" if cfg.sim_integrator else "")) / "posterior.npz"
        phys, big = physical15(z["samples"]), w > 1e-6 * w.max()
        prior_phys = physical15(prior.from_latent(rng.standard_normal((20000, D))))
        title = f"{o['name']}: {settings['points']} points, ln Z = {float(z['logz']):.2f} +- {float(z['logzerr']):.2f}"
        if net.exists() and o["name"] in np.load(net)["names"]:
            n = np.load(net)
            corner(physical15(n["params"][list(n["names"]).index(o["name"])]), prior_phys, out / "corner.png", title, weighted=(phys[big], w[big]),
                   blue=f"network posterior ({cfg.name})", orange="nested-sampling samples, weighted", **extra)  # fmt: skip
        else:
            corner(physical15(post), prior_phys, out / "corner.png", title, blue="nested sampling, equal-weight draws", **extra)

        pct = lambda x: [float(v) for v in quantile(x, [0.16, 0.5, 0.84], weights=w)]
        row |= dict(logz=float(z["logz"]), logzerr=float(z["logzerr"]), effective_samples=float(1.0 / np.sum(w**2)), samples=len(w), iterations=int(z["niter"]),
                    calls=int(z["ncall"]), hours=float(z["hours"]), calls_per_s=float(z["calls_per_s"]), logl_max=float(z["logl"].max()),
                    q=pct(orientation(z["samples"][:, 2:5])[:, 0]), parameters={k: pct(z["samples"][:, i]) for i, k in enumerate(PARAM_NAMES)})  # fmt: skip
        (out / "metrics.json").write_text(json.dumps(dict(settings=settings, **row), indent=1))
        print(f"wrote {out}: " + ", ".join(sorted(p.name for p in out.iterdir())), flush=True)


if __name__ == "__main__":
    worker(Connection(int(sys.argv[2]))) if sys.argv[1:2] == ["--worker"] else main()
