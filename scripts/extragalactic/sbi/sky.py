"""The figures of a finished evaluation_real.py that show the plane of the sky, redrawn from its files: the
posterior predictive check of every track (sky/sky_p*.png) and the corner plots, which carry that check in their
upper right corner (corner/corner_<track>.png).

    uv run python scripts/extragalactic/sbi/sky.py outputs/extragalactic/C26_test/models/<run>/evaluation_real_leapfrog
    uv run python scripts/extragalactic/sbi/sky.py <run>/evaluation_real <directory>      # write there instead

Reads what the evaluation left (posterior.npz, weights.npz, metrics.json, simulations/) and the settings and
prior of the run (<run>/config.yaml). Nothing is simulated, no network is loaded, and it runs on the CPU. The
plane is that of the track files: x = r cos(theta), y = r sin(theta) in kpc, host at the origin. Its orientation
on the sky (north, east) has not been checked. A model track has no orientation of its own (the simulation frame
puts the progenitor on the x axis); it is drawn bin by bin at the angles of the data, which is how the likelihood
compares the two.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from config import Config, pick_device

pick_device(cpu=True)  # plots only: stay off the GPUs. Must come before jax is imported

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from corner import physical18, track_corners  # noqa: E402
from data import PARAM_NAMES, Grid, Prior, load_observed  # noqa: E402
from mock import encode_observation  # noqa: E402
from predictive import accelerations, figures, read_simulations, replicate, sky_pages  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    to = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else out
    to.mkdir(parents=True, exist_ok=True)
    cfg, stored = Config.load(out.parent / "config.yaml")
    saved, metrics = np.load(out / "weights.npz"), json.loads((out / "metrics.json").read_text())
    with h5py.File(sorted((out / "simulations/data").glob("shard_*.h5"))[0], "r") as f:
        sc = json.loads(f.attrs["config"])
    grid = Grid.from_sims(sc)
    observed = {o["name"]: o for o in load_observed(cfg)}
    tracks = [encode_observation(o["theta_deg"], o["r"], o["r_err"], cfg, grid, dissolved=True, prog=0.5 if metrics["progenitor_given"] else None)
              | dict(title=o["name"], angle0=float(np.median(o["theta_deg"]))) for o in (observed[str(n)] for n in saved["names"])]  # fmt: skip
    z = replicate(read_simulations(out, cfg), saved["params"], tracks, grid, cfg)
    if not np.array_equal(z["cover"], saved["covering"]):
        raise SystemExit("the covering draws rebuilt from the files differ from those in weights.npz: settings or simulations have changed")
    # the acceleration arrows of the corner plots need simulations that hold the line-of-sight coordinate
    z |= dict(weights=saved["weights"]) | accelerations(z, saved["params"], tracks, grid, sc["halo_flattening"])
    sky_pages(tracks, z, saved["params"], metrics["tracks"], grid, figures(to, "sky"))
    post = np.load(out / "posterior.npz")
    if not np.array_equal(post["names"], saved["names"]):
        raise SystemExit("posterior.npz and weights.npz hold different tracks")
    prior = Prior(**{k: tuple(v) for k, v in stored["prior"].items()})
    prior_phys = physical18(prior.from_latent(np.random.default_rng(cfg.seed).standard_normal((20000, len(PARAM_NAMES)))))
    track_corners(tracks, metrics["tracks"], physical18(post["params"]), prior_phys, z, saved["params"], grid.bin_deg, figures(to, "corner"))
    print(f"wrote {to}/sky/sky_p*.png and {to}/corner/corner_<track>.png for {len(tracks)} tracks")


if __name__ == "__main__":
    main()
