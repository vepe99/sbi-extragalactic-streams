"""The posterior predictive check of a finished evaluation_real.py, redrawn in the plane of the sky: sky_p*.png.

    uv run python scripts/extragalactic/sbi/sky.py outputs/extragalactic/C26_test/models/<run>/evaluation_real_leapfrog

Reads what the evaluation left (weights.npz, metrics.json, simulations/) and the settings of the run
(<run>/config.yaml). Nothing is simulated, no network is loaded, and it runs on the CPU. The plane is that of
the track files: x = r cos(theta), y = r sin(theta) in kpc, host at the origin. Its orientation on the sky
(north, east) has not been checked. A model track has no orientation of its own (the simulation frame puts
the progenitor on the x axis); it is drawn bin by bin at the angles of the data, which is how the likelihood
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

from data import Grid, load_observed  # noqa: E402
from mock import encode_observation  # noqa: E402
from predictive import read_tracks, replicate, sky_pages  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    cfg = Config.load(out.parent / "config.yaml")[0]
    saved, metrics = np.load(out / "weights.npz"), json.loads((out / "metrics.json").read_text())
    with h5py.File(sorted((out / "simulations/data").glob("shard_*.h5"))[0], "r") as f:
        grid = Grid.from_sims(json.loads(f.attrs["config"]))
    observed = {o["name"]: o for o in load_observed(cfg)}
    tracks = [encode_observation(o["theta_deg"], o["r"], o["r_err"], cfg, grid, dissolved=True, prog=0.5 if metrics["progenitor_given"] else None)
              | dict(title=o["name"], angle0=float(np.median(o["theta_deg"]))) for o in (observed[str(n)] for n in saved["names"])]  # fmt: skip
    z = replicate(read_tracks(out / "simulations"), saved["params"], tracks, grid, cfg)
    if not np.array_equal(z["cover"], saved["covering"]):
        raise SystemExit("the covering draws rebuilt from the files differ from those in weights.npz: settings or simulations have changed")
    sky_pages(tracks, z | dict(weights=saved["weights"]), saved["params"], metrics["tracks"], grid, out)
    print(f"wrote {out}/sky_p*.png")


if __name__ == "__main__":
    main()
