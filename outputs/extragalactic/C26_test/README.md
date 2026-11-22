# Extragalactic stream tracks (C26 forward model): data schema

Code: `scripts/extragalactic/generate.py` (prior, simulator, tests, pilot, run, analyze) and
`scripts/extragalactic/plot.py` (figures). Both carry their uv environment in the file header:

```bash
uv run --script scripts/extragalactic/generate.py test      # checks; add --cpu to stay off the GPUs
uv run --script scripts/extragalactic/generate.py pilot     # 1,000 streams on one free GPU -> data/pilot/
uv run --script scripts/extragalactic/generate.py run       # 100,000 streams, one worker per free GPU -> data/
uv run --script scripts/extragalactic/generate.py analyze   # data/analysis.h5 + data/summary_table.{md,json}
uv run --script scripts/extragalactic/plot.py               # plots/*.pdf, *.png   (--subdir pilot for the pilot)
```

`run` resumes: finished shards are skipped (and refused if their settings differ, unless `--overwrite`).
Every `Config` field in `generate.py` is a command-line option (`--progenitor-mass-loss linear`, `--n-steps 99`,
`--gpus <uuid,uuid>`, `--cpu`, ...).

## Layout

```
data/
  manifest.json            config, versions, shard list (file, start, stop, n, wall_s, device), run timing
  shard_00000.h5 ...       1,000 streams each; stream index k lives in shard k // 1000
  particles_00000.h5 ...   projected particles of the first 5,000 streams (shards 0-4)
  analysis.h5              one row per stream: everything below except track/ (written by `analyze`)
  summary_table.md/.json   the step-7 table
  logs/worker_*.log        one JSON line per event (probe, shard, end)
  tests/                   test_report.json, energy.npz, streamax_compare.npz
  pilot/                   same layout, written by `pilot`   (pilot_cpu/ = the 1,000-stream CPU pilot)
plots/                     figures (plots/<subdir>/ for a pilot)
report.md, README.md
```

## Reproducibility

Stream `k` depends only on `(seed, k)`: `key_k = fold_in(key(seed), k)`, prior draws use `fold_in(key_k, 0)`
(one sub-key per parameter), the particle spray uses `fold_in(key_k, 1)`. The result does not depend on the
batch size, the shard size or the number of GPUs. Integration is float64; stored arrays are float32.

## Shard file (`shard_XXXXX.h5`), n = streams in the shard

Units: kpc, km/s, Gyr, Msun, degrees for the track angle, radians for `psi`.

| dataset | shape, dtype | content |
|---|---|---|
| `index` | (n,) i8 | stream index k |
| `theta` | (n, 13) f4 | `log10_M_halo, R_s, q, cos_i, psi, log10_m_prog, r_s, x0, z0, vx0, vy0, vz0, t_int` (attr `theta_names`) |
| `axis` | (n, 3) f4 | halo symmetry axis n = (sin i cos psi, sin i sin psi, cos i) |
| `c26_vec` | (n, 3) f4 | C26 vector (xhat, yhat, zhat) **in StreaMAX's convention**: r(q) x (-n_x, -n_y, n_z); pass it as `dirx, diry, dirz` to StreaMAX to get this halo |
| `status` | (n,) i4 | bit flags, see below |
| `track/count` | (n, 288) i2 | particles per bin; bin j covers phi in [-1440 + 10 j, -1430 + 10 j) deg |
| `track/r_mean`, `track/r_std` | (n, 288) f4 | mean and standard deviation (ddof 0) of the projected radius; NaN in empty bins |
| `track/r_med`, `track/r_hw` | (n, 288) f4 | median and (p84 - p16)/2: the two statistics StreaMAX `get_track_2D` returns |
| `diag/E`, `diag/Phi0` | (n,) f4 | progenitor specific energy today and host potential at its position, (km/s)^2; Phi -> 0 at infinity |
| `diag/dE_rel` | (n,) f4 | relative energy change of the progenitor orbit over t_int with the simulator's `n_steps` |
| `diag/dE_rel_fine`, `diag/dx_rel` | (n,) f4 | same with 2,000 steps; relative distance between the two orbits at -t_int |
| `diag/r_min`, `diag/r_max` | (n,) f4 | 3D radius range of the backward orbit (2,000 steps) |
| `diag/n_peri`, `diag/n_apo` | (n,) i4 | pericentre / apocentre passages inside t_int |
| `diag/swept_proj_deg`, `diag/swept_3d_deg` | (n,) f4 | projected (signed, forward in time) and 3D angle swept by the progenitor |
| `diag/n_bad`, `diag/n_overflow` | (n,) i4 | non-finite particles; finite particles outside the phi grid |
| `summary/n_valid` | (n,) i4 | bins with >= 100 particles ("valid") |
| `summary/ang_len_deg`, `summary/ang_span_deg` | (n,) f4 | valid bins x 10 deg; first-to-last valid bin |
| `summary/arc_len_kpc` | (n,) f4 | polyline length through the track points of adjacent valid bins |
| `summary/r_track_mean`, `_spread`, `_min`, `_max` | (n,) f4 | mean, std, min, max of `r_mean` over the valid bins |
| `flags/unbound` | (n,) bool | E >= 0 |
| `flags/completed_orbit` | (n,) bool | at least one full radial period inside t_int (`n_peri >= 2` or `n_apo >= 2`) |
| `flags/contiguous` | (n,) bool | the valid bins form one unbroken run |
| `flags/sel_ang`, `sel_arc`, `sel_rad`, `sel_all` | (n,) bool | C26 mock-selection cuts: angular length > 90 deg, arc length > 100 kpc, all valid radii in [10, 500] kpc, and their AND |

File attributes: `config` (JSON of every setting), `physics_hash`, `meta` (device, wall time), `theta_names`,
`status_bits`.

`status` bits: 1 unbound, 2 some particles non-finite, 4 tidal radius undefined at some release epoch
(Omega^2 - d2Phi/dr2 <= 0; StreaMAX returns NaN there), 8 particles outside the phi grid, 16 failed
(non-finite progenitor orbit or no finite particle), 32 projected progenitor angle not monotonic,
64 no valid bin.

## Particle file (`particles_XXXXX.h5`)

`index` (m,), `xy` (m, 10000, 2) f4 projected positions in kpc, `phi` (m, 10000) f4 unwrapped angle in degrees
(progenitor at 0, trailing < 0 < leading). Particle p was released at epoch p // 100 (epoch 0 = -t_int,
epoch 99 = today); even p come from the outer Lagrange point, odd p from the inner one.

## Notes for training (not implemented here)

- The tracks are noise-free. C26 likelihood: `r_obs ~ N(r, (0.02 r)^2)`; 2026b adds `sigma_sys ~ U(0, 25)` kpc in
  quadrature as a 14th parameter.
- Coverage rule: draw an observed window of bins independently of the parameters, keep the (stream, window) pair
  only if every bin of the window has `count >= 50` (C26), mask the rest. `track/count` is stored for this.
- The dataset is the raw prior: unbound streams, unresolved orbits and streams failing the cuts are all in it,
  flagged but not removed. Filter with `flags/*`, `status` and `diag/dE_rel` as needed.
- `theta[:, 9:12]` are km/s (C26 table 1). StreaMAX-fit's `priors.py` feeds the same numbers as kpc/Gyr
  (1 km/s = 1.0227 kpc/Gyr), so its velocity prior is 2.2 % narrower in physical units.
