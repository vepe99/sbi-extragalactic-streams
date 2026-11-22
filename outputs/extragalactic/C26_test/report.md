# C26 extragalactic stream dataset: report

**Status (2026-11-22): complete.** The 100,000-stream production set was generated on CPU on 2026-11-22
(`data/manifest.json` has `"complete": true`, no failed worker, no failed stream). The GPU path has still never
been run. Everything in `data/` and `plots/` was produced by commit `461e2aa`; the prior inspection
(`plots/prior_inspection/`) by the commit that added `analysis/inspect_prior.py`.

## Machine and environment

| | |
|---|---|
| CPU | 2 x AMD EPYC 7713 (64 cores each, 128 physical cores, 256 threads, 2 NUMA nodes), AVX2 |
| RAM | 1.0 TiB |
| GPU | none (`nvidia-smi` not installed) |
| OS / platform | Ubuntu 24.04.4 LTS, kernel 6.8.0-149-generic, Linux x86_64, glibc 2.39 |
| uv / Python | uv 0.11.8, CPython 3.12.3 |
| packages | jax 0.12.1, jaxlib 0.12.1, galax 0.1.0 (the latest release on PyPI on this date), unxt 2.2.0, coordinax 0.24.3, equinox 0.13.8, diffrax 0.7.2, numpy 2.5.5, scipy 1.18.1, h5py 3.16.0, StreaMAX 0.4.1 @ 9fd520e |
| jax line | `jax[cuda12]==0.12.1` left as received (Linux x86_64; the CUDA wheels are installed and unused) |
| lock files | `scripts/extragalactic/generate.py.lock`, `plot.py.lock` (script environments), `uv.lock` (project) |
| cores used | tests and pilot: 16 (`taskset -c 0-15`); production: 100 physical cores, one per worker (0-49 and 64-113, 50 per socket), `nice -n 10`; the machine is shared and was otherwise idle (load 1.5) |

The only code change against the received archive is in the launcher: `run --cpu` takes `--cpu-workers` and
`--cpu-cores` and starts several pinned CPU workers (`cmd_run`, `cpu_pins`, two `Config` fields). The simulator,
the prior, the track extraction and the physics hash (`e39332650ce2`) are untouched.

## What was run, in order

All through `uv run --script scripts/extragalactic/generate.py` (`G`) or `plot.py` (`P`).

| # | command | cores | wall time | result |
|---|---|---|---|---|
| 1 | `G analyze --help` | - | 85 s | first build of the environment (downloads included) |
| 2 | `taskset -c 0-15 G test --cpu --n-test-streams 96` | 16 | 30 s (38 s with start-up) | 19 / 19 checks pass |
| 3 | `taskset -c 0-15 G pilot --cpu` | 16 | 45 s (48 s with start-up) | 1,000 streams, matches the reference |
| 4 | `G pilot --cpu --pilot-n 256 --n-save-particles 0 --batch-size B --subdir tune_tT_bB`, 21 runs | 1 to 16 each | about 1 min each, run concurrently | batch and core scan below; folders deleted |
| 5 | `nohup nice -n 10 G run --cpu --batch-size 16 --cpu-workers 100 --cpu-cores 0-49,64-113 > run.log` | 100 | 435 s (439 s with start-up) | 100,000 / 100,000 streams, 100 shards, no failed worker |
| 6 | `G analyze` | 32 | 7 s | `analysis.h5`, `summary_table.md/.json` |
| 7 | `P` | 32 | 27 s | all figures |
| 8 | `P --only tests` | 32 | 2.5 s | the two test figures |
| 9 | `G pilot --cpu --pilot-n 200 --n-save-particles 0`, 5 runs with batch 8 / 16 and 1 / 16 cores | 1 or 16 | under 2 min | reproducibility check below; folders deleted |
| 10 | `uv run python analysis/inspect_prior.py` | 16 | 9 s | `plots/prior_inspection/`, `data/prior_inspection.json` |

No interruption, no resume, no failed worker, nothing in the simulator needed a fix. The resume path of `run` was
therefore not exercised.

## Timings

**One JAX process does not scale across cores on this machine.** Streams per second, 256 streams per
configuration, compile time excluded (the 2- to 16-core runs shared the machine with each other; the 1-core
runs and the 2-core batch-64 run were measured with the machine nearly idle):

| cores per worker | batch 4 | batch 8 | batch 16 | batch 32 | batch 64 | best, per core |
|---|---|---|---|---|---|---|
| 1 | - | 4.60 | 4.63 | 4.58 | 4.65 | 4.6 |
| 2 | 6.49 | 6.75 | 6.75 | 6.97 | 8.75 | 4.4 |
| 4 | 11.22 | 11.20 | 11.09 | 10.95 | - | 2.8 |
| 8 | 14.16 | 17.97 | 16.87 | 16.96 | - | 2.2 |
| 16 | 9.63 | 20.65 | 22.97 | 24.56 | - | 1.5 |

The batch size barely matters below 16 cores; the number of cores per worker does. Production therefore used
100 single-core workers with batch 16, one shard each.

| | pilot (16 cores, batch 8) | production (100 x 1 core, batch 16) |
|---|---|---|
| compile | 3.6 s | 14.7 to 22.2 s per worker (median 16.2 s) |
| rate | 24.5 streams/s (reference machine: 13.6) | 2.5 streams/s per worker (median; 3.0 for the fastest), 230 streams/s overall |
| wall time | 45 s for 1,000; projected 68 min for 100,000 | 435 s for 100,000; shards took 332 to 410 s (median 399 s) |
| peak memory | 1.3 GB (resident) | 138 GB summed over the 100 workers, 1.6 GB for the largest process |
| load average | - | about 105 of 256 logical CPUs |

A single core runs at 4.6 streams/s alone and at 2.5 with 99 neighbours busy, so 100 concurrent workers cost
about 45 % per worker; the cause (memory bandwidth, clock frequency) was not investigated.

Disk: `data/` is 784 MB in total.

| files | size |
|---|---|
| 100 `shard_*.h5` (tracks, parameters, diagnostics) | 94 MB (about 0.95 MB each) |
| 5 `particles_*.h5` (first 5,000 streams) | 573 MB (120 MB each) |
| `analysis.h5` | 18 MB |
| `pilot/` (1,000 streams with particles) | 116 MB |
| `tests/`, `logs/`, `manifest.json`, summary tables | 2 MB |
| `plots/` (PDF and PNG) | 11 MB |

## Tests (step 4), `data/tests/test_report.json`

Run on this machine: 19 / 19 pass. The values are the ones of the original machine except where a second number
is given in brackets (original machine).

| check | result |
|---|---|
| prior: KS of xhat, yhat vs N(0,1), zhat vs half-normal (1e5 draws) | p = 0.042, 0.134, 0.849 |
| prior: KS of the 13 parameters vs their analytic priors | smallest p = 0.089 (psi) |
| stored C26 vector maps back to q through C26 eq. 11 | max error 4e-15 |
| stream k independent of the batch it is drawn in | exact |
| potential: analytic gradient vs autodiff | < 4e-12 |
| potential: Laplacian / 4 pi G vs NFW density on m^2 = R^2 + Z^2/q^2 | 5e-6 (q = 1.5) to 3.8e-3 (q = 0.5) |
| potential vs galax `TriaxialNFWPotential` (50-node quadrature) | 3e-8 (q = 1.5) to 4.3e-5 (q = 0.5) |
| q = 1, two different axes, fixed seed | identical particles and tracks |
| n and -n | identical particles and tracks |
| whole system rotated by 70.7 deg about z | r(phi) unchanged (float32 storage: 2e-7) |
| StreaMAX progenitor orbit (float32) vs this code, 5 bound sets | max relative difference 1.4e-5 |
| StreaMAX binned tracks vs this code, same ordering, 80 bins | z mean 0.11, std 0.97, max 2.76 (0.07, 0.96, 2.76) |
| StreaMAX symmetry axis for the stored `c26_vec` | equals `axis` |

Figures: `plots/streamax_comparison.png`, `plots/energy_conservation.png`.

**Energy conservation of the progenitor orbit** (200 prior draws, 126 bound), relative energy change over t_int,
percentiles 50 / 90 / 99 / max:

| steps | all | bound | unbound |
|---|---|---|---|
| 99 | 1.7e-4 / 1.33 / 5.0 / 15.5 | 6.0e-3 / 1.83 / 6.0 / 15.5 | 3.5e-5 / 1.8e-4 / 4.5e-3 / 7.6e-3 |
| 100 | 1.6e-4 / 0.90 / 15 / 38 | 4.9e-3 / 1.37 / 19 / 38 | 3.4e-5 / 1.7e-4 / 4.4e-3 / 7.4e-3 |
| 2000 | 2.0e-7 / 8.8e-5 / 0.55 / 1.14 (0.54 / 1.05) | 8.0e-7 / 3.0e-3 / 0.61 / 1.14 (0.74 / 1.05) | 8.5e-8 / 4.3e-7 / 1.1e-5 / 1.8e-5 |

About 100 leapfrog steps do not resolve the deeply bound part of the prior (massive halo, small pericentre, many
orbits inside t_int): for roughly a third of the bound draws the energy error is of order one and the position at
-t_int differs from the 2,000-step orbit by more than the orbit's size. These orbits are an integrator artefact,
their leapfrog map is chaotic (round-off changes individual particles by O(r)), and they are reproduced here on
purpose, because the C26 / 2026b posteriors are computed with the same integrator. `diag/dE_rel` flags them.

## Step-7 table, all 100,000 streams (`data/summary_table.md`)

| fraction of streams | all (n = 100000) | bound only (n = 65026) |
|---|---|---|
| unbound (E >= 0) | 34.97 % | 0.00 % |
| bound, no completed radial orbit in t_int | 21.19 % | 32.59 % |
| failed / NaN | 0.00 % | 0.00 % |
| at least one valid bin | 100.00 % | 100.00 % |
| angular length > 90 deg | 41.43 % | 43.85 % |
| arc length > 100 kpc | 62.85 % | 66.43 % |
| valid radii in [10, 500] kpc | 78.99 % | 70.23 % |
| all three cuts | 23.51 % | 20.85 % |

Other flags (all streams): any non-finite particle 0.00 % (mean particle fraction 0.00 %), tidal radius undefined at some epoch 0.00 %, particles outside the phi grid 6.98 %, projected progenitor angle not monotonic 15.66 %, all three cuts and contiguous valid bins 12.02 %.

Of the streams passing all three cuts, 42.32 % are unbound. Progenitor orbit not resolved by 99 leapfrog steps (relative energy error > 1 %): 31.80 % of all, 48.03 % of the bound, 28.70 % of those passing the cuts.

| log10 M_halo | 11.0-11.5 | 11.5-12.0 | 12.0-12.5 | 12.5-13.0 | 13.0-13.5 | 13.5-14.0 |
|---|---|---|---|---|---|---|
| unbound | 93.67 % | 74.92 % | 35.75 % | 5.41 % | 0.13 % | 0.00 % |

| speed today [km/s] | 0-100 | 100-200 | 200-300 | 300-400 | 400-500 | 500-700 | >700 |
|---|---|---|---|---|---|---|---|
| unbound | 1.08 % | 10.65 % | 24.01 % | 34.31 % | 40.49 % | 48.96 % | 56.98 % |

Statistical error on these fractions: at most 0.16 percentage points. Failure and NaN rates: 0 failed streams,
0 non-finite particles, tidal radius never undefined; 1 stream of 100,000 has no valid bin.

**Pilot against the reference** (the same 1,000 streams, original machine in brackets): unbound 36.3 % (36.3),
bound with no completed radial orbit 21.9 % (21.9), failed 0 % (0), angular length > 90 deg 40.7 % (40.7), arc
length > 100 kpc 66.7 % (66.7), valid radii in [10, 500] kpc 79.6 % (79.5), all three cuts 24.4 % (24.3), 21.7 %
of the bound (21.5), 43.4 % of the passing streams unbound (43.6). One stream differs.

## Reproducibility across batch sizes and thread counts

Stream `k` depends only on `(seed, k)` for its parameters and its random keys, but the stored tracks are **not
bit-identical across batch sizes**:

- same batch size, 1 core against 16 cores, and the same configuration run twice: identical tracks (200 streams);
- batch 8 against batch 16 on one core: 5 of 200 streams differ, all 5 with an unresolved orbit (energy error > 1 %);
- pilot shard 0 (batch 8, 16 cores) against production shard 0 (batch 16, 1 core), the same 1,000 streams:
  parameters, flags and all diagnostics identical; `track/r_mean` differs in 32 streams, 31 of them with an
  unresolved orbit; median relative difference 8e-7, largest 24 %; no selection flag changes.

This is the round-off sensitivity of the unresolved orbits described above (the compiled code differs with the
batch size), not a seeding problem. A dataset is exactly reproducible only at the batch size it was made with
(16 here, recorded in every shard).

## Prior inspection ahead of amortized inference

`analysis/inspect_prior.py`, numbers in `data/prior_inspection.json`, figures in `plots/prior_inspection/`.

- **Usable fraction.** 23.5 % of the draws pass the three C26 cuts (23,511). Requiring a resolved progenitor
  orbit (energy error < 1 %) leaves 16.8 % (16,763); of those 58 % are unbound fly-bys (7,043 bound) and 66 %
  have contiguous valid bins (11,143).
- **The cuts leave the halo-shape prior almost untouched.** Kolmogorov-Smirnov distance between the accepted
  and the full prior (cuts + resolved orbit): q 0.04, cos i 0.005, psi 0.004, R_s 0.02. Acceptance falls from
  19 % at q < 0.7 to 15 % at q > 1.3. The cuts act on x0 (0.31), log10 M_halo (0.26, almost entirely through the
  resolved-orbit requirement: 0.8 % of the draws above 10^13.5 Msun survive), log10 m_prog (0.26), vy0 (0.23) and t_int (0.19).
- **Unresolved orbits follow the number of pericentres.** Among streams passing the cuts 28.7 % are unresolved:
  3 % with no pericentre inside t_int, 4 % with one, 17 % with two, 38 % with three, 88 % with six or more. 58 %
  of the passing streams have at most one pericentre.
- **What the network receives.** 10 to 24 valid bins (5th to 95th percentile, median 16), with valid bins on both
  sides of the progenitor in 99.99 % of them; the radial contrast (r_max - r_min) / mean is 0.4 to 2.6
  (median 1.2), far above the 2 % noise.
- **STRRINGS sample used: 32 of the 35 streams.** NGC1084, NGC1121 and PGC021008 are excluded (decision of
  2026-11-22, Giuseppe; `STRRINGS_EXCLUDED` in `analysis/inspect_prior.py`). Every number below is for the 32.
- **Against STRRINGS** (arXiv:2508.02154 table 4, transcribed from the arXiv HTML and not checked against the
  journal table; lengths and angles are measured differently in data and simulation). 15 of the 32 observed
  streams are shorter than the 100 kpc cut and 3 subtend less than 90 deg. Length / angle gives a median radius of
  35 kpc for the data (5th to 95th percentile 16 to 77) against 78 kpc (38 to 174) for the usable simulations.
  Only 2.3 % of the simulations passing the cuts are longer than 270 deg, against 10 of the 32 observed streams.
  Counting simulations with a resolved orbit within 0.2 dex in length and 30 deg in angle of each observed
  stream (no cut): median 1,128, fewer than 100 for 7 streams, none for 2 (PGC000902, PGC938075). With the three
  cuts: median 106, none for 11.
- **Nearest neighbours** (`strrings_neighbours_p1..4`, `strrings_neighbour_parameters`). Pool: resolved orbit and
  contiguous valid bins, no C26 cut (55,674 tracks); distance = hypot(dlog10 length / 0.2, dangle / 30 deg). 24 of
  the 32 streams have more than 50 tracks within distance 1 and a nearest neighbour closer than 0.35. NGC4256,
  IC0120 and NGC1162 have 37, 17 and 13; NGC0804 has 1; PGC1001085, UGC08717, PGC000902 and PGC938075 have none
  (nearest at distance 1.2, 3.0, 6.4 and 4.5). The neighbours (up to 100 per stream, 2,354 in all) come from a
  narrow part of the prior: median projected distance x0 = 29 kpc (prior 101), 3D distance 100 kpc (177),
  log10 M_halo 12.2, t_int 2.1 Gyr; 69 % are bound and 44 % pass the three cuts. Their q, cos i and psi follow the prior.
- **Bound against unbound among the usable streams** (`gallery_bound`, `gallery_unbound`,
  `bound_vs_unbound_tracks`). Length, angle and radius are alike (median angle 160 against 150 deg, radius 73
  against 82 kpc). The width is not: the median radial width of a bin relative to its radius is 0.08 for the
  bound (16th to 84th percentile 0.05 to 0.16) and 0.24 for the unbound (0.11 to 0.49). 82 % of the bound and
  28 % of the unbound (2,775 streams) are thinner than 0.15. Unbound progenitors have at most one pericentre
  (53 % none); the bound ones a median of two.
- **Corner view of the bound fraction** (`corner_bound_vs_all`, `corner_bound_vs_all_focus`): being bound is
  decided by halo mass against speed; no other pair of parameters shows structure.

## Figures (`plots/`; the pilot's are in `plots/pilot/`)

`prior_marginals`, `energy`, `binned_values`, `gallery` (PDF, 3 pages; `gallery_p1..3.png`), `noise_examples`,
`streamax_comparison`, `energy_conservation`; in `plots/prior_inspection/`: `funnel`, `prior_bound_vs_all`,
`corner_bound_vs_all`, `corner_bound_vs_all_focus`, `acceptance`, `acceptance_maps`, `integrator`, `observable`,
`strrings`, `strrings_neighbours_p1..4`, `strrings_neighbour_parameters`, `gallery_bound`, `gallery_unbound`,
`bound_vs_unbound_tracks`. Each as PDF and PNG; all were looked at.

## What was read in StreaMAX, and where this differs from the prompt

Sources: StreaMAX @ 9fd520e (v0.4.1, 2026-08-20) and its history (the January-2026 package 1f10dc8 and the
September-2025 scripts 27279f3 that C26 was written with), StreaMAX-fit @ 2e6a708, arXiv:2601.15373v3,
arXiv:2607.05510.

1. **q enters the density.** C26 v3 says "flattening in the density"; StreaMAX-fit uses host `NFWFlatDensity`
   (Chandrasekhar integral, 12-node Gauss-Legendre in v = cos theta). The code C26 was first written with
   flattened the potential (`NFW`: r^2 = x^2 + y^2 + z^2/q^2); the density version entered StreaMAX in May 2026.
   Default here: density, same 12 nodes. `--halo-flattening potential` gives the legacy model.
2. **M is the NFW scale mass**, 4 pi rho0 R_s^3: Phi = -G M ln(1 + r/R_s) / r at q = 1. Not a virial mass. With
   flattening the central density normalisation stays M / (4 pi R_s^3) (verified through the Laplacian).
3. **Axis convention.** StreaMAX evaluates the halo-frame potential at R x with R = `get_mat(dir)`, the rotation
   taking z onto `dir`. The symmetry axis of its halo is therefore R^T z = (-dirx, -diry, dirz) / |dir|, the mirror
   of `dir` in the sky-plane components. To describe the same halo, `c26_vec` is stored as r(q) x (-n_x, -n_y, n_z),
   not r x n as the prompt states. The prior is the same either way; the orientation posteriors are not.
   Checked numerically against StreaMAX.
4. **Release schedule and step count.** StreaMAX takes `n_steps` and releases at `n_steps + 1` epochs spanning
   [-t_int, 0] with both ends included, `n_particles / (n_steps + 1)` particles per epoch, alternating outer and
   inner Lagrange point; every particle then takes `n_steps` steps of (remaining time) / `n_steps`. With 10,000
   particles this only divides for **n_steps = 99** (the StreaMAX-fit default): 100 epochs x 100 particles, 99 steps.
   The prompt's "100 steps" with 10,000 particles is not expressible in StreaMAX (100 steps needs 10,100
   particles: `--n-steps 100 --n-particles 10100`). The 100 particles released today take zero-length steps and
   sit at their release offsets.
5. **Progenitor seen by the particles.** StreaMAX re-integrates the progenitor from each release epoch with that
   epoch's step (`precompute_prog_trajectories`) and evaluates its Plummer force at the step boundaries. Mirrored.
6. **Fardal+15 constants**: k_r = 2.0 +- 0.5, k_vphi = 0.3 +- 0.5 (multiplied by k_r), k_z = 0 +- 0.5, k_vz = 0 +- 0.5;
   r_t = (G m / (Omega^2 - d2Phi/dr2))^(1/3), Omega = |L| / r^2, v_t = Omega r_t. galax's `FardalStreamDF` has the
   same constants and tidal radius and is used as is; StreaMAX returns NaN where Omega^2 - d2Phi/dr2 <= 0 (galax
   would take a real cube root), which is reproduced and flagged (status bit 4).
7. **Mass loss `linear`.** StreaMAX-fit's `linear_to_zero` sets the final log10 mass to 0: log10 m falls linearly
   in time from log10 m_prog to 0, i.e. the mass decays exponentially to 1 Msun, not linearly to zero.
   `--progenitor-mass-loss linear` reproduces StreaMAX.
8. **Velocity units.** C26 table 1 and the C26-era code use km/s (converted with 1.0227 kpc/Gyr per km/s);
   StreaMAX-fit's `priors.py` passes N(0, 250) straight in as kpc/Gyr. This dataset follows the papers: km/s.
9. **Track statistics.** The papers describe the mean radius and the radial standard deviation; StreaMAX
   `get_track_2D` returns the median and (p84 - p16)/2. Both pairs are stored.
10. **Ordering.** The dataset uses the C26 per-particle unwrapping (`track_ordering=c26`). StreaMAX HEAD has only
    the Gibbons+14 ordering of 2026b; `--track-ordering gibbons14` reproduces it (implemented, compared against
    StreaMAX, not used). The two orderings differ in sparse tails and for near-radial streams: C26 ordering
    against StreaMAX's on the same 80 bins gives z std 1.65 with 7.5 % of bins beyond 3 sigma.
11. **Selection cuts.** Implemented as the prompt and the paper state them. The C26-era `mock_stream.py` differs:
    it also requires contiguous valid bins (stored as `flags/contiguous`), a monotonic progenitor angle (status
    bit 32), more than 9 valid bins, min particle radius > 2 kpc, and measures arc length as sum r tanh(dphi).
12. **Prior ranges** follow C26 table 1 (R_s in [10, 25], r_s in [1, 5], z0 > 0). StreaMAX-fit's `priors.py` uses
    R_s in [5, 25], r_s in [1, 3] and an untruncated z0.

## galax extensions (all inside `generate.py`)

- `OrientedNFWPotential`, a `galax.potential.AbstractSinglePotential`: NFW flattened along an arbitrary unit
  vector, in the density (StreaMAX quadrature, analytic gradient) or in the potential. galax's
  `TriaxialNFWPotential` is flattened along the coordinate axes only and uses another quadrature variable.
- `kdk`: kick-drift-kick leapfrog with a fixed number of steps and a per-particle step. galax integrates with
  diffrax solvers, none of which is this scheme.
- The StreaMAX release schedule around galax's `FardalStreamDF`, and the per-epoch progenitor table.

Used from galax unchanged: `PlummerPotential`, `FardalStreamDF`, the potential base class (Hessian for the tidal
radius), `NFWPotential` and `TriaxialNFWPotential` as test references, a kpc / Gyr / Msun unit system (its G equals
StreaMAX's to the last digit).

## Not verified

- Nothing has run on a GPU: the memory probe, the per-GPU worker launch and the GPU throughput are untested.
- The resume path of `run` (no interruption happened) and `--cpu-workers` with several shards per worker or
  several cores per worker: the production run used 100 workers with one core and one shard each.
- Tracks are not bit-identical across batch sizes for about 3 % of the streams (see above); whether the
  population statistics of those streams depend on the batch size was not tested beyond the pilot comparison.
- The STRRINGS table used in the prior inspection was transcribed from the arXiv HTML by an automated reading and
  not checked row by row; the comparison of lengths and angles ignores how they are measured in the images.
- The slowdown of a worker when 100 run at once (4.6 to 2.5 streams/s) was measured, not explained.
- The StreaMAX comparison was run for the default (`density`, `none`) and once for `--halo-flattening potential
  --progenitor-mass-loss linear` (60 bins: z mean 0.00, std 0.96, max 2.37; orbit 1.4e-5). The other two
  combinations were not run.
- `--track-ordering gibbons14` and `--n-steps 100 --n-particles 10100` run (32 streams on CPU) but have no
  dedicated check beyond the ordering function being the one used in the StreaMAX comparison.

Data schema: see `README.md`.
