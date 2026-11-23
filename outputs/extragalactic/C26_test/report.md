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
- **STRRINGS sample used: the 32 observed tracks released with StreaMAX-fit** (commit `2e6a708`, `data/strrings/`,
  fetched into `data/strrings/` on first use; `theta`, `r`, `r_err` in 10 deg bins). NGC1084, NGC1121 and
  PGC021008 are excluded (decision of 2026-11-22, Giuseppe): they are the three of the 35 streams of
  arXiv:2508.02154 that StreaMAX-fit does not release, its export script dropping them as measurement failures.
  Each track is centred on its median angle, the StreaMAX-fit convention for where the model puts the
  progenitor, and then moved by at most half a bin onto the simulation grid.
- **The observed tracks against the C26 cuts**, measured exactly as for the simulations: 13 of 32 pass all three.
  28 pass the angle cut, 15 the 100 kpc length cut and 22 the radius cut (10 have a bin inside 10 kpc).
- **The observed streams are smaller than the simulated ones.** Mean track radius: median 25 kpc for the data
  (5th to 95th percentile 10 to 47) against 130 kpc for all simulations and 65 kpc (30 to 155) for the usable
  ones; 16 % of all simulated tracks and 27 % of the usable ones lie below 47 kpc. Median observed length 89 kpc,
  angular length 165 deg; 10 of 32 are longer than 270 deg against 2.3 % of the simulations passing the cuts.
  Median radial error of the data 4 %; radial contrast 1.0, like the simulations (1.2).
- **Nearest neighbours on the radial bins** (`strrings_neighbours_p1..4`, third panel of `strrings`). Distance:
  root-mean-square of r_sim / r_obs - 1 over the observed bins, over the 68,201 simulations with a resolved
  orbit, each bin needing at least 50 particles, the simulation or its mirror image (about half of the best
  matches are mirrored). Simulations covering every observed bin: 8,600 to 28,600 for tracks of 7 to 12 bins,
  540 to 5,500 for 14 to 17 bins, 37 to 700 for 22 bins and more. Within 10 % rms: none for 23 of the 32
  tracks, at most 58 for the others (NGC2769 58, NGC4686 28, ESO298-016 17, ESO079-003 11, NGC3041 10, one
  each for four more). Within 20 % rms: fewer than 100 for 28 tracks, none for 12. Best match: 5 to 10 % rms for
  9 tracks, 11 to 21 % for 14, above 21 % for 9. PGC938075 (radii below 1 kpc) is outside the prior altogether.
- **Where the neighbours come from** (`strrings_neighbour_parameters`; up to 100 per track within 20 % rms, 862 in
  all): median projected distance x0 = 38 kpc (prior 102), 3D distance 74 kpc (177), log10 M_halo 12.3; 77 % are
  bound and 63 % pass the three cuts. Their q, cos i and psi follow the prior.
- **Bound against unbound among the usable streams** (`gallery_bound`, `gallery_unbound`,
  `bound_vs_unbound_tracks`). Length, angle and radius are alike (median angle 160 against 150 deg, mean
  radius 58 against 70 kpc). The width is not: the median radial width of a bin relative to its radius is 0.08 for the
  bound (16th to 84th percentile 0.05 to 0.16) and 0.24 for the unbound (0.11 to 0.49). 82 % of the bound and
  28 % of the unbound (2,775 streams) are thinner than 0.15. Unbound progenitors have at most one pericentre
  (53 % none); the bound ones a median of two.
- **Corner view of the bound fraction** (`corner_bound_vs_all`, `corner_bound_vs_all_focus`): being bound is
  decided by halo mass against speed; no other pair of parameters shows structure.

## Particles, steps and progenitor state: tests for the next set (2026-11-22)

Generator options added for this (defaults unchanged, 19 / 19 checks pass in both configurations):
`--progenitor-mass-loss mixed` (each stream draws a surviving or a dissolved progenitor, stored in
`prog/dissolved`), `--alt-ordering` (both orderings stored), the progenitor's sky position in `prog/xy`, and the
Gibbons ordering zeroed on the progenitor. Six test sets with the same seed, mixed progenitor state, Gibbons
ordering in `track/` and the C26 one in `track_alt/`, 48 single-core workers on a machine shared with another
56-core job; `analysis/convergence.py`, `plots/convergence/convergence`, `data/convergence.json`.

| set | steps | particles | streams | wall time | streams per core-second | 1,000,000 streams on 100 cores |
|---|---|---|---|---|---|---|
| `conv_p10` | 99 | 10,000 | 3,840 | 67 s | 1.19 | 2.3 h |
| `conv_p30` | 99 | 30,000 | 3,840 | 151 s | 0.53 | 5.2 h |
| `conv_p50` | 99 | 50,000 | 3,840 | 236 s | 0.34 | 8.2 h |
| `conv_s399b` | 399 | 20,000 | 3,840 | 309 s | 0.26 | 10.7 h |
| `conv_s399` | 399 | 30,400 | 3,840 | 481 s | 0.17 | 16.7 h |
| `conv_s1999` | 1,999 | 32,000 | 1,920 | 992 s | 0.040 | 69 h |

**Particles.** Share of the simulations that the StreaMAX-fit likelihood would accept as sampled well enough
(at least 3 particles in every observed bin and model variance at most 1/9 of the data variance), median over
the observed tracks of each length:

| observed tracks of | 10,000 | 30,000 | 50,000 particles |
|---|---|---|---|
| 7 to 12 bins | 3.0 % | 6.0 % | 7.3 % |
| 14 to 17 bins | 0.23 % | 0.63 % | 0.91 % |
| 22 to 48 bins | 0.10 % | 0.23 % | 0.26 % |

Bins with at least 50 particles per stream: 10.5, 15.7, 18.9. Monte Carlo error of the bin median in the bins
valid at 10,000 particles: 0.9 %, 0.5 %, 0.4 % of the radius. 30,000 particles double the usable share for
2.2 times the cost; 50,000 add a fifth to a half more for 1.6 times the cost of 30,000.

**Steps.** Progenitor orbits with an energy error above 1 %:

| | 99 steps | 399 steps | 1,999 steps |
|---|---|---|---|
| all draws | 29.7 % | 15.4 % | 4.0 % |
| bound | 46.3 % | 24.3 % | 6.4 % |
| bound, mean track radius below 47 kpc (the size of the observed streams) | 59.8 % | 30.1 % | 6.5 % |

Change of the binned track (rms of the radius ratio minus 1 over the bins valid in both sets), share of streams
changing by more than 4 %, the median error of the data; 1,920 common streams:

| | resolved at 99 steps (1,349) | unresolved at 99 steps (571) | bound and small (306) | all |
|---|---|---|---|---|
| two particle realisations, 99 steps | 7.3 % | 64.7 % | 48.6 % | 27.6 % |
| two particle realisations, 399 steps | 9.1 % | 46.9 % | 32.3 % | 21.3 % |
| 99 against 399 steps | 6.2 % | 82.9 % | 43.9 % | 17.0 % |
| 99 against 1,999 steps | 7.5 % | 79.6 % | 40.4 % | 16.2 % |
| 399 against 1,999 steps | 5.0 % | 29.3 % | 24.2 % | 9.8 % |

- Streams resolved at 99 steps do not depend on the step number: every comparison gives the 5 to 9 % that two
  particle realisations give.
- Streams unresolved at 99 steps do: four in five move by more than 4 % (median change 28 to 35 %) when the
  steps are raised, and at 99 steps two realisations of the same stream disagree for two in three. These are
  30 % of the prior and 60 % of the bound streams as small as the observed ones.
- 399 steps are not enough for the small bound streams (30 % still unresolved, a quarter still moving by more
  than 4 % towards 1,999 steps); 1,999 steps leave 6.5 %. Convergence beyond 1,999 steps was not tested.
- The Gibbons ordering is the more stable of the two: 17 % of the tracks move by more than 4 % from 99 to 399
  steps, against 26 % in the C26 ordering. Surviving and dissolved progenitors behave alike (16.7 % and 17.2 %).

## galax's adaptive integrator against the leapfrog (2026-11-22)

The generator now installs galax from GitHub main (commit `4c29842`, 199 commits after 0.1.0; `xmmutablemap`
0.3.1, every other pin unchanged; a checkout is in `external/galax`, not tracked). The leapfrog path passes the
19 checks under it (StreaMAX track comparison: z mean -0.02, std 0.96; the deterministic checks are identical).

`--integrator galax` integrates everything with galax's `OrbitSolver`, i.e. diffrax Dopri8 with a PID step-size
controller (`--rtol 1e-7`, galax's default) and at most `--max-steps 1999` steps per solve:

- progenitor: from today back to -t_int and forward on the release times;
- release: galax `FardalStreamDF`, one leading and one trailing particle at each of n_particles / 2 times
  uniform over [-t_int, 0] (15,000 release times for 30,000 particles, against the 100 to 2,000 epochs of
  StreaMAX);
- particles: each solved from its release to today together with its own copy of the progenitor, as galax's
  `MockStreamGenerator` does with a `progenitor_potential`. The field (`StreamField` in `generate.py`) is
  written here because galax's generator returns final positions only and a constant progenitor mass; the
  unwrapped angle and the Gibbons coordinate need points along each path (`--n-steps` of them, uniform in
  time) and the 2026b model needs the mass ramp;
- a solve that would need more than `--max-steps` steps returns non-finite values: the particle is counted in
  `diag/n_bad`, and a progenitor orbit that fails sets the stream's `failed` status bit. Steps taken are stored
  in `diag/solver_steps_prog`, `solver_steps_mean`, `solver_steps_max`.

Test set `conv_galax`: the 1,920 prior draws of `conv_s1999`, 30,000 particles, mixed progenitor state, 64
single-core workers, batch 1, 1,186 s wall; `plots/convergence/galax_integrator`, `data/convergence.json`.

| | leapfrog 99 | leapfrog 399 | leapfrog 1,999 | galax adaptive |
|---|---|---|---|---|
| core-seconds per stream (about 30,000 particles) | 1.2 | 4.3 | 20.3 | 16.7 (shards: 7.7 to 37) |
| progenitor energy error above 1 % | 29.7 % | 15.4 % | 4.0 % | 0 % (largest 8e-7, median 3e-9) |
| fit's sampling guards accept, observed tracks of 7-12 / 14-17 / 22-48 bins | 6.0 / 0.63 / 0.23 % | - | 7.4 / 1.4 / 0.63 % | 7.2 / 1.1 / 0.57 % |

Steps taken by the adaptive solver, percentiles 50 / 90 / 99 / max over the streams: progenitor orbit (back and
forth) 40 / 605 / 2,240 / 3,998; mean over a stream's particles 14 / 181 / 713 / 1,999; largest of a stream
36 / 468 / 1,747 / 1,999. 12 streams (0.6 %) have a solve at the cap: 4 lose the progenitor orbit and count as
failed, 8 lose some particles.

Share of streams whose binned track differs from the galax one by more than 4 % (rms radius ratio minus 1):

| | resolved at 99 leapfrog steps | unresolved at 99 steps | bound, as small as the data | all |
|---|---|---|---|---|
| leapfrog 99 against galax | 9.9 % | 82.1 % | 44.2 % | 19.3 % |
| leapfrog 399 against galax | 7.9 % | 34.2 % | 27.2 % | 13.2 % |
| leapfrog 1,999 against galax | 7.8 % | 24.7 % | 22.0 % | 12.0 % |
| two leapfrog realisations at 399 steps (particle noise) | 9.1 % | 46.9 % | 32.3 % | 21.3 % |

The leapfrog converges towards the galax tracks as its steps are raised, and where the leapfrog is resolved
the two agree to within particle noise. The cost is that of 1,999 leapfrog steps: the adaptive solver takes far
fewer steps (a median of 14 per particle) but each Dopri8 step is 13 force evaluations on a two-body state,
steps are rejected, and the solves run one particle at a time. Not tested: a looser tolerance, a cheaper
solver (Tsit5, Dopri5), a larger step cap, and galax's own `MockStreamGenerator.run` end to end.

### The galax set used for the galleries and the prior inspection (`galax_pilot`)

3,840 prior draws, 30,000 particles, galax integrator (1,999-step cap), mixed progenitor state, Gibbons ordering
in `track/` and C26 in `track_alt/`, particles saved for the first 384; 64 single-core workers, 1,864 s. Analysed
with `--n-valid-min 300` (1 % of the particles, the equivalent of 100 of 10,000). Figures in
`plots/galax_pilot/` (the standard set) and `plots/galax_pilot/prior_inspection/`.

Unbound 35.5 %; failed 0.23 % (9 streams whose progenitor orbit hit the step cap); angular length > 90 deg
19.5 %; arc length > 100 kpc 30.6 %; radii in [10, 500] kpc 93.1 %; all three cuts 11.2 % (431 streams, 182 of
them bound, 325 with contiguous bins). With 3,840 simulations the neighbour search against the observed tracks
is thin (at most a few hundred simulations cover a short observed track, a handful a long one).

**Why 11 % pass the cuts here against 23.5 % in the production set.** On the same 1,920 draws, with a valid bin
defined as 1 % of the particles:

| | passes the three cuts |
|---|---|
| production: constant progenitor mass, 99 leapfrog steps, C26 ordering | 23.8 % |
| mixed progenitor state (half dissolved), 99 steps, C26 ordering | 17.3 % |
| the same at 1,999 steps | 15.6 % |
| the same with the galax integrator | 14.9 % |
| galax integrator, Gibbons ordering | 11.9 % |

So three things each remove a few points: dissolved progenitors make shorter streams, resolving the orbits
removes the scattered particles that made unresolved streams look long (the mean number of valid bins falls
from 8.2 to 7.1), and the Gibbons ordering populates fewer bins than the C26 one.

**The Gibbons ordering runs away much more often than the C26 one.** Share of streams with more than 1 % of
their particles beyond the +-1,440 deg grid: Gibbons 22.6 % (leapfrog 99), 14.0 % (leapfrog 1,999), 12.3 %
(galax); C26 2.3 %, 2.2 %, 5.0 %. For the galax set, by pericentre passages (0, 1, 2, ..., 6+): Gibbons 4, 2, 7,
11, 15, 31, 43 %; C26 0, 0, 0, 1, 1, 3, 27 %. Streams with no pericentre cannot wrap four times, so the Gibbons
cases there are failures of the unwrapping, not long streams. The code follows StreaMAX (checked against its
source: particles sorted by the summed radial offset from the progenitor, polar angle unwrapped along that
order), whose sort key does not order particles released at different times along the stream. The fits use
only the bins of the observed track, where this matters less; a prior-wide training set is more exposed. Both
orderings are stored, so the choice can be made later; the galleries above use the Gibbons one.

## The 500,000-stream training set `train_v1` (started 2026-11-22 21:24)

Settings, chosen with the user: galax integrator with **Dopri5**, tolerance **1e-6** (one order looser than
galax's default), at most **2,500** steps per solve; 30,000 particles; mixed progenitor state (`prog/dissolved`
says whether a stream was made with the mass ramp); C26 ordering in `track/`, Gibbons in `track_alt/`; shards of
250; particles saved for the first 10,000 streams. Command: `data/train_v1/launch.sh` (rerun it to resume);
112 single-core workers on cores 0-7, 9-56, 65-79, 81-105, 107-122, `nice -n 10`, detached with `nohup setsid`;
code at commit `a212b8a`.

Tests behind the settings, 960 prior draws each at 30,000 particles (`--solver` was added for them):

| step cap | core-seconds per stream | streams failed | bound | bound, x0 < 70 kpc | 6 or more pericentres |
|---|---|---|---|---|---|
| 400 | 9.5 | 10.2 % | 16.1 % | 27.7 % | 57.1 % |
| 1,000 | 10.1 | 2.4 % | 3.8 % | 11.2 % | 13.5 % |
| 2,500 | 9.8 | 0.2 % | 0.3 % | 1.0 % | 1.2 % |

The cap does not change the cost: a failed stream still runs every particle to the cap, and most streams need
far fewer steps (mean per particle: median 18, 90th percentile 233, 99th percentile 865 at the 2,500 cap). The
saving against Dopri8 at 1e-7 (16.7 core-seconds per stream) comes from the solver and the tolerance. With
Dopri5 at 1e-6 the energy error of the streams that finish is at most 6e-5 and their C26 tracks agree with
the Dopri8 ones to well under 1 % (same particle draws). 99 or 399 path samples for the unwrapped angle make
no difference. The user first asked for a cap of 400 and chose 2,500 on these numbers.

Expected: about 12 hours for 500,000 streams (4.9 million core-seconds on 112 cores), 1.5 GB of tracks and
3.6 GB of particles. *Results, timings and the prior predictive checks are added when the run is complete.*

## Inference pipeline `scripts/extragalactic/train.py` (2026-11-23, written while `train_v1` was running)

Asked for: BayesFlow added to `pyproject.toml`; a training script with a normalizing flow, the JAX backend,
BayesFlow's offline dataset and augmentation hook, augmentations written in JAX (GPU when there is one), a log
file, a checkpoint at every best epoch, the likelihood written out for importance sampling, and a setup as
close as possible to 2026b. `bayesflow==2.1.0` (Keras 3.15, JAX backend) is pinned in `pyproject.toml`.

### What 2026b fixes, and how the script follows it

| item | 2026b (paper and StreaMAX-fit at `2e6a708`) | `train.py` |
|---|---|---|
| parameters | 14: `logM Rs dirx diry dirz logm rs x0 z0 vx0 vy0 vz0 time sig` | the same 14 in the same order |
| prior | table A1 | read from the configuration stored in the shards; identical to table A1 |
| likelihood | eq. 4: Gaussian per bin, variance sigma_data^2 + sigma_sys^2, sigma_sys ~ U(0, 25) kpc | mock radii are drawn from exactly this; `log_likelihood` is eq. 4 |
| model track | median radius per 10 deg bin (`get_track_2D`), 16-84 half-width for the model noise | `r_med`, `r_hw`, `count` of the shards |
| data frame | angle minus the median angle of the track; progenitor at angle 0 | the same |
| guard 1 | at least 3 model particles in every observed bin (of at least 10,000) | `--n-min 10` of 30,000 |
| guard 2 | model variance at most 1/9 of the data variance | `--var-ratio 9`, at the 30,000 particles of the set (below) |
| progenitor | mass ramped to zero if none is visible, constant if one is | `prog/dissolved` is a condition of the network |
| ordering | Gibbons et al. 2014 | `--ordering gibbons14` trains on `track_alt/`; default is the C26 ordering of `track/` |

Table A1 of 2026b is the prior the simulations already use (R_s in 10-25 kpc, r_s in 1-5 kpc, z0 > 0,
velocities in km/s). The public StreaMAX-fit code differs from that table in four places (R_s from 5 kpc,
r_s to 3 kpc, z0 of either sign, velocities of 250 in kpc/Gyr); the script follows the paper.

### Design decisions

- **Latent coordinates.** The flow is trained on zeta = Phi^-1(F(parameter)), F the prior cdf. The prior is
  N(0, I) in zeta and F(parameter) is dynesty's unit cube, so importance weights are
  N(zeta; 0, I) L / q with no Jacobian.
- **No rotation or mirror augmentation (confirmed with the user, 2026-11-23).** The prior keeps one copy of
  every set of equivalent systems: y0 = 0 with x0 > 0 removes rotations about the line of sight, vy0 > 0 the
  left-right mirror, z0 > 0 the front-back mirror, dirz >= 0 the sign of the halo axis. A rotated or mirrored
  example is identical or outside the prior, and flipping the track while keeping vy0 > 0 would swap the
  leading and trailing arms, which are only approximately alike. The only randomness added to a simulation is
  the observation: which bins are seen, with which errors, and the noise drawn from the likelihood.
- **Progenitor at the centre of the track**, as the paper assumes (`--prog-position centre`, the default).
  My first version drew the progenitor at a random bin edge of the window and gave its angle to the network in
  half of the examples; that stays as `--prog-position random` for a later step in which the position is
  inferred or passed as a condition. The centre costs no simulations: the share of streams that cover a
  24-bin track is 2.9 % centred against 2.5 % with a random position, because simulated streams extend about
  equally on both sides of the progenitor.
- **Windows without a hidden selection.** A mock track is a window of S bins with l bins left of the
  progenitor, drawn from a fixed p0(S, l) and restricted to windows the simulated stream covers. Drawing every
  simulation once per epoch would then favour simulated streams barely longer than the data, a factor the
  paper's posterior does not have. Instead a simulation is drawn with probability proportional to Z, the
  chance that a random window is covered (`SelectionDataset`); this is rejection sampling of p0, so the training
  distribution is prior x p0 x 1[model covers the data].
- **Track lengths.** p0(S) is set so that the lengths of the accepted examples are log-uniform between 6 and 64
  bins: 41 % have 24 bins or more before the model-noise guard and 35 % after it (observed tracks: 44 %). With p0 itself log-uniform only 3 % of the examples
  would have been that long, because few simulated streams are. S is observed, so this choice does not enter
  the posterior; it decides where the training effort goes.
- **Model-noise guard on, as in the paper.** Measured on 75,000 streams, with the guard off the largest ratio of
  model noise to data error along a mock track has a median of 1.6 (90th percentile 16): half of the tracks
  have a bin where the noise of the 30,000-particle model exceeds the data error. With the guard that ratio is
  at most 0.33. The price is fewer simulations for long tracks: the streams behind examples of 24 bins or more
  fall from 4,600 to 2,400 per 75,000. Because the comparable bins now depend on the data errors, simulations
  are drawn per stratum of the error level (8 strata) and a weight in the loss corrects for the scatter of the
  errors along the track; 89 % of the draws are used and the weights have an efficiency of 0.65 (0.29 before
  the strata). `--var-ratio 0` switches the guard off; all weights are then exactly 1.
- **Particle number against the paper.** The paper runs max(10,000, 1,500 per data point) particles, so its
  guard is looser than ours for tracks longer than 20 points (69,000 particles for 46 points) and stricter for
  shorter ones. The set has 30,000 for every stream; the guard is applied at that number.
- **Errors and gaps.** sigma_data is drawn independently of the parameters: track level log-uniform in
  0.1-5 kpc, log-normal scatter of 0.2-1.0 along the track (observed: medians 0.2-2.9 kpc, scatter 0.4-0.95).
  30 % of mock tracks lose interior bins (10 of the 32 observed tracks have gaps of up to 3 bins).
- **Usable simulations.** Dropped: failed integrations, streams that lost more than 1 % of their particles,
  and streams with fewer than 6 comparable bins around the progenitor. Unbound progenitors stay (raw prior).
  64.5 % of the first 50,000 streams are usable.

### Summary network

BayesFlow 2.1 offers `TimeSeriesTransformer`, `TimeSeriesNetwork` (convolutions + GRU), `SetTransformer`,
`DeepSet`, `ConvolutionalNetwork` and fusion variants. Only the two transformers take a padding mask.
`TimeSeriesTransformer` also takes an explicit time axis, which is the angle here, so missing bins and tracks
of 7 to 49 bins need no special handling. It is the default. `timeseries` and `mlp` (a flattened track) are
selectable for comparison; they see the observed flag as an input channel instead.

First comparison, 2026-11-23: the first 200 shards (32,264 usable streams, 1,613 of them held out with 2 fixed
mock tracks each), batch 256, 119 steps per epoch, 4 CPU cores per run, one seed, guard on, progenitor at a
random position (the runs started before the centre became the default). The loss is in nats; 19.87 is a
network that learnt nothing.

| summary network | best held-out loss | at epoch | training loss there | held-out loss at the end | seconds per epoch |
|---|---|---|---|---|---|
| `transformer` | 16.78 | 5 of 6 | 14.56 | 16.82 | 187 |
| `timeseries` | 17.12 | 5 of 16 | 14.86 | 18.41 | 62 |
| `mlp` | 17.67 | 4 of 80 | 16.15 | 24.64 | 11 |

The transformer reaches the lowest held-out loss and is the default. All three overfit on this subset: the
training loss runs 2 nats or more below the held-out loss within a few epochs, and the MLP ends worse than a
network that learnt nothing. The cause is the small set combined with drawing the few long streams many times
per epoch. The full set has 16 times more streams; whether that is enough is the first thing to read off the
`history.csv` of a full run. The ranking rests on one seed and 1,613 held-out streams.

On CPU the transformer is the slow one: 0.6 steps per second at batch 256 on 4 cores, against 12 for the MLP.
This machine has no GPU. A full epoch is about 610 steps at batch 512, which at the measured rate is about
half an hour for the transformer on 4 cores (not measured on more cores) and under two minutes for the MLP.

### Checks (`train.py check`, 200 shards, default settings)

All pass: latent transform round trip (relative error 2e-12); prior is N(0, 1) in every latent coordinate
(KS, smallest p 0.009 of 13); q recovered from the orientation vector (1e-7); windows inside the covered run;
every observed bin passes both guards; angles centred on the median; mock radii scatter with chi-square per
point 1.0006; the real-track encoder reproduces the mock inputs under a sky rotation (4e-6); `log_likelihood`
equals eq. 4 written by hand (3e-5); training batches have the shapes the estimator expects; streams are
drawn in proportion to Z; the importance weights recover an analytic posterior mean; all 32 observed tracks
encode on the 64-token grid and lie in the training range (spans 7-49 bins; PGC938075 with the paper's radius
scaling of 100). The same checks pass with `--var-ratio 0` and with `--prog-position random`.

Numbers the checks print: 17 % of the draws have no window and get weight 0 (weight efficiency 0.58);
weighted share of examples with 24 bins or more 35 %, with 40 or more 16 %; the 1 % of streams with the
largest Z give 24 % of the examples (each is drawn about 24 times per epoch, with new errors and noise each
time); share of streams that cover a centred track of S bins: 21 % (8), 5.9 % (16), 2.9 % (24), 2.0 % (32),
0.6 % (48), 0.2 % (64). For low-noise tracks of 12 bins or more the most likely progenitor offset is the true
one in 95 % of cases. The augmentation takes 67 ms per batch of 512 on 4 CPU cores.

Figure: `plots/train_v1/npe/mock_observations.png` (eight mock tracks with their model track, four observed
tracks in the same frame).

### Not done yet

- No model has been trained on the full set; the runs above are smoke tests on 32,000 usable streams.
- Simulations for long tracks are few (0.6 % of the streams cover 48 bins), and they are drawn
  many times per epoch. The held-out loss is the guard against memorising them; whether 500,000 streams are
  enough for the longest observed tracks is not known yet.
- Importance sampling needs the tracks of posterior samples, which means running `generate.py` on given
  parameters. `train.py sample` writes the samples and their log q; the simulation step is not written.
- The progenitor sits on a bin edge of the stored tracks, so its offset is a multiple of 5 deg; tracks with an
  odd number of points and the progenitor at the centre fall between two trained offsets.
- The mock errors are independent between bins; real errors may be correlated along a track.

## Figures (`plots/`; the pilot's are in `plots/pilot/`)

`prior_marginals`, `energy`, `binned_values`, `gallery` (PDF, 3 pages; `gallery_p1..3.png`), `noise_examples`,
`streamax_comparison`, `energy_conservation`; in `plots/prior_inspection/`: `funnel`, `prior_bound_vs_all`,
`corner_bound_vs_all`, `corner_bound_vs_all_focus`, `acceptance`, `acceptance_maps`, `integrator`, `observable`,
`strrings`, `strrings_neighbours_p1..4`, `strrings_neighbour_parameters`, `gallery_bound`, `gallery_unbound`,
`bound_vs_unbound_tracks`; in `plots/convergence/`: `convergence`, `galax_integrator`; the same standard and
prior-inspection figures for the galax set in `plots/galax_pilot/`. Each as PDF and PNG; all were looked at.

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

- (written before `train_v1`) No production set exists in the new configuration (mixed progenitor state, Gibbons ordering, more particles or
  steps): only the six test sets of 1,920 to 3,840 streams. The neighbour comparison with the observed tracks
  was not repeated for it. Convergence was not tested beyond 1,999 steps, and the test timings were taken on a
  machine shared with another job.
- Nothing has run on a GPU: the memory probe, the per-GPU worker launch and the GPU throughput are untested.
- The resume path of `run` (no interruption happened) and `--cpu-workers` with several shards per worker or
  several cores per worker: the production run used 100 workers with one core and one shard each.
- Tracks are not bit-identical across batch sizes for about 3 % of the streams (see above); whether the
  population statistics of those streams depend on the batch size was not tested beyond the pilot comparison.
- The observed tracks are compared in the StreaMAX-fit convention (centred on the median angle, no shift along
  the track); the nearest neighbours were not searched over other positions of the progenitor. The stream widths
  (one rug in `bound_vs_unbound_tracks`) come from table 4 of arXiv:2508.02154, transcribed from the arXiv HTML
  and not checked row by row.
- The production set uses a constant progenitor mass (C26); Chemaly et al. 2026b fit the STRRINGS tracks with the
  mass ramp (`--progenitor-mass-loss linear`), which was not generated.
- The slowdown of a worker when 100 run at once (4.6 to 2.5 streams/s) was measured, not explained.
- The StreaMAX comparison was run for the default (`density`, `none`) and once for `--halo-flattening potential
  --progenitor-mass-loss linear` (60 bins: z mean 0.00, std 0.96, max 2.37; orbit 1.4e-5). The other two
  combinations were not run.
- `--track-ordering gibbons14` and `--n-steps 100 --n-particles 10100` run (32 streams on CPU) but have no
  dedicated check beyond the ordering function being the one used in the StreaMAX comparison.

Data schema: see `README.md`.
