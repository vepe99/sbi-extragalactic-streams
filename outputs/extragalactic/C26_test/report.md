# C26 extragalactic stream dataset: report

**Status (2026-11-22): code written and tested on CPU only. The GPU pilot (step 5) and the 100,000-stream
production run (step 6) have NOT been run** (no GPU was free). The numbers below come from the CPU test suite
and a 1,000-stream CPU pilot (`data/pilot_cpu/`). Sections marked *pending* are filled after the production run.

## What was run

| step | command | device | result |
|---|---|---|---|
| tests (step 4) | `generate.py test --cpu --n-test-streams 96` | 16 CPU cores | 19 / 19 checks pass, 28 s |
| CPU pilot | `generate.py pilot --cpu --subdir pilot_cpu --shard-size 250 --n-save-particles 1000` | 16 CPU cores | 1,000 streams, 73 s |
| GPU pilot (step 5) | `generate.py pilot` | - | *pending* |
| production (step 6) | `generate.py run` | - | *pending* |

## Timings

CPU, 16 cores (`taskset -c 0-15`), batch 8: compile 3.8 s, **13.6 streams/s**, i.e. about 2.0 h for 100,000 streams
on 16 cores. No GPU timing yet; the worker picks its batch size from a memory probe (8 streams, peak allocation,
scaled to 60 % of the card, at most 256).

## Tests (step 4), `data/tests/test_report.json`

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
| StreaMAX binned tracks vs this code, same ordering, 80 bins | z mean 0.07, std 0.96, max 2.76 |
| StreaMAX symmetry axis for the stored `c26_vec` | equals `axis` |

Figures: `plots/streamax_comparison.png`, `plots/energy_conservation.png`.

**Energy conservation of the progenitor orbit** (200 prior draws, 126 bound), relative energy change over t_int,
percentiles 50 / 90 / 99 / max:

| steps | all | bound | unbound |
|---|---|---|---|
| 99 | 1.7e-4 / 1.33 / 5.0 / 15.5 | 6.0e-3 / 1.83 / 6.0 / 15.5 | 3.5e-5 / 1.8e-4 / 4.5e-3 / 7.6e-3 |
| 100 | 1.6e-4 / 0.90 / 15 / 38 | 4.9e-3 / 1.37 / 19 / 38 | 3.4e-5 / 1.7e-4 / 4.4e-3 / 7.4e-3 |
| 2000 | 2.0e-7 / 8.8e-5 / 0.54 / 1.05 | 8.0e-7 / 3.0e-3 / 0.74 / 1.05 | 8.5e-8 / 4.3e-7 / 1.1e-5 / 1.8e-5 |

About 100 leapfrog steps do not resolve the deeply bound part of the prior (massive halo, small pericentre, many
orbits inside t_int): for roughly a third of the bound draws the energy error is of order one and the position at
-t_int differs from the 2,000-step orbit by more than the orbit's size. These orbits are an integrator artefact,
their leapfrog map is chaotic (round-off changes individual particles by O(r)), and they are reproduced here on
purpose, because the C26 / 2026b posteriors are computed with the same integrator. `diag/dE_rel` flags them.

## Step-7 table from the 1,000-stream CPU pilot (*replaced after production*)

| fraction of streams | all (n = 1000) | bound only (n = 637) |
|---|---|---|
| unbound (E >= 0) | 36.30 % | 0.00 % |
| bound, no completed radial orbit in t_int | 21.90 % | 34.38 % |
| failed / NaN | 0.00 % | 0.00 % |
| at least one valid bin | 100.00 % | 100.00 % |
| angular length > 90 deg | 40.70 % | 43.49 % |
| arc length > 100 kpc | 66.70 % | 69.07 % |
| valid radii in [10, 500] kpc | 79.50 % | 69.86 % |
| all three cuts | 24.30 % | 21.51 % |

Other flags (all streams): no non-finite particle in any stream, tidal radius never undefined, particles outside
the phi grid 6.7 %, projected progenitor angle not monotonic 14.3 %, all three cuts and contiguous valid bins 13.0 %.
Of the streams passing all three cuts 43.6 % are unbound. Orbit not resolved by 99 steps (energy error > 1 %):
30.3 % of all, 46.6 % of the bound, 23.1 % of those passing the cuts.

| log10 M_halo | 11.0-11.5 | 11.5-12.0 | 12.0-12.5 | 12.5-13.0 | 13.0-13.5 | 13.5-14.0 |
|---|---|---|---|---|---|---|
| unbound | 96.4 % | 72.3 % | 40.2 % | 5.5 % | 0.0 % | 0.0 % |

| speed today [km/s] | 0-100 | 100-200 | 200-300 | 300-400 | 400-500 | 500-700 | >700 |
|---|---|---|---|---|---|---|---|
| unbound | 0.0 % | 19.0 % | 19.2 % | 30.9 % | 45.2 % | 49.3 % | 72.3 % |

With n = 1000 these fractions carry a statistical error of about 1.5 percentage points.

## Figures (`plots/pilot_cpu/` for now, `plots/` after production)

`prior_marginals`, `energy`, `binned_values`, `gallery` (PDF, 3 pages; `gallery_p1..3.png`), `noise_examples`;
in `plots/`: `streamax_comparison`, `energy_conservation`.

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
- The StreaMAX comparison was run for the default (`density`, `none`) and once for `--halo-flattening potential
  --progenitor-mass-loss linear` (60 bins: z mean 0.00, std 0.96, max 2.37; orbit 1.4e-5). The other two
  combinations were not run.
- `--track-ordering gibbons14` and `--n-steps 100 --n-particles 10100` run (32 streams on CPU) but have no
  dedicated check beyond the ordering function being the one used in the StreaMAX comparison.

Data schema: see `README.md`.
