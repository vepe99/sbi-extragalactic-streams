# sbi-extragalactic-streams

Simulation-based inference of dark-matter halo shapes from extragalactic stellar streams.

This repository holds the forward model and the prior-predictive dataset: **100,000 stellar-stream tracks drawn
from the prior** of Chemaly et al. 2026a ([arXiv:2601.15373](https://arxiv.org/abs/2601.15373), "C26") and 2026b
([arXiv:2607.05510](https://arxiv.org/abs/2607.05510)), reproducing their code
[StreaMAX](https://github.com/David-Chemaly/StreaMAX) at commit `9fd520e`. The observational target is the
STRRINGS sample of streams around nearby galaxies ([arXiv:2508.02154](https://arxiv.org/abs/2508.02154)); the
dataset is the starting point for testing whether an amortized posterior estimator can be trained on this model.

- `scripts/extragalactic/generate.py`: prior, simulator, tests, pilot, production run, diagnostics
- `scripts/extragalactic/plot.py`: figures
- `scripts/extragalactic/train.py`: amortised posterior estimation with BayesFlow (mock observations, the
  likelihood, training, posterior samples for the observed tracks); see "Amortised inference" below
- `scripts/extragalactic/inspect_prior.py`: prior inspection ahead of amortized inference (usable fraction, what the selection
  cuts do to the prior, where the integrator fails, the prior predictive against the STRRINGS streams)
- `scripts/extragalactic/prior_predictive.py`: prior predictive checks of a set (finished or partial) against the observed
  STRRINGS tracks
- `scripts/extragalactic/convergence.py`: how many particles and leapfrog steps a training set needs. It reads the `conv_*`
  test sets, which were deleted on 2026-11-23; its figures and `data/convergence.json` are kept, and the report
  lists the settings to regenerate the sets
- `outputs/extragalactic/C26_test/README.md`: data schema
- `outputs/extragalactic/C26_test/report.md`: what was tested and run, and every StreaMAX detail looked up

Generated data (`outputs/**/data/`) and figures (`outputs/**/plots/`) are not tracked. On disk since
2026-11-23: `data/c26_100k/` (the 100,000-stream set, merged into one file; read it with `--subdir c26_100k`),
`data/train_v1/` (the 500,000-stream training set), `data/strrings/` and `data/tests/`.

## Decisions

- **STRRINGS sample (2026-11-22, Giuseppe):** NGC1084, NGC1121 and PGC021008 are excluded from every comparison
  and from inference; 32 of the 35 streams are used. This is the sample of 2026b (its section 2): PGC021008 is
  bright enough to have been modelled away with the host light, NGC1084 could not be fitted and may be a spiral
  arm, and NGC1121 is a three-loop stream left to a dedicated study. Encoded as `STRRINGS_EXCLUDED` in
  `scripts/extragalactic/inspect_prior.py`.
- **Observed tracks:** the 32 tracks released with [StreaMAX-fit](https://github.com/David-Chemaly/StreaMAX-fit)
  at commit `2e6a708` (MIT licence). `scripts/extragalactic/inspect_prior.py` fetches them into
  `outputs/extragalactic/C26_test/data/strrings/` on first use; they are not tracked here.

## The five commands

Only [uv](https://docs.astral.sh/uv/) is needed. Both scripts carry their environment in a PEP 723 header; uv
fetches Python 3.12 and every package (network access and git are required: StreaMAX is installed from GitHub
at a pinned commit, and so is galax, from its main branch).

```bash
uv run --script scripts/extragalactic/generate.py test --cpu --n-test-streams 96   # 19 checks, under a minute
uv run --script scripts/extragalactic/generate.py pilot --cpu                      # 1,000 streams -> data/pilot/
uv run --script scripts/extragalactic/generate.py run --cpu --batch-size <N>       # 100,000 streams -> data/
uv run --script scripts/extragalactic/generate.py analyze                          # data/analysis.h5 + summary table
uv run --script scripts/extragalactic/plot.py                                      # plots/ (--only tests for the test figures)
```

`run` resumes: rerun the same command after an interruption and finished shards are skipped. Stream `k` depends
only on `(seed, k)`, so batch size, shard size and the number of workers do not change the results.

On a many-core machine one JAX process does not use the cores well. `run --cpu` therefore takes
`--cpu-workers N` (N worker processes, each with its own block of shards) and `--cpu-cores 0-47,64-111` (the
cores shared out among them, in `taskset` syntax; each worker is pinned to its group). Without these two flags
`run --cpu` starts a single unpinned worker, as before.

Options beyond the C26 defaults, all in the `Config` dataclass:

| flag | effect |
|---|---|
| `--n-steps N --n-particles M` | leapfrog steps and particles; `M` must be an even multiple of `N + 1` (StreaMAX releases `M / (N + 1)` particles at each of `N + 1` epochs) |
| `--progenitor-mass-loss mixed --p-dissolved 0.5` | each stream draws its progenitor state: still there (constant mass, C26) or dissolved (mass ramped away, 2026b); stored in `prog/dissolved` |
| `--solver dopri5` | with `--integrator galax`: the diffrax solver (`dopri8` is galax's default; also `tsit5`) |
| `--integrator galax --max-steps 1999 --rtol 1e-7` | integrate the progenitor and every particle with galax's adaptive integrator (diffrax Dopri8) instead of the fixed-step leapfrog; one leading and one trailing particle per release time; use `--batch-size 1` |
| `--track-ordering gibbons14` | the ordering of 2026b in `track/` (default: the C26 per-particle unwrapping) |
| `--alt-ordering` | also store the track in the other ordering, in `track_alt/` |

Each shard also stores the progenitor's sky position (`prog/xy`), which is `(x0, 0)` in the frame of the stored
track; see the data schema. Sets made with such options go in their own `--subdir`.

The resolved environments are pinned in `scripts/extragalactic/generate.py.lock` and `plot.py.lock`
(`uv lock --script <file>`); `uv run --script` uses them.

## CPU and GPU

The dataset in this repository was generated on CPU: pass `--cpu` to every command on a machine without a GPU.

The GPU path is kept. Without `--cpu`, `test` and `pilot` take one free GPU and `run` starts one worker per free
GPU (free = no compute process on the card, found with `nvidia-smi` and `autocvd`; `--gpus <uuid,uuid>` forces a
choice; under Slurm the allocation's `CUDA_VISIBLE_DEVICES` is used). To switch to GPUs:

1. keep `jax[cuda12]==0.12.1` in the header of `generate.py` (restore it if it was changed to `jax==0.12.1`),
2. drop `--cpu` from the commands.

**The GPU path has never been run.** The memory probe that picks the batch size and the per-GPU worker launch
are untested.

## Project environment

`pyproject.toml` and `uv.lock` describe the same pinned environment (CPU-only jax) for interactive work and for
the analysis and training code that will follow: `uv sync`, then `uv run python ...`. Add `--extra gpu` for the
CUDA 12 wheels. The five commands above do not use it; they run from the script headers.

```bash
uv run python scripts/extragalactic/inspect_prior.py --subdir c26_100k   # plots/c26_100k/prior_inspection/ + data/c26_100k/prior_inspection.json
```

## Amortised inference

`scripts/extragalactic/train.py` trains a normalizing flow on the 14 parameters of 2026b (table A1, in the order
of StreaMAX-fit: `logM Rs dirx diry dirz logm rs x0 z0 vx0 vy0 vz0 time sig`), conditioned on a summary of the
track. It runs in the project environment with the JAX backend of Keras:

```bash
uv run python scripts/extragalactic/train.py check     # tests of the transforms, mock observations, likelihood
uv run python scripts/extragalactic/train.py train     # -> outputs/extragalactic/C26_test/models/<name>/
uv run python scripts/extragalactic/train.py sample    # posterior samples + log q for the 32 observed tracks
```

Without `--cpu` a free GPU is taken with autocvd when the machine has one. A run directory holds `train.log`
(a line every `--log-every` steps and one per epoch), `history.csv`, `last.keras` (every epoch), `best.keras`
(every epoch that improves the held-out loss), `best.json` and `config.json`. `--resume` continues an
interrupted run; the learning-rate schedule is the one fixed at the first start.

| option | default | meaning |
|---|---|---|
| `--summary` | `transformer` | summary network: `transformer` (attention over the bins, the angle as explicit time axis), `timeseries` (convolutions + GRU), `mlp` (flattened track) |
| `--ordering` | `c26` | track to train on: `c26` or `gibbons14` (the ordering of 2026b, stored as `track_alt/`) |
| `--n-min` | 10 | model particles a bin needs to be compared with data (3 of 10,000 in the paper) |
| `--var-ratio` | 9 | the paper's model-noise guard: variance of the model track at most 1/9 of the data variance; 0 switches it off |
| `--span-prior` | `loguniform` | distribution of the track length (6 to 64 bins) of the training examples |
| `--prog-position` | `centre` | progenitor at the middle of the mock track, as 2026b assumes; `random` puts it on any bin edge of the track and gives its angle to the network in a fraction `--p-known` of the examples |
| `--prog-mode` | `centre` | `sample`: progenitor at the middle of the track (2026b) or `free` (marginalised) |
| `--sig-sys-power` | 1 | training draws of sigma_sys: 1 is the prior U(0, 25) kpc; larger concentrates them at small values |

The module docstring states the frame conventions, the mock observation model and the likelihood. In short:

- **Parameters** are trained in latent coordinates in which the prior is a standard normal (the normal quantile
  of dynesty's unit cube), so importance weights need no Jacobian.
- **No rotation or mirror augmentation.** The prior of the paper keeps one copy of every set of equivalent
  systems (y0 = 0, x0 > 0, vy0 > 0, z0 > 0, dirz >= 0), so a rotated or mirrored example is either identical or
  outside the prior. Tracks are centred on their median angle, as StreaMAX-fit does, and the progenitor sits
  at the centre of the mock track, as the paper assumes (decided 2026-11-23).
- **Each training example** is a new mock observation of a stored simulation: a random window of the track,
  random gaps, radial errors drawn over the observed range, sigma_sys from its prior, and radii scattered as
  N(model, sigma_data^2 + sigma_sys^2). Simulations are drawn in proportion to the chance that they cover a
  random window, so the training distribution is the prior times "the model covers the data" (the paper's two
  guards), with no preference for short or long simulated streams. 35 % of the examples are 24 bins or longer,
  against 44 % of the observed tracks.
- **The likelihood** of 2026b (eq. 4 with its two guards) is `log_likelihood`; `importance_weights` turns
  posterior samples, their log q and their likelihood into weights and an effective sample size. Evaluating
  the likelihood of new samples needs their tracks from `generate.py`; that step is not written yet.

## Hardware

Generated on a dual-socket AMD EPYC 7713 (2 x 64 cores, 256 threads), 1 TB RAM, Ubuntu 24.04, no GPU:
100 single-core workers, 100,000 streams in 435 s (230 streams/s), 138 GB of memory at the peak.

```bash
uv run --script scripts/extragalactic/generate.py run --cpu --batch-size 16 --cpu-workers 100 --cpu-cores 0-49,64-113
```

Tracks are bit-identical across thread counts but not across batch sizes (about 3 % of the streams, those whose
orbit the 99-step integrator does not resolve), so keep `--batch-size 16` to reproduce this set exactly.
Run details, timings and results: `outputs/extragalactic/C26_test/report.md`.
