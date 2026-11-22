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
- `outputs/extragalactic/C26_test/README.md`: data schema
- `outputs/extragalactic/C26_test/report.md`: what was tested and run, and every StreaMAX detail looked up

Generated data (`outputs/**/data/`) and figures (`outputs/**/plots/`) are not tracked.

## The five commands

Only [uv](https://docs.astral.sh/uv/) is needed. Both scripts carry their environment in a PEP 723 header; uv
fetches Python 3.12 and every package (network access and git are required: StreaMAX is installed from GitHub
at a pinned commit).

```bash
uv run --script scripts/extragalactic/generate.py test --cpu --n-test-streams 96   # 19 checks, under a minute
uv run --script scripts/extragalactic/generate.py pilot --cpu                      # 1,000 streams -> data/pilot/
uv run --script scripts/extragalactic/generate.py run --cpu --batch-size <N>       # 100,000 streams -> data/
uv run --script scripts/extragalactic/generate.py analyze                          # data/analysis.h5 + summary table
uv run --script scripts/extragalactic/plot.py                                      # plots/ (--only tests for the test figures)
```

`run` resumes: rerun the same command after an interruption and finished shards are skipped. Stream `k` depends
only on `(seed, k)`, so batch size, shard size and the number of workers do not change the results.

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

## Hardware

Generated on a dual-socket AMD EPYC 7713 (2 x 64 cores, 256 threads), 1 TB RAM, Ubuntu 24.04, no GPU.
Run details, timings and results: `outputs/extragalactic/C26_test/report.md`.
