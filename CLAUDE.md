# Instructions for Claude in this repository

## Git

- **Do not commit unless Giuseppe says so.** Leave changes in the working tree and say what changed. A go-ahead
  covers the commit it was given for, not later ones.
- Do not push unless asked.

## GPUs

- **On a machine with GPUs, use only GPUs that are free.** Never start a job on a card another process is
  using, and never set `CUDA_VISIBLE_DEVICES` by hand to a card that has not been checked.
- Pick the card with `autocvd` before JAX (or Keras) is imported. It waits for a free card and sets
  `CUDA_VISIBLE_DEVICES`:

  ```python
  from autocvd import autocvd
  autocvd(num_gpus=1)   # before `import jax`
  ```

  `scripts/extragalactic/sbi/config.py` (`pick_device`) and `scripts/extragalactic/generate.py` (`free_gpus`) already
  do this; new GPU code must do the same.
- To see who is using which card: `uv run gpustat`.
- Under Slurm, use the allocation as given (`CUDA_VISIBLE_DEVICES` set by the scheduler).
- On a machine without an NVIDIA driver the scripts fall back to the CPU; pass `--cpu` to force it.

Both `autocvd` and `gpustat` are in `pyproject.toml`.

## Project facts worth knowing

### Progenitor positions of the STRRINGS streams (searched 2026-11-23)

- The released track files (`outputs/extragalactic/C26_test/data/strrings/`, from StreaMAX-fit at `2e6a708`) hold
  no progenitor position and no visible-progenitor flag. The released code centres every track on its median
  angle and puts the progenitor there. Chemaly et al. 2026b (arXiv:2607.05510) says a subset of streams has a
  visible progenitor but names none.
- The STRRINGS catalogue paper (arXiv:2508.02154, section 5.3 and table 3) lists six streams with a candidate
  dwarf progenitor and its sky coordinates. All six are among our 32:

  | host | RA (deg) | Dec (deg) | note |
  |---|---|---|---|
  | ESO356-012 | 42.186 | -36.526 | no redshift for the dwarf |
  | NGC0804 | 30.996 | 30.817 | dwarf at the host's redshift |
  | NGC5513_GROUP | 213.306 | 20.424 | same redshift; identification called uncertain |
  | PGC1092512 | 337.795 | -2.407 | no redshift for the dwarf |
  | PGC430221 | 304.791 | -53.414 | no redshift for the dwarf |
  | UGC09239 | 216.287 | 13.753 | no reliable redshift |

- These are sky coordinates, not positions along the track. Converting them needs the host centre (not in the
  files) and the orientation of the `x`, `y` frame of the track files (not checked). Not done.
- The host position angle `PA` in the files is the disk's, used by the paper only after the fits (its section
  5.2). It is not a model input and is not passed to the network.
- **Decision (Giuseppe, 2026-11-23): the progenitor position is not passed to the network for now**
  (`p_known: 0`). It is a 15th parameter, inferred as a nuisance. The mechanism to pass it stays in the code
  (`p_known > 0` at training, `prog_given: true` at evaluation; see `scripts/extragalactic/sbi/mock.py`).

### SBI runs

- One YAML file drives training and both evaluations: `outputs/extragalactic/C26_test/npe_v2.yaml`.
  See `scripts/extragalactic/sbi/README.md`.
- Training speed (measured 2026-11-23, table in that README): the step time is the summary transformer's and
  grows with `embed_dim` and `n_blocks`, not with `n_heads`; the coupling flow costs 13 ms of a step. The
  training set is held on the GPU (`data_on_device: true`); `train.py` then lets JAX take 95 % of the card.
- **Decision (Giuseppe, 2026-11-23): a very small transformer summary, for quick training**
  (`embed_dim` 64, `n_blocks` 2, `n_heads` 4 in the YAML; the sizes are Claude's pick, not yet run).

### Nested sampling of an observed track (`scripts/extragalactic/sbi/nested.py`, written 2026-11-24)

- dynesty's dynamic nested sampling of the posterior the importance weights aim at. Sampler settings (`ns_*`)
  default to StreaMAX-fit's "paper" preset: 2000 live points, `rslice`, 10,000 effective samples. Results in
  `outputs/extragalactic/C26_test/nested/<track>/`.
- `generate.py` cannot be imported in the project environment: its own uv environment has galax from git
  (`4c29842`), the project pins `galax==0.1.0`, which has no `galax.orbits`. `nested.py` therefore starts its
  GPU workers with the interpreter of that environment (`uv python find --script scripts/extragalactic/generate.py`).
  The comment in `pyproject.toml` that calls the two sets of pins the same is wrong on this point.
- Speed on compgpu6 (RTX 2080 Ti, 30,000 particles, 64-bit): 75 streams per second and card at any batch size.
- The likelihood is noisy in the random key of the particle release: 0.8 in log-likelihood (standard deviation
  over 802 posterior draws of NGC1162). `nested.py` uses one key for every call.
