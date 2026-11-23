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

  `scripts/extragalactic/train.py` (`pick_device`) and `scripts/extragalactic/generate.py` (`free_gpus`) already
  do this; new GPU code must do the same.
- To see who is using which card: `uv run gpustat`.
- Under Slurm, use the allocation as given (`CUDA_VISIBLE_DEVICES` set by the scheduler).
- On a machine without an NVIDIA driver the scripts fall back to the CPU; pass `--cpu` to force it.

Both `autocvd` and `gpustat` are in `pyproject.toml`.
