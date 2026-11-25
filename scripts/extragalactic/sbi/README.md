# sbi: amortised inference for stream tracks

Posterior estimation of the 14 parameters of Chemaly et al. 2026b (plus the progenitor position along the track)
from a stream track, with BayesFlow on the JAX backend. Run from the repository root, in the project environment.

## Run

One YAML file holds the settings of a run and drives all three scripts:

```bash
uv run python scripts/extragalactic/sbi/train.py           --config outputs/extragalactic/C26_test/npe_v2.yaml
uv run python scripts/extragalactic/sbi/evaluation.py      --config outputs/extragalactic/C26_test/npe_v2.yaml
uv run python scripts/extragalactic/sbi/evaluation_real.py --config outputs/extragalactic/C26_test/npe_v2.yaml
uv run python scripts/extragalactic/sbi/nested.py          --config outputs/extragalactic/C26_test/npe_v2.yaml --real-tracks NGC1162
uv run python scripts/extragalactic/sbi/check.py --cpu     # self-tests
```

- On a machine with GPUs add `--extra gpu` after `uv run` (CUDA build of JAX); a free GPU is picked with autocvd.
  `--cpu` (or `cpu: true`) stays off them.
- A run lives in `outputs/extragalactic/C26_test/models/<name>/`: `config.yaml` (settings as used), `train.log`
  (one line per epoch), `history.csv`, `model.keras` (best held-out epoch), `evaluation/`, `evaluation_real/`.
- A training run whose loss stops being finite (NaN) ends at that epoch with exit code 1; `model.keras` is then the
  best epoch before it, and `train.log` ends with `stopped at epoch N` instead of `finished`.
  Two things that gave such NaNs on 2026-11-24 are closed. `networks.py` replaces the spline function of
  BayesFlow 2.1.0, which evaluates the spline also for points outside its domain and discards the result: with a
  knot derivative at exactly 0 the discarded log Jacobian is log(0), and its gradient is NaN in every weight
  upstream at an ordinary, finite loss. The guarded function gives the same values. And a draw with no valid
  window (weight 0, tokens far outside the range of real tracks) no longer reaches the network: it takes the
  inputs of an accepted draw of its batch (`accepted_inputs` in `mock.py`).
- `train.py --config <run>/config.yaml --resume` takes a run up again from its `model.keras` (after a NaN, a kill,
  a reboot). Weights, standardisation and the optimizer with its step count come from the file, so the
  learning-rate schedule goes on where it was; the epochs after the one in `model.keras` are trained again and
  their rows leave `history.csv` (`train.log` keeps them). Stop the first process before.
- Both evaluations put their figures in subfolders, one per kind, and keep the files other code reads
  (`metrics.json`, the `.npz` files, `simulations/`) at the top of the directory:

  ```
  evaluation/                         evaluation_real/
    diagnostics/  loss, recovery,       q.png
                  calibration_ecdf,     ppc/     ppc_p*.png
                  z_score_contraction   sky/     sky_p*.png
    by_length/    by_length.png and     corner/  corner_<track>.png
                  the three diagnostics metrics.json, posterior.npz, weights.npz,
                  per length class      importance.md, draws.npz, simulations/
    sky/          sky_p*.png
    corner/       corner_mock_track_<i>.png
    metrics.json, test_tracks.npz, weights.npz,
    draws.npz, simulations/
  ```

- `evaluation.py`: mock tracks of held-out streams. Loss curve, recovery, calibration, contraction
  (`diagnostics/`), and for `ppc_tracks` of them a posterior predictive check in the plane of the sky
  (`sky/sky_p*.png`: the frame of the simulation, true progenitor on the +x axis, with the true model track) and
  a corner plot with the true values marked (`corner/corner_mock_track_<i>.png`). Recovery, calibration and
  contraction are repeated for short, medium and long tracks (`by_length/*_short.png`, `*_medium.png`,
  `*_long.png`; the classes are set by `length_edges`, in bins), and `by_length/by_length.png` puts the three
  classes side by side per parameter.
- `evaluation_real.py`: the observed STRRINGS tracks. Posterior draws (`posterior.npz`), `q.png`, a posterior
  predictive check per track against the angle (`ppc/ppc_p*.png`) and in the plane of the track files
  (`sky/sky_p*.png`: x, y in kpc, host at the origin), importance weights with the paper's likelihood
  (`weights.npz`, `metrics.json`; `importance.md` is the table per track of the effective sample size
  n_eff = (sum w)^2 / sum w^2 and the sample efficiency n_eff / n in per cent) and a corner plot per track with the predictive check in the plane of the sky
  in its upper right corner (`corner/corner_<track>.png`).
  A corner plot has 18 columns: the 15 parameters, with q, cos i and psi after the halo vector they are computed
  from.
  The sky panel of a corner plot also shows, as arrows, the acceleration of the host halo at the bins of the
  track, projected on the sky (km/s/Myr, the key gives the scale): for every simulated draw the potential of
  the draw is evaluated at the place of its own stream in three dimensions (bin angle, median radius and the
  mean line-of-sight coordinate of the particles of the bin, `track/los_mean`, which `generate.py` stores since
  2026-11-24), and the arrow is the median over the draws, at the median model track. Blue: the network's
  posterior; orange: the reweighted one; purple, on a mock track: the truth, from a simulation of the true
  parameters. Simulations kept from before that date have no `los_mean` and are made again by the evaluations;
  `sky.py` draws no arrows from them.
  Corner plots and predictive checks show two posteriors: the network's (blue) and the same draws reweighted
  with their importance weights (orange). In a corner plot the orange contours need an effective sample size
  of 10; below it the weighted draws are points. `ppc_draws` = `n_samples` simulates and weights every draw.
  `--eval-tag <tag>` writes to `evaluation_real_<tag>/` and leaves the evaluation without a tag as it is, e.g.
  one track with many more draws: `--real-tracks NGC1162 --n-samples 20000 --ppc-draws 20000 --eval-tag NGC1162_20000`.
  The training set is not loaded (prior, grid and simulator settings come from its first shard).
- `sky.py <run>/evaluation_real*`: redraws `sky/sky_p*.png` and the corner plots (`corner/`) from the files of a
  finished evaluation (a second argument names another directory to write to). It simulates nothing, loads no
  network and runs on the CPU.
- Predictive checks and importance weights simulate posterior draws again with `generate.py`, `ppc_draws` per
  track: on GPUs by default (`sim_device: gpu`; the card the evaluation holds plus the free ones), or on CPU
  (`sim_device: cpu`, `sim_workers`, `sim_cores`). They use the integrator of the training set unless
  `sim_integrator` names another; the results then go to `evaluation*_<integrator>/`. The adaptive galax
  integrator needs the CPU (about 10 core-seconds per draw; on a GPU not one stream finished in 280 s), the
  leapfrog runs tens of streams per second on a GPU.
  Draws and simulations are kept and reused; delete `evaluation*/draws.npz` and `simulations/` to redo them.
- The evaluations refuse to run if the training settings in the file differ from those the model was trained with.
- `nested.py`: dynamic nested sampling (dynesty) of the observed tracks named in `real_tracks`, one after the
  other: the posterior the importance weights aim at (15 parameters, the paper's likelihood with both guards at
  the `n_min`, `var_ratio` and `sig_sys_max` of the file, the simulator of `subdir`), without a network. It
  needs no trained model. Results go to `<out>/nested/<track>[_<ns_tag>]/`: `result.npz` (weighted samples,
  equal-weight `posterior`, `logz`), `nested.log`, `metrics.json`, `ppc.png`, `sky.png` and `corner.png`.
  - `corner.png` is drawn as the corner plots of `evaluation_real.py`: 18 columns, and in the upper right corner
    the predictive check in the plane of the sky (`sky.png`) with the acceleration of the host at the bins as
    arrows. Blue is nested sampling. If `evaluation_real.py` was run for the `name` of the file (with
    `--eval-tag`, the evaluation of that tag; `--name` picks another network), the figure also
    shows what the network gave for the track: its posterior (green, dashed) and its simulated draws with their
    importance weights (orange; in the sky panel the orange lines and arrows). `metrics.json` holds the network's
    numbers under `network`.
  - The figures are drawn at every call; sampling and the simulations of the predictive check are skipped if
    `result.npz` and `ppc.npz` are there. Such a call takes one free GPU (none if `ppc.npz` is there too), so
    the same command with the YAML file of another network redraws the comparison with that network.
  - Sampler settings are the `ns_*` fields; the defaults are StreaMAX-fit's "paper" preset (2000 live points,
    random-slice sampling, 10,000 effective samples).
  - Sampling takes `ns_gpus` free GPUs (default 4; it waits until that many are free), one worker per card.
    The workers run in the uv environment of `generate.py`, whose galax the project environment does not have.
  - Speed (measured 2026-11-24, RTX 2080 Ti, 30,000 particles, 64-bit): 75 streams per second and card,
    whatever the batch size, so 300 likelihood calls per second on four cards at best. A run with the default
    settings reaches about 190: the 512 chains dynesty runs at once end one by one (160 calls each on average
    with `rslice`, a few many more), so most batches are far from full. That is about 5,000 iterations an hour
    with 2000 live points (NGC1162, while still outside the region where the guards hold).
  - Every likelihood call uses the same random key for the release of the particles, so the likelihood is a
    function of the parameters. With another key the log-likelihood of a draw changes by 0.8 (standard
    deviation over 802 posterior draws of NGC1162; 10.7 at most).
  - A run that is stopped resumes from `checkpoint.save` (written every 5 minutes) when the same command is
    given again; with other settings it refuses, so give those an `--ns-tag`.
- Training speed (measured 2026-11-23, A100 40 GB, batch 4096, 49 tokens, flow of depth 8 and width 512; network
  only, one batch replayed):

  | summary network | ms / step |
  |---|---|
  | transformer, `embed_dim` 256, `n_blocks` 6, `n_heads` 8 | 185 |
  | the same, 4 heads / 2 heads | 175 / 171 |
  | `embed_dim` 192, 6 blocks | 139 |
  | `embed_dim` 128, 6 blocks | 100 |
  | `embed_dim` 128, 4 blocks, 4 heads | 63 |
  | `embed_dim` 256, 4 blocks / 3 blocks | 127 / 97 |
  | `embed_dim` 256, 6 blocks, tracks cut to 25 tokens | 99 |
  | `embed_dim` 256, 6 blocks, Keras `mixed_float16` (not in the code, loss not checked) | 103 |
  | none (`summary: mlp`): the coupling flow alone | 13 |
  | coupling flow reduced to depth 6, width 256 (transformer 256 / 6 / 8) | 180 |

  The time of a step is the summary transformer's (93 % of it at 256 / 6 blocks). It grows with `n_blocks`, with
  `embed_dim` and with `span_max`: every track is padded to `span_max` tokens, and a mock track has 15 observed
  bins on average. `n_heads` splits `embed_dim` and barely changes it. Making a batch through the host costs
  18 ms (2.6 gather, 5.5 to the GPU, 1 mock, 6 back, 3 adapter). The training set therefore sits on the
  GPU and the batches are made there (`data_on_device: true`, 1.0 GB per million usable streams; `train.py` then
  lets JAX take 95 % of the card). `data_on_device: false` makes the same batches through the host, 17 ms a step
  slower: use it when the run stops with `RESOURCE_EXHAUSTED`, i.e. when the set and the training step (26.9 GiB
  for `embed_dim` 256 / 6 blocks at batch 4096) do not fit on the card together.

## Change the configuration

Edit the YAML file, or copy it under a new `name`. A file is a flat list of `field: value`; every field of `Config`
(`config.py`) can appear, the ones left out keep their defaults, and an unknown field is an error. Options on the
command line override the file (`_` becomes `-`):

```bash
uv run python scripts/extragalactic/sbi/train.py --config outputs/extragalactic/C26_test/npe_v2.yaml --name dm_v1 --inference diffusion --backbone transformer
uv run python scripts/extragalactic/sbi/evaluation_real.py --config outputs/extragalactic/C26_test/npe_v2.yaml --real-tracks NGC5907 --ppc-draws 256
```

The most used fields: `inference` (flow, diffusion, flow_matching), `backbone` (mlp, transformer), `summary`
(transformer, timeseries, mlp), `p_known` (share of examples with the progenitor position given; 0 = always
inferred), `var_ratio` (the paper's model-noise guard, 0 = off), `sig_sys_max` (upper bound of the prior of the
systematic scatter), `noise_scale` (factor on the scatter of the mock radii; 0 = the model track itself, a test of
what a track holds without observational noise: see `npe_v2_nonoise.yaml`; `evaluation_real.py` refuses such a
run), `standardize` (BayesFlow's standardisation of parameters, tokens and conditions: none, all or
a list), `subdir` (simulation set), `max_shards` (a subset, for tests), and for the evaluations `n_test`, `n_samples`, `ppc_draws`, `real_tracks`, `importance`.

Two settings added on 2026-11-25 (`npe_v2_long90k.yaml`, `npe_v2_long90k_scaleaug.yaml`):

- `replace_tracks: train_v2_long90k` puts the tracks of a set of streams that were simulated again with more
  particles (`scripts/extragalactic/long90k.py`: the long streams of `train_v2`, 90,000 particles) in place of the
  stored tracks of the same streams. Replaced, not added, so the prior is unchanged. Guard 1 is a fraction of the
  particles, so the data carry `n_min` per stream (10 for 30,000 particles, 30 for 90,000) and `data.py` and
  `mock.py` use it; guard 2 is as it was. Which streams are kept and their order (hence the held-out split) are
  decided by the stored tracks. A stream whose new simulation fails the loader's cuts, or whose new `diag/dE_rel`
  is above the 0.01 it was selected with, keeps its stored track (5,930 of the 644,256; the set's `README.md`
  has the numbers). The selection of the long streams was made on the stored tracks, so the particle
  number of a stream is a function of its parameters and of its first realisation. The predictive check and the
  importance weights of `evaluation.py` and `evaluation_real.py` follow that (`more_particles` in `predictive.py`):
  a posterior draw is simulated with 30,000 particles, and if its stream is long by the rule of the set
  (`data.chosen_for_replacement`) it is simulated again with 90,000 and compared with guard 1 at 30; the second
  simulations are in `simulations_more_particles/` of the evaluation. `prior_weights.py` and `nested.py` still
  use 30,000 particles and `n_min` for every stream.
- `scale_aug: true` moves every training example along the scale family of the simulator (lengths x L, masses x
  U, velocities x sqrt(U / L), t_int x sqrt(L^3 / U), track x L), with (L, U) drawn from the prior restricted to
  that family (`draw_scale` in `mock.py`; `scale_points` candidates per example). A dissolved progenitor moves in
  L only. The validation mocks and the mocks of the evaluations are never scaled.
  `scripts/extragalactic/scale_augmentation.py` holds its tests.

## Files

| file | holds |
|---|---|
| `config.py` | settings, command line, `config.yaml`, device choice |
| `data.py` | shards, prior and latent coordinates, window prior, observed tracks |
| `mock.py` | mock observations (the only augmentation), the training dataset (on the device or on the host), encoding of real tracks |
| `likelihood.py` | the paper's likelihood, its form for a sampler, and importance weights |
| `networks.py` | summary and inference networks, the BayesFlow workflow |
| `train.py` | the training run |
| `evaluation.py` | mock tracks: loss curve, recovery, calibration, contraction, predictive check |
| `evaluation_real.py` | observed tracks: posteriors, predictive checks, importance weights |
| `predictive.py` | posterior draws, their simulations, predictive check and importance weights (shared) |
| `sky.py` | the sky-plane predictive check and the corner plots of a finished `evaluation_real.py`, from its files |
| `corner.py` | corner plot of a posterior with the predictive check in its upper right corner (observed tracks: in the plane of the sky) |
| `nested.py` | dynamic nested sampling of an observed track with dynesty, the simulator on several GPUs |
| `check.py` | self-tests |

## Add pieces

- **A setting:** add a field to `Config`; it becomes an option and is saved in `config.yaml`.
- **A summary or inference network:** add a branch in `build_summary` or `build_inference` (`networks.py`) and name
  it in the `summary` / `inference` comment of `Config`. A summary network that takes a padding mask goes in `USES_MASK`.
- **A step of the mock observation** (another noise term, another kind of gap): add it in `mock` inside `make_mock`
  (`mock.py`), in JAX, with its own key from `jr.split`; return anything the tests need.
- **A network input:** add it in `featurise` (`mock.py`); mock and real tracks both go through it.
- **A diagnostic:** add a function in `evaluation.py` (or `evaluation_real.py`) and call it in `main`; put its numbers
  in `metrics`.
- **A test:** add a `check(...)` line in `check.py`.

After a change, run `check.py`, then a short run of the three scripts with `--cpu --max-shards 200 --epochs 2 --name test`.
