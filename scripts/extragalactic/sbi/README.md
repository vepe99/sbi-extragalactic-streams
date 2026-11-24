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
- `evaluation.py`: mock tracks of held-out streams. Loss curve, recovery, calibration, contraction, and a
  posterior predictive check on a few tracks.
- `evaluation_real.py`: the observed STRRINGS tracks. Posterior draws (`posterior.npz`), `q.png`, a posterior
  predictive check per track against the angle (`ppc_p*.png`) and in the plane of the track files (`sky_p*.png`:
  x, y in kpc, host at the origin), importance weights with the paper's likelihood (`weights.npz`,
  `metrics.json`) and a corner plot per track with the predictive check in its upper right corner
  (`corner_<track>.png`).
  Corner plots and predictive checks show two posteriors: the network's (blue) and the same draws reweighted
  with their importance weights (orange). In a corner plot the orange contours need an effective sample size
  of 10; below it the weighted draws are points. `ppc_draws` = `n_samples` simulates and weights every draw.
- `sky.py <run>/evaluation_real*`: redraws `sky_p*.png` from the files of a finished evaluation. It simulates
  nothing, loads no network and runs on the CPU.
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
  equal-weight `posterior`, `logz`), `nested.log`, `metrics.json`, `ppc.png`, `sky_p1.png` and `corner.png`,
  which shows the network's posterior next to it if `evaluation_real.py` was run for the `name` of the file.
  - Sampler settings are the `ns_*` fields; the defaults are StreaMAX-fit's "paper" preset (2000 live points,
    random-slice sampling, 10,000 effective samples).
  - It takes `ns_gpus` free GPUs (default 4; it waits until that many are free) and runs one worker per card.
    The workers run in the uv environment of `generate.py`, whose galax the project environment does not have.
  - Speed (measured 2026-11-24, RTX 2080 Ti, 30,000 particles, 64-bit): 75 streams per second and card,
    whatever the batch size.
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
systematic scatter), `standardize` (BayesFlow's standardisation of parameters, tokens and conditions: none, all or
a list), `subdir` (simulation set), `max_shards` (a subset, for tests), and for the evaluations `n_test`, `n_samples`, `ppc_draws`, `real_tracks`, `importance`.

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
| `sky.py` | the predictive check of a finished `evaluation_real.py` in the plane of the sky, from its files |
| `corner.py` | corner plot of a posterior with the predictive check in its upper right corner |
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
