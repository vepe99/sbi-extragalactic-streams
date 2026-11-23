# sbi: amortised inference for stream tracks

Posterior estimation of the 14 parameters of Chemaly et al. 2026b (plus the progenitor position along the track)
from a stream track, with BayesFlow on the JAX backend. Run from the repository root, in the project environment.

## Run

One YAML file holds the settings of a run and drives all three scripts:

```bash
uv run python scripts/extragalactic/sbi/train.py           --config outputs/extragalactic/C26_test/npe_v2.yaml
uv run python scripts/extragalactic/sbi/evaluation.py      --config outputs/extragalactic/C26_test/npe_v2.yaml
uv run python scripts/extragalactic/sbi/evaluation_real.py --config outputs/extragalactic/C26_test/npe_v2.yaml
uv run python scripts/extragalactic/sbi/check.py --cpu     # self-tests
```

- On a machine with GPUs add `--extra gpu` after `uv run` (CUDA build of JAX); a free GPU is picked with autocvd.
  `--cpu` (or `cpu: true`) stays off them.
- A run lives in `outputs/extragalactic/C26_test/models/<name>/`: `config.yaml` (settings as used), `train.log`
  (one line per epoch), `history.csv`, `model.keras` (best held-out epoch), `evaluation/`, `evaluation_real/`.
- `evaluation.py`: mock tracks of held-out streams. Loss curve, recovery, calibration, contraction, and a
  posterior predictive check on a few tracks.
- `evaluation_real.py`: the observed STRRINGS tracks. Posterior draws (`posterior.npz`), `q.png`, a posterior
  predictive check per track and importance weights with the paper's likelihood (`weights.npz`, `metrics.json`).
- Predictive checks and importance weights simulate posterior draws again with `generate.py` on CPU
  (`sim_workers`, `sim_cores`; about 10 core-seconds per draw, `ppc_draws` per track). Draws and simulations are
  kept and reused; delete `evaluation*/draws.npz` and `simulations/` to redo them.
- The evaluations refuse to run if the training settings in the file differ from those the model was trained with.

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
inferred), `var_ratio` (the paper's model-noise guard, 0 = off), `subdir` (simulation set), `max_shards` (a subset,
for tests), and for the evaluations `n_test`, `n_samples`, `ppc_draws`, `real_tracks`, `importance`.

## Files

| file | holds |
|---|---|
| `config.py` | settings, command line, `config.yaml`, device choice |
| `data.py` | shards, prior and latent coordinates, window prior, observed tracks |
| `mock.py` | mock observations (the only augmentation), the training dataset, encoding of real tracks |
| `likelihood.py` | the paper's likelihood and importance weights |
| `networks.py` | summary and inference networks, the BayesFlow workflow |
| `train.py` | the training run |
| `evaluation.py` | mock tracks: loss curve, recovery, calibration, contraction, predictive check |
| `evaluation_real.py` | observed tracks: posteriors, predictive checks, importance weights |
| `predictive.py` | posterior draws, their simulations, predictive check and importance weights (shared) |
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
