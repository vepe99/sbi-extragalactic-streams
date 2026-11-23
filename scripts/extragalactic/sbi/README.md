# sbi: amortised inference for stream tracks

Posterior estimation of the 14 parameters of Chemaly et al. 2026b (plus the progenitor position along the track)
from a stream track, with BayesFlow on the JAX backend. Run from the repository root, in the project environment.

## Run

```bash
uv run python scripts/extragalactic/sbi/check.py --cpu                # self-tests
uv run python scripts/extragalactic/sbi/train.py --name npe_v2        # train
uv run python scripts/extragalactic/sbi/evaluation.py --name npe_v2   # evaluate
```

- Without `--cpu` a free GPU is picked with autocvd.
- A run lives in `outputs/extragalactic/C26_test/models/<name>/`: `config.yaml`, `train.log` (one line per epoch),
  `history.csv`, `model.keras` (best held-out epoch), `evaluation/`, `posterior_observed.npz`.
- `evaluation.py --what loss,recovery,calibration,contraction,ppc,observed` chooses the parts (default: all).
  `ppc` re-simulates posterior draws with `generate.py`; give it cores with `--ppc-workers 16 --ppc-cores 0-15`.
- `--prog given` (default) passes the progenitor position to the network; `--prog free` infers it.

## Change the configuration

Every field of `Config` in `config.py` is a command-line option (`_` becomes `-`):

```bash
uv run python scripts/extragalactic/sbi/train.py --name dm_v1 --inference diffusion --backbone transformer --epochs 200
uv run python scripts/extragalactic/sbi/train.py --config path/to/config.yaml --name rerun --lr 2e-4   # file, then overrides
```

The most used ones: `--inference flow|diffusion|flow_matching`, `--backbone mlp|transformer`,
`--summary transformer|timeseries|mlp`, `--p-known` (share of examples with the progenitor position given),
`--var-ratio` (the paper's model-noise guard, 0 = off), `--subdir` (simulation set), `--max-shards` (a subset, for tests).
To change a default, edit the field in `config.py`.

## Files

| file | holds |
|---|---|
| `config.py` | settings, command line, `config.yaml`, device choice |
| `data.py` | shards, prior and latent coordinates, window prior, observed tracks |
| `mock.py` | mock observations (the only augmentation), the training dataset, encoding of real tracks |
| `likelihood.py` | the paper's likelihood and importance weights |
| `networks.py` | summary and inference networks, the BayesFlow workflow |
| `train.py` | the training run |
| `evaluation.py` | loss curve, recovery, calibration, contraction, predictive checks, observed tracks |
| `check.py` | self-tests |

## Add pieces

- **A setting:** add a field to `Config`; it becomes an option and is saved in `config.yaml`.
- **A summary or inference network:** add a branch in `build_summary` or `build_inference` (`networks.py`) and name
  it in the `summary` / `inference` comment of `Config`. A summary network that takes a padding mask goes in `USES_MASK`.
- **A step of the mock observation** (another noise term, another kind of gap): add it in `mock` inside `make_mock`
  (`mock.py`), in JAX, with its own key from `jr.split`; return anything the tests need.
- **A network input:** add it in `featurise` (`mock.py`); mock and real tracks both go through it.
- **A diagnostic:** add a function in `evaluation.py`, a name in `--what`, and a call in `main`; write numbers into `metrics`.
- **A test:** add a `check(...)` line in `check.py`.

After a change, run `check.py`, then a short training: `train.py --cpu --max-shards 200 --epochs 2 --name test`.
