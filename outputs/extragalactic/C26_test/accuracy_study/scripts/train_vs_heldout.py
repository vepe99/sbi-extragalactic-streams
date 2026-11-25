"""NLL of trained runs on mock tracks of TRAINING streams against held-out streams, by track length and class."""
import os, sys
import numpy as np
REPO = "/export/data/vgiusepp/sbi-extragalactic-streams"
sys.path[:0] = [f"{REPO}/scripts/extragalactic", f"{REPO}/scripts/extragalactic/sbi"]
import track_noise as TN
from config import Config, pick_device
from pathlib import Path
runs, N = sys.argv[2:], 40000
cfgs = [Config.load(Path(REPO) / "outputs/extragalactic/C26_test/models" / r / "config.yaml")[0] for r in runs]
cfg = cfgs[0]
tab = TN.stream_table(cfg, 32)
print("device", pick_device(False, int(sys.argv[1])), flush=True)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import keras, networks
from data import load_simulations, split
from mock import fixed_observations
sims = load_simulations(cfg)
assert np.array_equal(tab["theta"].astype(np.float64), sims["theta"])
tr, held = split(sims, cfg.val_frac)
cut = len(tr["zeta13"])
sets = {"train": (fixed_observations(tr, cfg, sims["grid"], sims["design"], cfg.seed + 7, N, unweighted=True), 0),
        "heldout": (fixed_observations(held, cfg, sims["grid"], sims["design"], cfg.seed + 7, N, unweighted=True), cut)}
out = {}
for name, (t, off) in sets.items():
    st = off + t["pick"]
    out |= {f"{name}_S": t["S"], f"{name}_cls": tab["cls"][st], f"{name}_dE": tab["dE"][st], f"{name}_stream": st}
for c in cfgs:
    est = keras.saving.load_model(str(c.run_dir() / "model.keras"))
    for name, (t, off) in sets.items():
        fixed = np.ones_like(t["zeta"])
        lp = [np.asarray(est.log_prob(dict(zeta=t["zeta"][s], tokens=t["tokens"][s], mask=t["mask"][s], cond=t["cond"][s], fixed_target_mask=fixed[s], fixed_target_value=0 * fixed[s])))
              for s in (slice(i, i + 4096) for i in range(0, N, 4096))]
        out[f"{name}_nll_{c.name}"] = -np.concatenate(lp)
    S_t, S_h = out["train_S"], out["heldout_S"]
    print(f"\n{c.name}: NLL on training streams | held-out streams | gap (nats)")
    for lab, f in (("short 6-12", lambda S: S <= 12), ("medium 13-25", lambda S: (S > 12) & (S <= 25)), ("long 26-35", lambda S: (S > 25) & (S <= 35)), ("long 36-49", lambda S: S > 35), ("all", lambda S: S > 0)):
        row = []
        for k, cn in list(enumerate(TN.CLASS_KEYS)) + [(None, "all")]:
            a = out[f"train_nll_{c.name}"][f(S_t) & ((out["train_cls"] == k) if k is not None else True)]
            b = out[f"heldout_nll_{c.name}"][f(S_h) & ((out["heldout_cls"] == k) if k is not None else True)]
            row.append(f"{cn}: {a.mean():6.2f} | {b.mean():6.2f} | {b.mean() - a.mean():+5.2f} (se {np.sqrt(a.var() / len(a) + b.var() / len(b)):.2f})")
        print(f"  {lab:13s} " + "   ".join(row), flush=True)
    del est
np.savez(f"{os.path.dirname(os.path.abspath(__file__))}/train_vs_heldout.npz", **out)
