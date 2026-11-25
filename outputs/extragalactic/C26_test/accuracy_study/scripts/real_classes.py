import sys, glob, json, h5py, numpy as np
run = sys.argv[1]
ev = f"{run}/evaluation_real"
w = np.load(f"{ev}/weights.npz", allow_pickle=True)
names, W, cov = w["names"], w["weights"], w["covering"]
K, M = W.shape
cols = {k: [] for k in ("index", "status", "diag/dE_rel", "diag/n_peri", "diag/n_apo")}
for p in sorted(glob.glob(f"{ev}/simulations/data/shard_*.h5")):
    with h5py.File(p) as f:
        for k in cols: cols[k].append(f[k][:])
o = np.argsort(np.concatenate(cols["index"]))
g = lambda k: np.concatenate(cols[k])[o][:K*M].reshape(K, M)
dE, st = g("diag/dE_rel"), g("status")
norb = np.maximum(g("diag/n_peri"), g("diag/n_apo"))
unres, fold = dE > 1e-2, (st & 32) > 0
imp = {r["track"]: r for r in json.load(open(f"{ev}/metrics.json"))["tracks"]} if True else {}
print(f"{'track':18s} bins  cover  n_eff | unresolved share: all draws, covering, reweighted | folds: covering, reweighted | radial periods (median, covering)")
rows = []
for i, n in enumerate(names):
    c = cov[i]; ww = np.nan_to_num(W[i]); ww = ww / ww.sum() if ww.sum() > 0 else ww
    r = imp.get(str(n), {})
    rows.append((r.get("points", 0), str(n), c.mean(), r.get("effective_samples", float("nan")), unres[i].mean(), unres[i][c].mean(), (ww * unres[i]).sum(), fold[i][c].mean(), (ww * fold[i]).sum(), np.median(norb[i][c])))
for r in sorted(rows):
    print(f"{r[1]:18s} {r[0]:4d}  {r[2]:5.2f} {r[3]:6.1f} | {r[4]:5.2f} {r[5]:5.2f} {r[6]:5.2f} | {r[7]:5.2f} {r[8]:5.2f} | {r[9]:.0f}")
a = np.array([r[2:] for r in rows]); b = np.array([r[0] for r in rows])
for lab, s in (("<=16 bins", b <= 16), (">16 bins", b > 16)):
    print(lab, "tracks", s.sum(), "mean unresolved share (covering)", a[s, 3].mean().round(2), "reweighted", a[s, 4].mean().round(2), "folds (covering)", a[s, 5].mean().round(2))
