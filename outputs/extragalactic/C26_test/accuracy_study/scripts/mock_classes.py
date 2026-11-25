import sys, glob, json, h5py, numpy as np
from scipy.stats import spearmanr
for run in sys.argv[1:]:
    ev = f"outputs/extragalactic/C26_test/models/{run}/evaluation"
    w = np.load(f"{ev}/weights.npz", allow_pickle=True)
    W, cov = w["weights"], w["covering"]; K, M = W.shape
    cols = {k: [] for k in ("index", "status", "diag/dE_rel")}
    for p in sorted(glob.glob(f"{ev}/simulations/data/shard_*.h5")):
        with h5py.File(p) as f:
            for k in cols: cols[k].append(f[k][:])
    o = np.argsort(np.concatenate(cols["index"]))
    dE = np.concatenate(cols["diag/dE_rel"])[o]
    dE_true, dE = dE[K*M:], dE[:K*M].reshape(K, M)
    rows = json.load(open(f"{ev}/metrics.json"))["ppc"]
    eff = np.array([r.get("importance_efficiency", np.nan) for r in rows])
    S = np.load(f"{ev}/test_tracks.npz")["S"][:K]
    un = (dE > 1e-2)
    share = np.array([un[i][cov[i]].mean() for i in range(K)])
    print(f"\n{run}: {K} mock tracks x {M} draws")
    print("  truth unresolved:", int((dE_true > 1e-2).sum()), "of", K)
    for lab, s in (("truth resolved", dE_true <= 1e-2), ("truth unresolved", dE_true > 1e-2)):
        print(f"   {lab:17s} n {s.sum():2d}  median efficiency {100*np.nanmedian(eff[s]):5.1f} %  median S {np.median(S[s]):.0f}  median unresolved share of the draws {np.median(share[s]):.2f}")
    for lab, s in (("unresolved share of draws < 0.15", share < 0.15), ("0.15 - 0.4", (share >= 0.15) & (share < 0.4)), (">= 0.4", share >= 0.4)):
        print(f"   draws: {lab:33s} n {s.sum():2d}  median efficiency {100*np.nanmedian(eff[s]):5.1f} %  median S {np.median(S[s]):.0f}")
    print("   Spearman(efficiency, unresolved share of draws) = %.2f ; Spearman(efficiency, S) = %.2f" % (spearmanr(eff, share)[0], spearmanr(eff, S)[0]))
    # efficiency if the unresolved draws are left out (restricted prior): weights of the resolved, covering draws only
    eff_res = []
    for i in range(K):
        ww = np.nan_to_num(W[i]) * ~un[i]
        eff_res.append(ww.sum()**2 / (ww**2).sum() / max((~un[i]).sum(), 1) if ww.sum() > 0 else np.nan)
    eff_res = np.array(eff_res); s = dE_true <= 1e-2
    print(f"   resolved truths only, weights of the resolved draws only (n_eff / resolved draws): median {100*np.nanmedian(eff_res[s]):.1f} %, against {100*np.nanmedian(eff[s]):.1f} % with all draws")
