import sys, glob, json, h5py, numpy as np
np.seterr(all="ignore")
for run in sys.argv[1:]:
    ev = f"outputs/extragalactic/C26_test/models/{run}/evaluation"
    w = np.load(f"{ev}/weights.npz", allow_pickle=True)
    W, cov = np.nan_to_num(w["weights"]), w["covering"]; K, M = W.shape
    cols = {k: [] for k in ("index", "diag/dE_rel")}
    for p in sorted(glob.glob(f"{ev}/simulations/data/shard_*.h5")):
        with h5py.File(p) as f:
            for k in cols: cols[k].append(f[k][:])
    o = np.argsort(np.concatenate(cols["index"])); dE = np.concatenate(cols["diag/dE_rel"])[o]
    res_true = dE[K*M:] <= 1e-2 if len(dE) > K*M else np.ones(K, bool)
    S = np.load(f"{ev}/test_tracks.npz")["S"][:K]
    ess = W.sum(1)**2 / (W**2).sum(1)
    Wn = W / W.sum(1, keepdims=True)
    fkl = np.sum(np.where(Wn > 0, Wn * np.log(Wn * M), 0), axis=1)      # E_p[log p/q], from self-normalised weights
    top = np.sort(Wn, axis=1)[:, -1]
    print(f"\n{run}: forward KL(p || q) estimate per mock track [nats] (biased low when n_eff is small), n_eff, largest weight")
    for lab, s in (("truth resolved, S<=12", res_true & (S <= 12)), ("truth resolved, S 13-25", res_true & (S > 12) & (S <= 25)), ("truth resolved, S>25", res_true & (S > 25)), ("truth unresolved", ~res_true)):
        if s.sum(): print(f"  {lab:24s} n {s.sum():2d}  KL median {np.median(fkl[s]):.2f} (16-84: {np.percentile(fkl[s],16):.2f}-{np.percentile(fkl[s],84):.2f})  n_eff median {np.median(ess[s]):.0f}  covering {np.median(cov[s].mean(1)):.2f}  largest weight median {np.median(top[s]):.2f}")
