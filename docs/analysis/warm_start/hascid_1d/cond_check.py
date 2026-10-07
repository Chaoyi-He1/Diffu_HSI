"""Step 2: decide how the sensor condition must be scaled for this checkpoint.

The 50bb95a loader divides y by max_sensor (13.4166) while the 5f10ad1/988a338 loader (getitem_PH5) feeds raw y.
The checkpoint (saved 2025-09-30 08:32, ~14 h of training) post-dates 5f10ad1 (2025-09-29 14:56), so raw y is
expected; we verify it from the network itself: one-step x0 prediction error from x_t = q(x_t|x0) with the true
condition (raw / normalised) and with a shuffled condition (a no-information reference).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from common import OUT, load_model_strict, make_schedule, metrics

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
model, snap, meta = load_model_strict()
sch = make_schedule(meta)
d = np.load(os.path.join(OUT, "data", "test_pixels.npz"))
n = 64
X = torch.tensor(d["X"][:n]).unsqueeze(1); Y = torch.tensor(d["Y"][:n])
MAXS = 13.41655092067913
g = torch.Generator().manual_seed(123)
eps = torch.randn(X.shape, generator=g)
perm = torch.randperm(n, generator=g)
res = {}
for t in [100, 300, 600, 900]:
    ab = sch.ab[t].float()
    xt = ab.sqrt() * X + (1 - ab).sqrt() * eps
    tt = torch.full((n,), t, dtype=torch.long)
    for name, c in [("raw", Y), ("normalised", Y / MAXS), ("raw_shuffled", Y[perm])]:
        with torch.no_grad():
            e = model(xt, c, tt)
        x0 = ((xt - (1 - ab).sqrt() * e) / ab.sqrt()).clamp(-1, 1)
        rm, sam, _ = metrics(x0[:, 0].numpy(), X[:, 0].numpy())
        eps_err = float(((e - eps) ** 2).mean())
        res[f"t{t}_{name}"] = {"x0_rmse_pct": float(rm.mean()), "sam_deg": float(sam.mean()), "eps_mse": eps_err}
        print(f"t={t:4d} cond={name:13s} x0-pred RMSE {rm.mean():6.2f}%  SAM {sam.mean():5.2f}deg  eps MSE {eps_err:.4f}",
              flush=True)
raw_better = np.mean([res[f"t{t}_raw"]["x0_rmse_pct"] < res[f"t{t}_normalised"]["x0_rmse_pct"] for t in [100, 300, 600, 900]])
choice = "raw" if raw_better >= 0.5 else "normalised"
print("choice:", choice)
with open(os.path.join(OUT, "data", "cond_scale.json"), "w") as f:
    json.dump({"choice": choice, "scale": 1.0 if choice == "raw" else 1.0 / MAXS, "checks": res,
               "snapshot": snap}, f, indent=1)
