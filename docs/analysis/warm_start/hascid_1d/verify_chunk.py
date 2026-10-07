"""Cheap reproducibility check (~a few minutes on CPU): recompute a few configurations of one stored chunk with the
same noise generators and compare with chunks/chunk_XXX.npz.  The sweep batched every job of a chunk together, so
batch composition differs here; agreement is expected up to float32 round-off amplified by the sampler."""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, json
import sweep
from sweep import Job, gen, ddim_times, run_jobs, lin_gauss, SELF_STEPS
from common import OUT, load_model_strict, make_schedule, metrics

ap = argparse.ArgumentParser()
ap.add_argument("--chunk", type=int, default=0)
ap.add_argument("--chunk_size", type=int, default=4)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
model, snap, meta = load_model_strict()
sch = make_schedule(meta); sweep.sch_global = sch
prior = dict(np.load(os.path.join(OUT, "data", "gauss_prior.npz")))
st = np.load(os.path.join(OUT, "chunks", f"chunk_{a.chunk:03d}.npz"))
idx = st["idx"]; d = np.load(os.path.join(OUT, "data", "test_pixels.npz"))
X, Y = d["X"][idx], d["Y"][idx]; n = len(idx); c = a.chunk
cond = torch.tensor(Y, dtype=torch.float32)
xlin, _ = lin_gauss(Y, prior)
jobs = []
for s in (0, 1):
    xT = torch.randn((n, 1, 160), generator=gen("self_xT", s, c))
    jobs.append(Job(f"s{s}/self_est", xT, cond, ddim_times(999, SELF_STEPS), "ddim"))
    x0h = torch.tensor(np.clip(xlin, -1, 1), dtype=torch.float32).unsqueeze(1)
    for t0, kind in [(200, "ddim10"), (50, "ddpm")]:
        z2 = torch.randn((n, 1, 160), generator=gen("renoise", s, c, t0))
        ab = sch.ab[t0].float(); xt0 = ab.sqrt() * x0h + (1 - ab).sqrt() * z2
        if kind == "ddpm":
            jobs.append(Job(f"s{s}/lin_ddpm_t{t0}", xt0, cond, range(t0, -1, -1), "ddpm", gen("ddpm_trunc", s, c, t0)))
        else:
            jobs.append(Job(f"s{s}/lin_ddim10_t{t0}", xt0, cond, ddim_times(t0, 10), "ddim"))
run_jobs(jobs, model, 256, log_every=0)
res = {}
for j in jobs:
    new = j.x[:, 0].numpy()
    old = st["est/" + j.name.replace("_est", "")] if j.name.endswith("self_est") else st["out/" + j.name]
    if j.name.endswith("self_est"):
        new = np.clip(new, -1, 1)
    rn = metrics(np.clip(new, -1, 1), X)[0].mean(); ro = metrics(np.clip(old, -1, 1), X)[0].mean()
    res[j.name] = {"max_abs_diff": float(np.abs(new - old).max()), "rmse_pct_rerun": float(rn), "rmse_pct_stored": float(ro)}
    print(f"{j.name:22s} max|new-old| {np.abs(new-old).max():.2e}  RMSE% rerun {rn:.3f} stored {ro:.3f}")
json.dump(res, open(os.path.join(OUT, "data", f"verify_chunk{c}.json"), "w"), indent=1)
