"""Diagnose the NaNs of full DDPM-1000: replay the exact trajectory of one chunk/seed (same generators as sweep.py)
and log max|x_t|, max|eps_hat| and the x0-prediction per pixel; report the step where a pixel first leaves a sane range.
Also reruns the same trajectory with the common 'clip x0' guard (x0_hat clipped to [-1,1] inside each ancestral
step, posterior mean computed from (x0_hat, x_t)) to see whether that stabilises it."""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from common import OUT, load_model_strict, make_schedule, metrics
from sweep import gen

ap = argparse.ArgumentParser()
ap.add_argument("--chunk", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chunk_size", type=int, default=4)
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
torch.set_num_threads(a.threads)
model, snap, meta = load_model_strict()
sch = make_schedule(meta)
d = np.load(os.path.join(OUT, "data", "test_pixels.npz"))
idx = list(range(a.chunk * a.chunk_size, (a.chunk + 1) * a.chunk_size))
X, Y = d["X"][idx], d["Y"][idx]
cond = torch.tensor(Y, dtype=torch.float32)
n = len(idx)
log = {}
for mode in ["eps", "clip_x0"]:
    x = torch.randn((n, 1, 160), generator=gen("xT", a.seed, a.chunk))
    g = gen("ddpm_full", a.seed, a.chunk)
    first_bad = [None] * n
    trace = []
    for t in range(999, -1, -1):
        with torch.inference_mode():
            e = model(x, cond, torch.full((n,), t, dtype=torch.long)).double()
        xd = x.double()
        ab, al, be = sch.ab[t], sch.alphas[t], sch.betas[t]
        x0 = (xd - torch.sqrt(1 - ab) * e) / torch.sqrt(ab)
        if mode == "eps":
            mean = (xd - be / torch.sqrt(1 - ab) * e) / torch.sqrt(al)
        else:
            abp = sch.ab[t - 1] if t > 0 else torch.tensor(1.0, dtype=torch.float64)
            x0c = x0.clamp(-1, 1)
            mean = (torch.sqrt(abp) * be / (1 - ab)) * x0c + (torch.sqrt(al) * (1 - abp) / (1 - ab)) * xd
        if t > 0:
            mean = mean + torch.sqrt(sch.post_var[t]) * torch.randn(x.shape, generator=g).double()
        x = mean.float()
        xm = x.abs().amax((1, 2)).numpy(); em = e.abs().amax((1, 2)).numpy(); x0m = x0.abs().amax((1, 2)).numpy()
        if t % 50 == 0 or t >= 990:
            trace.append({"t": t, "max_abs_x": xm.tolist(), "max_abs_eps": em.tolist(), "max_abs_x0hat": x0m.tolist()})
        for i in range(n):
            if first_bad[i] is None and (not np.isfinite(xm[i]) or xm[i] > 50):
                first_bad[i] = t
    out = x[:, 0].numpy()
    with np.errstate(all="ignore"):
        r, s_, _ = metrics(np.clip(out, -1, 1), X)
    log[mode] = {"first_t_with_abs_x_gt_50_or_nonfinite": first_bad, "final_rmse_pct": r.tolist(),
                 "final_sam_deg": s_.tolist(), "trace": trace}
    print(mode, "first bad t per pixel:", first_bad, "final RMSE %:", np.round(r, 2).tolist(), flush=True)
    for tr in trace[:6] + trace[-4:]:
        print("   t", tr["t"], "max|x|", np.round(tr["max_abs_x"], 2).tolist(), "max|eps|", np.round(tr["max_abs_eps"], 2).tolist(),
              "max|x0hat|", np.round(tr["max_abs_x0hat"], 1).tolist())
with open(os.path.join(OUT, "data", f"diag_full_ddpm_c{a.chunk}_s{a.seed}.json"), "w") as f:
    json.dump(log, f, indent=1)
