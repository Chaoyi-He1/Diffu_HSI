"""Step 3: warm-start / truncated reverse diffusion sweep (CPU), chunked and resumable.

For a chunk of test pixels and every noise seed, all sampler configurations are advanced together
("tick" loop: every active job takes one reverse step per tick; all jobs' samples go through the network
in shared batches, each sample with its own integer t).  Noise for every job comes from its own
torch.Generator seeded by (seed, chunk, purpose), so a chunk's numbers do not depend on how the batches
were composed, and the re-noising / ancestral noise is shared across the four inits (common random numbers).

Configs (per seed):
  full_ddpm               x_999 ~ N(0,I), DDPM t = 999..0                (1000 NFE)
  ddim50 / ddim100        same x_999, DDIM eta=0                         (50 / 100 NFE)
  <init>_ddpm_t<t0>       x_t0 = sqrt(ab_t0) x0_hat + sqrt(1-ab_t0) z, DDPM t = t0..0   (t0+1 NFE)
  <init>_ddim<n>_t<t0>    same x_t0, DDIM eta=0 on round(t0*(n-i)/n), i=0..n-1, final step to x0 (n NFE)
  inits: lin (Gaussian-prior MMSE), gauss (posterior sample), self (5-step DDIM of the model), mean (prior mean)
The ground truth never enters the sampler; it is stored only for scoring.
"""
import os, sys, json, time, argparse, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from common import OUT, load_model_strict, make_schedule

INITS = ["lin", "gauss", "self", "mean"]
DDPM_T0 = [600, 400, 300, 200, 100, 50]
DDIM_T0 = [400, 200]
DDIM_N = [10, 20]
SELF_STEPS = 5


def gen(*key):
    h = int(hashlib.sha256(repr(key).encode()).hexdigest()[:15], 16)
    return torch.Generator().manual_seed(h)


def ddim_times(t_start, n):
    return [int(round(t_start * (n - i) / n)) for i in range(n)]


class Job:
    def __init__(self, name, x, cond, times, method, noise_gen=None):
        self.name, self.x, self.cond, self.times, self.method = name, x.clone(), cond, list(times), method
        self.k = 0
        self.g = noise_gen
        self.nfe = len(times)

    @property
    def done(self):
        return self.k >= len(self.times)


def step(job, eps, sch):
    t = job.times[job.k]
    x = job.x.double(); e = eps.double()
    ab_t = sch.ab[t]
    if job.method == "ddpm":
        a_t, b_t = sch.alphas[t], sch.betas[t]
        mean = (x - b_t / torch.sqrt(1 - ab_t) * e) / torch.sqrt(a_t)
        if t > 0:
            z = torch.randn(job.x.shape, generator=job.g).double()
            mean = mean + torch.sqrt(sch.post_var[t]) * z
        job.x = mean.float()
    else:  # ddim, eta = 0
        x0 = (x - torch.sqrt(1 - ab_t) * e) / torch.sqrt(ab_t)
        if job.k + 1 < len(job.times):
            ab_n = sch.ab[job.times[job.k + 1]]
            job.x = (torch.sqrt(ab_n) * x0 + torch.sqrt(1 - ab_n) * e).float()
        else:
            job.x = x0.float()
    job.k += 1


def run_jobs(jobs, model, max_batch, log_every=50, tag=""):
    tick, pix_nfe, t_start = 0, 0, time.time()
    while True:
        active = [j for j in jobs if not j.done]
        if not active:
            break
        xs = torch.cat([j.x for j in active]); cs = torch.cat([j.cond for j in active])
        ts = torch.cat([torch.full((j.x.shape[0],), j.times[j.k], dtype=torch.long) for j in active])
        eps = torch.empty_like(xs)
        with torch.inference_mode():
            for s in range(0, xs.shape[0], max_batch):
                eps[s:s + max_batch] = model(xs[s:s + max_batch], cs[s:s + max_batch], ts[s:s + max_batch])
        pix_nfe += xs.shape[0]
        o = 0
        for j in active:
            n = j.x.shape[0]
            step(j, eps[o:o + n], sch_global)
            o += n
        tick += 1
        if log_every and tick % log_every == 0:
            el = time.time() - t_start
            print(f"{tag} tick {tick} active_jobs {len(active)} batch {xs.shape[0]} "
                  f"pixel-NFE {pix_nfe} elapsed {el:.0f}s ({el/max(pix_nfe,1)*1000:.1f} ms/pixel-NFE)", flush=True)
    return pix_nfe, time.time() - t_start


def lin_gauss(Y, prior, cond_unused=None):
    mu, H, b, K, Sp = prior["mu"], prior["H"], prior["b"], prior["K"], prior["Sig_post"]
    xh = mu + (Y.astype(np.float64) - (H @ mu + b)) @ K.T            # [n, 204]
    return xh[:, :160], mu[:160]


def chol_psd(S):
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 0, None)
    return V * np.sqrt(w)            # L with L L^T = S (PSD-projected)


def run_chunk(c, idx, args, model, sch, prior, cond_scale):
    d = np.load(os.path.join(OUT, "data", "test_pixels.npz"))
    X = d["X"][idx]; Y = d["Y"][idx]
    n = len(idx)
    cond = torch.tensor(Y * cond_scale, dtype=torch.float32)
    xlin, mu160 = lin_gauss(Y, prior)
    xlin_c = np.clip(xlin, -1, 1)
    Lp = chol_psd(prior["Sig_post"][:160, :160])
    est, jobs = {}, []
    if args.linreg_s is not None:
        # add-on: the same Gaussian-prior MMSE estimator with a larger regulariser s (a deliberately degraded,
        # noise-robust estimate that only uses the few strong directions of H C H^T); no baselines here.
        H, b, mu, C = prior["H"], prior["b"], prior["mu"], prior["C"]
        w, V = np.linalg.eigh(H @ C @ H.T)
        K2 = C @ H.T @ ((V / (np.clip(w, 0, None) + args.linreg_s ** 2)) @ V.T)
        xr = np.clip(mu + (Y.astype(np.float64) - (H @ mu + b)) @ K2.T, -1, 1)[:, :160]
        inits, pn_self, wall_self = ["linreg"], 0, 0.0
        for s in args.seeds:
            est[(s, "linreg")] = xr
    else:
        inits = INITS
    t_self0 = time.time()
    # --- self estimate: 5-step deterministic DDIM from t=999 (one job per seed) ---
    self_jobs = []
    for s in (args.seeds if args.linreg_s is None else []):
        xT = torch.randn((n, 1, 160), generator=gen("self_xT", s, c))
        self_jobs.append(Job(f"s{s}/self_est", xT, cond, ddim_times(999, SELF_STEPS), "ddim"))
    if self_jobs:
        pn_self, wall_self = run_jobs(self_jobs, model, args.max_batch, log_every=0)
    for s, j in zip(args.seeds, self_jobs):
        est[(s, "self")] = j.x[:, 0].numpy().clip(-1, 1).astype(np.float64)
    for s in args.seeds:
        if args.linreg_s is None:
            est[(s, "lin")] = xlin_c
            est[(s, "mean")] = np.repeat(mu160[None], n, 0)
            z1 = torch.randn((n, 160), generator=gen("gauss_z1", s, c)).double().numpy()
            est[(s, "gauss")] = xlin + z1 @ Lp.T            # posterior sample (mean not clipped: matches Sigma_post)
            xT = torch.randn((n, 1, 160), generator=gen("xT", s, c))
            T_full = 30 if args.smoke else 999          # --smoke only exercises the code paths
            jobs.append(Job(f"s{s}/full_ddpm", xT, cond, range(T_full, -1, -1), "ddpm", gen("ddpm_full", s, c)))
            for nn_ in (50, 100):
                jobs.append(Job(f"s{s}/ddim{nn_}", xT, cond, ddim_times(999, nn_), "ddim"))
        for init in inits:
            x0h = torch.tensor(est[(s, init)], dtype=torch.float32).unsqueeze(1)
            for t0 in sorted(set(DDPM_T0) | set(DDIM_T0)):
                z2 = torch.randn((n, 1, 160), generator=gen("renoise", s, c, t0))
                ab = sch.ab[t0].float()
                xt0 = ab.sqrt() * x0h + (1 - ab).sqrt() * z2
                if t0 in DDPM_T0:
                    jobs.append(Job(f"s{s}/{init}_ddpm_t{t0}", xt0, cond, range(t0, -1, -1), "ddpm",
                                    gen("ddpm_trunc", s, c, t0)))
                if t0 in DDIM_T0:
                    for nn_ in DDIM_N:
                        jobs.append(Job(f"s{s}/{init}_ddim{nn_}_t{t0}", xt0, cond, ddim_times(t0, nn_), "ddim"))
    if args.dry:
        tot = sum(j.nfe * n for j in jobs)
        print(f"chunk {c}: {len(jobs)} jobs, total pixel-NFE {tot}")
        return
    pn, wall = run_jobs(jobs, model, args.max_batch, log_every=args.log_every, tag=f"[chunk {c}]")
    out = {"X": X, "Y": Y, "idx": np.asarray(idx), "pixel_nfe": pn + pn_self, "wall": wall + wall_self}
    for j in jobs:
        out["out/" + j.name] = j.x[:, 0].numpy()
        out["nfe/" + j.name] = j.nfe
    for (s, init), v in est.items():
        out[f"est/s{s}/{init}"] = v
    tmp = os.path.join(args.out_dir, f"chunk_{c:03d}.tmp.npz")
    np.savez(tmp, **out)
    os.replace(tmp, os.path.join(args.out_dir, f"chunk_{c:03d}.npz"))
    print(f"[chunk {c}] saved; pixel-NFE {pn+pn_self}, wall {wall+wall_self:.0f}s "
          f"({(wall+wall_self)/(pn+pn_self)*1000:.1f} ms/pixel-NFE)", flush=True)


def main():
    global sch_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pixels", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--max_batch", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--out_dir", default=os.path.join(OUT, "chunks"))
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny grid, for testing the code only")
    ap.add_argument("--linreg_s", type=float, default=None,
                    help="add-on run: only the 'linreg' init (Gaussian MMSE with this s), no baselines")
    args = ap.parse_args()
    if args.smoke:
        global DDPM_T0, DDIM_T0, DDIM_N
        DDPM_T0, DDIM_T0, DDIM_N = [20, 10], [20], [4]
    torch.set_num_threads(args.threads)
    os.makedirs(args.out_dir, exist_ok=True)
    model, snap, meta = load_model_strict()
    assert meta["prediction_type"] == "eps"
    sch = make_schedule(meta); sch_global = sch
    prior = dict(np.load(os.path.join(OUT, "data", "gauss_prior.npz")))
    cs = json.load(open(os.path.join(OUT, "data", "cond_scale.json")))
    print(f"snapshot {snap}; condition = {cs['choice']} (scale {cs['scale']}); s = {float(prior['s']):g}", flush=True)
    n_chunks = (args.n_pixels + args.chunk - 1) // args.chunk
    for c in range(n_chunks):
        if c % args.n_workers != args.worker:
            continue
        if os.path.exists(os.path.join(args.out_dir, f"chunk_{c:03d}.npz")):
            print(f"[chunk {c}] exists, skipping"); continue
        idx = list(range(c * args.chunk, min((c + 1) * args.chunk, args.n_pixels)))
        run_chunk(c, idx, args, model, sch, prior, cs["scale"])


sch_global = None
if __name__ == "__main__":
    main()
