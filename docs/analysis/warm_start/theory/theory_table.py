#!/usr/bin/env python
"""
Analytic table for warm-started (truncated) diffusion sampling in this repo's setting.

Nothing here touches a network: everything is either a closed-form function of the repo's noise schedule
(linear betas 1e-4..0.02, T = 1000) or a cheap statistic of a y-only starting estimate measured on held-out files.

What it computes
  1. Schedule: sqrt(ab), sqrt(1-ab), SNR = ab/(1-ab), the OU time T(t0) = -0.5 ln ab (horizon entering
     Chen et al. 2023's score/discretisation terms), checked against a checkpoint's stored diffusion_config.
  2. HFD (MatFlower60/Train), the split of the code that trained every surviving HFD checkpoint (snapshots
     988a338 / a703252: no 10 000-file cap, sorted list, first 90 % train = 36 730 files, last 10 % test = 4 081):
       - per-pixel Gaussian prior (mu, C) on the 31-band loader scale [-1, 1], fitted on TRAINING files only;
       - the regulariser s of K = C R (R^T C R + s^2 I)^-1 chosen by held-out error on training-class files
         (training files split 80/20 by source image; y is noiseless up to float32 rounding, as in the loader);
       - on every test file (full 64x64 cubes): per-element RMSE e of the clipped linear-MMSE start
         x0_hat = clip(mu + K (y - R^T mu), -1, 1), mapped to the model's 64-band target by the loader's linear
         interpolation, for R_Device1 and PH5; same for the prior-mean control x0_hat = mu;
       - Gaussian-surrogate posterior covariance (64-band), its eigenvalues, and the measured error variance
         along each eigen-direction (calibration of the surrogate);
       - display-scale metrics of the starting estimate itself (0-NFE baseline): RMSE %, SAM, PSNR.
  3. Sensor super-resolution stage (current code: 10 000-file cap, 9 000 train / 1 000 test), target = full
     30-channel R_Device1 sensor image globally normalised to [-1, 1] with min/max over the training split
     (scripts/compute_sensor_stats.py logic), start = bilinear up-sampling of the low-resolution sensor image
     (area_resize as in job_sensor_sr_sweep.slurm for d = 2/3/4/6/8; strided for d = 2/4/8 as in job_sensor_sr.slurm,
     with torch's default bilinear and with grid-aligned linear interpolation).
  4. The table: for t0 in {999, 800, 600, 500, 400, 300, 200, 100, 50, 20}: per-element ratio sqrt(SNR) e, whole-item
     KL bound SNR ||e||^2 / 2 (64x64x64 cube, and a 64-band pixel for the 1-D models), Pinsker TV bound,
     the KL bound of the standard N(0, I) start at t = 999 for reference, surrogate r = SNR * lambda_max(Sigma_post),
     and the Gaussian exact-sampler bias / spread factors.
  5. t0 rules: t0*(tau) = min{t : sqrt(SNR(t)) e <= tau}.
  6. A Monte-Carlo check of the closed-form Gaussian warm-start formulas with the repo's own DDPM / DDIM updates.

Run (CPU only, ~10 min, 8 worker processes):
    /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python theory_table.py
Writes theory_results.json and theory_table.txt next to this script.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):  # 8 single-threaded workers on a shared box
    os.environ.setdefault(_v, '1')
sys.dont_write_bytecode = True
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d

DATA = '/data/chaoyi_he/HSI/Diffu/dataset/HFD100 Mat dataset'
OUT = os.path.dirname(os.path.abspath(__file__))
CKPT_CHECK = '/data/chaoyi_he/HSI/Diffu/results/1d_hsi_diffusion/HFD/PH5/checkpoint_epoch_111.pth'
WL = np.linspace(451, 855, 31)
WL64 = np.linspace(451, 855, 64)
T0S = [999, 800, 600, 500, 400, 300, 200, 100, 50, 20]
TAUS = [0.1, 0.25, 0.5, 1.0, 2.0]
PX_FIT = 64           # pixels per training file entering the prior moments
VAL_FRAC = 0.2        # training source images held out to choose s
S_GRID = [0.0, 1e-7, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
SR_CAP = 10000        # HFD_SensorSR_data's file cap (current code)
N_WORKERS = 8
SEED = 0
D_CUBE = 64 * 64 * 64  # HSI stage: 64 bands x 64 x 64
D_PIX = 64             # 1-D models: one 64-band spectrum
D_SR = 30 * 64 * 64    # sensor-SR stage: 30 channels x 64 x 64
HIST_BINS = np.linspace(0, 2, 4001)


# ----------------------------------------------------------------------------- data helpers (loader-identical)
def load_R(n):
    """HFD_data.load_sensor_response(_new): 30 uniformly chosen columns, resampled to WL, column min-max."""
    if n is None:
        R = np.array(sio.loadmat(os.path.join(DATA, 'PH5_interp_results.mat'))['PH5_interp'])
        swl = np.arange(400, 1560, 10)
    else:
        R = np.array(sio.loadmat(os.path.join(DATA, f'R_Device{n}.mat'))['R'])
        swl = np.linspace(400, 1000, R.shape[0])
    R = R[:, np.linspace(0, R.shape[1] - 1, 30, dtype=int).tolist()]
    R = interp1d(swl, R, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')(WL)
    return (R - R.min(0)) / (R.max(0) - R.min(0) + 1e-20)  # [31, 30]


def interp_matrix():
    """A [64, 31] with x64 = A x31, identical to HFD_data.expand_wavelens."""
    return interp1d(WL, np.eye(31), axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')(WL64)


def all_files():
    d = os.path.join(DATA, 'MatFlower60', 'Train')
    out = []
    for c in os.listdir(d):
        if os.path.isdir(os.path.join(d, c)):
            out += [os.path.join(d, c, f) for f in os.listdir(os.path.join(d, c)) if f.endswith('.mat')]
    return sorted(out)


def key_of(path):
    return os.path.basename(os.path.dirname(path)), os.path.basename(path).split('_')[0]


def load31(path):
    """[64, 64, 31] float32 on the loader's per-image [-1, 1] scale (same float32 arithmetic as the loader)."""
    g = np.array(sio.loadmat(path)['truth'], dtype=np.float32)
    mn, mx = g.min(), g.max()
    g = (g - mn) / (mx - mn + 1e-20)
    return g * 2.0 - 1.0


# ----------------------------------------------------------------------------- workers
G = {}


def _init(globals_):
    torch.set_num_threads(1)
    G.update(globals_)


def train_worker(args):
    i, path = args
    x = load31(path).reshape(-1, 31)
    rng = np.random.default_rng(SEED * 1_000_003 + i)
    px = x[rng.choice(len(x), PX_FIT, replace=False)].copy()
    smin = smax = None
    if i < G['sr_ntrain']:  # global sensor stats over the capped training split (compute_sensor_stats.py)
        y = x @ G['R1']    # float32 @ float64 -> float64, as HFD_data.__getitem__
        smin, smax = float(y.min()), float(y.max())
    return px, smin, smax


def test_worker(path):
    """Full-cube errors of the y-only starts on one HFD test file (64-band target)."""
    x = load31(path).reshape(-1, 31)
    A = G['A']
    X64 = x.astype(np.float64) @ A.T
    out = dict(n=X64.size, x2=float((X64 ** 2).sum()), key=key_of(path))
    e_mean = (G['mu'] @ A.T)[None] - X64
    out['mean'] = dict(sse=float((e_mean ** 2).sum()), rmse_item=float(np.sqrt((e_mean ** 2).mean())))
    for name, (R, K, V) in G['sensors'].items():
        y = (x @ R).astype(np.float32).astype(np.float64)   # loader: float64 product, collated to float32
        xh = G['mu'] + (y - G['mu'] @ R) @ K.T
        xc = np.clip(xh, -1, 1)
        e64 = xc @ A.T - X64
        e64u = xh @ A.T - X64
        a = (xc @ A.T + 1) / 2
        b = (X64 + 1) / 2
        mse01 = ((a - b) ** 2).mean()
        cos = np.clip((a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12), -1, 1)
        out[name] = dict(sse=float((e64 ** 2).sum()), sse_unclipped=float((e64u ** 2).sum()),
                         sse31=float(((xc - x) ** 2).sum()),
                         rmse_item=float(np.sqrt((e64 ** 2).mean())),
                         disp_rmse_pct=float(np.sqrt(mse01) * 100), psnr=float(10 * np.log10(1 / mse01)),
                         sam_deg=float(np.degrees(np.arccos(cos)).mean()),
                         hist=np.histogram(np.abs(e64), HIST_BINS)[0].astype(np.int64),
                         proj=((e64 @ V) ** 2).sum(0),  # error energy along the surrogate posterior eigenvectors
                         consist=float(np.abs(xc @ R - y).max()))
    return out


def _lin_up_matrix(n_low, ds, n=64):
    """Grid-aligned linear interpolation for strided sampling: low-res sample j sits at full-res index j*ds."""
    M = np.zeros((n, n_low))
    for i in range(n):
        p = min(i / ds, n_low - 1)
        j = int(np.floor(p))
        f = p - j
        M[i, j] += 1 - f
        if f > 0:
            M[i, j + 1] += f
    return torch.tensor(M, dtype=torch.float32)


def sr_worker(path):
    x = load31(path)
    Y = (x.reshape(-1, 31) @ G['R1']).reshape(64, 64, 30)          # float64, as HFD_SensorSR_data
    rng_ = G['smax'] - G['smin'] + 1e-20
    Yn = np.clip(2.0 * (Y - G['smin']) / rng_ - 1.0, -1.0, 1.0).astype(np.float32)
    t = torch.from_numpy(Yn).permute(2, 0, 1)[None]
    traw = torch.from_numpy(Y.astype(np.float32)).permute(2, 0, 1)[None]
    out = dict(n=t.numel(), y2=float((t ** 2).sum()), clip_frac=float(np.mean(np.abs(2.0 * (Y - G['smin']) / rng_ - 1.0) > 1)))
    for ds in [2, 3, 4, 6, 8]:
        hn = int(round(64 / ds))
        lo = F.interpolate(t, size=(hn, hn), mode='area')
        up = F.interpolate(lo, size=(64, 64), mode='bilinear', align_corners=False)
        lo_r = F.interpolate(traw, size=(hn, hn), mode='area')
        up_r = F.interpolate(lo_r, size=(64, 64), mode='bilinear', align_corners=False)
        out[f'area{ds}'] = dict(sse=float(((up - t) ** 2).sum()), rmse_item=float(((up - t) ** 2).mean().sqrt()),
                                rel_raw=float((up_r - traw).norm() / traw.norm()),
                                hist=np.histogram(np.abs((up - t).numpy()), HIST_BINS)[0].astype(np.int64))
    for ds in [2, 4, 8]:
        lo = t[..., ::ds, ::ds]
        up = F.interpolate(lo, size=(64, 64), mode='bilinear', align_corners=False)
        M = _lin_up_matrix(lo.shape[-1], ds)
        up_al = torch.einsum('ij,bcjk,lk->bcil', M, lo, M)
        out[f'strided{ds}'] = dict(sse=float(((up - t) ** 2).sum()), rmse_item=float(((up - t) ** 2).mean().sqrt()),
                                   sse_aligned=float(((up_al - t) ** 2).sum()),
                                   rmse_item_aligned=float(((up_al - t) ** 2).mean().sqrt()))
    return out


# ----------------------------------------------------------------------------- linear-Gaussian pieces
def moments(P):
    P = P.astype(np.float64)
    mu = P.mean(0)
    return mu, (P - mu).T @ (P - mu) / len(P)


def gain(C, R, s):
    # pinv: at s = 0 the 30x30 system is numerically singular for PH5 (cond(R^T) ~ 1e19); equals inv for s > 0 here
    return C @ R @ np.linalg.pinv(R.T @ C @ R + s ** 2 * np.eye(R.shape[1]), hermitian=True)


def lin_rmse64(P, mu, C, R, s, A):
    y = (P.astype(np.float32) @ R).astype(np.float32).astype(np.float64)
    xh = np.clip(mu + (y - mu @ R) @ gain(C, R, s).T, -1, 1)
    return float(np.sqrt((((xh - P) @ A.T) ** 2).mean()))


def quantile_from_hist(h, q):
    c = np.cumsum(h) / h.sum()
    return float(HIST_BINS[1:][np.searchsorted(c, q)])


# ----------------------------------------------------------------------------- Gaussian warm-start check
def gaussian_check(ab, n=100_000, seed=0):
    """1-D posterior N(m, sig^2); start x_t0 = sqrt(ab) (m + delta) + sqrt(1-ab) z; exact x0-prediction
    E[x0 | x_t] = m + a sig^2 / (a^2 sig^2 + b^2) (x_t - a m); the repo's DDPM ancestral update (x0
    parameterisation, final step returns x0_pred) and DDIM eta=0 update, no clamping. Compared with the closed forms
      exact reverse kernel : mean m + r/(1+r) delta,          var sig^2 (1+2r)/(1+r)^2
      probability-flow ODE : mean m + sqrt(r/(1+r)) delta,    var sig^2 / (1+r)        with r = SNR(t0) sig^2.
    The repo's samplers are themselves under-dispersed at t0 = 999 (DDPM uses the posterior variance beta-tilde and
    the last step returns x0_pred with no noise), so variances are compared relative to the same sampler's t0 = 999
    output (ddpm/ddim_var_ratio_rel999); means are compared directly."""
    rng = np.random.default_rng(seed)
    ab = ab.double().numpy()
    abp = np.concatenate([[1.0], ab[:-1]])
    m, res = 0.1, []
    for sig, delta in [(0.02, 0.05), (0.1, 0.1), (0.3, 0.2)]:
        for t0 in [999, 400, 200, 100, 50, 20]:
            a0, b0 = np.sqrt(ab[t0]), np.sqrt(1 - ab[t0])
            r = ab[t0] / (1 - ab[t0]) * sig ** 2
            outs = {}
            for method in ['ddpm', 'ddim']:
                x = a0 * (m + delta) + b0 * rng.standard_normal(n)
                for t in range(t0, -1, -1):
                    a, b2 = np.sqrt(ab[t]), 1 - ab[t]
                    x0p = m + a * sig ** 2 / (a * a * sig ** 2 + b2) * (x - a * m)
                    if t == 0:
                        x = x0p
                        break
                    at = ab[t] / abp[t]
                    if method == 'ddpm':
                        mean = np.sqrt(abp[t]) * (1 - at) / b2 * x0p + np.sqrt(at) * (1 - abp[t]) / b2 * x
                        var = max((1 - at) * (1 - abp[t]) / b2, 0.0)
                        x = mean + np.sqrt(var) * rng.standard_normal(n)
                    else:
                        eps = (x - a * x0p) / np.sqrt(b2)
                        x = np.sqrt(abp[t]) * x0p + np.sqrt(1 - abp[t]) * eps
                outs[method] = (float(x.mean() - m), float(x.var()))
            if t0 == 999:   # the sampler's own full-length output spread (its discretisation / final-step deficit)
                ref = {k: v[1] / sig ** 2 for k, v in outs.items()}
            res.append(dict(sig=sig, delta=delta, t0=t0, r=float(r),
                            ddpm_var_ratio_rel999=outs['ddpm'][1] / sig ** 2 / ref['ddpm'],
                            ddim_var_ratio_rel999=outs['ddim'][1] / sig ** 2 / ref['ddim'],
                            ddpm_bias=outs['ddpm'][0], ddpm_bias_formula=float(r / (1 + r) * delta),
                            ddpm_var_ratio=outs['ddpm'][1] / sig ** 2, ddpm_var_ratio_formula=float((1 + 2 * r) / (1 + r) ** 2),
                            ddim_bias=outs['ddim'][0], ddim_bias_formula=float(np.sqrt(r / (1 + r)) * delta),
                            ddim_var_ratio=outs['ddim'][1] / sig ** 2, ddim_var_ratio_formula=float(1 / (1 + r))))
    return res


# ----------------------------------------------------------------------------- main
def main():
    t_start = time.time()
    torch.set_num_threads(16)
    quick = int(os.environ.get('THEORY_QUICK', '0'))
    log = open(os.path.join(OUT, 'theory_table_quick.log' if quick else 'theory_table.log'), 'w')

    def P(*a):
        s = ' '.join(str(v) for v in a)
        print(s, flush=True)
        log.write(s + '\n')
        log.flush()

    # ---- 1. schedule (exactly DiffusionTrainer's float32 linspace, accumulated in float64)
    betas = torch.linspace(1e-4, 0.02, 1000, dtype=torch.float32).double()
    ab = torch.cumprod(1 - betas, 0)
    sched_check = None
    try:
        ck = torch.load(CKPT_CHECK, map_location='cpu', weights_only=False, mmap=True)
        dc = ck['diffusion_config']
        sched_check = dict(ckpt=CKPT_CHECK, prediction_type=dc['prediction_type'], loss_type=dc['loss_type'],
                           n_timesteps=dc['n_timesteps'],
                           max_abs_diff_alpha_bar=float((dc['alpha_bars'].double() - ab).abs().max()),
                           max_abs_diff_beta=float((dc['betas'].double() - betas).abs().max()))
        del ck
    except Exception as e:  # noqa
        sched_check = dict(error=repr(e)[:200])
    P('schedule check vs checkpoint:', sched_check)
    abn = ab.numpy()
    snr = abn / (1 - abn)
    alphas = 1 - betas.numpy()
    abp = np.concatenate([[1.0], abn[:-1]])
    # CCDF's per-step contraction of the reverse mean map under a point-mass (oracle) score, and the
    # first-order-exact posterior-variance threshold below which a step contracts (see notes in results)
    lam = np.sqrt(alphas) * (1 - abp) / (1 - abn)
    vstar = (1 - lam) * np.sqrt(alphas) * (1 - abn) ** 2 / (betas.numpy() * abn)

    # ---- 2. HFD files (split of the snapshots that trained the HFD checkpoints: no cap)
    files = all_files()
    QUICK = int(os.environ.get('THEORY_QUICK', '0'))  # >0: smoke test on every QUICK-th file (results not valid)
    n_eval = int(len(files) * 0.1)
    train_f, test_f = files[:len(files) - n_eval], files[len(files) - n_eval:]
    capped = files[:SR_CAP]
    sr_ntrain = SR_CAP - int(SR_CAP * 0.1)
    sr_test = capped[sr_ntrain:]
    if QUICK:
        train_f, test_f, sr_test = train_f[::QUICK], test_f[::QUICK], sr_test[::QUICK]
        sr_ntrain = max(1, sr_ntrain // QUICK)
    P(f'HFD files {len(files)}: train {len(train_f)} test {len(test_f)} | capped SR split train {sr_ntrain} test {len(sr_test)}')
    R1, PH5, A = load_R(1), load_R(None), interp_matrix()

    with Pool(N_WORKERS, initializer=_init, initargs=(dict(R1=R1, sr_ntrain=sr_ntrain),)) as pool:
        res = pool.map(train_worker, list(enumerate(train_f)), chunksize=64)
    PX = np.stack([r[0] for r in res])                      # [n_train_files, PX_FIT, 31]
    smin = min(r[1] for r in res if r[1] is not None)
    smax = max(r[2] for r in res if r[2] is not None)
    P(f'loaded training pixels {PX.shape} in {time.time() - t_start:.0f}s; SR global stats min {smin:.6f} max {smax:.6f}')

    # s by held-out error on training-class files: 80/20 split of training SOURCE IMAGES
    keys = [key_of(f) for f in train_f]
    uniq = sorted(set(keys))
    rng = np.random.default_rng(SEED)
    val_keys = set(uniq[i] for i in rng.permutation(len(uniq))[: int(VAL_FRAC * len(uniq))])
    is_val = np.array([k in val_keys for k in keys])
    mu_f, C_f = moments(PX[~is_val].reshape(-1, 31))
    Pval = PX[is_val].reshape(-1, 31).astype(np.float64)
    s_choice, s_curve = {}, {}
    for name, R in [('R_Device1', R1), ('PH5', PH5)]:
        curve = [lin_rmse64(Pval, mu_f, C_f, R, s, A) for s in S_GRID]
        s_curve[name] = dict(zip([str(s) for s in S_GRID], curve))
        s_choice[name] = S_GRID[int(np.argmin(curve))]
        P(f'{name}: held-out (training-class) RMSE64 vs s:', {s: round(c, 5) for s, c in zip(S_GRID, curve)}, '-> s =', s_choice[name])
    rms_y = {name: float(np.sqrt(((Pval @ R) ** 2).mean())) for name, R in [('R_Device1', R1), ('PH5', PH5)]}

    # final prior on ALL training files
    mu, C = moments(PX.reshape(-1, 31))
    sensors, surrogate = {}, {}
    for name, R in [('R_Device1', R1), ('PH5', PH5)]:
        K = gain(C, R, s_choice[name])
        Sig = C - K @ R.T @ C
        Sig64 = A @ Sig @ A.T
        w, V = np.linalg.eigh((Sig64 + Sig64.T) / 2)
        w, V = w[::-1].clip(min=0), V[:, ::-1]
        sensors[name] = (R, K, V)
        sv = np.linalg.svd(R.T, compute_uv=False)
        surrogate[name] = dict(eig=w.tolist(), trace_per_elem=float(w.sum() / 64), lam_max=float(w[0]),
                               pred_rmse=float(np.sqrt(w.sum() / 64)), cond_RT=float(sv[0] / sv[-1]),
                               s=s_choice[name], s_rel_rms_y=s_choice[name] / rms_y[name])
        P(f'{name}: cond(R^T) {sv[0] / sv[-1]:.3e}; surrogate posterior eig (64-band, top 5) {np.round(w[:5], 6)}; '
          f'predicted per-element RMSE {np.sqrt(w.sum() / 64):.5f}')

    # ---- test cubes (HSI stage) and capped test cubes (SR stage)
    with Pool(N_WORKERS, initializer=_init, initargs=(dict(A=A, mu=mu, sensors=sensors, R1=R1, smin=smin, smax=smax),)) as pool:
        tres = pool.map(test_worker, test_f, chunksize=16)
        sres = pool.map(sr_worker, sr_test, chunksize=16)
    P(f'test passes done at {time.time() - t_start:.0f}s')

    train_keys = set(keys)
    seen = np.array([tuple(r['key']) in train_keys for r in tres])
    n_el = sum(r['n'] for r in tres)
    x0_rms = float(np.sqrt(sum(r['x2'] for r in tres) / n_el))
    init = {}
    for name in ['R_Device1', 'PH5', 'mean']:
        def agg(sel):
            rs = [r for r, s in zip(tres, sel) if s]
            ne = sum(r['n'] for r in rs)
            return float(np.sqrt(sum(r[name]['sse'] for r in rs) / ne))
        rows = [r[name] for r in tres]
        d = dict(e_pooled=agg(np.ones(len(tres), bool)), e_unseen_sources=agg(~seen),
                 rmse_item_mean=float(np.mean([r['rmse_item'] for r in rows])),
                 rmse_item_std=float(np.std([r['rmse_item'] for r in rows])),
                 rmse_item_p90=float(np.percentile([r['rmse_item'] for r in rows], 90)))
        if name != 'mean':
            h = np.sum([r['hist'] for r in rows], 0)
            proj = np.sum([r['proj'] for r in rows], 0) / (n_el / 64)
            w = np.array(surrogate[name]['eig'])
            d.update(e_unclipped=float(np.sqrt(sum(r['sse_unclipped'] for r in rows) / n_el)),
                     e31=float(np.sqrt(sum(r['sse31'] for r in rows) / (n_el / 64 * 31))),
                     abs_err_p50=quantile_from_hist(h, .5), abs_err_p99=quantile_from_hist(h, .99),
                     abs_err_p999=quantile_from_hist(h, .999),
                     err_var_along_eig_top5=proj[:5].tolist(), surrogate_eig_top5=w[:5].tolist(),
                     err_var_ratio_top5=(proj[:5] / np.maximum(w[:5], 1e-30)).tolist(),
                     max_meas_residual=float(max(r['consist'] for r in rows)),
                     # x0-part of the (gauss) start, mu_post + L z1: E||.-x0||^2 = E||mu_post-x0||^2 + tr(Sigma_post)
                     e_gauss_x0part_pred=float(np.sqrt(agg(np.ones(len(tres), bool)) ** 2 + surrogate[name]['trace_per_elem'])),
                     disp_rmse_pct=[float(np.mean([r['disp_rmse_pct'] for r in rows])), float(np.std([r['disp_rmse_pct'] for r in rows]))],
                     sam_deg=[float(np.mean([r['sam_deg'] for r in rows])), float(np.std([r['sam_deg'] for r in rows]))],
                     psnr=[float(np.mean([r['psnr'] for r in rows])), float(np.std([r['psnr'] for r in rows]))])
        init[name] = d
        P(name, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()}))
    P(f'test cubes {len(tres)}, of which {int(seen.sum())} share a source image with training; rms(x0) on [-1,1] = {x0_rms:.4f}')

    n_sr = sum(r['n'] for r in sres)
    sr = dict(global_min=smin, global_max=smax, clip_frac_test=float(np.mean([r['clip_frac'] for r in sres])),
              y_rms=float(np.sqrt(sum(r['y2'] for r in sres) / n_sr)), n_test=len(sres))
    for k in [f'area{d}' for d in [2, 3, 4, 6, 8]] + [f'strided{d}' for d in [2, 4, 8]]:
        rows = [r[k] for r in sres]
        d = dict(e_pooled=float(np.sqrt(sum(r['sse'] for r in rows) / n_sr)),
                 rmse_item_mean=float(np.mean([r['rmse_item'] for r in rows])),
                 rmse_item_std=float(np.std([r['rmse_item'] for r in rows])))
        if k.startswith('area'):
            h = np.sum([r['hist'] for r in rows], 0)
            d.update(rel_raw_mean=float(np.mean([r['rel_raw'] for r in rows])), abs_err_p99=quantile_from_hist(h, .99))
        else:
            d.update(e_pooled_aligned=float(np.sqrt(sum(r['sse_aligned'] for r in rows) / n_sr)))
        sr[k] = d
        P('SR', k, json.dumps({kk: round(v, 5) for kk, v in d.items()}))

    # ---- 4. the table
    def at(t):
        return dict(t0=t, sqrt_ab=float(np.sqrt(abn[t])), sqrt_1mab=float(np.sqrt(1 - abn[t])), snr=float(snr[t]),
                    T_ou=float(-0.5 * np.log(abn[t])), ccdf_lambda_t=float(lam[t]), ccdf_lambda_max=float(lam[1:t + 1].max()),
                    # CCDF Thm 1 floor 2 C tau / (1 - lambda^2) per element (tau = 1 for A = I): with C = n max_{i<=t0} beta_i
                    # (H2 over the steps actually run) and with the paper's eq. (19) C = n (1 - alpha_N) = n beta_N
                    ccdf_floor_per_elem=float(2 * (1 - alphas[1:t + 1]).max() / (1 - lam[1:t + 1].max() ** 2)),
                    ccdf_floor_per_elem_paperC=float(2 * (1 - alphas[-1]) / (1 - lam[1:t + 1].max() ** 2)),
                    ccdf_oracle_prod_t0_to_1=float(np.prod(lam[1:t + 1])),
                    contraction_var_threshold=float(vstar[t]), first_order_threshold=float((1 + abn[t]) / (2 * snr[t])))

    e_ref = {'lin R_Device1': init['R_Device1']['e_pooled'], 'lin PH5': init['PH5']['e_pooled'],
             'mean (prior mean)': init['mean']['e_pooled'], 'reference 0.046 (40 dB, doc)': 0.046}
    lam_max = {'lin R_Device1': surrogate['R_Device1']['lam_max'], 'lin PH5': surrogate['PH5']['lam_max']}
    e_sr = {f'SR area d={d}': sr[f'area{d}']['e_pooled'] for d in [2, 3, 4, 6, 8]}
    e_sr.update({f'SR strided d={d}': sr[f'strided{d}']['e_pooled'] for d in [2, 4, 8]})
    e_sr.update({f'SR strided-aligned d={d}': sr[f'strided{d}']['e_pooled_aligned'] for d in [2, 4, 8]})
    table = []
    for t in T0S:
        row = at(t)
        s = snr[t]
        row['standard_start_KL_bound_cube'] = float(0.5 * s * D_CUBE * x0_rms ** 2)  # N(0,I) start vs q_t(.|y), same bound
        for k, e in e_ref.items():
            kl = 0.5 * s * D_CUBE * e ** 2
            row[k] = dict(ratio=float(np.sqrt(s) * e), KL_cube=float(kl), TV_cube_pinsker=float(min(1.0, np.sqrt(kl / 2))),
                          KL_pixel=float(0.5 * s * D_PIX * e ** 2))
            if k in lam_max:
                r = s * lam_max[k]
                rk = s * np.array(surrogate[k.split(' ')[1]]['eig'])
                # exact KL(start || q_t0) if the per-pixel Gaussian surrogate WERE the posterior and x0_hat its mean:
                # per eigen-direction 0.5 [ln(1+r) - r/(1+r)] ~ r^2/4 (variance mismatch only, no bias term)
                kl_pix = float(0.5 * np.sum(np.log1p(rk) - rk / (1 + rk)))
                row[k].update(r_max=float(r), bias_factor_exact=float(r / (1 + r)),
                              std_factor_exact=float(np.sqrt(1 + 2 * r) / (1 + r)),
                              KL_cube_if_surrogate_exact=kl_pix * 64 * 64, KL_pixel_if_surrogate_exact=kl_pix)
        for k, e in e_sr.items():
            row[k] = dict(ratio=float(np.sqrt(s) * e), KL_sr=float(0.5 * s * D_SR * e ** 2))
        table.append(row)

    # ---- 5. t0 rules: smallest t with sqrt(SNR(t)) e <= tau  (sqrt(SNR) decreases in t)
    def t_rule(e, tau):
        ok = np.where(np.sqrt(snr) * e <= tau)[0]
        return int(ok[0]) if len(ok) else None

    def t_kl(e, d, eps):
        ok = np.where(0.5 * snr * d * e ** 2 <= eps)[0]
        return int(ok[0]) if len(ok) else None

    rules = {}
    e_p99 = {'lin R_Device1 (p99 |err|)': init['R_Device1']['abs_err_p99'], 'lin PH5 (p99 |err|)': init['PH5']['abs_err_p99']}
    e_p99.update({f'SR area d={d} (p99 |err|)': sr[f'area{d}']['abs_err_p99'] for d in [2, 3, 4, 6, 8]})
    for k, e in {**e_ref, **e_p99, **e_sr}.items():
        rules[k] = dict(e=e, **{f'tau={tau}': t_rule(e, tau) for tau in TAUS})
        if k in e_sr:
            rules[k]['KL_sr<=standard_start'] = t_kl(e, D_SR, 0.5 * snr[999] * D_SR * sr['y_rms'] ** 2)
        if k in e_ref:
            rules[k].update({'KL_cube<=0.1': t_kl(e, D_CUBE, 0.1),
                             'KL_cube<=standard_start': t_kl(e, D_CUBE, 0.5 * snr[999] * D_CUBE * x0_rms ** 2)})
        if k in lam_max:
            ok = np.where(snr * lam_max[k] <= 0.25)[0]
            rules[k]['r_max<=0.25'] = int(ok[0]) if len(ok) else None

    # ---- 6. Gaussian closed forms vs the repo's own update rules
    gchk = gaussian_check(ab)
    worst = max(max(abs(g['ddpm_bias'] - g['ddpm_bias_formula']) / max(g['delta'], 1e-12),
                    abs(g['ddpm_var_ratio_rel999'] - g['ddpm_var_ratio_formula']),
                    abs(g['ddim_bias'] - g['ddim_bias_formula']) / max(g['delta'], 1e-12),
                    abs(g['ddim_var_ratio_rel999'] - g['ddim_var_ratio_formula'])) for g in gchk)
    for g in gchk:
        P('  gauss sig=%.2f delta=%.2f t0=%4d r=%.4f | DDPM bias %.4f (formula %.4f) var/var999 %.3f (formula %.3f) | '
          'DDIM bias %.4f (formula %.4f) var/var999 %.3f (formula %.3f) | sampler var ratio at 999: ddpm %.3f ddim %.3f' % (
              g['sig'], g['delta'], g['t0'], g['r'], g['ddpm_bias'], g['ddpm_bias_formula'], g['ddpm_var_ratio_rel999'],
              g['ddpm_var_ratio_formula'], g['ddim_bias'], g['ddim_bias_formula'], g['ddim_var_ratio_rel999'],
              g['ddim_var_ratio_formula'], g['ddpm_var_ratio'] / g['ddpm_var_ratio_rel999'],
              g['ddim_var_ratio'] / g['ddim_var_ratio_rel999']))
    P(f'Gaussian closed forms vs repo updates: worst abs deviation (bias/delta, or variance ratio relative to t0=999) = {worst:.4f}')

    # ---- text table
    lines = []
    hdr = f"{'t0':>4} {'sqrt_ab':>8} {'sqrt1-ab':>8} {'SNR':>10} {'T_ou':>5} | " \
          f"{'rat R1':>7} {'rat PH5':>7} {'rat mu':>7} {'rat .046':>8} | {'KLcube R1':>10} {'KLcube PH5':>10} {'KLcube .046':>11} {'KL std':>9} | " \
          f"{'rmax R1':>8} {'rmax PH5':>8} | " + ' '.join(f"{'SR' + str(d):>7}" for d in [2, 3, 4, 6, 8])
    lines.append(hdr)
    for row in table:
        lines.append(f"{row['t0']:>4} {row['sqrt_ab']:8.5f} {row['sqrt_1mab']:8.5f} {row['snr']:10.3e} {row['T_ou']:5.2f} | "
                     f"{row['lin R_Device1']['ratio']:7.4f} {row['lin PH5']['ratio']:7.4f} {row['mean (prior mean)']['ratio']:7.4f} "
                     f"{row['reference 0.046 (40 dB, doc)']['ratio']:8.4f} | {row['lin R_Device1']['KL_cube']:10.3e} "
                     f"{row['lin PH5']['KL_cube']:10.3e} {row['reference 0.046 (40 dB, doc)']['KL_cube']:11.3e} "
                     f"{row['standard_start_KL_bound_cube']:9.3e} | {row['lin R_Device1']['r_max']:8.3e} {row['lin PH5']['r_max']:8.3e} | "
                     + ' '.join(f"{row[f'SR area d={d}']['ratio']:7.4f}" for d in [2, 3, 4, 6, 8]))
    lines.append('')
    lines.append('t0 rule t0*(tau) = min{t: sqrt(SNR(t)) e <= tau}:')
    for k, v in rules.items():
        lines.append(f"  {k:32s} e={v['e']:.5f} " + ' '.join(f"{kk}:{vv}" for kk, vv in v.items() if kk != 'e'))
    txt = '\n'.join(lines)
    P(txt)
    open(os.path.join(OUT, 'theory_table_quick.txt' if QUICK else 'theory_table.txt'), 'w').write(txt + '\n')

    results = dict(
        schedule=dict(beta=[1e-4, 0.02], T=1000, alpha_bar_999=float(abn[999]), snr_999=float(snr[999]), check=sched_check),
        split=dict(hsi=dict(code='snapshots 988a338/a703252 (no file cap)', n_train=len(train_f), n_test=len(test_f),
                            test_sharing_source_with_train=int(seen.sum()),
                            test_classes=sorted(set(key_of(f)[0] for f in test_f))),
                   sr=dict(code='current HFD_SensorSR_data (10 000-file cap)', n_train=sr_ntrain, n_test=len(sr_test))),
        prior=dict(px_per_file=PX_FIT, n_pixels=int(PX.shape[0] * PX.shape[1]), s_grid=S_GRID, s_curve=s_curve,
                   s_choice=s_choice, rms_y_val=rms_y, val_source_images=len(val_keys), train_source_images=len(uniq)),
        surrogate=surrogate, initializers=init, x0_rms=x0_rms, sensor_sr=sr, table=table, rules=rules,
        gaussian_check=gchk, gaussian_check_worst_dev=worst,
        dims=dict(cube=D_CUBE, pixel=D_PIX, sr=D_SR), runtime_s=time.time() - t_start)
    out_json = os.path.join(OUT, 'theory_results_quick.json' if QUICK else 'theory_results.json')
    json.dump(results, open(out_json, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    P(f'wrote {out_json} in {time.time() - t_start:.0f}s')


if __name__ == '__main__':
    main()
