"""Reconcile the two verifiers' whiteness numbers on the 8 stored cubes with both definitions, at d = 4 and d = 8."""
import json, os
import numpy as np

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/uq/'
OUT = os.path.dirname(os.path.abspath(__file__))


def corr(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def lag1_A(e):            # recompute-d4: pooled, horizontal only
    return corr(e[..., :, :-1], e[..., :, 1:])


def band_A(e):            # pooled adjacent-band corr; band axis is -3
    return corr(np.take(e, range(e.shape[-3] - 1), axis=-3), np.take(e, range(1, e.shape[-3]), axis=-3))


def lag1_B(a):            # mechanism-d4: per-map mean removed, mean of horizontal and vertical
    a = a - a.mean(axis=(-2, -1), keepdims=True)
    return float(((a[..., 1:, :] * a[..., :-1, :]).mean() + (a[..., :, 1:] * a[..., :, :-1]).mean()) / 2 / (a ** 2).mean())


def band_B(a):            # [C, H, W] per-band normalised, mean over adjacent pairs
    a = a.reshape(a.shape[0], -1); a = a - a.mean(1, keepdims=True); n = np.sqrt((a ** 2).sum(1))
    return float(((a[1:] * a[:-1]).sum(1) / (n[1:] * n[:-1] + 1e-30)).mean())


H = 64
k = np.fft.fftfreq(H) * H
RAD = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)


def hf_frac(a):           # fraction of spatial power at radial index >= 12.5, pooled over bands
    P = np.abs(np.fft.fft2(a.astype(np.float64), axes=(-2, -1))) ** 2
    return float(P[..., RAD >= 12.5].sum() / P.sum())


res = {}
for tag, p in (('d4', R + 'd4/eval/d4_contrast__cubes.npz'), ('d8', R + 'd8/eval/d8_main__cubes.npz')):
    z = np.load(p)
    S = z['samples']; X = z['x']; XR = z['x_r']
    rows = []
    for i in range(X.shape[0]):
        s = S[:, i].astype(np.float32); x = X[i]; xr = XR[i]
        mu = s.mean(0); e = s - mu[None]; tr = x - xr; r0 = s[0] - xr
        rows.append(dict(
            A_lag1_dev=lag1_A(e), A_lag1_true=lag1_A(tr), A_band_dev=band_A(e), A_band_true=band_A(tr),
            B_lag1_dev4=float(np.mean([lag1_B(e[kk]) for kk in range(4)])), B_lag1_true=lag1_B(tr), B_lag1_s0res=lag1_B(r0),
            B_band_dev4=float(np.mean([band_B(e[kk]) for kk in range(4)])), B_band_true=band_B(tr), B_band_s0res=band_B(r0),
            hf_true=hf_frac(tr), hf_dev=float(np.mean([hf_frac(e[kk]) for kk in range(4)])), hf_s0res=hf_frac(r0),
        ))
    res[tag] = {kk: float(np.mean([r[kk] for r in rows])) for kk in rows[0]}
    print(tag, json.dumps({kk: round(v, 3) for kk, v in res[tag].items()}), flush=True)
json.dump(res, open(os.path.join(OUT, 'white.json'), 'w'), indent=1)
