"""Step 1 (CPU): choose the test sets and fit the per-pixel Gaussian prior used by the
'lin', 'gauss' and 'mean' starting estimates.

* primary test set: 16 files from classes P090-P100 (all in the eval split, never trained on),
  one per class plus a second file for 5 randomly chosen classes (seed 0).
* secondary set: 8 files from 8 random training classes P001-P088 (files the model trained on).
* prior (mu, C) on the loader's 31-band [-1,1] scale, fitted on pixels of 3000 random TRAINING
  files (first 36 730 sorted files), excluding the secondary-set files.
* the regulariser s of K = C R (R^T C R + s^2 I)^-1 is chosen by the per-file RMSE (on the [0,1]
  scale, 64 bands, after clipping) on 400 other held-out training files.
x64 = A x31 with A the 64x31 linear-interpolation matrix of HFD_data.expand_wavelens, so the
64-band MMSE estimate is exactly A times the 31-band one (equivalent to fitting the joint
Gaussian of (x64, y) directly, because x64 is a fixed linear function of x31).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
torch.set_num_threads(16)

rng = np.random.default_rng(0)
tr, ev = make_datasets()
R = np.asarray(tr.sensor_R_matrix, dtype=np.float64)            # [31, 30]
A = interp_matrix(tr.wavelens)                                    # [64, 31]
print('R', R.shape, 'A', A.shape, 'n_train files', len(tr.img_list), 'n_eval files', len(ev.img_list))
assert len(tr.img_list) == 36730 and class_of(tr.img_list[-1]) == 'P089'

# ---- check A against the loader's own expand_wavelens / y against its __getitem__ ----
x64_l, y_l = ev[0]
x31 = load_cube31(ev.img_list[0])
assert np.allclose(x31.reshape(-1, 31) @ A.T, x64_l.reshape(-1, 64), atol=1e-5)
assert np.allclose(x31.reshape(-1, 31) @ R, y_l.reshape(-1, 30), atol=1e-4)
print('A and R reproduce the loader output: OK')

# ---- test sets ----
by_cls = {}
for i, p in enumerate(ev.img_list):
    by_cls.setdefault(class_of(p), []).append(i)
test_classes = ['P%03d' % k for k in range(90, 101)]
extra = set(rng.choice(test_classes, size=5, replace=False).tolist())
primary = []
for c in test_classes:
    n = 2 if c in extra else 1
    for i in rng.choice(by_cls[c], size=n, replace=False):
        primary.append(ev.img_list[int(i)])
assert len(primary) == 16
tr_by_cls = {}
for i, p in enumerate(tr.img_list):
    tr_by_cls.setdefault(class_of(p), []).append(i)
sec_classes = sorted(rng.choice(['P%03d' % k for k in range(1, 89)], size=8, replace=False).tolist())
secondary = [tr.img_list[int(rng.choice(tr_by_cls[c]))] for c in sec_classes]
print('primary:', [os.path.relpath(p, DATA) for p in primary])
print('secondary:', [os.path.relpath(p, DATA) for p in secondary])

# ---- prior fit on training files ----
sec_set = set(secondary)
pool = np.array([i for i, p in enumerate(tr.img_list) if p not in sec_set])
perm = rng.permutation(pool)
fit_idx, val_idx = perm[:3000], perm[3000:3400]
t0 = time.time()
pix = []
for i in fit_idx:
    c = load_cube31(tr.img_list[i]).reshape(-1, 31)
    pix.append(c[rng.choice(c.shape[0], 256, replace=False)])
pix = np.concatenate(pix).astype(np.float64)
mu = pix.mean(0)
C = np.cov(pix, rowvar=False)
print('fitted prior on %d pixels from %d training files in %.1fs' % (len(pix), len(fit_idx), time.time() - t0))
print('eig(C) range %.3e .. %.3e; eig(R^T C R) range %.3e .. %.3e' % (
    np.linalg.eigvalsh(C).min(), np.linalg.eigvalsh(C).max(),
    np.linalg.eigvalsh(R.T @ C @ R).min(), np.linalg.eigvalsh(R.T @ C @ R).max()))


def gain(s):
    S = R.T @ C @ R + (s ** 2) * np.eye(R.shape[1])
    return np.linalg.solve(S, (C @ R).T).T      # = C R S^-1 (S symmetric)


def lin_est64(x31img, K):
    # y exactly as the model receives it: float32 cube @ float64 R, then cast to float32
    y = (x31img.reshape(-1, 31) @ R).astype(np.float32).astype(np.float64)
    xh = mu + (y - mu @ R) @ K.T
    return np.clip(xh @ A.T, -1, 1)


val_cubes = [load_cube31(tr.img_list[i]) for i in val_idx]
grid = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]  # s=0 is singular: R^T C R has a ~0 eigenvalue
sel = {}
for s in grid:
    K = gain(s)
    errs = []
    for c in val_cubes:
        gt = (c.reshape(-1, 31) @ A.T + 1) / 2
        e = (lin_est64(c, K) + 1) / 2
        errs.append(np.sqrt(((e - gt) ** 2).mean()) * 100)
    sel[s] = float(np.mean(errs))
    print('s = %-7g held-out RMSE %.4f %% (median %.4f, max %.4f)' % (s, sel[s], np.median(errs), np.max(errs)))
s_best = min(sel, key=sel.get)
K = gain(s_best)
Sig_post = C - K @ R.T @ C
Sig_post = 0.5 * (Sig_post + Sig_post.T)
ev_post = np.linalg.eigvalsh(Sig_post)
w, V = np.linalg.eigh(Sig_post)
L = V * np.sqrt(np.clip(w, 0, None))           # L L^T = Sig_post with round-off negatives clipped to 0
print('chosen s =', s_best, '; posterior cov eig range %.3e .. %.3e; trace %.4e (prior trace %.4e)' % (
    ev_post.min(), ev_post.max(), np.trace(Sig_post), np.trace(C)))

np.savez(os.path.join(OUT, 'prior.npz'), mu=mu, C=C, R=R, A=A, K=K, L=L, Sig_post=Sig_post, s=s_best)
json.dump(dict(primary=primary, secondary=secondary, s=s_best, s_grid=sel,
               fit_files=len(fit_idx), fit_pixels=int(len(pix)), val_files=len(val_idx)),
          open(os.path.join(OUT, 'testsets.json'), 'w'), indent=1)
print('saved prior.npz and testsets.json')
