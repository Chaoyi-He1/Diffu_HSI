"""
Per-class example spectra and split facts for the HFD100 flower dataset (MatFlower60, classes P001-P100),
for the "example classes" and "which classes the runs see" figures in docs/why_diffusion_recovers_hsi.html.

Preprocessing is the loader's (data_loader/HFD_dataset.py): per-image min-max normalisation to [-1, 1];
sensor values y = R^T x with R_Device1 (30 uniformly chosen filters, resampled, column min-max).
Spectra and errors are reported on the display scale (x + 1) / 2 in [0, 1], the scale the figures plot.

What it computes, using every file of every class in MatFlower60/Train:
  * each class's exact mean spectrum, its 10-90 % pixel band, and the range of all 100 class means
  * eight example classes by greedy farthest-point sampling on the class means
  * an approximate-colour thumbnail of one representative patch per example
  * each example's sensor signature R^T x_mean on the display scale, scaled to the map maximum
  * linear Gaussian-prior MMSE reconstruction from y at 40 dB SNR with two priors. Each class's source images
    are split in half; every class is scored on its second half.
      - "train" prior: fitted on the first half of classes P001-P022 only (what the capped loader trains on)
      - "all" prior:   fitted on the first half of all 100 classes
    per-class error = per-pixel RMSE over the 31 bands on the display scale (also §1's relative l2 on [-1, 1]);
    repeated for N_SPLITS random half-splits to check that the conclusions do not depend on the split
  * the median pairwise cosine similarity of the examples' sensor signatures and of their spectra
  * the loader's split with and without the 10 000-file cap, how source images are shared between patches and
    between MatFlower60/Train and MatFlower60/Test, and how many validation patches a file-level 90/10
    shuffle would leave with a sibling patch of the same image in training

Run from the repo root with the `hsi` environment (a few minutes):
    python docs/analysis/class_spectra.py class_spectra.json
"""
import base64
import io
import json
import os
import sys
from collections import defaultdict

import numpy as np
import scipy.io as sio
from PIL import Image
from scipy.interpolate import interp1d

ROOT = 'dataset/HFD100 Mat dataset'
WL = np.linspace(451, 855, 31)
SNR_DB = 40
PX_FIT = 128       # pixels per fit-half file entering the prior moments
PX_EVAL = 256      # pixels per eval-half file entering the reconstruction error
PX_BAND = 64       # pixels per file kept for the class's 10-90 % band
N_EXAMPLES = 8
N_SPLITS = 3     # random source-image half-splits; the figures use split 0
SEED = 0
TRAIN_CLASSES = {f'P{i:03d}' for i in range(1, 23)}  # what the capped loader trains on


def load_R():
    R = np.array(sio.loadmat(os.path.join(ROOT, 'R_Device1.mat'))['R'])
    swl = np.linspace(400, 1000, R.shape[0])
    R = R[:, np.linspace(0, R.shape[1] - 1, 30, dtype=int)]
    R = interp1d(swl, R, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')(WL)
    return (R - R.min(0)) / (R.max(0) - R.min(0) + 1e-20)  # [31, 30]


def src_id(path):
    """Patch files are named <source image>_<patch>.mat."""
    return os.path.basename(path).split('_')[0]


# ---- CIE 1931 2-degree colour matching functions, multi-lobe Gaussian fit (Wyman, Sloan, Shirley 2013)
def _g(x, mu, s1, s2):
    return np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)


def cmf(lam):
    xb = 1.056 * _g(lam, 599.8, 37.9, 31.0) + 0.362 * _g(lam, 442.0, 16.0, 26.7) - 0.065 * _g(lam, 501.1, 20.4, 26.2)
    yb = 0.821 * _g(lam, 568.8, 46.9, 40.5) + 0.286 * _g(lam, 530.9, 16.3, 31.1)
    zb = 1.217 * _g(lam, 437.0, 11.8, 36.0) + 0.681 * _g(lam, 459.0, 26.0, 13.8)
    return np.stack([xb, yb, zb], 1)


XYZ2RGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])


def thumbnail(path):
    """Approximate-colour sRGB rendering of a raw patch: the 31-band counts (451-855 nm only, not reflectance)
    weighted by the CIE 1931 functions under a flat illuminant, auto-scaled to the patch's 99.5th percentile.
    Light below 451 nm is missing, so blues are under-rendered; brightness is not comparable between patches."""
    g = np.array(sio.loadmat(path)['truth'], dtype=np.float64)
    rgb = np.clip((g.reshape(-1, 31) @ cmf(WL)) @ XYZ2RGB.T, 0, None)
    rgb = np.clip(rgb / (np.percentile(rgb, 99.5) + 1e-12), 0, 1)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    buf = io.BytesIO()
    Image.fromarray((rgb.reshape(64, 64, 3) * 255).round().astype(np.uint8)).save(buf, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def list_split(split):
    d = os.path.join(ROOT, 'MatFlower60', split)
    classes = sorted(c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c)))
    return classes, {c: sorted(os.path.join(d, c, f) for f in os.listdir(os.path.join(d, c)) if f.endswith('.mat'))
                     for c in classes}


def loader_split(all_files, cap):
    sub = all_files[:cap] if cap else all_files
    n_eval = int(len(sub) * 0.1)
    cls = lambda f: os.path.basename(os.path.dirname(f))
    count = lambda lst: dict(sorted(((c, sum(1 for f in lst if cls(f) == c)) for c in set(map(cls, lst)))))
    return dict(n_train=len(sub) - n_eval, n_val=n_eval, train=count(sub[:len(sub) - n_eval]), val=count(sub[len(sub) - n_eval:]))


def fit_and_eval(classes, files, src_of, store, R, split_seed, sig=None):
    """Split each class's source images in half (seeded), fit both priors on the fit halves, score every class
    on its eval half. Returns per-class stats and the noise level."""
    rng = np.random.default_rng(1000 + split_seed)
    fit_half = {}
    for c in classes:
        ids = sorted({src_of[f] for f in files[c]})
        perm = rng.permutation(len(ids))
        fit_half[c] = {ids[i] for i in perm[: len(ids) // 2]}
    mom = {k: [np.zeros(31), np.zeros((31, 31)), 0] for k in ('train', 'all')}
    for c in classes:
        for f in files[c]:
            if src_of[f] in fit_half[c]:
                p = store[f][:PX_FIT].astype(np.float64)
                for k in ('train', 'all') if c in TRAIN_CLASSES else ('all',):
                    mom[k][0] += p.sum(0); mom[k][1] += p.T @ p; mom[k][2] += len(p)
    priors = {k: (s1 / n, s2 / n - np.outer(s1 / n, s1 / n)) for k, (s1, s2, n) in mom.items()}
    if sig is None:  # 40 dB below the rms sensor value of the all-class fit pixels: rms^2 = trace(R^T E[x x^T] R) / 30
        sig = np.sqrt(np.trace(R.T @ (mom['all'][1] / mom['all'][2]) @ R) / 30) * 10 ** (-SNR_DB / 20)
    gain = {k: C @ R @ np.linalg.inv(R.T @ C @ R + sig ** 2 * np.eye(30)) for k, (mu, C) in priors.items()}
    stats = {}
    for c in classes:
        X = np.concatenate([store[f] for f in files[c] if src_of[f] not in fit_half[c]]).astype(np.float64)
        Yn = X @ R + sig * rng.standard_normal((len(X), 30))
        st = dict(mean_eval=X.mean(0))
        for k, (mu, C) in priors.items():
            Xh = mu + (gain[k] @ (Yn - mu @ R).T).T
            st['rmse_' + k] = float((np.sqrt(((Xh - X) ** 2).mean(1)) / 2).mean())       # display scale [0, 1]
            st['rel_' + k] = float((np.linalg.norm(Xh - X, axis=1) / np.linalg.norm(X, axis=1)).mean())  # §1's metric
            st['rec_' + k] = Xh.mean(0)
            st['neg_' + k] = float(((Xh < -1).any(1)).mean())  # below the patch's darkest value in some band
        stats[c] = st
    return stats, sig


def main(out_path):
    rng = np.random.default_rng(SEED)
    R = load_R()
    classes, files = list_split('Train')
    _, test_files = list_split('Test')
    all_files = sorted(f for c in classes for f in files[c])
    src_of = {f: src_id(f) for f in all_files}

    # ---- loader splits: current (10 000-file cap, commit 652f656) and before the cap
    capped, uncapped = loader_split(all_files, 10000), loader_split(all_files, None)

    # ---- source-image structure, Train/Test overlap, and what a file-level 90/10 shuffle would leak
    src = {c: defaultdict(list) for c in classes}
    for c in classes:
        for f in files[c]:
            src[c][src_of[f]].append(f)
    sizes = [len(s) for c in classes for s in src[c].values()]
    test_src = {(c, src_id(f)) for c in test_files for f in test_files[c]}
    train_src = {(c, s) for c in classes for s in src[c]}
    leak = []
    for seed in range(3):
        perm = np.random.default_rng(2000 + seed).permutation(len(all_files))
        n_val = int(len(all_files) * 0.1)
        val, tr = [all_files[i] for i in perm[:n_val]], [all_files[i] for i in perm[n_val:]]
        tr_keys = {(os.path.basename(os.path.dirname(f)), src_of[f]) for f in tr}
        leak.append(float(np.mean([(os.path.basename(os.path.dirname(f)), src_of[f]) in tr_keys for f in val])))
    min_ids = sorted({min(int(s) for s in src[c]) for c in classes})
    images = dict(train_patches=len(all_files), train_sources=len(sizes), train_sources_multi=int(sum(s > 1 for s in sizes)),
                  patches_per_source_mean=float(np.mean(sizes)), patches_per_source_max=int(max(sizes)),
                  image_ids_restart_per_class=min_ids == [1] if min_ids else False, smallest_image_id_per_class=min_ids,
                  test_patches=sum(len(v) for v in test_files.values()), test_sources=len(test_src),
                  test_sources_also_in_train=len(test_src & train_src),
                  file_shuffle_val_with_train_sibling=leak)
    print('images', images)

    # ---- one pass over every Train file: exact class means, band pixels, and a fixed pixel sample per file
    store, cls_sum, cls_n, band_px, file_means = {}, {}, {}, defaultdict(list), {}
    for c in classes:
        s, n = np.zeros(31), 0
        for f in files[c]:
            x = np.array(sio.loadmat(f)['truth'], dtype=np.float64).reshape(-1, 31)
            x = (x - x.min()) / (x.max() - x.min() + 1e-20) * 2 - 1  # loader scale [-1, 1]
            s += x.sum(0); n += len(x)
            file_means[f] = x.mean(0)
            band_px[c].append(x[rng.choice(len(x), PX_BAND, replace=False)])
            store[f] = x[rng.choice(len(x), max(PX_FIT, PX_EVAL), replace=False)].astype(np.float32)
        cls_sum[c], cls_n[c] = s, n
        print(c, len(files[c]), 'files', flush=True)

    # ---- the figures use split 0; splits 1 and 2 check that the conclusions do not depend on the split
    runs, sig = [], None
    for split_seed in range(N_SPLITS):
        st, sig = fit_and_eval(classes, files, src_of, store, R, split_seed, sig)
        runs.append(st)
        print('split', split_seed, 'done', flush=True)
    stats = runs[0]
    for c in classes:
        stats[c].update(n_files=len(files[c]), mean=cls_sum[c] / cls_n[c],
                        p10=np.percentile(np.concatenate(band_px[c]), 10, 0), p90=np.percentile(np.concatenate(band_px[c]), 90, 0))

    # ---- examples: greedy farthest-point sampling on the exact class means
    M = np.stack([stats[c]['mean'] for c in classes])
    chosen = [int(np.argmax(np.linalg.norm(M - M.mean(0), axis=1)))]
    while len(chosen) < N_EXAMPLES:
        chosen.append(int(np.argmax(np.min(np.stack([np.linalg.norm(M - M[i], axis=1) for i in chosen]), 0))))
    chosen.sort()

    disp = lambda v, nd=3: [round(float((a + 1) / 2), nd) for a in v]
    means01 = np.stack([(stats[classes[i]]['mean'] + 1) / 2 for i in chosen])
    sigs = means01 @ R
    sigs /= sigs.max()
    cosmed = lambda A: float(np.median([A[i] @ A[j] / np.linalg.norm(A[i]) / np.linalg.norm(A[j])
                                        for i in range(len(A)) for j in range(i + 1, len(A))]))
    examples = []
    for j, i in enumerate(chosen):
        c = classes[i]
        st = stats[c]
        rep = min(files[c], key=lambda f: np.linalg.norm(file_means[f] - st['mean']))
        examples.append(dict(id=c, n_files=st['n_files'], loader_train=capped['train'].get(c, 0), loader_val=capped['val'].get(c, 0),
                             mean=disp(st['mean']), p10=disp(st['p10']), p90=disp(st['p90']), mean_eval=disp(st['mean_eval']),
                             rec=disp(st['rec_all']), rec_train=disp(st['rec_train']),
                             rmse_all=round(st['rmse_all'], 4), rmse_train=round(st['rmse_train'], 4),
                             neg_all=round(st['neg_all'], 3), sig=[round(float(v), 3) for v in sigs[j]],
                             thumb=thumbnail(rep), thumb_file=os.path.relpath(rep, ROOT)))

    def group(st, k, sel):
        return float(np.mean([st[c]['rmse_' + k] for c in classes if sel(c)]))
    trained = lambda c: c in TRAIN_CLASSES
    other = lambda c: c not in TRAIN_CLASSES
    i693, i720 = int(np.argmin(abs(WL - 693.4))), int(np.argmin(abs(WL - 720.3)))  # bands 18 and 20

    def per_split(st):
        gaps = {c: st[c]['rmse_train'] - st[c]['rmse_all'] for c in classes}
        p = st['P064']
        return dict(rmse_train_on_train=group(st, 'train', trained), rmse_train_on_other=group(st, 'train', other),
                    rmse_all_on_train=group(st, 'all', trained), rmse_all_on_other=group(st, 'all', other),
                    rmse_all_mean=group(st, 'all', lambda c: True),
                    other_reduction=1 - group(st, 'all', other) / group(st, 'train', other),
                    other_improved=int(sum(gaps[c] > 0 for c in classes if other(c))),
                    largest_gap_class=max(gaps, key=gaps.get),
                    rel_all_mean=float(np.mean([st[c]['rel_all'] for c in classes])),
                    p064_step_true=float((p['mean_eval'][i720] - p['mean_eval'][i693]) / 2),
                    p064_step_all=float((p['rec_all'][i720] - p['rec_all'][i693]) / 2),
                    p064_step_train=float((p['rec_train'][i720] - p['rec_train'][i693]) / 2),
                    p021_neg_all=st['P021']['neg_all'])
    splits = [per_split(st) for st in runs]
    summary = dict(sigma=float(sig), splits=splits,
                   rmse_all_examples=float(np.mean([stats[classes[i]]['rmse_all'] for i in chosen])),
                   cos_median_sig=cosmed(sigs), cos_median_spectra=cosmed(means01))
    out = dict(wl=[round(float(w), 1) for w in WL], snr_db=SNR_DB, summary=summary, images=images,
               split=dict(capped=capped, uncapped=uncapped),
               env=dict(lo=disp(M.min(0)), hi=disp(M.max(0))),
               all=[[c, round(stats[c]['rmse_train'], 4), round(stats[c]['rmse_all'], 4), capped['train'].get(c, 0), capped['val'].get(c, 0)] for c in classes],
               ex=examples)
    json.dump(out, open(out_path, 'w'), separators=(',', ':'))
    print('examples', [classes[i] for i in chosen])
    print('summary', json.dumps(summary, indent=1))
    print('capped train classes', list(capped['train'])[0], '..', list(capped['train'])[-1], '| val', capped['val'])
    print('uncapped', uncapped['n_train'], uncapped['n_val'], 'train classes', list(uncapped['train'])[0], '..', list(uncapped['train'])[-1],
          '| val classes', list(uncapped['val'])[0], '..', list(uncapped['val'])[-1], 'first val class count', list(uncapped['val'].items())[0])
    print('wrote', out_path, os.path.getsize(out_path), 'bytes')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'class_spectra.json')
