"""Step 1 (numpy only, no network): build the test pixels and the linear-Gaussian estimator.

Outputs  data/test_pixels.npz   : 4096 test pixels (scene, index, target x[160] on [-1,1], sensor y[30])
         data/gauss_prior.npz   : mu, C (204-band prior on the loader's [-1,1] scale, TRAIN scenes only),
                                  H, b (affine forward operator y = H x + b), chosen s, s-validation table
Everything is computed exactly the way the 988a338/5f10ad1 HASCID_data.getitem_PH5 does it:
   y = resampled_gt[pixel] (float32, 61 wl 400..1000 nm) @ R (61x30, PH5, 30 uniformly selected columns,
       per-column min-max) -> float32 ; no sensor normalisation (5f10ad1 / 988a338 behaviour)
   x = (gtRef[pixel, :160] - 0.5) * 2   computed in float16 like the loader, then float32
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import DATA, OUT, get_loader_objects, interp_matrix

SEED_TEST = 20261006
N_TEST = 4096
TRAIN_PIX_PER_SCENE = 1024
N_VAL_SCENES = 68            # last 68 of the 688 training scenes -> held-out for choosing s
S_GRID = [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def sensor_from_resampled(resamp_rows, R):
    return np.matmul(resamp_rows.astype(np.float32), R).astype(np.float32)


def target_from_gt(gt_rows):
    g = gt_rows[:, :160]
    return ((g - 0.5) * 2.0).astype(np.float32)        # float16 arithmetic like the loader


def load_rows(path, idx):
    a = np.load(path, mmap_mode="r")
    H, W, C = a.shape
    flat = a.reshape(H * W, C)
    order = np.argsort(idx)
    out = np.empty((len(idx), C), dtype=a.dtype)
    out[order] = flat[np.asarray(idx)[order]]
    return out


def main():
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    tr, te = get_loader_objects("988a338")
    R = tr.sensor_R_matrix                      # [61, 30] float64
    assert R.shape == (61, 30)
    wl = tr.wavelens                            # 204
    swl = tr.sensor_wavelens                    # 61
    M = interp_matrix(wl, swl)                  # [61, 204]
    A = R.T @ M                                 # y = A @ gt_raw(204)
    # on the loader scale x' = 2 gt - 1  ->  gt = (x'+1)/2  ->  y = (A/2) x' + A 1 / 2
    Hop = A / 2.0
    bop = A.sum(1) / 2.0

    # ---------------- test pixels ----------------
    rng = np.random.default_rng(SEED_TEST)
    n_sc = len(te.img_name)
    scene_idx = rng.integers(0, n_sc, size=N_TEST)
    pix_idx = rng.integers(0, 512 * 512, size=N_TEST)
    X = np.zeros((N_TEST, 160), np.float32); Y = np.zeros((N_TEST, 30), np.float32)
    G204 = np.zeros((N_TEST, 204), np.float32)
    for s in np.unique(scene_idx):
        name = te.img_name[s]
        sel = np.where(scene_idx == s)[0]
        gt = load_rows(os.path.join(DATA, "gt_files", f"gtRef_{name}.npy"), pix_idx[sel])
        rs = load_rows(os.path.join(DATA, "resampled_gt", f"resamp_gt_{name}.npy"), pix_idx[sel])
        X[sel] = target_from_gt(gt)
        Y[sel] = sensor_from_resampled(rs, R)
        G204[sel] = ((gt - 0.5) * 2.0).astype(np.float32)
    # check: operator reproduces the loader's y (difference = float32 rounding of the resampled cube)
    y_op = (G204.astype(np.float64) @ Hop.T + bop)
    rel = np.abs(y_op - Y).max() / np.abs(Y).max()
    print(f"[test] {N_TEST} pixels from {len(np.unique(scene_idx))} of {n_sc} test scenes; "
          f"y range [{Y.min():.3f}, {Y.max():.3f}]; operator vs loader max rel diff {rel:.2e}")

    # cross-check a few pixels against the loader's own getitem_PH5 (image mode) on one test scene
    import copy
    te_img = copy.copy(te); te_img.data_format = "image"
    s0 = scene_idx[0]
    gt_l, y_l = te_img.getitem_PH5(int(s0))
    sel = np.where(scene_idx == s0)[0]
    r_, c_ = pix_idx[sel] // 512, pix_idx[sel] % 512
    dx = np.abs(gt_l[r_, c_, :].astype(np.float32) - X[sel]).max()
    dy = np.abs(y_l[r_, c_, :].astype(np.float32) - Y[sel]).max()
    print(f"[test] loader cross-check on scene {te.img_name[s0]} ({len(sel)} px): max|dx|={dx:.2e} max|dy|={dy:.2e}")
    assert dx == 0 and dy < 1e-5

    # ---------------- Gaussian prior from TRAIN scenes ----------------
    rng2 = np.random.default_rng(SEED_TEST + 1)
    n_tr = len(tr.img_name)
    feats, ys, sc_of = [], [], []
    t0 = time.time()
    for i, name in enumerate(tr.img_name):
        idx = rng2.choice(512 * 512, size=TRAIN_PIX_PER_SCENE, replace=False)
        gt = load_rows(os.path.join(DATA, "gt_files", f"gtRef_{name}.npy"), idx)
        feats.append(((gt - 0.5) * 2.0).astype(np.float32))
        if i >= n_tr - N_VAL_SCENES:
            rs = load_rows(os.path.join(DATA, "resampled_gt", f"resamp_gt_{name}.npy"), idx)
            ys.append(sensor_from_resampled(rs, R))
        sc_of.append(np.full(len(idx), i))
        if i % 100 == 0:
            print(f"  train scene {i}/{n_tr}  {time.time()-t0:.0f}s", flush=True)
    F = np.concatenate(feats).astype(np.float64)
    sc_of = np.concatenate(sc_of)
    Yv = np.concatenate(ys).astype(np.float64)

    def fit(Z):
        mu = Z.mean(0)
        C = np.cov(Z, rowvar=False)
        return mu, C

    def gain(C, s):
        # K = C H^T (H C H^T + s^2 I)^-1 via a symmetric eigendecomposition (H C H^T is rank 17 here)
        w, V = np.linalg.eigh(Hop @ C @ Hop.T)
        return C @ Hop.T @ ((V / (np.clip(w, 0, None) + s ** 2)) @ V.T)

    def lin_est(mu, C, y, s):
        K = gain(C, s)
        return mu + (y - (Hop @ mu + bop)) @ K.T, K

    fit_mask = sc_of < n_tr - N_VAL_SCENES
    mu_f, C_f = fit(F[fit_mask])
    Xv = F[~fit_mask][:, :160]
    table = []
    for s in S_GRID:
        xh, _ = lin_est(mu_f, C_f, Yv, s)
        xh = np.clip(xh[:, :160], -1, 1)
        rm = np.sqrt((((xh - Xv) / 2) ** 2).mean(1)).mean() * 100
        table.append({"s": s, "val_rmse_pct": float(rm)})
        print(f"  s={s:g}: held-out (train-split scenes) RMSE {rm:.3f} %")
    s_best = min(table, key=lambda r: r["val_rmse_pct"])["s"]
    print(f"[prior] chosen s = {s_best:g}")

    mu, C = fit(F)                               # refit on all training scenes
    K = gain(C, s_best)
    Sig_post = C - K @ Hop @ C
    Sig_post = (Sig_post + Sig_post.T) / 2
    ev_HCH = np.linalg.eigvalsh(Hop @ C @ Hop.T)[::-1]
    sv_A = np.linalg.svd(A, compute_uv=False)
    print(f"[prior] eig(H C H^T)/max: {np.array2string(ev_HCH / ev_HCH[0], precision=1, max_line_width=200)}")
    print(f"[prior] sv(A)/max: {np.array2string(sv_A / sv_A[0], precision=1, max_line_width=200)}")
    np.savez(os.path.join(OUT, "data", "gauss_prior.npz"), mu=mu, C=C, H=Hop, b=bop, s=s_best, K=K,
             Sig_post=Sig_post, n_train_pixels=F.shape[0], ev_HCH=ev_HCH, sv_A=sv_A)
    np.savez(os.path.join(OUT, "data", "test_pixels.npz"), X=X, Y=Y, scene_idx=scene_idx, pix_idx=pix_idx,
             scene_names=np.array(te.img_name))
    with open(os.path.join(OUT, "data", "s_validation.json"), "w") as f:
        json.dump({"grid": table, "s_chosen": s_best, "n_train_pixels": int(F.shape[0]),
                   "n_fit_pixels": int(fit_mask.sum()), "n_val_pixels": int((~fit_mask).sum())}, f, indent=1)


if __name__ == "__main__":
    main()
