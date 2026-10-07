"""Side check (numpy only): linear-Gaussian estimate quality for PH5 vs R_Device1 / R_Device2 on the same test pixels,
plus the x0-space noise level sigma(t) = sqrt((1-ab_t)/ab_t) of the re-noised start at each t0 of the grid.

R_Device{n} follows the 988a338 loader (getitem_new_R): y = gtRef(204 bands, raw [0,1]) @ R_n, R_n = 30 uniformly
selected columns resampled to the 204 HASCID wavelengths and min-max normalised per column.
The same 204-band Gaussian prior (TRAIN scenes) is used for every sensor; s is picked per sensor on the last 68
training scenes (these pixels are part of the prior fit, so this side check is slightly optimistic for every sensor).
This is NOT a diffusion result for R_Device1 (that needs the R_Device1 checkpoint); it only compares estimate quality.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from common import OUT, DATA, OLDCODE, get_loader_objects, metrics, _purge_modules
from prep_data import load_rows, S_GRID


def main():
    pr = dict(np.load(os.path.join(OUT, "data", "gauss_prior.npz")))
    d = np.load(os.path.join(OUT, "data", "test_pixels.npz"))
    tr, te = get_loader_objects("988a338")
    sys.path.insert(0, os.path.join(OLDCODE, "988a338")); _purge_modules()
    from data_loader.my_dataset import HASCID_data
    sensors = {"PH5": (pr["H"], pr["b"])}
    for n in (1, 2):
        ds = HASCID_data(DATA, train_mode="pixel", split="test", data_format="pixel", R_n=n)
        Rn = ds.sensor_R_matrix                       # [204, 30]
        sensors[f"R_Device{n}"] = (Rn.T / 2.0, Rn.sum(0) / 2.0)
    mu, C = pr["mu"], pr["C"]
    # held-out training scenes for s
    rng = np.random.default_rng(7)
    G = []
    for name in tr.img_name[-68:]:
        idx = rng.choice(512 * 512, size=512, replace=False)
        G.append(((load_rows(os.path.join(DATA, "gt_files", f"gtRef_{name}.npy"), idx) - 0.5) * 2.0).astype(np.float64))
    G = np.concatenate(G)
    # test pixels: full 204-band spectra are needed for the R_Device sensors
    Gt = np.zeros((len(d["X"]), 204))
    for s in np.unique(d["scene_idx"]):
        sel = np.where(d["scene_idx"] == s)[0]
        Gt[sel] = ((load_rows(os.path.join(DATA, "gt_files", f"gtRef_{d['scene_names'][s]}.npy"),
                              d["pix_idx"][sel]) - 0.5) * 2.0).astype(np.float64)
    Xt = d["X"].astype(np.float64)
    ab = torch.cumprod(1 - torch.linspace(1e-4, 0.02, 1000, dtype=torch.float32).double(), 0).numpy()
    out = {"sigma_x0_space": {int(t): float(np.sqrt((1 - ab[t]) / ab[t])) for t in [50, 100, 200, 300, 400, 600, 999]}}
    for name, (H, b) in sensors.items():
        def est(Y, s):
            w_, V_ = np.linalg.eigh(H @ C @ H.T)
            K = C @ H.T @ ((V_ / (np.clip(w_, 0, None) + s ** 2)) @ V_.T)
            return np.clip(mu + (Y - (H @ mu + b)) @ K.T, -1, 1)[:, :160]
        Yv = G @ H.T + b
        best = min(S_GRID, key=lambda s: metrics(est(Yv, s), G[:, :160].astype(np.float32))[0].mean())
        Yt = d["Y"].astype(np.float64) if name == "PH5" else Gt @ H.T + b
        if name == "PH5":
            best = float(pr["s"])                    # the s actually used in the sweep
        r, a, p = metrics(est(Yt, best), Xt)
        w = np.linalg.eigvalsh(H @ C @ H.T)[::-1]
        # noise-robust variant: s = 1 % of the training-prior rms sensor value (a 40 dB design), as for 'linreg'
        m_ = H @ mu + b
        s40 = 0.01 * float(np.sqrt((np.trace(H @ C @ H.T) + m_ @ m_) / 30))
        x40 = est(Yt, s40)
        r40, a40, _ = metrics(x40, Xt)
        out[name + "_s40dB"] = {"s": s40, "lin_rmse_pct": float(r40.mean()), "lin_sam_deg": float(a40.mean()),
                                "lin_err_x0_space_rms": float(np.sqrt(((x40 - Xt) ** 2).mean())),
                                "n_dirs_eig_above_s2": int((w > s40 ** 2).sum())}
        print(f"{name:10s} 40dB-design s={s40:.4f}: lin RMSE {r40.mean():.2f}%  SAM {a40.mean():.2f}  "
              f"x0-space rms err {out[name + '_s40dB']['lin_err_x0_space_rms']:.3f}  dirs used {int((w > s40 ** 2).sum())}")
        out[name] = {"s": best, "lin_rmse_pct": float(r.mean()), "lin_rmse_std": float(r.std()),
                     "lin_sam_deg": float(a.mean()), "lin_psnr": float(p.mean()),
                     "lin_err_x0_space_rms": float(np.sqrt((((est(Yt, best) - Xt)) ** 2).mean())),
                     "eig_HCHt_rel": (w / w[0]).tolist(),
                     "n_dirs_above_1e-4": int((w / w[0] > 1e-4).sum()),
                     "n_dirs_above_1e-6": int((w / w[0] > 1e-6).sum())}
        print(f"{name:10s} s={best:g}  lin RMSE {r.mean():.2f}% +- {r.std():.2f}  SAM {a.mean():.2f}  "
              f"x0-space rms err {out[name]['lin_err_x0_space_rms']:.3f}  "
              f"#eig(HCH^T)/max >1e-4: {out[name]['n_dirs_above_1e-4']}, >1e-6: {out[name]['n_dirs_above_1e-6']}")
    print("sigma(t) in x0 space:", out["sigma_x0_space"])
    with open(os.path.join(OUT, "data", "rdevice_compare.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
