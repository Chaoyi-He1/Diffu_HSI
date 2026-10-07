"""NaN-aware scoring helpers used by aggregate.py (full DDPM from pure noise diverges for a few pixel/seed pairs)."""
import numpy as np


def paired(R, refR):
    """pair-level difference (pixel x seed), seed-averaged per pixel over finite pairs; mean and SE over pixels"""
    D = R - refR
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dp = np.nanmean(D, 0)
    dp = dp[np.isfinite(dp)]
    return float(dp.mean()), float(dp.std(ddof=1) / np.sqrt(len(dp))), int(len(dp))


def make_row(init, t0, sampler, nfe, R, S, P, ref, sec_per_pixel_nfe):
    F = np.isfinite(R)
    per_seed = np.array([np.nanmean(r) for r in R])
    row = {
        "init": init, "t0": t0, "sampler": sampler, "nfe": nfe,
        "rmse_pct": float(np.nanmean(R)), "rmse_std": float(np.nanstd(R)),
        "rmse_median": float(np.nanmedian(R)), "rmse_seed_std": float(np.std(per_seed)),
        "sam_deg": float(np.nanmean(S)), "sam_std": float(np.nanstd(S)), "sam_median": float(np.nanmedian(S)),
        "psnr": float(np.nanmean(P)), "psnr_std": float(np.nanstd(P)),
        "n_pairs": int(R.size), "n_diverged": int((~F).sum()),
        "sec_per_item": float(nfe * sec_per_pixel_nfe),
    }
    for rn, (rR, rS, _) in ref.items():
        m, se, npx = paired(R, rR)
        row[f"d_rmse_vs_{rn}"] = m; row[f"d_rmse_se_vs_{rn}"] = se; row[f"n_pix_paired_vs_{rn}"] = npx
        ms, ses, _ = paired(S, rS)
        row[f"d_sam_vs_{rn}"] = ms; row[f"d_sam_se_vs_{rn}"] = ses
    return row


def t0_summaries(rows, inits):
    """usable_t0: smallest t0 such that it and every larger t0 is not worse than full DDPM by more than
    max(2 SE, 5 % of full-DDPM RMSE); forgotten_t0: smallest t0 such that it and every larger t0 is
    indistinguishable from full DDPM (|diff| <= the same margin), i.e. the start no longer matters."""
    full = [r for r in rows if r["sampler"] == "DDPM-1000 (full)"][0]
    tol = 0.05 * full["rmse_pct"]
    usable, forgotten = {}, {}
    for init in inits:
        rr = sorted([r for r in rows if r["init"] == init and r["sampler"] == "DDPM-trunc"], key=lambda r: -r["t0"])
        if not rr:
            continue
        u = f = None
        okU = okF = True
        for r in rr:
            lim = max(2 * r["d_rmse_se_vs_full_ddpm"], tol)
            okU = okU and r["d_rmse_vs_full_ddpm"] <= lim
            okF = okF and abs(r["d_rmse_vs_full_ddpm"]) <= lim
            if okU:
                u = r["t0"]
            if okF:
                f = r["t0"]
        usable[init], forgotten[init] = u, f
    return usable, forgotten


def print_table(rows):
    print(f"{'init':6s} {'t0':>4s} {'sampler':18s} {'NFE':>5s} {'RMSE% mean+-std':>16s} {'med':>5s} {'div':>3s} "
          f"{'dRMSE vs DDPM':>14s} {'dRMSE vs DDIM100':>16s} {'SAM deg':>13s} {'PSNR':>6s} {'s/item':>7s}")
    for r in rows:
        print(f"{r['init']:6s} {r['t0']:4d} {r['sampler']:18s} {r['nfe']:5d} "
              f"{r['rmse_pct']:7.2f}+-{r['rmse_std']:5.2f} {r['rmse_median']:5.2f} {r['n_diverged']:3d} "
              f"{r['d_rmse_vs_full_ddpm']:+7.2f}+-{r['d_rmse_se_vs_full_ddpm']:4.2f} "
              f"{r['d_rmse_vs_ddim100']:+8.2f}+-{r['d_rmse_se_vs_ddim100']:4.2f} "
              f"{r['sam_deg']:6.2f}+-{r['sam_std']:5.2f} {r['psnr']:6.2f} {r['sec_per_item']:7.1f}")
