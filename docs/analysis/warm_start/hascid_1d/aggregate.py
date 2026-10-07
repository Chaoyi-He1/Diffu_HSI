"""Step 4: score all finished chunks and write results.json (+ a printed table).

Metrics are computed on clip(output, -1, 1) against the loader-scale target, converted to [0,1].
Full DDPM from pure noise diverges (NaN) for a few pixel/seed pairs; every statistic is taken over finite pairs and
the number of diverged pairs is reported per row.  Paired differences are taken against full DDPM (its finite pairs)
and against full DDIM-100 (always finite).
"""
import os, sys, json, glob, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import OUT, metrics
from agg_core import make_row, t0_summaries, print_table

SELF_NFE = 5


def parse(name):
    """config name -> (init, t0, sampler)"""
    if name == "full_ddpm":
        return "noise", 999, "DDPM-1000 (full)"
    m = re.fullmatch(r"ddim(\d+)", name)
    if m:
        return "noise", 999, f"DDIM-{m.group(1)} (full)"
    m = re.fullmatch(r"(\w+?)_ddpm_t(\d+)", name)
    if m:
        return m.group(1), int(m.group(2)), "DDPM-trunc"
    m = re.fullmatch(r"(\w+?)_ddim(\d+)_t(\d+)", name)
    if m:
        return m.group(1), int(m.group(3)), f"DDIM{m.group(2)}-trunc"
    raise ValueError(name)


def main():
    files = sorted(f for f in glob.glob(os.path.join(OUT, "chunks", "chunk_*.npz")) if not f.endswith(".tmp.npz"))
    assert files, "no finished chunks"
    ch = [dict(np.load(f)) for f in files]
    # optional add-on chunks (the degraded 'linreg' init), merged only if present for every main chunk
    add = [os.path.join(OUT, "chunks_addon", os.path.basename(f)) for f in files]
    addon_used = all(os.path.exists(a) for a in add)
    if addon_used:
        for c, a in zip(ch, add):
            da = dict(np.load(a))
            assert np.array_equal(da["idx"], c["idx"])
            for k, v in da.items():
                if k.startswith(("out/", "est/", "nfe/")):
                    c[k] = v
            c["pixel_nfe"] = int(c["pixel_nfe"]) + int(da["pixel_nfe"])
            c["wall"] = float(c["wall"]) + float(da["wall"])
    else:
        print("add-on chunks missing for some main chunks -> add-on (linreg) not included")
    X = np.concatenate([c["X"] for c in ch]); idx = np.concatenate([c["idx"] for c in ch])
    N = len(X)
    pix_nfe = sum(int(c["pixel_nfe"]) for c in ch); wall = sum(float(c["wall"]) for c in ch)
    sec_per_pixel_nfe = wall / pix_nfe
    keys = [k for k in ch[0] if k.startswith("out/") or k.startswith("est/")]
    seeds = sorted({int(k.split("/")[1][1:]) for k in keys})
    confs = sorted({k.split("/", 2)[2] for k in keys if k.startswith("out/")})
    ests = sorted({k.split("/", 2)[2] for k in keys if k.startswith("est/")})

    def score(prefix, name):
        R, S, P = [], [], []
        for s in seeds:
            out = np.concatenate([c[f"{prefix}/s{s}/{name}"] for c in ch])
            with np.errstate(all="ignore"):
                r, a, p = metrics(np.clip(out, -1, 1), X)
            R.append(r); S.append(a); P.append(p)
        return np.array(R), np.array(S), np.array(P)          # [n_seeds, N]; NaN where the sampler diverged

    ref = {"full_ddpm": score("out", "full_ddpm"), "ddim100": score("out", "ddim100")}
    rows = []
    for name in confs:
        init, t0, sampler = parse(name)
        nfe = int(ch[0][f"nfe/s{seeds[0]}/{name}"]) + (SELF_NFE if init == "self" else 0)
        rows.append(make_row(init, t0, sampler, nfe, *score("out", name), ref, sec_per_pixel_nfe))
    for e in ests:
        rows.append(make_row(e, 0, "estimate only", SELF_NFE if e == "self" else 0, *score("est", e), ref,
                             sec_per_pixel_nfe))
    order_s = {"DDPM-1000 (full)": 0, "DDIM-100 (full)": 1, "DDIM-50 (full)": 2, "estimate only": 3,
               "DDPM-trunc": 4, "DDIM20-trunc": 5, "DDIM10-trunc": 6}
    order_i = {"noise": 0, "lin": 1, "gauss": 2, "self": 3, "linreg": 4, "mean": 5}
    rows.sort(key=lambda r: (order_i[r["init"]], order_s[r["sampler"]], -r["t0"]))
    usable, forgotten = t0_summaries(rows, ["lin", "gauss", "self", "linreg", "mean"])

    # which pixel/seed pairs diverged under full DDPM
    div = {}
    for s in seeds:
        out = np.concatenate([c[f"out/s{s}/full_ddpm"] for c in ch])
        div[f"seed{s}"] = np.where(~np.isfinite(out).all(1))[0].tolist()

    # estimate-only numbers on all 4096 test pixels (no network needed)
    d = np.load(os.path.join(OUT, "data", "test_pixels.npz")); pr = np.load(os.path.join(OUT, "data", "gauss_prior.npz"))
    Xa, Ya = d["X"], d["Y"].astype(np.float64)
    xl = pr["mu"] + (Ya - (pr["H"] @ pr["mu"] + pr["b"])) @ pr["K"].T
    rl, sl, pl = metrics(np.clip(xl[:, :160], -1, 1), Xa)
    rm_, sm_, pm_ = metrics(np.repeat(pr["mu"][None, :160], len(Xa), 0), Xa)
    est_all = {"n_pixels": int(len(Xa)),
               "lin": {"rmse_pct": float(rl.mean()), "rmse_std": float(rl.std()), "sam_deg": float(sl.mean()),
                       "psnr": float(pl.mean())},
               "mean": {"rmse_pct": float(rm_.mean()), "rmse_std": float(rm_.std()), "sam_deg": float(sm_.mean()),
                        "psnr": float(pm_.mean())}}

    res = {"addon_linreg_included": bool(addon_used), "n_pixels": N, "seeds": seeds,
           "pixel_indices_into_test_pixels_npz": idx.tolist(),
           "sec_per_pixel_nfe": sec_per_pixel_nfe, "total_pixel_nfe": pix_nfe, "total_wall_s": wall,
           "full_ddpm_diverged_pixels": div,
           "rows": rows, "usable_t0_ddpm": usable, "forgotten_t0_ddpm": forgotten,
           "estimate_only_all_4096": est_all,
           "cond_scale": json.load(open(os.path.join(OUT, "data", "cond_scale.json"))),
           "s_validation": json.load(open(os.path.join(OUT, "data", "s_validation.json")))}
    rd = os.path.join(OUT, "data", "rdevice_compare.json")
    if os.path.exists(rd):
        res["linear_estimate_sensor_compare"] = json.load(open(rd))
    diag = {}
    for f in sorted(glob.glob(os.path.join(OUT, "data", "diag_full_ddpm_c*_s*.json"))):
        dj = json.load(open(f))
        diag[os.path.basename(f)] = {m: {"first_t_with_abs_x_gt_50_or_nonfinite": v["first_t_with_abs_x_gt_50_or_nonfinite"],
                                         "final_rmse_pct": v["final_rmse_pct"],
                                         "max_abs_x_and_eps_and_x0hat_every_50": [
                                             (tr["t"], tr["max_abs_x"], tr["max_abs_eps"], tr["max_abs_x0hat"])
                                             for tr in v["trace"] if tr["t"] % 50 == 0 and 600 <= tr["t"] <= 900]}
                                     for m, v in dj.items()}
    if diag:
        res["full_ddpm_divergence_diagnosis"] = diag
    vf = os.path.join(OUT, "data", "verify_chunk0.json")
    if os.path.exists(vf):
        res["rerun_check_chunk0"] = json.load(open(vf))
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1)

    print(f"N = {N} pixels x {len(seeds)} seeds; {sec_per_pixel_nfe*1000:.1f} ms per pixel-NFE (CPU, shared)")
    print_table(rows)
    print("usable t0 (DDPM-trunc not worse than full DDPM by > max(2SE,5%), monotone):", usable)
    print("forgotten t0 (DDPM-trunc indistinguishable from full DDPM, monotone):", forgotten)
    print("full DDPM diverged (NaN) pixel indices:", div)
    print("estimate only on all 4096 test pixels:", json.dumps(est_all))


if __name__ == "__main__":
    main()
