"""Independent recomputation of the d = 8 verdict clauses from the per-cube arrays of d8_main.json, plus a cluster
bootstrap over the source images and per-class values. Does not import any repo code."""
import json
import math
import os
import re
import sys
import collections

import numpy as np

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/uq/d8'
E = os.path.join(R, 'ev' + 'al')
OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/recompute-bootstrap'
rec = json.load(open(os.path.join(E, 'd8_main.json')))
ver = json.load(open(os.path.join(R, 'verdict__d8_main.json')))
pc = rec['per_cube']
files = rec['files']
n = len(files)


def img_key(f):
    m = re.match(r'.*?/(P\d{3})/(\d+)_(\d+)\.mat$', f)
    assert m, f
    return m.group(1) + '/' + m.group(2)


keys = [img_key(f) for f in files]
cls = [k.split('/')[0] for k in keys]
assert cls == rec['classes']
uniq = sorted(set(keys))
G = len(uniq)
gid = np.array([uniq.index(k) for k in keys])
cls_of_g = np.array([u.split('/')[0] for u in uniq])
print(f'n = {n}, clusters G = {G}, cubes/class', dict(collections.Counter(cls)), 'images/class', dict(collections.Counter(cls_of_g)))

A = lambda m, k: np.asarray(pc[m][k], dtype=np.float64)
diffs = {
    'C1': A('diffusion', 'crps') - A('varhead', 'crps'),
    'P1': np.abs(A('diffusion', 'single_hf')) - np.abs(A('regressor', 'hf')),
    'P2': A('diffusion', 'single_rmse_pct') - A('baseline', 'rmse_pct'),
    'S1': A('diffusion', 'mean_rmse_pct') - A('regressor', 'rmse_pct'),
}
cov90 = A('diffusion', 'cov90')
cov50 = A('diffusion', 'cov50')
for a in list(diffs.values()) + [cov90, cov50]:
    assert np.all(np.isfinite(a)) and a.shape == (n,)


# ---------------------------------------------------------------- cluster SE, two independent routes
def cr1_loop(d, g):
    """Route 1: explicit loop over clusters, CR1 = G/(G-1) small-sample factor (the paired_stats definition)."""
    mu = d.mean()
    Gs = sorted(set(g.tolist()))
    tot = 0.0
    for c in Gs:
        tot += (d[g == c] - mu).sum() ** 2
    return math.sqrt(len(Gs) / (len(Gs) - 1) * tot) / len(d)


def sandwich(d, g):
    """Route 2: OLS of d on a constant with a cluster sandwich, via matrices; returns CR1 (G/(G-1)) and the Stata form
    G/(G-1) * (N-1)/(N-K)."""
    N = len(d)
    X = np.ones((N, 1))
    beta = np.linalg.lstsq(X, d, rcond=None)[0]
    u = d - X @ beta
    Gs = np.unique(g)
    M = np.zeros((1, 1))
    for c in Gs:
        s = X[g == c].T @ u[g == c]
        M += np.outer(s, s)
    bread = np.linalg.inv(X.T @ X)
    V = bread @ M @ bread
    Gn = len(Gs)
    return math.sqrt(V[0, 0] * Gn / (Gn - 1)), math.sqrt(V[0, 0] * Gn / (Gn - 1) * (N - 1) / (N - 1))


def iid_se(d):
    return d.std(ddof=1) / math.sqrt(len(d))


results = {'recompute': {}, 'bootstrap': {}, 'per_class': {}}
rule = {
    'C1': lambda m, s: m + 2 * s < 0,
    'P1': lambda m, s: m + 2 * s < 0,
    'P2': lambda m, s: m + 2 * s < 0,
    'S1': lambda m, s: abs(m) <= 2 * s,
}
print('\n=== recompute (own code) vs verdict file ===')
for k, d in diffs.items():
    m, se, sc1 = d.mean(), iid_se(d), cr1_loop(d, gid)
    sc2, _ = sandwich(d, gid)
    v = ver[k]
    passed = bool(rule[k](m, sc1))
    results['recompute'][k] = dict(mean=m, se_iid=se, se_cluster=sc1, se_cluster_sandwich=sc2, passed=passed,
                                   verdict_mean=v['mean'], verdict_se=v['se'], verdict_se_cluster=v['se_cluster'], verdict_passed=v['passed'],
                                   rule_stat=(m + 2 * sc1) if k != 'S1' else abs(m) / sc1)
    print(f"{k}: mean {m:+.6f} (verdict {v['mean']:+.6f}, diff {m - v['mean']:.2e}); se {se:.6f} ({v['se']:.6f}); "
          f"se_cl loop {sc1:.6f} sandwich {sc2:.6f} (verdict {v['se_cluster']:.6f}); pass {passed} (verdict {v['passed']}); "
          + (f"mean+2SEcl = {m + 2 * sc1:+.4f}" if k != 'S1' else f"|mean|/SEcl = {abs(m) / sc1:.2f}"))
c90, c50 = cov90.mean(), cov50.mean()
c2pass = bool(0.85 <= c90 <= 0.95 and 0.45 <= c50 <= 0.55)
results['recompute']['C2'] = dict(cov90=c90, cov50=c50, passed=c2pass, verdict=ver['C2'],
                                  cov90_se_cluster=cr1_loop(cov90, gid), cov50_se_cluster=cr1_loop(cov50, gid))
print(f"C2: cov90 {c90:.6f} (verdict {ver['C2']['cov90']:.6f}) SEcl {cr1_loop(cov90, gid):.4f}; cov50 {c50:.6f} "
      f"(verdict {ver['C2']['cov50']:.6f}) SEcl {cr1_loop(cov50, gid):.4f}; pass {c2pass} (verdict {ver['C2']['passed']})")
print(f"   cubes with cov50 in [0.45,0.55]: {np.mean((cov50 >= 0.45) & (cov50 <= 0.55)):.3f}; min cov50 {cov50.min():.3f}; "
      f"cubes with cov50 < 0.55: {(cov50 < 0.55).sum()}; cov90 range {cov90.min():.3f}-{cov90.max():.3f}")

# S2 on the y-swap cubes
ys = rec['yswap']
yfiles = ys['files']
dS2 = np.asarray(ys['swap_full'], dtype=np.float64) - np.asarray(ys['true'], dtype=np.float64)
ykeys = [img_key(f) for f in yfiles]
yuniq = sorted(set(ykeys))
ygid = np.array([yuniq.index(k) for k in ykeys])
mS2, seS2 = dS2.mean(), iid_se(dS2)
seS2c = cr1_loop(dS2, ygid)
s2pass = bool(mS2 - 2 * seS2 > 0)
B = len(yfiles)
partner = [(i - B // 2) % B for i in range(B)]          # torch.roll(k, 0): out[i] = in[(i - k) mod B]
same_img_partner = sum(ykeys[i] == ykeys[partner[i]] for i in range(B))
same_cls_partner = sum(ykeys[i].split('/')[0] == ykeys[partner[i]].split('/')[0] for i in range(B))
results['recompute']['S2'] = dict(mean=mS2, se_iid=seS2, se_cluster=seS2c, clusters=len(yuniq), passed=s2pass, verdict=ver['S2'],
                                  n_positive=int((dS2 > 0).sum()), median=float(np.median(dS2)),
                                  same_image_partners=int(same_img_partner), same_class_partners=int(same_cls_partner),
                                  rule_stat=mS2 - 2 * seS2)
print(f"S2: mean {mS2:+.6f} (verdict {ver['S2']['mean']:+.6f}); se_iid {seS2:.6f} ({ver['S2']['se']:.6f}); se_cluster ({len(yuniq)} images) {seS2c:.6f}; "
      f"pass {s2pass} (verdict {ver['S2']['passed']}); mean-2SE = {mS2 - 2 * seS2:+.4f}; "
      f"{(dS2 > 0).sum()}/32 positive; median {np.median(dS2):+.4f}; same-image partners {same_img_partner}, same-class partners {same_cls_partner}")
ytrue = np.asarray(ys['true']); yreg = np.asarray(ys['regressor']); ybase = np.asarray(ys['baseline'])
print(f"   y-swap cubes: true(meanK) {ytrue.mean():.4f}, swap_full {np.mean(ys['swap_full']):.4f}, regressor {yreg.mean():.4f}, baseline {ybase.mean():.4f}")
# consistency: y-swap 'regressor' and 'baseline' vs the per-cube gate arrays
idx_of = {f: i for i, f in enumerate(files)}
yi = np.array([idx_of[f] for f in yfiles])
print(f"   y-swap regressor vs gate per-cube regressor: max |diff| {np.abs(yreg - A('regressor', 'rmse_pct')[yi]).max():.2e}; "
      f"baseline max |diff| {np.abs(ybase - A('baseline', 'rmse_pct')[yi]).max():.2e}; "
      f"true(meanK, batch 32) vs gate mean_rmse (batch 4) mean diff {np.mean(ytrue - A('diffusion', 'mean_rmse_pct')[yi]):+.4f}, "
      f"max |diff| {np.abs(ytrue - A('diffusion', 'mean_rmse_pct')[yi]).max():.4f}")
# what S1 looks like on the 32 y-swap cubes alone (independent MC draws)
dS1y = ytrue - yreg
print(f"   S1 on the 32 y-swap cubes with their own (batch-32) draws: mean {dS1y.mean():+.4f} +- {iid_se(dS1y):.4f}")
results['recompute']['S1_on_yswap_draws'] = dict(mean=dS1y.mean(), se_iid=iid_se(dS1y))
results['recompute']['diffusion_wins_a'] = bool(results['recompute']['C1']['passed'] and c2pass and results['recompute']['P1']['passed']
                                                and results['recompute']['P2']['passed'] and results['recompute']['S1']['passed'] and s2pass)

# ---------------------------------------------------------------- cluster bootstrap over the 113 images
rng = np.random.default_rng(20261023)
NB = 5000
ng = np.bincount(gid, minlength=G).astype(np.float64)                    # cubes per image
W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(NB)]).astype(np.float64)   # [NB, G] multiplicities
Nstar = W @ ng


def boot_mean_se(d):
    S = np.bincount(gid, weights=d, minlength=G)                          # per-image sums
    mu = (W @ S) / Nstar
    # each drawn copy is its own cluster: G* = G draws
    resid = S[None, :] - ng[None, :] * mu[:, None]                        # [NB, G]
    se = np.sqrt(G / (G - 1) * (W * resid ** 2).sum(1)) / Nstar
    return mu, se


def q(a):
    return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]


boot = {}
print('\n=== cluster bootstrap over the 113 source images, 5000 draws ===')
passes = {}
for k, d in diffs.items():
    mu, se = boot_mean_se(d)
    p = np.array([rule[k](a, b) for a, b in zip(mu, se)])
    passes[k] = p
    stat = (mu + 2 * se) if k != 'S1' else np.abs(mu) / se
    boot[k] = dict(mean_q=q(mu), boot_sd_of_mean=float(mu.std(ddof=1)), analytic_se_cluster=results['recompute'][k]['se_cluster'],
                   rule_stat_q=q(stat), pass_frac=float(p.mean()), se_cluster_q=q(se))
    lab = 'mean+2SEcl' if k != 'S1' else '|mean|/SEcl'
    print(f"{k}: mean 95% [{q(mu)[0]:+.4f}, {q(mu)[1]:+.4f}]  boot SD {mu.std(ddof=1):.4f} vs analytic SEcl {results['recompute'][k]['se_cluster']:.4f};  "
          f"{lab} 95% [{q(stat)[0]:+.4f}, {q(stat)[1]:+.4f}];  pass fraction {p.mean():.4f}")
# S1 extra: how often |mean| is within 2 SEcl, and how often mean < 0 (drift direction)
mu, se = boot_mean_se(diffs['S1'])
boot['S1']['frac_mean_positive'] = float((mu > 0).mean())
# C2
m90 = (W @ np.bincount(gid, weights=cov90, minlength=G)) / Nstar
m50 = (W @ np.bincount(gid, weights=cov50, minlength=G)) / Nstar
p90 = (m90 >= 0.85) & (m90 <= 0.95)
p50 = (m50 >= 0.45) & (m50 <= 0.55)
passes['C2'] = p90 & p50
boot['C2'] = dict(cov90_q=q(m90), cov50_q=q(m50), cov90_in_range_frac=float(p90.mean()), cov50_in_range_frac=float(p50.mean()),
                  pass_frac=float(passes['C2'].mean()), cov90_max=float(m90.max()), cov50_min=float(m50.min()))
print(f"C2: cov90 95% [{q(m90)[0]:.4f}, {q(m90)[1]:.4f}] (in [0.85,0.95] in {p90.mean():.4f}); cov50 95% [{q(m50)[0]:.4f}, {q(m50)[1]:.4f}] "
      f"(min over draws {m50.min():.4f}; in [0.45,0.55] in {p50.mean():.4f}); pass fraction {passes['C2'].mean():.4f}")

# S2 inside the same joint draw: y-swap cubes inherit the multiplicity of their source image (113-image draw)
gate_g_of_y = np.array([uniq.index(k) for k in ykeys])                    # y-swap image -> gate cluster index
Wy = W[:, gate_g_of_y]                                                    # [NB, 32] multiplicity of each y-swap cube
Ny = Wy.sum(1)
ok = Ny >= 2
muy = np.where(ok, (Wy @ dS2) / np.maximum(Ny, 1), np.nan)
var = np.where(ok, (Wy * (dS2[None, :] - muy[:, None]) ** 2).sum(1) / np.maximum(Ny - 1, 1), np.nan)
sey = np.sqrt(var / np.maximum(Ny, 1))
pS2 = ok & (muy - 2 * sey > 0)
passes['S2'] = pS2
boot['S2_joint113'] = dict(mean_q=[float(np.nanquantile(muy, 0.025)), float(np.nanquantile(muy, 0.975))],
                           rule_stat_q=[float(np.nanquantile(muy - 2 * sey, 0.025)), float(np.nanquantile(muy - 2 * sey, 0.975))],
                           pass_frac=float(pS2.mean()), n_yswap_cubes_q=q(Ny), frac_mean_positive=float(np.nanmean(muy > 0)))
print(f"S2 (joint 113-image draw; y-swap cubes weighted by their image's multiplicity): mean 95% {boot['S2_joint113']['mean_q']}, "
      f"mean-2SE 95% {boot['S2_joint113']['rule_stat_q']}, pass fraction {pS2.mean():.4f}, mean>0 in {np.nanmean(muy > 0):.3f}, "
      f"n y-swap cubes per draw 95% {q(Ny)}")
# S2 own bootstrap: resample the 29 y-swap source images
Gy = len(yuniq)
ngy = np.bincount(ygid, minlength=Gy).astype(np.float64)
Sy = np.bincount(ygid, weights=dS2, minlength=Gy)
Wy2 = np.stack([np.bincount(rng.integers(0, Gy, Gy), minlength=Gy) for _ in range(NB)]).astype(np.float64)
Wc = Wy2[:, ygid]                                                         # per-cube multiplicity
Nc = Wc.sum(1)
mu2 = (Wc @ dS2) / Nc
se2 = np.sqrt((Wc * (dS2[None, :] - mu2[:, None]) ** 2).sum(1) / (Nc - 1) / Nc)
boot['S2_own29'] = dict(mean_q=q(mu2), rule_stat_q=q(mu2 - 2 * se2), pass_frac=float((mu2 - 2 * se2 > 0).mean()),
                        frac_mean_positive=float((mu2 > 0).mean()))
print(f"S2 (own bootstrap over the 29 y-swap images): mean 95% [{q(mu2)[0]:+.4f}, {q(mu2)[1]:+.4f}], mean-2SE 95% "
      f"[{q(mu2 - 2 * se2)[0]:+.4f}, {q(mu2 - 2 * se2)[1]:+.4f}], pass fraction {(mu2 - 2 * se2 > 0).mean():.4f}, mean>0 in {(mu2 > 0).mean():.3f}")

cal = passes['C1'] & passes['C2']
per = passes['P1'] & passes['P2']
wins = cal & per & passes['S1'] & passes['S2']
boot['joint'] = dict(calibration_pass_frac=float(cal.mean()), perceptual_pass_frac=float(per.mean()), wins_frac=float(wins.mean()))
print(f"joint: calibration pass {cal.mean():.4f}, perceptual pass {per.mean():.4f}, diffusion wins (a) {wins.mean():.4f}")
results['bootstrap'] = boot

# ---------------------------------------------------------------- per class
print('\n=== per class (cluster SE within class; bootstrap within class, 5000 draws) ===')
for c in ('P022', 'P023', 'P024'):
    sel = np.array([x == c for x in cls])
    g_c = gid[sel]
    uc = np.unique(g_c)
    remap = {u: i for i, u in enumerate(uc)}
    gl = np.array([remap[u] for u in g_c])
    Gc = len(uc)
    nc = np.bincount(gl, minlength=Gc).astype(np.float64)
    Wc_ = np.stack([np.bincount(rng.integers(0, Gc, Gc), minlength=Gc) for _ in range(NB)]).astype(np.float64)
    Ncs = Wc_ @ nc
    row = dict(n=int(sel.sum()), images=int(Gc))
    line = f"{c} (n {sel.sum()}, {Gc} images):"
    for k, d in diffs.items():
        dc = d[sel]
        m, se, sc = dc.mean(), iid_se(dc), cr1_loop(dc, gl)
        S = np.bincount(gl, weights=dc, minlength=Gc)
        mub = (Wc_ @ S) / Ncs
        seb = np.sqrt(Gc / (Gc - 1) * (Wc_ * (S[None, :] - nc[None, :] * mub[:, None]) ** 2).sum(1)) / Ncs
        pf = float(np.mean([rule[k](a, b) for a, b in zip(mub, seb)]))
        row[k] = dict(mean=m, se_iid=se, se_cluster=sc, passed=bool(rule[k](m, sc)), boot_mean_q=q(mub), boot_pass_frac=pf)
        line += f"  {k} {m:+.4f}+-{sc:.4f}({'P' if rule[k](m, sc) else 'F'}, boot [{q(mub)[0]:+.3f},{q(mub)[1]:+.3f}] pass {pf:.2f})"
    row['C2'] = dict(cov90=float(cov90[sel].mean()), cov50=float(cov50[sel].mean()))
    line += f"  cov90 {cov90[sel].mean():.3f} cov50 {cov50[sel].mean():.3f}"
    results['per_class'][c] = row
    print(line)

# component means per class (to see what drives each difference)
print('\n=== component means per class ===')
comp = {'crps_diff': A('diffusion', 'crps'), 'crps_vh': A('varhead', 'crps'), 'hf_single': A('diffusion', 'single_hf'),
        'hf_reg': A('regressor', 'hf'), 'rmse_single': A('diffusion', 'single_rmse_pct'), 'rmse_base': A('baseline', 'rmse_pct'),
        'rmse_meanK': A('diffusion', 'mean_rmse_pct'), 'rmse_reg': A('regressor', 'rmse_pct')}
for c in ('P022', 'P023', 'P024', 'all'):
    sel = np.ones(n, bool) if c == 'all' else np.array([x == c for x in cls])
    print(c, ' '.join(f"{k}={v[sel].mean():+.4f}" for k, v in comp.items()))
# sign heterogeneity at the cube level
print('\n=== cube-level sign counts ===')
for k, d in diffs.items():
    print(f"{k}: {(d > 0).sum()} of {n} cubes > 0; images with mean > 0: {(np.bincount(gid, weights=d, minlength=G) / ng > 0).sum()} of {G}")
# S1 if the signed hf were used in P1 (a sensitivity, not the rule)
dsig = A('diffusion', 'single_hf') - A('regressor', 'hf')
print(f"P1 sensitivity: signed hf_single - hf_reg = {dsig.mean():+.4f}; |hf_single| mean {np.abs(A('diffusion', 'single_hf')).mean():.4f}, "
      f"|hf_reg| mean {np.abs(A('regressor', 'hf')).mean():.4f}; |hf_meanK| mean {np.abs(A('diffusion', 'mean_hf')).mean():.4f}; "
      f"cubes where single hf > 0 (over-sharpened): {(A('diffusion', 'single_hf') > 0).sum()}")
# P1 with |hf_meanK| as the alternative (reported only)
d_alt = np.abs(A('diffusion', 'mean_hf')) - np.abs(A('regressor', 'hf'))
print(f"P1 alt (mean-of-K instead of single sample): {d_alt.mean():+.4f} +- {cr1_loop(d_alt, gid):.4f} (cluster)")

with open(os.path.join(OUT, 'recompute_bootstrap.json'), 'w') as f:
    json.dump(results, f, indent=1, default=float)
print('\nwrote', os.path.join(OUT, 'recompute_bootstrap.json'))
