"""Collect every number the Phase 2a results page shows into one compact JSON (docs/analysis/phase2a_page/page_data.json).

Sources (all read-only): the uq_eval records and verdict files under RESULTS_ROOT/uq/, and the saved outputs of the two
verification workflows under docs/analysis/phase2a_verify_d8/ and docs/analysis/phase2a_verify_d4/. The DDPM-1000
reference row is included when its record exists. Run from anywhere:

    python docs/analysis/phase2a_page/extract_page_data.py
"""
import json
import os

R = os.environ.get('RESIDUAL_WARMSTART_RESULTS', '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart')
DOCS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V8 = os.path.join(DOCS, 'phase2a_verify_d8')
V4 = os.path.join(DOCS, 'phase2a_verify_d4')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'page_data.json')

RUNS = {  # key: (record, verdict)
    'd8': ('uq/d8/eval/d8_main.json', 'uq/d8/verdict__d8_main.json'),
    'd8k64': ('uq/d8/eval/d8_fix_k64.json', 'uq/d8/verdict__d8_fix_k64.json'),
    'd4': ('uq/d4/eval/d4_contrast.json', 'uq/d4/verdict__d4_contrast__report.json'),
    'd8ddpm': ('uq/d8/eval/d8_fix_ddpm.json', 'uq/d8/verdict__d8_fix_ddpm__report.json'),
}
METHODS = ('baseline', 'regressor', 'diffusion', 'varhead', 'ensemble', 'b', 'regressor_halves', 'diffusion_halfcentre')


def r4(v):
    return None if v is None else round(float(v), 10)


def load(p):
    with open(p) as f:
        return json.load(f)


def clause_block(v):
    out = {}
    for k in ('C1', 'C2', 'P1', 'P2', 'S1', 'S2'):
        c = v[k]
        out[k] = {kk: (r4(vv) if isinstance(vv, (int, float)) and not isinstance(vv, bool) else vv)
                  for kk, vv in c.items() if kk in ('mean', 'se', 'se_cluster', 'n', 'clusters', 'cov50', 'cov90', 'passed', 'rule')}
    for k in ('calibration_pass', 'perceptual_pass', 'wins', 'report_only', 'K', 'n', 'sampler'):
        if k in v:
            out[k] = v[k]
    return out


def means_block(rec):
    return {m: {k: r4(x) for k, x in d.items()} for m, d in rec['means'].items() if m in METHODS}


page = {'runs': {}}
for key, (rp, vp) in RUNS.items():
    rp_, vp_ = os.path.join(R, rp), os.path.join(R, vp)
    if not (os.path.exists(rp_) and os.path.exists(vp_)):
        page['runs'][key] = None
        continue
    rec, ver = load(rp_), load(vp_)
    page['runs'][key] = dict(record=rp, verdict=vp, n=rec['n'], K=rec['K'], sampler=rec['sampler'], d=rec['d'],
                             sigma_r=r4(rec['sigma_r']), clauses=clause_block(ver), means=means_block(rec),
                             reported={k: ver['reported'][k] for k in ('C3',) if k in ver.get('reported', {})},
                             centre=(rec['provenance'].get('centre') or {}).get('run') + '@' + str((rec['provenance'].get('centre') or {}).get('epoch')))
    if key in ('d8', 'd4'):
        pc = rec['per_cube']
        page['runs'][key]['scatter'] = dict(
            diffusion=[round(x, 10) for x in pc['diffusion']['crps']], varhead=[round(x, 10) for x in pc['varhead']['crps']],
            classes=rec['classes'])

# constant-width Gaussian around x_r, CRPS by width c (sigma = c * sigma_r), all 200 cubes
rc = load(os.path.join(V8, 'condition-blindness', 'ref_c.json'))
page['constgauss'] = {'d8': {c: round(sum(v['crps']) / len(v['crps']), 5) for c, v in rc.items()}}
r4m = load(os.path.join(V4, 'mechanism-d4', 'ref200_d4.json'))['means']
page['constgauss']['d4'] = {k.split('_')[-1]: round(v, 5) for k, v in r4m.items() if k.startswith('crps_homo_')}
page['constgauss']['d4_diff_minus_best'] = {k: r4m[k] for k in ('diff_minus_homo_best_mean', 'diff_minus_homo_best_se_cluster', 'diff_better_than_homo_frac')}

# per-sensor-phase error / spread (8 stored cubes; [0,1] scale -> %)
page['phases'] = {}
page['deciles'] = {}
page['structure'] = {}
for s in ('d8', 'd4'):
    o = load(os.path.join(V4, 'mechanism-d4', f'stored_{s}.json'))
    ph = o['phase']['all']
    page['phases'][s] = dict(err=[round(100 * r['err'], 10) for r in ph['rows']], sd=[round(100 * r['sd_rms'], 10) for r in ph['rows']],
                             vh=[round(100 * r['vh_rms'], 10) for r in ph['rows']], err_ratio=r4(ph['err_ratio']),
                             sd_ratio=r4(ph['sd_ratio']), vh_ratio=r4(ph['vh_ratio']), phase00={k: r4(v) for k, v in ph['phase00'].items()})
    page['deciles'][s] = [{k: r4(r[k]) for k in ('d50', 'd90', 'v50', 'v90', 'er', 'sd', 'sv')} for r in o['strata']['sig_vh_dec']]
    st = o['structure']
    page['structure'][s] = {k: r4(st[k]) for k in ('lag1_true', 'lag1_dev', 'band_true', 'band_dev', 'hf_s0', 'hf_reg', 'hf_xr_plus_iid', 'w1_s0', 'w1_reg', 'w1_xr_plus_iid')}
    page['structure'][s]['esplit_true'] = {k: r4(v) for k, v in st['esplit_true'].items()}
    page['structure'][s]['esplit_samp'] = {k: r4(v) for k, v in st['esplit_samp'].items()}
    page['structure'][s]['exkurt_z'] = r4(o['summary']['exkurt_z'])
    page['structure'][s]['exkurt_zv'] = r4(o['summary']['exkurt_zv'])

# radially averaged spectra of the true residual and of a sample's deviation (d = 8, 8 stored cubes)
psd = load(os.path.join(V8, 'metric-validity', 'psd.json'))
page['psd_d8'] = {k: [round(x, 10) for x in psd[k]] for k in ('P_res_true', 'P_res_samp', 'coh')}

# low-noise denoiser probe at d = 4: x0 RMS error in r units, by timestep
pr = load(os.path.join(V4, 'mechanism-d4', 'probe_d4.json'))
page['probe_d4'] = [dict(t=row['t'], net=r4(row['all'][0]), spec=r4(row['all'][1]), wpos=r4(row['all'][2]), iid=r4(row['all'][3]), vhw=r4(row['all'][4]))
                    for row in pr['rows']]

# S1 and P2 analyses; dose-response summary
dose = load(os.path.join(V4, 'recompute-d4', 'dose.json'))
page['S1'] = {k: {kk: (vv if not isinstance(vv, dict) else {a: r4(b) for a, b in vv.items()}) for kk, vv in v.items()} for k, v in dose['S1'].items()}
page['P2'] = {k: {kk: (r4(vv) if not isinstance(vv, dict) else {a: r4(b) for a, b in vv.items()}) for kk, vv in v.items()} for k, v in dose['P2'].items()}
page['slope'] = {k: {kk: r4(vv) for kk, vv in dose['per_stride'][k]['slope'].items()} for k in ('8', '4')}
syn = load(os.path.join(V4, 'synthesis', 'synth.json'))
page['dose'] = {k: {kk: r4(vv) for kk, vv in syn[k].items() if isinstance(vv, (int, float))} for k in ('d8', 'd4', 'd8_k64')}
page['dose_se'] = {k: {kk: r4(vv) for kk, vv in v.items() if isinstance(vv, (int, float))} for k, v in syn['delta_cluster_se'].items()}
page['dose_paired'] = {k: {kk: r4(vv) for kk, vv in v.items() if isinstance(vv, (int, float))} for k, v in dose['paired_d4_minus_d8'].items()}
page['bootstrap8'] = load(os.path.join(V8, 'recompute-bootstrap', 'recompute_bootstrap.json'))['bootstrap']['joint']
SE_P = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spread_error.json')  # from spread_error_posthoc.py
if os.path.exists(SE_P):
    page['spread_err'] = {k: {m: dict(mor=r4(s['mean_of_ratios']), rom=r4(s['ratio_of_means']['value']),
                                    rom_ci=[r4(x) for x in s['ratio_of_means']['ci95']], r=r4(s['pearson_log']['value']),
                                    r_ci=[r4(x) for x in s['pearson_log']['ci95']], rho=r4(s['spearman']['value']))
                              for m, s in v.items() if m != 'record'} for k, v in load(SE_P).items() if v}
if page['runs'].get('d8ddpm'):
    page['ddpm_hours'] = round(load(os.path.join(R, RUNS['d8ddpm'][0]))['provenance']['seconds'] / 3600, 3)

with open(OUT, 'w') as f:
    json.dump(page, f, separators=(',', ':'))
print('wrote', OUT, os.path.getsize(OUT), 'bytes; runs present:', [k for k, v in page['runs'].items() if v])
