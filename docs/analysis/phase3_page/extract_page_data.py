"""Collect every number the Phase 3 results page shows into one compact JSON (docs/analysis/phase3_page/page_data.json).

Sources (all read-only, under SENSOR_DIR = RESULTS_ROOT/sensor unless stated):
  eval/*.json                      the sd_eval records: e1_dev1_seq_<snr>_{main,C,B}, e2_{device2,ph5}_sequential_<snr>_
                                   {main,C,B}, e3_device1_snapshot_<snr>_{main,C,B}; B records may be missing (pending)
  eval/vpred_prior/*.json          the archived B records of the failed v-prediction prior (E1 60 / 50 dB)
  stage0/summary.json, stage0/L__<device>__<mode>__<snr>.json   Stage 0 (MMSE curves, raw-vs-normalised check, angles)
  toy_guidance.json                the ΠGDM toy test (falls back to this checkout's results/ copy)
  config.json                      P_y, training sha, family, demosaic error table
  verdict__*.json                  the sd_verdict output (--verdict; default: the final e123 verdict when present)
  RESULTS_ROOT/runs/<run>/{train_log.jsonl,args.json}            the Phase B trainings and the three prior pilots
plus the device-1 response R (data_loader.sensor_model.load_sensor) and the B-prior x̂₀-error probe curves, which live in
no JSON and are embedded below (B_PRIOR_PROBE). Paired differences that involve B are computed here on the files common
to the B record (50 cubes) and the 200-cube records, with scripts/paired_stats.py::cluster_se. Run from anywhere:

    python docs/analysis/phase3_page/extract_page_data.py [--verdict verdict__e123__report.json]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from data_loader.warmstart_paths import DATA_ROOT, RESULTS_ROOT  # noqa: E402
from data_loader.sensor_config import SENSOR_DIR  # noqa: E402
from scripts.paired_stats import cluster_se, source_image  # noqa: E402

OUT = os.path.join(HERE, 'page_data.json')
EVAL = os.path.join(SENSOR_DIR, 'eval')
RUNS = os.path.join(RESULTS_ROOT, 'runs')

# Experiments and their record stems (sd_eval names). B is expected everywhere except PH5 (its B rows were dropped,
# spec drop order + the results note's ruling on B's cost).
EXPERIMENTS = [
    ('E1', 'device1', 'sequential', [60, 50, 40, 30, 20], 'e1_dev1_seq_{snr}'),
    ('E2', 'device2', 'sequential', [40, 30, 20], 'e2_device2_sequential_{snr}'),
    ('E2', 'ph5', 'sequential', [40, 30, 20], 'e2_ph5_sequential_{snr}'),
    ('E3', 'device1', 'snapshot', [60, 40, 30, 20], 'e3_device1_snapshot_{snr}'),
    ('E4', 'device2', 'snapshot', [30], 'e4_device2_snapshot_{snr}'),   # report-only (spec §6.1)
]
B_DROPPED = {'ph5'}
RMSE_KEY = {'L': 'mean_rmse_pct', 'L_demosaic': 'mean_rmse_pct', 'V': 'mean_rmse_pct', 'C': 'mean_rmse_pct',
            'B': 'mean_rmse_pct', 'D': 'rmse_pct', 'C1': 'rmse_pct'}
GAUSS_KEYS = ('mean_rmse_pct', 'crps', 'cov50', 'cov90', 'ause', 'spread_ratio', 'mean_hf', 'mean_sam_deg')
POINT_KEYS = ('rmse_pct', 'hf', 'sam_deg', 'w1')
SAMPLE_KEYS = ('mean_rmse_pct', 'single_rmse_pct', 'crps', 'cov50', 'cov90', 'ause', 'spread_ratio', 'mean_hf',
               'single_hf', 'mean_sam_deg')
KEEP = {'L': GAUSS_KEYS, 'L_demosaic': GAUSS_KEYS, 'V': GAUSS_KEYS, 'D': POINT_KEYS, 'C1': POINT_KEYS,
        'C': SAMPLE_KEYS, 'B': SAMPLE_KEYS}

# Results note, section "B — the guided unconditional prior fails on its prior" (and "B's retrained prior"):
# x̂₀-estimate error (% of range) of each prior against the diffusion timestep t, 16 gate cubes at 60 dB.
B_PRIOR_PROBE = {
    't': [25, 50, 100, 150, 200, 299, 400, 500, 700, 900],
    'series': [
        {'key': 'blind', 'name': 'input-blind predictor E[v | t]', 'err': [4.7, 8.6, 15.4, 20.9, 24.6, 26.8, 24.7, 21.5, 18.6, 18.3]},
        {'key': 'vpred', 'name': 'v-prediction prior (epoch 100)', 'err': [4.4, 8.0, 14.2, 19.1, 22.1, 22.8, 18.6, 13.1, 8.2, 14.6]},
        {'key': 'g0', 'name': 'uniform-weighting pilot (epoch 10)', 'err': [4.7, 8.5, 15.3, 20.6, 23.9, 24.9, 20.7, 15.3, 10.6, 14.9]},
        {'key': 'tt', 'name': 'time-token pilot (epoch 10)', 'err': [4.7, 8.5, 15.3, 20.5, 23.8, 24.6, 20.1, 14.2, 8.9, 15.0]},
        {'key': 'x0pilot', 'name': 'x₀-prediction pilot (epoch 10)', 'err': [3.3, 3.4, 3.4, 3.5, 3.6, 4.0, 4.6, 5.7, 10.6, 23.5]},
        {'key': 'x0', 'name': 'x₀-prediction prior (epoch 100)', 'err': [2.3, 2.3, 2.5, 2.6, 2.8, 3.2, 3.8, 4.6, 7.9, 15.3]},
    ],
    'mean_cube': 17.15,
    'saturation_pct': {'vpred': 29.7, 'x0': 0.0},
}

# Phase 0's noise-free MMSE reference (legacy axis, s = 3e-7 regulariser, no noise): quoted once on the page as the
# number the 60 dB noise floor replaces (spec §4.3 / §6.3; not directly comparable).
LEGACY_NOISE_FREE_RMSE_PCT = 0.266

RUN_TABLE = [  # run, role, pre-registered selection, failed?
    ('uncond_x0_b64', 'B prior (x₀-prediction)', 100, False),
    ('cond_fam_b64', 'C, conditional diffusion', 100, False),
    ('regress_fam_b64', 'D, regressor', 85, False),
    ('varhead_fam_b64', 'V, variance head', 38, False),
    ('uncond_b64', 'B prior (v-prediction, failed)', 100, True),
]
PILOTS = ['uncond_g0pilot_b64', 'uncond_tt_pilot_b64', 'uncond_x0pilot_b64']
VERDICT_DEFAULTS = ['verdict__e123.json', 'verdict__e123__report.json', 'verdict__e123_noB__report.json']


def rd(v, n=6):
    if v is None:
        return None
    v = float(v)
    return None if not np.isfinite(v) else float(f'{v:.{n}g}')


def load(p):
    with open(p) as f:
        return json.load(f)


def pick(means, keys):
    return {k: rd(means[k]) for k in keys if k in means}


def created_of(rec):
    return rec.get('provenance', {}).get('created')


def b_pred_type(rec):
    p = rec.get('provenance', {})
    t = p.get('b_prediction_type') or rec.get('b_prediction_type')
    if t is None:
        t = p.get('checkpoints', {}).get('uncond', {}).get('prediction_type')
    return t or 'v'


def paired(a_rec, a_m, b_rec, b_m, metric_a, metric_b, limit=None):
    """a - (limit or 1) * b on the files common to both records (a's order), clustered SE over source images."""
    ia = {f: i for i, f in enumerate(a_rec['files'])}
    ib = {f: i for i, f in enumerate(b_rec['files'])}
    common = [f for f in a_rec['files'] if f in ib]
    if len(common) < 2:
        return None
    x = np.array([a_rec['per_cube'][a_m][metric_a][ia[f]] for f in common], dtype=np.float64)
    y = np.array([b_rec['per_cube'][b_m][metric_b][ib[f]] for f in common], dtype=np.float64)
    d = x - y
    se, G = cluster_se(d, common)
    out = dict(n=len(common), clusters=G, a=rd(x.mean()), b=rd(y.mean()), diff=rd(d.mean()), se_cluster=rd(se),
               ratio=rd(x.mean() / y.mean()))
    if limit is not None:
        dl = x - limit * y
        sel, _ = cluster_se(dl, common)
        out['limit'] = limit
        out['ratio_2se'] = rd((dl.mean() + 2 * sel) / y.mean() + limit)
        out['passed'] = bool(dl.mean() + 2 * sel <= 0)
    return out


def b_block(brec, main, crec, exp, snr, mode):
    """B's means and its paired comparisons on the common files."""
    blk = dict(status='final', n=brec['n'], K=brec.get('K'), steps=brec.get('pigdm_steps'), pred_type=b_pred_type(brec),
               created=created_of(brec), hours=rd(brec['provenance'].get('seconds', 0) / 3600, 4),
               means=pick(brec['means']['B'], KEEP['B']), paired={})
    pr = blk['paired']
    if main is not None:
        for m in ('L', 'D', 'V', 'L_demosaic'):
            if m in main['per_cube']:
                lim = 1.05 if (m == 'D' and exp == 'E1') else (0.85 if (m == 'L_demosaic' and snr == 30) else None)
                pr['rmse_vs_' + m] = paired(brec, 'B', main, m, 'mean_rmse_pct', RMSE_KEY[m], lim)
                if m != 'D':
                    pr['crps_vs_' + m] = paired(brec, 'B', main, m, 'crps', 'crps')
    if crec is not None:
        pr['rmse_vs_C'] = paired(brec, 'B', crec, 'C', 'mean_rmse_pct', 'mean_rmse_pct')
        pr['rmse_vs_C1'] = paired(brec, 'B', crec, 'C1', 'mean_rmse_pct', 'rmse_pct')
        pr['crps_vs_C'] = paired(brec, 'B', crec, 'C', 'crps', 'crps')
    # the other methods' means on B's 50 cubes, for the like-for-like rows
    sub = {}
    fb = set(brec['files'])
    for rec, ms in ((main, ('L', 'L_demosaic', 'D', 'V')), (crec, ('C', 'C1'))):
        if rec is None:
            continue
        idx = [i for i, f in enumerate(rec['files']) if f in fb]
        for m in ms:
            if m in rec['per_cube']:
                sub[m] = {k: rd(np.mean([rec['per_cube'][m][k][i] for i in idx])) for k in KEEP[m] if k in rec['per_cube'][m]}
    blk['on_subset'] = sub
    return blk


def settings_block():
    """-> (settings, final, pending, stale): a B record whose prior is not x0-prediction is 'stale' (listed under
    status.stale, never under final)."""
    out, pending, final, stale = {}, [], [], []
    for exp, dev, mode, snrs, stem in EXPERIMENTS:
        for snr in snrs:
            key = f'{dev}_{mode}_{snr}'
            base = os.path.join(EVAL, stem.format(snr=snr))
            main = load(base + '_main.json') if os.path.exists(base + '_main.json') else None
            crec = load(base + '_C.json') if os.path.exists(base + '_C.json') else None
            s = dict(exp=exp, device=dev, mode=mode, snr=snr, methods={})
            if main is not None:                                     # source images of the 200 cubes (SE clusters)
                s['clusters'] = len({source_image(f) for f in main['files']})
            for rec in (main, crec):
                if rec is None:
                    continue
                s['n'] = rec['n']
                s['sigma'] = rd(rec['setting']['sigma'])
                for m, mm in rec['means'].items():
                    s['methods'][m] = pick(mm, KEEP.get(m, mm.keys()))
            for kind, rec in (('main', main), ('C', crec)):
                (final if rec is not None else pending).append(f'{stem.format(snr=snr)}_{kind}')
            if dev in B_DROPPED:
                s['B'] = dict(status='dropped')
            elif os.path.exists(base + '_B.json'):
                brec = load(base + '_B.json')
                s['B'] = b_block(brec, main, crec, exp, snr, mode)
                if s['B']['pred_type'] != 'x0':
                    s['B']['status'] = 'stale'
                    stale.append(f'{stem.format(snr=snr)}_B')
                else:
                    final.append(f'{stem.format(snr=snr)}_B')
            else:
                s['B'] = dict(status='pending')
                pending.append(f'{stem.format(snr=snr)}_B')
            out[key] = s
    return out, final, pending, stale


def vpred_block():
    out = {}
    for p in sorted(glob.glob(os.path.join(EVAL, 'vpred_prior', 'e1_dev1_seq_*_B.json'))):
        brec = load(p)
        snr = int(round(brec['setting']['snr_db']))
        stem = os.path.join(EVAL, f'e1_dev1_seq_{snr}')
        main = load(stem + '_main.json') if os.path.exists(stem + '_main.json') else None
        crec = load(stem + '_C.json') if os.path.exists(stem + '_C.json') else None
        out[str(snr)] = b_block(brec, main, crec, 'E1', snr, 'sequential')
        out[str(snr)]['status'] = 'archived'
    return out


def sensor_block():
    from data_loader.sensor_model import load_sensor
    out = {'wavelengths': [rd(w, 5) for w in np.linspace(451, 855, 31)], 'devices': {}}
    for dev in ('device1', 'device2', 'ph5'):
        s = load_sensor(dev, DATA_ROOT)
        sv = np.linalg.svd(s['R'], compute_uv=False)
        out['devices'][dev] = dict(sha256=s['sha256'], sv=[rd(v, 5) for v in sv], sv_rel=[rd(v / sv[0], 5) for v in sv],
                                   n_below_1pct=int((sv < 0.01 * sv[0]).sum()), cond=rd(sv[0] / sv[-1], 4),
                                   n_states_total=s['n_states_total'])
        if dev == 'device1':
            out['R_device1'] = [[rd(v, 4) for v in row] for row in s['R']]
    return out


def stage0_block():
    st = os.path.join(SENSOR_DIR, 'stage0')
    s = load(os.path.join(st, 'summary.json'))
    ang = {}
    for k, v in s['principal_angles_deg'].items():
        nz = [rd(a, 4) for a in v if a > 1e-6]
        ang[k] = dict(n=len(v), nonzero=nz, max=rd(max(v), 4))
    table = {}
    for p in sorted(glob.glob(os.path.join(st, 'L__*__*__*.json'))):
        _, dev, mode, snr = os.path.basename(p)[:-5].split('__')
        r = load(p)
        table.setdefault(dev, {}).setdefault(mode, {})[str(int(float(snr)))] = {
            m: pick(mm, ('mean_rmse_pct', 'crps', 'cov50', 'cov90', 'ause')) for m, mm in r['means'].items()}
    return dict(snrs=s['snrs'], L_curves=s['L_curves'], curves_raw_device1=s['curves_raw_device1'],
                shape_check=s['shape_check'], normalisation_check_passed=s['normalisation_check_passed'],
                principal_angles=ang, sanity_reference_rmse_pct=rd(s['sanity_reference_rmse_pct']), P_y=rd(s['P_y']),
                P_y_raw=rd(s['P_y_raw']), table=table, created=s['provenance']['created'])


def toy_block():
    for p in (os.path.join(SENSOR_DIR, 'toy_guidance.json'),
              os.path.join(REPO, 'results', 'residual_warmstart', 'sensor', 'toy_guidance.json')):
        if os.path.exists(p):
            t = load(p)
            rows = [{k: (rd(v) if isinstance(v, float) else v) for k, v in r.items() if k != 'seconds'} for r in t['results']]
            rel = os.path.relpath(p, REPO) if p.startswith(REPO + os.sep) else os.path.join(
                'results', 'residual_warmstart', os.path.relpath(p, RESULTS_ROOT))
            return dict(path=rel, rows=rows, gate=t['gate'],
                        steps_for_B=t['steps_for_B'], verdict=t['verdict'])
    return None


def config_block():
    c = load(os.path.join(SENSOR_DIR, 'config.json'))
    from data_loader.sensor_config import training_sha
    e2 = np.array(c['e2_demosaic'])
    return dict(P_y=rd(c['P_y']), sha=training_sha(c), family=c['family'], s_floor01=c['s_floor01'],
                snr_floor_db=c['snr_floor_db'], tile=c['tile'], n_files_P_y=c['n_files_P_y'],
                e2_demosaic=dict(min=rd(e2.min()), median=rd(np.median(e2)), max=rd(e2.max()), n=len(e2)))


def b_is_valid_in(verdict):
    """The verdict's B clauses are current only if every B record its manifest names is an x₀-prior record written
    before the verdict (the e123_noB verdict read the archived v-prediction records)."""
    try:
        man = load(verdict['manifest'])
    except (KeyError, OSError):
        return False, 'manifest not readable'
    bs = [r for recs in man['records'].values() for r in recs if r.endswith('_B')]
    if not bs:
        return False, 'no B records in the manifest'
    for r in bs:
        p = os.path.join(EVAL, r + '.json')
        if not os.path.exists(p):
            return False, f'{r} is not in eval/ (archived with the v-prediction prior or not yet written)'
        rec = load(p)
        if b_pred_type(rec) != 'x0' or (created_of(rec) or '') > (verdict.get('created') or ''):
            return False, f'{r} was replaced after the verdict (the verdict read the v-prediction prior record)'
    return True, ''


def verdict_block(name):
    p = name if os.path.isabs(name) else os.path.join(SENSOR_DIR, name)
    v = load(p)
    ok, why = b_is_valid_in(v)
    keep = {k: v[k] for k in ('T1', 'T2', 'T3', 'T4', 'sanity', 'counts', 'main_line', 'tie_break', 'report_only', 'missing',
                              'thresholds', 'name', 'created', 'min_common') if k in v}
    return dict(file=os.path.basename(p), b_valid=ok, b_invalid_reason=why, **keep)


def runs_block():
    def one(run):
        d = os.path.join(RUNS, run)
        if not os.path.exists(os.path.join(d, 'train_log.jsonl')):
            return None
        L = [json.loads(x) for x in open(os.path.join(d, 'train_log.jsonl')) if x.strip()]
        a = load(os.path.join(d, 'args.json'))
        vals = {r['epoch']: r['val_loss'] for r in L if 'val_loss' in r}
        return dict(run=run, mode=a.get('mode'), prediction_type=a.get('prediction_type'), loss_type=a.get('loss_type'),
                    snr_gamma=a.get('snr_gamma'), seed=a.get('seed'), epochs=L[-1]['epoch'], planned=a.get('num_epochs'),
                    sec_per_epoch=rd(np.mean([r['sec'] for r in L]), 4), hours=rd(L[-1]['elapsed_hours'], 4),
                    done=os.path.exists(os.path.join(d, 'DONE')), vals=vals,
                    skipped=int(sum(r.get('skipped_nonfinite', 0) for r in L)))
    rows = []
    for run, role, sel, failed in RUN_TABLE:
        r = one(run)
        if r is None:
            continue
        r.update(role=role, selected=sel, failed=failed, val_at_sel=rd(r['vals'].get(sel)))
        r['vals'] = {str(k): rd(v) for k, v in r['vals'].items()}
        rows.append(r)
    pilots = [p for p in (one(x) for x in PILOTS) if p]
    for p in pilots:
        p.pop('vals')
    return rows, pilots


def eval_hours():
    """GPU-h per record kind. 'B' and the B record count hold x0-prior records only; a B record in eval/ that is not
    x0-prior (stale, not yet archived) counts under 'B_stale'; the archived v-prediction records under 'B_vpred'."""
    tot = {'main': 0.0, 'C': 0.0, 'B': 0.0, 'B_stale': 0.0, 'B_vpred': 0.0}
    nb = 0
    for p in glob.glob(os.path.join(EVAL, '*.json')):
        r = load(p)
        kind = p[:-5].rsplit('_', 1)[1]
        if kind == 'B' and b_pred_type(r) != 'x0':
            kind = 'B_stale'
        tot[kind] = tot.get(kind, 0.0) + r['provenance'].get('seconds', 0) / 3600
        nb += kind == 'B'
    for p in glob.glob(os.path.join(EVAL, 'vpred_prior', '*.json')):
        tot['B_vpred'] += load(p)['provenance'].get('seconds', 0) / 3600
    return {k: rd(v, 4) for k, v in tot.items()}, nb


def facts_block(S, stage0, verdict, sensor, runs=()):
    """Derived numbers the page's prose reads (each one bound to the records)."""
    f = {}
    # the training seed (the page says "each network was trained once (seed N)": None unless every run shares one seed)
    seeds = {r.get('seed') for r in runs}
    f['train_seed'] = seeds.pop() if len(seeds) == 1 else None
    # source images behind the SEs: the 200 gate cubes, and B's 50-cube subset (paired with the 200 on the common files)
    s30 = S.get('device1_sequential_30', {})
    f['clusters_gate'] = s30.get('clusters')
    pb = (s30.get('B') or {}).get('paired', {}).get('rmse_vs_L') if (s30.get('B') or {}).get('status') == 'final' else None
    f['clusters_B'] = pb['clusters'] if pb else None
    # C's T2 at 30 dB as the verdict scored it (the KPI tile's "(T2 passed / failed)")
    c30 = (((verdict.get('T2') or {}).get('C') or {}).get('30.0') or {})
    f['C_T2_30_word'] = None if 'passed' not in c30 else ('passed' if c30['passed'] else 'failed')
    e1 = {snr: S[f'device1_sequential_{snr}']['methods'] for snr in (60, 50, 40, 30, 20)}
    f['L_floor'] = stage0['sanity_reference_rmse_pct']
    for snr, m in e1.items():
        f[f'L_{snr}'] = m['L']['mean_rmse_pct']
        f[f'D_{snr}'] = m['D']['rmse_pct']
        f[f'C_{snr}'] = m['C']['mean_rmse_pct']
        f[f'C1_{snr}'] = m['C1']['rmse_pct']
        f[f'Csingle_{snr}'] = m['C']['single_rmse_pct']
        f[f'D_below_L_{snr}'] = rd(100 * (1 - m['D']['rmse_pct'] / m['L']['mean_rmse_pct']), 4)
        f[f'C_crps_below_L_{snr}'] = rd(100 * (1 - m['C']['crps'] / m['L']['crps']), 4)
        f[f'C_crps_below_V_{snr}'] = rd(100 * (1 - m['C']['crps'] / m['V']['crps']), 4)
        f[f'C_hf_single_{snr}'] = m['C']['single_hf']
        f[f'C_hf_mean_{snr}'] = m['C']['mean_hf']
    f['D_best_every_snr'] = all(m['D']['rmse_pct'] <= min(m['L']['mean_rmse_pct'], m['C']['mean_rmse_pct'], m['C1']['rmse_pct'])
                                for m in e1.values())
    f['D_over_L_20'] = rd(e1[20]['D']['rmse_pct'] / e1[20]['L']['mean_rmse_pct'], 4)
    f['V_ause_min'] = min(m['V']['ause'] for m in e1.values())
    f['V_ause_max'] = max(m['V']['ause'] for m in e1.values())
    f['C_ause_min'] = min(m['C']['ause'] for m in e1.values())
    f['C_ause_max'] = max(m['C']['ause'] for m in e1.values())
    f['L_cov90_min'] = min(m['L']['cov90'] for m in e1.values())
    f['L_cov90_max'] = max(m['L']['cov90'] for m in e1.values())
    lo = [s for s in (40, 30, 20)]
    f['C_crps_gain_lo_min'] = min(min(f[f'C_crps_below_L_{s}'], f[f'C_crps_below_V_{s}']) for s in lo)
    f['C_crps_gain_lo_max'] = max(max(f[f'C_crps_below_L_{s}'], f[f'C_crps_below_V_{s}']) for s in lo)
    # sensor
    d1 = sensor['devices']['device1']
    f['d1_n_below'] = d1['n_below_1pct']
    f['d1_informative'] = 30 - d1['n_below_1pct']
    f['d1_cond'] = d1['cond']
    f['d2_n_below'] = sensor['devices']['device2']['n_below_1pct']
    f['ph5_n_below'] = sensor['devices']['ph5']['n_below_1pct']
    # stage 0
    sc = stage0['shape_check']
    f['norm_maxdev_seq'] = sc['sequential']['max_dev']
    f['norm_maxdev_snap'] = sc['snapshot']['max_dev']
    a12 = stage0['principal_angles']['device1_device2']
    f['ang_d1d2_nonzero'] = len(a12['nonzero'])
    f['ang_d1d2_shared'] = a12['n'] - len(a12['nonzero'])
    f['ang_d1d2_max'] = a12['max']
    a15 = stage0['principal_angles']['device1_ph5']
    f['ph5_rank'] = a15['n']
    f['ang_d1ph5_max'] = a15['max']
    Lc = stage0['L_curves']
    f['ph5_L_floor'] = Lc['ph5']['sequential']['60.0']
    f['d2_L_floor'] = Lc['device2']['sequential']['60.0']
    f['d2_L_20'] = Lc['device2']['sequential']['20.0']
    f['L_growth_per10'] = rd((Lc['device1']['sequential']['20.0'] / Lc['device1']['sequential']['60.0']) ** 0.25, 4)
    dm = stage0['table']['device1']['snapshot']
    f['Ldem_30'] = dm['30']['L_demosaic']['mean_rmse_pct']
    f['Ldem_target_30'] = rd(0.85 * f['Ldem_30'])
    f['Lexact_30'] = dm['30']['L']['mean_rmse_pct']
    # E2
    for dev in ('device2', 'ph5'):
        for snr in (40, 30, 20):
            m = S[f'{dev}_sequential_{snr}']['methods']
            f[f'{dev}_D_vs_L_{snr}'] = rd(100 * (m['D']['rmse_pct'] / m['L']['mean_rmse_pct'] - 1), 4)
            f[f'{dev}_C_vs_L_{snr}'] = rd(m['C']['mean_rmse_pct'] / m['L']['mean_rmse_pct'], 4)
            f[f'{dev}_C_cov50_{snr}'] = m['C']['cov50']
            f[f'{dev}_C_cov90_{snr}'] = m['C']['cov90']
    f['ph5_D_worse_min'] = min(f[f'ph5_D_vs_L_{s}'] for s in (40, 30, 20))
    f['ph5_D_worse_max'] = max(f[f'ph5_D_vs_L_{s}'] for s in (40, 30, 20))
    f['ph5_C_ratio_min'] = min(f[f'ph5_C_vs_L_{s}'] for s in (40, 30, 20))
    f['ph5_C_ratio_max'] = max(f[f'ph5_C_vs_L_{s}'] for s in (40, 30, 20))
    f['ph5_C_cov50_max'] = max(f[f'ph5_C_cov50_{s}'] for s in (40, 30, 20))
    f['ph5_C_cov90_max'] = max(f[f'ph5_C_cov90_{s}'] for s in (40, 30, 20))
    # E3
    e3 = {snr: S[f'device1_snapshot_{snr}']['methods'] for snr in (60, 40, 30, 20)}
    f['e3_D_30'] = e3[30]['D']['rmse_pct']
    f['e3_C_30'] = e3[30]['C']['mean_rmse_pct']
    f['e3_D_over_dem_30'] = rd(e3[30]['D']['rmse_pct'] / e3[30]['L_demosaic']['mean_rmse_pct'], 4)
    f['e3_C1_min'] = min(m['C1']['rmse_pct'] for m in e3.values())
    f['e3_C1_max'] = max(m['C1']['rmse_pct'] for m in e3.values())
    f['e3_D_min'] = min(m['D']['rmse_pct'] for m in e3.values())
    f['e3_D_max'] = max(m['D']['rmse_pct'] for m in e3.values())
    f['e3_C_cov50_min'] = min(m['C']['cov50'] for m in e3.values())
    f['e3_C_cov50_max'] = max(m['C']['cov50'] for m in e3.values())
    f['e3_C_cov90_min'] = min(m['C']['cov90'] for m in e3.values())
    f['e3_V_cov50_30'] = e3[30]['V']['cov50']
    f['e3_Lexact_min'] = min(m['L']['mean_rmse_pct'] for m in e3.values())
    f['e3_Lexact_max'] = max(m['L']['mean_rmse_pct'] for m in e3.values())
    # noise floor and the legacy noise-free reference
    f['legacy_noise_free'] = LEGACY_NOISE_FREE_RMSE_PCT
    f['floor_over_legacy'] = rd(f['L_floor'] / LEGACY_NOISE_FREE_RMSE_PCT, 4)
    # best point estimate per E1 SNR on B's common cubes (B and the others on the same files)
    best = {}
    for snr in (60, 50, 40, 30, 20):
        s = S[f'device1_sequential_{snr}']
        if s['B']['status'] != 'final':
            continue
        sub = s['B']['on_subset']
        cand = {'L': sub['L']['mean_rmse_pct'], 'D': sub['D']['rmse_pct'], 'C': sub['C']['mean_rmse_pct'],
                'C1': sub['C1']['rmse_pct'], 'B': s['B']['means']['mean_rmse_pct']}
        best[str(snr)] = min(cand, key=cand.get)
    f['e1_best_common'] = best
    f['B_e1_rmse'] = {str(snr): S[f'device1_sequential_{snr}']['B']['means']['mean_rmse_pct']
                      for snr in (60, 50, 40, 30, 20) if S[f'device1_sequential_{snr}']['B']['status'] == 'final'}
    # B on device2: best point estimate on the common cubes at every SNR?
    d2 = []
    for snr in (40, 30, 20):
        s = S[f'device2_sequential_{snr}']
        if s['B']['status'] == 'final':
            sub = s['B']['on_subset']
            others = min(sub['L']['mean_rmse_pct'], sub['D']['rmse_pct'], sub['C']['mean_rmse_pct'], sub['C1']['rmse_pct'])
            d2.append(s['B']['means']['mean_rmse_pct'] < others)
    f['B_best_dev2_all'] = bool(d2) and all(d2)
    f['B_dev2_n'] = len(d2)
    # E4 (device2 snapshot 30 dB, report-only)
    e4 = S.get('device2_snapshot_30')
    if e4 and e4['methods']:
        m = e4['methods']
        f['e4_Ldem'] = m['L_demosaic']['mean_rmse_pct']
        f['e4_Lexact'] = m['L']['mean_rmse_pct']
        f['e4_D'] = m['D']['rmse_pct']
        f['e4_C'] = m['C']['mean_rmse_pct']
        f['e4_C1'] = m['C1']['rmse_pct']
        f['e4_D_over_dem'] = rd(m['D']['rmse_pct'] / m['L_demosaic']['mean_rmse_pct'], 4)
        f['e4_C_over_dem'] = rd(m['C']['mean_rmse_pct'] / m['L_demosaic']['mean_rmse_pct'], 4)
    return f


def data_as_of(verdict):
    """The newest 'created' stamp among the records and the verdict (not the wall clock, so a rebuild on unchanged data
    writes identical bytes)."""
    stamps = [verdict.get('created') or '']
    for p in glob.glob(os.path.join(EVAL, '*.json')) + glob.glob(os.path.join(EVAL, 'vpred_prior', '*.json')):
        stamps.append(created_of(load(p)) or '')
    return max(stamps).replace('T', ' ')[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--verdict', default=None, help='verdict file name under SENSOR_DIR (default: the first of '
                    + ', '.join(VERDICT_DEFAULTS) + ' that exists, else the newest verdict__*.json)')
    args = ap.parse_args()
    vname = args.verdict
    if vname is None:
        vname = next((v for v in VERDICT_DEFAULTS if os.path.exists(os.path.join(SENSOR_DIR, v))), None)
        if vname is None:
            vs = sorted(glob.glob(os.path.join(SENSOR_DIR, 'verdict__*.json')), key=os.path.getmtime)
            vname = os.path.basename(vs[-1])
    S, final, pending, stale = settings_block()
    stage0 = stage0_block()
    sensor = sensor_block()
    verdict = verdict_block(vname)
    runs, pilots = runs_block()
    eh, nb = eval_hours()
    b_expected = [k for k, s in S.items() if s['B']['status'] != 'dropped']
    b_have = [k for k in b_expected if S[k]['B']['status'] == 'final']
    b_hours = [S[k]['B']['hours'] for k in b_have]
    page = dict(
        generated=data_as_of(verdict),
        status=dict(final=sorted(final), pending=sorted(pending), stale=sorted(stale), b_expected=b_expected, b_have=b_have,
                    b_pending=[k for k in b_expected if k not in b_have], verdict_file=verdict['file'],
                    verdict_b_valid=verdict['b_valid'], verdict_final=('noB' not in verdict['file'])),
        settings=S, b_vpred=vpred_block(), stage0=stage0, sensor=sensor, toy=toy_block(), config=config_block(),
        verdict=verdict, runs=runs, pilots=pilots, b_prior_probe=B_PRIOR_PROBE,
        budget=dict(train_hours=rd(sum(r['hours'] for r in runs), 4), pilot_hours=rd(sum(p['hours'] for p in pilots), 4),
                    eval_hours=eh, b_records=nb, b_expected=len(b_expected),
                    b_hours_mean=rd(np.mean(b_hours), 4) if b_hours else None,
                    b_hours_projected=rd(np.mean(b_hours) * len(b_expected), 4) if b_hours else None),
    )
    page['facts'] = facts_block(S, stage0, verdict, sensor, runs)
    s = json.dumps(page, separators=(',', ':'), ensure_ascii=False, sort_keys=True)
    with open(OUT, 'w') as fh:
        fh.write(s)
    print(f'wrote {OUT} ({len(s)} bytes); verdict {verdict["file"]} (B clauses current: {verdict["b_valid"]}'
          + (f'; {verdict["b_invalid_reason"]}' if not verdict['b_valid'] else '') + ')')
    print(f'B records: {len(b_have)} of {len(b_expected)} present')
    if page['status']['pending']:
        print('PENDING records: ' + ', '.join(page['status']['pending']))
    else:
        print('PENDING records: none')
    if page['status']['stale']:
        print('STALE B records (not x0-prior; not counted as final): ' + ', '.join(page['status']['stale']))


if __name__ == '__main__':
    main()
