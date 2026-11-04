"""The pre-registered Phase 3 verdict (spec §6.3): T1-T4 applied to sd_eval records named in a manifest, and the main
diffusion line (B = guided unconditional prior, C = family-conditioned residual diffusion). Written before any record
exists; the thresholds live here and nowhere else.

    python scripts/sensor/sd_verdict.py --manifest <manifest.json> [--report_only]

Manifest: {"name": "<verdict name>", "records": {"<device>/<mode>/<snr>": ["<record name>", ...]}} with the records read
from SENSOR_DIR/eval/<name>.json. Several records of one setting are merged by method (a method twice is an error).
Records of different sizes are paired on their common files (>= MIN_COMMON), in the first list's order.

T1  RMSE_line <= T1_RATIO * RMSE_D at every E1 SNR (paired, 2 clustered SE).
T2  at T2_SNRS: CRPS_line < CRPS_L and < CRPS_V (2 clustered SE) and cov90 of the line within T2_COV90 (K >= 19).
T3  RMSE ratio line / L on device2 at T3_SNR <= T3_RATIO * the same ratio on device1 (PH5 reported, never gated).
T4  RMSE_line <= T4_RATIO * RMSE_L_demosaic on device1/snapshot at T4_SNR (paired, 2 clustered SE).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from data_loader.sensor_config import SENSOR_DIR  # noqa: E402
from scripts.paired_stats import cluster_se  # noqa: E402
from scripts.uq.metrics import min_k_for  # noqa: E402

T1_RATIO = 1.05
T2_COV90 = (0.85, 0.95)
T2_SNRS = (30.0, 20.0)
T3_RATIO = 1.10
T3_SNR = 30.0
T4_RATIO = 0.85
T4_SNR = 30.0
MIN_COMMON = 45
MANIFEST_KEYS = ('name', 'records')

E1_SNRS = (60.0, 50.0, 40.0, 30.0, 20.0)
LINES = ('B', 'C')
RMSE_KEY = dict(L='mean_rmse_pct', L_demosaic='mean_rmse_pct', V='mean_rmse_pct', C='mean_rmse_pct', B='mean_rmse_pct',
                D='rmse_pct', C1='rmse_pct')
SANITY_METHODS = ('L', 'D', 'V', 'C', 'B', 'C1')
LEARNED = ('D', 'V', 'C', 'B', 'C1')


class Missing(KeyError):
    """A setting or a method the clause needs is not in the manifest's records."""

    def __str__(self):
        return str(self.args[0])


def setting_key(device, mode, snr):
    return f'{device}/{mode}/{float(snr):g}'


def _parse_key(key):
    dev, mode, snr = key.split('/')
    return dev, mode, float(snr)


def load_manifest(path):
    """-> settings[setting_key][method] = (per_cube[method], files, K)."""
    with open(path) as f:
        man = json.load(f)
    settings = {}
    for key, names in man['records'].items():
        dev, mode, snr = _parse_key(key)
        k = setting_key(dev, mode, snr)
        out = settings.setdefault(k, {})
        for name in names:
            with open(os.path.join(SENSOR_DIR, 'eval', name + '.json')) as f:
                rec = json.load(f)
            st = rec.get('setting') or {}
            if (st.get('sensor_device'), st.get('mode'), float(st.get('snr_db', float('nan')))) != (dev, mode, snr):
                raise ValueError(f'record {name} has setting {st} but the manifest files it under {key}')
            for method, pc in rec['per_cube'].items():
                if method in out:
                    raise ValueError(f'method {method} appears twice in setting {k} (record {name})')
                out[method] = (pc, rec['files'], rec['K'])
    return settings


def _get(settings, key, method):
    if key not in settings:
        raise Missing(f'setting {key}')
    if method not in settings[key]:
        raise Missing(f'method {method} at {key}')
    return settings[key][method]


def _common(a, af, b, bf, key_a, key_b):
    pos = {f: i for i, f in enumerate(bf)}
    ia = [i for i, f in enumerate(af) if f in pos]
    if len(ia) < MIN_COMMON:
        raise ValueError(f'only {len(ia)} files are common to both records (< MIN_COMMON = {MIN_COMMON})')
    ib = [pos[af[i]] for i in ia]
    x = np.asarray(a[key_a], dtype=np.float64)[ia]
    y = np.asarray(b[key_b], dtype=np.float64)[ib]
    return x, y, [af[i] for i in ia]


def _stats(d, files):
    se_c, G = cluster_se(d, files)
    if se_c is None:
        raise ValueError('the common files come from fewer than 2 source images; no clustered SE')
    return dict(mean=float(d.mean()), se=float(d.std(ddof=1) / math.sqrt(len(d))), se_cluster=float(se_c), n=int(len(d)), clusters=int(G))


def paired_on_common(a, af, b, bf, key_a, key_b):
    """Difference a - b on the files common to both lists, in af's order."""
    x, y, files = _common(a, af, b, bf, key_a, key_b)
    return dict(_stats(x - y, files), ratio_of_means=float(x.mean() / y.mean()))


def ratio_clause(num, den, files, key_num, key_den, limit, direction):
    """Paired test of num - limit * den with its clustered SE. num / den are per-cube metric dicts; files is the common
    list, or a pair (num_files, den_files). direction 'le': num <= limit * den by 2 clustered SE."""
    nf, df = files if isinstance(files, tuple) else (files, files)
    x, y, common = _common(num, nf, den, df, key_num, key_den)
    s = _stats(x - limit * y, common)
    if direction == 'le':
        passed = s['mean'] + 2 * s['se_cluster'] <= 0
    elif direction == 'ge':
        passed = s['mean'] - 2 * s['se_cluster'] >= 0
    else:
        raise ValueError(direction)
    return dict(passed=bool(passed), ratio=float(x.mean() / y.mean()), ratio_2se=float((s['mean'] + 2 * s['se_cluster']) / y.mean() + limit),
                mean=s['mean'], se_cluster=s['se_cluster'], n=s['n'])


def _clause(settings, key, m_num, m_den, limit):
    pn, fn, _ = _get(settings, key, m_num)
    pd, fd, _ = _get(settings, key, m_den)
    return ratio_clause(pn, pd, (fn, fd), RMSE_KEY[m_num], RMSE_KEY[m_den], limit, 'le')


def t1(settings, line, missing=None):
    out = {}
    for s in E1_SNRS:
        try:
            out[s] = _clause(settings, setting_key('device1', 'sequential', s), line, 'D', T1_RATIO)
        except Missing as e:
            if missing is None:
                raise
            missing.append(f'T1 {line}: {e}')
    return out


def t2(settings, line, report_only=False, missing=None):
    out = {}
    for s in T2_SNRS:
        key = setting_key('device1', 'sequential', s)
        try:
            pl, fl, K = _get(settings, key, line)
            pL, fL, _ = _get(settings, key, 'L')
            pV, fV, _ = _get(settings, key, 'V')
        except Missing as e:
            if missing is None:
                raise
            missing.append(f'T2 {line}: {e}')
            continue
        need = min_k_for(0.9)
        if K < need and not report_only:
            raise ValueError(f'T2 {line} at {key}: record K = {K} < min_k_for(0.9) = {need}; coverage is exact only from K >= {need} '
                             f'(use --report_only for a descriptive verdict)')
        clauses = {}
        for name, (p, f) in (('crps_vs_L', (pL, fL)), ('crps_vs_V', (pV, fV))):
            c = paired_on_common(pl, fl, p, f, 'crps', 'crps')
            clauses[name] = dict(c, passed=bool(c['mean'] + 2 * c['se_cluster'] < 0))
        cov90 = float(np.mean(pl['cov90']))
        cov_ok = bool(T2_COV90[0] <= cov90 <= T2_COV90[1])
        out[s] = dict(clauses, cov90=cov90, cov90_passed=cov_ok, K=int(K),
                      passed=bool(clauses['crps_vs_L']['passed'] and clauses['crps_vs_V']['passed'] and cov_ok))
    return out


def _rmse_ratio(settings, key, line):
    pl, fl, _ = _get(settings, key, line)
    pL, fL, _ = _get(settings, key, 'L')
    return paired_on_common(pl, fl, pL, fL, RMSE_KEY[line], RMSE_KEY['L'])['ratio_of_means']


def t3(settings, line):
    r1 = _rmse_ratio(settings, setting_key('device1', 'sequential', T3_SNR), line)
    r2 = _rmse_ratio(settings, setting_key('device2', 'sequential', T3_SNR), line)
    out = dict(ratio_dev1=r1, ratio_dev2=r2, passed=bool(r2 <= T3_RATIO * r1))
    try:
        out['ph5'] = _rmse_ratio(settings, setting_key('ph5', 'sequential', T3_SNR), line)
    except Missing:
        pass
    return out


def t4(settings, line):
    key = setting_key('device1', 'snapshot', T4_SNR)
    out = _clause(settings, key, line, 'L_demosaic', T4_RATIO)
    try:
        out['D'] = _clause(settings, key, 'D', 'L_demosaic', T4_RATIO)
    except Missing:
        pass
    return out


def sanity(settings):
    keys = {s: setting_key('device1', 'sequential', s) for s in E1_SNRS}
    monotone = {}
    for m in SANITY_METHODS:
        vals = [float(np.mean(settings[keys[s]][m][0][RMSE_KEY[m]])) for s in sorted(E1_SNRS) if keys[s] in settings and m in settings[keys[s]]]
        if len(vals) >= 2:
            monotone[m] = bool(all(a >= b for a, b in zip(vals, vals[1:])))
    floor_pass, ref = {}, None
    k60 = keys[60.0]
    if k60 in settings and 'L' in settings[k60]:
        pL, fL, _ = settings[k60]['L']
        ref = float(np.mean(pL['mean_rmse_pct']))
        for m in LEARNED:
            if m in settings[k60]:
                p, f, _ = settings[k60][m]
                c = paired_on_common(p, f, pL, fL, RMSE_KEY[m], 'mean_rmse_pct')
                floor_pass[m] = bool(c['mean'] <= 2 * c['se_cluster'])
    return dict(monotone=monotone, floor_reference_rmse_pct=ref, floor_pass=floor_pass)


def _counts(res):
    counts = {}
    for line in res['T1']:
        n = int(all(s in res['T1'][line] and res['T1'][line][s]['passed'] for s in E1_SNRS))
        n += sum(int(res['T2'][line][s]['passed']) for s in T2_SNRS if s in res['T2'][line])
        n += int(bool(res['T3'][line] and res['T3'][line]['passed']))
        n += int(bool(res['T4'][line] and res['T4'][line]['passed']))
        counts[line] = n
    return counts


def main_line(res):
    """Most passes among T1 (once, all five SNRs), T2 (per SNR), T3, T4; a tie goes to the lower paired CRPS - CRPS_V at 20 dB."""
    counts = _counts(res)
    top = max(counts.values())
    tied = [l for l in counts if counts[l] == top]
    if len(tied) == 1:
        return tied[0]
    return min(tied, key=lambda l: res['T2'][l][T2_SNRS[1]]['crps_vs_V']['mean'])


def verdict(settings, report_only=False):
    res = dict(T1={}, T2={}, T3={}, T4={})
    missing = []
    for line in LINES:
        res['T1'][line] = t1(settings, line, missing if report_only else None)
        res['T2'][line] = t2(settings, line, report_only, missing if report_only else None)
        for name, fn in (('T3', t3), ('T4', t4)):
            try:
                res[name][line] = fn(settings, line)
            except Missing as e:
                if not report_only:
                    raise
                missing.append(f'{name} {line}: {e}')
                res[name][line] = None
    try:
        san = sanity(settings)
    except Missing as e:
        if not report_only:
            raise
        missing.append(f'sanity: {e}')
        san = None
    if not report_only:
        for s in E1_SNRS:
            _get(settings, setting_key('device1', 'sequential', s), 'L')
    counts = _counts(res)
    ml = None
    if not report_only:
        ml = main_line(res)
    return dict(T1=res['T1'], T2=res['T2'], T3=res['T3'], T4=res['T4'], sanity=san, counts=counts, main_line=ml,
                report_only=bool(report_only), missing=sorted(set(missing)))


def verdict_path(name, report_only):
    return os.path.join(SENSOR_DIR, 'verdict__' + name + ('__report' if report_only else '') + '.json')


def _print(v):
    for line in LINES:
        for s, c in v['T1'][line].items():
            print(f"T1 {line} {s:g} dB: {'PASS' if c['passed'] else 'FAIL'}  ratio {c['ratio']:.4f} (2se {c['ratio_2se']:.4f}, limit {T1_RATIO})")
        for s, c in v['T2'][line].items():
            print(f"T2 {line} {s:g} dB: {'PASS' if c['passed'] else 'FAIL'}  crps-L {c['crps_vs_L']['mean']:+.4f}  crps-V {c['crps_vs_V']['mean']:+.4f}"
                  f"  cov90 {c['cov90']:.3f}")
        c = v['T3'][line]
        if c:
            print(f"T3 {line}: {'PASS' if c['passed'] else 'FAIL'}  dev2 {c['ratio_dev2']:.4f} vs {T3_RATIO} x dev1 {c['ratio_dev1']:.4f}")
        c = v['T4'][line]
        if c:
            print(f"T4 {line}: {'PASS' if c['passed'] else 'FAIL'}  ratio {c['ratio']:.4f} (2se {c['ratio_2se']:.4f}, limit {T4_RATIO})")
    for m in v['missing']:
        print('MISSING', m)
    print('passes', v['counts'], '| main line:', v['main_line'] if v['main_line'] else 'none (report only)' if v['report_only'] else 'none')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--report_only', action='store_true')
    args = ap.parse_args(argv)
    with open(args.manifest) as f:
        name = json.load(f)['name']
    out = verdict_path(name, args.report_only)
    if os.path.exists(out):
        raise FileExistsError(f'{out} exists; the verdict is taken once')
    settings = load_manifest(args.manifest)
    v = verdict(settings, args.report_only)
    _print(v)
    v.update(name=name, manifest=os.path.abspath(args.manifest), created=time.strftime('%Y-%m-%dT%H:%M:%S'),
             thresholds=dict(T1_RATIO=T1_RATIO, T2_COV90=T2_COV90, T2_SNRS=T2_SNRS, T3_RATIO=T3_RATIO, T3_SNR=T3_SNR, T4_RATIO=T4_RATIO,
                             T4_SNR=T4_SNR, MIN_COMMON=MIN_COMMON))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + '.tmp', 'w') as f:
        json.dump(v, f, indent=1)
    os.replace(out + '.tmp', out)
    print('wrote', out)
    return v


if __name__ == '__main__':
    main()
