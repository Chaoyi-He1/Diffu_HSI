"""Phase 1a gate (spec §8): at its own best <= 10-NFE sampler setting, B must beat the baseline, A and A' by more
than k paired standard errors on the same cubes; a failing y-swap makes B invalid (spec §11). Refuses anything that
is not a set of three comparable final full-grid evaluations.

python scripts/gate_1a.py --b <eval>/residual/d4/residual_d4_b64__checkpoint_epoch_100.json \
    --a <eval>/standard/d4/standard_d4_b64__checkpoint_epoch_100.json \
    --aprime <eval>/standard_xhat/d4/standard_xhat_d4_b64__checkpoint_epoch_100.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

EXPECTED = {'b': 'residual', 'a': 'standard', 'aprime': 'standard_xhat'}
PREREG = 'ddim10'


def check_comparable(evs):
    """All three must be full-grid, meta-checked evaluations with identical seeds, d, base_channels, prediction_type and prior."""
    for e in evs:
        if e.get('grid', 'full') != 'full':
            raise ValueError(f"not a full-grid evaluation: grid={e.get('grid')!r}")
        if not e['provenance'].get('meta_checked'):
            raise ValueError('checkpoint meta was not checked (meta_checked is false)')
    for key in ('seeds', 'd'):
        if len({json.dumps(e[key]) for e in evs}) != 1:
            raise ValueError(f'evaluations differ in {key}: {[e[key] for e in evs]}')
    for key in ('base_channels', 'prediction_type', 'prior_sha256'):
        if len({str(e['provenance'][key]) for e in evs}) != 1:
            raise ValueError(f'evaluations differ in provenance.{key}: {[e["provenance"][key] for e in evs]}')


def check_done(ev):
    """The checkpoint must be the run's final one: its epoch equals the DONE file next to it."""
    ckpt = ev['provenance']['ckpt']
    done = os.path.join(os.path.dirname(ckpt), 'DONE')
    if not os.path.exists(done):
        raise ValueError(f'no DONE file next to {ckpt}: the run has not finished')
    m = re.search(r'checkpoint_epoch_(\d+)\.pth$', ckpt)
    final = int(open(done).read().strip())
    if not m or int(m.group(1)) != final:
        raise ValueError(f'{ckpt} is not the final checkpoint (DONE says epoch {final})')


def eligible_rows(ev, max_nfe):
    return [r for r in ev['rows'] if r['name'] != 'ddpm1000' and r['nfe'] <= max_nfe]


def best_row(ev, max_nfe):
    rows = eligible_rows(ev, max_nfe)
    if not rows:
        raise ValueError(f'no sampler row with nfe <= {max_nfe}')
    return min(rows, key=lambda r: r['rmse_pct'])


def per_cube(row):
    return np.asarray(row['per_item']['rmse_pct'], dtype=float).mean(0)       # [seed][cube] -> [cube]


def _paired(d):
    return dict(diff=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float('nan'), n=int(len(d)))


def paired_vs_model(row_b, row_other):
    if row_b['files'] != row_other['files']:
        raise ValueError('rows score different cubes (or a different order); pair via files')
    return _paired(per_cube(row_b) - per_cube(row_other))


def paired_on_intersection(row_x, row_y):
    """x − y on the cubes both rows scored (e.g. DDIM-10 on 200 cubes vs DDPM-1000 on 50)."""
    cx = dict(zip(row_x['files'], per_cube(row_x))); cy = dict(zip(row_y['files'], per_cube(row_y)))
    common = [f for f in row_y['files'] if f in cx]
    if not common:
        raise ValueError('no common cubes')
    return _paired(np.array([cx[f] - cy[f] for f in common]))


def row_named(ev, name):
    return next((r for r in ev['rows'] if r['name'] == name), None)


def gate(ev_b, ev_a, ev_aprime, max_nfe=10, k=2.0, require_done=True):
    evs = dict(b=ev_b, a=ev_a, aprime=ev_aprime)
    for slot, ev in evs.items():
        if ev['provenance']['mode'] != EXPECTED[slot]:
            raise ValueError(f"--{slot} must be a {EXPECTED[slot]} evaluation, got mode {ev['provenance']['mode']!r}")
    check_comparable(list(evs.values()))
    if require_done:
        for ev in evs.values():
            check_done(ev)
    rb, ra, rp = (best_row(ev, max_nfe) for ev in (ev_b, ev_a, ev_aprime))
    vs = dict(vs_baseline=dict(diff=rb['paired_rmse_diff'], se=rb['paired_rmse_se'], n=len(rb['files'])),
              vs_a=paired_vs_model(rb, ra), vs_aprime=paired_vs_model(rb, rp))
    for key, r in vs.items():
        r['pass'] = bool(r['diff'] + k * r['se'] < 0)
    vs['vs_a']['ablation_blind'] = not bool(ev_a['yswap']['passes'])
    vs['vs_aprime']['ablation_blind'] = not bool(ev_aprime['yswap']['passes'])
    yswap_b = bool(ev_b['yswap']['passes'])
    if not yswap_b:
        verdict = 'INVALID'
    elif vs['vs_baseline']['pass'] and vs['vs_a']['pass']:
        verdict = 'PASS' if vs['vs_aprime']['pass'] else 'PASS_NOT_OVER_APRIME'
    else:
        verdict = 'FAIL'
    prereg = {s: dict(rmse_pct=row_named(ev, PREREG)['rmse_pct'], paired_rmse_diff=row_named(ev, PREREG)['paired_rmse_diff'])
              for s, ev in evs.items() if row_named(ev, PREREG)}
    at_b_setting = {s: paired_vs_model(rb, row_named(ev, rb['name'])) for s, ev in (('a', ev_a), ('aprime', ev_aprime)) if row_named(ev, rb['name'])}
    speed = {s: paired_on_intersection(row_named(ev, PREREG), row_named(ev, 'ddpm1000'))
             for s, ev in evs.items() if row_named(ev, PREREG) and row_named(ev, 'ddpm1000')}
    return dict(verdict=verdict, max_nfe=max_nfe, k=k, yswap_b=yswap_b,
                b=dict(selected_from_grid=rb['name'], nfe=rb['nfe'], rmse_pct=rb['rmse_pct'], rmse_pct_se=rb['rmse_pct_se'], ckpt=ev_b['provenance']['ckpt']),
                a=dict(selected_from_grid=ra['name'], nfe=ra['nfe'], rmse_pct=ra['rmse_pct'], ckpt=ev_a['provenance']['ckpt']),
                aprime=dict(selected_from_grid=rp['name'], nfe=rp['nfe'], rmse_pct=rp['rmse_pct'], ckpt=ev_aprime['provenance']['ckpt']),
                baseline_rmse_pct=ev_b['baseline']['rmse_pct'], prereg=dict(setting=PREREG, **prereg) if prereg else {},
                b_setting_applied_to=at_b_setting, ddim10_minus_ddpm1000=speed, **vs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b', required=True); ap.add_argument('--a', required=True); ap.add_argument('--aprime', required=True)
    ap.add_argument('--max_nfe', type=int, default=10); ap.add_argument('--k', type=float, default=2.0)
    ap.add_argument('--no_require_done', action='store_true'); ap.add_argument('--out', default=None)
    args = ap.parse_args()
    evs = [json.load(open(p)) for p in (args.b, args.a, args.aprime)]
    g = gate(*evs, max_nfe=args.max_nfe, k=args.k, require_done=not args.no_require_done)
    print(f"gate 1a (<= {g['max_nfe']} NFE, k = {g['k']}): {g['verdict']}")
    for lab, key in (('B ', 'b'), ('A ', 'a'), ("A'", 'aprime')):
        r = g[key]
        print(f"  {lab} {r['selected_from_grid']:>13} nfe={r['nfe']:3d} RMSE {r['rmse_pct']:.4f} %   {os.path.basename(r['ckpt'])}")
    print(f"  baseline (200 cubes) {g['baseline_rmse_pct']:.4f} %   y-swap B {'pass' if g['yswap_b'] else 'FAIL'}")
    for lab, key in (('B - baseline', 'vs_baseline'), ('B - A', 'vs_a'), ("B - A'", 'vs_aprime')):
        r = g[key]
        flag = '  [ablation y-swap failed]' if r.get('ablation_blind') else ''
        print(f"  {lab:<13} {r['diff']:+.4f} ± {r['se']:.4f} (n={r['n']})  {'pass' if r['pass'] else 'fail'}{flag}")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.b)), 'gate_1a.json')
    with open(out, 'w') as f:
        json.dump(g, f, indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
