"""Phase 1a gate (spec §8): at its own best <= 10-NFE sampler setting, B must beat the baseline, A and A' by more
than k paired standard errors on the same cubes; a failing y-swap makes B invalid (spec §11). Refuses anything that
is not a set of three comparable final full-grid evaluations trained for the same number of epochs (unless
--allow_unequal_epochs, which marks the verdict).

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
from scripts.paired_stats import cluster_se  # noqa: E402

EXPECTED = {'b': 'residual', 'a': 'standard', 'aprime': 'standard_xhat'}
PREREG = 'ddim10'
REQUIRED_ROWS = (PREREG, 'ddpm1000')          # a full-grid evaluation always has both
DEFAULT_MAX_NFE, DEFAULT_K = 10, 2.0
K_SE = ('se: the iid paired SE over cubes (std ddof=1 / sqrt(n)), as the spec §8 rule states; se_cluster (CR1 over '
        'source images, n_clusters images) is reported alongside and is not used by the rule')


def _need(e, key, where='evaluation'):
    if key not in e:
        raise ValueError(f'{where} has no {key!r} key')
    return e[key]


def _prov(e, key):
    return _need(_need(e, 'provenance'), key, 'provenance')


def check_comparable(evs):
    """All three must be full-grid, meta-checked evaluations with identical seeds, d, base_channels, prediction_type and prior."""
    for e in evs:
        if e.get('grid') != 'full':
            raise ValueError(f"not a full-grid evaluation: grid={e.get('grid')!r}")
        if not _need(e, 'provenance').get('meta_checked'):
            raise ValueError('checkpoint meta was not checked (meta_checked is false)')
    for key in ('seeds', 'd'):
        if len({json.dumps(_need(e, key)) for e in evs}) != 1:
            raise ValueError(f'evaluations differ in {key}: {[e[key] for e in evs]}')
    for key in ('base_channels', 'prediction_type', 'prior_sha256'):
        if len({str(_prov(e, key)) for e in evs}) != 1:
            raise ValueError(f'evaluations differ in provenance.{key}: {[e["provenance"][key] for e in evs]}')


def ckpt_epoch(ckpt):
    m = re.search(r'checkpoint_epoch_(\d+)\.pth$', ckpt)
    if not m:
        raise ValueError(f'cannot read a training epoch from the checkpoint name {ckpt!r} (expected checkpoint_epoch_<N>.pth)')
    return int(m.group(1))


def check_done(ev):
    """The checkpoint must be the run's final one: its epoch equals the DONE file next to it."""
    ckpt = _prov(ev, 'ckpt')
    done = os.path.join(os.path.dirname(ckpt), 'DONE')
    if not os.path.exists(done):
        raise ValueError(f'no DONE file next to {ckpt}: the run has not finished')
    with open(done) as f:
        final = int(f.read().strip())
    if ckpt_epoch(ckpt) != final:
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


def _paired(d, files):
    """Mean and iid SE of per-cube differences `d`, plus the cluster-robust SE over the cubes' source images."""
    if len(d) < 2:
        raise ValueError('need at least 2 paired cubes')
    se_cluster, n_clusters = cluster_se(d, files)
    return dict(diff=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), n=int(len(d)),
                se_cluster=se_cluster, n_clusters=n_clusters)


def paired_vs_model(row_b, row_other):
    if row_b['files'] != row_other['files']:
        raise ValueError('rows score different cubes (or a different order); pair via files')
    return _paired(per_cube(row_b) - per_cube(row_other), row_b['files'])


def baseline_by_file(ev):
    """file -> per-cube baseline RMSE of an evaluation, or None when it does not carry them."""
    if 'baseline_files' not in ev or 'baseline_per_item' not in ev:
        return None
    return dict(zip(ev['baseline_files'], ev['baseline_per_item']['rmse_pct']))


def paired_vs_baseline(row, base=None):
    """The row's stored paired difference to the baseline and its iid SE (computed by the harness). The clustered SE
    is the row's own when the harness stored it, else recomputed from `base` (file -> per-cube baseline), else None."""
    r = dict(diff=row['paired_rmse_diff'], se=row['paired_rmse_se'], n=len(row['files']), se_cluster=None, n_clusters=None)
    if 'paired_rmse_se_cluster' in row:
        r.update(se_cluster=row['paired_rmse_se_cluster'], n_clusters=row['n_clusters'])
    elif base is not None and all(f in base for f in row['files']):
        r['se_cluster'], r['n_clusters'] = cluster_se(per_cube(row) - np.array([base[f] for f in row['files']]), row['files'])
    return r


def paired_on_intersection(row_x, row_y):
    """x − y on the cubes both rows scored (e.g. DDIM-10 on 200 cubes vs DDPM-1000 on 50)."""
    cx = dict(zip(row_x['files'], per_cube(row_x))); cy = dict(zip(row_y['files'], per_cube(row_y)))
    common = [f for f in row_y['files'] if f in cx]
    if not common:
        raise ValueError('no common cubes')
    return _paired(np.array([cx[f] - cy[f] for f in common]), common)


def row_named(ev, name):
    return next((r for r in ev['rows'] if r['name'] == name), None)


def comparisons(rb, ra, rp, k, blind_a, blind_aprime, base=None):
    """B's row against the baseline (its stored paired diff/SE), A's row and A''s row; pass = diff + k·SE < 0, with
    the iid SE (se_cluster is reported only). `base`: file -> per-cube baseline, for the baseline's clustered SE."""
    vs = dict(vs_baseline=paired_vs_baseline(rb, base), vs_a=paired_vs_model(rb, ra), vs_aprime=paired_vs_model(rb, rp))
    for r in vs.values():
        r['pass'] = bool(r['diff'] + k * r['se'] < 0)
    vs['vs_a']['ablation_blind'] = blind_a
    vs['vs_aprime']['ablation_blind'] = blind_aprime
    return vs


def default_out_name(run_name_b, require_done=True, max_nfe=DEFAULT_MAX_NFE, k=DEFAULT_K):
    """A run that peeks (no DONE required) or uses other thresholds must not overwrite the official gate file."""
    return (f'gate_1a__{run_name_b}' + ('' if require_done else '__nodone') + ('' if max_nfe == DEFAULT_MAX_NFE else f'__nfe{max_nfe}')
            + ('' if k == DEFAULT_K else f'__k{k:g}') + '.json')


def gate(ev_b, ev_a, ev_aprime, max_nfe=DEFAULT_MAX_NFE, k=DEFAULT_K, require_done=True, allow_unequal_epochs=False):
    evs = dict(b=ev_b, a=ev_a, aprime=ev_aprime)
    for slot, ev in evs.items():
        if _prov(ev, 'mode') != EXPECTED[slot]:
            raise ValueError(f"--{slot} must be a {EXPECTED[slot]} evaluation, got mode {ev['provenance']['mode']!r}")
    check_comparable(list(evs.values()))
    for slot, ev in evs.items():
        for name in REQUIRED_ROWS:
            if row_named(ev, name) is None:
                raise ValueError(f"--{slot} ({EXPECTED[slot]}) evaluation has no {name!r} row; a full-grid evaluation always has it")
    if require_done:
        for ev in evs.values():
            check_done(ev)
    epochs = {slot: ckpt_epoch(_prov(ev, 'ckpt')) for slot, ev in evs.items()}
    unequal = len(set(epochs.values())) > 1
    if unequal and not allow_unequal_epochs:
        raise ValueError(f'models were trained for different numbers of epochs: B={epochs["b"]}, A={epochs["a"]}, '
                         f"A'={epochs['aprime']}; pass allow_unequal_epochs to compare them anyway (the verdict is then marked)")
    blind_a, blind_ap = not bool(ev_a['yswap']['passes']), not bool(ev_aprime['yswap']['passes'])
    rb, ra, rp = (best_row(ev, max_nfe) for ev in (ev_b, ev_a, ev_aprime))
    base = baseline_by_file(ev_b)
    vs = comparisons(rb, ra, rp, k, blind_a, blind_ap, base)
    yswap_b = bool(ev_b['yswap']['passes'])
    if not yswap_b:
        verdict = 'INVALID'
    elif vs['vs_baseline']['pass'] and vs['vs_a']['pass']:
        verdict = 'PASS' if vs['vs_aprime']['pass'] else 'PASS_NOT_OVER_APRIME'
    else:
        verdict = 'FAIL'
    if unequal:
        verdict += '_UNEQUAL_EPOCHS'
    pb, pa, pp = (row_named(ev, PREREG) for ev in (ev_b, ev_a, ev_aprime))
    prereg = dict(setting=PREREG, **comparisons(pb, pa, pp, k, blind_a, blind_ap, base),
                  rmse_pct=dict(b=pb['rmse_pct'], a=pa['rmse_pct'], aprime=pp['rmse_pct']))
    at_b_setting = {s: paired_vs_model(rb, row_named(ev, rb['name'])) for s, ev in (('a', ev_a), ('aprime', ev_aprime)) if row_named(ev, rb['name'])}
    speed = {s: paired_on_intersection(row_named(ev, PREREG), row_named(ev, 'ddpm1000')) for s, ev in evs.items()}
    return dict(verdict=verdict, max_nfe=max_nfe, k=k, k_se=K_SE, require_done=bool(require_done), yswap_b=yswap_b,
                run_names={s: ev.get('run_name') for s, ev in evs.items()}, final_epochs=epochs,
                b=dict(selected_from_grid=rb['name'], nfe=rb['nfe'], rmse_pct=rb['rmse_pct'], rmse_pct_se=rb['rmse_pct_se'], ckpt=ev_b['provenance']['ckpt']),
                a=dict(selected_from_grid=ra['name'], nfe=ra['nfe'], rmse_pct=ra['rmse_pct'], ckpt=ev_a['provenance']['ckpt']),
                aprime=dict(selected_from_grid=rp['name'], nfe=rp['nfe'], rmse_pct=rp['rmse_pct'], ckpt=ev_aprime['provenance']['ckpt']),
                baseline_rmse_pct=ev_b['baseline']['rmse_pct'], prereg=prereg,
                b_setting_applied_to=at_b_setting, ddim10_minus_ddpm1000=speed, **vs)


def _print_comparisons(g, indent='  '):
    for lab, key in (('B - baseline', 'vs_baseline'), ('B - A', 'vs_a'), ("B - A'", 'vs_aprime')):
        r = g[key]
        flag = '  [ablation y-swap failed]' if r.get('ablation_blind') else ''
        cl = f", clustered ± {r['se_cluster']:.4f} over {r['n_clusters']} images" if r.get('se_cluster') is not None else ''
        print(f"{indent}{lab:<13} {r['diff']:+.4f} ± {r['se']:.4f} (n={r['n']}{cl})  {'pass' if r['pass'] else 'fail'}{flag}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--b', required=True); ap.add_argument('--a', required=True); ap.add_argument('--aprime', required=True)
    ap.add_argument('--max_nfe', type=int, default=DEFAULT_MAX_NFE); ap.add_argument('--k', type=float, default=DEFAULT_K)
    ap.add_argument('--no_require_done', action='store_true'); ap.add_argument('--allow_unequal_epochs', action='store_true')
    ap.add_argument('--out', default=None, help='default: gate_1a__<B run_name>[__nodone][__nfe<N>][__k<K>].json next to --b')
    args = ap.parse_args(argv)
    paths = dict(b=args.b, a=args.a, aprime=args.aprime)
    evs = []
    for p in paths.values():
        with open(p) as f:
            evs.append(json.load(f))
    g = gate(*evs, max_nfe=args.max_nfe, k=args.k, require_done=not args.no_require_done, allow_unequal_epochs=args.allow_unequal_epochs)
    g['inputs'] = {s: os.path.abspath(p) for s, p in paths.items()}
    print(f"gate 1a (<= {g['max_nfe']} NFE, k = {g['k']}, DONE {'required' if g['require_done'] else 'NOT required'}): {g['verdict']}")
    print(f"  epochs  B={g['final_epochs']['b']}  A={g['final_epochs']['a']}  A'={g['final_epochs']['aprime']}")
    for lab, key in (('B ', 'b'), ('A ', 'a'), ("A'", 'aprime')):
        r = g[key]
        print(f"  {lab} {r['selected_from_grid']:>13} nfe={r['nfe']:3d} RMSE {r['rmse_pct']:.4f} %   {os.path.basename(r['ckpt'])}")
    print(f"  baseline (200 cubes) {g['baseline_rmse_pct']:.4f} %   y-swap B {'pass' if g['yswap_b'] else 'FAIL'}")
    _print_comparisons(g)
    pr = g['prereg']
    print(f"pre-registered {pr['setting']}:  RMSE B {pr['rmse_pct']['b']:.4f} %  A {pr['rmse_pct']['a']:.4f} %  A' {pr['rmse_pct']['aprime']:.4f} %")
    _print_comparisons(pr)
    run_b = evs[0].get('run_name') or os.path.splitext(os.path.basename(args.b))[0]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.b)), default_out_name(run_b, not args.no_require_done, args.max_nfe, args.k))
    with open(out, 'w') as f:
        json.dump(g, f, indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
