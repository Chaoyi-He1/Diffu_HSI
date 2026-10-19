"""Phase 1a-prime gate: the one-step residual regressor(s) against the baseline and B's one-step read-out (ddim1) on
the same 200 cubes, under the PROPOSED one-step y-swap validity rule (plan deviation 1: binding for the verdicts in
this phase; the spec §7 0.9 ratio is reported, and go_1b requires it too). Torch-free.

  python scripts/gate_1a_prime.py \
      --reg <eval>/regress/d4/regress_l1_d4_b64__checkpoint_epoch_100__direct.json \
            <eval>/regress/d4/regress_l2_d4_b64__checkpoint_epoch_100__direct.json \
      --b_yswap <results>/diagnostics/yswap_residual_d4_b64_t999_n32__xT.json \
      --b_diag <results>/diagnostics/b_diagnostics.json

Per regressor R: vs_baseline (R - baseline per cube) and vs_b_ddim1 (R - B's seed-mean ddim1 per cube) pass when
diff + k * SE < 0 (iid SE; the CR1 SE over source images is reported). Validity, on R's own y-swap at its single call
t = 999: (a) swap_full - true > k * SE and (b) swap_full - baseline + k * SE >= 0 (with a swapped condition the gain
over the baseline vanishes); invalid_reason names the failed part(s), 'a', 'b' or 'ab'. Verdict: INVALID unless valid;
else PASS (beats the baseline and B), PASS_NOT_OVER_B (beats the baseline only) or FAIL. target_met: R's RMSE <= 0.85 x
the baseline (the spec §8 1b bar). Report only (--b_diag): vs_b_t999_kmean, R minus the mean of B's K one-step draws
at t = 999 (K NFE). B is re-validated under the same rule from its t = 999 y-swap (--b_yswap). Two regressors with
different losses and the same training seed form a loss_pair; two with the same loss and different seeds form a
seed_pair, whose go_1b is true only when both are PASS (hence valid), target_met and pass the spec 0.9 ratio at
t = 999. best is the lowest-RMSE PASS regressor (the lowest-RMSE one when none passes); best_verdict is its verdict.
Every regressor must carry its loss_type and train_seed, and no two may share both ('duplicate runs').

Every y-swap record must hold YSWAP_N = 32 cubes (the pre-registered size) and be scored against its own evaluation's
per-cube baseline on its cubes: the regressor's own y-swap against the regressor evaluation's baseline, and --b_yswap
against B's (within YSWAP_BASELINE_TOL). --b_yswap must also be B's by provenance (mode residual, B's checkpoint, d,
prior, meta checked) and leak-free: made with scripts/yswap_one_step.py --x_t noise, B's own pure-noise x_T (the
default q_sample x_t leaks the true residual to B at t = 999). --b_diag's provenance.b_eval must name the --b file.

A regressor-ensemble evaluation (provenance.ensemble: scripts/eval_warmstart.py with several --ckpt and/or --tta) is
judged by the same rules; its epoch is the label ens:<run>@<epoch>,... and its ckpt the member list. It is exempt from
the equal-epoch check (its members are milestone checkpoints), every member's run must be DONE (a member need not be
the final checkpoint), it is told apart from the other regressors by its members and TTA, and it never forms a
loss_pair or seed_pair.
"""
import argparse
import json
import os
import sys
from itertools import combinations

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data_loader.warmstart_paths import EVAL_DIR, GATE_DEFAULT_K as DEFAULT_K  # noqa: E402
from scripts.gate_1a import (YSWAP_THRESHOLD, _need, _paired, _prov, baseline_by_file, check_done, ckpt_epoch,  # noqa: E402
                             paired_vs_baseline, per_cube, row_named)

B_EVAL = os.path.join(EVAL_DIR, 'residual', 'd4', 'residual_d4_b64__checkpoint_epoch_100.json')
TARGET_FRACTION = 0.85          # spec §8: the 1b bar is 15 % below the baseline
ONE_STEP_T = 999                # REGRESS_T and B's ddim1 call (repeated here: the gate stays torch-free)
YSWAP_N = 32                    # plan deviation 1: the proposed rule is pre-registered on 32 stratified y-swap cubes
YSWAP_BASELINE_TOL = 1e-5       # a y-swap scores its baseline in float32 on the evaluation device, the evaluation's own
#                                 baseline_per_item on the CPU: a few ulps apart (1 ulp is 4.8e-7 at 4-8 %), far below
#                                 what another cube or prior changes


def proposed_validity(ys, k=DEFAULT_K):
    """The proposed rule on a y-swap record with per-cube values at the single one-step t (see the module docstring)."""
    per = _need(ys, 'per_cube', 'y-swap record')
    if len(per) != 1:
        raise ValueError(f'the proposed rule needs a y-swap record at a single t, got t = {sorted(per)}')
    (t, cubes), = per.items()
    if int(t) != ONE_STEP_T:
        raise ValueError(f'the proposed rule is defined at the one-step call t = {ONE_STEP_T}, got t = {t}')
    files = _need(ys, 'files', 'y-swap record')
    if files is None:
        raise ValueError('the y-swap record has no files (needed for the clustered SE)')
    true, swap = np.asarray(cubes['true'], float), np.asarray(cubes['swap_full'], float)
    base = np.asarray(_need(ys, 'baseline_per_cube', 'y-swap record'), float)
    a, b, g = _paired(swap - true, files), _paired(swap - base, files), _paired(true - base, files)
    a['pass'] = bool(a['diff'] - k * a['se'] > 0)
    b['pass'] = bool(b['diff'] + k * b['se'] >= 0)
    ratio = float(true.mean() / swap.mean())
    reason = ''.join(part for part, ok in (('a', a['pass']), ('b', b['pass'])) if not ok) or None
    return dict(t=int(t), n=len(files), swap_full_minus_true=a, swap_full_minus_baseline=b, true_minus_baseline=g,
                valid=reason is None, invalid_reason=reason, spec_ratio=ratio, spec_threshold=YSWAP_THRESHOLD,
                spec_rule_passes=bool(ratio <= YSWAP_THRESHOLD))


def paired_by_file(row_x, row_y):
    """row_x - row_y per cube (seed means), paired by file name; both rows must score the same cubes."""
    y = dict(zip(row_y['files'], per_cube(row_y)))
    if set(row_x['files']) != set(y):
        raise ValueError('rows score different cubes')
    return _paired(np.array([v - y[f] for f, v in zip(row_x['files'], per_cube(row_x))]), row_x['files'])


def _ensemble(ev):
    """provenance.ensemble of a regressor-ensemble evaluation; None for a single checkpoint's."""
    return _need(ev, 'provenance').get('ensemble')


def ensemble_members(ens):
    """An ensemble's members as '<run>@<epoch>', in their order."""
    runs, epochs = _need(ens, 'runs', 'provenance.ensemble'), _need(ens, 'epochs', 'provenance.ensemble')
    return [f'{run}@{epoch}' for run, epoch in zip(runs, epochs)]


def ensemble_label(ens):
    """'ens:<run>@<epoch>,...': an ensemble's epoch label."""
    return 'ens:' + ','.join(ensemble_members(ens))


def check_members_done(ev):
    """check_done for an ensemble: every member's run must have finished (a DONE file next to its checkpoint). A member
    need not be its run's final checkpoint (ensembles use milestone checkpoints), but it cannot be past it."""
    ens = _ensemble(ev)
    for ckpt, run in zip(_need(ens, 'ckpts', 'provenance.ensemble'), _need(ens, 'runs', 'provenance.ensemble')):
        done = os.path.join(os.path.dirname(ckpt), 'DONE')
        if not os.path.exists(done):
            raise ValueError(f'ensemble member {ckpt}: no DONE file next to it: the run {run} has not finished')
        with open(done) as f:
            final = int(f.read().strip())
        if ckpt_epoch(ckpt) > final:
            raise ValueError(f'ensemble member {ckpt} is past the final checkpoint of the run {run} '
                             f'(DONE says epoch {final})')


def check_inputs(regs, ev_b, require_done):
    for ev in regs:
        if _prov(ev, 'mode') != 'regress' or ev.get('grid') != 'direct' or _prov(ev, 'prediction_type') != 'direct':
            raise ValueError(f"--reg must be a direct-grid regress evaluation, got mode={_prov(ev, 'mode')!r} "
                             f"grid={ev.get('grid')!r} prediction_type={_prov(ev, 'prediction_type')!r}")
    runs = {}                                    # (loss_type, train_seed) -> ckpt: the keys the pairs are formed on
    for ev in regs:
        key, ckpt = (_prov(ev, 'loss_type'), _prov(ev, 'train_seed')), _prov(ev, 'ckpt')
        if None in key:
            raise ValueError(f'{ckpt}: provenance loss_type={key[0]!r}, train_seed={key[1]!r}: the gate pairs regressors '
                             f'on both, so neither may be None')
        ens = _ensemble(ev)
        if ens is not None:                      # never paired: an ensemble is told apart by its members and TTA
            key = ('ensemble', ensemble_label(ens), ens.get('tta'))
        if key in runs:
            what = (f'the same ensemble {key[1]} with tta {key[2]!r}' if ens is not None else
                    f'both loss_type {key[0]!r}, train_seed {key[1]!r}')
            raise ValueError(f'duplicate runs: {runs[key]} and {ckpt} are {what}')
        runs[key] = ckpt
    if _prov(ev_b, 'mode') != 'residual' or ev_b.get('grid') != 'full':
        raise ValueError(f"--b must be B's full-grid residual evaluation, got mode={_prov(ev_b, 'mode')!r} "
                         f"grid={ev_b.get('grid')!r}")
    evs = list(regs) + [ev_b]
    for ev in evs:
        if not _prov(ev, 'meta_checked'):
            raise ValueError(f"{_prov(ev, 'ckpt')}: checkpoint meta was not checked")
    for key in ('d', 'base_channels', 'prior_sha256'):
        vals = {str(_prov(ev, key)) for ev in evs}
        if len(vals) != 1:
            raise ValueError(f'evaluations differ in provenance.{key}: {sorted(vals)}')
    epochs = {_prov(ev, 'ckpt'): ckpt_epoch(_prov(ev, 'ckpt')) for ev in regs if _ensemble(ev) is None}   # singles
    if len(set(epochs.values())) > 1:
        raise ValueError(f'the regressors stopped at different epochs: {epochs}')
    if require_done:
        for ev in evs:
            if _ensemble(ev) is not None:
                check_members_done(ev)
            else:
                check_done(ev)
    base_b = baseline_by_file(ev_b)
    for ev in regs:
        base = baseline_by_file(ev)
        if base is None or set(base) != set(base_b) or max(abs(base[f] - base_b[f]) for f in base) > 1e-6:
            raise ValueError(f"{_prov(ev, 'ckpt')}: per-cube baselines differ from B's (different cubes or prior)")
        where = f"{_prov(ev, 'ckpt')}: the regressor's y-swap"
        check_yswap_baseline(_need(ev, 'yswap'), base, where)
        check_yswap_n(ev['yswap'], ev['provenance'].get('yswap_n'), 'yswap_n', where)


def check_yswap_n(ys, declared, declared_key, where):
    """Plan deviation 1 pre-registers the proposed rule on YSWAP_N stratified cubes: the y-swap must hold exactly
    YSWAP_N cubes, and the size its producer declares (`declared`: an evaluation's provenance.yswap_n, a
    yswap_one_step record's n; None when it declares none) must be YSWAP_N too."""
    files = _need(ys, 'files', where)
    n = None if files is None else len(files)
    if n != YSWAP_N:
        raise ValueError(f'{where} has {n} cubes; the proposed rule is pre-registered on {YSWAP_N} stratified cubes '
                         f'(plan deviation 1)')
    if declared is not None and declared != YSWAP_N:
        raise ValueError(f'{where} declares {declared_key} {declared}, not the pre-registered {YSWAP_N}')


def check_yswap_baseline(ys, base, where):
    """A y-swap record's per-cube baseline must be its evaluation's (`base`: file -> baseline RMSE %) on its cubes."""
    files, yb = _need(ys, 'files', where), _need(ys, 'baseline_per_cube', where)
    if files is None:
        raise ValueError(f'{where} has no files (needed to pair it with the baseline)')
    if len(files) != len(yb) or any(f not in base for f in files) or \
            max(abs(base[f] - v) for f, v in zip(files, yb)) > YSWAP_BASELINE_TOL:
        raise ValueError(f"{where}: its per-cube baselines differ from its evaluation's on its cubes "
                         f"(different cubes or prior)")


def _same(key, got, want):
    if key == 'checkpoint':
        return os.path.realpath(str(got)) == os.path.realpath(str(want))
    return got is True if key == 'meta_checked' else str(got) == str(want)


def check_b_records(ev_b, b_yswap=None, b_diag=None, b_path=None):
    """--b_yswap must be B's own (scripts/yswap_one_step.py provenance: mode residual, B's checkpoint, d and prior, meta
    checked; its per-cube baseline is B's on its cubes), leak-free (provenance.x_t 'noise': B's ddim1 read-out starts
    from pure noise, while the default q_sample x_t feeds B 0.00635 x the true residual at t = 999; a record without
    x_t predates the option and used q_sample) and of YSWAP_N cubes. --b_diag's provenance.b_eval
    (scripts/diagnose_b.py merge) must name the --b file; that comparison needs `b_path`, which main always passes."""
    if b_yswap is not None:
        prov = _need(b_yswap, 'provenance', '--b_yswap record')
        for key, want in (('mode', 'residual'), ('checkpoint', _prov(ev_b, 'ckpt')), ('d', _prov(ev_b, 'd')),
                          ('prior_sha256', _prov(ev_b, 'prior_sha256')), ('meta_checked', True)):
            got = _need(prov, key, '--b_yswap provenance')
            if not _same(key, got, want):
                raise ValueError(f"--b_yswap is not B's y-swap: provenance.{key} is {got!r}, B's evaluation has {want!r}")
        x_t = prov.get('x_t', 'q_sample')
        if x_t != 'noise':
            note = '' if 'x_t' in prov else ' (no provenance.x_t: a record from before --x_t existed)'
            raise ValueError(f"--b_yswap was made with x_t = {x_t!r}{note}, not 'noise': B's one-step read-out (ddim1) "
                             f"starts from pure noise, while x_t = q_sample(true residual, {ONE_STEP_T}) = 0.00635 r + "
                             f"0.99998 eps leaks the true residual to B; use scripts/yswap_one_step.py --t {ONE_STEP_T} "
                             f"--x_t noise")
        ys = _need(b_yswap, 'yswap', '--b_yswap record')
        check_yswap_baseline(ys, baseline_by_file(ev_b), '--b_yswap')
        check_yswap_n(ys, b_yswap.get('n'), 'n', '--b_yswap')
    if b_diag is not None:
        named = _need(_need(b_diag, 'provenance', 'b_diagnostics'), 'b_eval', 'b_diagnostics provenance')
        if b_path is not None and os.path.realpath(named) != os.path.realpath(b_path):
            raise ValueError(f'--b_diag was made from {named}, not from --b {b_path}')


def judge(ev, ev_b, k):
    rr, rb = row_named(ev, 'direct'), row_named(ev_b, 'ddim1')
    if rr is None or rb is None:
        raise ValueError("needs the regressor's 'direct' row and B's 'ddim1' row")
    vs_base, vs_b = paired_vs_baseline(rr, baseline_by_file(ev)), paired_by_file(rr, rb)
    for r in (vs_base, vs_b):
        r['pass'] = bool(r['diff'] + k * r['se'] < 0)
    validity = proposed_validity(_need(ev, 'yswap'), k)
    verdict_if_valid = 'PASS' if vs_base['pass'] and vs_b['pass'] else ('PASS_NOT_OVER_B' if vs_base['pass'] else 'FAIL')
    baseline = float(_need(ev, 'baseline')['rmse_pct'])
    prov = ev['provenance']
    ens = _ensemble(ev)
    m = dict(run_name=ev.get('run_name'),
             ckpt=prov['ckpt'] if ens is None else list(_need(ens, 'ckpts', 'provenance.ensemble')),
             epoch=ckpt_epoch(prov['ckpt']) if ens is None else ensemble_label(ens),
             loss_type=prov.get('loss_type'), train_seed=prov.get('train_seed'), rmse_pct=rr['rmse_pct'],
             rmse_pct_se=rr['rmse_pct_se'], baseline_rmse_pct=baseline, target_rmse_pct=TARGET_FRACTION * baseline,
             target_met=bool(rr['rmse_pct'] <= TARGET_FRACTION * baseline), vs_baseline=vs_base, vs_b_ddim1=vs_b,
             validity=validity, verdict_if_valid=verdict_if_valid, invalid_reason=validity['invalid_reason'],
             verdict=verdict_if_valid if validity['valid'] else 'INVALID')
    if ens is not None:
        m['ensemble'] = dict(members=ensemble_members(ens), tta=ens.get('tta'), n_calls=ens.get('n_calls'))
    return m


def vs_b_kmean(ev, b_diag, k=DEFAULT_K):
    """Report only: R's direct row minus the mean of B's K one-step draws at t = 999 (K NFE), per cube, from
    b_diagnostics.json (scripts/diagnose_b.py merge)."""
    t9 = _need(_need(b_diag, 'rows', 'b_diagnostics'), 't999', 'b_diagnostics rows')
    km = dict(zip(t9['per_cube']['files'], t9['per_cube']['kmean_rmse']))
    rr = row_named(ev, 'direct')
    if set(rr['files']) != set(km):
        raise ValueError("b_diagnostics' t999 row scores different cubes from the regressor evaluation")
    r = _paired(np.array([v - km[f] for f, v in zip(rr['files'], per_cube(rr))]), rr['files'])
    r['pass'] = bool(r['diff'] + k * r['se'] < 0)
    r['nfe'] = int(t9['K'])
    return r


def gate(regs, ev_b, b_yswap=None, k=DEFAULT_K, require_done=True, b_diag=None, b_path=None):
    check_inputs(regs, ev_b, require_done)
    check_b_records(ev_b, b_yswap, b_diag, b_path)
    models = [judge(ev, ev_b, k) for ev in regs]
    for m, ev in zip(models, regs):
        m['vs_b_t999_kmean'] = None if b_diag is None else vs_b_kmean(ev, b_diag, k)
    rb = row_named(ev_b, 'ddim1')
    b = dict(rmse_pct=rb['rmse_pct'], vs_baseline=paired_vs_baseline(rb, baseline_by_file(ev_b)),
             validity=None if b_yswap is None else proposed_validity(_need(b_yswap, 'yswap'), k))
    b['vs_baseline']['pass'] = bool(b['vs_baseline']['diff'] + k * b['vs_baseline']['se'] < 0)
    b['verdict_under_proposed_rule'] = None if b['validity'] is None else (
        ('PASS' if b['vs_baseline']['pass'] else 'FAIL') if b['validity']['valid'] else 'INVALID')
    pairs = []
    for (i, mi), (j, mj) in combinations(enumerate(models), 2):
        if 'ensemble' in mi or 'ensemble' in mj:            # an ensemble is never one side of a loss or seed pair
            kind = 'other'
        elif mi['loss_type'] != mj['loss_type'] and mi['train_seed'] == mj['train_seed']:
            kind = 'loss_pair'
        elif mi['loss_type'] == mj['loss_type'] and mi['train_seed'] != mj['train_seed']:
            kind = 'seed_pair'
        else:
            kind = 'other'
        rec = dict(kind=kind, a=mi['run_name'], b=mj['run_name'],
                   a_minus_b=paired_by_file(row_named(regs[i], 'direct'), row_named(regs[j], 'direct')))
        if kind == 'seed_pair':
            rec['go_1b'] = all(m['verdict'] == 'PASS' and m['target_met'] and m['validity']['spec_rule_passes']
                               for m in (mi, mj))
        pairs.append(rec)
    passing = [m for m in models if m['verdict'] == 'PASS']
    best = min(passing or models, key=lambda m: m['rmse_pct'])
    return dict(k=k, require_done=require_done, models=models, b=b, pairs=pairs, best=best['run_name'],
                best_verdict=best['verdict'], go_1b=any(p.get('go_1b', False) for p in pairs))


def out_name(run_names, require_done):
    return 'gate_1a_prime__' + '+'.join(run_names) + ('' if require_done else '__nodone') + '.json'


def print_report(g):
    for m in g['models']:
        v, vb, vv = m['vs_baseline'], m['vs_b_ddim1'], m['validity']
        print(f"{m['run_name']} ({m['loss_type']}, seed {m['train_seed']}, epoch {m['epoch']}): {m['verdict']}  "
              f"RMSE {m['rmse_pct']:.4f} ± {m['rmse_pct_se']:.4f} %  target {m['target_rmse_pct']:.4f} "
              f"{'met' if m['target_met'] else 'not met'}")
        if 'ensemble' in m:
            e = m['ensemble']
            print(f"   ensemble of {len(e['members'])} members: {', '.join(e['members'])}; tta {e['tta']}, "
                  f"{e['n_calls']} calls per cube")
        print(f"   - baseline {v['diff']:+.4f} ± {v['se']:.4f} (clustered {v['se_cluster']:.4f})  "
              f"- B ddim1 {vb['diff']:+.4f} ± {vb['se']:.4f} (clustered {vb['se_cluster']:.4f})")
        if m['vs_b_t999_kmean'] is not None:
            vk = m['vs_b_t999_kmean']
            print(f"   - B t999 mean of {vk['nfe']} draws (report only) {vk['diff']:+.4f} ± {vk['se']:.4f} "
                  f"(clustered {vk['se_cluster']:.4f})")
        print(f"   validity at t = {vv['t']} on {vv['n']} cubes: swap_full - true {vv['swap_full_minus_true']['diff']:+.4f} ± "
              f"{vv['swap_full_minus_true']['se']:.4f}, swap_full - baseline {vv['swap_full_minus_baseline']['diff']:+.4f} ± "
              f"{vv['swap_full_minus_baseline']['se']:.4f} -> "
              f"{'valid' if vv['valid'] else 'INVALID (' + vv['invalid_reason'] + ')'} "
              f"(spec 0.9 ratio {vv['spec_ratio']:.3f})")
    b = g['b']
    print(f"B ddim1 {b['rmse_pct']:.4f} %  - baseline {b['vs_baseline']['diff']:+.4f} ± {b['vs_baseline']['se']:.4f}  "
          f"under the proposed rule: {b['verdict_under_proposed_rule']}")
    for p in g['pairs']:
        print(f"{p['kind']}: {p['a']} - {p['b']} {p['a_minus_b']['diff']:+.4f} ± {p['a_minus_b']['se']:.4f}"
              + (f"  go_1b {p['go_1b']}" if 'go_1b' in p else ''))
    print(f"best {g['best']} ({g['best_verdict']}); go_1b {g['go_1b']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--reg', nargs='+', required=True)
    ap.add_argument('--b', default=B_EVAL)
    ap.add_argument('--b_yswap', default=None,
                    help="B's leak-free t = 999 y-swap (scripts/yswap_one_step.py --t 999 --n 32 --x_t noise)")
    ap.add_argument('--b_diag', default=None, help="B's step-0 diagnostics (scripts/diagnose_b.py merge); adds the "
                                                   "report-only comparison with the mean of B's K one-step draws")
    ap.add_argument('--k', type=float, default=DEFAULT_K)
    ap.add_argument('--no_require_done', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)
    regs = [json.load(open(p)) for p in args.reg]
    ev_b = json.load(open(args.b))
    b_yswap = json.load(open(args.b_yswap)) if args.b_yswap else None
    b_diag = json.load(open(args.b_diag)) if args.b_diag else None
    g = gate(regs, ev_b, b_yswap, k=args.k, require_done=not args.no_require_done, b_diag=b_diag, b_path=args.b)
    g['inputs'] = dict(reg=[os.path.abspath(p) for p in args.reg], b=os.path.abspath(args.b),
                       b_yswap=os.path.abspath(args.b_yswap) if args.b_yswap else None,
                       b_diag=os.path.abspath(args.b_diag) if args.b_diag else None)
    print_report(g)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.reg[0])),
                                   out_name([m['run_name'] for m in g['models']], g['require_done']))
    tmp = out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(g, f, indent=1)
    os.replace(tmp, out)
    print('wrote', out)
    return g


if __name__ == '__main__':
    main()
