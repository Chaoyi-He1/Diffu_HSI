"""Verdict of a selection-run regressor at ANY stride, read from an eval_warmstart record (no B comparison): the B.7
reader (a verbatim copy of the scratch d8_verdict.py that produced the B.7 numbers; it reproduces the d = 4 gate's
paired-gain and validity numbers on the d = 4 records). Usage: python scripts/verdict_any_stride.py RECORD.json [more]
"""
import json
import os
import sys

import numpy as np


def cluster_se(diff, files):
    keys = [os.path.dirname(f) + '/' + os.path.basename(f).rsplit('_', 1)[0] for f in files]
    d = np.asarray(diff) - np.mean(diff)
    s = {}
    for k, v in zip(keys, d):
        s[k] = s.get(k, 0.0) + v
    return float(np.sqrt(sum(x * x for x in s.values())) / len(diff)), len(s)


def se(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x)))


for path in sys.argv[1:]:
    e = json.load(open(path))
    r = e['rows'][0]
    b = e['baseline']
    files = r['files']
    assert files == e['baseline_files']
    m = np.mean(np.asarray(r['per_item']['rmse_pct'], float), axis=0)        # mean over seeds (deterministic: identical)
    bi = np.asarray(e['baseline_per_item']['rmse_pct'], float)
    diff = m - bi
    se_c, ncl = cluster_se(diff, files)
    tgt = m - 0.85 * bi
    tse_c, _ = cluster_se(tgt, files)
    ys = e['yswap']
    t = [k for k in ys['per_cube']][0]
    pc = ys['per_cube'][t]
    a = np.asarray(pc['swap_full']) - np.asarray(pc['true'])
    bb = np.asarray(pc['swap_full']) - np.asarray(ys['baseline_per_cube'])
    valid_a = a.mean() - 2 * se(a) > 0
    valid_b = bb.mean() + 2 * se(bb) >= 0
    spec_ratio = float(np.mean(pc['true']) / np.mean(pc['swap_full']))          # the gate's "spec 0.9 ratio": true / swap_full
    prov = e['provenance']
    name = f"{e['run_name']} @ {os.path.basename(prov['ckpt'])} (d = {e['d']}, {prov.get('loss_type')}, seed {prov.get('train_seed')})"
    print(name)
    print(f"  model  {m.mean():.4f} ± {se(m):.4f} %   baseline {bi.mean():.4f} ± {se(bi):.4f} %   ratio {m.mean() / bi.mean():.4f}   (record: {r['rmse_pct']:.4f}, {b['rmse_pct']:.4f})")
    print(f"  paired gain {diff.mean():+.4f} ± {se(diff):.4f} (iid)  ± {se_c:.4f} (clustered, {ncl} source images)  = {100 * diff.mean() / bi.mean():.1f} % of the baseline"
          f"   [record {r['paired_rmse_diff']:+.4f} ± {r['paired_rmse_se']:.4f} / {r['paired_rmse_se_cluster']:.4f}]")
    print(f"  beats baseline by 2 iid SE: {diff.mean() + 2 * se(diff) < 0}")
    print(f"  target 15 %: m - 0.85 b = {tgt.mean():+.4f} ± {se(tgt):.4f} (iid) ± {tse_c:.4f} (clustered); met at the point estimate: {tgt.mean() <= 0}; "
          f"met by 2 iid SE: {tgt.mean() + 2 * se(tgt) <= 0}; by 2 clustered SE: {tgt.mean() + 2 * tse_c <= 0}; target value {0.85 * bi.mean():.4f} %")
    print(f"  validity at t = {t} on {len(a)} cubes: swap_full - true {a.mean():+.4f} ± {se(a):.4f} ({'ok' if valid_a else 'FAIL'}), "
          f"swap_full - baseline {bb.mean():+.4f} ± {se(bb):.4f} ({'ok' if valid_b else 'FAIL'}) -> {'valid' if valid_a and valid_b else 'INVALID'}; "
          f"spec 0.9 ratio {spec_ratio:.3f}; record passes = {ys['passes']}; swap_sensor_only {np.mean(pc['swap_sensor_only']):.4f}, zero {np.mean(pc['zero']):.4f}")
    for c in ['P022', 'P023', 'P024', 'unseen']:
        print(f"  {c}: model {r['rmse_pct_' + c]:.4f} ± {r['rmse_pct_' + c + '_se']:.4f}  baseline {b['rmse_pct_' + c]:.4f}  paired {r['paired_rmse_diff_' + c]:+.4f} ± {r['paired_rmse_se_' + c]:.4f}  ratio {r['rmse_pct_' + c] / b['rmse_pct_' + c]:.3f}")
    print(f"  psnr {r['psnr']:.2f} (baseline {b['psnr']:.2f})  sam {r['sam_deg']:.3f} deg (baseline {b['sam_deg']:.3f})  sec/cube {r['sec_per_cube']:.3f}  git {prov['git_head'][:7]}")
    verdict = 'PASS' if (diff.mean() + 2 * se(diff) < 0 and valid_a and valid_b) else 'FAIL'
    print(f"  VERDICT: {verdict}; target {'met' if tgt.mean() <= 0 else 'not met'} at the point estimate"
          f"{' (and by 2 iid SE)' if tgt.mean() + 2 * se(tgt) <= 0 else ' (not by 2 SE)' if tgt.mean() <= 0 else ''}")
    print()
