"""Idempotent driver for the Phase 1a pilot: observe the run directories, take the next actions, log them.
    python scripts/pilot/pilot_driver.py --once          # one pass
    python scripts/pilot/pilot_driver.py --loop 600      # every 10 min until the gate has run (nohup this)
Policy: B -> GPU 0 and A -> GPU 1; a dead run without DONE is relaunched with --resume auto; A' -> GPU 1 after A is DONE;
quick evals of epochs 25/50 on the run's own GPU; full eval of the final checkpoint on an idle GPU; gate when all three exist.
"""
import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
ROOT = os.environ.get('RESIDUAL_WARMSTART_RESULTS', '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart')
PY = '/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python'
MODES = ('residual', 'standard', 'standard_xhat')
HOME_GPU = {'residual': 0, 'standard': 1, 'standard_xhat': 1}
MILESTONES = (25, 50)
MAX_RELAUNCHES = 3            # a run that keeps dying (e.g. the non-finite abort) is left for a human after 3 relaunches


def run_name(mode):
    return f'{mode}_d4_b64'


def pid_alive(path):
    try:
        pid = int(open(path).read().strip()); os.kill(pid, 0); return True
    except (OSError, ValueError):
        return False


def run_state(run_dir, eval_dir):
    """Everything the policy needs, read from the filesystem."""
    name = os.path.basename(run_dir)
    mode = re.sub(r'_d\d+_b\d+$', '', name)                            # works for *_d4_b64, *_d4_b256 and smoke runs
    if not os.path.isdir(run_dir):
        return dict(mode=mode, exists=False, alive=False, done=None, last_epoch=0, checkpoints=[], quick_evals=[], full_evals=[], relaunches=0)
    cks = sorted(int(m.group(1)) for f in os.listdir(run_dir) for m in [re.match(r'checkpoint_epoch_(\d+)\.pth$', f)] if m)
    done_f = os.path.join(run_dir, 'DONE')
    done = int(open(done_f).read().strip()) if os.path.exists(done_f) else None
    last = 0
    log = os.path.join(run_dir, 'train_log.jsonl')
    if os.path.exists(log):
        for line in open(log):
            try:
                last = max(last, json.loads(line)['epoch'])
            except (ValueError, KeyError):
                pass                                                  # a half-written last line
    evs = glob.glob(os.path.join(eval_dir, mode, 'd4', f'{name}__checkpoint_epoch_*.json'))
    quick = sorted(int(m.group(1)) for p in evs for m in [re.search(r'checkpoint_epoch_(\d+)__quick\.json$', p)] if m)
    full = sorted(int(m.group(1)) for p in evs for m in [re.search(r'checkpoint_epoch_(\d+)\.json$', p)] if m)
    rl = os.path.join(run_dir, 'relaunches')
    relaunches = int(open(rl).read().strip() or 0) if os.path.exists(rl) else 0
    # gate_1a.py names its default output gate_1a__<B run_name>[...].json next to --b, so match by glob
    gate = bool(glob.glob(os.path.join(eval_dir, 'residual', 'd4', 'gate_1a*.json')))
    return dict(mode=mode, exists=True, alive=pid_alive(os.path.join(run_dir, 'pid')), done=done, last_epoch=last,
                checkpoints=cks, quick_evals=quick, full_evals=full, relaunches=relaunches, gate=gate)


def next_actions(states, gpu_status):
    """Pure policy. states: {mode: run_state}; gpu_status: {0: 'free'|'busy', 1: ...}. Returns a list of action tuples."""
    acts, free = [], {g for g, s in gpu_status.items() if s == 'free'}
    s = states
    if all(s[m]['done'] is not None and s[m]['done'] in s[m]['full_evals'] for m in MODES):
        return [] if s['residual'].get('gate') else [('gate',) + tuple(s[m]['done'] for m in MODES)]
    def launchable(st):                                                # not finished, not running, not stuck in a crash loop
        return st['done'] is None and not st['alive'] and st.get('relaunches', 0) < MAX_RELAUNCHES
    for mode in ('residual', 'standard'):                              # launch or relaunch B and A on their home GPUs
        st, g = s[mode], HOME_GPU[mode]
        if launchable(st) and g in free:
            acts.append(('launch', mode, g)); free.discard(g)
    st = s['standard_xhat']
    if s['standard']['done'] is not None and launchable(st) and HOME_GPU['standard_xhat'] in free:
        acts.append(('launch', 'standard_xhat', HOME_GPU['standard_xhat'])); free.discard(HOME_GPU['standard_xhat'])
    for mode in MODES:                                                 # quick milestone evals on the run's own GPU
        st = s[mode]
        for e in MILESTONES:
            if e in st['checkpoints'] and e not in st['quick_evals'] and st['done'] is None:
                acts.append(('eval_quick', mode, e, HOME_GPU[mode]))
    for mode in MODES:                                                 # one full eval per idle GPU
        st = s[mode]
        if st['done'] is not None and st['done'] not in st['full_evals'] and free:
            g = min(free); acts.append(('eval_full', mode, st['done'], g)); free.discard(g)
    return acts


def gpu_status():
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], capture_output=True, text=True).stdout
    uuids = subprocess.run(['nvidia-smi', '--query-gpu=index,gpu_uuid', '--format=csv,noheader'], capture_output=True, text=True).stdout
    idx = {u.strip(): int(i) for i, u in (l.split(',') for l in uuids.strip().splitlines())}
    busy = {idx[l.split(',')[0].strip()] for l in out.strip().splitlines() if l.strip()}
    return {i: ('busy' if i in busy else 'free') for i in idx.values()}


def log_event(msg):
    os.makedirs(os.path.join(ROOT, 'runs'), exist_ok=True)
    line = f'{datetime.datetime.now().isoformat(timespec="seconds")} {msg}'
    print(line, flush=True)
    with open(os.path.join(ROOT, 'runs', 'pilot_events.log'), 'a') as f:
        f.write(line + '\n')


def execute(action):
    kind = action[0]
    sh = os.path.join(REPO, 'scripts', 'pilot')
    if kind == 'launch':
        cmd = ['bash', os.path.join(sh, 'launch_run.sh'), action[1], str(action[2])]
    elif kind in ('eval_quick', 'eval_full'):
        cmd = ['bash', os.path.join(sh, 'eval_milestone.sh'), action[1], str(action[2]), kind[5:], str(action[3])]
    elif kind == 'gate':
        ev = os.path.join(ROOT, 'eval')
        cmd = [PY, os.path.join(REPO, 'scripts', 'gate_1a.py'),
               '--b', os.path.join(ev, 'residual', 'd4', f'residual_d4_b64__checkpoint_epoch_{action[1]}.json'),
               '--a', os.path.join(ev, 'standard', 'd4', f'standard_d4_b64__checkpoint_epoch_{action[2]}.json'),
               '--aprime', os.path.join(ev, 'standard_xhat', 'd4', f'standard_xhat_d4_b64__checkpoint_epoch_{action[3]}.json')]
    else:
        raise ValueError(action)
    log_event(f'START {action}')
    r = subprocess.run(cmd, cwd=REPO)
    log_event(f'END   {action} rc={r.returncode}')


def one_pass():
    states = {m: run_state(os.path.join(ROOT, 'runs', run_name(m)), os.path.join(ROOT, 'eval')) for m in MODES}
    acts = next_actions(states, gpu_status())
    for a in acts:
        execute(a)
    return states, acts


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--once', action='store_true'); ap.add_argument('--loop', type=int, default=0)
    args = ap.parse_args()
    while True:
        states, acts = one_pass()
        if states['residual'].get('gate') or not args.loop:
            break
        time.sleep(args.loop)


if __name__ == '__main__':
    main()
