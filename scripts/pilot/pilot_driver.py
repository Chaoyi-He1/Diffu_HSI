"""Idempotent driver for the Phase 1a pilot: observe the run directories, take the next actions, log them.
    python scripts/pilot/pilot_driver.py --once          # one pass
    python scripts/pilot/pilot_driver.py --loop 600      # every 10 min until the gate has run (nohup this)
Policy: B -> GPU 0 and A -> GPU 1; a dead run without DONE is relaunched with --resume auto; A' -> GPU 1 after A is DONE;
quick evals of epochs 25/50 on the run's own GPU; full eval of the final checkpoint on an idle GPU; gate when all three exist.
A run that is still alive gets no full eval and holds back the gate, even with a DONE file.
Evaluations run synchronously and one at a time. While one runs, the driver polls it every POLL_S s. Each poll re-reads
the state and executes any launch that has become possible, counting the eval's GPU as busy, and logs a BUSY heartbeat
every BUSY_EVERY_S s. A pass whose evaluation saw such a launch leaves the rest of its plan to the next pass (REPLAN).
Unattended safety: one driver per results root (flock on runs/pilot_driver.lock; a second one exits 3); a root whose
official gate file already exists is refused before anything runs (exit 4: use a fresh RESIDUAL_WARMSTART_RESULTS); every launch is
counted (a run gets 1 + MAX_RELAUNCHES launches); an action that fails (non-zero rc or timeout) MAX_ACTION_FAILURES times
is dropped (runs/action_failures.json -- delete the entry to retry); a pass never raises
(it logs ERROR); the loop exits 2 with STUCK when nothing is executable, no run is alive and nvidia-smi answered (e.g. foreign
jobs hold the GPUs) -- a failed nvidia-smi means wait, logged as SMI-FAIL on every pass -- and 0 as soon as the gate has run.
"""
import argparse
import datetime
import fcntl
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from time import monotonic

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from scripts import gate_1a  # noqa: E402  (the gate's output-name contract: gate_path)

ROOT =os.environ.get('RESIDUAL_WARMSTART_RESULTS', '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart')
PY = '/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python'
MODES = ('residual', 'standard', 'standard_xhat')
HOME_GPU = {'residual': 0, 'standard': 1, 'standard_xhat': 1}
MILESTONES = (25, 50)
MAX_RELAUNCHES = 3            # a run that keeps dying (e.g. the non-finite abort) is left for a human: 1 + 3 launches in total
MAX_ACTION_FAILURES = 2       # an action that failed this many times (rc != 0 or timeout) is no longer retried
TIMEOUT_S = {'launch': 3600, 'eval_quick': 2 * 3600, 'eval_full': 8 * 3600, 'gate': 3600}
POLL_S = 60                   # a running action is polled this often; an evaluation's poll may launch a run (eval_tick)
BUSY_EVERY_S = 600            # BUSY heartbeat while an evaluation runs: a silent log no longer means a hung driver
_SMI_FAILING = False          # nvidia-smi failure is logged once per failure episode, not once per pass


def run_name(mode):
    return f'{mode}_d4_b64'


def read_int(path):
    """Integer in a small marker file; None when absent, empty (the writer has opened but not written) or not an integer."""
    try:
        return int(open(path).read().strip())
    except (OSError, ValueError):
        return None


def pid_alive(path):
    pid = read_int(path)
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def gate_path(eval_dir):
    """The OFFICIAL gate file, named by gate_1a.default_out_name next to B's evaluation; --no_require_done / --max_nfe / --k
    peeks write gate_1a__<run>__nodone... .json beside it and do not count."""
    return os.path.join(eval_dir, 'residual', 'd4', gate_1a.default_out_name(run_name('residual')))


def run_state(run_dir, eval_dir):
    """Everything the policy needs, read from the filesystem."""
    name = os.path.basename(run_dir)
    mode = re.sub(r'_d\d+_b\d+$', '', name)                            # works for *_d4_b64, *_d4_b256 and smoke runs
    gate = os.path.exists(gate_path(eval_dir))
    if not os.path.isdir(run_dir):
        return dict(mode=mode, exists=False, alive=False, done=None, last_epoch=0, checkpoints=[], quick_evals=[], full_evals=[], launches=0, gate=gate)
    cks = sorted(int(m.group(1)) for f in os.listdir(run_dir) for m in [re.match(r'checkpoint_epoch_(\d+)\.pth$', f)] if m)
    done = read_int(os.path.join(run_dir, 'DONE'))
    last = 0
    log = os.path.join(run_dir, 'train_log.jsonl')
    if os.path.exists(log):
        for line in open(log):
            try:
                last = max(last, json.loads(line)['epoch'])
            except (ValueError, KeyError, TypeError):
                pass                                                  # a half-written last line
    evs = glob.glob(os.path.join(eval_dir, mode, 'd4', f'{name}__checkpoint_epoch_*.json'))
    quick = sorted(int(m.group(1)) for p in evs for m in [re.search(r'checkpoint_epoch_(\d+)__quick\.json$', p)] if m)
    full = sorted(int(m.group(1)) for p in evs for m in [re.search(r'checkpoint_epoch_(\d+)\.json$', p)] if m)
    launches = max(0, read_int(os.path.join(run_dir, 'launches')) or 0)       # launch_run.sh counts EVERY launch
    return dict(mode=mode, exists=True, alive=pid_alive(os.path.join(run_dir, 'pid')), done=done, last_epoch=last,
                checkpoints=cks, quick_evals=quick, full_evals=full, launches=launches, gate=gate)


def next_actions(states, gpu_status):
    """Pure policy. states: {mode: run_state}; gpu_status: {0: 'free'|'busy', 1: ...}. Returns a list of action tuples.
    A run that is still alive gets no full eval and holds back the gate, even with a DONE file: its process may still be
    exiting, or it was resumed past an old DONE (the trainer removes DONE when it starts)."""
    acts, free = [], {g for g, s in gpu_status.items() if s == 'free'}
    s = states
    if all(s[m]['done'] is not None and s[m]['done'] in s[m]['full_evals'] and not s[m]['alive'] for m in MODES):
        return [] if s['residual'].get('gate') else [('gate',) + tuple(s[m]['done'] for m in MODES)]
    def launchable(st):                                                # not finished, not running, not stuck in a crash loop
        return st['done'] is None and not st['alive'] and st.get('launches', 0) <= MAX_RELAUNCHES
    for mode in ('residual', 'standard'):                              # launch or relaunch B and A on their home GPUs
        st, g = s[mode], HOME_GPU[mode]
        if launchable(st) and g in free:
            acts.append(('launch', mode, g)); free.discard(g)
    st = s['standard_xhat']
    if s['standard']['done'] is not None and launchable(st) and HOME_GPU['standard_xhat'] in free:
        acts.append(('launch', 'standard_xhat', HOME_GPU['standard_xhat'])); free.discard(HOME_GPU['standard_xhat'])
    for mode in MODES:                                                 # quick milestone evals on the run's own GPU (also after DONE if missed)
        st = s[mode]
        for e in MILESTONES:
            if e in st['checkpoints'] and e not in st['quick_evals']:
                acts.append(('eval_quick', mode, e, HOME_GPU[mode]))
    for mode in MODES:                                                 # one full eval per idle GPU, of a finished run only
        st = s[mode]
        if st['done'] is not None and not st['alive'] and st['done'] not in st['full_evals'] and free:
            g = min(free); acts.append(('eval_full', mode, st['done'], g)); free.discard(g)
    return acts


def failures_path():
    return os.path.join(ROOT, 'runs', 'action_failures.json')


def load_failures(path):
    """{repr(action): failure count}; empty when the file is absent, corrupt or not a JSON object."""
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return {}
    if not isinstance(d, dict):
        return {}
    return {str(k): v for k, v in d.items() if isinstance(v, int) and not isinstance(v, bool)}


def save_failures(path, failures):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.tmp', 'w') as f:
        json.dump(failures, f, indent=1, sort_keys=True)
    os.replace(path + '.tmp', path)


def executable(actions, failures):
    """Drop the actions that have already failed MAX_ACTION_FAILURES times."""
    return [a for a in actions if failures.get(repr(a), 0) < MAX_ACTION_FAILURES]


def is_stuck(states, todo, gpus=None):
    """Nothing executable this pass and no run alive: nothing can progress without a human (e.g. foreign jobs hold the
    GPUs). gpus == {} -- the pass's gpu_status(), empty when nvidia-smi failed -- is never STUCK: the loop waits."""
    if gpus == {}:
        return False
    return not todo and not any(st['alive'] for st in states.values())


def parse_gpu_status(uuid_lines, app_lines):
    """uuid_lines: `nvidia-smi --query-gpu=index,gpu_uuid` output; app_lines: `--query-compute-apps=gpu_uuid,pid` output. Skips unparsable lines."""
    idx = {}
    for line in uuid_lines.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 2 and parts[0].isdigit():
            idx[parts[1]] = int(parts[0])
    busy = {idx[line.split(',')[0].strip()] for line in app_lines.splitlines() if line.split(',')[0].strip() in idx}
    return {i: ('busy' if i in busy else 'free') for i in idx.values()}


def _smi(args):
    """stdout of nvidia-smi, or None when it cannot be run, times out or fails."""
    try:
        r = subprocess.run(['nvidia-smi'] + args, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def gpu_status():
    """{gpu index: 'free'|'busy'}; {} (so nothing is launched this pass) when nvidia-smi fails."""
    global _SMI_FAILING
    uuids = _smi(['--query-gpu=index,gpu_uuid', '--format=csv,noheader'])
    apps = _smi(['--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader']) if uuids is not None else None
    if uuids is None or apps is None:
        if not _SMI_FAILING:
            log_event('nvidia-smi failed; no GPU is treated as free until it answers again')
        _SMI_FAILING = True
        return {}
    _SMI_FAILING = False
    return parse_gpu_status(uuids, apps)


def log_event(msg):
    os.makedirs(os.path.join(ROOT, 'runs'), exist_ok=True)
    line = f'{datetime.datetime.now().isoformat(timespec="seconds")} {msg}'
    print(line, flush=True)
    with open(os.path.join(ROOT, 'runs', 'pilot_events.log'), 'a') as f:
        f.write(line + '\n')


def run_cmd(cmd, timeout, on_tick=None):
    """Run cmd in its own session and wait up to `timeout` s, polling every POLL_S s and calling on_tick(elapsed_s) between
    polls (a tick that raises is logged, never propagated: the child is still waited for). On timeout kill the whole group
    (bash, python, tee). Returns rc or 'timeout'."""
    try:
        p = subprocess.Popen(cmd, cwd=REPO, start_new_session=True)
    except OSError as e:
        return f'oserror: {e}'                                         # counts as a failure like any non-zero rc
    t0 = monotonic()
    while True:
        left = timeout - (monotonic() - t0)
        if left <= 0:
            break
        try:
            return p.wait(timeout=min(POLL_S, left))
        except subprocess.TimeoutExpired:
            pass
        elapsed = monotonic() - t0
        if on_tick is not None and elapsed < timeout:
            try:
                on_tick(elapsed)
            except Exception:
                try:
                    log_event(f'ERROR in a tick of {cmd[1:]}: ' + traceback.format_exc().rstrip())
                except Exception:
                    pass
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except OSError:
        pass
    p.wait()
    return 'timeout'


def execute(action, on_tick=None):
    """Run one action synchronously; a non-zero rc or a timeout is counted in runs/action_failures.json, a success clears the count.
    on_tick is passed to run_cmd (one_pass gives evaluations eval_tick)."""
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
    rc = run_cmd(cmd, TIMEOUT_S[kind]) if on_tick is None else run_cmd(cmd, TIMEOUT_S[kind], on_tick=on_tick)
    log_event(f'END   {action} rc={rc}')
    path = failures_path(); failures = load_failures(path); key = repr(action)
    if rc != 0:
        failures[key] = failures.get(key, 0) + 1
        save_failures(path, failures)
    elif key in failures:
        del failures[key]
        save_failures(path, failures)
    return rc


def read_states():
    return {m: run_state(os.path.join(ROOT, 'runs', run_name(m)), os.path.join(ROOT, 'eval')) for m in MODES}


def eval_tick(action, executed):
    """on_tick for a running evaluation `action`: a BUSY heartbeat every BUSY_EVERY_S s, then re-read the state and execute
    (through execute(), so they are counted) the launches that became possible -- launches only, evaluations stay serial.
    The evaluation's own GPU counts as busy: nvidia-smi may not list its CUDA context yet. Executed launches go to `executed`."""
    gpu, next_busy = action[3], [BUSY_EVERY_S]

    def tick(elapsed_s):
        if elapsed_s >= next_busy[0]:
            log_event(f'BUSY {action} elapsed={elapsed_s / 60:.0f}min')
            while next_busy[0] <= elapsed_s:
                next_busy[0] += BUSY_EVERY_S
        gpus = dict(gpu_status())
        if gpus:
            gpus[gpu] = 'busy'
        launches = [a for a in next_actions(read_states(), gpus) if a[0] == 'launch']
        for a in executable(launches, load_failures(failures_path())):
            execute(a)
            executed.append(a)
    return tick


def one_pass(observed=None):
    """Observe, act, observe again. Returns (states after the actions, actions executed); never raises (logs ERROR, returns (None, [])).
    observed (a dict, optional) receives the pass's GPU status under 'gpus' ({} when nvidia-smi failed: SMI-FAIL is logged)."""
    try:
        states = read_states()
        gpus = gpu_status()
        if observed is not None:
            observed['gpus'] = gpus
        if not gpus:
            log_event('SMI-FAIL no GPU status from nvidia-smi; no GPU counts as free this pass; waiting for it to answer')
        acts = next_actions(states, gpus)
        failures = load_failures(failures_path())
        todo = executable(acts, failures)
        for a in acts:
            if a not in todo:
                log_event(f'SKIP {a} (failed {failures.get(repr(a), 0)} times)')
        if not acts:
            log_event(f"IDLE alive={[m for m in MODES if states[m]['alive']]} done={ {m: states[m]['done'] for m in MODES} }")
        executed = []
        for i, a in enumerate(todo):
            executed.append(a)
            if a[0] not in ('eval_quick', 'eval_full'):
                execute(a)
                continue
            n = len(executed)
            execute(a, on_tick=eval_tick(a, executed))
            if len(executed) > n and todo[i + 1:]:                     # a launch took a GPU this plan may have counted as free
                log_event(f'REPLAN {todo[i + 1:]} left for the next pass (launched during {a})')
                break
        return read_states(), executed
    except Exception:
        log_event('ERROR ' + traceback.format_exc().rstrip())
        return None, []


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--once', action='store_true'); ap.add_argument('--loop', type=int, default=0)
    args = ap.parse_args(argv)
    os.makedirs(os.path.join(ROOT, 'runs'), exist_ok=True)
    with open(os.path.join(ROOT, 'runs', 'pilot_driver.lock'), 'a+') as lock:     # held until this driver exits
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock.seek(0)
            log_event(f'LOCKED another pilot driver holds {lock.name} (last recorded pid {lock.read().strip() or "?"}); exiting')
            return 3
        lock.seek(0); lock.truncate(); lock.write(f'{os.getpid()}\n'); lock.flush()
        gate = gate_path(os.path.join(ROOT, 'eval'))
        if os.path.exists(gate):                                       # a finished campaign: its evals would be reused by name
            log_event(f'USED ROOT {gate} exists before the first pass; not launching anything '
                      '(start a new campaign under a fresh RESIDUAL_WARMSTART_RESULTS root)')
            return 4
        while True:
            observed = {}
            states, todo = one_pass(observed)
            if states is not None:
                if states['residual'].get('gate'):
                    log_event('GATE file present; the pilot driver is finished')
                    return 0
                if args.loop and is_stuck(states, todo, observed.get('gpus')):
                    log_event('STUCK no executable action and no run alive; a human must look: '
                              + str({m: dict(done=st['done'], launches=st['launches'], last_epoch=st['last_epoch'], full_evals=st['full_evals'])
                                     for m, st in states.items()})
                              + f" gpus={observed.get('gpus')} (failed actions: {failures_path()})")
                    return 2
            if not args.loop:
                return 0 if states is not None else 1
            time.sleep(args.loop)


if __name__ == '__main__':
    sys.exit(main())
