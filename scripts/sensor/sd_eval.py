"""Phase 3 evaluation: one record per (sensor device, measurement mode, SNR, subset) scoring, per cube, the classical
MMSE baseline L (and L_demosaic in snapshot mode), the family regressor D, the variance head V, the family-conditioned
diffusion C (K draws) and its one-call point C1, the guided unconditional prior B (K ΠGDM / DPS draws) and B+ (method
Bp: the same K draws, each replaced by a last-step Gaussian posterior draw around it, model/posterior_draw.py; needs
--b_kernel, the error covariance fit_b_kernel.py measured on held-out calibration cubes). Everything
runs in float32 without autocast (the guidance operator is cast to y's dtype). Records go to
results/residual_warmstart/sensor/eval/<name>.json and are never overwritten.

    python scripts/sensor/sd_eval.py --name dev1_seq_30 --sensor_device device1 --mode sequential --snr_db 30 \
        --regress <ckpt> --varhead <ckpt> --cond <ckpt> --uncond <ckpt> --K 32 --pigdm_steps 50 --device cuda:0

--methods defaults to every applicable method: L always, L_demosaic in snapshot mode, D / V / C and C1 / B when
--regress / --varhead / --cond / --uncond is given (Bp when --b_kernel is given too). A requested method without its checkpoint is refused; a checkpoint
whose method is not requested is ignored. Sample coverage is exact only for K >= min_k_for(0.9) (coverage_exact).

Phase 3b (spec 2026-11-16 Part B): --scenario NAME (scripts/sensor/scenarios_3b.py) evaluates one pre-registered change
of the forward model, sequential mode only; --snr_db and --sensor_device are taken from the scenario (given, they must
match). L / D / V / C / C1 read the batch's aware statistics; L_naive / D_naive (non-Gaussian noise only) the naive
ones; L_interp (pixel patterns only) interpolates the measured readings over the image and adds the interpolation-error
variance e2 (measured once per run on the first N_E2_CUBES calibration cubes, setting.e2_interp). B / Bp guide and
condition with the scenario's R, its mask (when not all ones) and its per-element noise variance (non-Gaussian noise).

    python scripts/sensor/sd_eval.py --name b3_stride_4_30 --scenario stride_4_30 --regress <ckpt> --uncond <ckpt> \
        --b_kernel <kernel> --K 20 --pigdm_steps 50 --device cuda:0
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from data_loader.hfd_sensor_sim import HFDSensorSim, load_cube01  # noqa: E402
from data_loader.mmse import demosaic_then_mmse  # noqa: E402
from data_loader.scenarios import (has_naive, has_pixels, interp_error_e2, interp_then_mmse, load_consts,  # noqa: E402
                                   needs_consts, scenario_sha)
from data_loader.sensor_config import CONFIG_PATH, SENSOR_DIR, load_config, training_sha  # noqa: E402
from data_loader.sensor_forward import guidance_operator  # noqa: E402
from data_loader.warmstart_paths import DATA_ROOT, PRIOR, file_sha256, git_head  # noqa: E402
from model.diffusion_trainer import DiffusionTrainer  # noqa: E402
from model.guidance import LinearMeasurement, draw_guided  # noqa: E402
from model.posterior_draw import kernel_sha256, load_kernel, posterior_draws  # noqa: E402
from model.residual_wrapper import LOGVAR_RANGE, build_residual_model, compose, regress_forward, varhead_forward  # noqa: E402
from data_loader.hfd_residual import limit_worker_threads  # noqa: E402
from scripts.eval_warmstart import class_of, load_subset  # noqa: E402
from scripts.sensor.scenarios_3b import SCENARIOS  # noqa: E402
from scripts.uq.metrics import min_k_for  # noqa: E402
from scripts.uq_eval import gaussian_metrics, point_metrics, sample_metrics  # noqa: E402

METHODS = ('L', 'L_demosaic', 'L_naive', 'L_interp', 'D', 'D_naive', 'V', 'C', 'C1', 'B', 'Bp')
EVAL_DIR = os.path.join(SENSOR_DIR, 'eval')
CKPT_FLAG = {'D': 'regress', 'D_naive': 'regress', 'V': 'varhead', 'C': 'cond', 'C1': 'cond', 'B': 'uncond', 'Bp': 'uncond'}
SHARED_MODEL = ('C1', 'Bp', 'D_naive')          # methods that reuse another method's model (C, B, D)
NAIVE_METHODS = ('L_naive', 'D_naive')          # Phase 3b: non-Gaussian noise scenarios only
SCENARIO_CONSTS = os.path.join(SENSOR_DIR, 'scenario_consts.json')
N_E2_CUBES = 20                                 # L_interp: e2 from the first 20 calibration cubes, seed 0
GUIDED = ('B', 'Bp')                            # both are scored on the same K guided draws
FLAG_MODE = {'regress': 'regress_fam', 'varhead': 'varhead_fam', 'cond': 'cond_fam', 'uncond': 'uncond'}
FLAG_ALLOWED_MODES = {**{f: (m,) for f, m in FLAG_MODE.items()}, 'uncond': ('uncond', 'uncond_tt')}   # B: either prior
SAMPLED_FLAGS = ('cond', 'uncond')              # sampled on N_TIMESTEPS steps
N_TIMESTEPS = 1000
C_CLAMP = (-8.0, 8.0)                           # the standardised residual's x0 estimate in C's sampler
C_STEPS = 10                                    # C: DDIM-10, draw k under torch.manual_seed(k)
C1_STEPS, C1_SEED = 1, 0                        # C1 (spec §5.3 one-call point of C): one DDIM step under torch.manual_seed(0)


# ----------------------------------------------------------------------------- methods and checkpoints
def scenario_of(args):
    """The scenario dict of args.scenario (None without one)."""
    name = getattr(args, 'scenario', None)
    if not name:
        return None
    if name not in SCENARIOS:
        raise ValueError(f'unknown scenario {name!r}; expected one of {sorted(SCENARIOS)}')
    return SCENARIOS[name]['scenario']


def apply_scenario_args(args):
    """Validate and fill the arguments of a scenario run (in place): --snr_db / --sensor_device come from the scenario
    (given, they must match), --mode must be sequential, and --scenario_consts (default SCENARIO_CONSTS) must exist when
    the noise kind needs the constants (it is set to None otherwise). Without --scenario, --snr_db is required.
    Returns the scenario dict or None."""
    scen = scenario_of(args)
    if scen is None:
        if args.snr_db is None:
            raise ValueError('--snr_db is required (unless --scenario is given)')
        return None
    s = SCENARIOS[args.scenario]
    if args.mode != 'sequential':
        raise ValueError(f'--scenario {args.scenario} is measured in sequential mode, got --mode {args.mode}')
    if args.sensor_device != s['sensor_device']:
        raise ValueError(f"--scenario {args.scenario} is built on {s['sensor_device']}, got --sensor_device {args.sensor_device}")
    if args.snr_db is not None and float(args.snr_db) != s['snr_db']:
        raise ValueError(f"--scenario {args.scenario} is at {s['snr_db']} dB, got --snr_db {args.snr_db}")
    args.snr_db = s['snr_db']
    if needs_consts(scen):
        args.scenario_consts = getattr(args, 'scenario_consts', None) or SCENARIO_CONSTS
        if not os.path.isfile(args.scenario_consts):
            raise FileNotFoundError(f'scenario_consts not found: {args.scenario_consts} (run scripts/sensor/make_scenario_consts.py)')
    else:
        args.scenario_consts = None
    return scen


def resolve_methods(args):
    """args.methods if given (validated), else every applicable method."""
    snapshot = args.mode == 'snapshot'
    scen = scenario_of(args)
    naive = scen is not None and has_naive(scen)
    pixels = scen is not None and has_pixels(scen)
    if args.methods:
        bad = [m for m in args.methods if m not in METHODS]
        if bad:
            raise ValueError(f'unknown methods {bad}; expected a subset of {METHODS}')
        if 'L_demosaic' in args.methods and not snapshot:
            raise ValueError('L_demosaic needs --mode snapshot (there is no mosaic to demosaic in sequential mode)')
        if 'Bp' in args.methods and not getattr(args, 'b_kernel', None):
            raise ValueError('Bp needs --b_kernel (the kernel file fit_b_kernel.py wrote)')
        for m in NAIVE_METHODS:
            if m in args.methods and not naive:
                raise ValueError(f'{m} needs a --scenario with non-Gaussian noise (poisson / saturation / quantisation)')
        if 'L_interp' in args.methods and not pixels:
            raise ValueError('L_interp needs a --scenario with a pixel pattern (dead / stride)')
        return [m for m in METHODS if m in args.methods]
    return [m for m in METHODS if m == 'L' or (m == 'L_demosaic' and snapshot)
            or (m == 'L_naive' and naive) or (m == 'L_interp' and pixels)
            or (m in CKPT_FLAG and getattr(args, CKPT_FLAG[m], None) and (m != 'Bp' or getattr(args, 'b_kernel', None))
                and (m != 'D_naive' or naive))]


def ckpt_epoch(path):
    m = re.search(r'checkpoint_epoch_(\d+)\.pth$', path)
    return int(m.group(1)) if m else None


def check_models(args, cfg, device):
    """Load the checkpoint of every requested method (C and C1 share --cond) and refuse (ValueError, naming the file)
    one whose meta mode is not the method's or whose sensor_config_sha is not training_sha(cfg), and a D / V / C checkpoint
    trained on another device than cfg['device_train'] (meta sensor_device; missing = device_train); the sampled models
    (cond must be v-prediction, uncond v or x0). Returns (models keyed D / V / C / B, provenance keyed by the flag)."""
    sha = training_sha(cfg)
    models, prov = {}, {}
    for flag in dict.fromkeys(CKPT_FLAG[m] for m in resolve_methods(args) if m in CKPT_FLAG):
        path = getattr(args, flag, None)
        if not path:
            need = [m for m, f in CKPT_FLAG.items() if f == flag]
            raise ValueError(f'methods {need} need --{flag} (a {FLAG_MODE[flag]} checkpoint)')
        ck = torch.load(path, map_location='cpu', weights_only=False)
        meta = (ck.get('meta') if isinstance(ck, dict) else None) or {}
        bad = []
        if meta.get('mode') not in FLAG_ALLOWED_MODES[flag]:
            want = ' or '.join(repr(m) for m in FLAG_ALLOWED_MODES[flag])
            bad.append(f"mode: checkpoint {meta.get('mode', '<missing>')!r} vs required {want}")
        if meta.get('sensor_config_sha') != sha:
            bad.append(f"sensor_config_sha: checkpoint {meta.get('sensor_config_sha', '<missing>')!r} vs config {sha!r}")
        # the sha hashes the constant device_train, not the device the run used; the uncond prior is sensor-free.
        # A missing field counts as device_train (launch_run.sh never passed --sensor_device, the trainer's default).
        if flag != 'uncond' and meta.get('sensor_device', cfg['device_train']) != cfg['device_train']:
            bad.append(f"sensor_device: checkpoint {meta.get('sensor_device')!r} vs config device_train {cfg['device_train']!r}")
        allowed_pt = {'cond': ('v',), 'uncond': ('v', 'x0')}.get(flag)   # B's prior may be v- or x0-prediction
        if allowed_pt and meta.get('prediction_type', 'v') not in allowed_pt:
            want = ' or '.join(repr(a) for a in allowed_pt)
            bad.append(f"prediction_type: checkpoint {meta.get('prediction_type')!r} vs required {want}")
        if bad:
            raise ValueError(f'--{flag} checkpoint {path} does not fit this evaluation: ' + '; '.join(bad))
        model = build_residual_model(meta['mode'], int(meta['base_channels']))
        model.load_state_dict(ck['model_state_dict'], strict=True)
        model = model.float().to(device).eval()
        for m, f in CKPT_FLAG.items():
            if f == flag and m not in SHARED_MODEL:
                models[m] = model
        prov[flag] = dict(path=os.path.abspath(path), epoch=ckpt_epoch(path), sha256=file_sha256(path),
                          mode=meta.get('mode'), git_head=meta.get('git_head'))
        if flag == 'uncond':
            prov[flag]['prediction_type'] = meta.get('prediction_type', 'v')
    return models, prov


# ----------------------------------------------------------------------------- the run
def _extend(acc, method, d):
    a = acc.setdefault(method, {})
    for k, v in d.items():
        a.setdefault(k, []).extend(v)


def _demosaic_baseline(batch, ds, sigma, e2):
    """demosaic_then_mmse per cube on the CPU -> (x_hat, std), [B, 64, H, W] float32 on the trainer scale."""
    xs, ss = [], []
    for i in range(batch['y'].shape[0]):
        y = batch['y'][i].permute(1, 2, 0).numpy()
        mask = batch['mask'][i].permute(1, 2, 0).numpy()
        xh, sd = demosaic_then_mmse(y, mask, ds.R, ds.prior, sigma, ds.A, e2)
        xs.append(torch.from_numpy(np.ascontiguousarray(xh.transpose(2, 0, 1))))
        ss.append(torch.from_numpy(np.ascontiguousarray(sd.transpose(2, 0, 1))))
    return torch.stack(xs).float(), torch.stack(ss).float()


def _interp_baseline(batch, ds, sigma, e2):
    """interp_then_mmse per cube on the CPU (the measured pixels' readings interpolated over the image) -> (x_hat, std),
    [B, 64, H, W] float32 on the trainer scale."""
    xs, ss = [], []
    for i in range(batch['y'].shape[0]):
        y = batch['y'][i].permute(1, 2, 0).numpy()
        measured = batch['measured'][i, 0].numpy() > 0.5
        xh, sd = interp_then_mmse(y, measured, ds.R, ds.prior, sigma, ds.A, e2)
        xs.append(torch.from_numpy(np.ascontiguousarray(xh.transpose(2, 0, 1))))
        ss.append(torch.from_numpy(np.ascontiguousarray(sd.transpose(2, 0, 1))))
    return torch.stack(xs).float(), torch.stack(ss).float()


def calibration_cubes01(data_root, n=N_E2_CUBES):
    """The first n calibration cubes (fit_b_kernel.calibration_files; never a test cube) on the x01 scale."""
    from scripts.sensor.fit_b_kernel import calibration_files     # lazy: fit_b_kernel imports this module
    return [load_cube01(os.path.join(data_root, f)) for f in calibration_files(n=n)]


def c_sampler_provenance(methods, K):
    """C's / C1's sampler as the record states it (B's is pigdm_steps / guidance / b_prediction_type); None without C or C1."""
    if 'C' not in methods and 'C1' not in methods:
        return None
    out = dict(method='ddim', prediction_type='v', n_timesteps=N_TIMESTEPS, clamp_x0=list(C_CLAMP))
    if 'C' in methods:
        out['C'] = dict(n_steps=C_STEPS, seeds=list(range(K)), rule='draw k under torch.manual_seed(k); point = mean of K')
    if 'C1' in methods:
        out['C1'] = dict(n_steps=C1_STEPS, seed=C1_SEED, rule=f'one DDIM call of the C model under torch.manual_seed({C1_SEED}) (spec §5.3)')
    return out


@torch.no_grad()
def run_eval(models, ds, loader, args, cfg, provenance, device):
    methods = resolve_methods(args)
    K, snapshot = int(args.K), args.mode == 'snapshot'
    scen = scenario_of(args)
    e2_interp = None
    if 'L_interp' in methods:                    # once per run, on clean calibration readings with this scenario's R
        e2_interp = interp_error_e2(calibration_cubes01(args.data_root), ds.R, scen['pixels'], 0)
    if scen is not None and needs_consts(scen):
        consts_prov = dict(path=os.path.abspath(args.scenario_consts), sha256=file_sha256(args.scenario_consts))
    else:
        consts_prov = None
    if 'D_naive' in methods and 'D' not in models:
        raise ValueError('D_naive needs the --regress model')
    trainer = DiffusionTrainer(device=device, n_timesteps=N_TIMESTEPS, loss_type='l1', prediction_type='v', snr_gamma=None)
    b_pt = (provenance.get('uncond') or {}).get('prediction_type', 'v')
    guided = [m for m in methods if m in GUIDED]
    kernel = None
    if 'Bp' in methods:
        Sigma_e, kmeta = load_kernel(args.b_kernel)
        if kmeta.get('uncond_sha256') and kmeta['uncond_sha256'] != (provenance.get('uncond') or {}).get('sha256'):
            raise ValueError(f"--b_kernel {args.b_kernel} was fitted for another prior checkpoint "
                             f"({kmeta['uncond_sha256'][:12]}... vs {provenance['uncond']['sha256'][:12]}...)")
        kernel = dict(path=os.path.abspath(args.b_kernel), sha256=kernel_sha256(Sigma_e), meta=kmeta)
    if guided:
        trainer_B = DiffusionTrainer(device=device, n_timesteps=N_TIMESTEPS, loss_type='l1', prediction_type=b_pt,
                                     snr_gamma=None)               # B's prior may be x0-prediction; C keeps the 'v' trainer
        Hn, bn = guidance_operator(ds.R, ds.A)
        H_op, b_op = torch.from_numpy(np.asarray(Hn)), torch.from_numpy(np.asarray(bn))
    acc, sigma, t0, saved = {}, None, time.time(), []
    for bi, batch in enumerate(loader):
        s_b = batch['sigma']
        if not bool(torch.all(s_b == s_b[0])) or (sigma is not None and float(s_b[0]) != sigma):
            raise ValueError('the cubes of one record must share one sigma (a fixed setting)')
        sigma = float(s_b[0])
        x, x0_hat = batch['x'].float().to(device), batch['x0_hat'].float().to(device)
        s_post, s_raw = batch['s_post'].float().to(device), batch['s_raw'].float().to(device)
        cond = {'sensor': None, 'x0_hat': x0_hat, 's_post': s_post}
        shape = tuple(x.shape)
        if 'L' in methods:
            _extend(acc, 'L', gaussian_metrics(x0_hat, s_raw, x))
        if 'L_demosaic' in methods:
            xd, sd = _demosaic_baseline(batch, ds, sigma, cfg['e2_demosaic'])
            _extend(acc, 'L_demosaic', gaussian_metrics(xd.to(device), sd.to(device), x))
        if 'L_naive' in methods or 'D_naive' in methods:
            x0_hat_n = batch['x0_hat_naive'].float().to(device)
            s_post_n, s_raw_n = batch['s_post_naive'].float().to(device), batch['s_raw_naive'].float().to(device)
        if 'L_naive' in methods:
            _extend(acc, 'L_naive', gaussian_metrics(x0_hat_n, s_raw_n, x))
        if 'L_interp' in methods:
            xi, si = _interp_baseline(batch, ds, sigma, e2_interp)
            _extend(acc, 'L_interp', gaussian_metrics(xi.to(device), si.to(device), x))
        if 'D' in methods:
            _extend(acc, 'D', point_metrics(compose(x0_hat, s_post, regress_forward(models['D'], cond)), x))
        if 'D_naive' in methods:
            cond_n = {'sensor': None, 'x0_hat': x0_hat_n, 's_post': s_post_n}
            _extend(acc, 'D_naive', point_metrics(compose(x0_hat_n, s_post_n, regress_forward(models['D'], cond_n)), x))
        if 'V' in methods:
            logvar = varhead_forward(models['V'], cond).clamp(*LOGVAR_RANGE)
            _extend(acc, 'V', gaussian_metrics(x0_hat, s_post * torch.exp(0.5 * logvar), x))
        if 'C' in methods:
            draws = []
            for k in range(K):
                torch.manual_seed(k)
                r = trainer.sample(models['C'], cond, shape, n_steps=C_STEPS, method='ddim', progress=False, clamp_x0=C_CLAMP)
                draws.append(compose(x0_hat, s_post, r))
            _extend(acc, 'C', sample_metrics(torch.stack(draws, 0), x))
        if 'C1' in methods:
            torch.manual_seed(C1_SEED)
            r1 = trainer.sample(models['C'], cond, shape, n_steps=C1_STEPS, method='ddim', progress=False, clamp_x0=C_CLAMP)
            _extend(acc, 'C1', point_metrics(compose(x0_hat, s_post, r1), x))
        if guided:
            y = batch['y'].float().to(device)
            if scen is None:                     # Phase 3: exactly as before
                mask = batch['mask'].float().to(device) if snapshot else None
                meas = LinearMeasurement(H_op, b_op, y, mask, sigma)
                sigma2 = sigma ** 2
            else:                                # the scenario's mask (unless all ones) and per-element noise variance
                mask = batch['mask'].float().to(device)
                mask = None if bool(torch.all(mask == 1)) else mask
                nv = batch['noise_var'].float().to(device) if scen['noise']['kind'] != 'gaussian' else None
                meas = LinearMeasurement(H_op, b_op, y, mask, sigma, noise_var=nv)
                sigma2 = sigma ** 2 if nv is None else nv
            draws = draw_guided(models['B'], trainer_B, meas, shape, K, int(args.pigdm_steps), args.guidance)
            if 'B' in methods:
                _extend(acc, 'B', sample_metrics(draws, x))
            if 'Bp' in methods:
                _extend(acc, 'Bp', sample_metrics(posterior_draws(draws, H_op, b_op, y, sigma2, Sigma_e, mask=mask), x))
            if getattr(args, 'save_draws', None):
                saved.append(draws.half().cpu().numpy())
        print(f'batch {bi + 1}/{len(loader)} ({time.time() - t0:.0f} s)', flush=True)
    means = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in acc.items()}
    if saved:                                    # B's raw guided draws [K, n, C, H, W] float16, for later re-scoring
        os.makedirs(args.save_draws, exist_ok=True)
        np.savez(os.path.join(args.save_draws, args.name + '.npz'), draws=np.concatenate(saved, 1),
                 files=np.array([os.path.relpath(f, args.data_root) for f in ds.img_list]), sigma=sigma, K=K)
    setting = dict(sensor_device=args.sensor_device, mode=args.mode, snr_db=float(args.snr_db), sigma=sigma,
                   R_sha256=ds.sensor['sha256'], seed_scheme=ds.checkpoint_extra()['seed_scheme'])
    prov = dict(sensor_config_sha=training_sha(cfg), config_path=os.path.abspath(args.sensor_config),
                checkpoints=provenance, b_prediction_type=b_pt if guided else None, b_kernel=kernel,
                c_sampler=c_sampler_provenance(methods, K),
                git_head=git_head(), created=time.strftime('%Y-%m-%dT%H:%M:%S'),
                seconds=time.time() - t0, seeds=list(range(K)), device=str(device),
                batch_size=int(args.batch_size), limit_files=int(args.limit_files))
    if scen is not None:                         # Phase 3b keys, only on scenario records
        setting.update(scenario=scen, scenario_sha=scenario_sha(scen), R_sha256=ds.R_sha256,
                       e2_interp=None if e2_interp is None else [float(v) for v in e2_interp])
        prov['scenario_consts'] = consts_prov
    return dict(name=args.name, setting=setting,
                subset=args.subset, K=K, pigdm_steps=int(args.pigdm_steps), guidance=args.guidance, n=len(ds),
                files=[os.path.relpath(f, args.data_root) for f in ds.img_list], classes=[class_of(f) for f in ds.img_list],
                per_cube=acc, means=means, coverage_exact=bool(K >= min_k_for(0.9)), provenance=prov)


def output_path(name):
    return os.path.join(EVAL_DIR, name + '.json')


def write_record(path, record):
    if os.path.exists(path):
        raise FileExistsError(f'{path} exists; records are never overwritten (pick another --name)')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(record, f)
    os.replace(tmp, path)


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--sensor_device', choices=['device1', 'device2', 'ph5'], default='device1')
    ap.add_argument('--mode', choices=['sequential', 'snapshot'], default='sequential')
    ap.add_argument('--snr_db', type=float, default=None, help='required unless --scenario is given (then its SNR)')
    ap.add_argument('--scenario', choices=sorted(SCENARIOS), default=None,
                    help='a Phase 3b setting of scripts/sensor/scenarios_3b.py (sequential mode)')
    ap.add_argument('--scenario_consts', default=None,
                    help=f'scenario constants (make_scenario_consts.py); default {SCENARIO_CONSTS}, read only by '
                         'poisson / saturation / quantisation')
    ap.add_argument('--subset', choices=['gate', 'ddpm'], default='gate')
    ap.add_argument('--methods', nargs='+', choices=METHODS, default=None)
    ap.add_argument('--regress', default=None); ap.add_argument('--varhead', default=None)
    ap.add_argument('--cond', default=None); ap.add_argument('--uncond', default=None)
    ap.add_argument('--K', type=int, default=32); ap.add_argument('--pigdm_steps', type=int, default=50)
    ap.add_argument('--guidance', choices=['pigdm', 'dps'], default='pigdm')
    ap.add_argument('--b_kernel', default=None, help='B+ kernel file (fit_b_kernel.py); enables the method Bp')
    ap.add_argument('--save_draws', default=None, help="directory for B's raw guided draws (<name>.npz, float16)")
    ap.add_argument('--sensor_config', default=CONFIG_PATH)
    ap.add_argument('--batch_size', type=int, default=4); ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--limit_files', type=int, default=0)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--data_root', default=DATA_ROOT); ap.add_argument('--prior', default=PRIOR)
    return ap


def main():
    args = get_parser().parse_args()
    out = output_path(args.name)
    if os.path.exists(out):
        raise FileExistsError(f'{out} exists; pick another --name')
    scen = apply_scenario_args(args)
    cfg = load_config(args.sensor_config)
    args.methods = resolve_methods(args)
    device = torch.device(args.device)
    models, prov = check_models(args, cfg, device)
    if scen is None:
        ds = HFDSensorSim(args.data_root, 'test', cfg, device=args.sensor_device,
                          setting=dict(mode=args.mode, snr_db=args.snr_db), prior_path=args.prior)
    else:
        ds = HFDSensorSim(args.data_root, 'test', cfg, device=args.sensor_device,
                          setting=dict(mode=args.mode, snr_db=args.snr_db), prior_path=args.prior, scenario=scen,
                          scenario_consts=load_consts(args.scenario_consts) if args.scenario_consts else None)
    files = load_subset(args.data_root, 'files' if args.subset == 'gate' else 'ddpm1000_subset')
    have = {os.path.abspath(f) for f in ds.img_list}
    missing = [f for f in files if os.path.abspath(f) not in have]
    if missing:
        raise ValueError(f'{len(missing)} of {len(files)} subset files are not in the test split, e.g. {missing[0]}')
    if args.limit_files:
        files = files[:args.limit_files]
    ds.img_list = list(files)
    # make_eval_loader hard-codes residual_collate_fn: the same loader with the dataset's dict collate_fn
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                         collate_fn=ds.collate_fn, worker_init_fn=limit_worker_threads)
    rec = run_eval(models, ds, loader, args, cfg, prov, device)
    write_record(out, rec)
    print('wrote', out)
    for m, d in rec['means'].items():
        print(f'  {m:10s} ' + ' '.join(f'{k}={v:.4f}' for k, v in d.items()
                                       if k in ('rmse_pct', 'mean_rmse_pct', 'crps', 'cov90', 'ause')))


if __name__ == '__main__':
    main()
