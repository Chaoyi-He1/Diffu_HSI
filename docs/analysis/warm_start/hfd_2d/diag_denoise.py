"""Diagnostic (GPU 0): how well does the checkpoint denoise, and does it use y?
For each test item x (ground truth, used here ONLY to make x_t for the diagnostic) and t:
  x_t = sqrt(ab_t) x + sqrt(1-ab_t) eps;  eps_hat = model(x_t, y, t);
  report eps-MSE (the training loss at that t) and the one-step x0 estimate
  x0_hat = (x_t - sqrt(1-ab) eps_hat)/sqrt(ab) RMSE on [0,1] (%), with the true y,
  with y = 0, and with y from a different item (shuffled) -- if the last two match the
  first, the network is not using the sensor values at that noise level.
Also: the noise-only floor (RMSE of x_t/sqrt(ab) itself) and the average eps-MSE over uniform t.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
torch.set_num_threads(8)
dev = pick_device()
model, betas, meta = load_model(dev, sdpa=True)
ab = torch.cumprod(1 - betas, 0)
tr, ev = make_datasets()
ts_ = json.load(open(os.path.join(OUT, 'testsets.json')))
idx = {p: (ev, i) for i, p in enumerate(ev.img_list)}
idx.update({p: (tr, i) for i, p in enumerate(tr.img_list)})
res = {}
for setname in ('primary', 'secondary'):
    X, Y = [], []
    for p in ts_[setname]:
        ds, i = idx[p]; g, s = ds[i]
        X.append(torch.tensor(g, dtype=torch.float32).permute(2, 0, 1)); Y.append(torch.tensor(s, dtype=torch.float32).permute(2, 0, 1))
    X = torch.stack(X); Y = torch.stack(Y)
    out = {}
    gen = torch.Generator(device=dev).manual_seed(123)
    for t in [999, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 20, 5]:
        acc = {k: [] for k in ('eps_mse', 'x0_rmse', 'x0_rmse_y0', 'x0_rmse_yshuf', 'xt_rmse', 'eps_mse_y0')}
        for b in range(0, len(X), 8):
            x = X[b:b + 8].to(dev); y = Y[b:b + 8].to(dev)
            e = torch.randn(x.shape, generator=gen, device=dev)
            a = float(ab[t])
            xt = math.sqrt(a) * x + math.sqrt(1 - a) * e
            tt = torch.full((x.shape[0],), t, device=dev, dtype=torch.long)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                eh = model(xt, y, tt).float()
                eh0 = model(xt, torch.zeros_like(y), tt).float()
                ehs = model(xt, y.roll(1, 0), tt).float()
            f = lambda E: metrics((xt - math.sqrt(1 - a) * E) / math.sqrt(a), x)[0]
            acc['eps_mse'] += ((eh - e) ** 2).flatten(1).mean(1).tolist()
            acc['eps_mse_y0'] += ((eh0 - e) ** 2).flatten(1).mean(1).tolist()
            acc['x0_rmse'] += f(eh).tolist(); acc['x0_rmse_y0'] += f(eh0).tolist(); acc['x0_rmse_yshuf'] += f(ehs).tolist()
            acc['xt_rmse'] += metrics(xt / math.sqrt(a), x)[0].tolist()
        out[t] = {k: float(np.mean(v)) for k, v in acc.items()}
        print('%-9s t=%4d  eps-MSE %.4f (y=0: %.4f) | one-step x0 RMSE %%: true y %7.3f  y=0 %7.3f  shuffled y %7.3f | x_t/sqrt(ab) %8.3f' % (
            setname, t, out[t]['eps_mse'], out[t]['eps_mse_y0'], out[t]['x0_rmse'], out[t]['x0_rmse_y0'],
            out[t]['x0_rmse_yshuf'], out[t]['xt_rmse']), flush=True)
    # training-loss estimate: uniform t, 40 draws per item
    ls = []
    for rep in range(40):
        for b in range(0, len(X), 8):
            x = X[b:b + 8].to(dev); y = Y[b:b + 8].to(dev)
            tt = torch.randint(0, 1000, (x.shape[0],), generator=gen, device=dev)
            e = torch.randn(x.shape, generator=gen, device=dev)
            a = ab.to(dev)[tt].float().view(-1, 1, 1, 1)
            xt = a.sqrt() * x + (1 - a).sqrt() * e
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                eh = model(xt, y, tt).float()
            ls.append(((eh - e) ** 2).mean().item())
    out['uniform_t_eps_mse'] = float(np.mean(ls))
    print('%s: eps-MSE averaged over uniform t = %.4f (checkpoint training loss 0.0355)' % (setname, out['uniform_t_eps_mse']), flush=True)
    res[setname] = out
json.dump(res, open(os.path.join(OUT, 'diag_denoise.json'), 'w'), indent=1)
