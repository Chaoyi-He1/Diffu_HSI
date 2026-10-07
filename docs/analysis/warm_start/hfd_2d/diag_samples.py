"""Diagnostic (CPU, 8 threads): what do samples from pure noise look like?
Deterministic DDIM (20 steps, eta=0) from the same x_999 ~ N(0,I) for 2 test items, conditioned
on the true y and on the other item's y.  Reports the raw (unclipped) output range, the fraction
of values outside [-1,1], the band-mean of the output vs the ground truth, and how much the
output changes when y is swapped."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
torch.set_num_threads(8)
model, betas, meta = load_model(torch.device('cpu'), sdpa=True)
ab = torch.cumprod(1 - betas, 0).tolist()
tr, ev = make_datasets()
ts_ = json.load(open(os.path.join(OUT, 'testsets.json')))
idx = {p: (ev, i) for i, p in enumerate(ev.img_list)}
items = [ts_['primary'][0], ts_['primary'][5]]
X = torch.stack([torch.tensor(idx[p][0][idx[p][1]][0], dtype=torch.float32).permute(2, 0, 1) for p in items])
Y = torch.stack([torch.tensor(idx[p][0][idx[p][1]][1], dtype=torch.float32).permute(2, 0, 1) for p in items])


def ddim(x, y, S=20):
    ts = [int(round(999 * (1 - i / S))) for i in range(S)]
    for i, t in enumerate(ts):
        with torch.no_grad():
            e = model(x, y, torch.full((x.shape[0],), t, dtype=torch.long))
        x0 = (x - math.sqrt(1 - ab[t]) * e) / math.sqrt(ab[t])
        a = ab[ts[i + 1]] if i + 1 < len(ts) else 1.0
        x = math.sqrt(a) * x0 + math.sqrt(1 - a) * e
    return x


g = torch.Generator().manual_seed(0)
xT = torch.randn(X.shape, generator=g)
out = ddim(xT, Y)
out_sw = ddim(xT, Y.flip(0))
rm, sa, ps = metrics(out, X)
for j, p in enumerate(items):
    o = out[j]
    print('%s: raw output range %.2f..%.2f, |x|>1 fraction %.3f, RMSE %.2f%%, SAM %.2f deg' % (
        os.path.relpath(p, DATA), o.min(), o.max(), (o.abs() > 1).float().mean(), rm[j], sa[j]))
    print('   band means (every 8th band) output:', np.round(o.mean((1, 2))[::8].numpy(), 2).tolist())
    print('   band means (every 8th band) truth :', np.round(X[j].mean((1, 2))[::8].numpy(), 2).tolist())
    print('   swapping y changes the output by RMS %.4f (output RMS %.3f, truth difference between the two items RMS %.3f)' % (
        (out_sw[j] - o).pow(2).mean().sqrt(), o.pow(2).mean().sqrt(), (X[0] - X[1]).pow(2).mean().sqrt()))
