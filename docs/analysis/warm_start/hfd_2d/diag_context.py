"""Diagnostic (CPU, 8 threads): why does the network ignore y?
The sensor values enter only as keys/values of cross-attention, as tokens
context_encoder(y)[pixel] + t_emb.  Compare the size of the y-dependent part
(and its variation across pixels / across items) with the time embedding."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
torch.set_num_threads(8)
model, betas, meta = load_model(torch.device('cpu'), sdpa=True)
tr, ev = make_datasets()
ts_ = json.load(open(os.path.join(OUT, 'testsets.json')))
idx = {p: (ev, i) for i, p in enumerate(ev.img_list)}
Y = torch.stack([torch.tensor(idx[p][0][idx[p][1]][1], dtype=torch.float32).permute(2, 0, 1) for p in ts_['primary'][:4]])
with torch.no_grad():
    ce = model.context_encoder(Y)                     # [4, 1024, 64, 64]
    tok = ce.flatten(2).permute(0, 2, 1)              # [4, 4096, 1024]
    print('y range %.3f..%.3f' % (Y.min(), Y.max()))
    print('context token norm: mean %.3f' % tok.norm(dim=-1).mean())
    print('  std across pixels within an item (norm of per-pixel deviation): %.3f' % (tok - tok.mean(1, keepdim=True)).norm(dim=-1).mean())
    print('  difference between item means (norm): %.3f' % (tok.mean(1) - tok.mean(1).mean(0)).norm(dim=-1).mean())
    for t in (999, 500, 200, 50, 10):
        te = model.time_embedding(torch.tensor([t]))
        print('t=%4d  |t_emb| = %.3f' % (t, te.norm()))
