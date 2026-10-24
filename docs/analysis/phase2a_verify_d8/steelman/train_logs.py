import json, math
import numpy as np
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/runs/'
for run in ('residual_reg_d8_b64', 'varhead_d8_b64', 'residual_d4_b64', 'residual_reg_d4_b64'):
    try:
        rows = [json.loads(l) for l in open(R + run + '/train_log.jsonl')]
    except FileNotFoundError:
        print(run, 'missing'); continue
    print(run, len(rows), 'rows; keys', sorted(rows[-1].keys()))
    for r in rows:
        e = r.get('epoch')
        if e in (1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100) or 'val_loss' in r and e % 10 == 0:
            print(f"  ep {e:3d} loss {r.get('loss', float('nan')):.5f} val {r.get('val_loss', float('nan')):.5f} lr {r.get('lr', 0):.2e} gn {r.get('grad_norm_mean') or float('nan'):.3f}")
    vals = [(r['epoch'], r['val_loss']) for r in rows if 'val_loss' in r]
    print('  val:', ' '.join(f'{e}:{v:.4f}' for e, v in vals))

# trivial-model reference for the min-SNR-weighted v-loss: v_pred = 0 is the exact denoiser of iid N(0,1) residuals
betas = np.linspace(1e-4, 0.02, 1000)
ab = np.cumprod(1 - betas)
snr = ab / (1 - ab)
w = np.minimum(snr, 5.0) / (snr + 1.0)
print('E_t w(t) =', w.mean(), '; 0.798 * E_t w =', 0.7979 * w.mean())
for t in (0, 49, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999):
    print(f't={t:4d} ab={ab[t]:.4f} snr={snr[t]:.4g} w={w[t]:.4f} sigma_t/alpha_t={math.sqrt((1-ab[t])/ab[t]):.3f}')
print('t where snr crosses 1:', int(np.argmin(np.abs(snr - 1))), ' crosses 5:', int(np.argmin(np.abs(snr - 5))), ' crosses 0.1:', int(np.argmin(np.abs(snr - 0.1))))
