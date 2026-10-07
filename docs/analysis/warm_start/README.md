# Warm-start (truncated) diffusion sampling — experiment scripts

Scripts and reference outputs behind `docs/warm_start_sampling.html`: start the reverse
process at step t0 < 999 from `x_t0 = sqrt(ab[t0]) * x0_hat(y) + sqrt(1 - ab[t0]) * eps`,
where `x0_hat(y)` is a cheap estimate of the cube from the sensor values.

| folder | model | what it does |
|---|---|---|
| `hfd_2d/` | `results/2d_hsi_diffusion/HFD/R_1/l2_loss/checkpoint_epoch_101.pth` (2-D, 64 bands, R_Device1, eps/MSE, epoch 100) | sweep of starting estimates × t0 × DDPM/DDIM on 16 unseen-class patches (P090–P100); `diag_denoise.py` is the y-swap test |
| `hascid_1d/` | `results/1d_hsi_diffusion/HASCID/PH5/final_model.pth` (1-D per pixel, 160 bands, PH5, eps/L1) | the same sweep on 32 test pixels × 2 seeds; CPU |
| `theory/` | — | schedule quantities, start-error bounds, measured estimate errors, t0 rules |

`results.json` and `theory_results.json` are the outputs the page quotes (the `.txt` summaries the
scripts also write are git-ignored).
`hascid_1d/noise_fragility_verifier.json` is the independent verifier's sweep of the
linear estimate against sensor SNR.

## Running

The checkpoints were trained with older code. Unpack the matching snapshots first
(from the repo root):

```bash
export WS_OLDCODE=/tmp/diffu_oldcode
for c in 988a338 50bb95a; do
  mkdir -p $WS_OLDCODE/$c
  git archive $c model data_loader misc train_eval | tar -x -C $WS_OLDCODE/$c
done
export WS_OUT=$PWD/ws_out      # where outputs go (defaults to an out/ folder next to each script)
bash docs/analysis/warm_start/hfd_2d/run_all.sh       # about 3 h on a shared GPU, capped at 2.4 GiB
bash docs/analysis/warm_start/hascid_1d/run_all.sh    # about 7 h on CPU
python docs/analysis/warm_start/theory/theory_table.py
```

Dataset paths point at `/data/chaoyi_he/HSI/Diffu/dataset`; edit `DATA` in each
`common.py` (and in `theory_table.py`) for another machine.
