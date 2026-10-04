# Why conditional diffusion recovers a hyperspectral cube from 30 sensor channels — and why the spectral axis must be treated as channels, not convolved

*Status: working note, 2026-10-03. Every number quoted below is produced by `docs/analysis/sensor_identifiability.py`, which uses the exact preprocessing of `data_loader/HFD_dataset.py` and `data_loader/my_dataset.py`.*

---

## 0. Setting, as implemented in this repository

| symbol | meaning | HFD100 | HASCID |
|---|---|---|---|
| $L$ | spectral bands of the ground truth $x$ | 31 (451–855 nm), linearly interpolated to 64 for the network | 204 measured, first 160 reconstructed |
| $N$ | sensor channels of the measurement $y$ | 30 | 30 |
| $R\in\mathbb R^{L\times N}$ | sensor response (30 columns of `R_Device1/2.mat` or `PH5_interp_results.mat`, resampled to the data wavelengths, each column min-max normalised) | | |
| $P=HW$ | pixels per cube | $64\times 64$ | $512\times512$ |
| $d$ | spatial down-sampling of the sensor grid (`--ds`) | 1 or 2 | 1 |

The forward model is **per pixel, linear, and wavelength-indexed**:

$$
y_p \;=\; R^{\mathsf T} x_p \in\mathbb R^{N},\qquad p=1,\dots,P,
\qquad\text{i.e.}\qquad Y = R^{\mathsf T} X,\; X\in\mathbb R^{L\times P}. \tag{0.1}
$$

With `--ds d` only the pixels on a stride-$d$ grid are measured: $Y_d = (R^{\mathsf T}X)\,S_d$ where $S_d$ selects every $d$-th row and column.

The network is trained in data space (no VAE) by DDPM $\varepsilon$-prediction (`DiffusionTrainer`, linear $\beta_t\in[10^{-4},0.02]$, $T=1000$) with an $L_1$ loss, conditioned on $Y$ through cross-attention (`U2NetHyperspectral`). The question this note answers is: **why is $x$ recoverable from $y$ at all, why does diffusion recover it, and why must the architecture treat $\lambda$ as a channel axis (2-D spatial + dense 1-D spectral) rather than as a third convolution axis (3-D)?**

---

## 1. The inverse problem: nominally almost square, effectively very under-determined

### 1.1 Nominal structure

$R^{\mathsf T}:\mathbb R^{L}\to\mathbb R^{N}$ has a null space of dimension $\ge L-N$. For HFD ($L=31,N=30$) that is a *single* direction; for HASCID ($L=160$) it is $\ge130$ directions. Nominally HFD looks almost well-posed. It is not:

### 1.2 Conditioning (measured)

Singular values of $R^{\mathsf T}$ after the loader's preprocessing:

| sensor | rank | $\kappa=\sigma_1/\sigma_{\min}$ | #$\{\sigma_i/\sigma_1>10^{-2}\}$ | #$\{>10^{-3}\}$ | #$\{>5\cdot10^{-4}\}$ (fp16 $\epsilon$) |
|---|---|---|---|---|---|
| `R_Device1`, HFD (30×31) | 30 | $2.3\times10^{9}$ | **11** | 17 | 19 |
| `R_Device2`, HFD (30×31) | 30 | $5.1\times10^{7}$ | **9** | 16 | 18 |
| `PH5`, HFD (30×31) | 16 | $2.9\times10^{19}$ | **4** | 6 | 7 |
| `R_Device1`, HASCID, 160 target bands (30×160) | 30 | $1.5\times10^{7}$ | **13** | 18 | 21 |

The singular values of `R_Device1` fall from 15.3 to $7\times10^{-9}$. Under *any* perturbation of $y$ — read-out noise, the fp16 autocast the network runs in ($\epsilon_{\text{fp16}}\approx5\times10^{-4}$), even float32 round-off relative to $\kappa$ — only about **11–19 of the 30 channels carry usable information**. The remaining directions of $x$ have to come from somewhere else. Concretely, for `R_Device1` on sampled HFD pixels (relative $\ell_2$ error per pixel):

| estimator | noiseless | SNR 60 dB | SNR 40 dB | SNR 30 dB |
|---|---|---|---|---|
| pseudo-inverse $(R^{\mathsf T})^{+}y$ | 0.017 | $2.7\times10^{5}$ | $2.7\times10^{6}$ | $8.5\times10^{6}$ |
| Gaussian-prior MMSE $\mu+CR(R^{\mathsf T}CR+\sigma^2 I)^{-1}(y-R^{\mathsf T}\mu)$ | — | **0.049** | **0.088** | **0.139** |

Direct inversion is useless at any realistic precision; a prior is not optional. The Gaussian-prior row is the *linear* Bayesian baseline — the floor that any learned method must beat, and a useful sanity check for the diffusion model (§6).

### 1.3 The prior is low-dimensional (measured)

PCA of per-image-normalised spectra:

| data | $k$ for 99 % variance | 99.9 % | 99.99 % |
|---|---|---|---|
| HFD (31 bands, 120 000 pixels) | **8** | 19 | 29 |
| HASCID (160 bands, 60 000 pixels) | **3** | 7 | 61 |

and the measurement sees the part that matters: for `R_Device1` on HFD, the 11 well-conditioned directions of $R^{\mathsf T}$ contain **98.0 %** of the spectral variance (17 directions: 99.4 %); on HASCID the 13 well-conditioned directions contain 99.8 %. For `PH5` the 4 usable directions contain only 61 % — which predicts, before training anything, that PH5 reconstructions will be markedly worse than `R_Device1/2`.

---

## 2. Identifiability: on the spectral manifold, $y$ determines $x$

Let $\mathcal M\subset\mathbb R^{L}$ be the set of spectra that occur (a compact set of intrinsic dimension $k\approx 8$ for HFD, $3$–$7$ for HASCID, by §1.3), and $A=R^{\mathsf T}$.

**Proposition 1 (generic injectivity).** If $N>2k$, then for Lebesgue-almost-every linear $A:\mathbb R^{L}\to\mathbb R^{N}$ the restriction $A|_{\mathcal M}$ is injective.
*Proof.* This is the embedology theorem of Sauer, Yorke & Casdagli (1991), the measure-theoretic version of Whitney's embedding theorem: a compact set of (box-counting) dimension $k$ is injectively mapped by almost every projection to $\mathbb R^{N}$ as soon as $N>2k$. $\square$

Here $N=30>2\cdot 8=16$ (HFD) and $30>2\cdot7=14$ (HASCID). The theorem is about *generic* $A$; for the *actual* $A$ the relevant certificate is the **restricted conditioning** on the principal subspace $V_k$ that carries the manifold,

$$
\rho_k \;=\; \frac{\sigma_{\min}(R^{\mathsf T}V_k)}{\sigma_{\max}(R^{\mathsf T}V_k)} ,
$$

because $\rho_k>0$ means $A$ is bi-Lipschitz on $\operatorname{span}V_k\supset\mathcal M$ (up to the 1 % of energy outside it), which implies injectivity on $\mathcal M$ with a quantitative inverse-Lipschitz constant $1/\rho_k$:

| $k$ | `R_Device1` HFD | `R_Device2` HFD | `PH5` HFD | `R_Device1` HASCID |
|---|---|---|---|---|
| 5 | $4.1\times10^{-2}$ | $3.2\times10^{-2}$ | $7.0\times10^{-4}$ | $6.6\times10^{-2}$ |
| 8 | $3.2\times10^{-2}$ | $1.2\times10^{-2}$ | $1.2\times10^{-4}$ | $3.1\times10^{-2}$ |
| 12 | $2.7\times10^{-3}$ | $1.3\times10^{-3}$ | $1.6\times10^{-5}$ | $9.1\times10^{-3}$ |

**Corollary 2.** Under a prior supported on $\mathcal M$, the noiseless posterior $p(x\mid y)\propto p(x)\,\delta(y-R^{\mathsf T}x)$ is supported on $\mathcal M\cap(x^\star+\ker R^{\mathsf T})$, which by Proposition 1 is the single point $x^\star$: the posterior is a point mass and $x=g(y)$ for a (nonlinear) inverse $g$ defined on $A(\mathcal M)$. With noise or finite precision, the posterior has spread of order $\sigma/\rho_k$ along the $k$ manifold directions and *no* spread elsewhere — in contrast to the linear-Gaussian posterior of §1.2, which spreads over all $L-11$ poorly measured directions.

That is the whole reason reconstruction is possible: **the 30 channels do not determine the 31 (or 160) numbers, but they determine the 8 (or 7) numbers that the spectra actually have.** A method succeeds exactly to the degree that it (i) knows $\mathcal M$ and (ii) can apply the dense, wavelength-specific operator that maps $y$ onto it.

---

## 3. Why *diffusion* computes this posterior

Notation as in `model/diffusion_trainer.py`: $\bar\alpha_t=\prod_{s\le t}(1-\beta_s)$, $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon$, $\varepsilon\sim\mathcal N(0,I)$, $t\sim\mathcal U\{0,\dots,T-1\}$.

**Proposition 3 (the trained network is the conditional score).** The minimiser of the conditional $\varepsilon$-objective
$$
\mathcal L(\theta)=\mathbb E_{t,x_0,y,\varepsilon}\big\|\varepsilon-\varepsilon_\theta(x_t,y,t)\big\|_2^2
$$
over all measurable functions is $\varepsilon^\star(x_t,y,t)=\mathbb E[\varepsilon\mid x_t,y]$, and
$$
\nabla_{x_t}\log p_t(x_t\mid y)\;=\;-\frac{\varepsilon^\star(x_t,y,t)}{\sqrt{1-\bar\alpha_t}} . \tag{3.1}
$$
*Proof.* The first statement is the $L_2$ projection property of conditional expectation. For (3.1), $p_t(x_t\mid y)=\int\mathcal N(x_t;\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)\,p(x_0\mid y)\,dx_0$; differentiating under the integral,
$\nabla_{x_t}\log p_t(x_t|y)=-\mathbb E\!\left[\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}\,\Big|\,x_t,y\right]=-\frac{\mathbb E[\varepsilon|x_t,y]}{\sqrt{1-\bar\alpha_t}}$ (Tweedie / Robbins; Efron 2011). $\square$

*Remark ($L_1$ loss).* The repository minimises $\mathbb E|\varepsilon-\varepsilon_\theta|$, whose minimiser is the coordinate-wise conditional *median* of $\varepsilon$. It coincides with the mean whenever the conditional law of $\varepsilon$ given $(x_t,y)$ is symmetric — in particular at small $t$, where the posterior is unimodal and nearly Gaussian — and is a common robustness choice, but strictly speaking the sampler below assumes the mean. This is a modelling choice, not a flaw in the argument; §6 lists it as something to ablate.

**Proposition 4 (sampling the posterior).** Let $s_\theta(x_t,y,t)\approx\nabla\log p_t(x_t|y)$ with $\mathbb E_{t,x_t}\|s_\theta-\nabla\log p_t\|^2\le\varepsilon_{\text{score}}^2$. The reverse-time process with drift built from $s_\theta$ (Anderson 1982; Song et al. 2021), discretised as the DDPM ancestral sampler implemented in `DiffusionTrainer.sample`, produces a distribution $\hat p$ with
$$
\mathrm{TV}\big(\hat p,\;p(\cdot\mid y)\big)\;\le\;C\big(\varepsilon_{\text{score}}\sqrt{T}+\text{discretisation}+e^{-T}\big)
$$
for **any** conditional data distribution with finite second moment — no log-concavity, smoothness, or full support is required (Chen, Lee, Lu et al., ICLR 2023, "Sampling is as easy as learning the score"; the manifold-supported case is covered by their early-stopping variant).

**Theorem 5 (reconstruction).** Combining Corollary 2 with Propositions 3–4: a conditional diffusion model trained to small $\varepsilon$-loss on $(x,y=R^{\mathsf T}x)$ pairs and sampled with the DDPM/DDIM reverse process returns samples concentrated at $g(y)=x^\star$, with error controlled by (a) the prior spread allowed by $\rho_k$ and the measurement precision, and (b) the score approximation and discretisation error. In the noiseless, perfectly trained limit the sample *is* $x^\star$. $\square$

### 3.1 What the score must contain (this drives the architecture question)

By Bayes,
$$
\nabla_{x_t}\log p_t(x_t\mid y)\;=\;\underbrace{\nabla_{x_t}\log p_t(x_t)}_{\text{prior: spatial }\times\text{ spectral}}\;+\;\underbrace{\nabla_{x_t}\log p_t(y\mid x_t)}_{\text{likelihood}} . \tag{3.2}
$$
Because (0.1) is pixel-wise, $p(y|x_0)=\prod_p p(y_p|x_{0,p})$ and the likelihood term is **per pixel and purely spectral**. Under the usual Gaussian approximation of $p(x_0\mid x_t)$ (Song et al. 2023, ΠGDM; Chung et al. 2023, DPS),
$$
\nabla_{x_t}\log p_t(y_p\mid x_t)\;\approx\;\sqrt{\bar\alpha_t}\;J_p^{\mathsf T}\,R\,\big(R^{\mathsf T}\Sigma_t R+\sigma^2I\big)^{-1}\big(y_p-R^{\mathsf T}\hat x_{0,p}(x_t)\big), \tag{3.3}
$$
a **dense $L\times N$ wavelength-indexed linear map** applied at every pixel. The conditional network does not need $R$ at test time — it learns (3.3) *amortised* from data, which is also why it tolerates calibration error in $R$ — but it must be *able to represent* an arbitrary dense map between the sensor index $j$ and the wavelength index $\lambda$. Section 4 shows that a 3-D convolutional network cannot do so without contradiction, and Section 5 that the 2-D-spatial + channel-spectral network does so in one layer.

### 3.2 Why not a plain regressor $y\mapsto x$?

A regressor trained with $L_2$ learns $\mathbb E[x\mid y]$. When the posterior is multimodal (folds of $\mathcal M$ seen through $A$, or $d>1$ where unmeasured pixels have genuinely several plausible spectra) the mean lies *off* $\mathcal M$ — a blurred, physically impossible spectrum. Diffusion samples the posterior and so stays on $\mathcal M$; in the identifiable regime of Theorem 5 both agree, but diffusion additionally regularises the ill-conditioned directions of §1.2 adaptively (through the learned prior score) instead of with a fixed ridge, and multiple samples give a calibrated uncertainty for free.

---

## 4. Why a 3-D (λ, h, w)-convolutional diffusion model is the wrong hypothesis class

"3-D diffusion" here means: treat the cube as a one-channel volume $[1,L,H,W]$ and build $\varepsilon_\theta$ from 3-D convolutions whose kernels are shared along $\lambda$ exactly as along $h,w$.

### 4.1 Lemma (a 3-D convolution is a Toeplitz-constrained 2-D convolution)

Let a 3-D convolution with $C_{\text{in}}\to C_{\text{out}}$ channels and kernel $w[c',c,\delta_\lambda,\delta_h,\delta_w]$ act on $u[c,\lambda,h,w]$. Reshape $u$ to a 2-D feature map with $C_{\text{in}}L$ channels indexed by $(c,\lambda)$. Then the 3-D convolution equals a 2-D convolution with $C_{\text{in}}L\to C_{\text{out}}L$ channels whose channel-mixing matrix at each spatial offset $(\delta_h,\delta_w)$ is
$$
W_{\delta_h\delta_w}\big[(c',\lambda'),(c,\lambda)\big]\;=\;w[c',c,\lambda'-\lambda,\delta_h,\delta_w]\cdot\mathbf 1_{|\lambda'-\lambda|<k_\lambda}, \tag{4.1}
$$
i.e. **block-Toeplitz and banded along $\lambda$**. *Proof:* write out the sum defining the 3-D convolution and regroup the $\lambda$ index into the channel index. $\square$

Consequently
$$
\mathcal F_{\text{3D}}\;\subsetneq\;\mathcal F_{\text{2D, }\lambda\text{ as channels}}
$$
with identical spatial kernels: everything a 3-D network can compute, the 2-D-with-spectral-channels network (this repository's `U2NetHyperspectral`) can compute, and the latter additionally realises **any** dense channel-mixing matrix — in particular $R$, $(R^{\mathsf T})^{+}$, the projector onto $V_k$, and the map (3.3) — in a single $1\times1$ layer. Nothing is lost by abandoning 3-D convolution; only a constraint is removed. The question is whether that constraint is *true* for this problem.

### 4.2 Proposition (the constraint is false: the problem is not shift-invariant along λ)

A stack of 3-D convolutions, point-wise nonlinearities and shared-weight pooling/up-sampling along $\lambda$ (with circular or infinite padding) is **equivariant to spectral translations** $S_\lambda$: $F(S_\lambda u)=S_\lambda F(u)$ (Cohen & Welling 2016; for strided pooling, to translations by multiples of the stride). By (3.1) the target of training is the Bayes denoiser $D^\star(x_t,y,t)=\mathbb E[x_0\mid x_t,y]$. For an equivariant class, the best achievable approximation of $D^\star$ is its Reynolds (shift-)average $\Pi D^\star$, and the **irreducible error** is $\|D^\star-\Pi D^\star\|$. Three independent facts make it large:

1. **The likelihood is not shift-covariant.** For $F$ to be jointly equivariant in $(x,y)$ there must be a shift $S'$ on the sensor index with $R^{\mathsf T}S_\lambda=S'R^{\mathsf T}$, i.e. $R^{\mathsf T}$ must be (block-)Toeplitz. The Reynolds average of a matrix over simultaneous row/column shifts is exactly its projection onto Toeplitz matrices, so the irreducible *relative* error of the best shift-equivariant linear approximation of the measurement operator is $\|R^{\mathsf T}-\Pi_{\text{Toep}}R^{\mathsf T}\|_F/\|R^{\mathsf T}\|_F$, measured:

   | `R_Device1` HFD | `R_Device2` HFD | `PH5` HFD | `R_Device1` HASCID | i.i.d. Gaussian (reference) |
   |---|---|---|---|---|
   | **0.59** | 0.26 | 0.49 | **0.60** | 0.97 |

   For the device actually used, a spectrally shift-equivariant model mis-represents the physics by 59 % before any learning happens.

2. **The prior is not stationary along λ.** The spectral covariance of HFD spectra is 0.44 (relative Frobenius) away from the nearest Toeplitz matrix (HASCID: 0.12); the mean spectrum has its steepest slope at 761 nm (the vegetation red edge) and its variance peak at 572 nm. Absolute wavelength carries meaning; a shift-equivariant prior cannot tell 572 nm from 761 nm.

3. **The condition cannot even be placed on the λ-axis.** $y_p\in\mathbb R^{30}$ while $x_p\in\mathbb R^{31}$, $\mathbb R^{64}$ or $\mathbb R^{160}$; the 30 sensor channels are a uniformly thinned list of broadband filters with no metric on their index $j$. Any lifting $y\mapsto$ volume requires a learned dense $N\to L$ map — which *is* a channel-mixing $1\times1$ operator, i.e. the 2-D ingredient. A "3-D" conditional model is therefore either ill-defined or already a 2-D-channel model in disguise.

Together: $\|D^\star-F\|\ge\|D^\star-\Pi D^\star\|>0$ for every spectrally equivariant $F$. The Toeplitz distances above are the exact value of this irreducible error for the *linear* part of the map ($R^{\mathsf T}$ itself, and the linear-Gaussian denoiser built from $C$ and $R$); for the full nonlinear $D^\star$ they quantify the obstruction rather than bound it tightly. $\square$

*Remark (what a real 3-D net can and cannot do about it).* Zero padding leaks absolute position near the boundaries (Islam, Jia & Bruce, ICLR 2020; Kayhan & van Gemert, CVPR 2020), so a deep 3-D network is only *approximately* equivariant and could in principle infer wavelength identity from distance to the spectral border, through many layers, without guarantee. Adding explicit $\lambda$-positional encodings removes the equivariance — and concedes the point, since the spectral axis is then being treated as a labelled channel axis at $L$-fold cost. The honest statement of this section is therefore: *the inductive bias of 3-D convolution is the negation of the structure that makes the problem solvable (§2–3.1); whatever a 3-D model achieves, it achieves by undoing that bias.*

### 4.3 Cost (not the mathematical core, but decisive in practice)

Activations of a 3-D model are $L$ times those of the 2-D model at equal width: the first stage of `U2NetHyperspectral` at `base_channels=128` holds $128\times64\times64$ values per sample; the 3-D analogue holds $128\times64\times64\times64$ — $64\times$ more. The 2-D model already uses 16 GB at batch 3 on this machine. FLOPs scale by a further $k_\lambda$. Dense 30↔64 spectral mixing, which the 2-D model does in one layer with $30\cdot64$ weights, needs $\lceil (L-1)/(k_\lambda-1)\rceil$ stacked 3-D layers merely to connect the first and last band.

---

## 5. Why the 2-D (spatial) + 1-D (spectral) factorisation works

The two axes of the cube have **opposite symmetry properties with respect to the measurement operator**, and the factorised architecture assigns to each the inductive bias that is actually true:

| axis | does $R^{\mathsf T}$ commute with translation? | is the prior stationary? | right operator | used in this repo |
|---|---|---|---|---|
| spatial $(h,w)$ | **yes, exactly**: $R^{\mathsf T}(XS_{h})=(R^{\mathsf T}X)S_{h}$ because (0.1) is pixel-wise | approximately (lag-1 autocorrelation 0.966, lag-2 0.921) | shared-weight 2-D convolution + attention | spatial U-Net |
| spectral $\lambda$ | **no** (59 % off Toeplitz) | no (44 % off Toeplitz) | dense, position-aware mixing | spectral bands as channels; 1×1 convs, attention; 1-D model per pixel |

### 5.1 The 1-D (per-pixel) model is already a consistent estimator

Since the likelihood factorises over pixels, the posterior with the *marginal* spectral prior, $p(x_p\mid y_p)\propto p(x_p)\,\delta(y_p-R^{\mathsf T}x_p)$, is well defined and — by Corollary 2 — a point mass at the true spectrum when $d=1$. The 1-D model (`main_1d.py`, `U2Net1D`, condition injected by cross-attention) learns exactly this posterior, and it does so from every pixel as an independent sample: $40{,}811\times4{,}096\approx1.7\times10^{8}$ spectra in 31 dimensions, instead of $4\times10^{4}$ cubes in $1.3\times10^{5}$ dimensions. Nonparametric density-estimation rates scale with the intrinsic dimension, $n^{-O(1/k)}$, so this is the statistically cheapest route to learning $\mathcal M$. It ignores the spatial prior, which costs it only the residual spread of §2 — and it cannot help at all when $d>1$.

### 5.2 The 2-D model adds the spatial prior where it is needed

With `--ds 2`, three quarters of the pixels have **no** measurement; per-pixel identifiability fails for them and the spatial prior term of (3.2) must supply their spectra from their neighbours. The spatial prior is rich: bilinear interpolation of a stride-2 cube is already within **6.4 %** relative error of the original, and the model's cross-attention to the bilinearly resized context (`create_context_for_resolution`) is precisely a learned, spectrum-aware version of that fill-in. Meanwhile the spectral axis remains a dense channel operator, so every pixel that *is* measured still gets the exact operator (3.3). The 2-D model is thus the minimal architecture that is correct for both $d=1$ and $d>1$.

### 5.3 Summary of the three options

| | 3-D conv over $(\lambda,h,w)$ | 1-D per pixel | 2-D spatial + spectral channels |
|---|---|---|---|
| can represent the dense $y\!\to\!x$ spectral operator (3.3)? | no (Toeplitz-constrained, Lemma 4.1) | yes (attention/dense) | yes (1×1 conv, one layer) |
| imposes a false symmetry? | yes, along $\lambda$ | no | no |
| uses the true spatial symmetry? | yes | no | yes |
| handles `--ds` $>1$? | — | no | yes |
| activation memory | $\times L$ | tiny | baseline |

---

## 6. Current results (as of 2026-10-03)

All 2-D runs: HFD, `R_Device1` (`--R-n 1`), 64 interpolated bands, `base_channels 128`, batch 3, AdamW lr $10^{-4}$ cosine to $10^{-5}$, AMP, $\varepsilon$-prediction, $L_1$ loss, $T=1000$. The reported metric is the training $\varepsilon$-$L_1$ averaged over uniformly random $t$. The trivial predictor $\varepsilon_\theta\equiv0$ scores $\mathbb E|\varepsilon|=\sqrt{2/\pi}=0.798$; a first-epoch average of 0.69–0.71 in the from-scratch logs is consistent with that starting point.

| run (log) | data | sensor grid | epochs in log | $\varepsilon$-$L_1$ start → end | notes |
|---|---|---|---|---|---|
| `log.txt` | full 36 730 cubes, lr $5\cdot10^{-5}$ | $d=1$ | 87 | 0.689 → 0.255 | crashed (CUBLAS error) |
| `log1.txt` | full 36 730 cubes | $d=1$ | 208 | 0.297 → 0.153 | warm start from `log.txt` lineage |
| `log2.txt` | 9 000-cube subset | $d=1$ | 910 | 0.153 → **0.151** (val 0.151 at launch) | plateau from epoch 1; ended on a failed checkpoint write |
| `log_down2.txt` | 9 000-cube subset | $d=2$ | 804 | 0.708 → 0.170 | crashed (CUBLAS error) |
| `log_down2_1.txt` | 9 000-cube subset | $d=2$ | 793 | 0.171 → **0.162** | warm start from the above |
| 1-D, HASCID, `PH5` | per pixel, 160 bands | $d=1$ | — | 0.031 → **0.026** | `results/1d_hsi_diffusion/HASCID/PH5/train_history.txt` |
| VAE, HASCID (160→12 latent) | — | — | — | best val loss 0.041 | `results/vae_hyperspectral/base_128_latent_12/` |

**Reading these numbers through §2–3.** For a perfectly identifiable problem the $\varepsilon$-loss can in principle reach zero at *every* $t$: once $x_0=g(y)$ is known, $\varepsilon=(x_t-\sqrt{\bar\alpha_t}\,g(y))/\sqrt{1-\bar\alpha_t}$ is exactly computable. A floor of 0.151 therefore reflects the sum of (a) posterior spread that the measurement precision and $\rho_k$ leave, (b) the 1 % of spectral energy outside the well-measured subspace, and (c) model capacity/optimisation. The $d=2$ floor is higher (0.162 vs 0.151), as §5.2 predicts: three quarters of the pixels are reconstructed from the spatial prior alone. The $d=1$ model made no progress over 900 epochs on the 9 000-cube subset; that is a capacity/optimisation plateau, not a data-limited one, and is the first thing to investigate (see below). The 1-D HASCID floor (0.026) is far lower, consistent with a $k\approx3$–7 manifold seen by 13 well-conditioned channels (§1.3) and $10^{8}$ training spectra.

**What is missing.** No reconstruction metric (PSNR, SAM, relative RMSE on held-out cubes) has been computed yet: `results/2d_visualization/.../l1_loss` is empty, and both surviving $L_1$ checkpoints are unusable — `checkpoint_epoch_911.pth` is a truncated file from a failed save, `checkpoint_epoch_811.pth` is 0 bytes. Until retrained, the only quantitative evidence for the 2-D model is the loss table above.

**Next steps that follow from this note.**
1. Retrain with atomic checkpointing (save to a temporary name, then `os.replace`) and evaluate PSNR/SAM on the test split against the **linear Gaussian-prior MMSE baseline of §1.2** (0.05–0.14 relative error depending on assumed SNR) — a free, strong baseline that the diffusion model must beat to justify itself.
2. Reconstruct 31 bands and interpolate to 64 afterwards: the 31→64 map has rank 31, so the 64-band target lives on a 31-dimensional subspace that the network currently has to learn (`expand_wavelens` in `HFD_dataset.py`).
3. Fix two code issues that bias the numbers: gradient clipping is applied to AMP-*scaled* gradients (`clip_grad_norm_` without `scaler.unscale_`), and `DiffusionTrainer.sample(n_steps=50)` starts the reverse process at $t=49$ from pure noise instead of striding 1000 steps (`train_eval/train_2d.py` `generate_samples`).
4. Ablate $L_1$ vs $L_2$ $\varepsilon$-loss (Remark after Proposition 3) and `R_Device1/2` vs `PH5` — §1–2 predict the ordering Device1 ≳ Device2 ≫ PH5.
5. For HASCID note that 10.7 % of the measurement energy comes from bands 161–204 that are not part of the 160-band target; either reconstruct all 204 bands or compute $y$ from the first 160 only, otherwise those bands act as structured noise in $y$.

---

## References

- Anderson, B. D. O. (1982). Reverse-time diffusion equation models. *Stoch. Proc. Appl.*
- Chen, S., Chewi, S., Li, J., Li, Y., Salim, A., Zhang, A. (2023). Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions. *ICLR*.
- Chung, H., Kim, J., Mccann, M. T., Klasky, M. L., Ye, J. C. (2023). Diffusion posterior sampling for general noisy inverse problems. *ICLR*.
- Cohen, T., Welling, M. (2016). Group equivariant convolutional networks. *ICML*.
- Efron, B. (2011). Tweedie's formula and selection bias. *JASA*.
- Ho, J., Jain, A., Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS*.
- Islam, M. A., Jia, S., Bruce, N. D. B. (2020). How much position information do convolutional neural networks encode? *ICLR*.
- Kayhan, O. S., van Gemert, J. C. (2020). On translation invariance in CNNs: convolutional layers can exploit absolute spatial location. *CVPR*.
- Sauer, T., Yorke, J. A., Casdagli, M. (1991). Embedology. *J. Stat. Phys.*
- Song, J., Vahdat, A., Mardani, M., Kautz, J. (2023). Pseudoinverse-guided diffusion models for inverse problems. *ICLR*.
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., Poole, B. (2021). Score-based generative modeling through stochastic differential equations. *ICLR*.
- Zheng, Y., Zhang, T., Fu, Y. (2022). A large-scale hyperspectral dataset for flower classification. *Knowledge-Based Systems*.
