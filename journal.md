# Discovery Journal

## Shift: 2026-03-21
Goals: Understand VertexModel behavior, fix training instability, tune from untuned baseline.

### Investigation 1: Baseline + standard tuning

**Question**: How does the untuned VertexModel perform, and do standard knobs help?

| Config | PSNR | Notes |
|---|---|---|
| Baseline (exp density, no densify) | 18.43 | 51k verts, ~1850 steps in 10 min |
| + densification 200/500/800 | 18.93 | 137k verts, barely helps |
| + Delaunay every 100 | 18.45 | No effect |
| n_quad_samples=2 | 18.75 | Slightly faster, similar quality |
| freeze_lr=2e-2 | 18.63 | Higher LR doesn't help |
| density_offset=-2 | 10.96 | Catastrophic |
| No vertex opt | 18.16 | Slightly worse |

**Discovery**: The vertex model is insensitive to standard tuning. Densification adds 85k vertices but barely improves quality. Something fundamental is wrong with convergence.

### Investigation 2: Training instability diagnosis (BREAKTHROUGH)

**Question**: Why does PSNR crash from 13.4 to -6 dB at step 50?

**Method**: Per-step logging of sigma, density, image range, and alpha for first 200 steps.

**Root cause**: **exp() density activation causes vanishing gradients**. With `density = exp(sigma - 4)`:
- At sigma=0: gradient = exp(-4) = 0.018 — 50x weaker than color gradients
- Color learns 100x faster than density, producing pixel values of 50+ through a nearly-transparent medium
- This causes catastrophic per-image oscillations

**Evidence**: At step 100, sigma max = 0.6, density max = 0.033, but image max still spiking to 9+. Even 10x sigma LR only reached density max = 0.111 at step 199.

**Fix**: Pass raw `sigma + offset` to the shader instead of `exp(sigma + offset)`. The shader's existing `max(x, 0)` provides the nonlinearity with unit gradient for positive density.

| Density activation | PSNR | Early convergence |
|---|---|---|
| exp(sigma-4) in Python | 18.43 | PSNR crashes to -6, slow recovery |
| **sigma+0 raw (max in shader)** | **19.42** | No crash, fast convergence |
| exp() in shader (log-space interp) | 18.24 | Same crash + Jensen's inequality loss |

**Why exp-in-shader is worse**: Barycentric interpolation in log-space gives lower density at midpoints (Jensen's inequality: `exp(mean) <= mean(exp)`). This makes the mesh "leakier" and loses 0.2 dB vs the original Python-side exp.

### Investigation 3: Vertex optimization frequency

**Question**: Does the optimal vertex opt frequency change with raw density?

| Frequency | PSNR |
|---|---|
| Every 5 steps | 19.42 |
| Every 3 steps | 19.65 |
| **Every 1 step** | **19.76** |

**Discovery**: Every-step vertex optimization is best for the vertex model. Unlike the old per-tet model (where every-5 was optimal), the vertex model benefits from continuous position updates because positions directly define both geometry and the interpolation field.

### Investigation 4: Delaunay with best config

| Config | PSNR |
|---|---|
| No Delaunay | **19.76** |
| Delaunay every 300 | 19.09 |

**Discovery**: Delaunay retriangulation consistently hurts the vertex model. The retriangulation changes the tet topology, which disrupts the learned per-vertex attribute interpolation. Since the vertex model has no per-tet parameters, the index buffer swap should be harmless — but the changed tet shapes alter how vertex attributes interpolate, effectively perturbing the rendered output.

### End-of-shift summary

- **Best PSNR**: 19.76 (raw density + vertex opt every step, 51k vertices)
- **Improvement**: +1.33 dB over untuned baseline (18.43)
### Investigation 5: Density activation — finding the right nonlinearity

**Question**: ReLU (max(x,0)) fixes convergence but creates discontinuities that break densification and retriangulation. Can we get the gradient benefits of ReLU while staying smooth?

**Key insight**: For any smooth positive function f, if f(x_init) is small then f'(x_init) is also small (the function must be climbing from near-zero). The exception: **softplus(x, beta)** = ln(1+exp(beta*x))/beta. At x=0: value = ln(2)/beta (tunable via beta), gradient = sigmoid(0) = **0.5 always** regardless of beta.

| Activation | PSNR | Grad@init | Smooth? | Densify-safe? |
|---|---|---|---|---|
| exp(sigma-4) [original] | 18.43 | 0.018 | yes | yes |
| raw sigma + max(x,0) | 19.76 | 1.0 | no | no |
| exp in shader (log interp) | 18.24 | 0.018 | yes | yes |
| softplus(sigma, beta=20) | ~19.4 | 0.5 | yes | yes |
| **softplus(sigma, beta=40)** | **19.68** | **0.5** | **yes** | **yes** |
| softplus(beta=5) | diverged | 0.5 | yes | — |

beta=40 gives initial density = ln(2)/40 = 0.017 (matching exp(-4)=0.018) with 28x stronger gradient. beta=5 was too opaque initially (density=0.139). beta=20 worked but slightly lower final PSNR.

With softplus + densification + Delaunay: **19.58** — no catastrophic failure (unlike ReLU which dropped 0.67 dB with Delaunay). But densification still doesn't help meaningfully.

**Discovery**: The densification problem is NOT caused by the activation function. Even with smooth softplus, adding 130k vertices via nearest-neighbor copy barely helps. The per-tet error-based densification strategy simply doesn't translate well to per-vertex needs.

- **Key discoveries**:
  1. **exp() density activation has vanishing gradients** — the dominant source of training instability
  2. **softplus(sigma, beta=40)** is the right fix: 28x stronger gradient, smooth, always positive, densify-safe
  3. **Every-step vertex optimization** gives +0.34 dB for vertex model
  4. **Delaunay retriangulation** is neutral with softplus, harmful with ReLU
  5. **Densification doesn't help** regardless of activation — the bottleneck is the densification strategy itself, not the activation
  6. **Log-space interpolation (exp in shader) is worse** due to Jensen's inequality
  7. Loss function tuning (SSIM, BW-SSIM) has negligible effect
- **Best config**: softplus(sigma, beta=40) + vertex opt every step + no densification + no Delaunay = **19.68 PSNR**
- **Open questions for next shift**:
  - Why doesn't densification help? The per-tet error metric doesn't target per-vertex needs. Need a vertex-native densification strategy.
  - Can per-vertex feature dimension be increased (e.g. more SH bands, or per-vertex gradient)?
  - Try softplus on the SimpleModel (per-tet) — does it help convergence there too?
  - The 51k vertex ceiling: is it capacity-limited or training-time-limited?
