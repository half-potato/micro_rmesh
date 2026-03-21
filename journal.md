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
- **Key discoveries**:
  1. **exp() density activation has vanishing gradients** at initialization — the dominant source of training instability
  2. **Raw density with ReLU in shader** fixes convergence and gives +1.0 dB
  3. **Every-step vertex optimization** gives +0.34 dB for vertex model (different optimum than per-tet model)
  4. **Delaunay retriangulation hurts** the vertex model (-0.67 dB)
  5. **Densification barely helps** (+0.13 dB for 3.5x more vertices) — the bottleneck is not vertex count
  6. **Log-space interpolation (exp in shader) is worse** due to Jensen's inequality
  7. Loss function tuning (SSIM, BW-SSIM) has negligible effect
- **Open questions for next shift**:
  - The 51k vertex model seems capacity-limited at ~19.8. Can per-vertex feature vectors increase capacity without more vertices?
  - Try the raw density fix on the SimpleModel (per-tet) — does it help there too?
  - Why does densification not help? Is the error-based selection bad, or is nearest-neighbor attribute initialization the problem?
  - Explore higher SH degree or additional per-vertex learnable features
