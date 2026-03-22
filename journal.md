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

## Shift: 2026-03-22
Goals: Improve densification and retriangulation. Break through the 51k vertex ceiling.

### Investigation 1: Gradient-based edge midpoint splitting

**Question**: Can we improve densification by (a) targeting edges with high vertex gradient norms and (b) initializing new vertices with averaged endpoint attributes instead of nearest-neighbor?

**Hypothesis**: The old densification fails because nearest-neighbor attribute initialization corrupts the interpolation field. Edge midpoints with averaged attributes should preserve it exactly.

**Method**:
- Accumulate vertex position gradient norms during training
- At densification, build edge list, score edges by endpoint gradient sum
- Split top-k edges by adding midpoint vertices with averaged (sigma, rgb, sh)
- Schedule: 3000 vertices every 100 steps, steps 200-1100

**Results**: 19.60 PSNR with 81k vertices (baseline: 19.77 with 51k)

**Discovery**: **Attribute initialization is not the bottleneck — Delaunay retriangulation is.** Even with perfect midpoint-averaged attributes, each Delaunay completely restructures the tet topology. 10 retriangulations over training cost ~500 steps of training time, and each one partially undoes learned interpolation patterns. The extra vertices don't compensate for the disruption.

### Investigation 2: Upfront vertex densification (avoid mid-training retriangulation)

**Question**: What if we add all extra vertices before training starts, avoiding mid-training retriangulation entirely?

**Hypothesis**: The model is capacity-limited at 51k vertices, but mid-training retriangulation is too disruptive. Starting with more vertices sidesteps the retriangulation problem entirely.

**Method**: Before training, duplicate each initial COLMAP vertex with a small random offset (noise_scale=0.05), then run Delaunay once. No mid-training densification or retriangulation.

| Config | PSNR | n_vertices | Notes |
|---|---|---|---|
| Baseline (51k, no densify) | 19.77 | 51358 | ~1750 steps in 10 min |
| Grad edge midpoint (10 retriangs) | 19.60 | 81358 | Retriangulation disrupts learning |
| 2x init (offset 0.05) | 20.09 | 102716 | +0.32 dB, no mid-training retriang |
| **3x init (offset 0.05)** | **20.19** | **154074** | **+0.42 dB, new best** |
| 2x init (offset 0.005) | crash | 102716 | NaN: close vertices create degenerate tets |
| 4x init (offset 0.05) | hung | 205432 | Stalled at step 750, 1.3M tets too heavy |

**Discovery**: **Upfront densification works.** 2x initial vertices gives +0.32 dB (19.77 to 20.09) with minimal per-step slowdown (~0.20s vs ~0.18s per step). The model IS capacity-limited, but the solution is to start dense, not to add vertices mid-training.

**Critical finding on offset scale**: Offset 0.005 creates NaN at step ~400 because near-duplicate vertices produce degenerate tets. Offset 0.05 is stable. This sets a lower bound on vertex spacing for Delaunay meshes.

**4x scaling failure**: 205k vertices with 1.3M tets appeared to run but hung silently at step 750. Likely GPU memory pressure (24GB VRAM) causing kernel stalls. There's a practical upper limit on mesh size.

### End-of-shift summary

- **Best PSNR**: 20.19 (3x upfront densification, 154k vertices)
- **Improvement**: +0.42 dB over previous best (19.77), +1.76 dB over original baseline (18.43)
- **Key discoveries**:
  1. **Mid-training retriangulation is the densification bottleneck** — not attribute initialization, not targeting strategy. Each Delaunay restructures the entire mesh topology.
  2. **Upfront densification avoids this entirely** — duplicate initial vertices with random offset before training. Simple, effective, no disruption.
  3. **Vertex spacing matters** — offsets < 0.01 create degenerate tets that cause NaN. Offset 0.05 is stable.
  4. **The model is capacity-limited** — 2x vertices immediately improves quality without changing anything else.
  5. **4x vertices hits GPU limits** — 1.3M tets causes stalls on 24GB VRAM.
### Investigation 3: Tile size as VRAM bottleneck

**Question**: Why does 4x (205k verts, 1.3M tets) use 24GB VRAM when the parameters are only ~10MB?

**Method**: Added instrumentation to `tile_shader_slang.py` and `train.py` to measure index buffer size and VRAM usage.

**Root cause**: The tile-based rasterizer creates an index buffer of size `O(T * tiles_per_tet)`. With tile_size=4 on 1237x822 images, the grid is 310x206 = 63k tiles, and each tet spans **126 tiles on average** (its screen-space bounding box covers ~44x44 pixels = 11x11 tiles). This creates a 32M-entry index buffer (368MB), plus CUB sort temporary storage inflates PyTorch's reserved memory to **3GB**.

**Fix**: tile_size=16 reduces tiles-per-tet from 126 to **9** (14x reduction). The index buffer shrinks from 32M to 3M entries. VRAM drops from 3058MB to **472MB**.

| tile_size | tiles/tet | idx_buf_MB | VRAM_reserved | steps/s |
|---|---|---|---|---|
| 4 | 126 | 368 | 3058 MB | ~2.5 |
| 16 | 9 | 36 | 472 MB | ~3.5 |

**Discovery**: **tile_size=16 is strictly better** for this mesh density — faster per step AND uses 6x less VRAM. The smaller sort dominates any wasted-work cost from larger tiles.

**Combined result**: 4x upfront + tile_size=16 = **20.61 PSNR** with 205k vertices. +0.84 dB over baseline. More training steps (2100 vs 1750) despite 4x more vertices, because tile_size=16 is faster per step.

### Updated end-of-shift summary

- **Best PSNR**: 20.61 (4x upfront densification + tile_size=16, 205k vertices)
- **Improvement**: +0.84 dB over previous best (19.77), +2.18 dB over original baseline (18.43)
- **Key discoveries**:
  1. **Mid-training retriangulation is disruptive** — topology change, not Delaunay compute cost, is the problem
  2. **Upfront densification works** — duplicate initial vertices with random offset before training
  3. **tile_size=4 was the VRAM bottleneck** — O(T*tiles/tet) index buffer + CUB sort temp = 3GB for 1M tets
  4. **tile_size=16 is strictly better** — 14x smaller index buffer, faster per step, 6x less VRAM
  5. **The model is capacity-limited** — 4x vertices = +0.84 dB with room to scale further
  6. **Vertex spacing** — offsets < 0.01 create degenerate tets (NaN)
- **Open questions for next shift**:
  - Scale to 500k+ vertices (10x+) — the VRAM headroom now exists
  - What's the PSNR ceiling with unlimited vertices? Are we training-time-limited at 500k?
  - Can we use even larger tiles (32x32) for further scaling?
  - Would the evaluation harness need tile_size=16 too? (test_util.py is read-only)
