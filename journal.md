# Discovery Journal

## Shift: 2026-03-21
Goals: Understand VertexModel behavior, fix training instability, improve from untuned baseline.

### Investigation 1: VertexModel baseline characterization

**Question**: How does the untuned VertexModel perform?

**Results**: 18.43 PSNR with 51k vertices, no densification, no retriangulation. ~1850 steps in 10 min (~0.31s/step). Per-image PSNR variance is extreme (0 to 20 in same epoch).

### Investigation 2: Enabling densification and retriangulation

**Question**: Do densification and Delaunay help the vertex model?

| Config | PSNR | Verts | Notes |
|---|---|---|---|
| Baseline (no densify, no Delaunay) | 18.43 | 51k | Reference |
| Densification 200/500/800 + Delaunay 300 | 18.93 | 137k | +0.5 barely helps |
| Delaunay every 100 (no densify) | 18.45 | 51k | No effect |
| No vertex position optimization | 18.16 | 51k | Slightly worse |

**Discovery**: Neither densification nor retriangulation helps meaningfully. The vertex model's bottleneck is not vertex count or mesh quality.

### Investigation 3: Parameter tuning

| Change | PSNR | Notes |
|---|---|---|
| n_quad_samples=2 (was 4) | 18.75 | ~7% faster, similar quality |
| freeze_lr=2e-2 (was 6e-3) | 18.63 | Higher LR doesn't help |
| density_offset=-2 (was -4) | 10.96 | Catastrophic — too opaque initial |
| voxel_size=0 (more initial verts) | ~17.8 | COLMAP only has ~55k points anyway |

**Discovery**: The vertex model is insensitive to these standard tuning knobs.

### Investigation 4: Training instability deep dive (BREAKTHROUGH)

**Question**: Why does PSNR crash from 13.4 to -6 dB in the first 50 steps?

**Method**: Added per-step instrumentation logging sigma stats, density range, image value range, and alpha for the first 200 steps.

**Key finding**: The instability is caused by a **density-color convergence mismatch**.

**Evidence** (with exp(sigma-4) activation, step-by-step):
- Step 0: sigma=0, density=exp(-4)=0.018, alpha=0.17, img=[0.41-0.42] (uniform gray)
- Step 3: sigma unchanged, density unchanged, img max=**24.4** (color exploding!)
- Step 6: sigma unchanged, density unchanged, img max=**53.2**
- Step 50: sigma max=0.2, density max=0.023, img max=9.3
- Step 100: sigma max=0.6, density max=0.033, img still spiking to 9+
- Step 200: sigma max still only 0.1 (at 1.5x LR), density max=0.028

**Root cause**: The gradient through `exp(sigma + offset)` at offset=-4 is `exp(sigma-4) ≈ 0.018`. This is 50x smaller than the color gradient. Adam can't fix this because the gradient magnitude is genuinely tiny. Color learns 100x faster than density, so colors must be extreme (50x normal) to render anything visible through the nearly-transparent medium. This creates catastrophic oscillations.

**Fix**: Replace `exp(sigma + offset)` with raw `sigma + offset` passed directly to the shader. The shader's existing `max(sigma_s, 0)` provides the nonlinearity. The gradient through `max` is 1 for positive density (vs 0.018 for exp), giving **50x stronger density gradients**.

**Results**:

| Density activation | PSNR | Convergence |
|---|---|---|
| exp(sigma-4) baseline | 18.43 | PSNR=13.4 at step 0, crashes to -6, slow recovery |
| **sigma+0 raw (max in shader)** | **19.42** | **PSNR=15.8 at step 0, no crash, fast convergence** |
| Raw + densification | 19.55 | Barely helps (+0.13) despite 3.5x more vertices |
| Raw + lambda_ssim=0.2 | 19.42 | No improvement |

With 10x sigma LR: density grew faster (max=0.111 at step 199 vs 0.024 with 1.5x), confirming the gradient bottleneck hypothesis. But even 10x LR wasn't enough to fully fix it — the raw activation is fundamentally better.

**Changes**: `get_vertex_values()` now returns `sigma + offset` instead of `exp(sigma + offset)`. `density_offset` changed from -4 to 0. `sigma` initialized to 0.01 (small positive) instead of 0. `n_quad_samples` changed from 4 to 2.

### End-of-shift summary (in progress)

- **Best PSNR**: 19.42 (raw density, no densification, 51k vertices)
- **Key discoveries**:
  1. **exp() density activation causes vanishing gradients** — the single biggest source of training instability
  2. **Raw density with ReLU in shader fixes it** — +1 dB, eliminates color explosions
  3. Densification barely helps the vertex model (+0.13 dB for 3.5x more vertices)
  4. Delaunay retriangulation has no effect for vertex model
  5. Quad samples 2 vs 4 is negligible quality difference
- **Open questions**:
  - Can we push past 19.5 with vertex model? Or is 51k vertices the real bottleneck?
  - Would a different densification strategy help (current one designed for per-tet model)?
  - Try vertex model with raw density on SimpleModel comparison
  - Try different sigma initialization values
  - Try disabling BW-SSIM loss (saves compute, may help or hurt)
