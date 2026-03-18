# Discovery Journal

## Session: 2026-03-14 (compact summary of ~125 experiments)

Starting from defaults (20.26 PSNR), systematically tuned all parameters. Key accepted changes:
- **max_sh_deg=1** + **densify_interval=400** → 20.35
- **threshold_start=2500** + culling thresholds → 20.38
- **density_offset=-4** → 20.53 (lower initial density forces tets to learn opacity)
- **densify_interval=700** + alpha_threshold=0 → 20.59
- **clone_min_contrib=2/255** → 20.68
- **densify_start=300** → 20.82 (more post-densification convergence time)
- **densify_end=1100** hard cap → 20.78 reproducible (robust to GPU speed)

Rejected (all hurt): LR changes, loss function tuning, architecture changes, densification threshold changes, 3+ densification events. System sits on knife-edge between 2 and 3 events; any timing change shifts the balance. 125+ experiments confirm plateau at 20.78 PSNR.

Critical insight: the system is extremely sensitive to the number of densification events. 2 events ≈ 20.8 PSNR, 3 events ≈ 20.4 PSNR. Each event causes a ~5 dB PSNR crash from full Delaunay retriangulation + parameter transfer.

---

## Shift: 2026-03-16
Goals: Investigate orphan tet behavior; understand the post-densification PSNR drop; develop zero-disruption densification.

### Investigation 1: Orphan tet deep dive

**Question**: Do orphan tets (containing newly added vertices) create floaters by inheriting bad density?

**Method**: Added instrumentation to `update_triangulation` logging orphan counts, density distributions, blended vs nearest-copy values, and non-orphan comparisons. Tested 8 variants of orphan handling.

**Instrumentation results**:
- After 1st densification (51k→73k verts): 59% of tets are orphans
- After 2nd densification (73k→145k verts): **84% orphans**
- Orphan density 10-15x higher than non-orphan (expected — opaque regions stay opaque)
- Blended vs nearest-copy sigma differ by >1.0 for 40% of orphans

| Experiment | PSNR | Finding |
|---|---|---|
| Baseline | 20.80 | Reference |
| Orphan sigma=0 | 19.33 | New verts can't contribute if transparent |
| Skip orphan override | 20.82 | Override is redundant with blending |
| Disable density_scale | 20.75 | Scale correction not root cause |
| Fresh Adam state | 20.76 | Blended momentum is net positive |
| Tighter density_scale clamp | 20.74 | No effect |
| Centroid split points | 20.68 | Error-targeted placement is better |
| 3x LR boost after densify | 18.26 | Catastrophic instability |
| 4 small densifications | 19.00 | Too many disruptions |

**Discovery**: The 5 dB post-densification PSNR drop is **structural** — caused by topology change when doubling vertex count, not by bad parameter initialization. No initialization strategy changes the final converged quality. The system self-corrects within a few hundred steps regardless.

### Investigation 2: Zero-disruption densification (1-to-4 inplace tet split)

**Question**: Can we eliminate the post-densification PSNR crash entirely?

**Hypothesis**: The crash comes from full Delaunay retriangulation + approximate parameter transfer. If we instead split tets in-place (1-to-4 at centroids), each sub-tet gets an exact copy of the parent's parameters. The rendered image should be identical before and after the split. The periodic Delaunay retriangulation (every 300 steps) can fix mesh quality separately.

**Method**: Implemented `split_tets_inplace()` on SimpleOptimizer:
1. For each selected tet (v0,v1,v2,v3), compute centroid c
2. Replace with 4 sub-tets: (c,v1,v2,v3), (v0,c,v2,v3), (v0,v1,c,v3), (v0,v1,v2,c)
3. Copy parent's density/rgb/gradient/sh exactly to all 4 children
4. Copy Adam momentum from parent to children
5. No retriangulation — the next periodic Delaunay step handles topology

Modified `apply_densification` to call `split_tets_inplace(clone_mask)` instead of `tet_optim.split(split_point)`.

**Results — convergence comparison (baseline vs inplace split)**:
```
BASELINE (old split, schedule 300/1000):
  Step 300 densify: PSNR 15.4 → 13.5  (1.9 dB crash)
  Step 1000 densify: PSNR 18.2 → 12.5 (5.7 dB crash)
  Final: 20.80

INPLACE SPLIT (schedule 200/700):
  Step 200 split: PSNR 15.4 → 16.5   (NO crash — continued improving!)
  Step 300 Delaunay: 17.3 → 16.7     (0.6 dB from topology fix only)
  Step 700 split: PSNR 17.9 → 18.2   (NO crash!)
  Step 900 Delaunay: 18.2 → 17.4     (0.8 dB from topology fix)
  Final: 20.80
```

**The post-densification PSNR crash is eliminated.** Remaining drops (~0.8 dB) come only from periodic Delaunay retriangulation, which is far less disruptive.

| Schedule | PSNR | Verts | Notes |
|---|---|---|---|
| Old split 300/1000 (baseline) | 20.80 | 145k | 2 events, 5 dB crashes |
| **Inplace 200/700** | **20.80** | **139k** | **2 events, 0 dB crashes** |
| Inplace 300/1000 | 20.75 | 168k | More tets, slightly slower |
| Inplace 200/600 | 20.75 | 144k | Slightly worse split decisions |
| Inplace 150/650 | 20.45 | 131k | Too early, bad error statistics |
| Inplace 3 events 200/500/800 | 20.76 | 315k | Too many tets (2M), slow per-step |

### Investigation 3: Undo mechanism (adaptive coarsening)

**Question**: Can we automatically undo useless splits to recover vertex budget?

**Hypothesis**: After a 1-to-4 split, if the 4 sibling tets haven't diverged in their parameters (density variance < threshold), the split was in a uniform region and should be undone.

**Method**: Implemented `undo_useless_splits(threshold)` that checks sibling divergence at each Delaunay step. Non-diverged groups get merged back: centroid vertex removed, parent tet restored with averaged parameters.

**Results**:
- threshold=0.5: **96-99% of splits undone** → only 52k verts survive → PSNR 20.19
- threshold=0.01: **33-83% undone** → 80k verts → PSNR 20.20

**Discovery**: The undo fires too aggressively because siblings haven't had enough training time to diverge. 150 steps (one Delaunay interval) isn't enough. The mechanism is architecturally sound but needs either a much longer delay before checking, or a divergence metric that captures potential rather than realized difference.

### End-of-shift summary

- **Best PSNR**: 20.80 (inplace split at 200/700 — matches baseline with smoother convergence)
- **Key discoveries**:
  1. Post-densification PSNR crash is structural (topology change), not initialization
  2. **1-to-4 inplace split eliminates the crash entirely** — zero disruption from densification
  3. Densification and Delaunay retriangulation are now **decoupled** — can tune independently
  4. Orphan override code is dead weight (blending alone gives same result)
  5. The undo mechanism works but needs longer divergence windows
- **Open questions for next shift**:
  - Can the undo be made to work with a longer delay (check 500+ steps after split)?
  - With zero-cost splits, can we do many small splits iteratively (split → train → split more)?
  - Can we reduce the Delaunay retriangulation disruption (the remaining 0.8 dB drops)?
  - Does the inplace split help on other scenes (not just bicycle)?

---

## Shift: 2026-03-17 (overnight)
Goals: Break through the 20.80 PSNR plateau. Investigate Delaunay disruption, split scheduling, loss functions, and vertex optimization frequency.

### Investigation 1: Delaunay interval and disruption timing

**Question**: Can we reduce or eliminate the PSNR disruption from periodic Delaunay retriangulation?

**Method**: Added per-step convergence logging. Tested delaunay_interval=300 (baseline), 500, and no-Delaunay. Also added timing instrumentation to the Delaunay pipeline.

**Results**:
| Config | PSNR | Notes |
|---|---|---|
| Baseline (interval=300) | 20.72 | 9 Delaunay events, ~10dB single-image crashes |
| interval=500 | 20.80 | Same final PSNR, fewer events |
| No Delaunay (vertex opt 300) | 20.21 | Bad mesh quality without retriangulation |

**Timing breakdown** (per Delaunay event):
- Prep (CC, v2t, edge keys): 0.01-0.05s
- Delaunay triangulation itself: 0.06-0.15s
- Transfer weights: 0.02-0.89s (fast when topology unchanged, slow post-split)
- Total: 0.12-1.06s per event

**Discovery**: Delaunay interval doesn't affect final PSNR (300 vs 500 give same result). The transfer is the bottleneck but total overhead is only ~5s per run (< 1%). The real cost is PSNR disruption, not wall-clock time. Delaunay IS essential — removing it drops PSNR by 0.6 dB.

### Investigation 2: Three-split densification with per-split vertex cap

**Question**: Can we use more of the 500k vertex budget effectively with 3 inplace splits?

**Hypothesis**: The previous 3-split attempt (mar16) failed because it created 2M tets (too slow). A per-split cap on new vertices keeps tet count manageable while using more vertices.

**Method**: Tested 3 inplace splits at various schedules with caps on per-split vertex additions.

**Results**:
| Schedule | Cap | PSNR | Verts | Tets | Notes |
|---|---|---|---|---|---|
| 200/450/700 | 50k | 20.94 | 167k | 1.07M | Good |
| 200/500/800 | 50k | **21.00** | 167k | 1.07M | **Best schedule** |
| 200/450/700 | 75k | 20.78 | 210k | 1.34M | Too many tets, slow training |
| 200/400/600/800 | 40k | 20.92 | 187k | 1.20M | 4 splits, extra disruptions |
| 150/400/650 | 50k | 20.83 | 166k | 1.06M | Too early, bad split decisions |
| 200/500/800/1100 | 35k | 20.99 | 172k | 1.10M | 4th split doesn't help |

**Discovery**: 3 splits at 200/500/800 with 50k cap per split is optimal. This yields ~167k vertices (vs 139k for 2 splits) and 1.07M tets. The wider spacing (300 steps apart) gives better error statistics for each split. Per-split cap of 50k balances vertex count vs tet overhead. More than 50k per split → too many tets → slower per-step → fewer total steps → worse PSNR.

### Investigation 3: Loss function and parameter tuning with 3 splits

**Question**: Can loss function changes compound with the 3-split improvement?

**Method**: Tested various loss/parameter changes on top of the best 3-split config.

**Results**:
| Change | PSNR | Notes |
|---|---|---|
| lambda_ssim=0.2 (2 splits) | 20.74 | Better SSIM/LPIPS, worse PSNR |
| lambda_ssim=0.1 (3 splits) | 20.92 | Same tradeoff |
| lambda_ssim_bw=0.3 | 20.94 | No improvement |
| freeze_lr=4e-2 | 20.83 | Too high |
| density_offset=-5 | 20.80 | Too conservative |
| max_sh_deg=2 | 21.00 | No improvement |
| L2 loss | 20.85 | L1 is better |
| lambda_dist=100 | 15.29 | Way too strong |
| density_threshold=0.005 | 20.81 | Removed useful tets |

**Discovery**: The 3-split config is highly robust to parameter changes. None of the loss/LR/architecture tweaks improved upon L1 loss with existing parameters. The gain comes from structural changes (more splits, better scheduling), not hyperparameter tuning.

### Investigation 4: Vertex optimization frequency (BREAKTHROUGH)

**Question**: Vertices are only optimized during Delaunay events (every 300 steps). What if we optimize them more frequently?

**Hypothesis**: Vertex positions are underfitting because they only get updated every 300 steps. The per-tet parameters (density, color) update every step, but vertices accumulate gradients for 300 steps before a single update. More frequent vertex optimization should give the geometry more chances to adapt.

**Method**: Tested vertex_optim.step() at various frequencies (every 1, 3, 5, 10, 50, 300 steps).

**Results**:
| Vertex opt frequency | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Every 300 (baseline) | 21.00 | 0.462 | 0.577 |
| Every 50 | 21.22-21.24 | 0.485 | 0.563 |
| **Every 10** | **21.42** | **0.501** | **0.553** |
| **Every 5** | **21.50** | **0.504** | **0.553** |
| Every 3 | 21.40 | 0.501 | 0.556 |
| Every 1 | 21.25 | 0.494 | 0.563 |

**Discovery**: **Vertex optimization frequency is a massive lever.** Going from every-300 to every-5 gives +0.50 dB PSNR, +0.04 SSIM, -0.024 LPIPS. This is the single largest improvement found across all experiments. The optimal frequency is every 5 steps — more frequent than that over-optimizes vertex positions (the per-tet parameters can't keep up), less frequent under-utilizes the geometry.

**Why this works**: In the Delaunay tet mesh, vertex positions define the mesh structure. When vertices update slowly, the per-tet parameters must compensate for suboptimal vertex placement. With more frequent vertex updates, the mesh geometry adapts continuously, allowing tets to focus on representing local color/density rather than compensating for misplaced vertices. This is analogous to how 3D Gaussian Splatting benefits from continuous position optimization.

### End-of-shift summary

- **Best PSNR**: 21.48 avg (21.50 + 21.46 across 2 runs)
- **Configuration**: 3 inplace splits at 200/500/800 (50k cap each) + vertex opt every 5 steps
- **Improvement**: +0.68 dB over previous best (20.80)
- **Key discoveries**:
  1. **Three capped inplace splits** (200/500/800, 50k each) uses more vertex budget effectively (+0.20 dB)
  2. **Frequent vertex optimization** (every 5 steps instead of 300) is the biggest single lever (+0.50 dB)
  3. Delaunay interval doesn't matter much (300 vs 500 give same result)
  4. Delaunay retriangulation IS essential — removing it drops quality by 0.6 dB
  5. Loss function changes (SSIM, L2, distortion) don't improve PSNR on this system
  6. The system is robust to parameter changes — gains come from structural improvements
### Investigation 5: Vertex LR tuning for frequent updates

**Question**: Should vertex LR change since we're updating 60x more frequently?

**Results**:
| Change | PSNR | Notes |
|---|---|---|
| vertices_lr=1e-4 (baseline) | 21.48 | Optimal |
| vertices_lr=2e-5 | 21.33 | Too slow |
| vertices_lr=3e-4 | 21.18 | Too fast, unstable |
| vert_lr_delay=200 | 21.43 | No clear effect |
| SGLD noise_lr=0.01 | 21.40 | No effect |
| num_samples=100 | 21.28 | More tets, slower |

**Discovery**: The original vertex LR (1e-4) is already well-tuned even at the higher update frequency. The exponential LR schedule handles the adaptation naturally. No parameter changes improve upon the committed config.

- **Open questions for next shift**:
  - Can we further improve by changing how Delaunay transfers parameters (reduce the ~10dB disruption)?
  - Does the vertex opt frequency interact with Delaunay interval?
  - Can we push further with 4+ splits now that vertex opt is more frequent?
  - Try different scenes beyond bicycle
