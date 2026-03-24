# micro_rmesh

LLM-driven research on neural radiance fields represented by a Delaunay tetrahedral mesh with per-vertex attributes.

## Setup

1. Agree on a run tag (e.g. `mar5`). Branch `micro_rmesh/<tag>` must not exist.
2. `git checkout -b micro_rmesh/<tag>` from master.
3. Read: `train.py`, `model.py`, `utils/densification.py`, `utils/decimation.py`, `utils/model_util.py`, `rmesh_renderer/slang/alphablend_shader_interp.slang`, `rmesh_renderer/slang/interp_version.slang`.
4. Create `results.tsv` (header only).
5. Run baseline: `uv run train.py`. Record result.
6. Read `journal.md` for prior discoveries. Begin shift.

## Experimentation

Fixed time budget (see `test_util.py`). Launch: `PYTHONUNBUFFERED=1 uv run python -u train.py 2>&1 | tee run.log`.

**Modifiable**: `train.py`, `model.py`, `utils/densification.py`, `utils/decimation.py`, `utils/model_util.py`, `utils/optim.py`, `rmesh_renderer/*`.

**Read-only**: `data/*`, `submodules/*`, `test_util.py`, `utils/train_util.py`. No new dependencies.

**Goal**: Highest PSNR. Understand *why* each change helps or hurts.

## Mesh Management Pipelines

The rasterizer sorts tets by depth per tile — **requires valid Delaunay triangulation**. Any vertex change must be followed by Delaunay. Delaunay is fast (~200ms) but **topology disruption** costs ~0.1-0.5 dB per call. Costs are cumulative.

- **Densification** (`apply_densification`): Adds vertices at centroids of high-error tets. Attributes = average of 4 corners. → Delaunay.
- **Refinement** (`refine_bad_tets`): Fixes sliver tets by inserting at circumcenters. Maintains triangulation quality for correct sorting. → Delaunay. The final tets MUST be Delaunay.
- **Decimation** (`apply_decimation`): Removes redundant vertices (short edges, uniform color, low density) to free budget. → Delaunay.
- **MCMC** (`apply_mcmc_relocation`): Teleports expendable vertices to high-error locations. Same count, positions change. → Delaunay.

See source files for detailed docs.

## Results logging

`results.tsv`: tab-separated, 5 columns: `commit PSNR n_vertices status description`. Never stage/commit this file.

## Research process

Organize work into **shifts**. Each shift: observe → hypothesize → test → update. Record findings in `journal.md`. Commit validated improvements. Failed experiments go in journal + results.tsv, not git history.

End each shift with a discovery report in `journal.md`: questions investigated, key discoveries, evidence, best config, open questions, failed ideas.

Prefer depth over breadth. Be autonomous — don't pause to ask permission.

## Syntax constraints

Run commands one at a time. No chaining (`&&`, `;`), subshells (`$()`), heredocs, piping (`|`), or redirection (`>`). Simple `git commit -m "message"`. No inline loops. Use `tee` for logging.
