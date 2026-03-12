# micro_rmesh

This is an experiment to have the LLM do its own research.

## Premise

This is an implementation of neural radiance fields represented by a delaunay triangulation, where each cell has a constant density, and linear color.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `floater/mar5`). The branch `floater/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b floater_/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `train.py` — one of the files you modify. Optimizer, training loop
   - `model.py` — one of the files you modify. Model architecture
   - `utils/densification.py` — controls how vertices are added
   - `utils/decimation.py` — controls how edges are removed
   - `utils/model_util.py` — Important details about how linear colors are processed and activated
   - `rmesh_renderer/slang/alphablend_shader_interp.slang` — The main alpha blending loop
   - `rmesh_renderer/slang/interp_version.slang` — Handles integration across each primitive.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 10 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `CUDA_VISIBLE_DEVICES=3 uv run train.py`.

**What you CAN do:**
- Modify `train.py` — training loop, hyperparameters, loss functions, scheduling.
- Modify `model.py` — model architecture, optimizer, Delaunay triangulation, parameter transfer.
- Modify `utils/densification.py` — vertex cloning/splitting strategy.
- Modify `utils/decimation.py` — edge collapse strategy.
- Modify `utils/model_util.py` — feature activation, color field processing.
- Modify `utils/optim.py` — custom Adam optimizer wrapper.
- Everything is fair game: architecture, optimizer, hyperparameters, training loop, batch size, model size, densification/decimation strategy, loss functions, etc.

**What you CANNOT do:**
- Modify `data/*`. It is read-only. It contains camera/dataset loading (COLMAP format, intrinsics, extrinsics, images).
- Modify `rmesh_renderer/*`. It is read-only (except for the entropy changes already made). It contains the Slang-based tile rendering pipeline and shaders.
- Modify `submodules/*`. It is read-only. It contains the LPIPS metric (`lpipsPyTorch`).
- Modify `test_util.py`. It is read-only (except for the entropy tracking already added). It contains the evaluation function and the constants `VERT_BUDGET` (500k) and `TIME_BUDGET` (600s).
- Modify `utils/train_util.py`. It is read-only (except for the entropy loss already added). It contains the `render()` function.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate` function in `test_util.py` is the ground truth metric.

**The goal has a dual mandate: get the highest PSNR while reducing ray weight entropy.** PSNR measures reconstruction quality; ray weight entropy (lower is better) measures how concentrated vs. diffuse the rendering weights are along each ray — low entropy means sharp surfaces, high entropy means floaters/fog. Since the time budget is fixed, you don't need to worry about training time — it's always 10 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful PSNR gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 PSNR improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 PSNR improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
----------
n_vertices: 51358
n_interior_vertices: 48201
n_tets: 301245
test_SSIM: 0.812
test_PSNR: 20.5
test_LPIPS: 0.312
test_ENTROPY: 0.1234
```

During training, each epoch also prints average ray weight entropy:
```
TRAIN PSNR: 20.50 ENTROPY: 0.1500 #V: 51358 #T: 301245
```

Note that the script is configured to always stop after 10 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metrics from the log file:

```
grep "^test_PSNR:\|^test_ENTROPY:\|^n_vertices:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	PSNR	ENTROPY	n_vertices	status	description
```

1. git commit hash (short, 7 chars)
2. PSNR achieved (e.g. 19) — use 0.000000 for crashes
3. ENTROPY (average ray weight entropy, lower is better) — use 0.000000 for crashes
4. number of vertices in final model
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	PSNR	ENTROPY	n_vertices	status	description
02c58f1	20.5	0.1234	51358	keep	baseline
b2c3d4e	20.9	0.1100	51358	keep	increase LR to 0.04
d4e5f6g	0.000000	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `floater_/mar5` or `floater_/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py, model.py` with an experimental idea by directly hacking the code.
3. git add -A, git commit
4. Run the experiment: `CUDA_VISIBLE_DEVICES=3 uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^test_PSNR:\|^test_ENTROPY:\|^n_vertices:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If the result is a Pareto improvement (PSNR increased or entropy decreased, without the other getting worse), you "advance" the branch, keeping the git commit. When in doubt, prioritize PSNR.
9. If neither metric improved, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~10 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~10 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
