# micro_rmesh

This is an experiment to have the LLM do its own research.

## Premise

This is an implementation of neural radiance fields represented by a delaunay triangulation, where each cell has a constant density, and linear color.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `micro_rmesh/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b micro_rmesh/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `train.py` — one of the files you modify. Optimizer, training loop
   - `model.py` — one of the files you modify. Model architecture
   - `utils/densification.py` — controls how vertices are added
   - `utils/decimation.py` — controls how edges are removed
   - `utils/model_util.py` — Important details about how linear colors are processed and activated
   - `rmesh_renderer/slang/alphablend_shader_interp.slang` — The main alpha blending loop
   - `rmesh_renderer/slang/interp_version.slang` — Handles integration across each primitive.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run. **Verify that `results.tsv` is listed in `.gitignore`** — it must never be staged, committed, or affected by git resets.
5. **Confirm and go**: Confirm setup looks good.
6. **Initial planning phase**: Before running the baseline, analyze the architecture for optimization opportunities. Read all in-scope files carefully and write an initial experiment plan to `plan.md` (see the **plan.md format** below) with 5–10 hypotheses ranked by priority. Each hypothesis must have explicit accept/reject/inconclusive criteria. Append the initial session to `journal.md` (see the **journal.md format** below). **Both `plan.md` and `journal.md` must stay untracked by git** — like `results.tsv`, they must never be staged, committed, or affected by git resets. Verify they are in `.gitignore`.
7. **Set up planning cron**: Create a recurring cron job (~every hour) to trigger a planning session. Use `CronCreate` with a prompt like: *"Follow the Planning Session procedure in program.md. Read program.md first for the full instructions."* This ensures periodic strategic review rather than ad-hoc experimentation.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 10 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

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
- Modify `rmesh_renderer/*`. It is read-only. It contains the Slang-based tile rendering pipeline and shaders.
- Modify `submodules/*`. It is read-only. It contains the LPIPS metric (`lpipsPyTorch`).
- Modify `test_util.py`. It is read-only. It contains the evaluation function and the constants `VERT_BUDGET` (500k) and `TIME_BUDGET` (600s).
- Modify `utils/train_util.py`. It is read-only. It contains the `render()` function.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate` function in `test_util.py` is the ground truth metric.

**The goal is simple: get the highest PSNR.** Since the time budget is fixed, you don't need to worry about training time — it's always 10 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

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
```

Note that the script is configured to always stop after 10 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metrics from the log file:

```
grep "^test_PSNR:\|^n_vertices:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	PSNR	n_vertices	status	description
```

1. git commit hash (short, 7 chars)
2. PSNR achieved (e.g. 19) — use 0.000000 for crashes
3. number of vertices in final model
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	PSNR	n_vertices	status	description
02c58f1	20.5	51358	keep	baseline
b2c3d4e	20.9	51358	keep	increase LR to 0.04
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `micro_rmesh/mar5` or `micro_rmesh/mar5-gpu0`).

LOOP FOREVER:

Each hypothesis gets a **series of ~5 experiments** (~53 minutes total) to properly test it. This allows two-tail exploration (the current setting could be too high or too low) and gives enough data points to draw a real conclusion.

**For each hypothesis:**

1. Look at the git state: the current branch/commit we're on. Note the starting commit — this is your **revert point** for the hypothesis.
2. Pick the next hypothesis from `plan.md` (highest priority first). If the plan is exhausted, trigger a planning session.
3. Run all experiments in the hypothesis's experiment plan sequentially (typically ~5 runs):
   a. Implement the next experiment in the plan by modifying the code.
   b. git commit (message should reference the hypothesis and experiment, e.g. "H1.3: LR 0.02")
   c. Run: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
   d. Read results: `grep "^test_PSNR:\|^n_vertices:" run.log`
   e. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't fix it quickly, log "crash" and move to the next experiment in the plan.
   f. Record in `results.tsv`. **CRITICAL: `results.tsv` is in `.gitignore` and must NEVER be staged or committed. It is the persistent experiment log that must survive all git resets. Before any `git add`, verify results.tsv is not being staged.**
   g. Git reset back to the revert point before starting the next experiment (each experiment tests independently from the same baseline).
4. **Judge the hypothesis**: After all ~5 experiments are done, evaluate the results as a group against the accept/reject/inconclusive criteria. Append the verdict to `journal.md`.
5. If ACCEPT: re-apply the best-performing experiment's changes (the one with highest PSNR), commit, and advance the branch. This becomes the new baseline for subsequent hypotheses.
6. If REJECT or INCONCLUSIVE: ensure you're at the revert point. Move on to the next hypothesis.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~10 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. Each hypothesis takes ~53 minutes (5 experiments × ~10 min + overhead), so you can test roughly 1 hypothesis per hour, or about 8 hypotheses over an 8-hour sleep. The user wakes up to a clean hypothesis journal showing which ideas panned out and which didn't.

## Planning Sessions

Planning sessions are triggered either by the hourly cron job (set up in step 7) or manually. When a planning session fires, follow these steps:

### Step 1: Deep run.log investigation

Read `run.log` fully — do NOT just grep the final metrics. Extract and analyze:

- **PSNR convergence curve**: The `TRAIN PSNR: X.XX #V: Y #T: Z` lines show how the model trains over time. Is PSNR still climbing at the end (more training time would help)? Does it plateau early (architecture bottleneck)?
- **Densification events**: Lines like `#Grow: X #Split: Y | #Alive: Z | Total Avg: A Within Avg: B` show when and how the mesh grows. Are densifications causing PSNR drops? How long does recovery take? Are grow/split ratios healthy?
- **Vertex trajectory**: Track #V over time. Is the model using the full vertex budget (500k)? If not, densification might be too conservative. If it hits 500k early, the budget is being used up before training converges.
- **Loss patterns**: Look for anomalies — sudden PSNR drops, oscillation, failure to recover after densification.

### Step 2: Review results.tsv

- Look at keep/discard/crash patterns across all experiments
- Identify which categories of changes tend to improve PSNR (e.g., LR changes, architecture changes, densification strategy)
- Note diminishing returns — are recent experiments showing smaller gains?

### Step 3: Read recent git log

- Understand what's been tried recently
- Look at the trajectory of ideas

### Step 4: Write plan.md

Overwrite `plan.md` with the new plan using the **plan.md format** below. Frame each planned experiment as a hypothesis with explicit accept/reject/inconclusive criteria. Hypotheses should be grounded in the run log analysis (Step 1) and results review (Step 2) — not just random ideas.

### Step 5: Append to journal.md

Start a new session section in `journal.md` using the **journal.md format** below. If there are completed experiments since the last planning session whose results haven't been logged yet, add their hypothesis outcomes to the table.

### Step 6: Resume

Continue the experiment loop with the top-ranked experiment from the plan.

---

## plan.md format

`plan.md` is a structured, overwritable working document. Overwrite it each planning session. Each experiment is framed as a **hypothesis** — a testable claim about what will improve PSNR and why. This forces precise thinking and makes results interpretable.

```markdown
# Experiment Plan
Date: YYYY-MM-DD HH:MM
Current best PSNR: <value>
Experiments completed: <count>

## Run Log Analysis
- Convergence: <was PSNR still rising at end? plateau? oscillating?>
- Densification: <how many events? PSNR drop/recovery pattern?>
- Vertex usage: <final #V vs 500k budget>
- Key observation: <most important insight from the training trajectory>

## Results Review
- Best result: <exp description> (PSNR X.XX)
- What's working: <pattern>
- What's not working: <pattern>
- Crash rate: <N/M experiments crashed>

## Hypotheses to Test
Each hypothesis is a **two-tail test**: the current setting might be too high, too low, or just wrong in some direction. The experiment plan explores both sides to find where the optimum lies. Each hypothesis gets ~5 experiments (~53 minutes).

### H<N>: <short title>
- **Claim**: <what you believe is suboptimal and why — rooted in the run log analysis or prior results>
- **Experiments** (~5 runs exploring the parameter space in both directions):
  1. <concrete change — e.g. parameter value or code modification>
  2. <another variation>
  3. <...>
  4. <...>
  5. <...>
- **Accept if**: <at least one experiment clearly beats baseline — specific threshold>
- **Reject if**: <all experiments perform at or below baseline — the current setting was already near-optimal>
- **Inconclusive if**: <results are noisy, crashes prevent clean comparison, or best is within noise floor>
- **Priority**: <1-5, 1=highest>

Examples:

### H1: Learning rates are suboptimal
- **Claim**: The default LRs may be too conservative (slow convergence visible in run.log) or too aggressive (oscillation in late training). Testing both directions reveals where the optimum lies.
- **Experiments**:
  1. Halve all LRs (position_lr, feature_lr, etc.)
  2. Double all LRs
  3. 1.5x all LRs
  4. 0.7x all LRs
  5. Best direction from above with LR schedule (cosine decay to 0.1x)
- **Accept if**: Any run beats baseline PSNR by >= 0.1
- **Reject if**: All runs within ±0.05 of baseline — current LRs are near-optimal
- **Inconclusive if**: Mixed results with no clear trend, or crashes obscure the picture
- **Priority**: 1

### H2: Densification schedule is poor
- **Claim**: The densification timing/frequency may be leaving PSNR on the table — too early wastes vertices before gradients are informative, too late leaves no time to train new geometry. The grow/split thresholds may also be miscalibrated.
- **Experiments**:
  1. Start densification 2x earlier (halve densify_from_iter)
  2. Start densification 2x later (double densify_from_iter)
  3. Densify more frequently (halve densify_every)
  4. Densify less frequently (double densify_every)
  5. Adjust grow/split thresholds based on best timing from above
- **Accept if**: Any run beats baseline PSNR by >= 0.1
- **Reject if**: All runs within ±0.05 of baseline
- **Inconclusive if**: Vertex counts vary wildly making comparison unfair, or crashes
- **Priority**: 2

### H3: SSIM loss weight is mistuned
- **Claim**: The SSIM/L1 loss balance affects what the model optimizes for. The current weight may over- or under-emphasize structural similarity vs. pixel accuracy, leaving PSNR on the table.
- **Experiments**:
  1. Increase SSIM weight by 50%
  2. Decrease SSIM weight by 50%
  3. Remove SSIM entirely (pure L1)
  4. Double SSIM weight
  5. Best weight from above + window size adjustment
- **Accept if**: Any run beats baseline PSNR by >= 0.1
- **Reject if**: All runs within ±0.05 of baseline — current balance is near-optimal
- **Inconclusive if**: PSNR improves but SSIM/LPIPS degrade significantly (metric tradeoff, not a clear win)
- **Priority**: 3
```

## journal.md format

`journal.md` is an append-only **hypothesis log**. It records tested hypotheses and their outcomes. This keeps the journal compact and makes patterns across experiments easy to spot.

Each planning session appends a new session header. Each hypothesis is appended when its full experiment series completes (~5 runs).

```markdown
# Hypothesis Journal

## Session: YYYY-MM-DD HH:MM
Best PSNR: <value> | Hypotheses tested: <count>
Key insight: <1 sentence — most important learning from this batch>

### H1: Learning rates are suboptimal — ACCEPT
Baseline: 27.4 | Best: 27.8 (1.5x LRs) | Worst: 26.9 (0.5x LRs)
| Run | Variation | PSNR | Notes |
|-----|----------|------|-------|
| H1.1 | 0.5x all LRs | 26.9 | Underfit, still climbing at end |
| H1.2 | 2x all LRs | 27.5 | Slight improvement, some oscillation |
| H1.3 | 1.5x all LRs | 27.8 | Best result, smooth convergence |
| H1.4 | 0.7x all LRs | 27.2 | Marginal, slow convergence |
| H1.5 | 1.5x + cosine decay | 27.7 | Decay didn't help over flat |
Verdict: Current LRs too conservative. 1.5x applied as new baseline.

### H2: Densification schedule is poor — REJECT
Baseline: 27.8 | Best: 27.85 (less frequent) | Worst: 27.3 (2x earlier)
| Run | Variation | PSNR | Notes |
|-----|----------|------|-------|
| H2.1 | 2x earlier start | 27.3 | Wasted vertices on noise |
| H2.2 | 2x later start | 27.6 | Fewer vertices but cleaner |
| H2.3 | 2x more frequent | 27.5 | Too many small densifications |
| H2.4 | 2x less frequent | 27.85 | Within noise floor |
| H2.5 | Adjusted thresholds | 27.7 | No clear win |
Verdict: Schedule is roughly optimal. No variation beat baseline beyond noise.
```

### Result categories
- **ACCEPT**: At least one experiment clearly beat baseline past the threshold. The best variation is kept and becomes the new baseline.
- **REJECT**: No experiment meaningfully beat baseline — the current setting is near-optimal. Revert all.
- **INCONCLUSIVE**: Results too noisy, too many crashes, or conflicting signals. Revert, note for potential revisit with tighter controls.

### Between planning sessions
As each hypothesis series completes, append its full block (header + table + verdict) to `journal.md`.

---

# Autonomous Execution & Syntax Constraints

You are operating in a continuous optimization loop. To ensure commands execute without requiring manual user approval, you MUST adhere to the following strict shell syntax rules:

1. **Sequential Execution**: Run commands one at a time. NEVER use command chaining (`&&`, `||`, `;`). 
2. **No Subshells**: NEVER use subshells (`$()`) or backticks inside terminal commands.
3. **No Heredocs**: NEVER use `<<EOF` or similar multi-line string inputs in the terminal.
4. **No Piping or Redirection**: NEVER use `|`, `>`, or `<`. If you need to read `run.log`, use your built-in file reading capabilities instead of `cat`, `sed`, or `grep`.
5. **Simple Git Commits**: For commits, use a single-line message: `git commit -m "Short message"`. If a multi-line message is absolutely required, write the text to a temporary file using the file system tool, then execute `git commit -F temp_msg.txt`.
6. **No Process Substitution Characters**: NEVER use the character sequences `>(` or `<(` anywhere in your commands, including inside quotes or commit messages. If logging parameter changes, use words instead of arrows (e.g., write "clamp 0.5 to 0.3" instead of "clamp ->(0.3)").
7. **No Inline Bash Loops**: NEVER use `while`, `for`, or command chaining (`;`) in the terminal to wait for a process or poll logs. 
8. **Scripted Polling**: To poll `run.log` for metrics, write the logic to a standalone script (e.g., `wait_for_metrics.sh`), make it executable, and run it with a simple `./wait_for_metrics.sh`.
