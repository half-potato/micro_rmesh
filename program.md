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
6. **Initial planning phase**: Before running the baseline, analyze the architecture for optimization opportunities. Read all in-scope files carefully and write an initial experiment plan to `plan.md` (see the **plan.md format** below) with the first 5–10 experiment ideas ranked by expected impact. Append the summary to `journal.md` (see the **journal.md format** below). **Both `plan.md` and `journal.md` must stay untracked by git** — like `results.tsv`, they must never be staged, committed, or affected by git resets. Verify they are in `.gitignore`.
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

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py, model.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^test_PSNR:\|^n_vertices:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv. **CRITICAL: `results.tsv` is in `.gitignore` and must NEVER be staged or committed. It is the persistent experiment log that must survive all git resets. Before any `git add`, verify results.tsv is not being staged.**
8. If PSNR improved (higher is better), you "advance" the branch, keeping the git commit
9. If PSNR is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~10 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~10 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

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

Overwrite `plan.md` with the new plan using the **plan.md format** below.

### Step 5: Append to journal.md

Read the `## Summary` section from `plan.md` and append it as a new entry in `journal.md` using the **journal.md format** below.

### Step 6: Resume

Continue the experiment loop with the top-ranked experiment from the plan.

---

## plan.md format

`plan.md` is a structured, overwritable working document. Overwrite it each planning session:

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

## Next Experiments
1. <idea> — rationale: <why, based on analysis above>
2. <idea> — rationale: <why>
3. <idea> — rationale: <why>
4. ...

## Summary
<2-3 sentence summary for journal>
```

## journal.md format

`journal.md` is an append-only log. Each planning session appends a new section:

```markdown
# Experiment Journal

---
### YYYY-MM-DD HH:MM — Planning Session
Best PSNR: <value> | Experiments: <count>
<summary from plan.md>
Next: <top 3 planned experiments>
```

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
