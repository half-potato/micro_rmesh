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
5. **Run the baseline**: Your very first run establishes the baseline. Run `uv run train.py` as-is and record the result.
6. **Confirm and go**: Confirm setup looks good with the user.

Once you get confirmation, begin the first shift.

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

**The goal is to get the highest PSNR and to deeply understand *why* each change helps or hurts.** Since the time budget is fixed, you don't need to worry about training time — it's always 10 minutes. Everything is fair game.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful PSNR gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

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

**CRITICAL: `results.tsv` is in `.gitignore` and must NEVER be staged or committed. Before any `git add`, verify results.tsv is not being staged.**

---

## The Research Process

You are an autonomous researcher. Your work is organized into **shifts** — focused work sessions with a clear deliverable. The user will discuss goals with you before each shift, and you'll produce a **discovery report** at the end.

### Pre-shift planning (with user)

Before each shift, discuss:
- What are the most interesting open questions about this system?
- What did we learn from the previous shift?
- What 2-3 things do we want to understand today?

The user may also just say "go" — in that case, identify your own questions based on the current state.

### During a shift

Your work follows four steps. These aren't rigid phases — you'll cycle through them naturally as you investigate. But every piece of work you do should fit into one of these:

#### 1. OBSERVE — Find something surprising or unexplained

Read `run.log` as a story, not a metric dump. Read the code paths that actually execute. Find the gap between what you expected and what happened.

- **PSNR convergence curve**: Are there unexpected drops? Is PSNR still climbing at the end? Does it plateau?
- **Densification events**: How much PSNR drops after densification and how long recovery takes is often the most important signal.
- **Vertex trajectory**: Is the model using the vertex budget effectively?
- **Code reading**: Trace through the actual code paths. Understand *what the code does*, not just what the parameters control.

The goal of observation is **a question**, not an answer. "Why does PSNR drop 5dB after densification?" is a good observation. "PSNR is 20.7" is just a number.

#### 2. HYPOTHESIZE — Explain *why*, not *what value*

Form a mechanistic hypothesis about the system's behavior. This is NOT "parameter X might be too high." It IS "orphan tets inherit high density from neighbors, creating opaque floaters that corrupt the rendered image."

A good hypothesis:
- Explains a *mechanism* — what's actually happening in the code
- Is *falsifiable* — you can design an experiment that would prove it wrong
- Suggests a *specific intervention* — not just "tune X" but "change how X works"

#### 3. TEST — Design experiments that distinguish between explanations

This is where you run experiments, but the experiments serve your understanding, not a parameter grid.

- **Sometimes the right experiment is adding instrumentation** — logging per-tet error, printing intermediate values, measuring timing of individual operations. Not every experiment needs to improve PSNR.
- **Sometimes it's an algorithmic change** — rewriting how orphan transfer works, changing the densification strategy, modifying the loss function structure.
- **Sometimes it's a parameter sweep** — but only when you have a specific reason ("I think the LR is too high because I see oscillation in the convergence curve").
- **Build iteratively.** Don't git-reset after every experiment. Work on a development branch. If an idea needs debugging and refinement, keep building on it. Only reset if you're abandoning the entire line of investigation.
- **Batch experiments when possible.** If you need to test 5 parameter values, write a script or use background tasks. Don't sit idle while the GPU works — read code, analyze previous results, plan the next investigation.
- **Verify results.** Run the baseline multiple times to understand noise. A result within ±0.1 of baseline is noise, not signal. Don't chase noise.

#### 4. UPDATE — Record what you learned

After each investigation, write down:
- **What question did you investigate?**
- **What did you learn?** (The answer might be "this doesn't matter" — that's still a discovery.)
- **What evidence supports this?**
- **What's the next question this raises?**

Update `journal.md` with your findings. Update `results.tsv` with experiment data. Commit your best changes to the branch.

### End of shift — Discovery Report

At the end of the shift, write a discovery report to `journal.md`. This is the primary deliverable. It should cover:

1. **Questions investigated**: What did you set out to understand?
2. **Key discoveries**: What did you learn about how the system works? These should be mechanistic insights, not just "X=4 is better than X=3."
3. **Evidence**: What experiments support each discovery?
4. **Current best config**: What's the PSNR and what changes are applied?
5. **Open questions**: What should the next shift investigate?
6. **Failed ideas and why**: What didn't work and what does that tell us?

### What makes a good shift

A good shift produces **understanding that compounds**. Discoveries from one shift should make the next shift more productive. Bad shifts produce lots of data but no insight — 50 experiments that all say "within noise" teaches nothing.

Prefer depth over breadth. One deep investigation that reveals *why* densification causes a 5dB PSNR drop is worth more than 20 parameter sweeps that each say "no improvement."

### Autonomy

You are autonomous during a shift. Do NOT pause to ask the user if you should continue or if your approach is right. Make judgment calls. If an investigation dead-ends, pivot to something more promising. If you discover something unexpected, follow it.

The user might be away during the shift. They expect to come back to a discovery report, not a list of questions.

---

## journal.md format

`journal.md` is an append-only **discovery journal**. It records what you investigated and what you learned.

```markdown
# Discovery Journal

## Shift: YYYY-MM-DD
Goals: <what questions we set out to answer>

### Investigation: <title>
**Question**: <what you wanted to understand>
**Hypothesis**: <your mechanistic explanation>
**Method**: <what you did — code changes, instrumentation, experiments>
**Results**: <what happened — include PSNR numbers but also qualitative observations>
**Discovery**: <what you learned — the insight, not just the number>
**Next question**: <what this raises>

### Investigation: <title>
...

### End-of-shift summary
- Best PSNR: <value> (from <description>)
- Key discoveries: <bulleted list>
- Open questions for next shift: <bulleted list>
```

---

# Autonomous Execution & Syntax Constraints

You are operating autonomously. To ensure commands execute without requiring manual user approval, you MUST adhere to the following strict shell syntax rules:

1. **Sequential Execution**: Run commands one at a time. NEVER use command chaining (`&&`, `||`, `;`).
2. **No Subshells**: NEVER use subshells (`$()`) or backticks inside terminal commands.
3. **No Heredocs**: NEVER use `<<EOF` or similar multi-line string inputs in the terminal.
4. **No Piping or Redirection**: NEVER use `|`, `>`, or `<`. If you need to read `run.log`, use your built-in file reading capabilities instead of `cat`, `sed`, or `grep`.
5. **Simple Git Commits**: For commits, use a single-line message: `git commit -m "Short message"`. If a multi-line message is absolutely required, write the text to a temporary file using the file system tool, then execute `git commit -F temp_msg.txt`.
6. **No Process Substitution Characters**: NEVER use the character sequences `>(` or `<(` anywhere in your commands, including inside quotes or commit messages. If logging parameter changes, use words instead of arrows (e.g., write "clamp 0.5 to 0.3" instead of "clamp ->(0.3)").
7. **No Inline Bash Loops**: NEVER use `while`, `for`, or command chaining (`;`) in the terminal to wait for a process or poll logs.
8. **Scripted Polling**: To poll `run.log` for metrics, write the logic to a standalone script (e.g., `wait_for_metrics.sh`), make it executable, and run it with a simple `./wait_for_metrics.sh`.
