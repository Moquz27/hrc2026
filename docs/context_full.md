

HRC Project Workflow & Environment Specification
This project uses a fixed two-machine workflow and it must not be changed unless there is a strong technical reason.

1. Roles of Each Machine
Mac (Development Machine)
The Mac is the primary development machine. It is used for:
Writing code
Reading the repository
Using Codex to analyze the codebase
Fixing logic
Reviewing diffs
Running lightweight tests
Committing and pushing code

Linux (Runtime Machine)
The Linux machine is the primary runtime environment. It is used for:
Running the actual repository
Storing the full dataset
Storing simulation assets
Storing checkpoints
Storing outputs and logs
Running Isaac Sim
Running evaluation and benchmarks
Running batch tests
Producing and validating real results

2. Core Principle
Mac = Development and orchestration
Linux = Execution and validation
Strict rules:
The Mac must NOT be used to run heavy simulations
The Mac must NOT store the full dataset
Dataset, checkpoints, and heavy outputs must NOT be frequently synced between machines
Only code, configs, documentation, and small text files are synced via Git

3. Directory Structure
On Mac
~/hrc-dev/hrc2026

Contains source code only
May include a local .venv for lightweight testing
This environment is NOT the production/runtime environment

On Linux
Code remains in:
~/hrc2026/repo

Runtime assets live under:
~/hrc-runtime/

Structure:
~/hrc2026/repo                # existing runnable code repository
~/hrc-runtime/data            # dataset + simulation assets
~/hrc-runtime/checkpoints     # model weights
~/hrc-runtime/outputs         # metrics, replay, video, results
~/hrc-runtime/logs            # logs

Optional deeper structure:
data/
  ├── raw/
  └── processed/

outputs/
  ├── metrics/
  └── replays/


4. Git Rules
Heavy files must NEVER be committed:
dataset
.pt, .ckpt
videos
replays
output images
cache
build artifacts
.gitignore must include at least:
.venv/
__pycache__/
*.pyc
.env
.env.local
data/
checkpoints/
outputs/
logs/
*.pt
*.ckpt
*.mp4
*.mov
*.png
*.jpg
.DS_Store


5. Path Handling (Critical Rule)
Code must NEVER hardcode local paths such as:
/Users/...
/home/...

All paths must be read from environment variables.

6. Environment Variables (Linux Only)
Add to ~/.bashrc:
export HRC_ROOT=$HOME/hrc-runtime
export HRC_REPO=$HOME/hrc2026/repo
export DATA_ROOT=$HOME/hrc-runtime/data
export CKPT_ROOT=$HOME/hrc-runtime/checkpoints
export OUTPUT_ROOT=$HOME/hrc-runtime/outputs
export LOG_ROOT=$HOME/hrc-runtime/logs

Usage in code:
import os

data_path = os.environ["DATA_ROOT"]
ckpt_path = os.environ["CKPT_ROOT"]
output_path = os.environ["OUTPUT_ROOT"]
log_path = os.environ["LOG_ROOT"]

Or via config mapping.
Goal:
The same code must run on both Mac and Linux without modification

7. Repository Structure
Minimum required structure:
src/
scripts/
configs/
tests/
docs/

AGENTS.md
PROJECT_CONTEXT.md
TASK_LOG.md
.gitignore
requirements.txt

Optional extension:
src/perception
src/planner
src/controller
src/evaluator
However:
Do NOT refactor heavily early unless there is benchmark evidence.

8. Required Project Files
AGENTS.md
Rules for Codex and all agents:
Must include:
Always read PROJECT_CONTEXT.md first
Do not refactor unrelated files
Keep changes minimal and testable
Prioritize stability over speed
Do not hardcode local paths
Update TASK_LOG.md after meaningful changes

PROJECT_CONTEXT.md
Contains:
Two-machine workflow
Competition assumptions
Simulator version
Current strategy
Active technical decisions

TASK_LOG.md
Progress log:
What has been tried
What errors occurred
What fixes were applied
Benchmark results
Next steps

9. Daily Workflow
On Mac
Open repo at:
~/hrc-dev/hrc2026


Read PROJECT_CONTEXT.md
Edit code (editor or Codex)
Run lightweight tests or smoke tests
Commit and push:
git add .
git commit -m "..."
git push


On Linux
SSH from Mac:
ssh hrc-linux

Go to repo:
cd ~/hrc2026/repo
git pull

Activate environment:
source .venv/bin/activate

Run tests:
pytest

Run simulation / evaluation:
bash scripts/run_task.sh


Output Locations
~/hrc-runtime/logs
~/hrc-runtime/outputs/metrics
~/hrc-runtime/outputs/replays


After Execution
Inspect logs, metrics, replay
Write summary into TASK_LOG.md
Return to Mac and continue iteration

10. Standard Loop
Edit on Mac
→ Push to Git
→ Pull on Linux
→ Run on Linux
→ Inspect logs/metrics
→ Iterate


11. Git Usage Rules
Git is the ONLY official sync mechanism
Never use Git for:
dataset
checkpoints
heavy outputs

12. Remote Development
Mac must have SSH key access to Linux
Define alias in ~/.ssh/config (e.g. hrc-linux)
Can use:
SSH terminal
Remote editor
But:
Linux is always the runtime source of truth

13. Codex Usage
Install Codex CLI on both machines via npm
Use Codex CLI inside repo for coding
Codex app (if used):
Mac only
for review, diff, workflow organization
Linux:
prefer terminal / CLI

14. Strategy Rules
Build baseline first
Optimize later
Core principle:
Baseline stable first, optimize later
Strict constraints:
No early large refactor
Changes must be:
easy to debug
easy to rollback
clearly logged

15. Testing Rules
Heavy tests → Linux only
Batch evaluation → Linux only
Simulation → Linux only
Mac:
only for lightweight testing
Results on Mac are NOT final ground truth

16. Dataset Handling
Never copy full dataset to Mac
Only small subsets for debugging

17. Simulation Environment
Isaac Sim must be installed outside the repo
Simulator version must be documented in PROJECT_CONTEXT.md
Do NOT change simulator version arbitrarily

18. Final Summary
Project memory model:
Mac:
~/hrc-dev/hrc2026
→ coding, Codex, lightweight testing, Git push
Linux:
~/hrc2026/repo
~/hrc-runtime/data
~/hrc-runtime/checkpoints
~/hrc-runtime/outputs
~/hrc-runtime/logs
→ full runtime, data, evaluation

19. Core System Design Rules
Code must be path-agnostic via environment variables
Repo must include:
AGENTS.md
PROJECT_CONTEXT.md
TASK_LOG.md

20. Final Workflow Rule
Edit on Mac → Sync via Git → Run on Linux → Inspect → Iterate
