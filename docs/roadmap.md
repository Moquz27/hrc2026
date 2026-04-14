# HRC 2026 Phase Plan

## Phase 0: Repo & Workflow Setup
Status: Done

Goal:
- Clean repo skeleton
- Git sync between Mac and Linux
- Runtime path rules
- Ignore heavy files and generated artifacts

Scope:
- Repository structure only
- Workflow and collaboration rules only
- No simulation logic

Done:
- Created repo structure
- Added AGENTS.md, PROJECT_CONTEXT.md, TASK_LOG.md
- Added .gitignore
- Defined Mac/Linux workflow

Exit Criteria:
- Repo structure is stable
- Git workflow is usable across Mac and Linux
- Heavy runtime artifacts are excluded from version control

## Phase 1: Isaac Runtime Smoke
Status: Done

Goal:
- Verify Linux Isaac Sim can run repo code
- Validate required environment variables
- Write runtime logs outside repo

Scope:
- Isaac runtime smoke only
- No task logic
- No control logic
- No learning or perception

Done:
- Added scripts/smoke_isaac.py
- Preflight passed
- Isaac smoke passed

Exit Criteria:
- Isaac runtime starts correctly from project workflow
- Required env vars are validated
- Logs can be written outside the repo

## Phase 2: Walker S2 Load Baseline
Status: Done

Goal:
- Load Walker S2 USD from runtime assets
- Detect articulation and joints
- Avoid task/control/perception logic

Scope:
- Robot asset load and structural validation only
- No control
- No manipulation
- No task scenes

Done:
- Added scripts/load_walker_s2.py
- Added Git LFS pointer detection
- Walker S2 loaded successfully
- Detected articulation root and 42 joints

Known Issues / Risks:
- Joint state read warning remains non-blocking

Exit Criteria:
- Real Walker S2 asset payload is present
- Robot loads in Isaac Sim
- Articulation root and joints are discoverable
- Structural inspection is stable enough to move to control validation

## Phase 3: Competition Stack Integration & Validation
Status: Next

Goal:
- Make the full competition stack run end-to-end at integration level
- Validate official robot, assets, baseline, dataset access, and motion sanity
- Reach a state where the system is fully debuggable before optimization

Scope:
- Walker S2 in Isaac Sim
- Official assets / scenes / task environments
- Official baseline repo inspection and smoke use
- Dataset inspection and practical evaluation
- Basic motion command and primitive tests
- Competition-like smoke tests
- No serious model optimization yet

Not Included:
- Final task-solving policies
- Aggressive ML optimization
- Large refactors
- Submission packaging
- Performance chasing before integration is stable

### 3.1 Workspace and Official Resource Setup
Goal:
- Standardize the runtime workspace and download all official resources correctly

Required resources:
- WalkerS2-Model-Challenge for USD/URDF/STL robot assets
- challenge2026_assets for scenes and task objects
- GlobalHumanoidRobotChallenge_2026_Baseline for reference pipeline
- challenge2026_dataset for inspection of available data
- Working Isaac Sim runtime

Success targets:
- No missing official repo
- No temporary or ambiguous path layout
- Git LFS payloads are real, not pointer files
- Robot / assets / baseline / dataset locations are explicit

Why it matters:
- If 3.1 is not clean, later debugging will mix path errors with simulation errors

### 3.2 Minimal Robot Control Validation
Goal:
- Prove the robot is controllable, not just loadable

Scene limits:
- Robot only
- Ground or very simple table
- No conveyor
- No foam
- No carton
- No task objects

Checks:
- Repeated reset stability
- Per-joint or per-group command checks
- Waist response
- Left/right arm response
- Gripper open/close
- No obvious physics explosion, violent jitter, or severe frame mismatch

Why it matters:
- "Model loads" and "robot is controllable" are different milestones

### 3.3 Official Scene Loading Validation
Goal:
- Verify official competition scenes and task assets load correctly

Must validate loading of:
- table
- material box
- conveyor
- foam
- carton
- workpieces A/B as applicable

Success targets:
- Scene loads cleanly
- Robot spawns correctly
- Objects are present
- No missing asset dependencies
- No broken resource mapping
- Repeated reset is possible

Important:
- 3.3 does not solve tasks
- It only proves the official scene stack is alive

### 3.4 Motion Smoke Test in Competition-like Scenes
Goal:
- Test basic robot motion inside official or near-official scenes

Example checks:
- Reach a point near an object
- Move end-effector from A to B
- Open/close gripper near objects
- Approach carton flap
- Bring end-effector near foam hole
- Move near conveyor object without attempting full pick

Validation focus:
- Frame correctness
- Controller stability in real scenes
- Obvious collision problems
- Basic contact feasibility
- Whether scene complexity breaks motions that worked in minimal scenes

Important:
- Failures here are still integration/runtime failures, not model-quality failures

### 3.5 Baseline Architecture Inspection
Goal:
- Map the real baseline architecture before using or modifying it

Must inspect:
- env entrypoints
- observation structure
- action format
- reset logic
- task config locations
- scoring / success logic
- scene path and asset root mapping
- dataset schema assumptions inside baseline

Why it matters:
- The goal is not blind compliance with baseline
- The goal is to understand what can be reused and what may later be replaced

### 3.6 Dataset Inspection and Practical Evaluation
Goal:
- Evaluate the real usefulness of available dataset subsets

Must inspect:
- Which tasks currently have data
- Number of episodes per subset
- Observation keys and action keys
- Data cleanliness
- Variation level
- Whether the data is enough for:
  - smoke validation
  - narrow baseline imitation
  - robust competition policy

Important:
- Do not assume available dataset automatically implies strong policy training

### 3.7 Per-Task Primitive Tests
Goal:
- Probe each task at primitive level before attempting full task solutions

Primitive checks:
- Task 1: reach / pick-like primitive on table
- Task 2: tracking and intercept prediction on moving conveyor objects
- Task 3: alignment primitive near foam insertion target
- Task 4: carton flap contact and simple fold motion

Why it matters:
- This reveals the true difficulty of each task before algorithm design

### 3.8 Initial Strategy Decision per Task
Goal:
- Produce a concrete initial technical direction for each task

Decision outputs should cover:
- Which task should start with scripted / FSM logic
- Which task may use hybrid perception + control
- Which task may benefit from imitation learning
- Which task may need data augmentation or self-collected demos

Important:
- The output of 3.8 must be a clear engineering decision, not vague notes

### Exit Criteria for Phase 3
Phase 3 is complete only when all of the following are true:
- Walker S2 loads stably
- Official assets and scenes load correctly
- Baseline repo is usable at inspect/smoke-test level
- Dataset has been inspected and evaluated at a practical level
- Robot can perform basic motions in a minimal scene
- Robot can perform primitive motions in official scenes
- Repeated reset does not break the stack
- There is a clear initial technical direction for each task

Summary:
Phase 3 ends when the competition stack is truly running and understood well enough to begin optimization.

## Phase 4: Algorithm & ML Optimization
Status: Future

Goal:
- Optimize task performance only after Phase 3 is fully passing
- Improve success rate, speed, stability, and competition score
- Turn the validated stack into a competitive solution

Scope:
- Per-task algorithm design
- Controller / state machine / planner refinement
- ML / imitation learning using available datasets
- Additional data collection if needed
- Perception improvements
- Benchmarking, ablation, and stress testing
- Submission preparation

Not Included:
- Basic path fixing
- Missing asset recovery
- Frame debugging
- First-time control validation
- Scene reset debugging

### 4.1 Task-by-Task Practical Strategy Selection
Goal:
- Finalize the initial practical solving direction per task

Examples to evaluate:
- Task 1: perception + stable scripted pick-place
- Task 2: tracking + timing + intercept logic
- Task 3: precise alignment, possibly hybrid
- Task 4: phase-based dual-arm carton controller, possibly without heavy ML

Important:
- These are candidate directions, not hardcoded facts

### 4.2 Build a Usable Baseline Policy from Available Data
Goal:
- Train something that actually runs, not something that only looks good offline

Must answer:
- Which dataset is used for which task
- Which subsets are only bootstrap references
- Which tasks need extra demo collection
- Which tasks should not depend heavily on currently available dataset

### 4.3 Optimize for Competition Metrics
Goal:
- Optimize against actual evaluation behavior

Primary metrics:
- success rate
- wrong class / wrong placement / missed object rate
- completion time
- multi-episode stability
- tolerance to randomization

### 4.4 Stress Testing and Submission Preparation
Goal:
- Harden the solution before submission

Must include:
- many seeds
- many resets
- structured logging
- edge-case fixes
- packaging model/source in submission-ready form

## Condensed Project Memory
Phase 3 = Integration & validation
- 3.1 official resources and workspace
- 3.2 minimal robot control
- 3.3 official scene loading
- 3.4 motion smoke tests in competition-like scenes
- 3.5 baseline architecture inspection
- 3.6 dataset inspection
- 3.7 per-task primitive tests
- 3.8 initial strategy decision

Phase 4 = Optimization
- task strategy refinement
- controller / ML optimization
- data usage and extra collection if needed
- benchmark, stress test, submission preparation
