# Task 1 Thinker4B Recorded-Camera Input Evaluation

This workflow evaluates Thinker4B on recorded Phase 1 camera inputs only.
It does not run robot execution, generate final grasp poses, call IK, modify
the planner, or integrate with runtime motion.

## Entrypoint

```bash
python3 scripts/task1_run_thinker4b_input_eval.py \
  --run-id test_phase1_initfix_1 \
  --seeds 1,2,3,4,5 \
  --cases-per-seed 10 \
  --output-dir "$OUTPUT_ROOT/test_runs/task1_thinker4b_input_eval/test_phase1_initfix_1_thinker4b_5x10" \
  --report-path docs/output01.txt
```

The command above requires a configured Thinker4B provider. Without one, add
`--allow-provider-failure` only to test selection, logging, metrics, and report
generation without fake model output.

## Providers

OpenAI-compatible HTTP server:

```bash
export THINKER4B_API_BASE=http://localhost:8000/v1
export THINKER4B_MODEL=Thinker4B
export THINKER4B_API_KEY=optional_key
python3 scripts/task1_run_thinker4b_input_eval.py --provider openai-compatible
```

Ollama-compatible local server:

```bash
export OLLAMA_HOST=http://localhost:11434
export THINKER4B_MODEL=thinker4b
python3 scripts/task1_run_thinker4b_input_eval.py --provider ollama
```

Command provider:

```bash
export THINKER4B_CMD='python3 /path/to/thinker4b_wrapper.py'
python3 scripts/task1_run_thinker4b_input_eval.py --provider command
```

The command provider receives JSON on stdin containing a `request_path`. That
file contains the prompt, case metadata, and PNG base64 camera images. The
command must print one JSON object to stdout.

Cached provider:

```bash
python3 scripts/task1_run_thinker4b_input_eval.py \
  --provider cache \
  --thinker-output-cache "$OUTPUT_ROOT/test_runs/my_real_thinker_outputs.jsonl"
```

The cache is for replaying previously generated real Thinker4B outputs, not
for synthetic outputs.

## Output

Runtime outputs are written outside the repo:

```text
$OUTPUT_ROOT/test_runs/task1_thinker4b_input_eval/<run_name>/
  summary.json
  cases.jsonl
  cases/
    seed_1_case_00.json
```

The human-readable report is written to:

```text
docs/output01.txt
```

That report includes every case, not just the aggregate summary.

## Current Limitation

The current environment did not expose `THINKER4B_API_BASE`,
`THINKER4B_MODEL`, `THINKER4B_CMD`, an Ollama Thinker4B server, or cached real
Thinker outputs. The checked-in `docs/output01.txt` therefore records the full
50-case workflow with Thinker4B provider failures, not real Thinker4B
performance.
