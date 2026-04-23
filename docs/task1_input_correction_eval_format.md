# Task 1 Input Correction Evaluation Format

This format is for the `test-camera-kiet` input-correction experiment only.
It evaluates whether AI output improves input data quality before any full
runtime integration.

This is not Thinker runtime control, not final grasp-pose generation, and not
motion execution.

## Scope

Allowed AI correction targets:

- `selected_object_id` / `object_id`
- `class`
- `center_2d`
- `roi`
- `orientation_bucket`
- `recommended_arm`
- `recommended_preset`

Forbidden AI output is ignored by the correction layer:

- final 3D grasp pose
- 6D target pose
- world/base/table 3D position overwrite
- joint commands
- motion commands
- execution phases
- trajectories or waypoints

Simulator truth is never changed.

## Prepared Case Input

The runner accepts either a JSON object with `cases` or a JSONL file with one
case object per line.

```json
{
  "format_version": "0.1.0",
  "cases": [
    {
      "case_id": "case_000",
      "sample_id": "sample_000000",
      "original_input": {
        "selected_object_id": "task1_part_001",
        "class": "B",
        "center_2d": [120.0, 80.0],
        "roi": [100.0, 60.0, 140.0, 100.0],
        "orientation_bucket": "front",
        "recommended_arm": "right",
        "recommended_preset": "topdown"
      },
      "ai_output": {
        "selected_object_id": "task1_part_000",
        "class": "A",
        "center_2d": [110.0, 75.0],
        "roi": [90.0, 55.0, 130.0, 95.0],
        "orientation_bucket": "left",
        "recommended_arm": "left",
        "recommended_preset": "side_approach",
        "confidences": {
          "selected_object_id": 0.92,
          "class": 0.90,
          "center_2d": 0.88,
          "roi": 0.87,
          "orientation_bucket": 0.86,
          "recommended_arm": 0.85,
          "recommended_preset": 0.84
        }
      },
      "truth": {
        "selected_object_id": "task1_part_000",
        "class": "A",
        "center_2d": [109.0, 74.0],
        "roi": [89.0, 54.0, 129.0, 94.0],
        "orientation_bucket": "left",
        "recommended_arm": "left",
        "recommended_preset": "side_approach"
      },
      "allowed_object_ids": [
        "task1_part_000",
        "task1_part_001",
        "task1_part_002",
        "task1_part_003"
      ]
    }
  ]
}
```

Confidence can be provided in `confidences.<field>` or as field-specific keys
such as `class_confidence`, `center_2d_confidence`, `roi_confidence`,
`orientation_bucket_confidence`, `recommended_arm_confidence`, and
`recommended_preset_confidence`.

## Generated Cases

When `--cases-json` is omitted, `scripts/task1_run_input_correction_eval.py`
can generate deterministic 10-case inputs from a Phase 1 collected run:

```bash
python3 scripts/task1_run_input_correction_eval.py \
  --run-id test_phase1_initfix_1 \
  --limit 10
```

Generated cases use Phase 1 labels as truth and deterministic synthetic
original/AI estimates to test gating and metric plumbing. These synthetic AI
outputs are not real Thinker predictions and must not be reported as runtime
AI performance.

For generated cases, `recommended_arm` and `recommended_preset` truth fields
are deterministic evaluator references derived from table position and coarse
orientation. They are not simulator-truth control labels.

## Output Layout

Outputs are written outside the repo, normally under:

```text
$OUTPUT_ROOT/test_runs/task1_input_correction_eval/<run_label>_<timestamp>/
  summary.json
  cases/
    case_000.json
    case_001.json
```

Each case log contains:

- `case_id`
- `original_input`
- `ai_output`
- `corrected_input`
- `truth`
- `correction_decisions`
- `confidence_values`
- `ignored_forbidden_fields`
- `metrics.before`
- `metrics.after`
- `before_after_deltas`

`summary.json` contains:

- number of accepted AI corrections
- number of rejected AI corrections
- number of cases improved
- number of cases unchanged
- number of cases worsened
- before/after correctness aggregates
- before/after mean 2D center error

## Pass/Fail Meaning

This experiment only measures input quality:

- class correctness
- selected-object correctness
- 2D center error
- orientation bucket correctness
- arm recommendation correctness
- preset recommendation correctness

It does not measure grasp success, task success, physical execution, or final
competition runtime behavior.
