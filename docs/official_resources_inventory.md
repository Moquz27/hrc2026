# Official Resources Inventory

Last updated: 2026-04-14

Runtime source of truth:
- `HRC_ROOT`
- `DATA_ROOT`
- `LOG_ROOT`

No official resource is stored inside the code repo.

## Verification Summary

Initial state before download:
- Walker S2 model repo existed under `HRC_ROOT/assets` and was not recloned.
- Baseline repo was missing.
- Official challenge assets were missing.
- Official dataset was missing.

Downloaded during this pass:
- `HRC_ROOT/baseline/GlobalHumanoidRobotChallenge_2026_Baseline`
- `HRC_ROOT/assets/challenge2026_assets`
- `DATA_ROOT/challenge2026_dataset`

Validation logs:
- `LOG_ROOT/official_resources_inventory.log`
- `LOG_ROOT/official_scene_smoke.log`

## Resource Status

| Resource | Local Path | Size | Status | Notes |
| --- | --- | ---: | --- | --- |
| Baseline repo | `$HRC_ROOT/baseline/GlobalHumanoidRobotChallenge_2026_Baseline` | 2.0M | OK | Contains `run.sh`, `Ubtech_sim/main.py`, task config YAMLs, and simulator source files. |
| Walker S2 model | `$HRC_ROOT/assets/WalkerS2-Model-Challenge` | 557M | OK | Existing repo verified first; root USD payload is real, not an LFS pointer. |
| Official assets | `$HRC_ROOT/assets/challenge2026_assets` | 2.5G | OK | USD resources present and no LFS pointer files detected. |
| Official dataset | `$DATA_ROOT/challenge2026_dataset` | 1.7G | OK | Contains `Packing_box/box_closing_001` parquet episodes and metadata. |

Filesystem validation:
- No Git LFS pointer files were detected in the verified resources.
- Official assets and dataset are large enough to be real payloads, not placeholder clones.

USD / Isaac validation:
- `scripts/load_official_scene_smoke.py` loaded
  `$HRC_ROOT/assets/challenge2026_assets/resources/Collected_table_v2/table_v2.usd`
  together with Walker S2.
- The smoke log reports `status=official_scene_smoke_ok`.
- Stage evidence: `/World/OfficialScene` had 11 prims, Walker S2 had 260 prims,
  articulation root was `/World/WalkerS2/base_link`, and joint count was 42.
- No unresolved dependencies were reported by the smoke script.
- Isaac emitted non-fatal Walker S2 USD warnings about material binding scope,
  one non-existent collision mesh path, and one corrupted normal primvar. These
  did not prevent stage load, articulation discovery, or joint discovery, but
  they should be treated as known asset warnings.

## Candidate Entrypoints

Baseline:
- `$HRC_ROOT/baseline/GlobalHumanoidRobotChallenge_2026_Baseline/run.sh`
- `$HRC_ROOT/baseline/GlobalHumanoidRobotChallenge_2026_Baseline/Ubtech_sim/main.py`
- `$HRC_ROOT/baseline/GlobalHumanoidRobotChallenge_2026_Baseline/Ubtech_sim/config/Part_Sorting.yaml`
- `$HRC_ROOT/baseline/GlobalHumanoidRobotChallenge_2026_Baseline/Ubtech_sim/config/Conveyor_Sorting.yaml`
- `$HRC_ROOT/baseline/GlobalHumanoidRobotChallenge_2026_Baseline/Ubtech_sim/config/Foam_Inlaying.yaml`
- `$HRC_ROOT/baseline/GlobalHumanoidRobotChallenge_2026_Baseline/Ubtech_sim/config/Packing_Box.yaml`

Walker S2:
- `$HRC_ROOT/assets/WalkerS2-Model-Challenge/WalkerS2-Model-Challenge/s2_v1.usd`
- `$HRC_ROOT/assets/challenge2026_assets/resources/Collected_s2_v1_ecbg/s2_v1.usd`

Task/object assets:
- Workpiece A: `$HRC_ROOT/assets/challenge2026_assets/resources/Task1_PartA.usd`
- Workpiece A collected variants:
  `$HRC_ROOT/assets/challenge2026_assets/resources/Collected_Task1_PartA_red/Task1_PartA.usd`,
  `$HRC_ROOT/assets/challenge2026_assets/resources/Collected_Task1_PartA_ori_color/Task1_PartA.usd`
- Workpiece B:
  `$HRC_ROOT/assets/challenge2026_assets/resources/Part_B.usd`
- Workpiece B collected variants:
  `$HRC_ROOT/assets/challenge2026_assets/resources/Collected_Part_B_red/Part_B.usd`,
  `$HRC_ROOT/assets/challenge2026_assets/resources/Collected_Part_B_blue/Part_B.usd`,
  `$HRC_ROOT/assets/challenge2026_assets/resources/Collected_Part_B_ori_color/Part_B.usd`
- Conveyor:
  `$HRC_ROOT/assets/challenge2026_assets/resources/ConveyorBelt.usd`,
  `$HRC_ROOT/assets/challenge2026_assets/resources/Collected_ConveyorBelt/ConveyorBelt.usd`,
  `$HRC_ROOT/assets/challenge2026_assets/resources/Collected_ConveyorBelt_New/Collected_ConveyorBelt/ConveyorBelt.usd`
- Table:
  `$HRC_ROOT/assets/challenge2026_assets/resources/Collected_table_v2/table_v2.usd`
- Bins / cartons / boxes:
  `$HRC_ROOT/assets/challenge2026_assets/resources/Box/box_60_40_23_cut_0.usd`,
  `$HRC_ROOT/assets/challenge2026_assets/resources/Box_blue/box_60_40_23_cut_0.usd`,
  `$HRC_ROOT/assets/challenge2026_assets/resources/Box_gray/box_60_40_23_cut_0.usd`,
  `$HRC_ROOT/assets/challenge2026_assets/resources/Box_blank/box_60_40_23_cut_0.usd`
- Foam / insertion:
  `$HRC_ROOT/assets/challenge2026_assets/resources/Collected_foam/foam.usd`,
  `$HRC_ROOT/assets/challenge2026_assets/resources/Collected_foam/Box_with_foam.usd`,
  `$HRC_ROOT/assets/challenge2026_assets/resources/Collected_foam_collision/foam.usd`,
  `$HRC_ROOT/assets/challenge2026_assets/resources/Task3_Part_A.usd`
- Carton/task 4:
  `$HRC_ROOT/assets/challenge2026_assets/resources/Collected_Task4/Task4.usd`,
  `$HRC_ROOT/assets/challenge2026_assets/resources/Collected_Task4/SubUSDs/Foldable_box.usd`,
  `$HRC_ROOT/assets/challenge2026_assets/resources/task4_box_foam.usd`

Dataset:
- `$DATA_ROOT/challenge2026_dataset/Packing_box/box_closing_001/meta/info.json`
- `$DATA_ROOT/challenge2026_dataset/Packing_box/box_closing_001/meta/episodes.jsonl`
- `$DATA_ROOT/challenge2026_dataset/Packing_box/box_closing_001/meta/episodes_stats.jsonl`
- `$DATA_ROOT/challenge2026_dataset/Packing_box/box_closing_001/meta/tasks.jsonl`
- `$DATA_ROOT/challenge2026_dataset/Packing_box/box_closing_001/data/chunk-000/episode_000000.parquet`
  through `episode_000049.parquet`

## Final Readiness

Resource availability is ready for Phase 3 official scene and baseline
inspection work.

Not yet ready:
- Baseline repo has not been executed.
- Official task reset/scoring entrypoints have not been mapped.
- Only one official table USD was loaded with Walker S2.
- Full Task 1/2/3/4 scene composition has not been validated.
- Dataset schema has not been inspected beyond filesystem presence.

## Exact Next Step

Inspect the baseline repo task configs and scene builder to determine the
official asset-root mapping, reset flow, action format, and intended task scene
composition before writing any task logic.
