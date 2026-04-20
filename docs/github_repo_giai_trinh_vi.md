# Giải Trình Repo GitHub HRC 2026

Nguồn mô tả: `https://github.com/Moquz27/hrc2026`

Snapshot kiểm tra local:
- Remote đã xác nhận: `origin` trỏ tới `https://github.com/Moquz27/hrc2026.git`
- Branch local: `main`
- HEAD local: `212daed`
- Thời điểm lập tài liệu: 2026-04-20

Ghi chú phạm vi:
- Tài liệu này mô tả repo source-code theo các file tracked trong snapshot local hiện tại.
- Không xem `.git`, `.venv`, `__pycache__`, runtime assets, dataset, checkpoint, logs, outputs là source của GitHub repo.
- Các tài nguyên lớn của ban tổ chức nằm ngoài repo code và được ghi nhận trong `docs/resources.md` / `docs/official_resources_inventory.md`.
- Nội dung được viết bằng tiếng Việt, tập trung vào vai trò file, quy trình vận hành, trạng thái phase hiện tại và các giới hạn đã biết.

---

## 1. Tóm Tắt Repo

Repo `hrc2026` là workspace source-code cho dự án HRC 2026 trên Isaac Sim với robot Walker S2.

Mục tiêu hiện tại của repo không phải là lời giải cuối cùng cho tất cả task. Repo đang ở giai đoạn dựng baseline, tích hợp tài nguyên chính thức, xác minh robot/scene/runtime và làm cho pipeline debug được trước khi tối ưu thuật toán.

Các nguyên tắc chính:
- Ưu tiên ổn định runtime trước tốc độ.
- Viết code trên Mac, chạy simulation thật trên Linux.
- Không commit asset nặng, dataset, checkpoint, log, output hoặc cache.
- Mọi path runtime phải đi qua environment variables, không hardcode path local.
- Thay đổi phải nhỏ, dễ test, dễ rollback.
- Sau thay đổi có ý nghĩa phải cập nhật `TASK_LOG.md`.

---

## 2. Phase Hiện Tại

### 2.1 Phase dự án tổng thể

Theo `PROJECT_CONTEXT.md` và `docs/roadmap.md`:

- Phase hiện tại: **Phase 3 - Competition Stack Integration & Validation**
- Trạng thái: phase active tiếp theo sau khi Phase 2 đã pass
- Phase 0, Phase 1, Phase 2: đã hoàn thành
- Phase 4: chỉ bắt đầu sau khi toàn bộ exit criteria của Phase 3 pass

Ý nghĩa của Phase 3:
- Làm cho robot Walker S2, assets chính thức, scene, baseline repo, dataset và các primitive motion chạy được ở mức tích hợp.
- Mục tiêu là debuggable stack, không phải tối ưu ML hoặc tối ưu điểm số.
- Các failure trong phase này được xem là integration/runtime/control validation failure, chưa phải model-quality failure.

### 2.2 Phase Task 1 trong `CURRENT_PLAN.md`

Theo `CURRENT_PLAN.md`, kế hoạch Task 1 hybrid grasp hiện tại là:

- Phase Task 1 hiện tại: **Phase 0 - Baseline Source Lock**
- Trạng thái: active
- Source baseline đã chọn: `scripts/task1_dualarmik_phase_baseline.py`

Lý do chọn source này:
- Dùng official `Part_Sorting.yaml` và `SceneBuilder`.
- Override asset root sang verified challenge assets.
- Build table, parts, Walker S2 và diagnostic static bin collider.
- Dùng official startup joint map.
- Dùng official `DualArmIK` / `CoordinateTransform`.
- Có phase sequence rõ: pregrasp, align, descend, close, micro-lift, lift, carry/place, release, retreat.
- Có logging đủ dày cho target selection, candidate diagnostics, object trace, coordinate diagnostics và real-grasp-center diagnostics.

Các phase Task 1 tiếp theo trong plan:
- Phase 1: Minimal Hybrid Skeleton
- Phase 2: Geometric Planner Hardening
- Phase 3: Thinker Advisor Integration
- Phase 4: Perception Abstraction For Future YOLO
- Phase 5: Reliability Tuning For Competition

Ghi chú quan trọng:
- `CURRENT_PLAN.md` là nguồn sự thật cho công việc Task 1 plan-driven.
- Nếu `CURRENT_PLAN.md` mâu thuẫn với log cũ hoặc context cũ, `CURRENT_PLAN.md` thắng.

### 2.3 Bottleneck Task 1 gần nhất

Theo `PROJECT_CONTEXT.md`, script smooth multi-object hiện có diagnostics và one-knob tuning support tại:

- `scripts/task1_smooth_autoseed_multi_object_baseline.py`

Kết quả sweep Linux gần nhất cho `seed=1`, `target-index=2`:
- `--grasp-depth-offset 0.0` fail trước grasp với `pre_grasp_unreachable`
- `--grasp-depth-offset -0.005` fail cùng lý do
- `--grasp-depth-offset -0.010` fail cùng lý do
- `pregrasp_error` gần như không đổi khoảng `0.304866 m`

Kết luận kỹ thuật hiện tại:
- Grasp depth không phải lever chính cho target này.
- Hướng tuning ưu tiên tiếp theo là approach / soft waypoint reachability.
- Chưa nên tune contact dwell, carry stabilization, place depth hoặc release timing cho target này vì failure xảy ra trước grasp.

---

## 3. Quy Trình Làm Việc Của Repo

### 3.1 Vai trò máy Mac và Linux

Mac là máy development:
- đọc repo
- sửa code
- dùng Codex phân tích
- chạy test nhẹ
- commit và push

Linux là máy runtime:
- chạy Isaac Sim
- chạy simulation/evaluation
- chứa dataset
- chứa assets
- chứa checkpoints
- chứa logs và outputs
- sinh kết quả thật để phân tích

Nguyên tắc:
- Mac = development và orchestration
- Linux = execution và validation
- Linux runtime logs là nguồn sự thật khi nói về kết quả simulation.

### 3.2 Vòng lặp chuẩn

Quy trình chuẩn:
1. Sửa code / docs trên Mac.
2. Commit và push qua Git.
3. Pull trên Linux.
4. Chạy Isaac/runtime script trên Linux.
5. Inspect logs, metrics, replay nếu có.
6. Ghi kết quả vào `TASK_LOG.md`.
7. Quay lại Mac để sửa tiếp.

### 3.3 Runtime environment variables

Các biến môi trường Linux quan trọng:
- `HRC_ROOT`
- `HRC_REPO`
- `DATA_ROOT`
- `CKPT_ROOT`
- `OUTPUT_ROOT`
- `LOG_ROOT`

Quy tắc:
- Code không hardcode `/Users/...` hoặc `/home/...`.
- Script runtime đọc path qua env var hoặc CLI override.
- Logs ghi dưới `LOG_ROOT`.
- Metrics ghi dưới `OUTPUT_ROOT/metrics` khi có.

### 3.4 Thứ tự kiểm tra runtime nên dùng

Thứ tự kiểm tra từ nhẹ đến nặng:
1. `scripts/smoke_isaac.py`
2. `scripts/minimal_scene_baseline.py`
3. `scripts/load_walker_s2.py`
4. `scripts/control_walker_s2_arms.py`
5. `scripts/right_arm_joint_space_sanity.py`
6. `scripts/front_seeded_manipulation_motion.py`
7. Các script official resource / Task 1 scene validation
8. Các script Task 1 manipulation baseline

---

## 4. Cấu Trúc Thư Mục

Repo tracked hiện tại gồm các nhóm chính:

```text
.
├── AGENTS.md
├── CURRENT_PLAN.md
├── PROJECT_CONTEXT.md
├── README.md
├── TASK_LOG.md
├── configs/
├── docs/
├── requirements.txt
├── scripts/
├── src/
├── task1_no_phase_patch.diff
└── tests/
```

Ý nghĩa từng thư mục:

| Thư mục | Vai trò |
| --- | --- |
| `configs/` | Placeholder cho config nội bộ trong repo; hiện chỉ giữ `.gitkeep`. |
| `docs/` | Tài liệu dự án, roadmap, resource inventory, baseline inventory, giải trình kỹ thuật. |
| `scripts/` | Entrypoint runtime/diagnostic/baseline cho Isaac Sim, Walker S2 và Task 1. |
| `src/` | Placeholder cho source package tương lai; hiện chưa có module runtime chính. |
| `tests/` | Test nhẹ local; hiện có smoke test đơn giản. |

---

## 5. File Gốc Ở Root Repo

| File | Vai trò | Nội dung chính | Ghi chú vận hành |
| --- | --- | --- | --- |
| `.gitignore` | Hygiene Git | Ignore `.venv`, cache, env, data, checkpoints, outputs, logs, media, `.DS_Store` | Giữ repo nhẹ, không commit runtime artifact. |
| `AGENTS.md` | Rule cho agent/Codex | Bắt buộc đọc context, giữ thay đổi nhỏ, không hardcode path, cập nhật log | Là file cần đọc trước khi sửa repo. |
| `CURRENT_PLAN.md` | Contract active cho Task 1 | Phase Task 1, source baseline đã chọn, phase tiếp theo | Nếu mâu thuẫn với log cũ thì file này thắng. |
| `PROJECT_CONTEXT.md` | Context dự án ngắn | Phase hiện tại, workflow Mac/Linux, env vars, current focus | Nguồn nhanh để biết repo đang ở đâu. |
| `README.md` | Overview public | Mục tiêu repo, quick checks, runtime expectations | Nói rõ repo là baseline/integration, chưa phải final solver. |
| `TASK_LOG.md` | Nhật ký kỹ thuật | Lịch sử thử nghiệm, kết quả runtime, bugfix, next step | Phải cập nhật sau thay đổi có ý nghĩa. |
| `requirements.txt` | Dependency nhẹ local | `numpy`, `pytest`, `PyYAML`, `tqdm` | Không đại diện toàn bộ Isaac runtime; Isaac dùng môi trường riêng. |
| `task1_no_phase_patch.diff` | Patch artifact | Patch nháp cho smooth continuous Task 1 no-phase/streaming changes | Không phải script chạy trực tiếp; dùng như diff tham khảo. |

---

## 6. File Placeholder

| File | Vai trò |
| --- | --- |
| `configs/.gitkeep` | Giữ thư mục `configs/` trong Git. |
| `docs/.gitkeep` | Giữ thư mục `docs/` trong Git từ skeleton ban đầu. |
| `scripts/.gitkeep` | Giữ thư mục `scripts/` trong Git từ skeleton ban đầu. |
| `src/.gitkeep` | Giữ thư mục `src/` trong Git cho package tương lai. |

---

## 7. Tài Liệu Trong `docs/`

| File | Vai trò | Mô tả |
| --- | --- | --- |
| `docs/baseline_full_inventory.txt` | Inventory official baseline | Kiểm kê chi tiết repo official baseline `GlobalHumanoidRobotChallenge_2026_Baseline` ở runtime. |
| `docs/baseline_full_inventory_vi.txt` | Bản tiếng Việt của baseline inventory | Giải thích từng file của official baseline bằng tiếng Việt. |
| `docs/baseline_status.md` | Trạng thái baseline hiện tại | Phân loại script diagnostic/baseline/experiment và nói rõ cái gì đã/ chưa được chứng minh. |
| `docs/context_full.md` | Context workflow đầy đủ | Mô tả workflow Mac/Linux, Git rules, env vars, daily workflow, chiến lược baseline-first. |
| `docs/hrc2026_full_inventory.txt` | Full inventory repo HRC local | Kiểm kê rất rộng toàn cây local, gồm cả cache/vendor trong lần scan đó. |
| `docs/hrc2026_full_inventory_vi.txt` | Bản tiếng Việt của full inventory | Bản dịch/diễn giải tiếng Việt của inventory toàn repo local. |
| `docs/official_resources_inventory.md` | Inventory tài nguyên chính thức | Ghi trạng thái baseline repo, Walker S2 model, challenge assets, dataset, candidate entrypoints. |
| `docs/resources.md` | Danh sách canonical resources | Link chính thức: UBTECH baseline, Hugging Face assets, dataset, Walker S2 model. |
| `docs/roadmap.md` | Roadmap Phase 0-4 | Phase 0-2 done, Phase 3 active, Phase 4 future optimization. |

Ghi chú:
- `docs/hrc2026_full_inventory*.txt` là tài liệu kiểm kê rất lớn và có thể bao gồm cache/local artifacts của thời điểm quét cũ. Không nên xem mọi entry trong đó là source cần sửa.
- File canonical để hiểu phase hiện tại là `docs/roadmap.md` kết hợp `PROJECT_CONTEXT.md`.

---

## 8. Scripts Runtime Và Diagnostic

### 8.1 Nhóm smoke/runtime nền

| Script | Vai trò | Quy trình liên quan |
| --- | --- | --- |
| `scripts/smoke_isaac.py` | Minimal Isaac Sim runtime smoke test | Verify Isaac khởi động, env vars hợp lệ, stage step được, ghi log được vào `LOG_ROOT`. |
| `scripts/minimal_scene_baseline.py` | Minimal scene baseline | Tạo một scene Isaac đơn giản với static cube, step frames, ghi log/metrics. |
| `scripts/load_walker_s2.py` | Walker S2 load baseline | Load Walker S2 USD, phát hiện articulation root, joint count, reject Git LFS pointer. |
| `scripts/load_official_scene_smoke.py` | Official asset scene smoke | Load một official USD asset cùng Walker S2 để xác minh resource mapping. |
| `scripts/inspect_official_assets.py` | Filesystem inventory resource | Kiểm tra baseline/assets/dataset/Walker S2 ngoài repo, size, key files, Git LFS pointer. |

Ý nghĩa:
- Đây là ladder kiểm tra từ môi trường đến robot/asset.
- Phase 1 dùng `smoke_isaac.py`.
- Phase 2 dùng `load_walker_s2.py`.
- Phase 3 dùng thêm official assets/baseline/dataset smoke.

### 8.2 Nhóm robot control cơ bản

| Script | Vai trò | Quy trình liên quan |
| --- | --- | --- |
| `scripts/control_walker_s2_arms.py` | Basic arm control smoke | Acquire articulation, đọc DOF, gửi target nhỏ cho arm, log observed motion. |
| `scripts/move_walker_s2_end_effector.py` | Cartesian EE target experiment | Damped least-squares IK cho right arm; hữu ích nhưng có rủi ro posture xấu. |
| `scripts/right_arm_joint_space_sanity.py` | Joint-space sign sanity | Kiểm tra dấu joint right arm, front pose, raise/lower, gripper open/close. |
| `scripts/front_seeded_manipulation_motion.py` | Motion baseline mạnh nhất giai đoạn đầu | Phased joint-space right-arm motion từ front seed; không chứng minh object transport. |
| `scripts/grasp_static_object_smoke.py` | Fixed target grasp-position smoke | Test wrist pose và gripper quanh cube tĩnh; không phải proof grasp vật động. |

Ý nghĩa:
- Nhóm này thuộc Phase 3.2 minimal robot control validation.
- Kết quả tốt ở nhóm này chỉ chứng minh robot có thể command/move, chưa chứng minh task solving.

### 8.3 Nhóm official Task 1 scene/assets

| Script | Vai trò | Quy trình liên quan |
| --- | --- | --- |
| `scripts/validate_task1_object_assets.py` | Validate official Task 1 workpiece assets | Kiểm tra Part A/B root và collected variants có collision/rigid body, rơi và nằm trên table. |
| `scripts/validate_task1_scene_builder_scene.py` | Validate official SceneBuilder Task 1 scene | Build table + parts qua official SceneBuilder với root_path override; kiểm tra physics/schema. |
| `scripts/diagnose_task1_bin_physics.py` | Diagnose Task 1 bin physics | So sánh standalone box asset với SceneBuilder box; tạo diagnostic static bin workaround. |
| `scripts/inspect_task1_pick_place_gui.py` | GUI-first Task 1 visual inspection | Build official Task 1 scene, Walker S2, diagnostic bin, marker; chạy visual phase flow. |

Ý nghĩa:
- Nhóm này thuộc Phase 3.3 và 3.4.
- Repo đã ghi nhận collected Task 1 Part A/B variants là physics-ready hơn root visual USD.
- Official composed box path qua SceneBuilder có vấn đề physics hierarchy, nên có diagnostic static bin collider workaround.

### 8.4 Nhóm Task 1 manipulation baseline

| Script | Vai trò | Trạng thái / ý nghĩa |
| --- | --- | --- |
| `scripts/task1_single_target_random_scene_baseline.py` | One-object Task 1 pick-place attempt | Dùng official randomized scene, right gripper effort hold, object-centric pass/fail cho một target. |
| `scripts/task1_smooth_autoseed_multi_object_baseline.py` | Smooth auto-seed multi-object Task 1 variant | Có auto-seed, loop nhiều object, continuous-motion refactor, diagnostics failure phase/reason. |
| `scripts/task1_cartesian_dls_phase_baseline.py` | Cartesian DLS phase baseline | Task 1 baseline dùng Cartesian DLS phase structure; là nhánh thử nghiệm cũ hơn so với DualArmIK plan. |
| `scripts/task1_dualarmik_phase_baseline.py` | Selected Task 1 source baseline | Baseline source đã chọn trong `CURRENT_PLAN.md`; dùng official DualArmIK, SceneBuilder, gates và logging tốt. |
| `scripts/task1_dualarmik_phase_nogate.py` | No-gate diagnostic duplicate | Bản bypass gate để chẩn đoán; không phải baseline source an toàn. |
| `scripts/task1_hybrid_geometric_phase1.py` | Hybrid geometric Phase 1 skeleton | Giữ official table frame/object info/candidate generation/DualArmIK, chưa là final optimizer. |
| `scripts/task1_hybrid_geometric_phase2.py` | Phase 2 contact-hardening variant | Thêm geometric grasp filtering và local contact/descent hardening. |
| `scripts/task1_phase2_contact_centric_patch.py` | Contact-centric patch variant | Bản patch chi tiết cho real grasp center, fingertip/contact reference, close gate và final descent diagnostics. |

Ý nghĩa:
- Đây là vùng code Task 1 phát triển mạnh nhất.
- `task1_dualarmik_phase_baseline.py` là source chính cho plan hiện tại.
- `task1_phase2_contact_centric_patch.py` là bản nghiên cứu/patch rất chi tiết về contact-centric, nhưng không phải source lock hiện tại theo `CURRENT_PLAN.md`.
- `task1_smooth_autoseed_multi_object_baseline.py` có dữ liệu failure gần nhất cho target-index 2: bottleneck trước grasp.

### 8.5 GUI wrapper scripts

| Script | Vai trò |
| --- | --- |
| `scripts/watch_task1_gui.sh` | Wrapper chạy `task1_smooth_autoseed_multi_object_baseline.py` với Isaac python, seed/target/depth env override, GUI và hold-open. |
| `scripts/watch_task1_phase2_gui.sh` | Wrapper chạy `task1_hybrid_geometric_phase2.py` với Isaac python, seed/target/arm env override, GUI và hold-open. |

Quy tắc:
- Các wrapper này cần `ISAAC_SIM_PYTHON` hoặc `ISAAC_SIM_ROOT`.
- Dùng trên Linux runtime, không dùng để chạy heavy simulation trên Mac.

---

## 9. Test Và Dependency

| File | Vai trò | Mô tả |
| --- | --- | --- |
| `requirements.txt` | Dependency local nhẹ | Gồm `numpy`, `pytest`, `PyYAML`, `tqdm`; không thay thế Isaac Sim environment. |
| `tests/test_smoke.py` | Test smoke đơn giản | `assert 1 + 1 == 2`; dùng để xác nhận pytest hoạt động, không kiểm tra logic robot. |

Ghi chú:
- Lightweight test trên Mac chỉ giúp bắt lỗi cơ bản.
- Runtime validation thật nằm ở Linux Isaac logs.

---

## 10. Quy Trình Tích Hợp Hiện Có

### 10.1 Phase 0 - Repo & workflow setup

Đã làm:
- Tạo skeleton `src`, `scripts`, `configs`, `tests`, `docs`.
- Thêm `AGENTS.md`, `PROJECT_CONTEXT.md`, `TASK_LOG.md`.
- Thêm `.gitignore`.
- Định nghĩa workflow Mac/Linux.

Kết quả:
- Repo có cấu trúc ổn định.
- Heavy runtime artifacts bị loại khỏi Git.
- Git là cơ chế sync chính giữa Mac và Linux.

### 10.2 Phase 1 - Isaac runtime smoke

Đã làm:
- Thêm `scripts/smoke_isaac.py`.
- Preflight pass.
- Isaac smoke pass.
- Log ghi ngoài repo.

Kết quả:
- Isaac runtime có thể khởi động từ workflow dự án.
- Env vars cần thiết được validate.
- `LOG_ROOT` write path hoạt động.

### 10.3 Phase 2 - Walker S2 load baseline

Đã làm:
- Thêm `scripts/load_walker_s2.py`.
- Thêm Git LFS pointer detection.
- Walker S2 loaded successfully từ runtime assets.
- Detect articulation root và 42 joints.

Known issue:
- Joint state read warning còn tồn tại nhưng non-blocking.

Kết quả:
- Robot asset payload thật đã có.
- Robot load được trong Isaac Sim.
- Articulation root và joints discoverable.

### 10.4 Phase 3 - Competition stack integration & validation

Đã có tiến triển:
- Official resources đã được ghi nhận trong `docs/resources.md`.
- `docs/official_resources_inventory.md` ghi baseline repo, Walker S2 model, official assets, dataset đã có ở runtime.
- Official table USD đã smoke-load cùng Walker S2.
- Task 1 collected workpiece variants đã được validate physics tốt hơn root visual USD.
- SceneBuilder table + parts usable hơn composed official box path.
- Diagnostic static bin collider workaround đã được dùng để tránh invalid composed box physics.
- Task 1 one-object baseline từng đạt success cho target-index 1 với gripper effort 100.0 theo `TASK_LOG.md`.
- Smooth multi-object baseline có diagnostics nhưng chưa robust toàn bộ object.

Chưa xong:
- Phase 3 chưa pass toàn bộ exit criteria.
- Baseline repo chưa được coi là usable đầy đủ ở inspect/smoke-test level cho mọi task.
- Dataset mới ở mức resource/inspection, chưa thành training pipeline.
- Task 2/3/4 primitive tests chưa hoàn thiện trong repo.
- Chưa có strategy decision cuối cho từng task.

---

## 11. Resource Chính Thức Ngoài Repo

Canonical file: `docs/resources.md`

| Resource | Link | Vai trò |
| --- | --- | --- |
| Baseline | `https://github.com/UBTECH-Robot/GlobalHumanoidRobotChallenge_2026_Baseline` | Official workflow/reference package cho training/deployment. |
| Assets | `https://huggingface.co/UBTECH-Robotics/challenge2026_assets` | Simulation assets cho scenes, task objects, environment resources. |
| Dataset | `https://huggingface.co/datasets/UBTECH-Robotics/challenge2026_dataset` | Official challenge dataset, hiện dùng như Packing_Box dataset reference. |
| WalkerS2 USD | `https://github.com/UBTECH-Robot/WalkerS2-Model-Challenge` | Walker S2 USD/URDF/STL model cho Isaac Sim. |

Quy tắc:
- Không đưa các resource lớn này vào Git repo.
- Các path runtime được resolve qua `HRC_ROOT`, `DATA_ROOT`, `LOG_ROOT`.
- Nếu thiếu asset hoặc còn Git LFS pointer thì phải sửa ở runtime resource setup, không sửa bằng hardcoded path trong code.

---

## 12. Trạng Thái Kỹ Thuật Theo Từng Mảng

| Mảng | Trạng thái | Bằng chứng trong repo | Giới hạn |
| --- | --- | --- | --- |
| Repo/workflow | Pass | `AGENTS.md`, `PROJECT_CONTEXT.md`, `.gitignore`, `docs/context_full.md` | Cần tiếp tục giữ log sạch. |
| Isaac smoke | Pass | `scripts/smoke_isaac.py`, `TASK_LOG.md` | Chỉ smoke runtime, không robot/task. |
| Walker S2 load | Pass | `scripts/load_walker_s2.py`, roadmap Phase 2 | Joint state warning non-blocking còn ghi nhận. |
| Basic control | Partial pass | `control_walker_s2_arms.py`, `right_arm_joint_space_sanity.py` | Chưa chứng minh full-body/dual-arm/task-scene robustness. |
| Official resources | Present/usable for inspection | `docs/resources.md`, `docs/official_resources_inventory.md` | Baseline/dataset chưa thành pipeline tối ưu. |
| Task 1 scene | Partial pass | object/SceneBuilder/bin diagnostic scripts | Official composed box physics có vấn đề, dùng diagnostic workaround. |
| Task 1 manipulation | Experimental/partial | single-target và smooth multi-object scripts | Không robust; target-index 2 đang fail trước grasp. |
| Task 2/3/4 | Not implemented / not validated | roadmap nêu subphases cần làm | Chưa có primitive tests hoàn chỉnh. |
| ML optimization | Not started | Phase 4 future | Chỉ bắt đầu sau Phase 3 pass. |

---

## 13. Cách Đọc Repo Khi Người Mới Vào

Thứ tự đọc khuyến nghị:
1. `AGENTS.md`
2. `PROJECT_CONTEXT.md`
3. `CURRENT_PLAN.md`
4. `TASK_LOG.md`
5. `docs/roadmap.md`
6. `docs/resources.md`
7. `docs/official_resources_inventory.md`
8. `docs/baseline_status.md`

Sau đó mới đọc script theo mục tiêu:
- Muốn kiểm tra runtime: bắt đầu từ `scripts/smoke_isaac.py`.
- Muốn kiểm tra robot load: đọc `scripts/load_walker_s2.py`.
- Muốn kiểm tra arm control: đọc `scripts/control_walker_s2_arms.py`.
- Muốn hiểu Task 1 source hiện tại: đọc `scripts/task1_dualarmik_phase_baseline.py`.
- Muốn hiểu contact-centric patch: đọc `scripts/task1_phase2_contact_centric_patch.py` và tài liệu giải trình riêng nếu có.

---

## 14. Quy Trình Task 1 Hiện Tại

Quy trình Task 1 trong repo đã tiến hóa qua nhiều lớp:

1. Kiểm tra object/asset:
   - Validate Part A/B root và collected variants.
   - Kết luận: collected variants phù hợp hơn cho physics manipulation.

2. Kiểm tra SceneBuilder:
   - Build table + parts qua official `Part_Sorting.yaml`.
   - Override root path sang official assets tree.

3. Chẩn đoán bin:
   - Standalone box asset có thể inspect được.
   - SceneBuilder composed box path tạo invalid rigid body hierarchy.
   - Chọn diagnostic static bin collider để unblock manipulation validation.

4. Kiểm tra visual pick/place:
   - Dùng GUI inspection với marker để xác nhận placement, target, bin, phase direction.

5. Single-target manipulation:
   - Dùng official startup joint map, official DualArmIK hoặc direct phase path tùy script.
   - Right gripper effort hold 100.0 đã giúp one-object target-index 1 pass trong log.

6. Smooth multi-object:
   - Có auto-seed.
   - Có loop nhiều object.
   - Có diagnostics failure phase/reason.
   - Kết quả hiện không robust.

7. DualArmIK baseline source lock:
   - `CURRENT_PLAN.md` chọn `scripts/task1_dualarmik_phase_baseline.py` làm source để fork tiếp.
   - No-gate và contact-centric variants là diagnostic/experimental, không phải source chính mặc định.

---

## 15. Rủi Ro Và Giới Hạn Hiện Tại

Các điểm không nên hiểu nhầm:
- Repo chưa có final task-solving policy.
- Repo chưa có submission-ready package.
- Repo chưa có ML optimization.
- `front_seeded_manipulation_motion.py` là motion sanity, không phải proof sorting.
- `grasp_static_object_smoke.py` không chứng minh object transport.
- `task1_dualarmik_phase_nogate.py` là diagnostic bypass, không phải đường an toàn mặc định.
- Contact-centric scripts có nhiều diagnostics nhưng không thay thế `CURRENT_PLAN.md`.

Các rủi ro kỹ thuật:
- Official composed Task 1 box/bin physics path có vấn đề.
- Một số motion có thể phụ thuộc target region/orientation preset.
- Target-index 2 trong smooth baseline đang fail trước grasp.
- Nếu tune sai family, có thể tối ưu nhầm grasp depth/carry/place trong khi bottleneck là approach/pregrasp reachability.
- Isaac runtime chỉ xác thực trên Linux, không nên suy luận kết quả thật từ Mac-only static inspection.

---

## 16. Next Engineering Direction Theo Repo Hiện Tại

Ở mức project phase:
- Tiếp tục Phase 3 integration & validation.
- Không nhảy sang Phase 4 optimization khi Phase 3 chưa pass exit criteria.

Ở mức Task 1 plan:
- Theo `CURRENT_PLAN.md`, tiếp tục từ source lock `scripts/task1_dualarmik_phase_baseline.py`.
- Phase kế tiếp là minimal hybrid skeleton nếu người dùng cho phép implement.

Ở mức bottleneck smooth baseline gần nhất:
- Ưu tiên kiểm tra/tune approach và soft waypoint reachability cho target-index 2.
- Chưa ưu tiên grasp depth, contact dwell, carry stabilization, place depth, release timing.

Ở mức repo hygiene:
- Tiếp tục ghi kết quả chạy Linux vào `TASK_LOG.md`.
- Giữ docs và runtime scripts tách bạch.
- Không commit runtime assets/datasets/logs.

---

## 17. Tóm Tắt Ngắn

Repo `Moquz27/hrc2026` là workspace HRC 2026 cho Walker S2 trên Isaac Sim.
Phase tổng thể hiện tại là Phase 3: tích hợp và validation competition stack.
Phase 0-2 đã hoàn thành: repo/workflow setup, Isaac smoke, Walker S2 load baseline.
Phase 4 optimization chưa bắt đầu.

Repo có ba lớp chính:
- Tài liệu điều phối: `PROJECT_CONTEXT.md`, `CURRENT_PLAN.md`, `TASK_LOG.md`, `docs/roadmap.md`.
- Script validation nền: Isaac smoke, minimal scene, Walker S2 load, arm control, official resource checks.
- Script Task 1: asset/scene/bin diagnostics, single-target baseline, smooth multi-object baseline, DualArmIK baseline, hybrid/contact-centric variants.

Tình trạng thật:
- Stack đang ngày càng debuggable.
- Task 1 có nhiều bước đã validate từng phần, nhưng chưa robust toàn bộ.
- Task 2/3/4 và ML optimization chưa phải trọng tâm hiện tại.
- Hướng kỹ thuật đúng lúc này là tiếp tục integration/validation, không tối ưu sớm.
