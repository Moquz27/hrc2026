# Đặc Tả Giải Trình Thuật Toán
File nguồn: scripts/task1_phase2_contact_centric_patch.py
Ngôn ngữ: Tiếng Việt
Phạm vi: giải thích ý tưởng tổng quát, sau đó giải thích theo từng block code và từng dòng/nhóm dòng có chức năng.

Ghi chú về số dòng:
- Số dòng được ghi theo trạng thái repo tại thời điểm lập tài liệu này.
- Nếu file nguồn thay đổi, số dòng có thể lệch, nhưng tên hàm và vai trò logic vẫn là điểm đối chiếu chính.
- File nguồn dài gần 10.000 dòng, nên các dòng liên tiếp có cùng một thao tác được gộp thành khoảng dòng để tránh lặp lại cơ học.

## 1. Ý Tưởng Tổng Quát

### 1.1. Mục tiêu của file

File scripts/task1_phase2_contact_centric_patch.py là một bản Task 1 Phase 2 "contact-centric" cho Walker S2 trong Isaac Sim.
Nó không phải là một pipeline học máy, không thêm Thinker, không thêm YOLO, không thêm planner mới.
Nó lấy nền từ scripts/task1_hybrid_geometric_phase1.py và bổ sung lớp làm cứng tiếp xúc:
- Lọc ứng viên grasp bằng hình học rẻ.
- Ưu tiên tham chiếu tiếp xúc thực tế của hai đầu ngón tay thay vì chỉ tin vào proxy point_B.
- Điều khiển final descent theo điểm tiếp xúc runtime.
- Gate close dựa trên metric runtime của hai ngón tay, symmetry, alignment, table clearance, support gap và z-stall.
- Ghi log rất dày để debug từng pha, từng nguồn tham chiếu, từng điều kiện cho phép close.

### 1.2. Vấn đề file đang giải quyết

Task 1 cần robot gắp vật trên bàn và bỏ vào bin.
Trong các bản baseline trước, việc "target pose" thường được định nghĩa bằng EE origin hoặc một proxy point_B suy ra từ TCP offset.
Vấn đề là:
- EE origin không nằm tại điểm tiếp xúc thật.
- point_B là proxy hình học, không chắc trung với đầu ngón tay thật.
- Nếu close dựa trên proxy sai, robot có thể close khi ngón tay chưa đến vật, hoặc từ chối close vì proxy trong log không phản ánh tiếp xúc thật.

Bản contact-centric đổi cách nhìn:
- DualArmIK vẫn điều khiển pose EE.
- Nhưng điều kiện tiếp xúc, final descent và close gate cố gắng đo bằng "real grasp center" runtime.
- Real grasp center được ưu tiên theo thứ tự tin cậy:
  1. midpoint của actual fingertip frame pair nếu resolve được;
  2. stable finger-link midpoint nếu actual frame không có;
  3. calibrated distal proxy chỉ là diagnostic, không được tự động authorize close;
  4. point_B chỉ là fallback compatibility.

### 1.3. Luồng tổng quát của thuật toán

Bước 1: Đọc tham số CLI và validate runtime paths.
Script lấy baseline root, asset root, Walker S2 USD/URDF, log root, seed, target index, các threshold Phase 2.

Bước 2: Khởi tạo Isaac Sim và scene Task 1.
Script dùng official SceneBuilder để build table và parts.
Bin official có vấn đề physics trong các log trước, nên script dùng visual official box nếu cần và diagnostic static bin colliders để thao tác ổn định.

Bước 3: Load Walker S2 và acquire articulation.
Script tìm articulation root, acquire dynamic_control handle, đọc joint/body names, apply startup joint map.

Bước 4: Chọn target object.
Script lấy bbox của các part, tính center, sắp xếp theo target-index/seed/near, gắn category A/B nếu có.

Bước 5: Thiết lập frame làm việc.
Script tạo table frame, robot-base/world coordinate transform, EE frame compensation, TCP offset, point_B offset.

Bước 6: Phân loại target region.
Target được chiếu về frame robot base.
Nếu target xa thì dùng family world_y_approach/far_low_side_B_driven.
Nếu target mid/near thì dùng vertical Z descent.

Bước 7: Tạo ứng viên hướng tiếp cận và pose pregrasp.
Script dùng library orientation preset theo arm và approach family.
Mỗi preset được biến thành geometry gồm:
- pregrasp point_B world;
- align point_B world;
- contact point_B world;
- lift, carry, place, retreat pose;
- AB semantics và các metric debug.

Bước 8: Lọc ứng viên bằng IK và hình học.
Pregrasp candidate phải qua DualArmIK với error dưới tolerance.
Phase 2 candidate phải qua các check hình học: alignment, symmetry, width, clearance, predicted early contact asymmetry.

Bước 9: Chạy phase motion.
Tùy motion policy:
- Far: pregrasp -> outboard transition -> low side prepare -> XY align -> final descent.
- Vertical/mid/near: pregrasp -> align -> descend -> final descent.

Bước 10: Final descent contact-centric.
Thay vì chỉ lệnh cho EE đến pose cố định, mỗi tick script:
- đo contact reference runtime;
- nếu có trusted fingertip midpoint thì dùng tip_mid;
- nếu không, fallback point_B;
- tính delta đến locked target;
- clamp XY step;
- clamp Z chỉ đi xuống, không cho tăng Z;
- clamp yaw;
- đổi contact-reference target thành EE pose rồi đưa cho DualArmIK.

Bước 11: Pre-close gate.
Trước khi close, script thu thập:
- current EE pose;
- point_B;
- real grasp center;
- fingertip pair metrics;
- table clearance;
- selected candidate filter;
- final descent samples;
- support gap và z-stall.

Bước 12: Quyết định close.
Close chỉ được authorize nếu pass một trong các đường:
- primary close gate dựa trên runtime truth;
- vertical support/stall fallback;
- runtime commit fallback.
Nhưng vẫn bị chặn bởi hard blocker như catastrophic table clearance, catastrophic orientation, width sai, hoặc close-critical reference không tin cậy.

Bước 13: Close, lift, retry.
Script close hai stage, giữ effort, short lift verify.
Nếu fail và còn retry, recover và lặp lại một phần.

Bước 14: Carry/place/release/settle và ghi log.
Nếu grasp/retention pass, script carry đến bin, place, release, retreat, settle, tính inside-bin và stability.

### 1.4. Ý nghĩa của "contact-centric"

"Contact-centric" trong file này không có nghĩa là có physics contact sensor chính thức.
Nó có nghĩa là vòng điều khiển gần tiếp xúc không lấy EE origin hay point_B làm sự thật cuối.
Nó cố gắng resolve tham chiếu thực hơn từ runtime:
- actual fingertip frame;
- midpoint của cặp ngón tay;
- link midpoint ổn định;
- runtime two-finger geometry.

Khi không có dữ liệu đủ tin cậy, script vẫn có fallback point_B để tiếp tục debug, nhưng log sẽ đánh dấu rõ fallback/proxy, và một số gate sẽ không cho close nếu nguồn đó không đủ tin cậy.

## 2. Giải Thích Theo Block CODE

### Block 1: Header, import, hằng số và orientation preset cơ bản

**Dòng liên quan: 1-220**

**Mục tiêu block:**
- Khai báo mục đích script.
- Import các module và helper từ các script runtime đã có.
- Định nghĩa các threshold Phase 1/Phase 2.
- Định nghĩa các preset orientation cho approach theo Z và theo world-Y.

**Giải thích từng dòng/nhóm dòng:**

Dòng 1:
Shebang chạy bằng python3 khi script được execute trực tiếp.

Dòng 2-8:
Docstring mô tả rõ file là Task 1 Phase 2 hybrid geometric contact-hardening.
Dòng này là contract thiết kế: giữ table-frame, scene_state object_info, finite candidate generation, DualArmIK backend và phase-machine execution từ Phase 1.
Dòng này cũng nói rõ Phase 2 chỉ thêm geometric grasp filtering và contact/descent hardening, không thêm Thinker, YOLO, hay planner backend mới.

Dòng 10:
Dùng from __future__ import annotations để type hint có thể tham chiếu tên class/chưa cần evaluate ngay.

Dòng 12-23:
Import các thư viện chuẩn:
- argparse cho CLI.
- importlib.util để load official baseline module động.
- json cho log.
- math cho geometry.
- random cho seed.
- sys/traceback cho runtime error.
- dataclass cho ServoSpec.
- datetime/timezone cho timestamp log.
- pathlib.Path cho path handling.
- typing.Any/Callable cho type hint linh hoạt.

Dòng 25:
Import numpy làm nền cho vector, matrix, pose và metric.

Dòng 27-33:
Import helper từ control_walker_s2_arms:
- acquire articulation;
- hold GUI;
- read DOF observation;
- send position target;
- start timeline.
Nghĩa là file này không tự viết lại nhưng primitive runtime đã có.

Dòng 34-38:
Import helper từ diagnose_task1_bin_physics:
- box relative path mặc định;
- add diagnostic static bin colliders;
- disable physics under visual box.
Đây là dấu vết của chiến lược bin diagnostic thay cho composed official box bị lỗi physics.

Dòng 39:
Import helper từ load_walker_s2:
- init steps mặc định;
- tạo minimal scene;
- tìm joint names;
- load SimulationApp;
- validate environment.

Dòng 40:
Import bbox và physics summary từ validate_task1_object_assets để dùng lại logic inspect asset/object.

Dòng 41-49:
Import helper từ validate_task1_scene_builder_scene:
- default asset/baseline/config path;
- NullDataLogger;
- category_from_reference;
- official SceneBuilder loader;
- reference paths.
Đây là cầu nối vào official baseline/resource stack.

Dòng 52-54:
SCRIPT_NAME, LOG_STEM và SOURCE_BASELINE_SCRIPT đặt danh tính log.
Lưu ý file hiện tại tên task1_phase2_contact_centric_patch.py nhưng SCRIPT_NAME/LOG_STEM vẫn ghi task1_hybrid_geometric_phase2.py; đây là chi tiết cần cảnh giác khi tìm log.

Dòng 56:
TABLE_UNIT_M = 0.035, dùng để đổi đơn vị table/config sang meter nếu cần.

Dòng 57-64:
Hằng số Phase 1:
- yaw preset;
- weight score reach/side/yaw/width;
- min/max object width;
- symmetry tolerance.
Nhưng hằng số này giữ compatibility với lớp lọc coarse cũ.

Dòng 66-105:
Hằng số Phase 2:
- alignment error max;
- symmetry/contact asymmetry max;
- table clearance min;
- width min/max;
- violation weight;
- allow least bad candidate mặc định False;
- close tolerance;
- catastrophic orientation/table clearance;
- runtime commit fallback thresholds;
- vertical fallback thresholds;
- final descent step limits;
- close stage fraction;
- retention/lift/retry thresholds;
- fingertip proxy thresholds.
Đây là "bảng điều khiển" chính của contact-centric patch.

Dòng 107-111:
Official robot prim/name và gripper width/effort.
DEFAULT_GRIPPER_HOLD_EFFORT = 100.0 là giá trị lực giữ đã được các run trước xác nhận tốt hơn 35.0.

Dòng 112-143:
Debug marker paths, radius, color cho proxy midpoint, object grasp center, pregrasp target, tip1/tip2/tip_mid, object center, runtime object grasp center, real grasp center, contact point, point_B.
Mục đích là làm rõ trên viewport/log: điểm nào là proxy, điểm nào là runtime truth.

Dòng 145-146:
TCP offset fallback và epsilon cho offset nhỏ.

Dòng 147-161:
OFFICIAL_TORSO_COMPENSATION_MATRIX là ma trận compensate frame official torso/EE.
Nó là một calibration matrix dùng khi cần align frame DualArmIK với world/runtime.

Dòng 153-160, hàm _preset_euler_xyz_to_rot:
- Dòng 154-156 tính cos/sin roll, pitch, yaw.
- Dòng 157 tạo Rx.
- Dòng 158 tạo Ry.
- Dòng 159 tạo Rz.
- Dòng 160 trả về R = Rz @ Ry @ Rx, tức convention Euler XYZ nhưng nhân ma trận theo thứ tự yaw-pitch-roll.

Dòng 163-173, hàm _preset_rot_to_euler_xyz:
- Dòng 164 ép input thành numpy array.
- Dòng 165 check gimbal lock bằng r[2,0].
- Dòng 166-168 tính pitch/roll/yaw nếu không gimbal.
- Dòng 170-172 xử lý gimbal lock: pitch +/- pi/2, roll = 0, yaw tính từ phần còn lại.
- Dòng 173 trả về list float.

Dòng 176-179, hàm _preset_with_local_ab_axis_roll:
- Dòng 177 đổi rpy gốc thành rotation.
- Dòng 178 tạo roll quanh trục local AB bằng Euler z local trong convention của preset.
- Dòng 179 trả về rpy mới sau khi nhân base_rot @ local_ab_roll.

Dòng 182-191, hàm _build_world_y_approach_presets:
- Dòng 183 tạo list rỗng.
- Dòng 184 duyệt từng preset cơ bản.
- Dòng 185 duyệt từng axial roll variant.
- Dòng 186-187 nếu roll xấp xỉ 0 thì giữ rpy.
- Dòng 188-189 nếu khác 0 thì tạo rpy rolled.
- Dòng 190 thêm label mới có suffix variant.
- Dòng 191 trả về preset list.

Dòng 194-202:
RIGHT_Z_APPROACH_PRESETS và LEFT_Z_APPROACH_PRESETS.
Đây là các hướng grasp từ trên xuống: straight và tilted_forward.

Dòng 204-220:
Comment và base preset cho world_y_approach.
Ý tưởng: FAR diagnostics cũ cho thấy family world_y_approach cũ có approach_axis_world lệch gần world +X.
Vì vậy preset mới thêm quarter-turn yaw để đưa local +Z/AB axis về base +/-X, map sang world +/-Y theo transform quan sát.

### Block 2: Preset còn lại, exception và ServoSpec

**Dòng liên quan: 221-398**

**Mục tiêu block:**
- Hoàn tất orientation preset library.
- Định nghĩa exception có cấu trúc cho failure.
- Định nghĩa ServoSpec để truyền thông số servo phase.

**Giải thích từng dòng/nhóm dòng:**

Dòng 221-376:
Tiếp tục khai báo preset, variant axial roll, map preset theo arm/family.
Các hằng số này quyết định vùng search orientation trước khi DualArmIK solve.
Đây là điểm cần xem khi target "pregrasp unreachable" vì orientation/approach family có thể quá chặt.

Dòng 378-381, class RunFailure:
- Dòng 378 tạo exception riêng cho run-level failure.
- Dòng 379 lưu reason dạng machine-readable.
- Dòng 380 lưu message để đọc log.
- Dòng 381 gọi RuntimeError constructor với message.

Dòng 384-388, class ServoEarlyStop:
- Dòng 384 tạo exception riêng cho servo stop sớm.
- Dòng 385 lưu reason.
- Dòng 386 lưu details dict.
- Dòng 387 tạo message có reason.
- Dòng 388 gọi RuntimeError constructor.
Hàm final descent dùng exception này khi vertical tip đạt table-z threshold.

Dòng 391:
@dataclass(frozen=True) làm ServoSpec immutable.

Dòng 392-398, class ServoSpec:
- phase_name: tên phase ghi log.
- target_pose_base: pose target trong base frame.
- position_tolerance: tolerance vị trí.
- rotation_tolerance: tolerance xoay.
- max_ticks: số tick servo tối đa.
ServoSpec giúp _execute_dualarmik_servo_phase nhận input gọn và nhất quán.

### Block 3: Path, fail, USD reference, xform, articulation

**Dòng liên quan: 401-541**

**Mục tiêu block:**
- Chuẩn hóa path.
- Fail có reason.
- Thêm USD reference vào stage.
- Set transform prim.
- Tìm và acquire articulation root của Walker S2.

**Giải thích từng dòng/nhóm dòng:**

Dòng 401-403, _as_path:
Nếu raw_path có giá trị thì trả Path(raw_path).expanduser(), nếu không trả default_path.
Hàm này giữ rule không hardcode local path trong code chính; path đến từ CLI/env/default relative.

Dòng 405-407, _fail:
Raise RunFailure với reason và message.
Tất cả failure chính nên đi qua hàm này để log có reason ổn định.

Dòng 409-414, _add_reference:
- Tạo prim tại prim_path.
- Add reference đến USD path.
- Trả về prim.
Dùng để load robot/asset vào stage mà không copy nội dung USD.

Dòng 416-433, _set_xform:
- Set translation, rotation, scale cho prim.
- Tạo XformCommonAPI nếu cần.
- Convert tuple/list sang Gf types.
Đây là primitive để đặt robot/object/bin.

Dòng 436-440, _valid_prim_path:
Kiểm tra prim_path có tồn tại trong stage và prim valid.
Dùng trước khi tin một path lấy từ config/log.

Dòng 443-464, _prim_has_articulation_api:
Kiểm tra prim có UsdPhysics.ArticulationRootAPI hoặc schema tương đương.
Vi official USD có thể đặt articulation API ở nested prim, script cần scan.

Dòng 467-474, _find_articulation_roots_anywhere:
Duyệt stage, gom các prim có articulation API, trả về path.
Đây là fallback khi path official/expected không dùng.

Dòng 477-524, _choose_robot_prim_path:
- Ưu tiên official robot prim path nếu valid.
- Nếu robot_prim_path input valid thì dùng.
- Nếu không, scan articulation roots.
- Ghi details về source, candidates, selected path.
Hàm này làm giảm rủi rõ path robot khác giữa official SceneBuilder và script từ load USD.

Dòng 507-524, _articulation_acquire_candidates:
Tạo danh sách path có thể acquire dynamic_control articulation.
Thứ từ gom detected_path, robot_prim_path, parent/root variants.

Dòng 527-541, _acquire_articulation_with_fallback:
- Thứ từng candidate path.
- Gọi _acquire_articulation.
- Nếu thành công trả articulation, handle, details.
- Nếu thật bai ghi error từng candidate.
Đây là lớp chống lỗi quan trọng vì dynamic_control đôi khi không acquire dùng path prim đầu tiên.

### Block 4: Vector, pose, SE3, coordinate transform, EE compensation

**Dòng liên quan: 544-985**

**Mục tiêu block:**
- Đổi object runtime sang vector numpy.
- Đọc pose body/prim.
- Chuyển đổi SE3/matrix.
- Tạo transform world <-> robot base.
- Verify và compensate EE frame.

**Giải thích từng dòng/nhóm dòng:**

Dòng 544-549, _vector3:
Lấy x/y/z từ object có field hoặc sequence, trả numpy vector.

Dòng 551-561:
_body_pose_position và _body_pose_orientation đọc pose body từ dynamic_control.
Position trả vector.
Orientation trả dict quaternion nếu có.

Dòng 563-575, _world_se3_from_prim:
Lấy world transform của prim USD và đổi thành SE3 cho IK.

Dòng 577-585, _pin_frame_se3:
Lấy frame SE3 từ ik_solver theo frame_name.
Nếu không có frame thì fail có reason.

Dòng 588-600:
_se3_from_matrix và _matrix_from_se3 đổi qua lại giữa ma trận 4x4 và object SE3 của official IK.
Đây là cầu nối giữa numpy và DualArmIK.

Dòng 602-605, _rotation_delta_rad:
Tính sai khác rotation từ delta_se3.
Dùng cho diagnostics/tolerance.

Dòng 608-614, _fk_ee_world_se3:
Lấy forward kinematics EE world SE3 cho arm side.

Dòng 617-629:
_ee_compensation_se3_from_map và _ee_compensation_se3 lấy compensation theo arm.
Nếu args bắt compensation thì IK target sẽ được biến đổi qua compensation.

Dòng 632-637, _ik_target_pose_se3:
Chuyển target_pose_base thành SE3 target cho IK, có áp dụng EE compensation nếu active.

Dòng 640-645, _ee_pose_base_from_ik_state:
Đọc EE pose hiện tại từ IK state, trả xyzrpy trong base frame.

Dòng 648-729, _coordinate_transform_from_anchor và _refresh_coordinate_transform_from_selection:
Hai hàm này tạo/refresh object coordinate transform từ robot anchor.
Nó lưu robot_world_R, inverse, world_to_robot, robot_to_world.
Đây là nền của mỗi công thức pose: object bbox trong world sẽ được đưa về robot base để DualArmIK solve.

Dòng 732-752, _resolve_link_prim_path:
Tìm prim path của link theo tên link.
Trả path và log candidates.

Dòng 755-800, _verify_ee_alignment_dynamic:
Số sánh body dynamic_control, prim USD, IK FK để xem EE frame có lệch không.
Kết quả dùng để bật/tắt compensation.

Dòng 803-884, _compute_ee_frame_delta_diagnostics:
Tính delta giữa IK EE frame và runtime/USD frame.
Ghi translation/rotation delta, source, per-arm diagnostic.

Dòng 887-907, _configure_ee_frame_compensation:
Dựa trên diagnostics và CLI, chọn có dùng compensation không.
Nếu active thì lưu map compensation vào args.

Dòng 910-985, _select_coordinate_transform_with_alignment:
Chọn coordinate transform tốt nhất, verify alignment, gắn compensation.
Đây là block quan trọng để tránh bug "pose dùng trong world nhưng sai trong base".

### Block 5: End-effector, fingertip, contact reference và runtime truth

**Dòng liên quan: 988-2492**

**Mục tiêu block:**
- Tìm body/link của EE và fingertip.
- Resolve midpoint hai ngón tay.
- Phần biet proxy, diagnostic proxy và close-critical runtime truth.
- Tính runtime two-finger metrics.
- Đổi contact reference target thành EE pose.

**Giải thích từng dòng/nhóm dòng:**

Dòng 988-994, _list_articulation_bodies:
Duyệt articulation bodies từ dynamic_control, trả index, handle, name/path.

Dòng 996-1027, _identify_end_effector_body:
Chọn body EE theo requested token/arm side.
Scoring dựa trên tên body/path để ưu tiên wrist/finger/palm phù hợp.

Dòng 1030-1042, _arm_side_match_score:
Cho điểm tên/path theo left/right arm.
Dùng để tránh chọn nhầm link đổi bên.

Dòng 1045-1070, _prim_bbox_center_world:
Tính bbox center world của prim.
Dùng khi body transform không có hoặc cần fallback từ USD.

Dòng 1073-1170, _resolve_named_body_or_prim_position:
Tìm position của một named body/prim.
Thứ từ logic: dynamic_control body trước, USD prim/bbox sau.
Trả cả world position và log source.

Dòng 1173-1176, _is_finger_midpoint_vertical_reference:
Kiểm tra token có yêu cầu midpoint finger/fingertip cho vertical reference không.

Dòng 1178-1271, _resolve_finger_midpoint_reference_position:
Resolve midpoint của hai finger link theo requested token.
Hàm này có thể dùng cho debug proxy midpoint.
Nó không mặc định là close-critical fingertip truth.

Dòng 1274-1362, _resolve_named_transform_position:
Resolve một transform theo tên/link/prim.
Dùng cho palm, link reference, vertical XY reference.

Dòng 1365-1402, _resolve_active_palm_reference_world:
Tìm palm reference của arm đang dùng.
Dùng khi cần offset từ palm/hand thay vì EE origin.

Dòng 1405-1448, _resolve_actual_fingertip_frame_world:
Tìm actual fingertip frame world.
Đây là nguồn runtime truth mạnh nhất nếu có đủ cặp tip1/tip2.

Dòng 1451-1515, _resolve_actual_fingertip_pair_midpoint_reference_position:
Lấy actual fingertip frame của hai ngón, tính midpoint.
Nếu thành công, log source là actual_fingertip_pair_midpoint và close-critical có thể true.

Dòng 1518-1643, _resolve_finger_link_distal_tip_proxy_world:
Tính distal tip proxy từ bbox/link geometry.
Đây là proxy hình học của đầu ngón tay, có ích để vẽ marker/so sánh.
Sau các fix gần đây, proxy này không được xem là close-critical truth nếu chưa runtime-validated.

Dòng 1646-1697, _resolve_single_fingertip_end_world:
Resolve một đầu ngón tay riêng lẻ.
Dùng bởi pair midpoint và diagnostics.

Dòng 1700-1775, _resolve_calibrated_distal_proxy_pair_midpoint_reference_position:
Tính midpoint từ calibrated distal proxy pair.
Nghĩa là "gần với đầu ngón" theo hình học, nhưng vẫn là proxy.
Theo log gần đây, đây chỉ là diagnostic-only nếu không được validate bằng actual/stable runtime reference.

Dòng 1778-1884, _resolve_fingertip_midpoint_reference_position:
Hàm tổng hợp các nguồn fingertip midpoint.
Thứ từ ưu tiên:
1. actual fingertip pair;
2. stable finger-link midpoint;
3. calibrated distal proxy diagnostic;
4. fallback nếu cần.
Đây là hàm trung tâm cho contact-centric truth hierarchy.

Dòng 1887-1907:
_finite_world_vector_or_none và _first_finite_vector_from_mapping lọc vector hợp lệ.
Chúng ngăn NaN/None làm gate sai.

Dòng 1910-2054, _compute_runtime_two_finger_metrics:
Tính metric runtime của hai ngón:
- tip1/tip2/tip_mid;
- tip axis;
- error của tip_mid đến object grasp center;
- xy/z error;
- symmetry error;
- z asymmetry;
- alignment error;
- primary_runtime_truth.
Đây là hàm quan trọng nhất của close gate.
Nếu primary_runtime_truth false, metric vẫn được log nhưng không nên authorize close như sự thật.

Dòng 2057-2080, _pose_for_contact_reference_world:
Nhận target contact reference world và contact_control_offset local.
Nó gọi logic tương đương _pose_for_point_b_world nhưng semantics đổi thành contact_reference_world_driven.
Công thức chính:
- rot_base = EulerXYZ(rpy).
- reference_base = world_to_robot(contact_reference_world).
- offset_base = rot_base @ contact_control_offset_local.
- point_A_base = reference_base - offset_base.
- pose = [point_A_base.x, point_A_base.y, point_A_base.z, rpy].
Ý nghĩa: DualArmIK vẫn điều khiển point_A/EE origin, nhưng target được định nghĩa theo contact reference.

Dòng 2083-2148, _resolve_vertical_finger_midpoint_reference_position:
Resolve vertical reference cho vertical policy.
Nó ưu tiên actual/stable midpoint và không dùng calibrated distal proxy làm truth.

Dòng 2151-2186, _upsert_two_finger_runtime_debug_markers:
Vẽ/update marker tip1, tip2, tip_mid, object center, runtime object grasp center.
Dùng để nhìn trực quan "robot đang close quanh đầu".

Dòng 2189-2208, _reference_comparison_payload:
Tạo payload so sánh giữa các reference.
Dùng trong log để biết point_B/proxy/real center lệch bao nhiêu.

Dòng 2211-2246, _resolve_finger_link_midpoint_diagnostic_world:
Resolve midpoint diagnostic của finger link.
Mục đích là so sánh, không phải primary close truth.

Dòng 2249-2489, _resolve_real_grasp_center_world:
Hàm trung tâm resolve real grasp center.
Nó áp dụng hierarchy:
- actual fingertip midpoint;
- stable finger-link midpoint;
- calibrated proxy diagnostic-only;
- fallback point_B/proxy.
Nó ghi rõ source, fallback_used, close_critical_reference.
Nếu close_critical_reference false thì close gate không nên coi đây là bằng chứng quyết định.

Dòng 2492-2514, resolve_real_grasp_center_world:
Wrapper public quanh _resolve_real_grasp_center_world.
Có thể dùng ngoài hàm private nếu cần debug/inspect.

### Block 6: Joint, gripper, IK loader, logging, bbox và workspace helper

**Dòng liên quan: 2517-2908**

**Mục tiêu block:**
- Đọc current vertical reference.
- Đọc joint state.
- Chọn DOF arm/gripper.
- Load official startup joint map và IK classes.
- Apply effort gripper.
- Chạy simulation ticks.
- Ghi phase/log JSON.
- Helper bbox/workspace.

**Giải thích từng dòng/nhóm dòng:**

Dòng 2517-2563:
_resolve_current_vertical_xy_reference_world và _resolve_vertical_xy_reference_offset tính reference XY runtime cho vertical descent.
Chúng cho phép align theo midpoint/fingertip thay vì point_B đơn thuần.

Dòng 2618-2634:
_current_positions, _named_positions, _read_positions, _targets_from_map là helper đọc/chuyển đổi joint target.

Dòng 2637-2644, _all_joint_state_for_ik:
Đọc joint names và positions từ articulation để sync vào IK solver.
Nếu joint state fail, IK sẽ solve trên state sai, nên hàm này là bước bắt buộc trước solve.

Dòng 2647-2656, _select_dofs_in_name_order:
Chọn DOF theo danh sách tên có thứ tự.
Đảm bảo target vector khớp đúng joint.

Dòng 2659-2674, _select_gripper_dofs:
Chọn finger DOF theo arm side.
Dùng cho open/close và effort hold.

Dòng 2677-2698, _select_dofs_by_target_names:
Chọn DOF cho arm target names.
Nếu thiếu DOF, fail để tránh command nhầm.

Dòng 2701-2715, _load_official_startup_joint_map:
Load joint value map từ official baseline.
Mục tiêu là startup posture dùng official, không tự chế local pose.

Dòng 2718-2727, _load_official_ik_classes:
Load DualArmIK/CoordinateTransform từ official baseline.
Đây là điểm script reuse official IK backend.

Dòng 2730-2751, _seed_joint_positions_for_initialization:
Apply startup joint positions để đưa robot về trạng thái ban đầu.

Dòng 2754-2778, _apply_gripper_effort:
Set effort cho gripper DOF.
Sử dụng trong close/retention/lift để giữ object, không chỉ position target.

Dòng 2781-2797, _run_updates:
Step simulation, tăng counter, có thể hold GUI.

Dòng 2800-2802, _gripper_values:
Đọc giá trị gripper hiện tại vào dict.

Dòng 2804-2821, _append_phase:
Ghi một phase vào phase_log với start/end step, condition_met, details.
Đây là nền của debug timeline.

Dòng 2824-2835, _json_safe:
Chuyển numpy/path/object thành kiểu JSON-safe.

Dòng 2838-2868, _write_logs:
Ghi payload JSON ra LOG_ROOT, có rolling log và timestamp log.

Dòng 2871-2873, _finite:
Check list float hợp lệ.

Dòng 2875-2881, _bbox_state:
Lấy bbox và center của prim.

Dòng 2884-2886, _center_from_bbox:
Tính center = (min + max) / 2.

Dòng 2888-2890, _distance:
Tính Euclidean distance.

Dòng 2892-2903, _workspace_check:
Check x/y/z có nằm trong limit không, trả dict pass/error.

Dòng 2906-2908, _table_units:
Đổi meter sang table unit nếu cần log/debug.

### Block 7: Table frame và object info trong table frame

**Dòng liên quan: 2911-3147**

**Mục tiêu block:**
- Tính frame của table.
- Đổi world <-> table.
- Lấy bbox object theo table frame.

**Giải thích từng dòng/nhóm dòng:**

Dòng 2911-2922, _bbox_corners_from_bbox:
Tạo 8 corner của bbox từ min/max.
Dùng để transform bbox sang table frame.

Dòng 2925-2933, _axis_aligned_with_world_xy:
Check hai trục table có gần song song world XY không.
Nếu table gần axis-aligned, có thể đơn giản hóa frame.

Dòng 2936-2972, _inspect_table_xform_axes:
Lấy trục x/y/z của table prim từ transform.
Ghi orientation, axis alignment, source.

Dòng 2975-3057, build_or_resolve_table_frame:
Dùng table path và bbox để xây table_frame.
Output gom origin_world, x_axis_world, y_axis_world, z_axis_world, table_top_z, source.
Đây là frame chuẩn để tính object pose, clearance và table z.

Dòng 3060-3069, world_to_table:
Trừ vector world với origin table, dot với các axis table, trả coordinate trong table frame.

Dòng 3072-3080, table_to_world:
Nhận point table theo x/y/z và cộng origin + x_axis*x + y_axis*y + z_axis*z.

Dòng 3083-3099, _bbox_in_table_frame:
Transform 8 bbox corner sang table frame, tính min/max/center mới.

Dòng 3102-3147, get_object_info_in_table_frame:
Lấy bbox world của object, tính bbox table, center world/table, top/bottom, size.
Đây là object_info chính cho grasp frame và geometry filter.

### Block 8: Candidate generation và Phase 2 geometric grasp filter

**Dòng liên quan: 3150-3572**

**Mục tiêu block:**
- Tạo candidate arm/approach.
- Score nhanh theo reach/yaw/width.
- Ước lượng object grasp frame.
- Đủ đoạn contact asymmetry.
- Kiểm tra table clearance và target gap.
- Chọn Phase 2 candidate.

**Giải thích từng dòng/nhóm dòng:**

Dòng 3150-3154, _candidate_arm_list:
Nếu requested_arm là left/right thì trả một arm.
Nếu auto/both thì trả ca hai.

Dòng 3156-3158, _preferred_arm_from_components:
Chọn arm dựa trên lateral component của target trong robot base.

Dòng 3160-3220, generate_approach_candidates_for_object:
Tạo candidate theo arm, yaw preset, object info.
Mỗi candidate có contact point, approach vector, width estimate, side penalty.
Đây là lớp coarse Phase 1, trước khi vào contact-centric.

Dòng 3223-3261, fast_score_candidate:
Tính score nhanh:
- reach distance;
- side penalty;
- yaw penalty;
- width penalty nếu width ngoài min/max.
Score càng thấp cang tốt.

Dòng 3264-3293, select_best_candidate:
Sắp xếp candidates theo score và chọn best.
Trả cả selected candidate và summary.

Dòng 3296-3301, _clamp_norm:
Giới hạn norm vector theo max_norm.
Hàm này được dùng lại trong final descent step clamp.

Dòng 3304-3308, _angle_between_axes_unsigned:
Tính gốc không hướng giữa hai axis.
Dùng cho alignment.

Dòng 3310-3318, _table_axis_to_world:
Đổi axis table sang world theo table_frame.

Dòng 3320-3380, estimate_object_grasp_frame:
Dùng object bbox/table frame để suy ra:
- object grasp center;
- major/minor/width axis;
- top/support;
- estimated width.
Đây là "hình học vật" cho filter.

Dòng 3383-3386, _candidate_hand_closing_axis_table:
Lấy closing axis của candidate trong table frame.

Dòng 3388-3412, predict_early_contact_asymmetry:
Đủ đoạn hai ngón sẽ chạm vật lệch báo nhiều nếu theo candidate.
Nếu asymmetry lớn, candidate có nguy cơ đẩy vật hơn là gắp.

Dòng 3415-3427, estimate_table_clearance_margin:
Tính margin giữa fingertip/contact geometry với table.
Dùng để chặn candidate có khả năng cắm xuống mặt bàn quá sâu.

Dòng 3430-3446, compute_target_gap:
Tính target gripper gap dựa trên object width + margin, clamp trong min/max.

Dòng 3449-3512, fast_geometric_grasp_filter:
Check bắt buộc:
- width trong rànge;
- alignment error <= max;
- symmetry error <= max;
- predicted contact asymmetry <= max;
- table clearance >= min.
Trả pass/fail, violation list, weighted score.

Dòng 3515-3572, select_best_phase2_candidate:
Chạy filter cho tất cả candidates.
Mặc định PHASE2_ALLOW_LEAST_BAD_CANDIDATE = False.
Nếu không có candidate pass mandatory, fail sớm thay vì lấy least-bad và lao vào descent.
Đây là thay đổi an toàn quan trọng của contact-centric patch.

### Block 9: Target/bin utility, target selection, TCP offset, region classification, orientation preset selection

**Dòng liên quan: 3575-4115**

**Mục tiêu block:**
- Check inside bin.
- Settle object.
- Tạo debug marker.
- Tính target component trong robot base.
- Chọn arm.
- Chọn target object.
- Resolve TCP offset.
- Phân loại target region và chọn orientation preset.

**Giải thích từng dòng/nhóm dòng:**

Dòng 3575-3582, _inside_bin:
Kiểm tra center object có nằm trong bin bbox trừ wall/floor margin không.

Dòng 3585-3599, _settle_and_measure:
Step physics settle và đo pose/jitter cuối.

Dòng 3602-3610, _create_debug_marker:
Tạo sphere marker tại position với radius/color.

Dòng 3613-3635, _upsert_debug_marker:
Nếu marker đã có thì update transform, nếu chưa có thì tạo.

Dòng 3638-3652, _compute_robot_base_target_components:
Chuyển object_world về robot base.
Tính forward, lateral, distance.
Đây là input cho near/mid/far và arm side.

Dòng 3655-3658, _choose_arm_side:
Nếu CLI chọn arm thì dùng; nếu auto thì dựa vào target_components.

Dòng 3661-3672, _category_from_target:
Suy ra category A/B theo index hoặc reference.

Dòng 3675-3683, _target_nearness_sort_key:
Sort key để ưu tiên target gần/rõ ràng.

Dòng 3686-3739, _build_target_candidate_records:
Lấy các prim part, bbox, center, category, sort key.

Dòng 3742-3802, _select_target_record:
Chọn target theo target-index hoặc policy.
Trả record có path, bbox, category, object_info.

Dòng 3805-3830, _resolve_tcp_offset:
Lấy TCP offset từ config/CLI/fallback.
Nếu không có offset chính thức, dùng fallback x.

Dòng 3833-3853:
_euler_xyz_to_rot và _rot_to_euler_xyz lặp lại conversion cho runtime pose.

Dòng 3856-3861, _normalize:
Normalize vector, nếu norm quá nhỏ thì dùng fallback.

Dòng 3864-3872, _approach_axis_from_mode:
Lấy trục approach từ rotation theo args.approach_axis_mode.

Dòng 3875-3878, _base_down_vector:
Tính vector down trong base frame từ coordinate transform.

Dòng 3880-3885, _classify_target_region:
Dùng forward_base và threshold CLI để phân target thành near_body/mid/far.
Đây ảnh hưởng trực tiếp đến motion_family.

Dòng 3888-3895, _approach_family_order_for_region:
Nếu far thì ưu tiên world_y_approach.
Nếu mid/near thì ưu tiên z_approach/vertical.

Dòng 3898-3908, _raw_orientation_presets_by_arm_and_family:
Trả map preset theo arm và family.

Dòng 3911-3916, _debug_fixed_rpy_for_arm:
Cho phép CLI force RPY debug cho một arm.

Dòng 3919-3938, _world_y_axis_diagnostics:
Tính dot approach axis với world +/-Y để debug family world_y.

Dòng 3941-3951, _preset_axial_roll_metadata:
Parse label axial roll variant.

Dòng 3954-3981, _orientation_preset_record:
Tạo record preset đầy đủ:
- label/index/family/rpy/rotation;
- approach axis base/world;
- up axis;
- dot với world +/-Y;
- axial roll metadata.

Dòng 3984-3997, _orientation_presets_by_arm_and_family:
Tạo list preset records cho arm/family.

Dòng 4000-4055, _region_filtered_orientation_presets:
Lọc preset theo target_region và approach_family_order.
Nếu debug fixed RPY active thì chỉ dùng preset debug.

Dòng 4058-4115, _fixed_downward_rpy_by_arm:
Tạo downward RPY mặc định theo arm.
Dùng cho geometry nếu không dùng preset library riêng.

### Block 10: Pose construction, point_B semantics và grasp geometry

**Dòng liên quan: 4118-4704**

**Mục tiêu block:**
- Đổi contact/point_B target world thành EE pose base.
- Định nghĩa point_A/point_B semantics.
- Tính contact_z.
- Lập kế hoạch pregrasp/align/contact/lift/carry/place/retreat cho far và vertical policy.

**Giải thích từng dòng/nhóm dòng:**

Dòng 4118-4149, _pose_contact_base:
- Dòng 4125 tính rotation base từ rpy.
- Dòng 4126 đổi contact_world sang robot base.
- Dòng 4127-4128 thêm bias forward/lateral trong base.
- Dòng 4130 tính tcp_offset_base = rot @ tcp_offset_local.
- Dòng 4132-4135 nếu target_mode == contact_axis thì EE origin trung contact_base, không compensate TCP.
- Dòng 4136-4138 ngược lại, EE origin = contact_base - tcp_offset_base.
- Dòng 4140-4142 tạo pose [x,ý,z,r,p,ý].
- Dòng 4143-4149 trả pose và log contact_world/contact_base/tcp_offset/target_mode.
Hàm này là path cũ theo contact axis/TCP, không phải contact-reference runtime truth mới.

Dòng 4153-4164:
_pose_position_world, _pose_rotation_base, _pose_rotation_world.
Ba hàm này đổi pose_base thành world position/rotation để log point_A/point_B.

Dòng 4166-4182, _resolve_point_b_offset_local:
- Nếu CLI cũng cặp point_b_offset_local thì dùng.
- Nếu không, lấy length = max(norm(tcp_offset), tcp_fallback_x).
- Đặt offset = [0,0,length].
- Log source và note: local +Z được dùng vì preset top-down hiện tại làm trục đó gần vertical world.
Đây là cách script suy ra point_B compatibility.

Dòng 4185-4192:
_point_world_from_pose và _point_b_world_from_pose.
Công thức:
point_A_world = robot_to_world(pose_base[:3])
rot_world = robot_world_R @ rot_base
point_B_world = point_A_world + rot_world @ point_b_offset_local.

Dòng 4195-4224, _point_b_target_for_xy_reference:
- Bắt đầu từ point_b_world.
- Nếu không có xy_reference_offset_local thì giữ XY point_B.
- Nếu có, tính delta_world = rot_world @ (xy_reference_offset_local - point_b_offset_local).
- Đặt target.x/y = desired_xy - delta_world.x/y.
Mục tiêu: point_B vẫn control Z/contact mark, nhưng XY của reference link/fingertip mỗi nam trên vật.

Dòng 4227-4245, _pose_for_point_b_world:
- Dòng 4233 tính rot_base.
- Dòng 4234 đổi point_B target world sang base.
- Dòng 4235 tính point_b_offset_base.
- Dòng 4236 tính point_A_base = point_B_base - point_b_offset_base.
- Dòng 4237 tạo pose [point_A_base, rpy].
- Dòng 4238-4245 log target point_B, offset, point_A, semantics point_B_proxy_driven.
Đây là công thức chính của geometry planning trước contact-centric final descent.

Dòng 4248-4276, _ab_pose_semantics:
Tính point_A_world, point_B_world, AB vector, length, AB axis, dot với world z, slant angle, dot với world +/-Y.
Đây là diagnostic để biet hand đang nằm ngang/dọc/lệch như thế nào.

Dòng 4279-4285, _target_world_from_pose_key:
Lấy pose trong geometry theo key rồi tính AB semantics.

Dòng 4288-4298, _compute_contact_z_world:
Công thức:
contact_z = max(
    bbox_top_z + descend_clearance + grasp_depth_offset,
    table_top_z + min_ee_table_clearance
)
Ý nghĩa:
- không đi thấp hơn table_top + min clearance;
- grasp_depth_offset có thể am để chèn sau hơn;
- nhưng trong sweep target-index=2, thay grasp_depth không đổi pregrasp_error, nên nó không phải lever của approach.

Dòng 4301-4637, _plan_grasp_geometry:
Đây là block lập kế hoạch geometry lớn nhất.

Dòng 4320-4324:
Lấy bbox, object center, đặt contact_world ban đầu bằng object center.
Thêm contact_world_y_bias.
Đặt contact_world.z bằng _compute_contact_z_world.

Dòng 4325:
object_support_z_world = max(table_top_z, bbox_min_z).
Đây là mốc support để tính gắp gần mặt bàn/vật.

Dòng 4327-4335:
Lấy rpy preset, point_b_offset, rotation, x/y axis base, world_up, base_up.

Dòng 4336-4345:
Tính AB vector world từ local point_B offset.
Tính horizontal span của AB và các biến far extra height/slant.

Dòng 4346-4356:
Nếu motion_family == world_y_approach, tính thêm point_A extra height để đặt requested downward slant.
Nếu cận, điều chỉnh point_b_offset local để point_B thấp hơn point_A theo world up.

Dòng 4357-4363:
Tính AB axis base/world, dot với world_z, horizontal norm, downward slant deg.

Dòng 4364-4368:
Tính robot_belly_forward_world từ base +X.
Dùng để xem AB có vuong gốc/hướng dùng với than robot không.

Dòng 4374-4438, nhanh far/world_y_approach:
- motion_policy = far_low_side_B_driven.
- far_contact_z_world = object_support_z + far_point_b_gap_above_support.
- far_reach_axis_world = horizontal projection của AB axis.
- far_xy_align_z_world = object_top_z + far_xy_align_clearance_above_object.
- far_xy_align_b_world = object XY, z = align z.
- contact_b_world = same XY, z = support gap.
- legacy_side_contact_b_world = contact_b_world + reach_axis * forward_extension.
- low_side_prepare_b_world = legacy side contact tru standoff, z = align z.
- far_outboard_transition_b_world = low_side_prepare + outboard offset + vertical clearance.
- pregrasp_b_world = low_side_prepare.
- align_b_world = far_xy_align.
Ý nghĩa: target xa không lao thang từ trên xuống, ma đi qua outboard/low-side/align rồi mỗi descend theo world Z.

Dòng 4440-4491, nhanh vertical/mid/near:
- motion_policy = mid_vertical_Z_descend hoặc near_body_vertical_Z_descend.
- vertical_contact_z_world = object_support_z + vertical_point_b_gap_above_support.
- vertical_uncorrected_reference_world = [contact_world.x, contact_world.y, vertical_contact_z].
- Tính lateral correction theo arm: right arm shift -baseY, left arm shift +baseY.
- vertical_xy_reference_world = uncorrected XY + correction.
- raw_contact_b_world = vertical_xy_reference_world.
- Gọi _point_b_target_for_xy_reference để điều chỉnh point_B target nếu có reference link offset.
- pregrasp_b_world = contact_b_world + world_up * pregrasp_clearance.
- align_b_world = contact_b_world + world_up * align_clearance.
Ý nghĩa: target mid/near là vertical descent, có bias lateral để sửa lệch inward của arm.

Dòng 4493-4512:
Convert pregrasp/align/contact/far_outboard B-world sang pose_base bằng _pose_for_point_b_world.

Dòng 4513-4522:
Tạo micro_lift và lift point_B world/pose.
Lift = contact_b_world + world_up * lift_height.

Dòng 4524-4537:
Tạo carry pose ở bin center, z = bin max z + safe_drop_height.
Tạo place pose ở bin center, z = max(bin_floor_top + 0.12, bin max + place_clearance).
Retreat pose = place pose + base_up * retreat_lift.

Dòng 4539-4637:
Return geometry dict rất lớn.
Nó luu:
- arm side, target region, motion family/policy;
- object center, bbox top, support z;
- contact z, rpy, axes;
- point_A/point_B definitions;
- far/vertical policy details;
- pregrasp/align/contact/lift/carry/place/retreat poses;
- world EE origins;
- AB semantics;
- carry/place details;
- approach axis.
Đây là payload debug chính để truy nguồn mỗi target pose.

Dòng 4640-4674, _plan_grasp_geometry_for_preset:
Wrapper gọi _plan_grasp_geometry với rpy từ orientation preset.
Thêm orientation_source và orientation_preset vào geometry.

Dòng 4677-4698, _pregrasp_candidates:
- Lấy base pregrasp_pose_base.
- Tạo 2 variant: nominal và slightly_higher.
- slightly_higher = base_up * candidate_higher_offset.
Nếu pregrasp unreachable, đây là nơi chỉ có hai biến thể height nhỏ, chưa có staging pose hay multi-candidate XY.

### Block 11: IK solve, candidate diagnostics và pregrasp acceptance

**Dòng liên quan: 4701-5330**

**Mục tiêu block:**
- Sync IK từ dynamic_control.
- Solve single-arm pose.
- Classify candidate failure.
- Chọn pregrasp candidate đầu tiên pass strict tolerance.
- Ghi diagnostics khi không có candidate pass.

**Giải thích từng dòng/nhóm dòng:**

Dòng 4701-4704, _sync_ik_from_dc:
Đọc all joint state và sync vào ik_solver.
Đây là bước đầu của mỗi candidate evaluation.

Dòng 4707-4714, _current_ee_pose_base:
Lấy EE pose hiện tại trong base frame.

Dòng 4717-4741, _ee_pose_base_for_arm_solution:
Gần q solution vào solver, tính FK EE pose, trả xyzrpy.

Dòng 4744-4756, _pose_error:
Tính position error và rotation error giữa reached pose và target pose.

Dòng 4759-4772, _limit_joint_delta:
Giới hạn delta joint mỗi tick để servo không giật.

Dòng 4775-4788, _ik_kwargs:
Lấy các tham số IK từ args và override.

Dòng 4791-4803, _solve_single_arm_pose:
Gọi official IK solve cho một arm.
Trả q_sol và ok.

Dòng 4806-4812, _candidate_ik_overrides:
Trả override riêng cho candidate search, vì candidate có thể cần tolerance/iteration khác servo.

Dòng 4815-4822, _finite_vector:
Check vector finite và dùng size.

Dòng 4825-4832, _candidate_failure_summary:
Đếm các loại failure candidate.

Dòng 4835-4864, _candidate_acceptance_tolerances:
Chọn tolerance theo target_region.
Far có thể dùng far_candidate_position_tolerance và far_candidate_rotation_tolerance.

Dòng 4867-4872, _finite_float_or_none:
Chuyển value sang float nếu finite, ngược lại None.

Dòng 4875-4906, _candidate_gate_metrics:
Tính pass/fail theo position/rotation tolerance.

Dòng 4909-4977, _candidate_diagnostic_view:
Rút gọn candidate để log để đọc.

Dòng 4980-5001, _best_candidate_by_metric và _best_candidate_diagnostics:
Tìm candidate tốt nhất theo position/rotation/combined error.
Đây là payload cần xem khi tất cả fail.

Dòng 5004-5022, _classify_candidate_result:
Phân loại:
- catastrophic_no_solution nếu q/reached/error invalid.
- solved_but_internal_not_ok nếu IK có solution nhưng solver ok false.
- candidate_error_exceeded_tolerance nếu solve được nhưng error vượt tolerance.
- valid_candidate nếu pass.

Dòng 5025-5330, _evaluate_pregrasp_candidates:
Đây là nơi pregrasp unreachable được quyết định trong contact-centric file.

Dòng 5049:
Ghi start_step để phase log.

Dòng 5050:
Sync IK từ dynamic_control.

Dòng 5051-5052:
Lưu reference_q và reference_pose để reset solver sau mỗi candidate.

Dòng 5053:
Lấy candidate_pos_tol/candidate_rot_tol theo region.

Dòng 5054-5057:
Khởi tạo preset_results, flat_results, selected, IK settings.

Dòng 5059-5078:
Với từng orientation preset, gọi _plan_grasp_geometry_for_preset để tạo geometry.

Dòng 5079:
Lấy candidate_point_b_offset từ geometry.

Dòng 5081:
Duyệt _pregrasp_candidates, hiện chỉ có nominal và slightly_higher.

Dòng 5082:
Reset ik_solver.q về reference_q để mỗi candidate công bằng.

Dòng 5083-5084:
Lấy target_pose và convert sang target_pose_for_ik để log.

Dòng 5085-5092:
Gọi _solve_single_arm_pose.

Dòng 5093-5094:
Tính reached_pose và pose error.

Dòng 5095-5097:
Check q solution, reached pose, error có finite không.

Dòng 5098-5106:
Classify candidate bằng _classify_candidate_result.

Dòng 5108-5115:
valid true khi:
- q valid;
- reached pose valid;
- error finite;
- position error <= tolerance;
- rotation error <= tolerance.
Lưu ý valid không yêu cầu ok true nếu solved và error pass, nhưng log accepted_despite_internal_not_ok.

Dòng 5117-5121:
Nếu valid thì failure_reason None, ngược lại failure_reason = classification.

Dòng 5122-5199:
Tạo result dict chi tiết:
- preset label/index/family/rpy;
- AB axis/far/vertical geometry;
- reference pose;
- target pose;
- dualarmik status;
- errors/tolerance;
- q solution;
- target point_A/B world;
- accepted_for_pregrasp_selection.

Dòng 5200-5205:
Tạo selection_record gom result + geometry + selected_orientation_preset.

Dòng 5206-5214:
Nếu valid và chưa có selected, chọn candidate đầu tiên strict valid.

Dòng 5215-5275:
Tạo preset_summary và preset_results cho log từng preset.

Dòng 5277:
Reset ik_solver.q về reference_q sau khi search.

Dòng 5278-5292:
Tính success và failure_reason tổng hợp:
- no_pregrasp_candidates_generated;
- all_candidates_catastrophic_no_solution;
- reachable_candidates_failed_strict_acceptance;
- mixed_candidate_reachability_failure.

Dòng 5293-5323:
Tạo details log với best candidate diagnostics, tolerance, preset count, candidate_results, selected fields.

Dòng 5324-5329:
Append phase select_pregrasp_candidate.

Dòng 5330:
Nếu không success, _fail("pregrasp_candidate_failed", ...).
Đây là failure cấp run. Nếu các log nói pre_grasp_unreachable trong script khác, logic gốc vẫn từ nhóm check pregrasp IK/tolerance này.

### Block 12: Servo executor và gripper phase

**Dòng liên quan: 5333-5857**

**Mục tiêu block:**
- Thức thì một phase servo DualArmIK.
- Hỗ trợ target pose động mỗi tick.
- Refresh coordinate transform.
- Stop theo metric EE/point_B/contact reference.
- Log trace, final error, early stop.

**Giải thích từng dòng/nhóm dòng:**

Dòng 5333-5763, _execute_dualarmik_servo_phase:
Đây là executor trung tâm cho pregrasp, align, descend, lift, carry, place.

Dòng 5333-5355:
Signature gom ServoSpec, ik_solver, dc, articulation, arm DOF, coord_transform, gripper, sim_app, args, phase_log, EE info.
Có target_pose_fn optional, position_metric_offset_local và position_metric_label.
Hai tham số metric offset này là fix quan trọng gần đây để final descent stop theo contact reference thay vì EE.

Dòng 5356-5365:
Khởi tạo start step, trace, best error, current q/pose.

Dòng 5366-5395:
Nếu có target_pose_fn thì mỗi tick lấy target pose mới.
Nếu có coord_transform_refresh_fn thì refresh transform để robot/world frame không bị stale.

Dòng 5396-5428:
Solve IK cho target pose, limit joint delta, gui position targets.

Dòng 5429-5468:
Step sim, đọc pose hiện tại, tính position/rotation error.
Nếu position_metric_offset_local có, metric point = EE pose + rot @ offset.
Nếu không, fallback point_B hoặc EE.
Đây là nối contact-centric final descent bước servo dùng đúng metric.

Dòng 5469-5508:
Cập nhật best error, trace sample theo interval.

Dòng 5509-5536:
Kiểm tra tolerance; nếu pass và hold đủ thì condition_met true.

Dòng 5537-5575:
Xử lý ServoEarlyStop nếu per_tick_monitor hoặc target_pose_fn raise.
Early stop vẫn được log với reason/details.

Dòng 5576-5759:
Append phase log với:
- target pose;
- final pose/error;
- best pose/error;
- trace;
- metric label;
- extra details.
Trả result dict.

Dòng 5765-5833, _command_gripper_phase:
Mô/dòng gripper bằng position target và effort optional.
Ghi gripper values, effort, steps.

Dòng 5836-5849, _resolve_torso_prim_path:
Resolve torso prim path từ robot root/articulation.

Dòng 5852-5857, _work_area_world_from_cfg:
Lấy work area world từ config/table bbox.

### Block 13: Pre-close gate và execute_pregrasp

**Dòng liên quan: 5860-6263**

**Mục tiêu block:**
- Thu thập evidence trước khi close.
- Tính real grasp center, runtime two-finger metrics, table clearance, selected candidate filter.
- Execute pregrasp servo với fallback yaw debug.

**Giải thích từng dòng/nhóm dòng:**

Dòng 5860-6176, _pre_close_gate:
Đây là hàm gom diagnostics trước close.

Dòng 5860-5885:
Signature nhận stage, target, geometry, transform, IK/DC/articulation, arm/gripper, table/object/filter.

Dòng 5886-5905:
Đọc current EE pose, current point_B, current target/object state.

Dòng 5906-5935:
Resolve real grasp center bằng _resolve_real_grasp_center_world.
Nếu source close-critical false, log vẫn có nhưng không được coi là proof close.

Dòng 5936-5965:
Build object_grasp_frame runtime và gọi _compute_runtime_two_finger_metrics.

Dòng 5966-6005:
Update debug markers tip/object/contact.

Dòng 6006-6065:
Tính table clearance, orientation/alignment, support gap, drift, selected candidate pass/fail.

Dòng 6066-6176:
Return payload đầy đủ:
- close_critical_uses_real_grasp_center;
- real_grasp_center source;
- point_B delta;
- runtime metrics;
- geometry/filter;
- table clearance;
- object state.
Đây là input của evaluate_close_gate.

Dòng 6179-6263, execute_pregrasp:
Wrapper gọi _execute_dualarmik_servo_phase cho pregrasp.
Nếu pregrasp IK fail theo yaw/orientation có thể fallback yaw=0 trong một số câu hình.
Mục đích là tach pregrasp servo khởi final descent contact-centric.

### Block 14: Final descent local IK contact-centric

**Dòng liên quan: 6266-6922**

**Mục tiêu block:**
- Dùng closed-loop descent dựa trên contact reference runtime.
- Clamp step XY/Z/yaw.
- Không cho Z command tăng len trong descent.
- Dùng contact reference offset để chuyển target về EE pose cho DualArmIK.
- Log samples chi tiết để quyết định close fallback.

**Giải thích từng dòng/nhóm dòng:**

Dòng 6266-6302:
Signature final_descent_local_ik nhận phase, scene, target, geometry, transform, IK/DC, robot root, arm/gripper, target locked, rpy locked, point_B offset, table/object/filter.

Dòng 6303:
Docstring: local closed-loop descent đo real grasp center mới target refresh.

Dòng 6304-6316:
Chuyển locked target/rpy/point_B offset sang numpy.
Lấy object_center_world từ geometry nếu có, nếu không NaN.
Lấy step limit:
- phase2_descent_xy_step;
- phase2_descent_z_step;
- phase2_descent_yaw_step.

Dòng 6317-6332:
Khởi tạo samples, debug samples, call_count, previous_commanded_world, last_yaw, vertical tip stop rule state.

Dòng 6333-6337:
Đọc initial EE pose/world rotation/point_B world.

Dòng 6338-6348, nested support_gap_m:
Tính support gap = world_pos.z - object_support_z.
Nếu không có object_support_z hoặc pos invalid thì None.

Dòng 6350-6361:
Resolve initial_real_center_world.
Fallback world là initial point_B.
Có diagnostic bypass flag nếu bắt.

Dòng 6362-6372:
use_real_contact_offset true chỉ khi:
- real_center_world có;
- source không fallback_proxy;
- close_critical_reference true.
Nếu true, contact_control_offset = initial_ee_rot_world.T @ (contact_control_world - initial_ee_world).
Nếu false, contact_control_offset = point_b_offset.
Đây là quyết định: final descent sẽ điều khiển tip_mid/contact reference thật nếu tin cậy, ngược lại fallback point_B.

Dòng 6373-6379:
Tính delta point_B -> contact_control_reference.
Tạo nominal target pose bằng _pose_for_contact_reference_world.

Dòng 6381-6437, nested _update_proxy_middle_point_debug_marker:
Resolve legacy finger-link midpoint và vẽ marker debug.
Nếu unresolved, ghi reason.
Nếu resolved, update marker /World/DebugProxyMiddlePoint và lưu component positions.

Dòng 6439-6551, nested monitor_vertical_tip_stop_rule:
Theo dõi tip_mid table z trong vertical descent.
Nếu tip table z <= threshold thì raise ServoEarlyStop.
Log tip1/tip2/tip_mid table z, object top/support delta, source, fallback.
Quan trọng: đây là stop/diagnostic rule, không phải auto-pass close nếu close-critical truth không đủ.

Dòng 6553-6760, nested contact_target_pose_fn:
Đây là vòng điều khiển contact-centric mỗi tick.

Dòng 6554-6557:
Tăng call_count và đọc current EE pose.

Dòng 6558-6567:
Resolve real_center_world mỗi tick.

Dòng 6568:
Tính current_point_b_world từ pose hiện tại.

Dòng 6569-6573:
trusted_real_center_for_descent true nếu real center có và close_critical_reference true.

Dòng 6574-6588:
Nếu trusted, lấy fingertip_midpoint_world.
Nếu không có midpoint nhưng có component tip1/tip2, tính midpoint = 0.5*(tip1+tip2).

Dòng 6589-6597:
Nếu tip_mid available, measured_world = tip_mid và source = tip_mid.
Ngược lại measured_world = current_point_b_world và source = point_B fallback.

Dòng 6598-6614:
Build descent_object_grasp_frame, thêm object_grasp_center_world và các axis nếu thiếu.

Dòng 6615-6622:
Tính close_runtime_metrics bằng _compute_runtime_two_finger_metrics.

Dòng 6624-6628:
Gọi monitor_vertical_tip_stop_rule.

Dòng 6630:
delta = locked_target - measured_world.

Dòng 6631:
xy_step = clamp_norm(delta[:2], xy_step_max).

Dòng 6632:
z_step = clamp(delta.z, giới hạn [-z_step_max, 0]).
Nghĩa là chỉ cho đi xuống hoặc đứng yên, không cho đi len.

Dòng 6633-6635:
commanded_world = measured_world + xy_step và z_step.

Dòng 6636-6637:
Nếu command z cao hơn previous_commanded z thì ep bằng previous.
Đây là monotonic Z downward policy.

Dòng 6639-6643:
Clamp yaw delta về [-yaw_step_max, yaw_step_max].
Cập nhật commanded_rpy.

Dòng 6644-6649:
Chuyển commanded contact reference world thành EE pose base bằng _pose_for_contact_reference_world.
Lưu previous_commanded_world và commanded_delta_to_target.

Dòng 6651-6759:
Theo trace interval, log sample rất chi tiết:
- locked target;
- real grasp center;
- source;
- tip1/tip2/tip_mid;
- runtime metrics;
- point_B proxy;
- measured/commanded world;
- support gap;
- z/xy error;
- command step;
- pose conversion;
- control_reference_source;
- monotonic_z_downward;
- yaw step.

Dòng 6761-6817:
Gọi _execute_dualarmik_servo_phase với:
- ServoSpec phase final descent;
- target_pose_fn = contact_target_pose_fn;
- position_metric_offset_local = contact_control_offset;
- position_metric_label = contact_reference_world;
- per_tick_monitor_fn nếu vertical tip rule active;
- extra details về contact_centric_command_path.
Đây là nối final descent gần target dòng vào servo executor.

Dòng 6818-6868:
Sau servo, tính xy_drift_stable, max_xy_step, recent_xy_drift, support_gap_values.

Dòng 6869-6898:
Tính z-stall:
- recent_z_gap_change;
- recent_z_progress;
- recent_z_motion_abs;
- stalled_in_z nếu recent_z_progress <= threshold.
Đây là evidence cho vertical/runtime commit fallback.

Dòng 6899-6919:
Tạo phase2_log với samples, marker updates, vertical stop samples, drift, support gap, z-stall policy.

Dòng 6920-6922:
Gần phase2_local_descent vào result và phase_log cuối, trả result.

### Block 15: Close gate, vertical fallback, runtime fallback, close debug summary

**Dòng liên quan: 6925-7330**

**Mục tiêu block:**
- Quyết định có được close hay không.
- Phần biet hard blockers và soft warnings.
- Cho fallback khi robot đã chạm/gần support gap và stalled in Z.
- Tạo summary ngắn để log.

**Giải thích từng dòng/nhóm dòng:**

Dòng 6925-7069, evaluate_close_gate:
Hàm này là primary close gate.

Dòng 6925-6950:
Nhận pre_close payload, geometry/filter, args.
Đọc runtime metrics và close-critical flag.

Dòng 6951-6990:
Check hard blockers:
- close-critical real grasp center không tin cậy;
- catastrophic orientation;
- catastrophic table clearance;
- width ngoài rànge;
- runtime metric không primary khi cần.

Dòng 6991-7025:
Check pass điều kiện:
- real center/tip_mid error <= tolerance;
- orientation <= tolerance;
- xy drift trong ngưỡng;
- table clearance pass;
- candidate filter pass hoặc chỉ warning tùy loại.

Dòng 7026-7069:
Return dict có close_allowed, reasons, warnings, runtime_truth, thresholds.
Ý nghĩa: candidate metric có thể là warning, nhưng runtime truth và hard blockers quyết định close.

Dòng 7072-7203, evaluate_vertical_support_or_stall_close_fallback:
Fallback cho trường hợp final descent đã gần support và z không tiến thêm.
Điều kiện chính:
- support gap trong [min, max];
- stalled_in_z;
- xy drift ổn;
- orientation ok;
- far policy chỉ được nếu allow_far_policy true.
Đây không phải relax tùy tiện; nó yêu cầu gần contact/support rõ ràng.

Dòng 7206-7283, evaluate_runtime_commit_fallback:
Generic runtime commit fallback.
Dùng khi tip_mid error nhỏ, z progress stall, xy drift nhỏ, samples đủ.
Vẫn cần runtime primary truth; nếu không, không authorize.

Dòng 7286-7330, build_close_debug_summary:
Rút gọn các metric quan trọng:
- close_allowed;
- hard blockers;
- support gap;
- z stall;
- tip_mid error;
- source reference;
- table clearance.
Dùng để đọc log nhanh.

### Block 16: Close hai stage, short lift, recover/retry

**Dòng liên quan: 7332-7525**

**Mục tiêu block:**
- Thức thì gripper close có hai stage.
- Verify lift ngắn sau close.
- Recover và retry nếu fail.

**Giải thích từng dòng/nhóm dòng:**

Dòng 7332-7401, execute_two_stage_close:
Stage A close theo fraction PHASE2_CLOSE_STAGE_A_FRACTION.
Stage B close đến target gap.
Dùng effort hold trong close/retention.
Ghi gripper values và phase log.

Dòng 7404-7480, verify_short_lift:
Nâng nhẹ sau close.
Đó object bbox/center trước-sau.
Pass nếu object z delta >= PHASE2_SHORT_LIFT_MIN_DELTA_M.
Đây là proof "có nam vật" trước carry dài.

Dòng 7483-7525, recover_and_retry:
Mô gripper, retreat/lift nhẹ, reset một số phase để retry.
Chỉ chạy khi retry count còn trong max.

### Block 17: main orchestration

**Dòng liên quan: 7528-9931**

**Mục tiêu block:**
main() nối tất cả các block trên thành một run Task 1.

**Giải thích từng dòng/nhóm dòng:**

Dòng 7528:
Định nghĩa main() trả int exit code.

Dòng 7529-7900:
Tạo argparse và khai báo CLI.
Nhóm tham số gồm:
- path/runtime;
- seed/target;
- scene/robot;
- workspace/clearance;
- IK;
- Phase 2 thresholds;
- debug marker/log;
- far/vertical policy;
- close/fallback/retry.

Dòng 7901-8050:
Parse args và validate threshold.
Ví dụ table clearance min phải không nhỏ hơn catastrophic min.

Dòng 8051-8150:
Resolve HRC_ROOT, LOG_ROOT, baseline_root, asset_root, config, robot USD/URDF.
Validate environment trước khi load Isaac.

Dòng 8151-8250:
Khởi tạo SimulationApp, stage, timeline, seed.

Dòng 8251-8420:
Load official SceneBuilder/config.
Override root_path đến official assets.
Build table và Task 1 parts.
Dùng diagnostic static bin collider thay cho composed box physics nếu cần.

Dòng 8421-8550:
Load/build robot Walker S2.
Acquire articulation.
Đọc joint/body names.
Apply official startup joint map.

Dòng 8551-8650:
Load official IK classes, tạo IK solver/coordinate transform.
Chọn EE body/path, verify EE alignment, configure compensation.

Dòng 8651-8750:
Build table_frame, lấy object candidate records, chọn target.
Ghi marker target/object/pregrasp nếu enabled.

Dòng 8751-8840:
Tính target_components trong robot base.
Chọn arm side.
Chọn target region near/mid/far.
Resolve TCP offset và point_B offset.

Dòng 8841-8950:
Tạo orientation presets theo region.
Generate coarse candidates và Phase 2 geometric filter.
Select best Phase 2 candidate nếu có.

Dòng 8951-9050:
Resolve vertical XY reference offset nếu vertical policy.
Evaluate pregrasp candidates bằng DualArmIK.
Nếu không pass, run fail trước khi chạy contact/grasp.

Dòng 9051-9200:
Lấy selected geometry, selected orientation, target poses.
Chọn motion policy:
- far_low_side_B_driven;
- mid_vertical_Z_descend;
- near_body_vertical_Z_descend.

Dòng 9201-9400:
Thức thì motion phases.
Far policy:
- execute_pregrasp;
- far outboard transition;
- low-side prepare;
- XY align;
- final_descent_local_ik.
Vertical policy:
- execute_pregrasp;
- align;
- descend;
- final_descent_local_ik.

Dòng 9401-9550:
Gọi _pre_close_gate.
Gọi evaluate_close_gate, evaluate_vertical_support_or_stall_close_fallback, evaluate_runtime_commit_fallback.
Tổng hợp final_close_allowed.
Nếu hard block thì fail trước close.

Dòng 9551-9700:
Nếu close allowed:
- execute_two_stage_close;
- verify_short_lift;
- nếu fail và retry được thì recover_and_retry.

Dòng 9701-9850:
Nếu grasp pass:
- lift/carry đến bin;
- place/release;
- retreat;
- settle object;
- check inside bin/stability.

Dòng 9851-9931:
Build payload kết quả cuối, ghi logs, cleanup SimulationApp, return exit code.
Nếu RunFailure/runtime error thì ghi failure_reason và traceback vào log.

### Block 18: Các điểm cần đọc khi debug "pregrasp unreachable"

Nếu một run báo pregrasp unreachable/pre_grasp_unreachable, cần đọc theo thứ tự:

1. _classify_target_region, dòng 3880-3885.
Nếu target bị xếp far, family orientation và path khác vertical.

2. _approach_family_order_for_region, dòng 3888-3895.
Nếu family ưu tiên world_y_approach mà orientation preset lệch, pregrasp có thể unreachable trước grasp.

3. _plan_grasp_geometry, dòng 4301-4637.
Đây là nối tính pregrasp_b_world:
- Far: pregrasp_b_world = low_side_prepare_b_world.
- Vertical: pregrasp_b_world = contact_b_world + world_up * pregrasp_clearance.

4. _pregrasp_candidates, dòng 4677-4698.
Hiện chỉ có nominal và slightly_higher.
Nếu ca hai fail với cũng error, việc tăng/ha depth grasp không tác động pregrasp nhiều.

5. _evaluate_pregrasp_candidates, dòng 5025-5330.
Đây là nối solve IK, tính pos_err/rot_err, số với tolerance và fail nếu không candidate pass.

6. _candidate_acceptance_tolerances, dòng 4835-4864.
Nếu error nhỏ hơn nhưng tolerance quá chặt, root cause có thể là threshold/logic.
Nếu error lớn và giống nhau qua các grasp-depth offset, root cause nghiêng về pose/orientation/path trước grasp.

### Block 19: Các điểm cần đọc khi debug close/contact-centric

1. _resolve_real_grasp_center_world, dòng 2249-2489.
Đọc source và close_critical_reference.
Nếu source là calibrated distal proxy diagnostic-only, không được coi là proof close.

2. _compute_runtime_two_finger_metrics, dòng 1910-2054.
Đọc primary_runtime_truth, tip_mid_error, tip_axis_alignment_error, symmetry, z_asymmetry.

3. final_descent_local_ik, dòng 6266-6922.
Đọc measured_world_source_for_descent:
- tip_mid: đang dùng runtime truth.
- point_B_fallback: đang fallback proxy.

4. _pre_close_gate, dòng 5860-6176.
Đọc close_critical_uses_real_grasp_center và table clearance.

5. evaluate_close_gate, dòng 6925-7069.
Đọc hard blockers trước khi xem warnings.

6. evaluate_vertical_support_or_stall_close_fallback, dòng 7072-7203.
Đọc support_gap, stalled_in_z, xy drift, orientation.

7. evaluate_runtime_commit_fallback, dòng 7206-7283.
Đọc tip_mid error và z/xy stability.

## 3. Tóm Tắt Thuật Toán Theo Pseudocode

main():
    parse_args()
    validate_paths_and_thresholds()
    start_isaac()
    build_official_task1_scene()
    add_diagnostic_static_bin_if_needed()
    load_or_build_walker_s2()
    acquire_articulation()
    apply_official_startup_joint_map()
    load_dualarmik()
    build_coordinate_transform()
    resolve_table_frame()
    target = select_target_object()
    target_components = project_target_to_robot_base(target.center)
    arm = choose_arm(target_components)
    region = classify_target_region(target_components.forward)
    presets = orientation_presets(region, arm)
    phase1_candidates = generate_approach_candidates(target)
    object_grasp_frame = estimate_object_grasp_frame(target)
    phase2_candidate = select_best_phase2_candidate(phase1_candidates, object_grasp_frame)
    pregrasp_candidate = evaluate_pregrasp_candidates(presets, target, region)
    geometry = pregrasp_candidate.geometry

    if geometry.motion_policy == far_low_side_B_driven:
        servo(pregrasp)
        servo(outboard_transition)
        servo(low_side_prepare)
        servo(xy_align)
        final_descent_local_ik(contact_reference)
    else:
        servo(pregrasp)
        servo(align)
        servo(descend)
        final_descent_local_ik(contact_reference)

    pre_close = pre_close_gate()
    primary = evaluate_close_gate(pre_close)
    vertical_fallback = evaluate_vertical_support_or_stall_close_fallback(pre_close)
    runtime_fallback = evaluate_runtime_commit_fallback(pre_close)
    if not any_allowed(primary, vertical_fallback, runtime_fallback):
        fail_before_close()

    execute_two_stage_close()
    if not verify_short_lift():
        recover_and_retry_if_allowed()
    carry_place_release_settle()
    score_inside_bin_and_stability()
    write_logs()

## 4. Các Công Thức/Quy Tắc Quan Trọng

### 4.1. Contact z

contact_z_world =
    max(
        bbox_top_z + descend_clearance + grasp_depth_offset,
        table_top_z + min_ee_table_clearance
    )

Ý nghĩa:
- bbox_top_z + descend_clearance là mục gần mặt trên của vật.
- grasp_depth_offset am làm target sau hơn.
- min_ee_table_clearance chặn target xuống quá sát table.

### 4.2. point_B world từ EE pose

point_A_world = robot_to_world(pose_base[:3])
rot_world = robot_world_R @ EulerXYZ(pose_base.rpy)
point_B_world = point_A_world + rot_world @ point_b_offset_local

### 4.3. EE pose từ point_B target

point_B_base = world_to_robot(point_B_world)
point_b_offset_base = rot_base @ point_b_offset_local
point_A_base = point_B_base - point_b_offset_base
pose_base = [point_A_base.x, point_A_base.y, point_A_base.z, rpy]

### 4.4. EE pose từ contact reference target

reference_base = world_to_robot(contact_reference_world)
offset_base = rot_base @ contact_control_offset_local
point_A_base = reference_base - offset_base
pose_base = [point_A_base.x, point_A_base.y, point_A_base.z, rpy]

Khác nhau giữa 4.3 và 4.4:
- 4.3 dùng point_B proxy.
- 4.4 dùng contact_control_offset, có thể là offset từ EE đến fingertip midpoint runtime.

### 4.5. Final descent step clamp

delta = locked_target - measured_world
xy_step = clamp_norm(delta.xy, phase2_descent_xy_step)
z_step = clamp(delta.z, -phase2_descent_z_step, 0.0)
commanded_world.xy = measured_world.xy + xy_step
commanded_world.z = measured_world.z + z_step
commanded_world.z không được lớn hơn previous_commanded_world.z

Ý nghĩa:
- XY tiên từ tự về target.
- Z chỉ đi xuống.
- Nếu measured target nhay, command vẫn không bắt len trên.

### 4.6. Close truth priority

Truth priority:
1. actual fingertip pair midpoint
2. stable finger-link midpoint
3. calibrated distal proxy diagnostic-only
4. point_B fallback

Quy tắc:
- Close-critical authorization cần close-critical reference true.
- Diagnostic proxy có thể vẽ marker/log nhưng không tự nó cho phép close.

## 5. Nhận Xét Kỹ Thuật Và Các Cảnh Báo Debug

### 5.1. File này ổn định theo hướng "debuggable trước, tối ưu sau"

Mỗi phase deu ghi details dày:
- source của reference;
- pose base/world;
- point_A/point_B/contact reference;
- IK error;
- threshold;
- pass/fail reason.

Điều này đúng với Phase 3 của dự án: integration/validation trước optimization.

### 5.2. Không nên tune grasp depth khi failure ở approach/pregrasp

Trong logic file:
- grasp_depth_offset tác động contact_z_world.
- Pregrasp far/vertical có thể dùng contact_b_world để suy ra pregrasp, nhưng nếu pregrasp_error gần như không đổi khi thay grasp_depth_offset, thì bottleneck không nằm ở depth.
- Khi failure_phase là approach/pregrasp, cần đọc orientation preset, pregrasp pose construction, IK tolerance, target region và possible staging.

### 5.3. Contact dwell/carry/place/release không liên quan nếu chưa qua pregrasp

Nếu run fail trước grasp_window:
- contact dwell chưa phát huy tác dụng.
- carry stabilization chưa vào phase.
- place depth chưa vào phase.
- release timing chưa vào phase.
Tune các nhóm này lúc đó sẽ tạo nhiều biến nhiều hơn, không giải quyết root cause.

### 5.4. Nếu runtime truth không resolve được, log phải được đọc theo source

Khi đọc log:
- source actual_fingertip_pair_midpoint là mạnh.
- stable finger-link midpoint có thể chấp nhận tùy context.
- calibrated distal proxy là diagnostic.
- fallback point_B là compatibility.

Không được chỉ nhìn "real_grasp_center_world" có số mà kết luận close truth hợp lệ.
Cần xem close_critical_reference và primary_runtime_truth.

## 6. Bản Đồ Hàm Chính Để Trả Cứu Nhanh

Orientation:
- _preset_euler_xyz_to_rot: dòng 153
- _preset_rot_to_euler_xyz: dòng 163
- _build_world_y_approach_presets: dòng 182
- _orientation_presets_by_arm_and_family: dòng 3984
- _region_filtered_orientation_presets: dòng 4000

Runtime/articulation:
- _choose_robot_prim_path: dòng 477
- _acquire_articulation_with_fallback: dòng 527
- _coordinate_transform_from_anchor: dòng 648
- _select_coordinate_transform_with_alignment: dòng 910

Contact reference:
- _resolve_actual_fingertip_pair_midpoint_reference_position: dòng 1451
- _resolve_fingertip_midpoint_reference_position: dòng 1778
- _compute_runtime_two_finger_metrics: dòng 1910
- _pose_for_contact_reference_world: dòng 2057
- _resolve_real_grasp_center_world: dòng 2249

Object/table:
- build_or_resolve_table_frame: dòng 2975
- world_to_table: dòng 3060
- table_to_world: dòng 3072
- get_object_info_in_table_frame: dòng 3102

Candidate/filter:
- generate_approach_candidates_for_object: dòng 3160
- estimate_object_grasp_frame: dòng 3320
- fast_geometric_grasp_filter: dòng 3449
- select_best_phase2_candidate: dòng 3515

Pose planning:
- _pose_for_point_b_world: dòng 4227
- _compute_contact_z_world: dòng 4288
- _plan_grasp_geometry: dòng 4301
- _pregrasp_candidates: dòng 4677

IK/pregrasp:
- _solve_single_arm_pose: dòng 4791
- _classify_candidate_result: dòng 5004
- _evaluate_pregrasp_candidates: dòng 5025

Servo/descent/close:
- _execute_dualarmik_servo_phase: dòng 5333
- _pre_close_gate: dòng 5860
- execute_pregrasp: dòng 6179
- final_descent_local_ik: dòng 6266
- evaluate_close_gate: dòng 6925
- evaluate_vertical_support_or_stall_close_fallback: dòng 7072
- evaluate_runtime_commit_fallback: dòng 7206
- execute_two_stage_close: dòng 7332
- verify_short_lift: dòng 7404

Orchestration:
- main: dòng 7528

## 7. Kết Luận Ngắn

scripts/task1_phase2_contact_centric_patch.py là một pipeline Task 1 hybrid geometric + contact-centric.
Nó vẫn dùng official SceneBuilder và DualArmIK, nhưng bổ sung một lớp truth hierarchy quanh fingertip/contact reference để tránh close dựa trên proxy sai.
Thuật toán chính không nằm ở một hàm duy nhất, mà nằm ở chuỗi:
geometry planning -> IK candidate validation -> servo phase -> final descent contact-centric -> pre-close gate -> close/fallback -> short lift verification.

Nếu cần debug failure:
- Fail trước pregrasp: đọc region, preset, _plan_grasp_geometry, _pregrasp_candidates, _evaluate_pregrasp_candidates.
- Fail trong final descent: đọc measured_world_source_for_descent, samples, z-stall, support gap.
- Fail close: đọc close_critical_reference, primary_runtime_truth, hard blockers.
- Fail sau close: đọc execute_two_stage_close, verify_short_lift, retention/carry/place logs.
