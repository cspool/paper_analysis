## Multi-Object Tracking (MOT) in Videos

术语是什么？
Multi-Object Tracking（MOT，多目标跟踪）是计算机视觉中的核心任务，指在视频序列中同时检测、识别并持续跟踪多个目标物体（如行人、车辆、手部等）的运动轨迹。与Single Object Tracking（SOT）不同，MOT需要处理目标间的遮挡、身份切换（ID switch）、目标进出画面等复杂情况。FoundationMotion中的MOT pipeline使用**hierarchical multi-stage detection + two-stage tracking**策略：预先通过多个专用检测器（Grounded-DINO for open-vocab objects、Cascade Mask R-CNN for persons、Hands23 for hands）获得初始检测，然后使用SAM2进行cross-frame propagation和keyframe refinement。

从算法pipeline角度拆解术语：
FoundationMotion的MOT pipeline：
```
# Hierarchical Detection
objects  = GroundedDINO(frame_0, categories_from_QwenVL)  # open-vocab
persons  = CascadeMaskRCNN_ViTDetH(frame_0, tau=0.8)      # person (high conf)
for each detected person:
    keypoints = ViTPose+(person_region)                    # 42 hand kpts
    hands     = Hands23(expand_region(hand_kpts, 1.5x))   # left/right hand
    # Hands23 output: (bbox, side{L/R}, contact_state, object_bbox)

# ID Assignment (hierarchical encoding)
person_id in [0, 99]
left_hand_id  = person_id * 10 + 1
right_hand_id = person_id * 10 + 4
object_id >= 1000  # non-person objects

# Two-Stage Tracking with SAM2
M_0 = SAM2.init(frame_0, prompts=all_detections)  # Stage 1: init
for t in 1..T:
    M_t = SAM2.propagate(M_{t-1})                  # Stage 1: per-frame propagation
    if t % 5 == 0:                                 # Stage 2: keyframe refine
        B_new = Hands23.detect(frame_t)            # re-detect hands
        M_t = SAM2.propagate(M_{t-1}, B_new)       # inject new prompts

# Trajectory Output
for each tracked object:
    trajectory[obj] = {
        bbox: [[l/width, t/height, r/width, b/height]_t for t in 0..T],
        object_type: str,
        interactions: [neighbor_obj_ids_at_t for t in 0..T]
    }
```

实际输出用于GPT-4o-mini生成motion captions的JSON格式：每个object有跨所有帧的bbox序列（归一化坐标）、object_type和interactions（记录每帧与其他object的空间关系）。

术语一般如何实现？如何使用？
Video MOT通常通过tracking-by-detection范式实现：先用detector获取每帧检测，再用association算法（如Kalman filter + Hungarian matching in SORT/DeepSORT）或learned tracker（如SAM2的memory-based propagation）跨帧关联。FoundationMotion选择SAM2-based方法因为其memory module对遮挡和外观变化更鲁棒。两阶段设计（全帧propagation + keyframe re-detection）平衡了效率（全帧SAM2 propagation计算量大）和精度（纯propagation会drift）。

涉及论文标题：
- FoundationMotion: Auto-Labeling and Reasoning about Spatial Movement in Videos
