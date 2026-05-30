## FoundationMotion: Auto-Labeling and Reasoning about Spatial Movement in Videos

- 属于算法pipeline的实现是什么？实验比较什么？
  实现一个全自动数据标注pipeline（FoundationMotion Pipeline），包含四个阶段：(1) Video Preprocessing——对视频做5-10秒temporal cropping，使用VGGT检测并过滤显著相机运动的视频；(2) Object Detection and Tracking——使用Qwen2.5-VL-7B做开放词汇目标识别+Grounded-DINO做定位，使用Cascade Mask R-CNN (ViTDet-H)+ViTPose++Hands23做人体/手部检测，使用SAM2做跨帧时序tracking（两阶段：初始tracking每帧传播+每5帧keyframe refinement）；(3) Caption Generation——将tracking输出（归一化bbox轨迹JSON、视频帧、彩色bbox overlay）输入GPT-4o-mini，按7维度prompt生成motion caption；(4) QA Generation——基于caption和视频帧，使用GPT-4o-mini生成5类QA（Motion Recognition, Action Order/Temporal Ordering, Motion-related Objects, Location-based Motion, Repetition Count）。在InternVid的46.7k视频上运行该pipeline，生成467K caption/QA-video pairs作为FoundationMotion Dataset。然后fine-tune NVILA-Video-15B/8B和Qwen2.5-VL-7B在MotionBench、VLM4D和自建zero-shot benchmarks上对比评估，baseline包括Gemini-2.5-Flash、Qwen2.5-VL-72B、PerceptionLM同量数据finetune。

- 硬件平台是什么，配置是什么。
  8×A100 GPUs（training和testing均使用）。论文未明确说明A100的具体显存配置（40GB或80GB）。

- 模型是什么。数据集和bench分别是什么。
  - 被fine-tune的base模型：NVILA-Video-15B、NVILA-Video-8B、Qwen2.5-VL-7B
  - 数据标注pipeline中使用的模型：Qwen2.5-VL-7B（开放词汇检测）、Grounded-DINO（目标定位）、Cascade Mask R-CNN with ViTDet-H backbone（人体检测）、ViTPose+（关键点提取）、Hands23（手部检测+交互分析）、SAM2（视频tracking）、VGGT（相机运动检测）、GPT-4o-mini（caption和QA生成）
  - 训练数据：InternVid（随机采样5秒clips，通过FoundationMotion pipeline标注）
  - 评估benchmarks：
    - 公开benchmark：MotionBench（5385 videos, 8052 QA pairs, 6 motion tasks）、VLM4D（1000 videos, 1800 QA pairs）
    - 自建zero-shot benchmark：AV-Car（NuScenes, 1968 QAs）、AV-Hand（NuScenes, 108 QAs）、Daily（100 Days of Hands, 832 QAs）、Robotics（YouTube, 102 QAs）
  - Baseline对比模型：Gemini-2.5-Flash、Qwen2.5-VL-72B、NVILA-Video-15B/8B base、Qwen2.5-VL-7B base、PLM（PerceptionLM）同量数据finetune

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  全面开源：Code: https://github.com/Wolfv0/FoundationMotion；Dataset: https://huggingface.co/datasets/WoWolf/v2-dev；Model: https://huggingface.co/WoWolf/models。
  
  **Algorithm Pipeline 伪代码**：
  ```
  # Stage 1: Video Preprocessing
  t_s ~ U(5, min(10, t_v))
  t_start = max(0, min(t_v - t_s, t_v/2 - t_s/2 + eps)), eps ~ U(-0.2*t_v, 0.2*t_v)
  clip = video[t_start : t_start + t_s]
  if VGGT.camera_motion_score(clip) > 0.3: skip  # filter high camera motion
  
  # Stage 2: Object Detection & Tracking
  O = Qwen2.5-VL-7B.detect(clip[0])           # open-vocab object categories
  B_obj = GroundedDINO(clip[0], O)              # bounding boxes per category
  B_person = CascadeMaskRCNN_ViTDetH(clip[0])   # person detections (tau=0.8)
  K = ViTPose+(clip[0], B_person)               # keypoints incl. 42 hand kpts
  for each person p:
      hands[p] = Hands23.detect(expand_region(K.hand_kpts[p], 1.5x))
  M_0 = SAM2.initialize(clip[0], B_obj + B_person + hands)  # init masks
  for t in 1..T:
      M_t = SAM2.propagate(M_{t-1})
      if t % 5 == 0:  # keyframe refinement
          B_new = Hands23.detect(clip[t])
          M_t = SAM2.propagate(M_{t-1}, B_new)
  
  # Stage 3: Caption Generation
  json_trajectories = {obj_id: {bbox: [[l,t,r,b]_t for t], object_type, interactions}}
  frames_2fps = sample(clip, fps=2)
  overlay = draw_bbox_overlay(frames_2fps, json_trajectories, color_coded=True)
  caption = GPT4o_mini(frames_2fps, json_trajectories, overlay, prompt_7dim)
  
  # Stage 4: QA Generation (5 types)
  qa_pairs = GPT4o_mini(frames, caption, prompt_5categories)
  # Categories: MotionRecognition, ActionOrder, MotionRelatedObjects,
  #             LocationBasedMotion, RepetitionCount
  # Output: [{"Q":..., "A":..., "B":..., "C":..., "D":..., "type":...}]
  
  # Stage 5: Fine-tuning
  for model in [NVILA-Video-15B, NVILA-Video-8B, Qwen2.5-VL-7B]:
      if model is Qwen-based:
          trainer = llamafactory(model, lr=1e-5, optimizer=Adam)
      else:  # NVILA
          trainer = NVILA_official(model, lr=1.5e-5, optimizer=Adam)
      trainer.fit(FoundationMotion_Dataset, cosine_annealing, no_weight_decay)
  ```

  **张量计算示意（Tracking核心）**：
  - BBox轨迹存储：traj in R^{N_obj x T x 4}，其中4维为 [left/width, top/height, right/width, bottom/height] 归一化坐标
  - SAM2 propagation：M_t = SAM2.predictor.propagate_in_video(M_{t-1})，M_t为frame t的segmentation masks {m_i in {0,1}^{HxW}}
  - ID层级编码：persons ID in [0,99]，left_hand=ID*10+1，right_hand=ID*10+4，objects ID >= 1000
