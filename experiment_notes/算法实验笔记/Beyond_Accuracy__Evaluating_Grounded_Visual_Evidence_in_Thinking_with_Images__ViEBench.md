## Beyond_Accuracy__Evaluating_Grounded_Visual_Evidence_in_Thinking_with_Images__ViEBench

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：ViEBench 是一个 process-verifiable 的 VLM 评估 benchmark，核心创新包含两个组件：(1) Benchmark 数据集 —— 200 张高分辨率多场景图像（retail, urban, industry, daily life），每个样本有 expert-annotated 黄金 BBox 标注视觉证据的精确空间位置，关键视觉证据平均仅占图像面积的 < 0.7%，迫使模型必须执行精确的 zooming/cropping 操作（"Thinking-with-Images"）；任务分为 perception（50%）和 reasoning（50%）两类，其中 reasoning 任务要求模型整合局部视觉线索与先验知识进行多步逻辑推理；(2) Dual-Axis Capability Matrix —— 基于 Intersection-over-Area (IoA) 指标构建二维诊断矩阵，将 Grounding 轴（IoA > 0.5 为 G⁺，否则 G⁻）与 Answer 轴（答案对/错为 A⁺/A⁻）交叉形成四个诊断象限：Valid Grounded Reasoning (G⁺·A⁺)、Ground-Success Answer-Failure (G⁺·A⁻)、Ungrounded Correct Answer (G⁻·A⁺)、Dual Ground-Answer Failure (G⁻·A⁻)，实现过程级细粒度诊断。
  实验比较：(a) Agentic Models (7个) vs End-to-end VLMs (9个) 在 ViEBench perception 和 reasoning 任务上的 accuracy 对比；(b) Agentic Models 的七项细粒度指标（Acc, GS, G⁺·A⁺, G⁺·A⁻, G⁻·A⁺, G⁻·A⁻, TR）在 perception 和 reasoning 子集上的全面过程级审计；(c) 双向 IoA 分析（IoA(B_pred, B_gt) vs IoA(B_gt, B_pred)）揭示不同模型的 crop 策略（expansive coverage vs tight focus）；(d) 对比 ViEBench 与现有 benchmark（V* Bench, HRBench, InfoVQA, VisualProbe）的功能覆盖差异。

- 硬件平台是什么，配置是什么。
  论文未明确说明评估所用 GPU 型号和硬件配置。Agentic models 严格按其官方仓库的评估设置和环境配置执行；End-to-end VLMs 使用 VLMEvalKit 框架进行统一评估。

- 模型是什么。数据集和bench分别是什么。
  模型（Agentic Models）：Pixel Reasoner, Thyme, DeepEyes, Mini-o3, Qwen3-VL-8B-Instruct, Qwen3-VL-235B-A22B-Instruct, Qwen3-VL-32B-Instruct。这些模型具备 tool-use 能力（自主 zooming/cropping），评估时遵循各模型官方仓库配置。
  模型（End-to-end VLMs）：GPT-4o, o3, Qwen2.5-VL-7B-Instruct, InternVL3-8B, LLaVA-CoT, LLaVA-OneVision (standard + SI variant), Keye-VL-1.5-8B, MiMo-VL-7B-RL。这些模型不具备显式 cropping 机制，仅报告 perception 和 reasoning accuracy。
  数据集（ViEBench）：200 个高分辨率多选 QA pairs，来自 Web search + VisualProbe。场景分布：urban (32%), daily life (32%), industrial (19%), retail (17%)。任务分布：perception (50%), reasoning (50%)。关键证据空间稀疏度：perception 任务 gold BBox 平均占图像面积 0.32%，reasoning 任务 0.63%。
  评价指标：Accuracy (Acc.)、Grounded Score (GS)、Valid Grounded Reasoning (G⁺·A⁺)、Ground-Success Answer-Failure (G⁺·A⁻)、Ungrounded Correct Answer (G⁻·A⁺)、Dual Ground-Answer Failure (G⁻·A⁻)、Tool Ratio (TR)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/Xuchen-Li/ViEBench（代码将在论文发表后发布）
  标注平台：基于 Label Studio (https://github.com/HumanSignal/label-studio) 定制的 web-based 标注界面

  ViEBench 评估协议伪代码：
  ```
  # === ViEBench 评估流程 ===
  # 输入: 模型 M, ViEBench benchmark 样本集 D = {(I_i, Q_i, A_gt_i, B_gt_i, task_type_i)}

  for each sample (I, Q, A_gt, B_gt, task_type) in D:
      # Step 1: 模型推理
      if M is agentic (with tools):
          # 模型自主决策是否调用 zooming/cropping tool
          response, crop_history = M(I, Q)  # crop_history 包含 B_pred 列表
      else:
          # End-to-end 模型直接推理（无显式 crop 输出）
          response = M(I, Q)
          crop_history = None

      # Step 2: Answer 评估
      A_pred = extract_answer(response)
      answer_correct = (A_pred == A_gt)

      # Step 3: Grounding 评估（仅对 agentic models）
      if crop_history is not None:
          B_pred = crop_history  # 模型生成的 crop 区域
          # 双向 IoA 计算
          IoA_pred_gt = Area(B_pred ∩ B_gt) / Area(B_gt)  # coverage: crop 覆盖了多少证据
          IoA_gt_pred = Area(B_pred ∩ B_gt) / Area(B_pred)  # concentration: crop 中证据占比
          IoA = max(IoA_pred_gt, IoA_gt_pred)
          grounding_success = (IoA > 0.5)

      # Step 4: 象限分配
      if grounding_success and answer_correct:
          quadrant = "G⁺·A⁺ (Valid Grounded Reasoning)"
      elif grounding_success and not answer_correct:
          quadrant = "G⁺·A⁻ (Ground-Success Answer-Failure)"
      elif not grounding_success and answer_correct:
          quadrant = "G⁻·A⁺ (Ungrounded Correct Answer)"
      else:
          quadrant = "G⁻·A⁻ (Dual Ground-Answer Failure)"

  # === 汇总指标 ===
  Acc = count(answer_correct) / |D|
  GS = count(grounding_success) / |D|  # 仅 agentic models
  G⁺A⁺ = count(G⁺·A⁺) / |D|
  G⁺A⁻ = count(G⁺·A⁻) / |D|
  G⁻A⁺ = count(G⁻·A⁺) / |D|
  G⁻A⁻ = count(G⁻·A⁻) / |D|
  TR = count(tool_invoked) / |D|  # Tool Ratio
  ```

  IoA 计算公式：
  ```
  IoA(B_pred, B_gt) = (Area(B_pred) ∩ Area(B_gt)) / Area(B_gt)
  IoA(B_gt, B_pred) = (Area(B_pred) ∩ Area(B_gt)) / Area(B_pred)
  IoA_final = max(IoA(B_pred, B_gt), IoA(B_gt, B_pred))
  # IoA > 0.5 → G⁺ (successful grounding)
  # IoA ≤ 0.5 → G⁻ (failed grounding)
  ```

  关键设计要点：
  - ViEBench 的 IoA 使用 max 而非标准 IoU，同时容忍 precise tight crop（高 IoA(B_gt, B_pred)）和 conservative expansive crop（高 IoA(B_pred, B_gt)）
  - Perception 任务要求模型识别细粒度视觉属性，reasoning 任务要求多步逻辑推理整合视觉线索与先验知识
  - Gold BBox 平均仅占 0.32%-0.63% 图像面积，确保在全局视图下 sub-perceptual，强制 tool-use
  - 标注流程：专业标注员标注 → 资深审查员验证 → 模糊样本精炼或丢弃
