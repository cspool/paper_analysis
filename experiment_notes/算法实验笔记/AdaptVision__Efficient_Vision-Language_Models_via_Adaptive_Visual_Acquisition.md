## AdaptVision__Efficient_Vision-Language_Models_via_Adaptive_Visual_Acquisition

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：AdaptVision 是一个基于强化学习的自适应视觉 token 获取框架，借鉴人类 active vision 的 coarse-to-fine 机制：(1) 首先用 1/4 低分辨率图像（25% visual tokens）处理，模型自主决定是直接回答还是调用 bounding box tool 裁剪高分辨率关键区域；(2) 若 tool call，从原始高分辨率图像中裁剪 bbox 区域 Icrop，合并到序列中继续推理后生成最终回答；(3) 训练使用 Decoupled Turn Policy Optimization (DTPO) —— 将策略损失按 turn 解耦为 Tool Token 和 Answer Token 两部分分别归一化，并分别计算 tool advantage A_tool 和 outcome advantage A_oc 进行精确 credit assignment；(4) 奖励函数包含 Outcome Reward（准确度 R_acc + 格式 R_form + 平衡 R_bal）和 Tool Reward（裁剪正确性 R_crop - α×面积惩罚 R_area）。
  实验比较：(a) 与静态 token 压缩方法（FastV 50%, SparseVLM 50%/70%, VisionZip 50%/70%）和动态方法（VisionThink, VisionThink†）在 9 个 VQA benchmark 上的性能和 token 消耗对比（Table 1, Table 4）；(b) 与 Down-Sample（25% token）对比验证 coarse-to-fine 有效弥补低分辨率的信息损失（+5.8% avg 仅 +7% token）；(c) 推理延迟对比（Fig. 4，vs Vanilla/VisionThink†，1.67× speedup）；(d) 消融实验：reward 设计（balance reward / tool reward 的单独移除对训练稳定性的影响，Fig. 5a）；(e) GRPO vs DTPO 训练动力学对比（Fig. 5b, Fig. 6a）；(f) 自适应 tool-use 分析（各 benchmark 的 tool call ratio，Fig. 6b）；(g) 超参数 sensitivity（λ 和 α，Table 2）；(h) 不同 reward model 对比（GPT-4o vs Qwen3-VL-4B 作为 judge，Table 3）。

- 硬件平台是什么，配置是什么。
  训练：4 节点 × 每节点 8× NVIDIA H20 GPU（共 32× H20 GPU），FP16 mixed-precision 训练。
  推理评估：使用 vLLM 框架，temperature=0。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen2.5-VL-7B-Instruct（视觉编码器 CLIP-ViT + modality projector + 7B LLM decoder）。
  RL 训练框架：veRL（https://github.com/volcengine/verl）。
  训练数据：来自 Yang et al. VisionThink-Smart-Train 数据集（https://huggingface.co/datasets/Senqiao/VisionThink-Smart-Train），包含可用低分辨率直接回答的 VQA 样本和需要高分辨率才能准确回答的样本。
  Benchmark（9 个）：ChartQA (test)、OCRBench (test)、DocVQA (val)、MME (test)、MMVet (test)、RealWorldQA (test)、POPE (test)、MathVista (testmini)、MathVerse (testmini)。
  评价指标：LLM-as-judge (GPT-4o) 进行 binary correctness 判断（1=正确，0=错误），format reward 检查推理/回答/工具调用格式合规性。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/AdaptVision/AdaptVision
  
  算法 pipeline 伪代码：
  ```
  # === AdaptVision 推理流程（两阶段 coarse-to-fine） ===
  # 输入: 高分辨率图像 I_high (H×W), 问题 q
  # 系统提示 x_sys (含 bounding box tool 定义)

  # Phase 1: 低分辨率编码与首轮推理
  I_low = resize(I_high, scale=1/4)           # 1/4 分辨率
  V_low = VisualEncoder(I_low)                 # ViT encode → n_low 个 visual tokens, n_low ≈ 0.25×n_full
  V_low = Projector(V_low)                     # 对齐到 LLM embedding 维度 d
  x = concat([x_sys, V_low, q])               # 拼接输入序列

  # 首轮生成 (policy π_θ 自回归采样)
  o_1:N = autoregressive_sample(LLM, x, temperature=1.0 during training / 0 during inference)

  # 判断响应类型
  if "<tool call>" in o_1:N:
      # Phase 2: Tool call → 裁剪高分辨率区域
      bbox = parse_bbox(o_1:N)                 # 提取 [x1, y1, x2, y2] in 绝对像素坐标
      I_crop = crop(I_high, bbox)              # 从原始高分辨率图像裁剪
      V_crop = VisualEncoder(I_crop)            # ViT encode → n_crop 个 visual tokens
      V_crop = Projector(V_crop)
      # 拼接裁剪区域 tokens 继续推理
      x_ext = concat([x, o_1:T, V_crop])
      o_T+1:N = autoregressive_sample(LLM, x_ext)
      # 总 visual tokens = n_low + n_crop
  else:
      # 直接回答: 总 visual tokens = n_low (仅低分辨率)
      pass

  # 最终答案: o_N 中的 <answer> 标签内容

  # === DTPO 训练流程 ===
  # 每步训练:
  # 1. 从 policy π_θ_old 采样 G=16 个候选响应 {o_i}
  # 2. 计算每个响应的 reward:
  #    R_acc = LLM_judge(pred_answer, ground_truth) ∈ {0,1}
  #    R_form = 0.5 if format合规 else 0
  #    R_bal = -0.1 if (tool_call ∧ correct) or (direct ∧ low_confidence ∧ correct)
  #    R_crop = GPT4o_judge(cropped_region, question) ∈ {0,1}
  #    R_area = clip(r_a/μ_a - 1, 0, 1) if (R_acc=1 ∧ R_crop=1)
  #    R_oc = R_acc + R_form + R_bal
  #    R_tool = R_crop - α·R_area  (α=2)
  #    R = R_oc + R_tool
  # 3. 分别计算 advantage:
  #    A_oc^(i) = (R_oc^(i) - mean({R_oc}))/std({R_oc})
  #    A_tool^(i) = (R_tool^(i) - mean({R_tool}))/std({R_tool})
  # 4. Token-level advantage (Eq.13):
  #    if direct_answer: A_i,t = A_oc^(i) + λ·A_tool^(i)
  #    if tool_call:     A_i,t = A_oc^(i) + λ·A_tool^(i)·I(1≤t≤T_i)
  # 5. Decoupled loss (Eq.12):
  #    L_tool = (1/ΣT_i)· Σ_i Σ_{t=1..T_i} clip_ratio_loss(π_θ, π_θ_old, A_i,t)
  #    L_answer = (1/Σ(N_i-T_i))· Σ_i Σ_{t=T_i+1..N_i} clip_ratio_loss(π_θ, π_θ_old, A_i,t)
  #    L = L_tool + L_answer
  # 6. 用 AdamW (lr=1e-6) 更新 π_θ，80 steps
  ```

  DTPO 相比 GRPO 的核心改进：
  - GRPO: 整个序列用同一个 advantage 归一化 → tool token 被序列长度 N_i 和组数 G 压制，梯度信号不平衡
  - DTPO: tool token 和 answer token 分别在各自组内按 token 数归一化 + 分别计算独立的 advantage → 精确 credit assignment + 平衡优化
