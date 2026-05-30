## TSPO__Temporal_Sampling_Policy_Optimization_for_Long-form_Video_Language_Understanding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**TSPO (Temporal Sampling Policy Optimization)**，一个基于强化学习的可训练稀疏帧采样框架，用于长视频语言理解。核心包括三部分：(1) **Event-aware Temporal Agent**：基于 CLIP-Large (400M参数，冻结) + 3.5M 可学习参数，通过 local window attention 捕获 event-query 关联，使用 Gumbel-Softmax 进行概率化关键帧选择；(2) **TSPO RL 优化算法**：将关键帧选择和语言生成建模为联合决策过程，通过 GRPO (Group Relative Policy Optimization) 进行端到端策略优化，Video-MLLM 骨干保持冻结；(3) **双风格训练数据构建管道**：Comprehensive Temporal Data（从 LLaVA-Video-178K 过滤出的多关键帧依赖 QA）+ Video Needle-in-a-Haystack Data（拼接目标视频和无关视频段合成的 10∼60 分钟长视频）。奖励机制包含 Answering Accuracy Reward（R_A = 多选题答案是否正确）和 Temporal Localization Reward（R_T = 采样帧中目标视频帧占比）。
  实验比较：与 uniform sampling、training-free keyframe search（LongVU/DINOv2、CoS/MLLM-13B）、training-based 方法（FrameVOYAGER、MLLM-VFS）、以及 SFT-based keyframe 学习在 LongVideoBench / MLVU / Video-MME / LVBench 四个长视频 benchmark 上的准确率。同时进行跨 Video-MLLM 迁移实验（LLaVA-Video→Qwen2VL/Qwen2.5VL/LLaVA-Video-72B）、推理效率对比（frame extraction time、token 数、LLM time）。

- 硬件平台是什么，配置是什么。
  训练：8 × NVIDIA A800 80GB GPU。推理效率实验：单 GPU（论文未明确说明推理 GPU 型号，但 toy_example 要求 ≥28GB 显存）。

- 模型是什么。数据集和bench分别是什么。
  模型：Temporal Agent 基于 CLIP-Large (400M 参数，冻结) + 3.5M 可训练参数（local window attention + MLP projector）。Video-MLLM 骨干：LLaVA-Video-7B（训练时冻结），迁移实验使用 Qwen2VL-7B、Qwen2.5VL-7B、LLaVA-Video-72B。
  训练数据集：TSPO-10K（自建，包含 Comprehensive Temporal Data 和 Video Needle-in-a-Haystack Data，共 10,000 条样本，源自 LLaVA-Video-178K）。
  Benchmarks：LongVideoBench（验证集 1,337 视频，平均 12min）、MLVU（Dev 集 M-Avg 部分，3min∼2h）、Video-MME w/o sub（900 视频，短/中/长三档，2700 QA）、LVBench（平均 4101 秒，超长视频 benchmark）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/Hui-design/TSPO（Public，AAAI 2026）。包含训练脚本（train_deepspeed.sh）、demo 脚本（llava_video_tspo.py / qwen25vl_tspo.py）、评估脚本（evaluation/）、模型实现（src/open_tspo/）、特征提取工具（mp_tools/）。

  算法 pipeline（张量计算级别）：

  ```
  输入: 长视频 V (T 帧), 文本查询 q, Video-MLLM π_l (冻结)

  # Step 1: 候选帧采样
  V_c = UniformSample(V, fps=1)  # T_c 帧候选

  # Step 2: CLIP 特征提取 (CLIP-Large, 冻结)
  F_f = CLIP_visual(V_c)         # [T_c, D], D 为特征维度
  F_t = CLIP_text(q)             # [1, D]

  # Step 3: Event-aware 特征增强 (3.5M 可训练参数)
  # Local window attention with sinusoidal positional encoding
  F_e = LocalWindowAttention(F_f + SinusoidalPE, window_size=w)  # [T_c, D]
  F_e = MLP_project(F_e)                                          # [T_c, D]

  # Step 4: 跨模态相似度计算
  Sim_event = CosineSimilarity(F_e, F_t)   # [T_c]
  Sim_frame = CosineSimilarity(F_f, F_t)   # [T_c]
  S = Sim_event + Sim_frame                # [T_c], 融合分数

  # Step 5: Gumbel-Softmax 概率采样
  γ ~ Gumbel(0, 1)                         # [T_c], 探索噪声
  P = Softmax(S/τ + γ)                     # [T_c], 概率分布
  P_sorted, I = TopK(P, T_s)               # 选 T_s 个关键帧索引+概率

  # Step 6: Video-MLLM 推理 (冻结)
  V_s = V_c[I]                              # 关键帧
  o = π_l(q, V_s, V_c)                     # 自回归生成回答

  # Step 7: 奖励计算
  R_A = 1 if (predicted_option == ground_truth) else 0   # 准确性奖励
  R_T = count(I in target_video_range) / T_s             # 定位奖励
  R = R_A + R_T  (Needle-in-a-Haystack) 或 R = R_A + 1  (Comprehensive)

  # Step 8: GRPO 策略更新 (仅更新 Temporal Agent 参数 θ_ts)
  # 对每组 G 个采样得到优势 A_i = (R_i - mean(R)) / std(R)
  # 目标: max E[π_ts(V_s|q,V_c) / π_ts_old(V_s|q,V_c) * A_i]
  # Video-MLLM 比率恒为 1 (冻结)，仅优化 θ_ts
  θ_ts ← θ_ts + lr * ∇J*_tspo(θ_ts)
  ```

  **关键设计**：
  - Temperature annealing: τ 从 0.025 逐渐退火至 0.01，训练初期鼓励探索，后期收敛到确定性关键段。
  - 推理时去除 Gumbel 噪声，直接 Softmax + TopK 确定性采样。
  - 训练时 T_s=16 帧，推理时 T_s=64 帧（候选帧 T_c=1FPS）。
  - GRPO 的 group size G 由 DeepSpeed 分布式训练管理。
  - Window size w=12, batch size=1, learning rate=5×10⁻⁴, 单 epoch 训练。
