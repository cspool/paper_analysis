## VTPerception-R1: Enhancing Multimodal Reasoning via Explicit Visual and Textual Perceptual Grounding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**VTPerception-R1**，一个两阶段训练框架，显式地将感知（perception）与推理（reasoning）解耦，增强多模态大语言模型的感知接地（perceptual grounding）能力。核心包括：
  
  **Stage I — Perception-Augmented SFT**：将原始 CoT 数据转换为结构化目标格式 `<description>...<think>...<answer>`，其中 `<description>` 专门总结与问题相关的视觉/文本证据（非通用 caption），`<think>` 保留原始推理链，`<answer>` 给出最终解。训练目标为 token 级交叉熵损失 L_SFT = -Σ_t log π_θ(y_t | x, y_<t)。SFT 数据从 LLaVA-CoT (4K) 和 Vision-SR1 (8K) 采样共约 12K 样本，经过自动清洗流水线（VLM dense caption + Grounding DINO + EasyOCR → 结构化描述 → LLM 重建 CoT → 多维度质量评分过滤）。

  **Stage II — Perception-Aware RL**：基于 DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）目标，引入六个奖励项：
  - R_acc：答案正确性奖励
  - R_fmt：格式合规奖励（强制 `<description> → <think> → <answer>` 模板）
  - R_rep：重复惩罚（惩罚重复 n-gram，鼓励简洁描述）
  - R_vkey：视觉关键信息奖励，衡量 `<description>` 覆盖关键视觉元素的比例，离散化为三档（1.0 / 0.5 / 0.0）
  - R_tkey：文本关键信息奖励，衡量 `<think>` 覆盖关键文本元素（OCR 文本、数值、单位、约束、常识）的比例
  - R_cons：描述-推理一致性奖励，检查 `<think> + <answer>` 中的实体/属性/数值是否被 `<description> + question` 的证据支持，存在冲突时直接为 0
  - 采用 perception-first 加权调度：训练早期侧重感知接地，后期切换到正确性

  实验比较：对比以下基于 Qwen2.5-VL-7B 的 baseline 方法——
  - Vision-SR1-7B（自奖励视觉推理）
  - Vision-R1-7B（cold-start + RLVR）
  - Perception-R1-7B（视觉感知奖励）
  - Visionary-R1（caption→reason→answer 结构）
  - MM-Eureka-Qwen-7B（GRPO + 规则奖励）
  - VL-Rethinker-7B（自反思强化学习）
  - Qwen2.5-VL-7B-Instruct（原始指令微调基座）
  - VTPerception-R1-7B (Before RL)（SFT 后 RL 前的自身消融）
  在 MathVista、MMMU、EMMA、AI2D、Creation-MMBench、C-MMBench-TO 六个 benchmark 上评估。

- 硬件平台是什么，配置是什么。
  SFT 阶段：DeepSpeed ZeRO-3 + bf16 精度 + 梯度检查点。RL 阶段：基于 EasyR1-perc 框架，使用 Ray 分布在 1 个主节点 + 1 个 ORM 节点上，Tensor Parallel Size = 4。论文未明确说明 GPU 型号和数量。

- 模型是什么。数据集和bench分别是什么。
  模型：**Qwen2.5-VL-7B-Instruct** 作为基座模型，全参数微调。
  
  训练数据：
  - SFT 数据：从 LLaVA-CoT（4K）和 Vision-SR1（8K）采样合并 ~12K 样本，经自动化清洗流水线处理
  - RL 数据：从 MMK12（5K）、LLaVA-CoT（5K）、Vision-R1-rl（5K）、Mulberry（5K）聚合 ~22K 多模态推理样本，覆盖数学、科学、图表理解；通过教师模型集成 + 预算验证 + 关键信息提取流水线构建
  
  训练超参数：SFT 阶段 lr=1e-5, weight decay=0.1, batch size=1, gradient accumulation=8, 3 epochs。RL 阶段训练 2 epochs。
  
  Benchmarks（6 个）：MathVista（6141 题，数学推理）、MMMU（11.5K 题，大学多学科）、AI2D（科学图表理解）、EMMA（数理化学科交叉推理）、Creation-MMBench（765 实例，51 个细粒度任务）、C-MMBench-TO（纯文本变体）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/yizhuoDi/VTPerceprion-R1
  
  算法 Pipeline 伪代码：
  ```
  # ===== Stage I: Perception-Augmented SFT =====
  # 输入: 图像 x_img, 问题 q
  # 目标: 模型生成 <description> d, <think> t, <answer> a
  
  d = model.generate(x_img, q, prefix="<description>")
  # d 仅包含与问题相关的视觉/文本证据，非通用 caption
  
  t = model.generate(x_img, q, d, prefix="<think>")
  # t 为基于 d 的推理链
  
  a = model.generate(x_img, q, d, t, prefix="<answer>")
  # a 为最终答案
  
  L_SFT = CrossEntropy(concat(d, t, a), target)
  # 目标序列: <description>...<think>...<answer>
  
  # ===== Stage II: Perception-Aware RL (DAPO) =====
  # 对每个 prompt x，采样 G 个响应 {o_i}
  # o_i 格式: <description> d_i <think> t_i <answer> a_i
  
  # 计算各奖励分量:
  R_acc = exact_match(a_i, ground_truth)         # 答案正确性
  R_fmt = check_format(o_i)                        # 模板合规
  R_rep = -count_repeated_ngrams(o_i)             # 重复惩罚
  
  # 视觉关键信息奖励:
  D = extract_facts(d_i)                          # 从 description 提取事实
  cov_v = |K_v ∩ D| / |K_v|                        # K_v 为标注的关键视觉线索
  R_vkey = 1.0 if cov_v >= τ_hi else (0.5 if cov_v >= τ_lo else 0.0)
  
  # 文本关键信息奖励:
  D_t = extract_facts(t_i)                        # 从 think 提取事实
  cov_t = |K_t ∩ D_t| / |K_t|                      # K_t 为标注的关键文本线索
  R_tkey = 1.0 if cov_t >= τ_hi else (0.5 if cov_t >= τ_lo else 0.0)
  
  # 一致性奖励:
  F_ans = extract_entities(t_i) ∪ extract_entities(a_i)
  E = extract_entities(d_i) ∪ extract_entities(q)
  cons = |F_ans ∩ E| / max(1, |F_ans|)
  R_cons = 0 if has_conflict(F_ans, E) else cons
  
  # 总奖励:
  R_i = R_acc + R_fmt + R_vkey + R_tkey + R_rep + R_cons
  
  # DAPO 目标 (token-level clipped policy gradient):
  A_i_t = normalize(R_i - mean({R_j}))  # group-relative advantage
  r_i_t(θ) = π_θ(o_i_t | ...) / π_old(o_i_t | ...)
  J(θ) = E[ 1/(Σ|o_i|) Σ_i Σ_t min(r_i_t * A_i_t, clip(r_i_t, 1-ε_low, 1+ε_high) * A_i_t) ]
  ```
  
  关键设计：
  - DAPO 四项技术：非对称裁剪（ε_low ≠ ε_high）、动态采样、token 级优化、过长响应惩罚
  - 排除所有 rollout 全对或全错的无效 group，避免退化更新
  - Perception-first 加权调度：早期训练增大 R_vkey 和 R_tkey 权重，后期切换到 R_acc

  - 开源情况：代码已开源在 GitHub，基于 EasyR1-perc 框架（DAPO 实现）和 DeepSpeed ZeRO-3。
