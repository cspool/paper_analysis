## VTPerception-R1: Enhancing Multimodal Reasoning via Explicit Visual and Textual Perceptual Grounding

- baseline方法是什么？
  Baseline 方法为 **RLVR-only 的多模态推理方法**（如 GRPO/DAPO 目标 + 格式奖励 + 答案正确性奖励），代表方法包括 MM-Eureka、Vision-R1、R1-VL、Visionary-R1 等。这类方法的核心思路是将文本大语言模型的 RLVR 直接迁移到多模态场景：通过强化学习优化答案正确性和结构化推理格式，让模型在 GRPO/DAPO 目标下自动探索更好的推理路径。
  
  全栈执行例子：
  - **算法 Pipeline**：输入（图像 x_img + 问题 q）→ MLLM 编码 → 自回归生成推理链（可能包含隐式视觉参考）→ 最终答案 a。RL 只奖励答案匹配度 R_acc 和格式合规 R_fmt，感知过程完全隐式（模型自行决定关注图像的哪些部分）。
  - **系统框架层**：基于 EasyR1 或类似 DAPO/GRPO 实现框架，Ray 分布式 RL 训练，DeepSpeed ZeRO-3 部署。训练流程：前向生成 → 规则奖励计算 → 策略梯度更新。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：标准 Transformer attention（Causal Self-Attention + Cross-Attention for visual tokens），Qwen2.5-VL 原生实现，无特殊 kernel 优化。
  - **硬件架构层**：论文未明确说明 GPU 型号，使用 bf16 精度训练。

  Baseline 的核心缺陷（论文系统研究发现）：
  (1) **感知错误是 RLVR 失败的主因**：PAPO 人工审计发现 67% 的 GRPO 错误源于感知问题。
  (2) **正确性奖励不足以改善感知**：纯粹的 answer-correctness RLVR 无法有效提升模型的视觉/文本感知能力。
  (3) **小模型感知能力更弱，结构化 prompting 反而有害**：实验发现 7B 模型在 structured visual grounding prompting 下性能下降，因为其自身感知能力不足以支撑结构化描述，产生幻觉性观察。
  (4) **缺乏文本感知**：现有方法几乎仅关注视觉感知，忽略了文本线索（OCR、数值、约束条件）对推理的关键影响。
  (5) **推理与感知耦合导致不可审计**：隐式感知使得无法检查模型是否"看到了正确的证据"。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **VTPerception-R1**，一个显式解耦感知与推理的两阶段训练框架，系统性地解决上述缺陷。

  **对应解决关系**：
  
  (a) **针对"感知错误是主因"→ 引入感知增强 SFT（Stage I）**：通过 `<description>` 字段显式训练模型提取视觉/文本证据，将感知从隐式过程变为显式、可检查的输出。SFT 数据经过自动化清洗（VLM dense caption + Grounding DINO + EasyOCR → 重建 CoT → 质量评分），确保感知训练数据的质量。

  (b) **针对"正确性奖励不足以改善感知"→ 引入感知感知 RL（Stage II）**：在 DAPO 目标上增加三个感知专用奖励——视觉关键信息 R_vkey（衡量 description 覆盖标注关键视觉元素的比例）、文本关键信息 R_tkey（衡量 think 覆盖关键文本线索的比例）、一致性 R_cons（确保推理引用的实体/属性/数值被感知证据支持）。这些奖励直接提供感知级别的学习信号，而非仅依赖下游答案正确性的间接信号。

  (c) **针对"小模型感知弱"→ Perception-First 加权调度**：训练早期增大 R_vkey 和 R_tkey 权重，先建立稳健的感知基础，后期才切换到以 R_acc 为主。这种渐进式策略尤其适合 7B 等感知能力较弱的模型。

  (d) **针对"缺乏文本感知"→ R_tkey 奖励**：专门衡量模型是否在推理中使用了问题中的关键文本信息（OCR 文本、数值、单位、约束、常识），确保推理不是纯视觉驱动的。

  (e) **针对"推理与感知耦合不可审计"→ R_cons 奖励**：检查 `<think> + <answer>` 中引用的实体/属性/数值是否在 `<description> + question` 中有据可查，存在冲突时直接给 0 分奖励。这使得模型的推理链可审计——任何人可以检查 reasoning 是否 grounded in perception。

  **全栈执行例子（VTPerception-R1）**：
  - **算法 Pipeline**：
    1. 输入（图像 x_img + 问题 q）
    2. MLLM 编码（Qwen2.5-VL-7B-Instruct 全参数）
    3. Stage I SFT 训练：模型学习生成 `<description>`（提取视觉/文本证据）→ `<think>`（基于证据推理）→ `<answer>`（输出答案），损失 L_SFT = -Σ log π_θ(y_t|x, y_<t)
    4. Stage II RL 训练：对 prompt x 采样 G 个 response {o_i}，计算 R = R_acc + R_fmt + R_vkey + R_tkey + R_rep + R_cons，用 DAPO token-level clipped policy gradient 更新
    5. 推理时：模型生成 `<description> d <think> t <answer> a`，d 可被外部检查验证感知是否正确
  - **系统框架层**：基于 EasyR1-perc（DAPO 实现），Ray 分布式（1 主节点 + 1 ORM 节点），DeepSpeed ZeRO-3 + bf16，TP=4。RL 数据通过教师模型集成（72B 级模型）→ 预算验证（top-B by log-probability → correctness + coherence scoring）→ 关键信息提取流水线构建。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：标准 Qwen2.5-VL attention 实现，无特殊 kernel 优化。SFT 阶段有梯度检查点（gradient checkpointing）优化显存。
  - **硬件架构层**：论文未明确说明 GPU 型号，仅说明使用 DeepSpeed ZeRO-3 分布式训练。

  关键对比：Baseline 方法（如 Vision-R1）仅奖励 "最终答案对不对"（R_acc + R_fmt），VTPerception-R1 还同时奖励 "感知到了什么"（R_vkey）"有没有用文本线索"（R_tkey）和 "推理是否忠于感知"（R_cons），从而让模型在强化学习中主动优化感知质量。
