## TSPO__Temporal_Sampling_Policy_Optimization_for_Long-form_Video_Language_Understanding

- baseline方法是什么？
  Baseline 为 Video-MLLM 的 **uniform frame sampling**（均匀帧采样）和 **training-free keyframe search**（无训练关键帧搜索）。全栈执行例子：
  - 模型推理算法层：Video-MLLM（如 LLaVA-Video-7B）对长视频以固定 FPS（如 1 FPS）均匀采样 64 帧，所有帧权重相同，不考虑查询内容。Training-free 方法如 LongVU 使用 DINOv2-1B 提取帧间差异选择关键帧，或 CoS 使用 LLaVA-1.5-13B 进行查询相关帧过滤——但这些 selector 是预训练模型，无法针对 Video-MLLM 的最终任务进行优化。所有方法中帧采样和语言生成是两个独立阶段。
  - 系统框架层：论文未明确说明。使用标准 Video-MLLM 推理流程：视频解码→帧采样→视觉编码→token 拼接→LLM 自回归生成，无专门的调度或编译框架。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。使用 PyTorch + DeepSpeed 标准训练栈，Flash-Attention 加速注意力计算。
  - 硬件架构层：论文未明确说明。运行在 8×NVIDIA A800 80GB GPU 上。

  Baseline 的核心缺陷：
  1. **无监督性 (Unsupervised)**：通用视频理解训练中缺乏帧级标注，uniform sampling 无法知道哪些帧对回答关键，training-free 方法依赖预训练 selector 的跨模态理解能力，无优化空间。
  2. **不可微性 (Non-differentiable)**：帧采样是离散子集选择问题，输出为帧索引而非连续变量，无法通过 SFT 反向传播直接优化采样策略。
  3. **计算冗余**：Training-free 方法如 CoS 额外调用 MLLM-13B 做帧选择，推理开销大（28.4s frame time vs TSPO 1.2s）。
  4. **查询无关**：Uniform sampling 对所有查询采样相同帧，忽略查询-事件关联。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  TSPO 将关键帧选择和语言生成建模为联合决策过程，通过强化学习（GRPO）端到端优化时序采样策略。全栈执行例子：
  - 模型推理算法层：
    (1) **Event-aware Temporal Agent**：基于 CLIP-Large（400M 冻结）+ 3.5M 可学习参数。输入候选帧（1FPS 均匀采样）和查询文本的 CLIP 特征 → local window attention 注入事件感知和时序位置编码 → 融合 event-level 和 frame-level 的 cross-modal similarity → Gumbel-Softmax 概率化 TopK 采样 → 输出关键帧索引和概率。
    (2) **TSPO RL 优化**：将采样策略 π_ts(V_s|q,V_c) 和语言生成 π_l(o|q,V_s,V_c) 联合建模为 π(o,V_s|q,V_c) = π_l · π_ts。Video-MLLM (π_l) 保持冻结，仅通过 GRPO 优化 Temporal Agent (π_ts)。GRPO 对每组 query 采样 G 个关键帧组合，以 rule-based reward（答案准确性 R_A + 时序定位 R_T）计算组内相对优势 A_i，最大化期望奖励。无需帧级标注——语言级答案正确性（多选题选项匹配）直接监督帧选择策略。
    (3) **双风格训练数据**：Comprehensive Temporal Data（过滤太易/太难的多选题，保留需多关键帧的题目）+ Video Needle-in-a-Haystack Data（合成超长视频 10∼60min，训练长程时序定位能力）。
    推理时去除 Gumbel 噪声，直接确定性采样 64 帧（可降至 32 帧仍超 baseline），比 CoS 节省 90% 帧提取时间。
  - 系统框架层：论文未明确说明。基于 DeepSpeed 分布式训练，Video-MLLM 骨干可替换（迁移实验验证了 LLaVA-Video→Qwen2VL/Qwen2.5VL 的 zero-shot 迁移能力）。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。
  - 硬件架构层：论文未明确说明。

  **解决 Baseline 缺陷的映射**：
  1. 无监督性 → **语言级奖励替代帧级标注**：TSPO 利用最终回答的准确性（R_A）和粗粒度定位准确率（R_T）作为奖励信号，language supervision 通过 GRPO 的期望最大化间接指导帧选择，无需任何帧级 ground-truth 标注。
  2. 不可微性 → **RL 替代 SFT 反向传播**：Gumbel-Softmax 提供可微的离散采样近似，GRPO 的 policy gradient 方法天然处理离散动作空间（帧索引选择），避免了对不可微采样的直接梯度需求。
  3. 计算冗余 → **轻量级 Temporal Agent (3.5M)**：相比 CoS 使用 MLLM-13B 做帧选择（28.4s），TSPO 的 CLIP-based agent 仅需 1.2s 帧提取时间，且推理时可降低采样帧数（32 帧实现 token 减半、LLM 时间减半）。
  4. 查询无关 → **查询驱动的自适应采样**：Event-aware agent 计算帧-查询 cross-modal similarity，对每个查询动态选择不同的关键帧组合，而非对所有查询使用相同帧。
