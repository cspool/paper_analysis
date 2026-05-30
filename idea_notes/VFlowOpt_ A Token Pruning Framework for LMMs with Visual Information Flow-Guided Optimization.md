## VFlowOpt: A Token Pruning Framework for LMMs with Visual Information Flow-Guided Optimization

- baseline方法是什么？
  Baseline 是 **基于 attention 的训练无关 token 剪枝方法**，以 FastV (ECCV 2024)、SparseVLM、VisionZip (CVPR 2025) 为代表。

  全栈执行例子（以 VisionZip + LLaVA-OneVision-7B 为例）：
  - 模型推理算法层：SigLIP ViT 将 1152×1152 图像编码为 7290 个视觉 tokens → MLP Projector 映射到 LLM 空间 → LLaVA-OneVision (Qwen2-7B) 逐层处理。VisionZip 在 ViT 编码器内用 [CLS] token 的 attention scores 评估各视觉 token 重要性，无 [CLS] 时退化为各 token 收到的平均 attention。按固定剪枝比率一次性丢弃低分 tokens（单阶段剪枝），丢弃的 token 信息永久丢失。剪枝策略（保留率、剪枝层位置）对所有 LMM 模型统一使用手工设定，无模型特定优化。
  - 系统框架层：基于 LMMs-Eval 框架的标准推理 pipeline（HuggingFace Transformers），无特殊 Serving 修改。推理时 KV-cache = O(L × T_vision × D)，token 剪枝后 KV-cache 等比例缩小。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：使用标准 PyTorch + HuggingFace 推理，GPU (A100) 上运行，无自定义 kernel。
  - 硬件架构层：NVIDIA A100-SXM4-80GB GPU，无硬件定制。

  核心缺陷：(1) **Attention-based importance map 有偏**：冗余 token（如背景区域）对同类冗余 token 分配不恰当的高 attention，用所有 token 的 attention 均值估计重要性会导致背景 token 重要性被高估、关键 token 被低估；(2) **单阶段粗粒度剪枝导致信息丢失**：所有冗余 token 一次性丢弃且不可恢复，尤其在浅层剪枝时，某些看似不重要的 token 在深层可能变得关键；(3) **剪枝策略缺乏模型适配性**：同一手工设定的剪枝超参数（如保留率、剪枝位置）不加区分地应用于所有 LMM，而不同模型（LLaVA vs Qwen2-VL）的 visual information flow 特征不同，统一策略导致性能显著退化；(4) **纯 attention 信号忽略视觉内容丰富度**：仅依赖 attention 分数评估重要性，但 attention 反映的是 token 间的交互强度而非图像块的信息丰富度。信息熵高的区域（纹理复杂、物体边界）即使 attention 不高也可能包含关键视觉信息。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **VFlowOpt** 引入三大创新：(1) Attention Calibration + 熵增强的重要性估计；(2) Progressive Pruning + Token Recycling；(3) Visual Information Flow-Guided Bayesian Optimization。

  对应解决四个缺陷：

  **(1) Attention Calibration 消除冗余 token 偏差：**
  Baseline → 用所有 token 的 attention 均值评估重要性，冗余背景 token 注意力偏向同类，使背景似乎"重要"。
  VFlowOpt → 先通过全局 attention 阈值 τ = t·mean(Σ_i Σ_j A_{ij}) 筛选出"相对重要"的 token 集合 K（被 attention 关注较多的 token），再仅用 K 中 token 的 attention 计算 I_i = Σ_{k∈K} A_{ki}。这样排除了冗余 token 的 noisy attention，重要性估计更可靠。消融实验证实：移除 importance calibration 导致 MMStar 从 57.8→56.2、SQA 从 92.3→91.8。

  **(2) Progressive Pruning + Token Recycling 避免信息丢失：**
  Baseline → 单阶段剪枝，所有目标 token 一次性丢弃，信息永久丢失。
  VFlowOpt → 三阶段逐步剪枝：LLM 处理过程平均保留率 R̄ = (R1·L1 + R1·R2·L2 + R1·R2·R3·L3)/L。初始阶段保留更多 token（浅层视觉信息更关键），深层逐渐激进剪枝（冗余增加）。Token Recycling：将剪枝 token 按 a×a 空间网格分组，组内以重要性为权重做加权平均融合 t_merged = Σ I_i·t_i / Σ I_i，融合 token 替换该网格最高重要性 token 位置归入保留集合。这样在减少 token 数量的同时，压缩保留了低重要性区域的视觉特征。消融实验证实：移除 token recycling 导致 POPE 从 89.1→86.8；移除 progressive pruning 导致 MMStar 从 57.8→56.0。

  **(3) Visual Information Flow Optimization 自动适配不同 LMM：**
  Baseline → 手工设定保留率，所有模型用同一策略。
  VFlowOpt → 基于 LMM 可解释性研究的发现（视觉信息从 vision tokens → query text tokens → 最后位置 last token 聚合），将剪枝策略设计建模为优化问题：max CosineSim(h_f, g_s(h_f))，即最大化无剪枝与剪枝最后 token 表示的余弦相似度。用 Bayesian Optimization（GP + Expected Improvement, 50 迭代, 30 样本, ≈30 分钟）自动搜索超参数 (R1, R2, R3, t, α, a)。核心洞察：不同的 LMM 有不同的内部信息流特征——LLaVA、LLaVA-NeXT、Qwen2-VL 三者的 ViT 架构、LLM backbone、信息聚合模式各不同——统一策略必然 suboptimal，自动优化实现"一模型一策略"的定制化。

  **(4) 熵项捕捉视觉内容丰富度：**
  Baseline → 纯 attention 信号评估重要性，忽略图像内容。
  VFlowOpt → 将图像块信息熵 H(V_i) = -Σ p_k·log(p_k)（256 灰度级）归一化后加权加入重要性得分，α 控制熵贡献权重。高熵区域（纹理复杂、细节丰富）即使 attention 一般也会获得加分，实现"attention 告诉你 token 间谁重要，熵告诉你谁的内容值得看"的双信号评估。两个信号均由 Bayesian Optimization 自动调权，无需手工设定 α。

- baseline方法是什么？
  Baseline 是 **Transformer-based MLLMs for long video understanding**，以 Video-XL (Qwen2-7B + 内部 token 压缩)、VideoChat-Flash (Qwen2-7B + 层级压缩)、LLaVA-Video (Qwen2-7B + ViT fine-tuning) 为代表。

  全栈执行例子（以 Video-XL 为例）：
  - 模型推理算法层：ViT 编码视频帧为 vision tokens → Projector 压缩（可选 ToMe 等）→ 全 Transformer LLM (Qwen2-7B) 逐层计算 self-attention：每层 attention 复杂度 O(T²)，其中 T = T_vision + T_text。Video-XL 在 LLM 内部做 token 压缩，基于 attention score 选择保留的 vision tokens 并生成新的 compact tokens。所有层均为 self-attention，KV-cache 随序列长度线性增长。
  - 系统框架层：基于 HuggingFace Transformers 的标准 MLLM pipeline，无特殊 Serving 框架修改。推理时 KV-cache 内存 = O(L × T × D)，长视频场景下 KV-cache 成为主要瓶颈。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：使用标准 Flash Attention 加速 Transformer attention 计算，无自定义 kernel。长序列下 attention 仍是计算瓶颈。
  - 硬件架构层：NVIDIA GPU（如 A100/H100），无硬件定制。

  核心缺陷：(1) **Attention 计算复杂度高**：全 Transformer backbone 的 self-attention 为 O(T²)，处理小时级视频（如 2.7M tokens）不可行；(2) **KV-cache 内存爆炸**：全 Transformer 的 KV-cache 为 O(L×T)，长序列推理时显存成为瓶颈；(3) **Vision token 冗余未在 hybrid 架构中探索**：现有 token dropping/compression 策略均基于 Transformer attention scores 设计（如基于 attention 分数排序丢弃），在 hybrid Mamba-Transformer 架构中信息存储/传递机制不同，直接迁移不可行；(4) **Compression 信息丢失**：token dropping 策略不可逆，被丢弃的 token 信息永久丢失；token compression 生成新 special tokens 则破坏了原始 token 的身份和位置信息。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **TimeViper** 引入两大创新：(1) Hybrid Mamba-Transformer LLM Backbone（Nanov2-9B），用 SSM 替代大部分 attention 层；(2) TransV token transfer mechanism，通过 gated cross-attention 将被丢弃的视觉 token 信息转移到指令 token 中。

  对应解决四个缺陷：

  **(1) O(n) 替代 O(n²)：**
  Baseline (Video-XL) → 全 Transformer 每层 O(T²) attention。
  TimeViper → 27 层 Mamba-2 每层 O(n) 递推计算（h_t = A_t·h_{t-1} + B_t·x_t），仅 4 层 self-attention 保留 O(n²)。实际效果：输入 32K tokens（约 2K frames×16 tokens）、输出 1K tokens、batch_size=32 时，TimeViper 每秒生成 token 数比 Qwen3 高 40.1%。

  **(2) O(1) KV-cache 替代 O(L×T) KV-cache：**
  Baseline → KV-cache 随序列长度线性增长，长视频推理显存爆炸。Vanilla 模型在 128 frames 即 OOM。
  TimeViper → Mamba-2 层仅需维护固定大小隐状态 h_t ∈ R^{N×D}（N≈128），无 KV-cache 存储需求；4 层 self-attention 的 KV-cache 远小于全 Transformer。ToMe+TransV 联合压缩后，4K frames 时内存节省 54.8%，可处理 10K+ frames。

  **(3) Hybrid 架构专用的 token 压缩策略：**
  Baseline → Attention-based token dropping（如 PDrop、PyramidDrop），依赖 Transformer attention scores 识别冗余 token，在 hybrid 架构中 Mamba 层缺乏显式 attention scores。
  TimeViper → 首先通过信息交换分析实验（blocking V2I/V2R）揭示 **vision-to-text information aggregation 现象**：视觉信息在浅层从 vision tokens 汇聚到 instruction tokens，深层 vision tokens 近乎 100% 冗余。基于此发现设计 TransV：(a) 浅层用 uniform dropping 避免 attention score 不可靠的问题；(b) 深层用 attention-guided dropping（以最后一个 instruction token 为 query 计算 attention scores）此时 attention 已可靠；(c) 在丢弃前通过 gated cross-attention 将被丢弃 token 的信息转移到 instruction tokens，避免信息丢失。α_l 初始化为 0 → 训练后学习到最优转移比例。

  **(4) Token Transfer 替代 Token Dropping/Compression：**
  Baseline → token dropping 不可逆丢失信息；token compression 生成新的 special tokens 破坏原始 token 身份。
  TimeViper → TransV 的 cross-attention 将被丢弃的 vision tokens 作为 KV、instruction tokens 作为 Q，计算得到的信息增量通过 tanh(α)·CrossAttn 加回到 instruction tokens。这意味着视觉信息被"转移"而非"丢弃"，instruction tokens 承载了原属于被丢弃 vision tokens 的关键信息。保留的 5% vision tokens + enriched instruction tokens 足以支撑任务性能。

  全栈执行例子（对比 baseline）：
  - 模型推理算法层：
    Baseline (Video-XL) → ViT encode → Projector → Qwen2 self-attention 逐层计算（每层 O(T²) attention + O(T) KV-cache 增长）→ 基于 attention score 选择保留/丢弃 tokens → 生成回答。
    TimeViper → SigLIP encode frames (768 tokens/frame) → ToMe 压缩至 16 tokens/frame → Concat [vision_tokens, instruction_tokens] → 逐层交替通过 Mamba-2 (SSM 递推，h_t = A_t h_{t-1} + B_t x_t，O(n)，无 KV-cache) 和 Self-Attention（标准 QK^T/sqrt(D)，O(n²)，有 KV-cache）→ 第 7 层 TransV：UniformDrop 50% vision tokens，CrossAttn(Q_inst, KV_dropped_vis)→ instruction tokens 吸收视觉信息 → 丢弃被转移的 vision tokens → 第 39 层 TransV：Attention-guided drop 90% remaining vision tokens（仅保留原 vision tokens 的 5%）→ CrossAttn 转移信息 → 深层继续 processing（512 frames 时 context 从 8192+text 降至 ~435+text tokens）→ 生成回答。关键区别：Mamba-2 层通过遗忘-记忆门控机制隐式建模时序位置（无需 MRoPE），TVG 任务 mIoU 仍达 40.5，超过使用 MRoPE 的 Qwen2.5-VL-7B(43.6) 的 93%。
  - 系统框架层：
    Baseline → HuggingFace Transformers 标准 MLLM，推理时 prefill → decode 两阶段，KV-cache 管理。
    TimeViper → 同标准 MLLM 训练 pipeline：两阶段训练（Image-Text Alignment 3M → Video Instruction Tuning 4.8M），Data Packing 支持 TransV 可变序列长度。推理时 TransV 在 prefill 阶段执行 token 压缩，decode 时 context 已大幅削减 → prefilling time 在 4096 frames 时减少 15.7% vs ToMe-only。兼容 HuggingFace Transformers。
  - 编译框架层：论文未明确说明。使用 PyTorch + 标准训练框架。
  - kernel 调度层：论文未明确说明。Mamba-2 SSM 使用标准 selective scan kernel，self-attention 使用 Flash Attention 加速。
  - 硬件架构层：论文未明确说明。NVIDIA GPU 推理。
