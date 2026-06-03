# Q-VDiT Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers

- 导出时间：2026-06-03
- 来源：当前 Claude Code session 可用对话上下文
- 保存路径：human_notes/Q-VDiT Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers.md
- 范围：仅用户输入与 Claude 最终输出

## 对话记录

### 001 User
教我,'量化	Q-VDiT (视频 DiT 量化)	TQE (Token-Quantization Error compensation, rank-1 低秩) + TMD (Temporal Model Distillation)；W4A6 几乎无损, W3A6 SC 23.40	ViDiT-Q, FlatQuant, LoRunner Kernel'

### 002 Claude
[Claude 给出了 Q-VDiT 的完整讲解，包括：
- 五个语义分段（S1: 量化、S2: Q-VDiT、S3: TQE+TMD、S4: W4A6/W3A6 实验结果、S5: ViDiT-Q/FlatQuant/LoRunner Kernel）
- 每个分段包含 What/Why/How 三维度、结构化例子（公式/伪代码/Mermaid 流程图/序列图/Gantt 图）、Annotations、Note Evidence、Uncertainties
- Paragraph Synthesis 含 Segment Relationships 流程图、Integrated Understanding、Technical Comparison Table
- 核心结论：Q-VDiT 是首个视频 DiT 专用 PTQ 框架，通过 TQE (信息论指导的 rank-1 误差补偿) + TMD (KL 散度对齐帧间时序分布)，在 W4A6 下几乎无损，W3A6 下 SC 达 23.40 (近 2× SOTA)，推理时通过 LoRunner Kernel 融合低秩分支与量化 GEMM，额外延迟 <5%。]

### 003 User
TMD (Temporal Model Distillation)是**考虑帧间差异的量化参数优化?

### 004 Claude
不完全准确——TMD 比「考虑帧间差异」更精细。

**你的描述**：「考虑帧间差异的量化参数优化」
- 「帧间差异」→ 不够精确
- 「量化参数优化」→ 不完全（TMD 也优化 TQE 参数 α, β, M）

**实际机制**：TMD 对齐的是帧间相似度的概率分布，而非直接的帧间差异：

D^{FP}_i = softmax([cos_sim(S^{FP}_i, S^{FP}_1), ..., cos_sim(S^{FP}_i, S^{FP}_t)])
L_{temporal} = Σ KL(D^{FP}_i || D^{Q}_i)

关键区别：
- 比较的是帧间相似度的 softmax 分布，而非帧间差异大小
- 帧 i 的分布受所有 t 帧共同归一化约束，而非仅两两比较
- 同时优化量化参数 (s, z) 和 TQE 参数 (α, β, M)

为什么 softmax 分布对齐更强：通过 softmax 归一化，每帧的分布 D_i 是一个概率向量——帧 i 与帧 j 的相似度不仅取决于 (i,j) 这一对，还受帧 i 与所有其他帧的关系影响。梯度 ∂L_temporal/∂T^Q_{i,j} = D^Q_{i,j} - D^FP_{i,j}，任意一对 (i,j) 的梯度都受所有帧 k=1...t 的共同数值影响。

更精确的描述：TMD 是一种时序维护蒸馏损失——通过 KL 散度对齐 FP 教师模型与量化学生模型的帧间余弦相似度分布，使每帧的优化受视频中所有帧共同引导，从而在 PTQ 校准训练中联合优化量化参数和 TQE 参数。它不是简单地"考虑帧间差异"，而是构建了一个全局归一化的帧间关系分布约束。