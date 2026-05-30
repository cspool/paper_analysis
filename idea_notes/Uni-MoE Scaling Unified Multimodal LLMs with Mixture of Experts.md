## Uni-MoE Scaling Unified Multimodal LLMs with Mixture of Experts

- baseline方法是什么？
  Baseline 是传统的 **稠密（Dense）统一多模态大语言模型**，如 Macaw-LLM、X-InstructBLIP，以及单专家 Dense 模型（Single-Modality-Expert）。这些模型在处理每种输入时激活全部参数（稠密计算），导致训练和推理的计算开销随模型规模和多模态数据种类增加而成比例增长。全栈执行例子（以 X-InstructBLIP 处理"语音提问+图像"三模态输入为例）：
  - **模型推理算法层**：语音编码器（Whisper）+ 图像编码器（CLIP）分别编码，通过 Connector 映射到 LLM 语言空间，所有 tokens 串联后送入 LLM 的每一层，每一层的 FFN 均为稠密计算——所有参数被激活处理每个 token。多模态数据混合训练时，Dense 模型一个专家/MLP 需要同时学习图像、语音、文本等所有模态的表示，容易产生模态间干扰，训练 loss 波动大，且对长语音等复杂模态的外推泛化能力差。
  - **系统框架层**：标准 PyTorch + HuggingFace Transformers 训练栈，数据并行（DP）训练。多模态多任务数据混合训练时所有样本经过相同模型参数，无专家级模型并行和模态级数据并行。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Uni-MoE**——基于稀疏 MoE 架构的统一多模态 LLM，通过三个关键设计解决 Baseline 缺陷：
  1. **稀疏 MoE 替代稠密 FFN**——将 LLM 中部分层的稠密 FFN 替换为含 4~8 个专家的稀疏 MoE 层，每个 token 仅激活 top-2 专家（激活参数远小于总参数），大幅降低推理计算开销。例如 Uni-MoE-7B×4-Top2 激活 8.9B 参数但总参数 13.2B，相比 Dense 7B 模型仅增加约 2.2B 激活参数即可处理 5 种模态。
  2. **模态特定专家预训练（阶段二）**——每个专家在不同模态数据上分别预训练（如 Expert 2 用文本-图像数据，Expert 3 用语音-图像数据，Expert 4 用纯音频数据），使各专家发展出模态偏好。Router 在学习过程中能自动将不同模态的 tokens 路由到对应专业专家（如音频 tokens → Expert 4，图像 tokens → Expert 2），解决 Dense 模型中单一 FFN 需同时学习所有模态导致的模态间干扰。
  3. **LoRA 微调 + 专家级模型并行**——在阶段三使用 LoRA（rank=8）微调预训练专家和自注意力层，冻结专家本体参数，仅更新低秩适配器和 Router。同时实现专家级模型并行（expert-level model parallelism）和模态级数据并行（modality-level data parallel），使训练可扩展到多节点多 GPU。
  
  全栈执行例子（以 Uni-MoE MoE-Task3 处理"视频+音频+文本"为例，含 4 个预训练专家）：
  - **模型推理算法层**：视频 8 帧通过 CLIP-V 编码后平均池化→视觉 tokens；音频通过 BEATs 编码→Audio-QFormer（4 层 cross-attention 蒸馏）→音频 tokens；文本通过 Word-Embedding→文本 tokens。三类 tokens 串联后输入 LLM。在 MoE 层中，Router 对每个 token 计算 softmax(W_router · x)，选择 top-2 专家。可视化分析（Figure 4-5）显示：音频 tokens 主要由 Expert 4（音频预训练）处理，图像 tokens 主要由 Expert 2（图像-文本预训练）处理，视频的多模态 tokens 在后期层由多专家协作处理。相比 Dense 模型，MoE 路由实现了模态感知的负载分配，使得长语音理解（RACE-Audio high）从 Dense 的 29.02% 提升至 49.37%，且在混合多模态数据上训练 loss 更稳定收敛（Figure 3 蓝色线）。
  - **系统框架层**：基于 PyTorch 分布式训练，实现数据并行（modality-level data parallelism——不同模态数据在不同 device 上处理）+ 专家并行（expert-level model parallelism——不同专家分布在不同 GPU 上）。阶段三支持多节点多 GPU（16×A800）训练 8 专家模型。论文公布了两种分布式并行训练方法。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明。
