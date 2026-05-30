## Machine_Mental_Imagery__Empower_Multimodal_Reasoning_with_Latent_Visual_Tokens

- baseline方法是什么？
  Baseline 分为两类：
  (1) **文本-only 推理 VLM**（Zero-Shot / Direct SFT / CoT SFT / GRPO / CoT SFT+GRPO）：VLM 仅输出文本 token，通过 chain-of-thought prompting 或 RL fine-tuning 延长文本推理轨迹。VLM 的 vision encoder 编码输入图像后，通过 MLP projection 映射到 LLM 的 text embedding space，随后全部 token 走 LLM 自回归解码。缺陷：视觉推理需要将视觉信息"翻译"为自然语言描述（verbalize），在 jigsaw puzzle、spatial navigation 等需要视觉想象的任务上，语言中介成为瓶颈，丢失了隐空间的视觉结构信息。
  (2) **Unified multimodal 模型**（Anole, MVoT）：在 Chameleon 类统一 token-based 框架下训练模型同时输出 text tokens 和 image tokens（pixel patches），再通过 external image decoder 渲染成显式图像。缺陷：(a) 大规模像素级图像生成预训练的计算开销极大，且常损害推理质量（Wang et al., 2025 指出同时优化逻辑推理和像素合成会导致推理退化）；(b) 生成的显式图像难以与输入图像交互形成 interleaved trajectory；(c) Anole 论文复现在 spatial planning 任务上甚至无法产生有效答案。

  Baseline（以 CoT SFT 为例）全栈执行例子：
  - **算法层**：Qwen2.5-VL-7B-Instruct → 输入 (image, text_query) → vision encoder (ViT) 输出 patch features → MLP projection → 与 text token embeddings 拼接 → LLM 自回归生成 text-only CoT → LM head 输出每个 text token 的概率分布 → 最终答案。SFT loss = CE(text_tokens)。无 latent visual 通道。
  - **系统框架层**：PyTorch + HuggingFace Transformers。标准 VLM 推理流程。GRPO 使用 VERL 框架。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 Transformer 推理 kernel（FlashAttention for self-attention）。无自定义 kernel。
  - **硬件架构层**：单 NVIDIA H100 GPU 训练。Qwen2.5-VL 7B 标准 batch_size=8。

  Baseline 的核心缺陷：
  1. **视觉推理被迫语言化（Verbalization Bottleneck）**：VLM 必须将视觉空间推理转化为自然语言描述，对 jigsaw 的边缘连续性、spatial navigation 的空间关系等需要"视觉想象"的任务，语言是次级表示，丢失精确的几何与空间结构。
  2. **Unified model 的推理-生成冲突**：同时学习逻辑推理和图像合成导致两个目标冲突，推理能力退化。
  3. **隐空间视觉信道未被利用**：VLM 的 hidden states 本身富含视觉信息（vision encoder 输出经过 LLM 各层后的中间表示），但 text-only decoding 将这些信息全部丢弃（仅 LM head projection 到 vocab space），浪费了 LLM 内部已编码的视觉知识。
  4. **合成推理链的质量依赖**：CoT SFT 的性能受限于合成 reasoning chain 的质量（由 Qwen2.5-VL-32B 生成），且长 CoT 文本会导致推理效率下降。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Mirage 通过三阶段训练范式实现 latent visual token 的 interleaved reasoning：

  (1) **Latent Visual Tokens → 解决 Verbalization Bottleneck**。不生成显式图像或文本描述视觉信息，而是直接复用 VLM 当前 hidden state 作为 compact visual embedding（k=4 个连续向量），插入文本 token 流中。这些 latent tokens 承载 task-relevant 视觉线索（如 jigsaw 的边缘匹配信息、navigation 的路径空间关系），供后续 token 的 attention 机制直接访问。类比人类 "mental imagery"——不生成照片级画面，而是构建简化的心智草图。

  (2) **Stage 1 Joint Supervision → 解决隐空间视觉信道未利用**。先用 helper image I 生成 compressed visual embeddings {ê_j} 作为 ground-truth 信号，通过 cosine similarity loss 将模型在 <latent> slot 的 hidden states 锚定到 visual subspace。同时训练左右文本段的 CE loss，让模型学会在文本推理中自然编织 visual cues。L_1 = Σ cos_sim(ê_j, h_j) + γ·Σ CE(text)。γ=0.1 控制视觉约束强度。

  (3) **Stage 2 Text-Only Relaxation → 解决 Unified model 推理-生成冲突**。移除 visual alignment loss，仅保留文本 CE loss。模型自回归生成自己的 latent tokens {e_i}，梯度通过 o_post 的 CE loss 反向传播到这些连续变量（fully differentiable）。这使 latent tokens 在 visual subspace 内自适应偏移，不再强制匹配固定 ground-truth embedding，从而更灵活地服务于最终答案生成。即 stage 1 提供 grounding（锚定语义），stage 2 提供 flexibility（任务自适应）。

  (4) **Stage 3 GRPO RL → 进一步优化 interleaved trajectory**。使用 VERL + GRPO，基于 accuracy + format rewards 优化整个 interleaved 序列。latent tokens 同样接收梯度（但排除 KL penalty），使模型可探索更优的 latent-text 交织模式。

  对比 baseline 的全栈执行例子（Mirage, Qwen2.5-VL 7B, VSP Spatial Reasoning, k=4）：

  - **算法层**：Qwen2.5-VL-7B → 输入 (map_image, text_query) → vision encoder 编码图像 → MLP projection → LLM 开始自回归生成。生成触发机制：当模型需要 "think visually" 时，输出特殊 token `<latent>` → 此时不通过 LM head 映射到 vocab，而是直接取当前最后一层的 hidden state h ∈ R^d 作为 latent visual token e_1 → 将 e_1 作为连续向量拼入 key-value context（类比 KV cache 中的一条）→ 继续生成 e_2, e_3, e_4（共 k=4 个 latent tokens）→ 这 4 个连续向量携带压缩后的视觉空间信息 → 后续 o_post text tokens 的 self-attention 可以 attend 到这些 visual embeddings → LM head 输出最终答案。对比 baseline text-only：{image → text_thoughts → answer}；Mirage：{image → text_thoughts_pre → [e_1, e_2, e_3, e_4] → text_thoughts_post → answer}。
  - **系统框架层**：PyTorch + HuggingFace Transformers for SFT；VERL framework for GRPO RL。修改点：在 LLM decoding loop 中插入 latent token generation path（bypass LM head, 直接取 hidden state）；训练时需要 helper image 预处理 pipeline（vision encoder forward + average pooling compression）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 Transformer kernel。latent tokens 的生成和梯度反向传播均遵循标准 PyTorch 计算图。latent token attention 与 text token attention 共享同一 self-attention kernel（隐向量作为连续的 key-value 条目参与 scaled dot-product attention）。
  - **硬件架构层**：单 NVIDIA H100 GPU。Stage 1 ~3.5h, Stage 2 ~7.2h (VSP task)。与 text-only CoT SFT (~5.5h) 相比，Mirage 总训练时间 10.7h ≈ 2× CoT SFT，但性能提升显著（VSP Spatial Planning 从 47% → 58%, Spatial Reasoning 从 84% → 87%）。推理时 latent tokens 增加的计算开销极小（k=4 个向量参与 attention，相比 576+ vision tokens 可忽略）。
