## Free-MoRef__Instantly_Multiplexing_Context_Perception_Capabilities_of_Video-MLLMs_within_Single_Inference

- baseline方法是什么？
  Baseline 方法：现有的 Video-MLLM（LLaVA-Video-7B）在长视频处理中面临两种典型策略：(1) 直接扩展输入帧数 —— 将更多帧通过 Vision Encoder 编码为 token 序列，送入 LLM 的 full self-attention 推理。当帧数超过 64 时，vision token 数量暴增（每帧 182 tokens），序列长度超出 Qwen2-7B 的 32768 token 阈值，导致 OOM 或显著的性能退化（如 256 frames full attention FLOPs 为 64 frames 的 1600%）；(2) Token Compression（如 FastV、LLaMA-Vid）—— 在推理前压缩/剪枝 vision tokens，高压缩率导致信息损失；(3) Streaming Inference —— 多次调用 LLM，复用历史 KV Cache，延迟与上下文长度成正比。

  Baseline（LLaVA-Video-7B, 64 frames, full attention）全栈执行例子：
  - 算法层：视频 → FPS=1 采样最多 64 帧 → SigLIP Vision Encoder 逐帧编码为 182 tokens/frame → spatial pooling (2×2) → 64×182=11648 vision tokens → Projector 映射到 LLM embedding space → 拼接 system prompt + vision tokens + question text → Qwen2-7B 28 层 causal self-attention（每层对全部 11648 tokens 做 full FlashAttention）→ 自回归 decode → 答案。若扩展至 128 frames，序列长度 23296 tokens，full attention 的 FLOPs 为基准的 400%；256 frames 则达 46592 tokens（1600% FLOPs），单卡 A100 直接 OOM。
  - 系统框架层：基于 HuggingFace Transformers + lmms-eval 评估框架，使用 accelerate 工具包管理显存
  - 编译框架层：论文未明确说明
  - kernel 调度层：标准 FlashAttention（causal mask），无特殊 kernel 优化
  - 硬件架构层：单张 NVIDIA A100 GPU

  Baseline 的缺陷：
  1. **O(N²) 计算复杂度导致长上下文难以扩展**：vision token 数量随帧数线性增长，但 full attention 计算代价按 O(n²) 增长，128 frames（2×帧数）需要 4× FLOPs，256 frames（4×帧数）需要 16× FLOPs，512 frames（8×帧数）需要 64× FLOPs。单卡 A100 在 256 frames 时直接 OOM。
  2. **Token Compression 导致信息丢失**：压缩/剪枝 vision tokens 可以控制序列长度，但高压缩率意味着关键视觉信息可能被丢弃，尤其在需要细粒度推理的长视频问答中损失严重。
  3. **Streaming Inference 引入线性延迟增长**：KV Cache 复用虽支持任意长上下文，但延迟与上下文长度成正比，无法在单次推理中高效完成长视频理解。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：Free-MoRef 是一个 training-free 方法，受 MoE（Mixture-of-Experts）范式启发，通过将长 vision token 序列划分为多个 short reference chunks（Multi-Reference Partition），在 shallow decoder layers 中用 MoRef Attention 并行 query 各 chunk 并聚合统一 activation，在 mid-layer 通过 Reference Fusion 合并 parallel chunks 为 global reference 后在 deep layers 中标准推理。

  对比 baseline 的全栈执行例子（Free-MoRef@LLaVA-Video-7B, 512 frames, 8 parallel chunks）：
  - 算法层：视频 → FPS=1 采样 512 帧 → SigLIP Vision Encoder 编码 → 512×182=93184 vision tokens → Multi-Reference Partition：M=64 temporal units, N=8 chunks → 每个 chunk 约 11648 tokens → 拼接相同 system prompt + question → 输入 LLM。Shallow layers 0-11（L=12）：各 chunk 独立执行 FlashAttention，产生 O_i = [O_i^sys, O_i^vis, O_i^ques]，计算跨模态注意力 A_i = softmax(Q^ques × (K^vis)^T) → gating weights w_i = max(A_i) / Σ max(A_j) → 聚合 O^fusion = Σ w_i · O_i^ques → 组装 MoRef 输出并替换 O^ques → 残差 + FFN。在 layer 12 执行 Reference Fusion：基于 E_i = mean(A_i, dim=ques)，每个 chunk 保留 top 1/8 vision tokens 并聚合为 global reference（约 11648 vision tokens）。Deep layers 12-27：仅用 global reference 做标准 self-attention → 自回归 decode → 答案。
  - 系统框架层：基于 HuggingFace Transformers + lmms-eval，使用 accelerate toolkit 辅助显存管理。512 frames@Free-MoRef 仅需 400% FLOPs（vs baseline 6400%），且可直接在单卡 A100 上运行（baseline 在 256 frames 已 OOM）。
  - 编译框架层：论文未明确说明
  - kernel 调度层：标准 FlashAttention（MoRef Attention 仍然兼容 FlashAttention 的 causal 接口），额外仅需一次 query-vision cross-modal attention（计算量可忽略）
  - 硬件架构层：单张 NVIDIA A100 GPU，512 frames Free-MoRef 可在不使用 accelerate 的情况下推理，而 baseline 在 256 frames 已 OOM

  解决 baseline 缺陷的对应关系：
  1. **Multi-Reference Partition + MoRef Attention → 解决 O(N²) 计算复杂度**：将长度为 N·L 的序列划分为 N 个长度为 L 的 chunk 并行处理，每层 attention 计算复杂度从 O((N·L)²) 降至 O(N·L²) ≈ O(1/N · full attention)。128 frames 时 FLOPs 仅 110.4%（vs full attention 400%），256 frames 时 163.2%（vs 1600%），512 frames 时 400%（vs 6400%），实现了随帧数线性增长的 FLOPs 而非二次增长。
  2. **MoRef Attention 的全感知机制 → 解决 Token Compression 的信息丢失**：通过 query-vision cross-modal attention 计算 gating weights 来聚合各 chunk 的 question token activation，使得所有 vision tokens 都参与到每个 decoder layer 的 question token 更新中，实现了"equivalent to full attention"的全上下文感知，无需丢弃任何 token。
  3. **单次推理并行处理 → 解决 Streaming Inference 的延迟增长**：所有 chunks 在单个 forward pass 中并行处理，first token latency 保持恒定（与 64-frame baseline 相当），不支持额外的逐 chunk 串行推理延迟，实现 "instant responses"。
