## YOCO (You Only Cache Once): Decoder-Decoder Architectures for Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  YOCO 提出了 decoder-decoder 架构替代传统 decoder-only Transformer，核心实现包含两个组件：(1) **Self-Decoder**：使用高效自注意力（gated retention 或 sliding-window attention），仅需 O(1) 常量 KV cache 内存；(2) **Cross-Decoder**：通过 cross-attention 复用 Self-Decoder 生成的全局 KV cache，使整体模型仅需 O(N) 而非 O(NL) 缓存。Self-Decoder 占前 L/2 层，Cross-Decoder 占后 L/2 层。实验比较了：(a) 与 OpenLLaMA-3B、StableLM-3B 的 LM Eval Harness 零样本下游任务性能（1T/1.6T tokens 训练）；(b) 从 160M 到 13B 的 scaling curves（对比 Llama-Transformer、YOCO_gRet、YOCO_SWA）；(c) 1M 上下文长度的 needle retrieval 和长序列 PPL；(d) 推理效率：GPU memory、prefill latency、throughput（32K-1M 长度，H100-80GB）。

- 硬件平台是什么，配置是什么。
  NVIDIA H100-80GB GPU（推理 profiling 实验）。训练平台论文未明确说明（使用内部 CUBE 分布式训练系统，GPU 集群由 Ben Huntley 维护）。Triton kernel 实现基于 FLA (Flash-Linear-Attention) 库。

- 模型是什么。数据集和bench分别是什么。
  模型：3B YOCO 主模型（hidden=3072, layers=26, query heads=24, KV heads=8 with GQA, non-embedding params=2.8B）；scaling 模型从 160M 到 13B（7 种尺寸）。
  数据集：训练语料类似 StableLM-3B-4E1T 的 curated corpus，tokenizer 为 tiktoken-c1100k_base。
  Benchmark：LM Eval Harness（ARC-C, ARC-E, BoolQ, HellaSwag, OBQA, PIQA, Winogrande, SciQ）、Needle-in-a-Haystack（1M 长度）、Multi-Needle Retrieval（128K 长度）、长序列 NLL（book + repository-level code, >1M tokens）。Scaling 曲线使用 validation loss。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源：https://aka.ms/YOCO。基于 FLA（https://github.com/sustcsonglin/flash-linear-attention）实现 gated retention 的 Triton kernel。

  算法pipeline（YOCO 推理流程）：
  1. **输入**：序列 x = [x_1, ..., x_N] ∈ R^{N×d_model}，输入 embedding X^0
  2. **Self-Decoder 前向**（L/2 层，gated retention 或 sliding-window attention）：
     - Gated Retention（默认，推理用 recurrent 模式）：
       - Q_n = (X_n W_Q) ⊙ Θ, K_n = (X_n W_K) ⊙ Θ̄, V_n = X_n W_V
       - γ_n = sigmoid(X_n W_γ)^{1/τ}
       - S_n = γ_n S_{n-1} + K_n^T V_n  （recurrent state update, O(1) memory）
       - gRet(X_n) = Q_n S_n
     - 或 Sliding-Window Attention：每个 query 仅关注窗口大小 C 内的 key
  3. **全局 KV Cache 生成**：K̂ = LN(X^{L/2}) W_K, V̂ = LN(X^{L/2}) W_V（单层全局缓存）
  4. **Cross-Decoder 前向**（L/2 层）：
     - Q̂^l = LN(X^l) W_Q^l
     - Y^l = Attention(Q̂^l, K̂, V̂) + X^l  （cross-attention，复用共享 KV cache）
     - X^{l+1} = SwiGLU(LN(Y^l)) + Y^l
  5. **Prefill 优化**：Cross-Decoder 的 cross-attention 仅依赖 K̂, V̂，prefill 阶段可在 Self-Decoder 完成后提前退出，仅需 L/2 层前向计算
  6. **输出**：X^L → softmax classifier → next-token prediction

  关键张量计算（gated retention, recurrent mode, 单 head）：
  - S_0 = 0 ∈ R^{d×d}
  - 对 timestep n=1..N：K_n ∈ R^d, V_n ∈ R^d, γ_n ∈ R
    - S_n = γ_n · S_{n-1} + K_n^T · V_n  （outer product, O(d²) state）
    - O_n = Q_n · S_n  （vector-matrix product, O(d²)）
  - 推理时仅维护 S_n 为中间状态，不存储 per-token KV cache
