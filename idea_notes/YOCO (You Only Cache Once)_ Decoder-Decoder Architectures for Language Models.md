## YOCO (You Only Cache Once): Decoder-Decoder Architectures for Language Models

- baseline方法是什么？
  Baseline 是标准 **decoder-only Transformer**（Llama 架构优化版），包含 RMSNorm、SwiGLU、RoPE、Grouped-Query Attention (GQA) 等现代改进。每层执行全局因果 self-attention：Q=XW_Q, K=XW_K, V=XW_V，计算 Attention(Q,K,V)=softmax(QK^T/√d_k+M)V，需存储全部 L 层的 per-token KV cache。

  Baseline 全栈执行例子（Transformer 7B, 512K context, 4×H100-80GB）：
  - 算法pipeline：序列 x → Embedding → L=32 层 decoder：每层 Masked MHA（QKV 投影 → QK^T/√d → causal mask → softmax → ×V → output proj，Flash-Decoding + kernel fusion 优化）→ SwiGLU FFN → 残差连接 → classifier。prefill 阶段：512K tokens 全部并行前向，需存储 32 层 × 512K × 2(KV) × d_head × h_kv × 2bytes。decode 阶段：每步从 HBM 读取全部 KV cache，HBM 带宽是瓶颈。
  - 系统框架：HuggingFace Transformers / 自定义推理框架，Flash-Decoding attention kernel，Triton fused kernel。
  - 编译框架：论文未明确说明。
  - kernel调度：Flash-Decoding（适用于长序列 attention）、kernel fusion（融合 LayerNorm/QKV 投影等操作）。H100 GPU 上执行。
  - 硬件架构：NVIDIA H100-80GB GPU（Ampere 下一代，Hopper 架构），无自定义硬件。

  **Baseline 的核心缺陷：**
  1. **KV Cache 内存随 L 线性增长**：每层都需存储 N 个 token 的 K,V，总 KV cache = O(L×N×D)。65B 模型 512K tokens 时 KV cache 占用约 86GB，超过单张 H100-80GB 容量。这是 LLM 长上下文推理的主要内存瓶颈。
  2. **Prefill 延迟 O(N²)**：softmax(QK^T) 的计算复杂度与序列长度平方成正比。7B 模型 1M tokens prefill 需约 380s（4×H100），严重影响用户体验。
  3. **长序列训练通信瓶颈**：分布式长序列训练时，每层的 all-gather 通信随序列长度和层数增加，吞吐受限。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **YOCO (You Only Cache Once)** decoder-decoder 架构，通过三个核心设计解决 baseline 缺陷：

  **(1) Self-Decoder + Cross-Decoder 分离解决 KV Cache 内存瓶颈**
  Baseline 每层都自注意力和自产 KV cache。YOCO 将 L 层平分为 Self-Decoder（前 L/2 层）和 Cross-Decoder（后 L/2 层）。Self-Decoder 使用高效自注意力（gated retention 或 sliding-window attention），仅需 O(1) 常量内存（如 retention 的 recurrent state S ∈ R^{d×d}，或 sliding-window 的固定窗口 C）。Cross-Decoder 所有层共享同一组全局 KV cache（K̂, V̂），仅由 Self-Decoder 最终输出生成一次。总 KV cache = O(N + CL) ≈ O(N)（C 为常量）vs Baseline O(NL)，大约节省 L 倍内存。65B 模型 KV cache memory 降低约 80×。

  **(2) Prefill Early Exit 解决 Prefill 延迟瓶颈**
  Baseline prefill 必须执行全部 L 层的前向计算（含 O(N²) attention）。YOCO 的 Cross-Decoder 仅依赖 Self-Decoder 的输出 K̂, V̂，因此 prefill 阶段可在 Self-Decoder（L/2 层）完成后立即退出。又因 Self-Decoder 使用高效 attention（如 retention 的 chunkwise recurrent），prefill 复杂度从 O(LN²D) 降至 O(LND)（线性于 N）。512K context：prefill 从 180s 降至 <6s（带 Flash-Decoding 优化的 Transformer baseline vs YOCO），1M context 加速 71.8×。

  **(3) Chunk Parallelism 解决分布式训练通信瓶颈**
  Baseline 的数据并行/序列并行中，每层都需 all-gather 通信。YOCO 的 Cross-Decoder 解耦了层间注意力依赖：Self-Decoder 仅需相邻设备的边界通信（如 retention 的 recurrent state 传递，或 sliding-window 的窗口边界）；Cross-Decoder 的 K̂, V̂ 仅需一次 all-gather（而非每层一次），大幅减少通信频率和 GPU memory fragmentation。

  论文方法全栈执行例子（YOCO_gRet 3B, 512K context, H100-80GB）：
  - 算法pipeline：序列 x → Embedding X^0 → **Self-Decoder（L/2=13 层，gated retention）**：每层 recurrent/chunkwise 计算 → S_n = γ_n S_{n-1} + K_n^T V_n → O_n = Q_n S_n → GroupNorm + swish gate → SwiGLU FFN → 输出 M = X^{L/2} → **生成全局 KV cache**：K̂ = LN(M)W_K, V̂ = LN(M)W_V（单次，共享给所有 Cross-Decoder 层）→ **Cross-Decoder（13 层，cross-attention）**：Q̂^l = LN(X^l)W_Q^l → Attention(Q̂^l, K̂, V̂) → SwiGLU FFN → 输出 X^L → classifier。**Prefill**：仅执行 Self-Decoder + 生成 K̂,V̂（提前退出，略过 Cross-Decoder），13 层而非 26 层。**Decode**：Self-Decoder 用 recurrent（O(1) state），Cross-Decoder 用标准 attention 复用 K̂,V̂。结果：GPU memory 12.4GB（Transformer 9.4× more），prefill <6s（Transformer 180s），throughput 43.1 tok/s（Transformer 4.5 tok/s）。
  - 系统框架：内部 CUBE 分布式训练系统（SuperScaler-based, https://github.com/microsoft/nnetscaler），HuggingFace Transformers 兼容 API。H100 GPU。
  - 编译框架：论文未明确说明。
  - kernel调度：Triton kernel：gated retention 的 chunkwise recurrent（prefill, chunk=256）+ recurrent（decode），基于 FLA 库。Baseline 使用 Flash-Decoding + kernel fusion 优化。
  - 硬件架构：NVIDIA H100-80GB GPU。无自定义硬件。论文在 Conclusion 中展望 YOCO + BitNet + Groq 的组合可进一步将部署成本降低数个数量级。

  关键设计动机映射：
  - Transformer 每层 KV cache 内存 O(LND) → YOCO 的单层全局 KV cache O(ND) + Self-Decoder 常量 cache O(CL)
  - Transformer prefill 延迟 O(LN²D) → YOCO prefill early exit + 高效 attention O(LND)
  - Transformer 分布式训练每层 all-gather → YOCO Cross-Decoder 仅一次 all-gather（KV cache） + Self-Decoder 仅边界通信
  - 不同场景可选用不同 Self-Decoder 实现：gated retention（性能最优）或 sliding-window attention（实现简单）
