## Block Transformer Global-to-Local Language Modeling for Fast Inference (NeurIPS 2024)

- baseline方法是什么？
  Baseline 是 **vanilla transformer**（Pythia 架构），即标准自回归 Transformer decoder，每层为全局因果 self-attention + FFN。其推理时存在两个核心瓶颈：
  
  (1) **Prefill 瓶颈**：生成第一个 token 前，必须先前向传播所有 prompt token，计算并缓存其 key-value（KV）状态。prompt 长时（如 2048 tokens），预填充延迟显著。
  
  (2) **Decode 瓶颈**：自回归生成阶段每步仅计算一个 token，但必须从 HBM 中检索所有先前 token 的 KV cache。KV cache 大小与序列长度 L 和 batch size B 线性增长（L×B），KV cache 内存访问总量随 L 二次增长（O(L²)）。在 batch decoding 场景下（实际部署常见），KV cache IO 成为主要吞吐瓶颈，远超参数 IO 开销。
  
  Baseline 全栈执行例子（vanilla Pythia 302M, L=2048, B=16, prefill-heavy scenario）：
  - 算法pipeline：prompt token ids → Embedding → 24 层 transformer decoder：每层 Masked MHA（QKV 投影 → QK^T/√d → causal mask → softmax → ×V → output proj）→ FFN（gate+up → SiLU → ×down）→ 残差连接。prefill 阶段：所有 2048 tokens 并行前向，缓存 24 层 × 2048 tokens × 2(KV) × 1024 dim × 16 heads × 2 bytes = 3.2GB KV cache。decode 阶段：每步生成 1 token，从 HBM 读取全部 KV cache 和模型参数（302M × 2 bytes = 604MB），受 HBM 带宽限制，batch size 受限，MFU 典型仅 ~1%。
  - 系统框架：HuggingFace Transformers + GPT-NeoX 库，PyTorch eager mode。
  - 编译框架：论文未明确说明（使用标准 PyTorch）。
  - kernel调度：论文未明确说明（标准 cuBLAS GEMM/GEMV kernel）。
  - 硬件架构：8× A100 40GB 训练，1× H100 推理。

  Baseline 的核心缺陷：**KV cache 内存和 IO 是推理吞吐的主要瓶颈**。随着模型规模增大和 context length 增长（百万 token 级别趋势），KV cache 开销愈发严重，超越了参数 IO，成为 batch inference 吞吐量的硬上限。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出**Block Transformer**，通过分层全局到局部（global-to-local）建模，将标准自注意力的两个功能（全局上下文理解 vs 局部细节建模）分解到两个独立 decoder 中：
  
  **(1) Block Decoder 解决预填充和全局 KV cache 开销问题**：
  Baseline 的每一层都对全部 L=2048 tokens 做全局 attention，KV cache 存储、预填充计算和 decode IO 均随 L 线性或二次增长。Block Decoder 在块级别（L/LB=512 blocks）而非 token 级别做全局自注意力，将上下文长度降至 1/LB。这使得：(i) 预填充计算量降低 LB 倍；(ii) KV cache 存储量降低 LB 倍；(iii) KV cache IO 降低 LB² 倍（因 IO ∝ 上下文长度²）；(iv) 每 LB 个 token 仅执行一次前向（而非每 token 一次），参数 IO 降低 LB 倍。
  
  **(2) Token Decoder 解决 decode 阶段 KV cache IO 瓶颈**：
  Baseline 中上层 decoder 的 KV cache 存储和 IO 随整个序列长度 L 二次增长。Token Decoder 将全局上下文压缩为单个 context embedding（由 Block Decoder 输出），仅对当前块内 LB 个 token 做局部注意力（跨所有 token decoder 层）。这使得：(i) KV cache 存储从 O(L) 降至 O(LB)，对 L=2048, LB=4 降低 R=L/LB=256 倍；(ii) 预填充可完全跳过（除最后一个块外），大幅降低首 token 延迟；(iii) KV cache IO 从 O(L²) 降至 O(L·LB)，即线性复杂度，从根本上解决了长上下文场景的 KV cache IO 瓶颈。prefix token 机制允许通过增加 prefix 长度扩展 token decoder 的"计算宽度"，在几乎不影响推理吞吐的前提下提升性能（因推理受 memory-bound 而非 compute-bound 限制）。
  
  **(3) Embedder 简化全局到局部信息传递**：
  Baseline 无此组件。Embedder 通过 lookup table 将每 LB 个 subword token 拼接为一个 block embedding，为 Block Decoder 提供粗粒度输入。简单 lookup 策略（vs 小型 RoBERTa encoder）既高效又不损失性能。
  
  **(4) 1:1 参数分配比 + prefix token 设计逆转了先前工作（MEGABYTE）的结论**：
  MEGABYTE 认为全局模块（block decoder）应占 6 倍于局部模块的参数，且局部模块可很小。Block Transformer 通过参数分配消融证明：(i) 1:1 的 block:token decoder 参数比在固定总参数约束下达到最优 perplexity（U-shaped trade-off）；(ii) 更大的 token decoder 可在稍有性能妥协下显著提升吞吐量（因局部 KV cache 极小）；(iii) prefix token 机制赋予 token decoder 进一步处理上下文的能力（类似 pause tokens），这在先前工作中被完全忽视。

  论文方法全栈执行例子（Block Transformer 302M, LB=4, prefix=2, L=2048, prefill-heavy scenario）：
  - 算法pipeline：prompt token ids → **Embedder**：lookup table 将每 4 token 的 D/4-dim embedding 拼接为 D-dim input block embedding（共 512 blocks）→ **Block Decoder（12层）**：对 512 blocks 做全局 causal self-attention → 输出 context embedding [B, D] → 投影为 2 个 prefix tokens [B, 2, D] → **Token Decoder（12层）**：每步将 prefix + 当前块 4 token embedding [B, 6, D] 输入 → 局部 causal attention（仅 6 token）→ FFN → classifier 输出下一个块 4 token 的 logits → 重复 L/LB=512 次生成完整序列。prefill 阶段：Block Decoder 预填充 512 个 block（vs baseline 2048 tokens，4× 降低），Token Decoder 仅预填充最后一个块。decode 阶段：Block Decoder 每 LB=4 token 仅执行一次前向（参数 IO 降低 4×），Token Decoder 虽每 token 执行一次前向但 KV cache 仅 6 token（vs baseline 2048），KV cache IO 降低 256×。总吞吐量：prefill-heavy (2048/128) 吞吐量 21.0K tok/s vs vanilla 0.8K tok/s（~26×），decode-heavy (128/2048) 吞吐量 44.1K tok/s vs vanilla 2.1K tok/s（~21×）。PPL: LAMBADA=29.5 vs vanilla=10.0, WikiText=27.7 vs vanilla=20.1, HellaSwag=31.13 vs vanilla=35.05。batch size 约 6× vanilla（KV cache 节省 105.0MB/sample vs 1140.0MB/sample）。
  - 系统框架：HuggingFace Transformers + GPT-NeoX 库 + DeepSpeed ZeRO，PyTorch mixed precision training。
  - 编译框架：论文未明确说明。
  - kernel调度：论文验证 FlashAttention 应用于 Block Decoder 的全局 attention 可进一步提升吞吐（最高 31%），但整体趋势不变（附录 I）。其他 kernel 论文未明确说明。
  - 硬件架构：8× A100 40GB 训练，1× H100 推理。Uptraining 策略：从预训练 vanilla transformer 分割层初始化 block/token decoder，仅需 10% 训练数据即可接近全量训练性能，提供了从现有模型迁移的低成本路径。
