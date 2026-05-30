## MOM: Memory-Efficient Offloaded Mini-Sequence Inference for Long Context Language Models

- baseline方法是什么？
  Baseline 是标准 LLM 推理流程（Standard）和现有内存优化方法（Chunked Prefill、KV Cache Offloading alone）。

  **Standard（无优化）**：全量 prefill + 全量 KV cache on GPU。全栈执行例子（Llama-3-8B, S=128K, d=4096, I=4d=16384, L 层）：
  - **算法pipeline**：输入 X → 逐层 Transformer Block → Attention(X) 使用 FlashAttention → MLP(X) 计算 SwiGLU（W_gate, W_up, W_down），中间激活大小 S×I → LM_Head(last_token)。每层 MLP 产生峰值中间内存 S×I = 128K×16384 ≈ 2.1B floats ≈ 4.2GB (bfloat16)。
  - **系统框架**：PyTorch + HuggingFace Transformers 标准推理流程。FlashAttention-2 优化 attention。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：FlashAttention-2 kernel 做 attention。MLP 使用标准 cuBLAS GEMM。
  - **硬件架构**：单张 NVIDIA A100 80GB GPU。Standard 在 155K tokens 时 OOM（超出 80GB 显存）。

  Baseline 的核心缺陷：(1) **MLP 中间激活主导 peak memory**——prefill 阶段 MLP 层的中间激活 S×I（I≈4d）是峰值内存的最大贡献者，远超 attention 优化后的 KV cache 和 attention 计算内存；(2) **Chunked Prefill 重复 forward-pass 开销**——将整个 prefill 切分为多个 chunk 串行处理，每个 chunk 需完整 forward（attention+MLP+LM Head），导致重复 kernel launch 和 extra computation；(3) **Offloading alone 收益有限**——仅 offloading KV cache 不降低 MLP 中间激活，因此 peak memory 减少不明显（因为 MLP 中间激活仍是瓶颈）。

  Chunked Prefill 的全栈执行例子：
  - **算法pipeline**：输入 X 按 chunk_size=C 切分为多个 chunk → 对每个 chunk X^(i) 执行完整 Transformer forward → 累积 KV cache → 所有 chunk 完成后进行 decode。每 chunk 计算 attention+MLP+LM Head 全部子层。
  - **系统框架**：PyTorch + HuggingFace。通过多次 forward 调用模拟分批处理。TensorRT-LLM 中也有实现。
  - **kernel调度**：每次 forward 有独立的 kernel launch overhead，多次 forward 累积开销显著。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MOM 通过两项核心设计解决 baseline 缺陷：

  **1. Mini-Sequence MLP Partitioning → 解决缺陷(1)（MLP 中间激活主导 peak memory）和缺陷(2)（Chunked Prefill 重复开销）**：
  Attention 层保持完整序列处理，仅将 MLP 层输入沿序列维度切分为 M 个 mini-sequences（每个 N=S/M），逐个通过 MLP。由于 attention 层不变，KV cache 的生成和使用不受影响。Mini-sequences 的中间激活从 S×I 降至 (S/M)×I，且均在单次 forward pass 中完成（非多次 forward），无 Chunked Prefill 的重复开销。最后一个 MLP 层和 LM Head 仅处理最后一个 token，进一步减少计算。

  全栈执行例子（MOM, Llama-3-8B, S=128K, C=8192）：
  - **算法pipeline**：
    1. Attention 层：完整处理 S=128K，使用 FlashAttention/GQA 保持不变，生成完整 A ∈ R^{128K×4096}
    2. KV cache：更新后 offload 到 CPU（每层 attention 完成时立即 offload）
    3. 非最后 MLP 层：将 A 按 C=8192 切分为 16 个 mini-sequences A_i ∈ R^{8192×4096}，逐个计算 MLP(A_i)，拼接后传入下一层
    4. 最后 MLP 层：仅取 A_last = A[-1:] ∈ R^{1×4096}，计算 MLP(A_last)
    5. LM_Head(A_last) → logits → 开始 decode
    6. Decode 前：将所有层 KV cache 从 CPU reload 到 GPU
  - **系统框架**：HuggingFace Transformers，使用 OffloadedCache 管理 KV cache 的 CPU/GPU 传输。仅需修改 MLP 层和 LM Head 的输入处理逻辑，attention 和其余组件不变。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：MLP 的 GEMM 操作由原本的 [S, d]×[d, I] 变为 [N, d]×[d, I]（N=S/M），更小的矩阵乘更利于 GPU L2 cache 命中。论文观察到 Mini-sequence 甚至能提升吞吐量（因为 shorter sequence chunks fit better into GPU cache）。
  - **硬件架构**：单 A100 80GB GPU。MOM 将最大 context 从 155K 扩展至 455K（~3×），内存节省 >50%。

  **2. KV Cache Offloading → 解决缺陷(3)（Offloading alone 收益有限）**：
  当 MLP 中间激活被 Mini-sequence 大幅降低后，KV cache 成为剩余内存中的主要占用者。此时 offloading KV cache 到 CPU 变得有意义——因为 MLP 内存不再是瓶颈，offloading 能进一步释放 GPU 内存供更长的序列使用。Offloading 与 Mini-sequence 结合产生协同效应：Mini-sequence 降低 MLP 中间激活 → offloading 降低 KV cache → 两者叠加释放的 GPU 内存远多于各自单独使用。

  **对比 Chunked Prefill 的关键差异**：
  - Chunked Prefill：多次 forward pass，每 chunk 重复 attention + MLP + LM Head → overhead 随 chunk 数增加
  - MOM：单次 forward pass，attention 一次完成，仅 MLP 逐 mini-sequence 循环 → 无重复 forward overhead
  - MOM 比 Chunked Prefill 延长 context 35% more（455K vs Chunked Prefill 的扩展量）
  - MOM + offloading throughput 远优于 Chunked Prefill + offloading（后者数据传输开销 >75%）

  **效果量化**（Llama-3.2-8B, A100 80GB）：
  - Peak memory @155K: Standard 72GB → MOM 35GB（~51% reduction）
  - Max context: Standard 155K → MOM 455K（~3×）
  - Prefill TTFT @144K: Standard 34.9s → Mini-sequence 34.0s（slightly faster），MOM 37.3s
  - Decode speed @144K: Standard 11.63 tok/s → MOM 11.60 tok/s（几乎无退化）
  - Accuracy: Logits identical，Needle test 等同
