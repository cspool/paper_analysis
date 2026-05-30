## FlatQuant: Flatness Matters for LLM Quantization

- **属于kernel调度/运行时计算的实现是什么？实验比较什么？**
  设计了融合 kernel，将仿射变换 Q(P₁^T ×₁ X̃ ×₂ P₂) 在单个 OpenAI Triton kernel 中完成，避免中间结果写回全局内存。kernel 设计动机：仿射变换使用 Kronecker 乘积的两个轻量矩阵（如 64×64），计算强度低，属于 memory-bound 操作；量化也是 memory-bound。融合后在单个 thread block 内完成：加载 P₁、P₂ 到 SRAM → slicing tile block X̄ ∈ R^{n₁×n₂} → 计算 P₁ X̄ P₂ → 即时量化 → 写回全局内存。实验对比了融合前后的 prefill/decoding latency 加速比（不同 hidden dimension 4096/5120/8192/11008/13824/14336，batch size 1-64）。同时集成了 CUTLASS INT4 matmul kernel 和 FlashInfer KV cache quantization。

  三种 SRAM 容量场景的 kernel 设计：
  - **默认设计**（图 8a）：共享内存足够容纳 P₁、P₂、X̄ 及其中间结果，单 kernel 完成所有操作
  - **Corner Case 1**（图 8b）：n 和 n₁ 过大，对 P₁ 非规约维度 tiling，分两个 kernel（先 P₁X̄P₂，再独立量化 kernel）
  - **Corner Case 2**（图 8c）：n 和 n₂ 极大，分三步（先 P₁^T X̄ 写回全局内存释放 SRAM，再乘以 P₂ 并即时量化）

- **后端平台是什么，配置是什么。**
  NVIDIA RTX 3090 GPU。SRAM 大小决定 kernel 设计策略，hidden_dim ≤ 14336 且 n₁,n₂ ≤ 128 时使用默认设计。评测覆盖 hidden dimensions：4096（LLaMA-2-7B）、5120、8192（LLaMA-3-8B）、11008（LLaMA-2-7B FFN intermediate）、13824、14336。Corner case 测试使用 hidden_dim=28762。

- **评估性能的软件/脚本是什么。修改了什么。**
  评估软件：OpenAI Triton（编写融合 kernel）、CUTLASS（INT4 矩阵乘法）、FlashInfer（KV cache 量化）、PyTorch（baseline 对比）。修改内容：
  1. **融合 kernel 实现**：将原本分离的 3 个操作（加载 → 仿射变换 → 写回 → 加载 → 量化 → 写回）融合为单 kernel 内流水线
  2. **SRAM tiling 策略**：根据 P₁(n₁×n₁)、X̄(n₁×n₂)、P₂(n₂×n₂) 的总 FP16 字节数（×2）判断是否超过 shared memory per block（m），自动选择默认/Corner Case 1/Corner Case 2 路径

- **开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。**
  开源代码：https://github.com/ruikangliu/FlatQuant

  **Kernel 执行全过程（以默认设计，hidden_dim=4096, n₁=n₂=64, RTX 3090, batch=1, prefill）**：

  **输入**：激活 X ∈ R^{k×4096}（k=tokens），P₁ ∈ R^{64×64}，P₂ ∈ R^{64×64}（FP16），量化参数（scale, zero-point）

  **Kernel Launch 配置**：Grid = (k, 1, 1)，每个 thread block 处理一个 token（X 的一个 64×64 tile）

  **Thread Block 内执行流程**：
  ```
  // Step 1: 加载到 SRAM
  P₁_sram ← load(P₁)                    // 64×64×2B = 8KB
  P₂_sram ← load(P₂)                    // 64×64×2B = 8KB
  X̄_sram ← load(X[token_i])             // 64×64×2B = 8KB (tile from X̃)

  // Step 2: 仿射变换 (在 SRAM 内)
  X̄' = P₁_sram^T @ X̄_sram @ P₂_sram     // 64×64 matmul × 2

  // Step 3: 即时量化 (在 SRAM 内)
  scale = max(|X̄'|) / (2^{b-1} - 1)     // per-token symmetric
  X̄'_q = round(X̄' / scale)              // quantize to INT4
  X̄'_q = clamp(X̄'_q, -2^{b-1}+1, 2^{b-1}-1)

  // Step 4: 写回全局内存
  store(X̄'_q, scale) → global memory    // INT4 + FP16 scale
  ```

  **Profiling 结果**（Table 6, hidden_dim=4096, batch=1, seq_len=2048 prefill / decode_1token）：
  - 无融合：prefill 0.1956ms, decode 0.0184ms
  - 有融合：prefill 0.0625ms, decode 0.0082ms
  - 加速比：prefill 3.13×, decode 2.25×

  **端到端加速**（LLaMA-2-7B, batch=64, prefill 2048 tokens + decode 256 tokens）：
  - vs FP16：prefill 2.30×, decode 1.76×
  - 与纯 INT4 量化（无变换）相比，仅损失 0.07× 加速比
