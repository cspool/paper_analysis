## AdaSplash: Adaptive Sparse Flash Attention

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现了一套基于 Triton 语言的自定义 GPU kernel，用于高效计算 α-entmax 稀疏注意力。包括：(1) 前向 kernel：block-wise tiling 避免 materialize 完整的 S ∈ R^{n×n} 和 P ∈ R^{n×n}，recomputation 策略以 FLOPs 换 memory，Halley-bisection block version 在 SRAM 中累积 f, f', f'' 不写出 S；(2) 反向 kernel：分 dK/dV kernel 和 dQ kernel 两个独立 kernel，利用 α-entmax 稀疏 Jacobian (U_i^{(j)} = P_i^{(j)}^{2-α}) 计算 dS = U ⊙ (dP - δ)；(3) 稀疏调度优化：前向 pass 最后迭代中动态构建 block mask M ∈ {0,1}^{T_r×T_c}，通过 pointer-increment lookup tables (K_j, Q_i) 跳过 null blocks 的 HBM 读写。实验比较：(1) ADASPLASH Triton kernel vs. FlashAttention-2 (CUDA + Triton) 在 synthetic data 上的 runtime vs. input sparsity 关系；(2) 不同序列长度 (512-8192) 和不同注意力实现 (Torch sorting, Torch bisection, Halley-bisection Triton, ADASPLASH Triton) 的 ModernBERT training step 时间与内存。

- 后端平台是什么，配置是什么。
  - Efficiency benchmark (synthetic data, Figures 1, 3) 和 GPT-2 训练：单张 Nvidia H100 GPU (80GB HBM, large SRAM)
  - ModernBERT runtime 和下游任务训练：Nvidia RTX A6000 GPU (48GB VRAM)
  - 所有 kernel 均用 Triton 语言编写（https://github.com/triton-lang/triton），利用 Triton 的 block-level programming 模型实现对 GPU SRAM 和 HBM 的精细控制

- 评估性能的软件/脚本是什么。修改了什么。
  基于 PyTorch (torch.compile 未使用，因 attention 太复杂需手动优化) + Triton 语言 + HuggingFace Transformers 框架。核心 Triton kernel 修改/实现：
  
  1. **Halley-bisection block version kernel（Algorithm 3）**：
  - 输入：Q_1..T_r, K_1..T_c 分块，HBM 中
  - 过程：对每 Q_i block，循环 T_c K_j blocks，在 SRAM 中计算 S_i^{(j)} = Q_i K_j^T，但不写出；直接累积 f(τ), f'(τ), f''(τ) 的 block 部分贡献；用 Halley-bisection 在 M 次迭代后输出 τ_i
  - 关键优化：block-accumulated f/f'/f'' 避免了 materialize 完整的 S 矩阵，仅需 O(B_r × B_c) SRAM workspace
  
  2. **ADASPLASH forward kernel（Algorithm 2）**：
  - block sizes: B_c（column block size for K,V）, B_r（row block size for Q）
  - 每 Q_i block：先调 Halley-bisection block kernel 得 τ_i；再 loop K_j/V_j blocks，在 SRAM 中 re-compute S_i^{(j)}，计算 P_i^{(j)} = [(α-1)S_i^{(j)} - τ_i]_+^{1/(α-1)}；累积 O_i += P_i^{(j)} V_j
  - 比 FlashAttention-2 多 2 次 K 加载（用于 τ 计算），故前向 pass 永远稍慢于 FA2
  
  3. **Block Mask 生成与 Lookup Table 构造**：
  - 在 Halley-bisection 最后迭代中：对每个 (i,j) block pair，check any(S_i^{(j)} > τ_i)，存为 binary M_{ij}
  - 用 torch.argwhere 提取 M_{ij}=1 的 (i,j) 索引 → K_j, Q_i lookup tables
  - M 为二进制值且跨 attention 层可共享，内存开销 O(T_r×T_c) 远小于 P ∈ R^{n×n}
  
  4. **ADASPLASH backward dK/dV kernel（Algorithm 4）**：
  - 外层 loop j=1..T_c（K_j/V_j 粒度），内层 loop i ∈ K_j（仅有效 Q_i）
  - 在 SRAM 中计算：S_i^{(j)} → P_i^{(j)} → U_i^{(j)} (= P_i^{(j)}^{2-α}) → dS_i^{(j)} = U_i^{(j)} ⊙ (dP_i^{(j)} - δ_i)（利用 α-entmax 稀疏 Jacobian）
  - dV_j += (P_i^{(j)})^T dO_i, dK_j += (dS_i^{(j)})^T Q_i
  - Block masking 跳过 M_{ij}=0 的 block，大幅减少无效 HBM 访问
  
  5. **ADASPLASH backward dQ kernel（Algorithm 5）**：
  - 外层 loop i=1..T_r（Q_i 粒度），内层 loop j ∈ Q_i（仅有效 K_j）
  - dQ_i += dS_i^{(j)} K_j
  - 与 dK/dV kernel 分离，允许独立的并行化和 block masking

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  代码开源：https://github.com/deep-spin/adasplash（ICML 2025）

  **评估原理与 Kernel 执行全流程（以一个 attention head 的前向+反向 pass 为例）**：

  ```
  输入：Q, K, V ∈ R^{n×d}（HBM 中），block sizes B_r, B_c，参数 α, iterations T
  输出：O ∈ R^{n×d}（前向），dQ, dK, dV ∈ R^{n×d}（反向）

  === 前向 Pass ===
  Step 1: 分块
    将 Q 分为 T_r = ⌈n/B_r⌉ blocks，K,V 分为 T_c = ⌈n/B_c⌉ blocks
    分配 τ ∈ R^n 的 T_r blocks（HBM 中）
  
  Step 2: τ 计算（Halley-Bisection Block Kernel，Algorithm 3）
    for i = 1..T_r:
        Load Q_i (B_r × d) from HBM → SRAM
        Initialize τ_i, τ_lo_i, τ_hi_i
        for iter = 1..T:  // T=3 即可达到 machine precision
            f, f', f'' = 0 (in SRAM, shape B_r)
            for j = 1..T_c:
                Load K_j (B_c × d) from HBM → SRAM
                Compute S_i^{(j)} = Q_i @ K_j^T on SRAM  // B_r × B_c
                Accumulate f, f', f'' using block contributions  // Equations 3,6,7
                // S_i^{(j)} NOT written to HBM (recomputed later)
            // Store binary mask M_{ij} if last iteration
            M_{ij} = 1 if any(S_i^{(j)} > τ_i) else 0
            // Update τ_i via Halley-bisection (Algorithm 1)
            if Halley update within [τ_lo, τ_hi]:
                τ_i = Halley_update
            else:
                τ_i = bisection_update
        Write τ_i to HBM
  
  Step 3: 构造 Lookup Tables
    M_full = M (T_r × T_c, bool, on GPU)
    indices = torch.argwhere(M)  // (i,j) pairs where M_{ij}=1
    K_j lookup: per-column j, row indices i where M_{ij}=1
    Q_i lookup: per-row i, column indices j where M_{ij}=1
  
  Step 4: 前向 Attention 计算（Algorithm 2 + Block Masking）
    for i = 1..T_r:
        Load Q_i, τ_i from HBM → SRAM
        Initialize O_i = 0 (B_r × d, on SRAM)
        for j in Q_i:  // only non-null blocks!
            Load K_j, V_j from HBM → SRAM
            Recompute S_i^{(j)} = Q_i @ K_j^T  // re-computation (not stored)
            Compute P_i^{(j)} = [(α-1)S_i^{(j)} - τ_i]_+^{1/(α-1)}
            O_i += P_i^{(j)} @ V_j  // accumulate in SRAM
        Write O_i to HBM

  === 反向 Pass ===
  Step 5: δ 计算（Separate Kernel, Equation 25）
    需要前向存下的 O^{(2)} ∈ R^{n×d}（存储增量 vs softmax 的 O）
    for i = 1..T_r:
        δ_i = dO_i @ O_i^{(2)} / ||U_i||_1  // per-row normalization
  
  Step 6: dK, dV Kernel（Algorithm 4 + Block Masking）
    for j = 1..T_c:
        Load K_j, V_j from HBM → SRAM
        Initialize dK_j, dV_j = 0 (on SRAM)
        for i in K_j:  // only Q_i rows that contribute
            Load Q_i, dO_i, τ_i, δ_i from HBM → SRAM
            Recompute S_i^{(j)}, P_i^{(j)}, U_i^{(j)}
            dP_i = dO_i @ V_j^T
            dS_i^{(j)} = U_i^{(j)} ⊙ (dP_i - δ_i)  // sparse Jacobian product
            dK_j += (dS_i^{(j)})^T @ Q_i
            dV_j += (P_i^{(j)})^T @ dO_i
        Write dK_j, dV_j to HBM
  
  Step 7: dQ Kernel（Algorithm 5 + Block Masking）
    for i = 1..T_r:
        Load Q_i, dO_i, δ_i, τ_i from HBM → SRAM
        Initialize dQ_i = 0 (on SRAM)
        for j in Q_i:  // only non-null blocks
            Load K_j, V_j from HBM → SRAM
            Recompute S_i^{(j)}, P_i^{(j)}, U_i^{(j)}
            dP_i = dO_i @ V_j^T
            dS_i^{(j)} = U_i^{(j)} ⊙ (dP_i - δ_i)
            dQ_i += dS_i^{(j)} @ K_j
        Write dQ_i to HBM
  ```

  **额外内存开销（vs FlashAttention-2）**：
  - O^{(2)} ∈ R^{n×d}：取代 softmax 中存 O 的需求，实际与 FA2 的 O 等大
  - M ∈ {0,1}^{T_r×T_c}：二进制 block mask，可跨 attention 层共享
  - τ ∈ R^n：每行一个标量阈值，O(n) 额外开销

  **评估指标**：
  - 平均 training step 时间（前向+反向）（单位：s 或 ms/step）
  - Peak GPU memory usage（GB）
  - 50 steps 的平均（warmup 后）

  **ModernBERT-base runtime 对比（Table 5, A6000 GPU）**：
  | 算法 | 512 | 1024 | 2048 | 4096 | 8192 |
  |------|-----|------|------|------|------|
  | Sorting (Torch) | 0.09s | 0.11s | 0.26s | 0.76s | OOM |
  | Bisection (Torch) | 0.11s | 0.15s | 0.42s | 1.35s | 4.99s |
  | Halley-bisection (Triton) | 0.10s | 0.11s | 0.26s | 0.46s | 1.61s |
  | **ADASPLASH (Triton)** | 0.10s | 0.12s | 0.21s | 0.48s | **1.53s** |

  **GPT-2 training step（Table 4, H100 GPU, 1024 ctx）**：
  - FlashAttention-2 (softmax): 0.98 s/step, 52.5 GB
  - ADASPLASH (α=1.5): 1.03 s/step, 52.5 GB
  - Torch sorting (α=1.5): 3.61 s/step, 73.8 GB
  - Torch bisection (α=1.5): 7.78 s/step, 77.6 GB
