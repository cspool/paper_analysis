## Mamba: Linear-Time Sequence Modeling with Selective State Spaces

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Mamba 设计了**硬件感知的选择性扫描（Hardware-Aware Selective Scan）**内核，用 kernel fusion + 并行扫描（parallel scan）+ 重计算（recomputation）三种经典技术使选择性 SSM 在现代 GPU 上高效运行。核心问题是：选择性 SSM 不再是 time-invariant，无法使用卷积模式（FFT）高效计算，而朴素循环模式下需要物化 (materialize) 大小为 (B, L, D, N) 的中间状态 h（N≈16 时比输入 x 大 N 倍），导致大量 HBM 读写。Mamba 的解法是：
    i) **Kernel Fusion**：将离散化（discretization）、并行扫描、与 C 的乘加融合进单个 CUDA kernel。直接从慢速 HBM 加载 O(BLD) 的 (Δ, A, B, C) 参数到快速 SRAM，在 SRAM 内完成离散化得到 (Ā, B̄) → 并行扫描计算 h → 乘以 C 得到 y，仅将最终 O(BLD) 的输出写回 HBM。减少 IO 量约 O(N) 倍
    ii) **Parallel Scan**：使用 Blelloch 工作高效的并行关联扫描算法（work-efficient parallel associative scan），将序列循环转化为 O(log L) 次并行操作
    iii) **Recomputation**：前向时不保存中间状态 h (B, L, D, N)，反向传播时重新加载输入从 HBM 到 SRAM 并重计算 h。由于输入/输出/梯度之和 O(BLD) 远小于 h 的 O(BLDN)，重计算比存储并读取 h 更快
  - 实验比较：
    - 扫描速度：Mamba fused scan vs 标准 PyTorch scan（完整物化 Ā, B̄, C 于 HBM）vs FlashAttention-2 vs 卷积（FFT）。测量序列长度 512–500K，batch size=1, D=1024, N=16, BF16
    - 端到端推理吞吐：Mamba-1.4B/6.9B vs Transformer-1.3B/6.7B（HuggingFace 实现），prompt 长度 2048，生成长度 128，batch size 1–128
    - 训练内存：Mamba-125M vs Transformer-125M（w/ FlashAttention-2 + torch.compile），batch size 1–32，序列长度 2048

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA A100 80GB PCIe GPU
  - 计算后端：CUDA（自定义 fused scan kernel），PyTorch（标准 scan baseline + Transformer 端到端推理）
  - 精度：BF16（训练/推理主流配置）

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件：自定义 CUDA scan kernel + PyTorch benchmark 脚本
  - 修改内容：在选择性 SSM 层中，将原本需先物化 (Ā, B̄, C) ∈ R^{B,L,D,N} 到 HBM → 调用 PyTorch scan → 乘 C → 写回 HBM 的标准流程，替换为单一 fused kernel
  - 对比的 baseline：
    - 标准 PyTorch scan：使用 `torch.cumsum` 或手工并行 scan 实现，完整物化中间状态
    - 卷积（FFT）：PyTorch 的 FFT-based convolution，O(L log L) FLOPs 但 LTI 限制
    - FlashAttention-2 (Dao 2024)：带 causal mask，当前最快的 attention kernel

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源地址：https://github.com/state-spaces/mamba
  - **Fused Selective Scan Kernel 执行流程**：

  ```
  输入（从 HBM 加载到 SRAM）:
    Δ ∈ R^{B,L,D}  (step size)
    A ∈ R^{D,N}    (diagonal state matrix, 可复用)
    B ∈ R^{B,L,N}  (input projection)
    C ∈ R^{B,L,N}  (output projection)
    x ∈ R^{B,L,D}  (optional, 若未在上游 projection 融合)

  SRAM 内计算:
    1. 离散化 (Discretization) — 对每个 timestep t:
       Ā_t = exp(Δ_t ⊙ A)       → R^{D,N}, element-wise exp
       B̄_t = Δ_t ⊙ B_t          → R^{D,N}, 一阶 Taylor 近似
       (注意: ⊙ 表示沿 N 维度的 broadcast element-wise 乘法)

    2. 并行关联扫描 (Parallel Associative Scan) — Blelloch 算法:
       // 输入: 序列 {(Ā_t, B̄_tᐧx_t)} for t=1..L
       // 关联操作: (a, b) ∘ (a', b') = (a'⊙a, a'⊙b + b')
       // h_t 为扫描输出: h_t = Ā_t⊙h_{t-1} + B̄_t⊙x_t

       Up-sweep (reduce phase): O(log L) parallel steps
         for d = 0 .. log₂(L)-1:
           for k = 0 .. L/2^{d+1}-1 in parallel:
             combine elements at indices 2^d·(2k+1)-1 and 2^d·(2k+2)-1

       Down-sweep (distribution phase): O(log L) parallel steps
         将中间结果传播, 输出完整 h_{0..L-1}

    3. 输出乘加:
       y_t = C_t ⊙ h_t  → R^{D}, for t=1..L

  输出（写回 HBM）:
    y ∈ R^{B,L,D}  (最终输出，与输入同形状)

  内存 IO 对比:
    标准方法: Read O(BLDN) + Write O(BLDN) + Read O(BLDN) = O(3BLDN) HBM I/O
    Fused方法: Read O(BLD) + Write O(BLD) = O(2BLD) HBM I/O
    加速比 ≈ N (当 N=16 时约 16×)，实测 20–40× 比 PyTorch naive scan
  ```

  - **长序列分块处理**：当序列长度 L 超过 SRAM 容量时，将序列分成 chunks。每个 chunk 内的 fused scan 在 SRAM 执行，通过保存 chunk 间的中间扫描状态（scan state）在 HBM 中连接相邻 chunks

  - **反向传播重计算**：
    ```
    前向: 不保存 h ∈ R^{B,L,D,N}（太大）
    反向:
      1. 从 HBM 加载 Δ, A, B, C, x (O(BLD))
      2. 在 SRAM 中重计算 h（与正向相同计算）
      3. 用 h 和 upstream gradient (从 HBM 加载, O(BLD)) 计算 Δ, A, B, C, x 的梯度
      4. 写回梯度 (O(BLD)) 到 HBM
    总 HBM I/O = O(BLD), 相比于保存/加载 h 的 O(BLDN) 更少
    ```

  - **内存消耗对比**（Table 15）：Mamba-125M ≈ 4.8GB (batch=1) ~ 38.2GB (batch=32) vs Transformer w/ FlashAttention-2 ≈ 4.6GB ~ 34.5GB，处于同一量级。每个 selective SSM 层约 16 bytes/token 激活内存，两层 ≈ 32 bytes/token（等价于 attention+MLP）

  - **关键结果**：
    - Fused scan 在序列长度 >2K 后超越 FlashAttention-2，在 32K 时快约 7×
    - 比 naive PyTorch scan 快 20–40×（所有序列长度）
    - Mamba-6.9B 推理吞吐 > Transformer-1.3B 的 5×（因无 KV cache 可用更大 batch size）
    - 扫描比 FFT 卷积 O(L log L) 在长序列上常数因子优势增大
