## Fused Selective Scan Kernel (Hardware-Aware SSM Scan)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused Selective Scan 是 Mamba (Gu & Dao, 2023) 为选择性 SSM (S6) 设计的硬件感知 CUDA kernel，通过 kernel fusion + parallel scan + recomputation 使 time-varying SSM 在现代 GPU 上高效运行。核心问题：选择性 SSM 不再是 LTI（线性时不变），无法用卷积（FFT）模式；朴素循环需物化 (B,L,D,N) 中间状态 h（N=16 时比输入大 16 倍），HBM IO 量巨大。该 kernel 将离散化、parallel scan、输出乘加融合在 GPU SRAM 内，仅将 O(BLD) 最终输出写回 HBM，IO 减少约 N 倍。在 A100 GPU 上，该 kernel 在序列长度 >2K 后超越 FlashAttention-2，32K 时快约 7×；vs PyTorch naive scan 快 20–40×。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Fused Selective Scan Kernel (per chunk, in SRAM):

// Phase 1: Load from HBM to SRAM
Δ [B,L,D], A [D,N], B [B,L,N], C [B,L,N], x [B,L,D]

// Phase 2: Discretize in SRAM (per timestep, fused with scan)
for t in 0..L-1:
    Ā_t = exp(Δ_t ⊙ A)     // (D,N), element-wise vector*scalar exp
    B̄_t = Δ_t ⊙ B_t        // (D,N), 一阶近似: (exp(ΔA)-I)/(ΔA) ≈ I for small Δ

// Phase 3: Parallel Associative Scan (Blelloch) in SRAM
// element = (a: R^{N}, b: R^{N})
// binop: (a,b) ⊕ (a',b') = (a'⊙a, a'⊙b + b')
// Up-sweep → Down-sweep → outputs h_{0..L-1} in O(log L) parallel steps

// Phase 4: Output multiply in SRAM
y_t = C_t ⊙ h_t for t in 0..L-1  // (D,)

// Phase 5: Write y [B,L,D] to HBM (ONLY this goes to HBM)

HBM IO: Read O(BLD) + Write O(BLD) = O(2BLD)
Naive IO: Read O(3BLDN) + Write O(BLDN) + const = O(4BLDN)
```
反向传播采用重计算：不保存 h [B,L,D,N] → backward 重新加载 O(BLD) 输入到 SRAM → 重计算 h → 计算梯度 → 写回 O(BLD)。总 backward IO = O(BLD) vs 保存方案 O(BLDN)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源: https://github.com/state-spaces/mamba (CUDA C++)。利用 GPU memory hierarchy: HBM→L2→SRAM→register。长序列分 chunk 处理，chunk 间通过 HBM 传递 scan state 连接。Mamba-125M activation memory ≈ 4.8-38.2GB (batch 1-32)，与 FlashAttention-2 Transformer (4.6-34.5GB) 可比。

涉及论文标题：
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
