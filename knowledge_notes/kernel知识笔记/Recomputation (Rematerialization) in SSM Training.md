## Recomputation (Rematerialization) in SSM Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Recomputation（梯度重计算/rematerialization）是内存优化技术：前向不保存大中间激活，反向时重新计算。在 Mamba 中，中间状态 h ∈ R^{B,L,D,N}（N=16 比输入大 16 倍）若保存到 HBM 内存开销巨大。解决：前向仅保存 O(BLD) 输入 → 反向重新加载到 SRAM → 重计算 h → 计算梯度。因输入+梯度 O(BLD) 远小于 h O(BLDN)，总 HBM IO 反而更少（memory-bandwidth 是瓶颈）。总效果：每个 selective SSM 层 ≈ 16 bytes/token 激活内存 vs Transformer (FlashAttention+MLP) ≈ 32 bytes/token。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Forward: don't save h (O(BLDN)), save only inputs (O(BLD))
fused_scan_forward(x, Δ, A, B, C) → y

// Backward: reload inputs, recompute h, compute gradients
fused_scan_backward(dL/dy, x, Δ, A, B, C):
    // 1. Load x, Δ, A, B, C from HBM → SRAM  (O(BLD) read)
    // 2. Recompute h in SRAM:
    //    discretize(Δ, A, B) → Ā, B̄
    //    parallel_scan(Ā, B̄⊙x) → h
    // 3. Compute gradients using h and dL/dy:
    //    dL/dC = dL/dy ⊙ h  → scan backprop → dL/dΔ, dL/dB
    // 4. Write gradients to HBM  (O(BLD) write)

IO对比:
  重计算: Read O(BLD) + Write O(BLD) = O(2BLD)
  保存: Read h O(BLDN) + Read grad O(BLD) = O(BLD(N+1))
  N=16: 重计算 IO ≈ 2BLD vs 保存 IO ≈ 17BLD → 8.5× 节约
```
该技术与 FlashAttention 的重计算策略一致（不保存中间 attention matrix，反向重计算 softmax）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源: https://github.com/state-spaces/mamba (fused scan kernel 内实现)。Mamba 还将重计算扩展到整个 SSM block：不保存 activation 输出和 short convolution 中间结果，需要时快速重计算。适用条件：重计算开销 < 额外 HBM IO 开销，对 memory-bandwidth-bound 操作（scan, attention）通常成立。

涉及论文标题：
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
