## Cooperative Softmax with Cross-Warp Shared Memory Reduction

术语是什么？通过联网搜索让回答具体和精准。

Cooperative Softmax 是 BitDecoding 提出的跨 warp softmax 实现，用于配合 Wn>1 的 warp layout。FlashAttention 原始实现中 softmax 在两轮迭代间完全在 register 内完成（online softmax with running max/sum），依赖所有 KV tile 被同一 warp 串行处理。当 Wn>1（多 warp 并行处理不同 KV segments），每个 warp 仅持有部分 P 矩阵，register-level softmax 不再可行。BitDecoding 引入两个 shared memory buffer：sTMP ∈ R^{Wn}（跨 warp reduction 计算 row-wise max）和 sAcc ∈ R^{Tm×Tn}（暂存 P 矩阵用于后续 reload）。流程：各 warp 独立完成 QK^T→intra-warp register reduction for rowmax→store local max to sTMP→cross-warp shared memory reduction (parallel reduction tree)→broadcast global max→各 warp 计算 exp(P - global_max)→store P to sAcc (r2s)→reload P via ldmatrix from sAcc (s2r)→PV mma。Shared memory overhead 极小（Wn 通常 ≤8，sAcc 复用 sTMP pointer），仅引入 0.5% latency overhead（Table III）。

从 kernel 调度角度拆解术语：

```
// === Cooperative Softmax Algorithm (Algorithm 1 in paper) ===
// 输入: Qi ∈ RTm×d, Ki/Vi ∈ RTn×d (in REG)
// Shared Memory: sTMP ∈ RWn, sAcc ∈ RTm×Tn

// Step 1: QK^T mma (Tensor Cores)
Si = Qi × Kj^T;  // Si ∈ RTm×Tn, in TC registers

// Step 2: Cross-warp max reduction
// 2a: Intra-warp max (register-level shuffle reduction)
local_max = warp_reduce_max(Si);  // __shfl_xor_sync reduction

// 2b: Inter-warp max (shared memory)
if (lane_id == 0):
    sTMP[warp_id] = local_max;    // 每个warp写入shared mem
__syncwarp();
global_max = shared_mem_parallel_reduce(sTMP, Wn);  // log2(Wn)步

// Step 3: Online softmax update
mnew = max(m_old, global_max);
Pi = exp(Si - mnew);  // Pi ∈ RTm×Tn

// Step 4: Store P to shared memory (r2s)
sAcc[tile_row][tile_col] = Pi;  // tiled copy, register→shared

// Step 5: Reload P via ldmatrix (s2r) for proper TC alignment
ldmatrix.sync.aligned.m16n8k16.shared.b16 [...], [sAcc];

// Step 6: PV mma (Tensor Cores), using reloaded P
Onew = Pi_reloaded × Vj + diag(exp(m_old - mnew)) × O_old;
```

关键设计：(i) sAcc 复用 sTMP 的 shared memory 指针以最小化 memory overhead；(ii) s2r 通过 ldmatrix 重载 P 确保后续 mma 需要的 interleaved layout；(iii) Hopper 上 sAcc 可直接被 wgmma_SS 访问，省去 s2r step。

术语一般如何实现？如何使用？

Cooperative softmax 实现在 BitDecoding 的 Packing Kernel 中，约 200 行 CUDA PTX。Wn 的典型取值为 4 或 8（受限于 shared memory size 和 SM warp 数）。性能 trade-off：增加 Wn 提升 parallelism 但增加 cross-warp reduction overhead（O(log Wn) shared memory accesses）；paper 表明 Wn=4 在 A100 上接近最优。对于 prefill（Q_len 大），可回退到 register-level softmax（Wn=1 的 FlashAttention 模式）。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

