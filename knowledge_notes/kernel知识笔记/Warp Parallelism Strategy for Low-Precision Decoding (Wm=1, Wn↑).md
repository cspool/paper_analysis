## Warp Parallelism Strategy for Low-Precision Decoding (Wm=1, Wn↑)

术语是什么？通过联网搜索让回答具体和精准。

Warp Parallelism Strategy for Low-Precision Decoding 是 BitDecoding 为解决低比特 dequantization 导致 warp stall 而提出的 warp 分配策略。核心思想：在 decode 阶段 Q length=1（极小的 M 维度），将 M 维度的 warp 数压缩到最小(Wm=1)，将释放的 warp 资源重新分配到 N 维度(Wn↑)。这样做的好处——多 warp 在 N 维度上并行处理 K/V 的不同 segment，每个 warp 的 dequantization（CUDA Cores）与 Tensor Cores mma 可以被 SM warp scheduler 自然地 overlap：当 warp_i 在 Tensor Cores 上执行 mma 时，warp_{i+1} 同时在 CUDA Cores 上做 dequantization。在 FlashAttention 原始 warp layout 下（Wn=1, Wm 较大），单个 warp 沿 N 维串行处理所有 tile，每次 dequant 都 stall 该 warp → Tensor Cores utilization 仅 10.91%（Table III）。BitDecoding 将 Wn 增至 4 后 TC utilization 提升至 19.66%。

从 kernel 调度角度拆解术语：

```
// === Warp Layout对比 ===
// 
// FlashAttention原始layout (register-level softmax):
//   Grid: Wm × Wn, 其中Wm>1 (沿M分割Q rows), Wn=1
//   Decode时Q_len=1 → M维度极小 → Wm个warp均严重underutilized
//   单个warp沿N串行: for each K_tile → dequant → mma → dequant stall
//
// BitDecoding layout:
//   Grid: Wm=1 × Wn (Wn≥4)
//   每个warp负责一段K/V: warp_0→K[0:Tn], warp_1→K[Tn:2Tn], ...
//   SM warp scheduler自动overlap:
//     cycle 0: warp_0发射mma, warp_1开始dequant
//     cycle 1: warp_0 still in mma, warp_1完成dequant→发射mma, warp_2开始dequant
//     ...
//   No warp idling — dequant latency hidden by parallel warp execution

// Dequantization stall消除原理:
// Original (Wn=1): Time = T_dequant + T_mma (串行)
// BitDecoding (Wn=4): Time ≈ max(T_dequant, T_mma) + T_dequant/Wn (parallel)
// 当T_mma >> T_dequant/Wn时dequant几乎零overhead
```

术语一般如何实现？如何使用？

在 CUDA kernel 中通过调整 grid/block 维度和 warp 分配实现：`blockDim = (32, Wn)` 或等效逻辑，query 沿 M 维度不拆分（Wm=1）。需要配套的 cooperative softmax（以 shared memory 替代 register-level softmax）处理 P 矩阵的跨 warp 聚合。此策略尤其适合 decode 场景（Q_len 极小，M 维度 compute 压力低），prefill 阶段（Q_len 大）不适用——prefill 下 M 维度本身可填满多个 warp。Paper Table III 验证：Wn=1→4 时 latency 从 3.746ms 降至 0.613ms（6.1× 改善），TCs utilization 从 10.91%升至 19.66%。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

