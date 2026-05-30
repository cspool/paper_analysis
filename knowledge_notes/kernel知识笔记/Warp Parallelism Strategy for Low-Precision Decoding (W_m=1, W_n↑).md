## Warp Parallelism Strategy for Low-Precision Decoding (W_m=1, W_n↑)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Warp Parallelism Strategy 是 BitDecoding 为解决低比特 dequantization 导致 warp stall 而提出的 warp 分配策略。核心思想：在 decode 阶段 Q length=1（极小 M 维度），将 M 维度的 warp 数压缩到 W_m=1，将释放的 warp 资源重新分配到 N 维度（W_n↑）。多 warp 在 N 维度并行处理 K/V 的不同 segment，SM warp scheduler 自然 overlap 各 warp 的 dequantization（CUDA Cores）与 mma（Tensor Cores）。在 FlashAttention 原始 layout 下（W_n=1），单 warp 沿 N 串行处理所有 tile，dequant 每次都 stall 该 warp → TC utilization 仅 10.91%。BitDecoding 将 W_n 增至 4 后，TC utilization 提升到 19.66%（1.8×），latency 从 3.746ms 降至 0.613ms（6.1×）。

从kernel调度角度拆解术语。

```
// 原始 FlashAttention warp layout (decode, W_m>1, W_n=1)
// 多个 warp 沿 M (seq_len_q=1) → M 维极小 → 大部分 warp 闲置
// 单个 warp 沿 N 串行:
for k_tile in 0..ceil(L/T_n):
    K_pack = load(k_tile) → dequant → mma Q @ K^T → ...
    // dequant stall warp → TC idle during dequant

// BitDecoding warp layout (decode, W_m=1, W_n>1)
// W_n 个 warp 沿 N 并行处理 K/V 的不同 tile segment:
// Warp 0: dequant(tile_0) → mma(tile_0) → dequant(tile_Wn) → mma(tile_Wn) → ...
// Warp 1: dequant(tile_1) → mma(tile_1) → ...
// ...
// SM warp scheduler: when warp_0 doing mma (TC), warp_1 doing dequant (CUDA Cores)
// → CUDA Core dequant 与 TC mma 在 warp 粒度 overlap
```

术语一般如何实现？如何使用？

实现在 BitDecoding Packing Kernel 的 kernel launch configuration。W_n 典型值 4 或 8（受限于 shared memory size 和 SM 最大 warp 数）。需配合 Cooperative Softmax 保证 W_n>1 时计算正确性。适用前提：decode 阶段 Q length 小（1-16 tokens），M 维 warp 并行无收益。Prefill 阶段（Q length 大）仍沿用 FlashAttention 原有 warp layout。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache

---
