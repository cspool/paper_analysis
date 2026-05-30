## Symmetric Tensor Layout for Conflict-Free One-Sided Access

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Symmetric Tensor Layout（对称张量布局）是 FlashMoE 为跨 GPU 非阻塞 one-sided 访问设计的内存组织方案。定义 L ∈ R^{P×R×B×E×C×H}，P = world size, R = 2 rounds (dispatch+combine), B = 2 buffers (outgoing+incoming), E = local experts, C = capacity, H = hidden dim。核心是 temporal buffering：4× overprovision（2 rounds × 2 buffers）使每个数据流有独立 buffer。Theorem 3.1 证明 layout write-write conflict-free。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Layout: L[P][R][B][E][C][H]
// Write rules: inter-device → p*=p_s, b=1; intra-device staging → b=0, p_s=p_t
// Conflict-free proof: p_s1 ≠ p_s2 → p*_1 ≠ p*_2 → i1 ≠ i2
// Size(L) ≈ 4 × Size(T), Memory overhead ≤ 2.15%

// 数据流:
// GPU i dispatch: write L[i,0,0,:,:,:] → NVSHMEM put → GPU j L[i,0,1,:,:,:]
// GPU j compute: Subscriber decode → Processor GEMM0→GEMM1
// GPU j combine:  write L[j,1,0,:,:,:] → NVSHMEM put → GPU i L[j,1,1,:,:,:]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- NVSHMEM 对称内存分配（`nvshmem_malloc`）确保地址对称
- Capacity upscaling 对齐 tile block size bM=128
- 2-round × 2-buffer overprovision 确保 producer-consumer 无需同步
- Memory overhead ≤ 2.15%（Mixtral 8x7B, 32K seqlen），DeepSeek-V3 仅 0.11%

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel
