## AllGather in Distributed Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

AllGather 是一种 collective communication 原语：每个进程贡献一个数据块，AllGather 将所有进程的块拼接后返回给每个进程。在 APB 中，AllGather 用于在每层 attention 计算前收集所有 host 的压缩 KV cache：每个 host 将自己的 K_h^C 和 V_h^C（l_p 个 token 的 KV）通过 AllGather 广播给所有其他 host，使每个 host 获得完整压缩 KV cache 视图以构造 passing block。

通信量：每层 2 × l_p × H × d_model × 2 (K+V) bytes。对于默认配置（H=8, l_p=2K, d=4096, FP16），通信量 = 2 × 2048 × 8 × 4096 × 2 = ~256 MB/layer。利用 NVLink 带宽（A800: 400 GB/s bidirectional），理论延迟 <1 ms。

从kernel调度角度拆解术语。

**APB 中 AllGather 的调度**：

```
// 每层每 host
// Step 1: 本地压缩
K_h^C, V_h^C = compress via retaining heads  // [l_p, d] each

// Step 2: AllGather（可与后续操作 pipeline）
K_all = AllGather(K_h^C)    // [H*l_p, d]，每 host 获得完整结果
V_all = AllGather(V_h^C)    // [H*l_p, d]

// Step 3: 构造 passing block（仅取前序 host）
K_p = K_all[0 : (h-1)*l_p]  // host 1: K_p = []
V_p = V_all[0 : (h-1)*l_p]

// Step 4: Attention
A = modified_flash_attn(Q, [K_a, K_p, K_h], [V_a, V_p, V_h])
```

**实测通信开销（Table 16, 128K, 8 hosts, per block）**：
- AllGather time: 0.62 ms（仅 ~0.8% of total 80.18 ms）
- 对比 RINGATTN P2P: 18.40 ms（~9% of 205.19 ms）
- 对比 ULYSSES All-to-All: 3.90 ms（~3% of 124.51 ms）

APB 的 AllGather 开销极小，因为压缩后的 KV cache 仅 l_p=2K（原始 l_b=16K 的 1/8）。

**LASP-2 中 AllGather 的使用**：

LASP-2 利用线性注意力的 right-product kernel trick，将 AllGather 应用于 memory state M_t ∈ R^{d×d}（而非 K_t, V_t）。由于 M_t 的大小与序列/chunk 长度无关，仅取决于 hidden dim d 和 head 数 H，通信量为 BHd²（常量）；

```
// LASP-2: AllGather on memory states
M_t = K_t^T @ V_t                    // [B, H, d, d] per device
[M_1, ..., M_T] = AllGather([M_1, ..., M_T])  // 通信量 = BHd²

// vs LASP-1: ring P2P, 2(W-1) steps, same per-step data but sequential
// LASP-2 reduces steps from 2(W-1) to 2 per iteration
```

LASP-2 在 Linear-Llama3-1B (B=16, H=16, d=2048) 上，每个 M_t 约 1.07B 参数（~2.14GB FP16），AllGather 通信量固定与序列长度无关。在 sequence length=2048K 时，计算量远大于通信量，通信开销被充分稀释。此外，LASP-2 的 AllGather 可与 intra-chunk left-product 计算在不同 CUDA stream 上 overlap。

术语一般如何实现？如何使用？

AllGather 通过 NCCL 的 `ncclAllGather` 或 PyTorch 的 `torch.distributed.all_gather` 实现。在 APB 中，通信在每个 Transformer 层的 attention 前同步进行。利用 CUDA stream 可与 retaining head 计算部分重叠。NCCL 自动选择最优算法（ring vs tree）基于消息大小和拓扑。

涉及论文标题：
- APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs
- LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid
