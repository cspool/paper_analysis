## ULYSSES Sequence Parallelism

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

ULYSSES (Jacobs et al., 2023, DeepSpeed) 是一种基于 All-to-All 通信的序列并行方法。与 RINGATTN 沿序列维度切分不同，ULYSSES 的核心思路是"在 attention 时将 sequence layout 转换为 head layout"：(1) attention 之前：输入按 sequence 维度切分到各 GPU，各 GPU 独立执行 MLP/LN（不同 token 间无依赖）；(2) attention 计算时：通过 All-to-All 通信将 sequence layout 转为 head layout——每个 GPU 持有完整序列的部分 attention heads；(3) attention 之后：再通过 All-to-All 转回 sequence layout。

ULYSSES 保持精确注意力的计算语义不变（FULLATTN），但受到模型 attention head 数量的限制——序列并行度 H 不能超过 head 数量（否则有些 GPU 分不到 head）。对于 GQA/MQA 模型（KV head 数远小于 Q head 数），ULYSSES 需要特殊处理（如 KV cache replication）。

从kernel调度角度拆解术语。

**ULYSSES 的 All-to-All 布局转换流程**：

```
// 假设 H 个 GPU, N 个 token, h 个 total heads
// 每 GPU 初始持有 N/H 个 token 的所有 heads

// Step 1: Attention 前（sequence layout → head layout）
// 输入：每 GPU [N/H, h/H, d]（自己的 tokens, 自己的 heads）
// 输出：每 GPU [N, h/H, d]（全部 tokens, 自己的 heads）
Q = AllToAll(Q, scatter_dim=0, gather_dim=1)
K = AllToAll(K, scatter_dim=0, gather_dim=1)
V = AllToAll(V, scatter_dim=0, gather_dim=1)

// Step 2: 每 GPU 独立计算 attention（完整序列的部分 heads）
A = flash_attn(Q, K, V)   // [N, h/H, d], 本地计算，无通信

// Step 3: Attention 后（head layout → sequence layout）
A = AllToAll(A, scatter_dim=1, gather_dim=0)
// 输出：每 GPU [N/H, h/H, d]（自己的 tokens, 自己的 heads）
// 继续 MLP/LN（不同 token 间独立）
```

**Wall-time 分解（128K, Llama-3.1-8B, 8 GPUs, per block）**：
- QKV Projection: 3.31 ms
- Communication: 3.90 ms（3× All-to-All, ~3% total）
- Attention: 84.53 ms
- FFN: 25.88 ms
- Total: 124.51 ms/block

**ULYSSES vs RINGATTN**：
| 维度 | ULYSSES | RINGATTN |
|------|---------|----------|
| 通信模式 | All-to-All (collective) | P2P ring |
| 通信量 | O(N×h/H) per All-to-All | O(N×h/H) per ring step |
| 通信轮数 | 3 (Q,K,V) + 1 (A) = 4 | H-1 |
| Head 限制 | H ≤ num_heads | 无 |
| 适用场景 | intra-node (高带宽) | cross-node (可 overlap) |

术语一般如何实现？如何使用？

ULYSSES 通过 NCCL All-to-All collective 实现。在 DeepSpeed Ulysses 中可以直接配置 `sequence_parallel_size` 参数。在 Shift Parallelism 中，通过 `--ulysses-sequence-parallel-size SP` 指定。APB 论文中使用 ULYSSES 作为 FULLATTN 的性能代表。开源：https://github.com/microsoft/DeepSpeed（Ulysses 实现）。

涉及论文标题：
- APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs

---
