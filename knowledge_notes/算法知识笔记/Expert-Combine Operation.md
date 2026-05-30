## Expert-Combine Operation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert-Combine Operation（专家组合操作）是 MoE layer 中将多个 activated expert 的输出融合为单个 token 输出的操作。在 Top-K routing 机制下，每个 token 被路由到 K 个 expert，各 expert 产生独立输出 h_i^k，Combine 操作将这些输出按 gate affinity score 加权求和（FlashMoE 式 2-3）：

$$C_i = \sum_{j=1}^k g_{i,e}, \quad \mathbf{h}_{i} = \sum_{j=1}^{k} \frac{g_{i,e}}{C_{i}} \cdot \mathbf{h}_{i}^{k}$$

其中 g_{i,e} 为 gate affinity score，C_i 为归一化因子，h_i^k 为第 k 个选定 expert 的输出。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Expert-Combine 在 MoE pipeline 中:
# ... Dispatch → Expert FFN → Combine → ...

# Top-2 routing by FlashMoE Task 统一抽象:
# Combine task: t = (M, ⊙, identity)
# F_t(A, S, C, C) := C ← A ⊙ S + C
# A=expert output, S=gate weight, C=accumulator

for each token i:
    C_norm = sum(g_i[e] for e in selected_experts[i])
    output[i] = zeros(H)
    for k in range(top_k):
        e = selected_experts[i][k]
        output[i] += (g_i[e]/C_norm) * expert_outputs[e][i]
```

在分布式 MoE 中，不同 expert 可能位于不同 GPU，Combine 需要跨 GPU 通信——这是第二轮 AlltoAll 或 one-sided transfer 的触发点。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- GShard-style: 加权求和，所有 K 个 expert 输出参与
- Switch Transformer: Top-1, Combine 退化为 copy
- DeepSeek-V3: Top-K + shared expert, K routed + 1 shared
- FlashMoE: Combine 统一到 Task 抽象，在 Processor actor 内与 GEMM task 交错调度
- 分布式场景中，Combine 触发的跨 GPU 通信传统上用 AlltoAll，FlashMoE 用 NVSHMEM one-sided put（每 GPU individually 回传 GEMM1 结果）

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel
