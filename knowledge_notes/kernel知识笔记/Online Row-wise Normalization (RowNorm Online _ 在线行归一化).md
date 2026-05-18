## Online Row-wise Normalization (RowNorm Online / 在线行归一化)

术语是什么？通过联网搜索让回答具体和精准。

Online Row-wise Normalization (RowNorm Online) 是 MetaAttention 提出的一种通用 online 行归一化接口，将 row-wise normalization（如 softmax、RetNet 的 reduceAbsSum normalization）拆分为 online_prologue、online_forward、online_epilogue 三段式，使得 normalization 可以在分 tile 遍历 K/V sequence 时逐步更新归一化状态，无需物化完整 score matrix 到 global memory。该设计源自 FlashAttention 的 online softmax 思想，但被泛化为通用接口以支持任意 row-wise normalization（不仅仅是 softmax）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

RowNorm Online 三段式接口的 kernel 执行流程：

```
// 以 RetNet reduceAbsSum-based RowNorm 为例
// 用户定义 RowNorm Online 接口的三个函数

class scores_RowNorm_Online:
    def online_prologue():
        row_sum_wo_clamp = 0           // 初始化未 clamp 的累积和
        row_sum = 0                     // 初始化 clamp 后的累积和
        return row_sum_wo_clamp, row_sum

    def online_forward(scores_tile, row_sum_wo_clamp_prev, row_sum_prev):
        // 当前 tile 的局部分量
        row_sum_cur = scores_tile.reduceAbsSum()
        // 更新全局未 clamp 和
        row_sum_wo_clamp = row_sum_wo_clamp_prev + row_sum_cur
        // clamp 防止除零
        row_sum = max(row_sum_wo_clamp, 1)
        // 用当前全局和归一化，并 rescale 之前 tiles 的输出
        scores_tile = scores_tile / row_sum
        scale = row_sum_prev / row_sum   // 传递给 aggregation 阶段 rescale 已累积的 output
        return scores_tile, row_sum_wo_clamp, row_sum, scale

    def online_epilogue(scores_tile):
        return scores_tile                // 最终输出（通常 identity）
```

Kernel 执行时，runtime 在遍历 KV tile 的主循环中：
```
// MetaAttention Parallel Pattern kernel 简化伪代码
row_sum_wo_clamp, row_sum = online_prologue()
output = zeros[head_dim_v]

for kv_tile in range(0, seq_len_kv, kv_tile_size):
    // 1. 异步加载 K_tile, V_tile 从 global → shared memory (TMA on H100)
    // 2. relevance scoring: scores_tile = Q × K_tile^T (Tensor Cores MMA)
    // 3. 应用 scores_Mod (如 mask, scaling) - SIMT fused
    // 4. online_forward: 更新归一化状态 + 归一化 scores_tile + 计算 scale
    scores_tile, row_sum_wo_clamp, row_sum, scale = online_forward(scores_tile, ...)
    // 5. 用 scale rescale 之前累积的 output
    output = output * scale
    // 6. aggregation: output += scores_tile × V_tile (Tensor Cores MMA)
    // 7. online_epilogue (通常 identity)

return output
```

按 RowNorm Online 标准的 online softmax 表达：
```
class scores_RowNorm_Online:
    def online_prologue():
        row_max = -inf; row_sum = 0
        return row_max, row_sum

    def online_forward(scores_tile, row_max_prev, row_sum_prev):
        row_max_cur = scores_tile.reduceMax()
        row_max = max(row_max_prev, row_max_cur)
        // rescale: 用新 max 修正之前累积和
        row_sum = row_sum_prev * exp(row_max_prev - row_max)
        row_sum += scores_tile.exp().reduceSum() * exp(row_max_cur - row_max)
        scores_tile = exp(scores_tile - row_max) / row_sum  // 归一化当前 tile
        scale = row_sum_prev / row_sum * exp(row_max_prev - row_max)
        return scores_tile, row_max, row_sum, scale

    def online_epilogue(scores): return scores
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

RowNorm Online 接口的泛化设计使得 MetaAttention 能支持任意 row-wise normalization（softmax、sigmoid、ReLU norm、RetNet reduceAbsSum norm 等），而不仅仅局限于 FlashAttention 内置的 online softmax。在 MetaAttention 实现中：online_prologue/forward/epilogue 作为 customizable function 被 trace 为 tensor DAG，其产生的中间状态（row_sum、row_max 等）作为 IntermediateTensor 纳入 scheduling（通常分配在 register 以最小化 latency），forward 中的 elementwise/scaling 操作被 SIMT fused，reduce 操作使用 intra-warp reduction。该接口的实现受到 FlashAttention 的 online softmax [Milakov & Gimelshein 2018] 和 FlashAttention [Dao et al. 2022] 的启发，但被抽像为通用接口。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends

