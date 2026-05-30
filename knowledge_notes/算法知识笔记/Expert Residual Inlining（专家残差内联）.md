## Expert Residual Inlining（专家残差内联）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Residual Inlining 是 NetMoE 为支持 Dynamic Sample Placement 而提出的计算顺序重排技术。在标准 Transformer MoE 层中，残差连接（residual connection）在 All-to-All Gather 之后执行：`output = gather(expert_outputs) + residual_input`。但在 NetMoE 中，All-to-All Gather 阶段会改变 token 的放置位置（按优化后的 SmpDev 重分配），若残差在原位置执行将导致计算错误。Expert Residual Inlining 将残差加法从 gather 之后移到 scatter 之后、gather 之前：`output_on_expert_device = scatter_input + expert_output`，然后 gather 只需将结果传输到新的 sample 位置。这样保证：不论 token 最终被 gather 到哪个 GPU，其残差连接的计算结果都是正确的——因为残差已经在 expert 所在的 GPU 上被加到了 expert 输出中。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
**标准 MoE 层前向传播（无 sample placement 调整）**：
```
input: x ∈ R^{S×H}  (tokens on device d)
route = gating_network(x)  # K selected experts per token
x_scattered = all_to_all_scatter(x, route)  # dispatch to expert GPUs
x_expert = experts(x_scattered)  # FFN on each expert's GPU
x_gathered = all_to_all_gather(x_expert, reverse(route))  # return to original GPU
output = x_gathered + x  # residual connection
```

**NetMoE 的 Expert Residual Inlining**（Algorithm 1, lines 10-12）：
```
input: x ∈ R^{S×H}  (tokens on device d)
route = gating_network(x)
x_scattered = all_to_all_scatter(x, route)  # dispatch to expert GPUs
x_expert = experts(x_scattered)  # FFN on expert's GPU
x_inlined = x_expert + x_scattered  # EXPERT RESIDUAL INLINING: residual added HERE
# CPU 后台求解最优 SmpDev
x_gathered = all_to_all_gather(x_inlined, SmpDev_optimized)  # gather to NEW positions
# output = x_gathered  # 无需再 + x，已内联
```
注意：残差加法的输入是 `x_scattered`（scatter 到 expert GPU 上的 token 数据）而非原始的 `x`（原始 GPU 上的数据），因为两者在 scatter 前后是等价的（只是位置不同），且 inlining 后不再需要 gather 回原 GPU——这是实现零额外通信开销 place 调整的关键。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现要求：修改 MoE 层的 forward 函数，将 `x_residual = x` 的保存和 `output = gathered + x_residual` 的执行逻辑替换为 `gathered = all_to_all_gather(x_expert + x_scattered, new_placement)`。
- 正确性保证：数学上等价——残差加法的交换律和结合律保证无论在哪台 GPU 上执行 `expert_output + original_input` 结果相同。
- 与标准 Transformer 的差异：标准实现中残差独立于 MoE 层（在 Transformer block 级别），而 Expert Residual Inlining 将残差嵌入 MoE 层内部。
- 适用范围：仅在需要改变 token 返回位置（如动态 sample placement）时才需要。若不做 placement 调整，标准残差方式更简单。
- 限制：论文未讨论对 gradient checkpointing（重计算）和混合精度训练（FP16/BF16）的具体影响，这些是实际部署中需要验证的问题。

涉及论文标题：
- NetMoE: Accelerating MoE Training through Dynamic Sample Placement
