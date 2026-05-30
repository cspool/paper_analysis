## Expert Choice Routing（专家选择路由）

术语是什么？
Expert Choice Routing 是 Zhou et al. (2022) 提出的 MoE 路由算法，与传统的 Token Choice Routing（token 选择 expert）相反，Expert Choice 让每个 expert 选择 top-k 个 token 进行处理。具体地，对于 N 个 expert 和 M 个 token，计算 gating score 矩阵 S ∈ R^{N×M}，每个 expert（行）选择 score 最高的 k 个 token。这种方法天然解决了 Token Choice 中的 expert 负载不均衡问题——每个 expert 恰好处理 k 个 token（或按 capacity factor 调整），避免了某些 expert 过载而其他闲置。PEER 论文使用 Expert Choice MoE（128 experts）作为 coarse-grained MoE baseline。然而，Expert Choice 仍需要在整个 N×M 的 gating score 矩阵上操作（通过 top-k），路由复杂度至少 O(N)，限制了 expert 数量通常 < 128。

从算法pipeline角度拆解术语：
Expert Choice Routing 算法流程：
```
# S ∈ R^{N×M}: gating score 矩阵，S[i,j] = expert i 对 token j 的 score
# capacity = k × capacity_factor  (每个 expert 的 token 容量)

for each expert i in {1..N}:
    # expert i 选择 score 最高的 capacity 个 token
    selected_tokens = TopK(S[i, :], capacity)
    # expert i 仅计算被选中 token 的 FFN
    for token j in selected_tokens:
        output[j] += softmax(S[:, j])[i] × ExpertFFN_i(x_j)
```
与 Token Choice 的对比：Token Choice 是每个 token 选 top-k expert（行方向 top-k），Expert Choice 是每个 expert 选 top-k token（列方向 top-k）。Expert Choice 保证了每个 expert 的负载均衡，但需要所有 token 同时可用（训练时 batch 内），且每个 expert 处理的 token 可能来自 batch 中不连续的位置。

术语一般如何实现？
Expert Choice 在训练中应用：batch 内所有 token 共同参与 gating score 计算，每个 expert 选择 score 最高的 ⌈k × M / N⌉ 个 token（或按 capacity factor 调整）。推理时一般仍用 Token Choice（因 token 按流式到达）。PEER 论文中 Expert Choice MoE baseline 使用 128 个 expert，每个 expert 大小等于对应 dense 模型的 FFW 层。与 PEER 的对比：Expert Choice 为 O(N) 路由复杂度，限制 N < 128；PEER 为 O(√N) 复杂度，支持 N ≥ 10⁶。

涉及论文标题：
- Mixture of A Million Experts
