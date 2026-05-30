## Expert-Choice Routing for Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Expert-Choice Routing 最初由 Zhou et al. (NeurIPS 2022) 在 Mixture of Experts 中提出，核心思想是反转传统 token-choice routing：不是让每个 token 选择 top-k 个专家，而是让每个专家从 batch 中选择自己偏好的 top-k 个 token。这天然保证每个专家处理恰好 k 个 token，实现完美负载均衡，无需 auxiliary load-balancing loss。

MoSA 将此范式移植到 attention 机制：每个 attention head 作为一个"专家"，通过可学习的 per-head router 从输入序列中选择自己需要处理的 k 个 token。与标准 MoE 中 Expert-Choice 的关键区别：MoSA 在每个序列内独立选择 token（而非跨 batch），且选择基于 per-head scoring function 而非跨 head 共享的 gating network。

与传统 token-choice routing（如 Switch Transformer）相比：(1) 完美负载均衡——每个 head 恰好 k 个 token；(2) 动态计算分配——重要 token 可被多 head 选中获得更多计算；(3) 避免 expert collapse。代价是某些 token 可能不被任何 head 选中（需 dense head 兜底）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Expert-Choice vs Token-Choice Routing in Attention

# Token-Choice (Switch Transformer style):
#   gate = softmax(X @ W_gate)          # [T, E]
#   selected = topk(gate, k_experts)     # per-token
#   问题: 某些 expert 可能被大量 token 选中 → collapse
#   需要 auxiliary loss: L_balance = α·E·Σ_i f_i·P_i

# Expert-Choice (MoSA style):
#   for each head i:
#     scores = σ(X @ W^r_i)              # [T], sigmoid
#     tokens_i = topk(scores, k)          # head selects k tokens
#   优势: |tokens_i| ≡ k → 完美均衡
#   代价: ∃j 可能不被任何 head 选中

# 对比: Routing Transformer (online K-means):
#   centers_i = EMA of similar tokens    # 慢收敛
#   tokens_i = argmin dist(Q_j, centers)  # 需要先算全部 Q
#   FLOP 远高于 MoSA（投影 T 级 vs k 级）
```

术语一般如何实现？如何使用？

在 MoSA 中，Expert-Choice Routing 通过 per-head W^r 实现，sigmoid 激活（非竞争，遵循 σ-MoE 发现），避免 softmax 造成的 token 间竞争。Router 输出 r_topk 在 attention 后 gating 输出使梯度可反向传播。训练中 teacher-forcing 使 TopK 可看到全部 token；推理中需像 Mixture-of-Depths 学习自回归 router 预测 token 被选中的概率（论文列为 future work）。MoSA 不需要 auxiliary load-balancing loss。

涉及论文标题：
- Mixture of Sparse Attention: Content-Based Learnable Sparse Attention via Expert-Choice Routing
