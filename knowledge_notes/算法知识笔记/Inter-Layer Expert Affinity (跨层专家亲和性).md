## Inter-Layer Expert Affinity (跨层专家亲和性)

术语解释
Inter-Layer Expert Affinity（跨层专家亲和性）是 ExFlow 论文提出的概念：在 pre-trained GPT MoE 模型中，给定一个 token 在 layer i 被 route 到某个 expert，该 token 在 layer i+1 及其后续层中被 route 到特定 experts 的条件概率并非均匀分布，而是表现出强烈的非随机模式——某些 expert pairs 在跨层 routing 中具有显著更高的共现概率。这种跨层 conditional routing probability 即为 Expert Affinity。

术语是什么？
Expert Affinity 的数学形式为 conditional probability $P(E_{p,j+1}|E_{i,j})$，即在 layer j 被 route 到 expert $E_i$ 的 token，在 layer j+1 被 route 到 expert $E_p$ 的概率。ExFlow 论文通过采样 Pile 数据集的 token 并 trace 其在每层的 routing 决策，构建了完整的 cross-layer routing heatmap（Fig. 2），证实了所有 layers 对之间均存在显著的 expert affinity，且该 affinity 是模型固有属性——在 OOD 数据集（C4、Dolma、Yelp Reviews）上归一化 affinity 保持 0.989-1.005 的高度一致性。

ExFlow 进一步将 expert affinity 分为两级：
- **Intra-GPU Affinity**（第一级）：expert 对其在后续层中与该 expert 同 GPU 的 affiliated expert 的偏好
- **Intra-Node Affinity**（第二级）：expert 对其在后续层中与该 expert 同节点但不同 GPU 的 expert 的偏好

从算法pipeline角度拆解术语：
Expert Affinity 通过以下算法流程捕捉和利用：

```
# ===== 1. Profiling: 捕捉 Expert Affinity =====
# 从 Pile 随机采样 N=3000 tokens
route_log = []  # route_log[k][j] = expert_idx

for token k in sampled_tokens:
    hidden = embedding(token)
    for layer j in 0..L-1:
        gate_scores = softmax(hidden @ W_gate[j])  # Top-1 gating
        expert_idx = argmax(gate_scores)
        route_log[k][j] = expert_idx
        hidden = expert_compute(expert_idx, hidden)
        hidden = attention(hidden, context)

# ===== 2. 构建 Conditional Probability Matrix =====
# P[j][i][p] = P(E_{p,j+1} | E_{i,j})
for layer j in 0..L-2:
    for token k in 0..N-1:
        i = route_log[k][j]      # token k 在 layer j 的 expert
        p = route_log[k][j+1]    # token k 在 layer j+1 的 expert
        count[j][i][p] += 1

for layer j, expert i, expert p:
    P[j][i][p] = count[j][i][p] / sum_over_p(count[j][i][:])

# ===== 3. Combined Affinity for Multi-Expert GPU =====
# GPU with capacity C_1 experts/层: holding experts {x_1,...,x_C1} at layer j
# Find experts {y_1,...,y_C1} at layer j+1 maximizing:
# score = sum_{k} sum_{p=1..C1} sum_{q=1..C1} P(E_{y_q,j+1} | E_{x_p,j}, T_k)
```

Expert affinity 的演化特性（Fig. 12）：
- 训练初期（iteration 0-200）：expert routing 高度不平衡（少数 expert 被频繁激活），apparent affinity 极高
- 过渡期（iteration 200-2000）：GShard load balancing loss 生效，expert 激活趋于均匀，affinity 下降
- 稳定期（iteration 2000+）：expert 逐渐变得 domain-specific，affinity 稳步上升并趋于稳定

术语一般如何实现？如何使用？
- **Profiling 成本极低**：仅需 1000-3000 个 token 即可精确捕捉 expert affinity（MoE-8 仅需 1000，MoE-64 需 3000）
- **Offline 使用**：在推理前一次性完成 profiling 和 ILP 求解，结果直接用于模型加载时的 expert placement
- **无需修改模型**：不改变 gating function、不添加 training loss、不需要 fine-tuning
- **硬件拓扑无关**：expert affinity 是 model-intrinsic 属性，placement 算法可适应任意硬件拓扑
- **与 expert popularity 的区别**：之前的工作（Lina、EdgeMoE）仅考虑 per-layer 的 expert popularity（单一层的热门 expert），而 Expert Affinity 是 cross-layer 的 conditional relationship

涉及论文标题：
- Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference
