## Cross-Layer Mutual Information in MoE Routing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Cross-Layer Mutual Information (MI) in MoE Routing 是 RMoE 论文提出的分析工具，用于量化 MoE 模型中不同层 router 之间共享的路由信息量。基于各层 router 输出的 gating probability distribution（softmax 后 N 个 expert 的概率分布）计算 pairwise MI：对每对层 (i,j)，将 token 的 routing distribution 离散化（100 bins）后使用 sklearn 的 mutual_info_score 计算 MI，形成 L×L 的 MI 矩阵。高 MI 值表示两层 router 决策高度相关。论文使用 MI 矩阵验证了 RMoE 的设计直觉：标准 SMoE/XMoE/HyperMoE 的跨层 MI 均很低（<0.1），router 层间独立决策；RMoE 的 MI 显著高于所有 baseline，验证 GRU 促进了跨层信息共享。该框架可推广用于评估任何跨层路由方法的有效性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Cross-Layer MI 计算
import numpy as np
from sklearn.metrics import mutual_info_score

def discretize(prob_dist, bins=100):
    return np.digitize(prob_dist, bins=np.linspace(0, 1, bins))

def calc_cross_layer_mi(all_gate_probs):
    # all_gate_probs: shape (L, N_tokens, N_experts)
    L = all_gate_probs.shape[0]
    mi_matrix = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            mi_vals = [mutual_info_score(
                discretize(all_gate_probs[i, t]),
                discretize(all_gate_probs[j, t])
            ) for t in range(N_tokens)]
            mi_matrix[i, j] = np.mean(mi_vals)
    return mi_matrix
```

MI 矩阵解读（L×L heatmap）：(1) SMoE/XMoE/HyperMoE: 全体 MI < 0.1 → router 独立决策；(2) RMoE: 对角线附近 MI 高（邻近层信息共享强），远距离层 MI 逐渐衰减但仍 > 0；(3) RMoE-NP-r1.0: 仅对角线高，远距离层 MI 快速衰减 → logits 残差难以保证长程信息共享。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

该分析框架适用于：(1) 验证跨层路由方法是否建立了真正的信息共享（而非仅增加参数量）；(2) 对比不同跨层机制的传递效率（GRU vs logits residual vs attention-based）；(3) 诊断路由训练问题（异常高 MI 可能表示 representation collapse）。计算复杂度 O(L²·T)，在验证集上采样即可。注意 MI 受 expert 数 N 影响（N 越大基线越低），应在相同 N 下跨方法比较。

涉及论文标题：
- Layerwise Recurrent Router for Mixture-of-Experts
