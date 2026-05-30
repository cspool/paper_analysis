## PageRank-based Token Importance Scoring

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

PageRank-based Token Importance Scoring 是一种基于图中心性算法评估 token 在 Attention 图中重要性的方法。将 Self-Attention 权重矩阵 A^l ∈ R^{(N+M)×(N+M)} 视为有向图的邻接矩阵，每个 token 是图中的一个节点，A_{i,j} 表示 token j 对 token i 的"投票"权重。在此基础上运行 PageRank 算法——稳态分布中得分高的节点即"被众多重要节点所关注的节点"——得分者即为重要的 token。

与 A2S（简单累加 Attention Score）不同，PageRank 递归考虑"被谁关注"：一个 token 若被高分 token 关注，其自身得分也高。这更准确反映 Attention 图中的信息流结构。

公式（AIM 公式 1）：

$$s_i^l = \frac{1}{N^l + M^l} \sum_{j=1}^{N^l + M^l} \mathbf{A}_{i,j}^l \cdot s_j^l$$

其中 $\mathbf{A}^l$ 为 softmax 归一化的 Attention 权重矩阵，$s$ 初始均匀分布。AIM 仅对 visual tokens 按 PageRank 分数排序剪枝，text tokens 始终保留。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**PageRank 用于 Token 重要性评估的完整流程**：

```
def pagerank_token_importance(A, num_visual, num_text, tol=1e-6):
    """
    A: Attention 权重矩阵 [N_v+N_t, N_v+N_t]
    num_visual: 视觉 token 数量
    num_text: 文本 token 数量
    返回: visual tokens 的 PageRank 重要性分数 [N_v]
    """
    N = num_visual + num_text
    
    // 初始化：均匀分布
    s = ones(N) / N
    
    // 幂迭代（Power Iteration）直到收敛
    for _ in range(max_iter):
        s_new = A.T @ s / N
        if norm(s_new - s) < tol:
            break
        s = s_new
    
    // 仅返回 visual token 的分数
    return s[:num_visual]
```

**为什么使用 PageRank 而非直接累加 Attention Score**：
- A2S：只看"∑_{query} 对 key 的关注总量"——与 token 位置相关，早期 token 累积次数多
- PageRank：考虑 Attention 图的全局结构——"被重要 token 关注的 token 更可能是重要的"——递归纠正位置偏差

术语一般如何实现？如何使用？

在 AIM 实现中，每层 Attention 计算后取 softmax 后的 A 矩阵（不用 FlashAttention），运行少量迭代的幂迭代法（矩阵-向量乘法 O(N²)）。因为仅对 visual tokens 排序，且 visual token 数量随层递减，实际开销很小（4.18 GFLOPs for Qwen2-7B）。该技术与 Zero-TPrune（Wang et al., CVPR 2024）中的 PageRank-based token pruning 一脉相承，AIM 将其扩展到多模态场景。

涉及论文标题：
- AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning

---
