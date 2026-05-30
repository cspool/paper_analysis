## Tree Attention（树注意力机制）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Tree Attention 是一种为树状结构的 token 序列设计的因果注意力变体。在标准 self-attention 中，causal mask 保证每个 token 只能 attend 其左侧（前缀）的所有 token。在 tree-based speculative decoding 中，draft model 构建了一个 token 树（多条候选路径），需要 target model 在一次前向中并行验证整个树的所有节点。Tree Attention 使用 **topology-aware causal mask** 替代标准 causal mask：每个树节点只能 attend 其祖先节点（从根到该节点的路径上的 token），而不能 attend 树中的兄弟节点或无关分支的 token。这保证了因果性的同时允许并行处理整棵树。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Tree Attention 的 topology-aware causal mask 构造
# 树结构: 每个节点有 parent_id, position_id

def build_tree_attention_mask(tree_nodes, N):
    # tree_nodes: [{id, parent_id, position_id, token}]
    # N: 总节点数
    mask = zeros(N, N)  # 全 -inf / 全 masked
    
    for i, node_i in enumerate(tree_nodes):
        for j, node_j in enumerate(tree_nodes):
            # node_j 是 node_i 的祖先（或自己）
            if is_ancestor(node_j, node_i, tree_nodes):
                mask[i][j] = 0  # 允许 attention
    
    return mask

# is_ancestor: node_j 在 node_i 的根到 node_i 路径上
def is_ancestor(node_j, node_i, tree_nodes):
    cur = node_i
    while cur is not None:
        if cur.id == node_j.id:
            return True
        cur = cur.parent
    return False
```

TwigVLM++ 的 Tree-based SSD 设置：
- Draft model 构建 token tree：expansion width E=10, selection width K=10, depth D=4
- 树结构：Level 1: E=10 children; Level 2-4: 每层选 top-K=10 节点各扩展 E=10 children
- 验证前裁剪至 Nmax=60 节点
- Target model 用 tree attention 并行验证

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Tree Attention 最早由 SpecInfer (Miao et al., 2023) 和 Medusa (Cai et al., 2024) 提出。实现方式：在 FlashAttention 或标准 attention kernel 的基础上，将 topology-aware mask 传入 attention 的 mask 参数即可。关键开销权衡：(1) 树越大 → 每次接受更多 tokens → 速度更快；(2) 但验证的前向计算量也更大（需处理 Nmax 个节点）；(3) 需要在实际中获得接受 token 率和验证开销之间的平衡。TwigVLM++ 使用 E=K=10, D=4, Nmax=60 的配置，在长 response 场景下实现 ~197% RelSpd（vs 标准 SSD 的 ~154%）。

涉及论文标题：
- Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models
