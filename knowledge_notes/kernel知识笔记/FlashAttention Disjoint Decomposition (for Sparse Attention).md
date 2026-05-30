## FlashAttention Disjoint Decomposition (for Sparse Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FlashAttention Disjoint Decomposition 是 Focus 论文提出的将分组稀疏注意力 mask 分解为两个不相交的 FlashAttention 调用的技术。核心思想：Focus 的注意力 mask M(i,j) = 1[j≤i] ∧ (1[g(i)=g(j)] ∨ 1[|i-j|≤w]) 天然可分解为两个互斥且完备的集合——A = {(i,j): j≤i ∧ g(i)=g(j)}（同组 causal 对）和 B = {(i,j): j≤i ∧ |i-j|≤w ∧ g(i)≠g(j)}（跨组 local 对）。由于 A∩B=∅（一个要求同组、另一个要求异组）且 A∪B=M（覆盖所有应关注的 pair），两路 FlashAttention 输出可通过 logsumexp merge 数学精确合并（cosine similarity 1.0000 vs O(n²) reference），消除 double-counting 问题和数值不稳定性（直接加减法在 logsumexp 空间中数值灾难性，cosine similarity 仅 0.79）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
分解的 kernel 调度流程（320 行 Python，仅调用 flash_attn_func）：

```
def focus_flash_attention(q, k, v, group_ids, w, K):
    """
    q, k, v: [batch, heads, T, d_head]
    group_ids: [T]  每个 token 的 group (0..K-1)
    w: local window size
    """
    # ===== Phase A: 同组 causal attention (O(T²/K)) =====
    # Step 1: Stable sort by group (保持 causal order)
    sorted_idx = torch.argsort(group_ids, stable=True)   # [T]
    reverse_idx = torch.argsort(sorted_idx)              # inverse map
    
    q_sorted = q[:, :, sorted_idx, :]
    k_sorted = k[:, :, sorted_idx, :]
    v_sorted = v[:, :, sorted_idx, :]
    
    # Step 2: 统计每组大小, reshape 为 K 个独立序列并 pad
    group_sizes = torch.bincount(group_ids, minlength=K)
    max_len = group_sizes.max()
    
    # Step 3: 对 K 个 group 分别调用 FlashAttention
    o_A_parts, lse_A_parts = [], []
    for k_idx in range(K):
        # flash_attn_func 内部: tiled QK^T → online softmax → V 加权
        # 每个 group 约 T/K 个 token, 单组复杂度 O((T/K)²)
        # K 组合计 O(K · (T/K)²) = O(T²/K)
        o_g, _, lse_g = flash_attn_func(
            q_sorted_padded[k_idx],
            k_sorted_padded[k_idx],
            v_sorted_padded[k_idx],
            causal=True
        )
        o_A_parts.append(o_g)
        lse_A_parts.append(lse_g)
    
    # Step 4: Unpad + unsort 还原原始顺序
    o_A = unsort(unpad(o_A_parts, group_sizes), reverse_idx)
    lse_A = unsort(unpad(lse_A_parts, group_sizes), reverse_idx)
    
    # ===== Phase B: 跨组 local attention (O(Tw)) =====
    # flash_attn_func 的 windowed attention
    # mask 跨组 pair 为 -inf, 同组 pair 在 Phase A 已处理
    o_B, _, lse_B = flash_attn_func(
        q, k, v,
        causal=True,
        window_size=(w, 0),        # 局部窗口
        # 附加自定义 mask: 同组 pair → -inf (避免与 Phase A 重复)
    )
    
    # ===== Merge: logsumexp 空间精确合并 =====
    # o[i] = (exp(lse_A[i]) * o_A[i] + exp(lse_B[i]) * o_B[i])
    #        / (exp(lse_A[i]) + exp(lse_B[i]))
    w_A = torch.exp(lse_A) / (torch.exp(lse_A) + torch.exp(lse_B))
    w_B = torch.exp(lse_B) / (torch.exp(lse_A) + torch.exp(lse_B))
    output = w_A * o_A + w_B * o_B
    
    return output
```

性能特征：
- Phase A: K 次 FlashAttention 调用, 每次处理约 T/K 个 token, 总 O(T²/K)
- Phase B: 单次 windowed FlashAttention, O(Tw)
- Sort + gather/scatter: O(T log T), ~12ms @ T=1M (vs 1.5s for full attention)
- Merge: O(T), negligible
- K=8, T=1M: 理论加速 8×, 实测 8.6× (FlashAttention 在短序列上效率更高带来额外收益)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：
- 无需自定义 CUDA kernel / Triton / 编译——纯 Python + PyTorch + flash_attn_func
- stable sort 保证 causal order 不被打乱（关键约束：group 内 token 的顺序必须保持）
- Pad 策略：每组 pad 到 max_len，padding 部分 causal mask 自动忽略（FA 内置）
- 验证：所有配置下 cosine similarity 1.0000 vs O(n²) reference
- 限制：短序列 (≤4K) 下 sort 开销 12ms 占主导，不加速

与直接 FlashAttention 的关系：
- 不修改 FlashAttention kernel 本身（复用标准 flash_attn_func）
- 通过 token 重排（sort by group）将稀疏 mask 问题转化为标准密集 attention 问题
- 这是 "data reorganization + existing kernel" 策略而非 "custom kernel" 策略

涉及论文标题：
- Why Attend to Everything? Focus is the Key (Composing Sparse Attention via Learned Grouping)

涉及论文标题：
- Trainable_Dynamic_Mask_Sparse_Attention
