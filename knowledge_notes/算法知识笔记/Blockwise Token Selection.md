## Blockwise Token Selection

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Blockwise Token Selection 是一种在稀疏注意力中以**连续 token block 为粒度**选择重要 KV 子集参与注意力计算的方法。与 token-level selection（逐 token 选择，如 HashAttention）相比，blockwise selection 的核心优势在于：(1) 连续内存访问匹配 GPU 的 coalesced HBM 读取模式，blockwise 加载吞吐远高于 scatter/gather 的随机索引读取；(2) blockwise 计算兼容 Tensor Core 的矩阵乘法 tile 要求（16/32/64/128 block sizes）；(3) 注意力分数在空间上往往呈块状聚集（blockwise clustering），相邻 token 重要性相似。

NSA 的 blockwise selection 采用「免费重要性分数」策略：利用 Token Compression 分支中已计算的压缩注意力分数 $\mathbf{p}_t^{\text{cmp}}$ 来推导 selection block 的重要性。当 compression block 与 selection block 共享 blocking scheme 时（l=32, l'=64, d=16 均整除），可通过空间对应关系的加权求和直接得到 selection block 分数。GQA 架构下跨 head 聚合确保 group 内 KV block 选择一致，解码时一次加载服务所有 head。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// Blockwise Selection (复用了 Compression 的 p_cmp)
// p_cmp ∈ R^{num_comp_blocks} 已在上一步计算

// Step 1: 映射 compression block 分数到 selection block
// compression: l=32, d=16; selection: l'=64
// 每个 selection block (l'=64) 覆盖 4 个 compression strides
p_slc = zeros(t // l')
for j in range(len(p_slc)):
    start_cmp = (j * l') // d
    // 聚合该 selection block 覆盖的所有 compression block 分数
    for m in range(l' // d):   // l'/d = 4
        for n in range(l // d): // l/d = 2 (块内 stride 覆盖)
            idx = start_cmp - m - n
            if 0 <= idx < len(p_cmp):
                p_slc[j] += p_cmp[idx]

// Step 2: GQA 跨 head 聚合 (H=16 heads per group)
p_slc_shared = sum(p_slc_h for h in range(H))  // [t/l']

// Step 3: Top-n 选择 (n=16, 含 1 个初始块 + 2 个局部块)
I_t = topk_indices(p_slc_shared, n=16)  // sorted by importance
// 将连续 token block 拼接到 K_sel, V_sel
K_sel = concat(K_cache[i*l' : (i+1)*l'] for i in sorted(I_t))  // [nl', d_k]
V_sel = concat(V_cache[i*l' : (i+1)*l'] for i in sorted(I_t))

// Step 4: 精细 attention（仅对选中 block）
scores = q_t @ K_sel^T / sqrt(d_k)  // [1, nl'], nl'=1024
output = softmax(scores) @ V_sel
```

术语一般如何实现？如何使用？

Blockwise selection 的重要性分数计算有三种典型方式：(a) NSA 的「免费复用」——利用已有 compression attention 分数的空间聚合，零额外计算开销，可端到端训练；(b) 辅助 loss-based（如 SeerAttention）——训练单独的 block 重要性预测网络，用 KL 散度监督，增加额外参数和训练复杂度；(c) Heuristic-based（如 Quest 的 min-max chunk product）——无参数启发式计算，无需训练但召回率较低。

实现要点：selection block size l' 必须是 Tensor Core tile size 的倍数（通常 64 或 128）。GQA/MQA 场景必须跨 head 聚合分数再统一选择，否则每个 head 独立选择导致 KV block 加载并集远大于交集。Top-n 选择数 n 体现 sparsity-quality trade-off（NSA n=16 @ l'=64 → 1024 tokens，平均约 2560 tokens 含压缩和窗口）。

涉及论文标题：
- Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

---
