## Tree-structured KV Cache Compression / TreeKV (树形结构 KV 缓存压缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

TreeKV 是一种 training-free 的 KV cache 压缩方法（He et al., 2025），核心创新是用**树形结构（tree structure）**替代全局贪心淘汰，实现"左侧稀疏、右侧密集"的平滑 cache 压缩。其设计动机来自 wavelet 分析：对 attention-weighted values 做 multi-level Haar wavelet 分解发现，token 的信息贡献从远到近平滑递增，且与邻居 token 的差异性也逐渐增大（高频分量增长显著）。基于此，TreeKV 设计了一种循环淘汰范围（cyclic eviction scope）机制：在 decoding 每 step，当 cache 满时仅在相邻两个 token {idx, idx+1} 间淘汰重要性较低者，idx 从 1 到 c 循环递增，使得淘汰均匀分布在序列全程。

与 H2O/TOVA 全局贪心排序（O(t log t)，产生区域偏差）不同，TreeKV 每 step 淘汰 O(1)（仅比较两个值），且循环结构自然产生 coarse-to-fine 的信息层次。TreeKV 同时适用于 decoding（token 级）和 prefilling（block 级）两个阶段。Ablation 实验表明树结构本身（而非 attention-weight-based selection）才是性能的核心来源。

从算法pipeline角度拆解术语。

**TreeKV Decoding Stage 伪代码**（论文 Algorithm 1）：

```
参数: cache_size = c (含 4 sink + 508 recent + 512 selected)
S = zeros(c), C = zeros(c), idx = 1
K_cache, V_cache = [], []

for t in 1..T:
    q, k, v = x[t] @ W_Q, x[t] @ W_K, x[t] @ W_V
    K_cache.append(k); V_cache.append(v)
    a = softmax(q @ K_cache^T / sqrt(d))

    C = (C union {0}) + 1
    S = (S union {0}) + a

    if len(K_cache) > c:
        S_avg = S / C  # mean attention weight
        if S_avg[idx] > S_avg[idx+1]:
            evict (idx+1)-th elements
        else:
            evict idx-th elements
        idx = (idx + 1) mod c + 1

    # Position encoding re-assignment per relative order
```

**Prefilling 阶段差异**: prompt 切分为 blocks（block size = b），用最后一个 block query 得到 per-block importance，在 block 级别并行执行上述树形淘汰。

**Tree-structured KV Cache Competition 树形竞争示意**:
```
初始: [T1, T2, T3, T4, T5, T6, T7, T8]
idx=1: T1 vs T2 → 淘汰较低分者
idx=2: T3 vs T4 → 淘汰较低分者（假设前一淘汰后 cache 重新索引）
...
多轮后形成 "左疏右密" 的树形结构
```

术语一般如何实现？如何使用？

论文声明开源 https://github.com/ZiweiHe/TreeKV（截至检索时为空）。HuggingFace Transformers 使用，Llama-2-7B + Llama-3.2-1B-Instruct，NVIDIA RTX 4090 bf16。Cache 组成: 4 sink + 508 recent + 512 TreeKV-selected = 1024 total。16k context 下 16× 压缩。10M token 序列 NLL 稳定（H2O/TOVA 退化）。Longbench 6% budget 达最优效率。每 step O(1) 淘汰开销。

涉及论文标题：
- TreeKV: Smooth Key-Value Cache Compression with Tree Structures
