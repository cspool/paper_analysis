## Modular Hierarchical Token Pruning (模块化层次化Token剪枝)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Modular Hierarchical Token Pruning 是 InfiniteHiP 提出的免训练长上下文 token 剪枝算法。核心思想：通过堆叠多个剪枝模块（pruning stage），每个模块基于当前 query block 动态评估 key chunk 的重要性，逐步将候选 key token 从全量缩减到常数级别（~2K-4K），最终生成 block sparse attention mask。与 HiP Attention 的迭代式 top-k 不同，每个剪枝模块使用 per-chunk top-1 代表 token 选择（而非全局 top-k），消除了全局 thread synchronization，实现 key sequence dimension 上的高并行度。

每个剪枝 stage S^(i) = (b_q^(i), l_c^(i), k^(i)) 包含三个参数：query block size b_q、chunk size l_c、保留 token 数 k。Stage 间数据流：全量 key → Stage 0: 分 chunk(l_c=256)→每 chunk 选代表 token→保留 top K chunk→Stage 1: 分 chunk(l_c=32)→选代表→保留 top K chunk→Stage 2: 分 chunk(l_c=8)→选代表→保留 top K chunk→输出 ~2K-4K key indices 用于 BSA。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**三阶段模块化剪枝 pipeline（3K preset, T_kv=1M tokens）**：

```
Input: Q ∈ R^(H×T_q×d), K ∈ R^(H×T_kv×d), n_sink=256, n_stream=1024
Output: Sparse key indices I^(3) for each query block

// Stage 0: 全量 key → 32K tokens
I^(0) = [n_sink, ..., T_kv - n_stream]  // 排除 sink/streaming
S^(0) = (b_q=64, l_c=256, k=32K)
For each query block m:
  C_j = chunk(I^(0), l_c=256)  // 约 4000 chunks
  For each chunk j:
    r_j = SelectRep(q_m, C_j, K)  // O(log₂ 256)=8 次点积
    s_j = max_{h,t} (q_{h,t}^T · k_{h, r_j})  // chunk 注意力分数估计
  T = argtop_{125}(s)  // 保留 top 125 chunks (125×256≈32K)
  I'^(0) = ∪_{j∈T} C_j
// Stage 1: 32K → 8K
S^(1) = (b_q=64, l_c=32, k=8K)
For each query block m:
  C_j = chunk(I'^(0), l_c=32)  // 约 1000 chunks
  r_j = SelectRep(q_m, C_j, K)
  s_j = max_{h,t} (q_{h,t}^T · k_{h, r_j})
  T = argtop_{250}(s)  // 保留 top 250 chunks (250×32=8K)
  I'^(1) = ∪_{j∈T} C_j
// Stage 2: 8K → ~3K (2K for layers >3, 4K for layers ≤3)
S^(2) = (b_q=64, l_c=8, k=2048|4096)
For each query block m:
  C_j = chunk(I'^(1), l_c=8)  // 约 256-512 chunks
  r_j = SelectRep(q_m, C_j, K)
  s_j = max_{h,t} (q_{h,t}^T · k_{h, r_j})
  T = argtop_{K}(s)
  I^(2) = ∪_{j∈T} C_j

// Final: BSA with I^(2) ≈ 2K-4K selected keys + sink + streaming
```

术语一般如何实现？如何使用？

InfiniteHiP 使用单个参数化的 Triton kernel 实现所有剪枝 stage，通过不同 (b_q, l_c, k) 参数区分。SelectRep 算法每次迭代仅访问 2 个 token（左右分支首 token），因此无需全局同步——这与 HiP Attention 的迭代式 top-k（需要全局同步）形成关键差异。剪枝 module 的数量 N=3 经实验确定的延迟-性能最优组合。

涉及论文标题：
- InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU
