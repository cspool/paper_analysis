## Memory-Centric Cost Model for Attention CTA Packing（面向内存的注意力CTA打包代价模型）

术语是什么？通过联网搜索让回答具体和精准。
Memory-Centric Cost Model是PAT在pack阶段使用的CTA打包决策模型。与FastTree的compute-oriented cost model（以最小化计算量为优化目标）不同，PAT的memory-centric model以最小化global memory访问量为目标，因为decode attention是memory-bound的（瓶颈在HBM bandwidth而非Tensor Core compute）。Model计算将queries打包进同一CTA的profit（节省的KV cache加载量）与overhead（因CTA splitting产生的FP32 partial intermediate写回和读回），通过profit-overhead ratio决定parent-child node之间的split/merge策略。该model使得PAT的memory read/write比compute-oriented model（PAT-compute ablation）低10.9%，比naive每node独立pack（PAT-naive ablation）低16.7%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Memory-centric profit model的核心公式：

```
给定prefix tree node u (shared KV length l_u, s_u个共享queries):

// === Intra-node profit (node u自身打包) ===
Saving = (s_u - 1) * l_u * d     // d = head_dimension
  // 相比one-query-per-CTA: s_u次KV加载 → 1次，节省(s_u-1)次
Overhead = 8 * s_u * d            // FP32 intermediate writes + reads
  // s_u个query × 2(写+读) × 2(来自node u + 来自children)
  // × 2(FP32 vs FP16) = 8*s_u*d
Profit_Ratio = l_u / 16 ≥ 1     // 因为KV block size ≥ 16

// === Inter-node profit (parent u + child v_i) ===
// Scheme 1: Split (父和子各自独立CTA)
Profit_split = (s_u - 1)*l_u*d - 4*s_u*d + Σ_i (s_i - 1)*l_i*d
// 开销项4*s_u*d: s_u个query × 2(写+读) × 2(仅来自node u overhead)

// Scheme 2: Merge (child v_i merge进parent u的CTA)
Profit_merge = (s_u - s_i - 1)*l_u*d - 4*(s_u - s_i)*d 
              + Σ_{k≠i} (s_k - 1)*l_k*d + (s_i - 1)*(l_u + l_i)*d
// 合并后: v_i的queries不再需要intermediate write/read
// 增量: Profit_merge - Profit_split = 4*s_i*d - l_u*d

// === 决策规则 ===
if 4*s_i > l_u:  选择 Scheme 2 (Merge)
  // child query数足够多且parent prefix足够短时merge更优
else:             选择 Scheme 1 (Split)
  // parent prefix长时split避免过大的CTA和intermediate overhead
```

关键设计理念：(1) profit以global memory bytes saved计量，而非compute FLOPs——与decode attention的memory-bound性质一致；(2) overhead明确计入FP32 intermediate results的双向global memory traffic；(3) 线性复杂度O(|V|+|E|)适合online serving（每个node和edge仅处理一次）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PAT在pack scheduler中以C++实现TreeHeuristic算法。Scheduler读取vLLM block table→构建prefix tree→对每棵树调用TreeHeuristic递归遍历→按profit model决策每个internal node的split/merge→生成CTA partition。Lazy update机制使得调度结果在block table不变时跨continuous-batching iterations复用，并与pre-attention tasks异步重叠。对比实验显示：PAT-compute（替换为FastTree的compute-oriented cost model）attention latency比PAT高4.6%，memory read/write高10.9%；PAT-naive（简单每node独立pack）latency比PAT高10.4%，memory read/write高16.7%。这验证了memory-centric model设计对memory-bound decode attention的适配性。

涉及论文标题：
- PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

