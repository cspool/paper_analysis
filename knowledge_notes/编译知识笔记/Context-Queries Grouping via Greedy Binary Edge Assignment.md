## Context-Queries Grouping via Greedy Binary Edge Assignment

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Context-Queries Grouping via Greedy Binary Edge Assignment 是 FastTree 提出的 runtime 优化方法，将 radix tree 结构转换为最优的 attention 计算分组方案。问题形式化：给定 KV cache radix tree T=(V,E)，每条边 e 赋值 0（分离，context 独立）或 1（拼接，context 串联），生成 virtual tree，再通过 node-centric query aggregation 得到 (context, {queries}) grouping plan。目标是最小化 grouping plan 对应的 GPU kernel latency。由于搜索空间为 $2^{|E|}$（指数级），FastTree 使用 BFS 贪心启发式算法：遍历每条 parent→child 边，估计 SplitKVCost（分离开销 = padding cost + intermediate result cost）和 SplitQCost（拼接开销 = padding cost），选开销更小的赋值。Pad cost 模型：$C_{P,q} = \mathrm{Pad}(TS_q, nQ) \cdot len \cdot d$（query dim padding）和 $C_{P,c} = nQ \cdot \mathrm{Pad}(TS_c, \min(len, TS_c)) \cdot d$（context dim padding），组合为 $C_P = \alpha C_{P,q} + \beta C_{P,c}$（α, β 为经验系数）。Intermediate result cost：$SplitKVCost_R = \gamma \cdot nQ_l \cdot d$（γ 为经验系数，nQ_l 为 leaf query 数）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

Greedy heuristic 的完整算法（Algorithm 1 from FastTree paper）：

```
Input: Radix tree T = (V, E)
1: Initialize A[v] = number of associated queries of node v (∀v ∈ V)
2: Initialize L[v] = context length of node v (∀v ∈ V)
3: for node v in BFS(V) do
4:     nQ_curr = A[v]            // current aggregated queries
5:     len_v = L[v]              // accumulated context length
6:     for leaf l in children(v) do
7:         nQ_l = A[l], len_l = L[l]
8:         // Estimate cost of separating (edge=0)
9:         C0 = SplitKVCost(nQ_curr, nQ_l, len_l, len_v)
10:            = C_P(nQ_curr, len_v) + C_P(nQ_l, len_l) + γ·nQ_l·d
11:        // Estimate cost of concatenating (edge=1)
12:        C1 = SplitQCost(nQ_curr, nQ_l, len_l, len_v)
13:            = C_P(nQ_curr - nQ_l, len_v) + C_P(nQ_l, len_v + len_l)
14:        if C0 >= C1 then
15:            Assign 1 to edge v → l      // concatenate
16:            nQ_curr = nQ_curr - nQ_l
17:            L[l] = len_l + len_v
18:        else
19:            Assign 0 to edge v → l      // separate
20:        end if
21:    end for
22: end for

Output: Binary edge assignments f: E → {0,1}
```

决策驱动流程 — 两种 plan 的对比示例（Figure 4 from paper）：

```
Tree: Root(A1) → [B1, B2], each leaf has M, N queries respectively

Plan 1 (separate all levels, edges=00):
  Groups: (A1, {Q1..Q_{M+N}}), (B1, {Q1..Q_M}), (B2, {Q_{M+1}..Q_{M+N}})
  优势: A1 group 聚合所有 queries → 大 GEMM → 高 tensor core util
  劣势: 3 groups → 更多 intermediate results → 更多 HBM write + reduction steps
  适用: M+N ≤ tile_size (无需 padding), 或 tree shallow

Plan 2 (concatenate A1+B1 and A1+B2, edges=11):
  Groups: (A1+B1, {Q1..Q_M}), (A1+B2, {Q_{M+1}..Q_{M+N}})
  优势: 2 groups → 更少 intermediate results → 更少 HBM IO
  劣势: 若 M, N < tile_size → 需 padding → wasted shared memory + compute
  适用: M, N large enough to fill tile (无需 padding), 或 tree deep
```

Padding cost 的作用：
```
场景 (d): M=N=8, tile_size=16
  Plan 1: A1 group: 16 queries OK (no pad)
          B1 group: 8 queries → Pad(16,8)=8 → 50% waste
          B2 group: 8 queries → 50% waste
  Plan 2: (A1+B1): context 128+32=160, queries=8 → Pad(16,8)=8 → Q dim waste
          (A1+B2): 同上
  → Plan 1 更优: A1 group 无 waste 抵消 B1/B2 的 waste
  
场景 (e): M=N=128, tile_size=16
  Plan 1: 所有 group > tile_size → 无 padding waste
          但 3 groups → 更多 intermediate results + reduction steps
  Plan 2: 所有 group > tile_size → 无 padding waste
          仅 2 groups → 更少 IO + reduction
  → Plan 2 更优: 无 padding difference 时，minimize group count
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Greedy heuristic 在 FastTree 的 CPU 端 runtime 中执行（Python/C++ 实现）。每次 radix tree 结构变化时（请求 arrival/completion），runtime 重新执行 BFS 贪心搜索。Heuristic cost model 中的 α、β、γ 经验系数通过 H100 GPU 的 profiling 离线校准。BFS 遍历保证线性时间 $O(|V|)$（对比枚举 $O(2^{|E|})$），预处理 overhead < 1ms（被 SGLang 的多步 continuous decoding 摊销）。Cost model 设计的关键简化：仅对 context length < tile_size 的 case 估算 padding overhead（长 context 的 padding 已是固定比例），且不尝试精确预测 latency（仅用 relative cost comparison）。FastTree 开源在 https://github.com/PanZaifeng/FastTree-Artifact。

涉及论文标题：
- FastTree Optimizing Attention Kernel and Runtime for Tree-Structured LLM Inference
