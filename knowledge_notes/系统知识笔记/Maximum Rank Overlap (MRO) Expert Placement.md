## Maximum Rank Overlap (MRO) Expert Placement

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Maximum Rank Overlap (MRO) Expert Placement 是 Lazarus 提出的可证明最优的 expert placement 算法，在给定的 expert replica allocation（r_e）下最大化均匀随机节点故障后的恢复概率。MRO 的核心原则是将 experts 分为 ⌈E/c⌉ 组（每组最多 c 个 experts，c 为每节点 replica slot 数），每组内最大化 experts 的跨节点重叠度：S_{lo} ⊂ S_{lo+1} ⊂ ... ⊂ S_{hi}（从最不流行到最流行的 expert）。

直觉解释："把所有鸡蛋放在一个篮子里"——通过让多个 experts 共享相同的故障节点集合，减少总体的 failure patterns 数量。每个 non-representative expert 的故障组合集是其 representative expert 故障组合集的子集，从而最小化"导致无法恢复的节点故障组合"的总数。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

MRO 算法流程（E > c 的困难情况）：

```
Input: E experts, r_e replicas per expert, N nodes, c slots/node
  Experts sorted by popularity (ascending)

Step 1: Partition experts into ⌈E/c⌉ groups
  Group 1: experts {1, ..., c}        (least popular)
  Group 2: experts {c+1, ..., 2c}
  ...
  Group ⌈E/c⌉: experts {c*(⌈E/c⌉-1)+1, ..., E}  (most popular)

Step 2: Partition nodes
  Group 1: r_1 nodes (representative expert=1)
  Group 2: r_{c+1} nodes (representative expert=c+1)
  ...
  Group ⌈E/c⌉: min(N - sum of previous, r_{representative}) nodes

Step 3: For each group i, each node in its node group gets
  one replica of each expert in the group:
  Node j in group i:
    Col_j = {c*(i-1)+1, c*(i-1)+2, ..., min(c*i, E)}
  
  Result: S_{c*(i-1)+1} ⊂ S_{c*(i-1)+2} ⊂ ... ⊂ S_{min(c*i, E)}
  (subset relationship within each group)

Step 4: Fill vacant slots uniformly with remaining replicas
```

**示例**（4 experts, c=2, N=5, r=[2,2,3,3]）：
- Group 1: experts {1,2} → r_1=2 nodes → nodes 1,2 各有 {1,2}
- Group 2: experts {3,4} → r_3=3 nodes → nodes 3,4,5 各有 {3,4}
- Recovery needs: ≥1 node alive from {1,2} AND ≥1 node alive from {3,4,5}

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MRO 在 Lazarus Controller 上以 Python 实现，placement plan 计算 <100ms（对 1K GPU 规模 <1s）。由于 expert load distribution 在不同层之间不同，MRO 为每层独立计算 placement（每层有独立的 r_e 和 placement plan）。当故障发生时，Controller 仅用剩余 alive nodes 重新计算 MRO plan。当新节点加入时，Controller 将新节点纳入计算。

MRO 的 theoretical guarantee (Theorem 1) 证明：对于任意给定的 replica 分配 r_e 和均匀随机节点故障，MRO plan 最大化 Pr(∪_{a∈A} Col_a = [E])，即所有 experts 能被恢复的概率。在 GPT-L (16 experts) 上，4 节点故障时 MRO 恢复概率为 41%，而 spread placement（round-robin）仅 12%。

涉及论文标题：
- Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement
