## Bottleneck Matching Problem (瓶颈匹配问题)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bottleneck Matching Problem（瓶颈匹配问题）是一种组合优化问题：在赋权二分图中，寻找一个完美匹配使得匹配中**最大边权重**最小化（即 min-max 目标）。与经典的 minimum-weight perfect matching（最小化总权重和）不同，bottleneck matching 关注的是最坏情况的优化。

在 Aurora 中，bottleneck matching 用于求解 Colocating+Heterogeneous 场景中的最优 expert 共置。二分图左侧为 Model a 的 n 个 expert，右侧为 Model b 的 n 个 expert，边权重为 max(a_i+b_j, a_{n+i}+b_{n+j})——表示两个 expert 共置在同一 GPU 后该 GPU 的最大发送/接收流量。求解 bottleneck matching 得到使瓶颈流量最小的 expert 配对。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
Aurora 中 bottleneck matching 的求解流程：

```
输入: 二分图 G=(A∪B, E), 边权重 w(e) for e∈E
输出: 最小化 max w(e) 的完美匹配 M

1. 将所有边按权重升序排列: w_1 ≤ w_2 ≤ ... ≤ w_{n²}
2. 二分搜索最小权重 w_min:
   low=0, high=n²
   while low < high:
     mid = (low+high)/2
     取所有权重 ≤ w_mid 的边构成子图 G_mid
     if G_mid 中存在完美匹配:  // 使用 Hopcroft-Karp O(n²√n)
       high = mid
     else:
       low = mid+1
3. w_min = w_low 对应的匹配即为最优解
```

复杂度：Hopcroft-Karp 为 O(n²√n)，二分搜索需要 O(log n) 次调用，总复杂度 O(n²√n log n)。

**Case I 的简化（Theorem 6.2）**：当每个 GPU 的发送流量等于接收流量（a_i = a_{n+i}），bottleneck matching 简化为排序配对：将 Model a 的 expert 按流量升序排列，Model b 按流量降序排列，顺序配对即可得到最优解。这相当于交替选择热门和冷门 expert。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Bottleneck Matching 的标准解法是二分搜索 + 最大匹配检验。Hopcroft-Karp 算法是二分图最大匹配的标准多项式算法（O(E√V)）。
- 在 Aurora 中，该算法运行于**离线优化阶段**，基于历史 traffic 统计做出 expert 共置决策，而非在线实时运行。
- 在 Colocating+Heterogeneous 的最复杂场景中，Aurora 将 3D matching（NP-hard）解耦为两个 bottleneck matching：先求解 expert 共置（expert-expert matching），再求解 GPU 分配（GPU-expert_pair matching），以仅偏离最优 1.07× 的代价获得高效多项式解。
- Bottleneck matching 也广泛应用于其他负载均衡和资源分配场景（如任务调度中最小化最大完成时间 makespan）。

涉及论文标题：
- Optimizing Mixture-of-Experts Inference Time Combining Model Deployment and Communication Scheduling
