## Non-Uniform Hierarchical Expert Grouping for Distributed MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Non-Uniform Hierarchical Expert Grouping 是 GRACE-MoE 的 expert placement 策略，替代传统的 uniform grouping（每设备等量 expert）。分层设计：跨节点层面使用 fully non-uniform（无 size 约束）最大化 intra-node affinity 以减少昂贵的跨节点通信；节点内层面使用 controlled non-uniform（ratio r 约束 group size deviation $\delta = E \cdot r$）保留 affinity 的同时限制负载倾斜。r 通过绘制 U(r)（intra-group affinity utilization）vs S(r)（size deviation）曲线取 knee point 确定。r=0 即 uniform grouping，r=1 即 fully non-uniform。

从算法pipeline角度拆解：

```
E = floor(n_experts / D); delta = max(1, round(E * r))
{C_d} = SpectralClustering(A, D)
for each oversized C_d: trim to num_max, push overflow to Omega
for e in Omega: assign to group maximizing intra-group affinity
for undersized groups: move weakest-affinity experts from oversized groups
# U(r) = sum_{C} sum_{i,j in C} A[i,j] / sum_{i<j} A[i,j]
# S(r) = sqrt(1/D * sum (|C_d| - E)^2)
# Select r at knee point of (S(r), U(r))
```

Table 2 验证：controlled non-uniform (r=0.15) 实现 end-to-end 5698ms vs uniform 6328ms vs fully non-uniform 5747ms。Fully non-uniform 通信最优（2826ms All-to-All）但 GPU idle 从 502ms 增至 617ms，controlled 实现最佳平衡。

涉及论文标题：
- GRACE-MoE: Grouping and Replication with Locality-Aware Routing for Efficient Distributed MoE Inference
