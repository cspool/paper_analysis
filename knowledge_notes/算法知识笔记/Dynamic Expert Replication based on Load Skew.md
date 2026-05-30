## Dynamic Expert Replication based on Load Skew

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Dynamic Expert Replication (DR) 是 GRACE-MoE 的自适应 expert 复制策略，用于补偿 affinity-based grouping 的计算负载倾斜。定义 load skew factor $\rho = W_{\max} / \bar{W}$，由 $n_{\text{replica}} = \min(\max(1, \lfloor \rho \rfloor), n_{\text{gpu}} - 1)$ 确定副本数。仅复制 heaviest group 中最热的 expert（cumulative load > $W_{\max} \cdot n_{\text{replica}}/(1+n_{\text{replica}})$），作为 secondary copies 放置到最少负载的 GPU。对比 fixed replication（1 replica always）：GPU idle 仅 −1.59%；DR：−19.71%，且 GPU load std 从 +90.03%（HG only）降至 +31.92%（HG+DR+WRR）。

从算法pipeline角度拆解：

```
rho = W_max / W_mean; n_replica = min(max(1, floor(rho)), n_gpu - 1)
# In heaviest group, sort experts by load, select hot experts:
#   cumulative_load > W_max * n_replica / (1 + n_replica)
# Post-replication load prediction (for routing weights):
W_p = W_max / (n_replica + 1)  # evenly split assumption
W'_max = W_max - W_r + W_p; W'_i = W_i + W_p
# routing weights ∝ 1/W' (inverse proportional)
```

涉及论文标题：
- GRACE-MoE: Grouping and Replication with Locality-Aware Routing for Efficient Distributed MoE Inference
