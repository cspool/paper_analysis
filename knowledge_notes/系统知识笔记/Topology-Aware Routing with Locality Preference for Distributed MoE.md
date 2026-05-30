## Topology-Aware Routing with Locality Preference for Distributed MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Topology-Aware Routing with Locality Preference (TAR) 是 GRACE-MoE 的在线路由策略，用于在多副本 distributed expert 场景下为每个 token 选择最优 expert 实例。在分布式集群中通信开销 hierarchy：intra-GPU < intra-node NVLink < cross-node Ethernet。TAR 三级优先：(i) 同 GPU 副本（零跨设备通信）；(ii) 同节点内其他 GPU 副本（NVLink）；(iii) 跨节点 fallback（Ethernet）。每级内使用 Weighted Round-Robin with Load Prediction——weights ∝ 1/W'（post-replication predicted GPU load）。TAR 相比纯 WRR 减少 All-to-All time 9.47%（OLMoE）、cross-node traffic 12.12%，GPU idle 仅增加 2.58%（trade-off 可接受）。

从系统架构角度拆解：

```
Algorithm 4 (Topology-Aware Routing):
  local_gpu_replicas = {g in replica_gpus | g == token_gpu}
  local_node_replicas = {g in replica_gpus | Node(g) == Node(token)}
  if local_gpu_replicas: → token_gpu (no comm)
  elif local_node_replicas: → WRR(local_node_replicas, predicted_loads)
  else: → WRR(all_replica_gpus, predicted_loads)
# per-tier WRR: ChooseByPollingWeight(local_weights)
# prevents all tokens going to same nearest replica (new hotspot)
```

需配合 Hierarchical Sparse Communication (HSC)：TAR 决定"去哪"，HSC 决定"怎么传"。需维护 per-layer per-expert replica map 和 GPU-node topology map。

涉及论文标题：
- GRACE-MoE: Grouping and Replication with Locality-Aware Routing for Efficient Distributed MoE Inference
