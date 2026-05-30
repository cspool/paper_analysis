## RBD (Redundancy-Bypassing Dispatch / 冗余绕行调度)

术语是什么？

RBD 是 X-MoE 提出的分层两级 token dispatch 算法，旨在消除 expert-specialized MoE（large top-k 路由）在层次化 HPC 网络（如 Dragonfly 拓扑）上的跨节点通信冗余。当 top-k 较大时，同一 token 可能被路由到多个位于同一目标节点的 expert，传统方法会通过跨节点链路重复发送同一 token 的多个副本。RBD 将 token 分为 Pilot tokens（跨节点去重后的最小集合）和 Local replica（节点内重复 token），仅 Pilot tokens 走慢速跨节点链路，Local replica 在目标节点从已到达的 Pilot 重建并通过快速节点内链路分发。

术语是什么？

RBD 执行流程（三阶段）：

```
# Stage 0 (S0): Pilot Selection
for each token's k destinations:
    group experts by destination node
    for each (src_node, dst_node) group:
        randomly select 1 → Pilot token
        mark rest → Local replica
# 构建 s1_mapping_indices: local_replica -> pilot_token 的相对索引映射
# 相对索引从0开始，按expert排序后转为绝对索引

# Stage 1 (S1): Inter-Node Exchange (Pilot Only)
pilot_tokens = gather_kernel(x, pilot_token_ids)
pilot_tokens = uneven_alltoallv(pilot_tokens)  # 跨节点，仅pilot
# 同时传输 local replica 的轻量元数据（ERI-arrays + mapping_indices）
# Local replica 在目标节点从 pilot 重建:
local_replica_buffer[i] = pilot_tokens[s1_mapping_indices[i]]

# Stage 2 (S2): Intra-Node Exchange (Local Replica Only)
local_replica = intra_node_uneven_alltoallv(local_replica_buffer)  # 节点内高速
# Merge: 合并 pilot + local replica，按 expert index 排序
```

从系统架构角度拆解：

RBD 的核心洞察是 Dragonfly 等 HPC 拓扑存在显著的 intra-node vs inter-node 带宽不对称（Infinity Fabric 200 GB/s vs Slingshot 25 GB/s in Frontier，约 8:1）。传统 alltoall 将所有 GPU 平等对待，将同一 token 的多份拷贝都通过跨节点链路发送。RBD 通过拓扑感知的分层调度将通信量从慢速链路卸载到快速链路。

X-MoE 在 32 GPU (EP=32)、54.8% 冗余率的场景下，RBD 减少跨节点通信时间 52.5%，总体 dispatch 加速 1.55×。

术语一般如何实现？

RBD 的 combine 阶段是 dispatch 的逆过程：Local replica 先通过 intra-node alltoall 聚合到 pilot token，然后 pilot token 通过 inter-node alltoall 返回原始设备。Combine weights 的缩放必须在 Stage 1（合并 local replica 之前）完成以保证正确性。

RBD 的 randomized pilot selection 策略（而非总是选最小 expert ID）是为了避免 alltoall 负载倾斜。

涉及论文标题：
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms
