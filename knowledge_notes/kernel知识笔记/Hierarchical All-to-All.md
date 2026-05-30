## Hierarchical All-to-All

术语解释
分层All-to-All是针对MoE Expert Parallelism的通信优化策略，将全局All-to-All拆分为intra-node（节点内高带宽）和inter-node（节点间低带宽）两个层次，充分利用两级带宽。

术语是什么？
传统All-to-All允许所有GPU直接相互通信，带宽使用效率低。分层All-to-All的关键洞察：
- 同节点GPU通过NVLink/NVSwitch互联，带宽极高（~900GB/s）
- 跨节点GPU通过InfiniBand/RoCE互联，带宽相对较低（~200GB/s）
- 应先在同节点内聚合数据，再跨节点传输，减少跨节点通信量

从kernel调度角度拆解术语。
```
# 分层All-to-All伪代码
def hierarchical_alltoall(tokens, expert_assignment, node_size):
    # tokens: [num_gpus, tokens_per_gpu]
    # node_size: GPUs per node
    
    # 阶段1：Intra-node gather（高带宽NVLink）
    for node in nodes:
        node_buffer = []
        for gpu in node.gpus:
            # 收集该节点内所有GPU需要发往其他节点的token
            for dst_gpu in range(num_gpus):
                if dst_gpu not in node.gpus:
                    node_buffer.append(tokens[gpu][dst_gpu])
    
    # 阶段2：Inter-node exchange（低带宽网络）
    # 每节点仅一个gateway GPU参与跨节点通信
    for src_node in nodes:
        gateway_gpu = src_node.gateway
        for dst_node in nodes:
            if src_node != dst_node:
                NCCL_Send(gateway_gpu, node_buffer, dst_node.gateway)
    
    # 阶段3：Intra-node scatter（高带宽NVLink）
    for node in nodes:
        distribute_received_tokens_to_gpus()
    
    # 加速比：1.4x-2x（ScheMoE vs 标准All-to-All）
```

实现框架：
- Tutel：统一的权重/数据布局支持自适应切换，无需重新格式化
- HetuMoE：针对大规模MoE的分层All-to-All实现
- DeepSpeed-MoE：分层通信 + 同数据路径token合并

术语一般如何实现？如何使用？
- 基于NCCL的send/recv primitives
- 需要配置node topology信息（哪些GPU在同一节点）
- gateway GPU选择策略影响性能
- 与expert放置策略配合使用效果最佳

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

---
