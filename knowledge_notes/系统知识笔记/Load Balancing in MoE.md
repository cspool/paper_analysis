## Load Balancing in MoE

术语解释
MoE负载均衡是确保分布式MoE推理中所有设备的工作负载均匀分布的技术，防止某些设备的expert因接收过多token而成为瓶颈，导致其他设备闲置。

术语是什么？
负载不均衡的根源：Router可能倾向于将大量token路由到少数"热门"expert，使得持有这些expert的设备过载。
- **辅助损失**：训练时添加负载均衡损失项（如Switch Transformer的auxiliary loss），使router均匀分配token
- **Hash-based路由**：使用特殊哈希函数替代学习型router，天然保证负载均衡
- **Expert放置优化**：Prophet构建负载均衡性能模型 + 贪心搜索最优expert放置；MoE-Prediction使用经典预测算法预测expert负载比例
- **Expert复制**：Lazarus为热门expert分配更多副本到不同设备；FlexMoE细粒度复制特定heavy expert并动态调整expert-to-device映射
- **Token重分配**：BaseLayers将token-to-expert分配形式化为线性分配问题保证公平分配；MoE-ECR让expert选择top-k token而非token选择expert
- **动态batch**：Lynx在batch推理中减少激活expert数量

从系统架构角度拆解术语。
以Prophet的负载均衡为例：
```
# 阶段1：性能建模
for each expert placement configuration:
    # 估计每个设备的执行时间
    for device d:
        T_d = T_compute(experts_on_d, tokens_to_d) + 
              T_communication(tokens_cross_device)

# 阶段2：贪心搜索最优放置
best_placement = None
best_max_time = INF
for each candidate placement:
    max_device_time = max(T_d for d in devices)
    if max_device_time < best_max_time:
        best_placement = candidate
        best_max_time = max_device_time

# 结果：1.75x-12.06x负载均衡改善（vs FasterMoE）
```

术语一般如何实现？如何使用？
- 训练时：辅助损失 + capacity factor限制 = 确保训练稳定性
- 推理时：静态度量（profile-based）+ 动态调整（runtime rebalancing）
- 与expert并行度紧密耦合——更大的EP度数意味着对负载不均衡更敏感
- 关键指标：expert利用率方差、设备空闲时间占比

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- Accelerating MoE Model Inference with Expert Sharding
- Capacity-Aware Inference Mitigating the Straggler Effect in Mixture of Experts

**MoEShard 的 Perfect Load Balancing 方法**：
MoEShard 通过 expert tensor sharding 实现 perfect load balancing——所有 GPU 处理完全相同数量的计算（全部 token × 全部 expert 的 1/|G| shard），无论 routing 分布如何倾斜。与现有方法的根本区别：
- 辅助损失/CF：训练时约束 routing，推理时可能丢 token
- Expert 复制（Lazarus/Prophet）：需要 profiling + 额外 GPU 内存
- Token 重分配（EC routing）：改变 routing 机制
- **MoEShard**：不改 routing 机制，不丢 token，不复制 expert，通过"计算全均匀化"实现负载均衡——将不可控的 routing skew 问题转化为可控的均匀张量计算问题
- **Capacity-Aware Inference**：在推理时通过 Expert Capacity 约束 max(N_i) ≤ γN̄ 来限制 straggler expert 负载；用 Expanded Drop 提升低负载 expert 利用率。属于 token 重分配类方案，但不改 router 训练——仅在推理时通过容量约束和候选扩展做 token-to-expert 重调度。本质是将 routing skew 转化为可控的延迟上限

---
