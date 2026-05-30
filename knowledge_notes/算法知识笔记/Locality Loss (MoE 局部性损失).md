## Locality Loss (MoE 局部性损失)

术语解释
Locality Loss 是 LocMoE/ETR 系列提出的负载均衡辅助损失，在 traditional auxiliary loss 基础上引入数据局部性约束：通过 KL 散度惩罚 token 被路由到非本地节点 expert，鼓励同节点路由，减少跨节点 All-to-All 通信。

术语是什么：
L_loc = μ · KL(D_c || D_l)，D_c 为当前 token-to-expert 分配的经验分布，D_l 为全局部化分布 (所有 token 仅分配至本地节点 expert)。当 expert 数 ≥ 节点数时效果最优——每个节点至少有一个 expert。论文验证: 32N/64N 配置下 locality loss 加速效果最显著。

术语一般如何实现？如何使用？
在 MindSpeed-LLM 中作为附加 loss 项实现：根据 expert-to-node 映射表构建 D_l 分布，计算 KL 散度加到总 loss。Locality Loss 仅影响路由决策 (gradient 通过 router 反向传播)，不影响 expert FFN 权重。

涉及论文标题：
- Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection
