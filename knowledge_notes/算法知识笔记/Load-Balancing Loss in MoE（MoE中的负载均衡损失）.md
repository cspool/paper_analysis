## Load-Balancing Loss in MoE（MoE中的负载均衡损失）

术语是什么？
Load-balancing loss是MoE训练中的辅助损失函数，鼓励gate network将tokens均匀分布到所有expert，防止部分expert过度使用（hot expert）而其他闲置。作为总损失附加项权重较小。虽提升训练稳定性和expert利用率，但对推理时expert offloading有负面影响：均匀路由使expert选择更分散、更难预测，降低了request-level热度统计的prefetch hit rate。

从算法pipeline角度拆解术语：
```
L_lb = α · num_experts · Σ_i(f_i · P_i)
# f_i: expert i被分配的token比例
# P_i: expert i的平均gate probability
# α: 辅助系数 (如0.01)
# L_total = L_main + L_lb
```

术语一般如何实现？
常见于Switch Transformer、GShard、Mixtral等MoE训练。DeepSeek-V3提出auxiliary-loss-free策略用动态bias调整替代。FineMoE指出load-balancing loss使expert使用更均匀但粗粒度统计可预测性下降，因此需要更细粒度的iteration-level expert map预测。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

