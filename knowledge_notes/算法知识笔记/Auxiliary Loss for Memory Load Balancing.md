## Auxiliary Loss for Memory Load Balancing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Auxiliary Loss for Memory Load Balancing 是 MoM 中用于确保 Router 将 token 均匀分配到各 memory state 的辅助训练损失函数。类似 Switch Transformer 中的 load balancing loss，目标是防止某些 memory 被过度激活（"hot memory"问题）而其他 memory 空闲。

MoM 的 auxiliary loss 公式：L_aux = α · Σ_m f_m · P_m，其中 f_m 是路由到 memory m 的 token 比例，P_m 是分配给 memory m 的平均 routing probability，α 是 auxiliary loss 的 scale 系数。最小化该损失鼓励均匀路由。

从算法pipeline角度拆解术语。

MoM 实验（Table 6）测试了不同 α 值的效果：
- α = 1e-2: Recall avg 27.59
- α = 1e-3: Recall avg 28.16 (best)
- α = 0: Recall avg 27.23

结果显示合适的 auxiliary loss weight 能提升性能（过大干扰主任务学习，过小导致负载不均衡）。

术语一般如何实现？如何使用？

实现为训练循环中的额外损失项：total_loss = language_modeling_loss + α · L_aux。MoM 的 Fig 5 热力图验证了施加 auxiliary loss 后各 memory 在各层的路由分布近乎均匀。代码开源：https://github.com/OpenSparseLLMs/MoM。

涉及论文标题：
- MoM: Linear Sequence Modeling with Mixture-of-Memories

---
