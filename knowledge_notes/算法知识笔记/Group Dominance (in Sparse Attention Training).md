## Group Dominance (in Sparse Attention Training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Group Dominance 是 Focus 论文发现的一种训练不稳定性：当使用 softmax 归一化训练 centroid-based 稀疏注意力时，一个 group 在约 600 步内吸收所有 token，导致稀疏性崩溃——Focus 退化为昂贵的 full attention。这类似于 Mixture of Experts 中的 expert collapse / load imbalance 问题（Fedus et al., 2022），但发生在注意力路由而非 FFN 路由中。论文识别出三条独立 escape pathway：(A) centroid drift——LM 梯度推动 centroid 漂移使所有 token 匹配同一 centroid；(B) representational bypass——即使 centroid 冻结，hidden states 也会向同一 centroid 方向偏移；(C) projection bypass——即使 centroids 和 hidden states 都被约束，W_g 投影也会学习将所有 token 映射到同一方向。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Group dominance 的根本原因：full attention 始终最小化 training loss（因模型可访问所有 token），因此梯度总是推动移除注意力限制。这形成了一个悖论——稀疏注意力在推理时提升质量，但训练时梯度推动消除稀疏性。

三种 escape pathway 的关系：
```
       训练前的均衡状态 (K=8, 每组 ~12.5%)
                |
       ┌───────┼───────┬──────────────┐
       v       v       v              v
   Path A   Path B   Path C         Combined
   centroid hidden    W_g            full FT
   drift    shift     collapse       all active
       |       |       |              |
       v       v       v              v
   1 group absorbs all tokens → sparsity lost
```

Dominance 度量：最大 group 中的 token 占比。K=8 时完美均衡 = 12.5%，collapse = 100%。

论文尝试的缓解方法及其失败原因：
- **Entropy + balance loss**：仅处理 Path A，第 600 步 collapse
- **Stop-gradient on inputs**：阻断 Path B 但不阻断 A/C
- **EMA centroids + detached projection**：阻断 A 但 projection 抹除结构 (Path C)
- **Periodic reclustering (每 100 步)**：周期性重置平衡但 group 不稳定
- **Balance weight ×5**：8 个 group 中 6 个死亡
- **Sinkhorn (论文方案)**：同时阻断三条路径——即使 centroids 漂移/representations 偏移/projection collapse，Sinkhorn 迭代强制重新分布

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Group dominance 的检测与防御：
- 监控每个 group 的 token 占比，发现某个 group 持续超过 30-40%（K=8 时代）即为 dominance 信号
- Sinkhorn 归一化作为结构性约束（非软损失），在每次前向传播中强制执行均衡分组
- Sinkhorn 在 full fine-tuning（最严峻的测试）中保持 15.9% dominance（K=8，完美 = 12.5%），而 softmax 方案 collapse 到 99.4%
- Sinkhorn 对超参稳健：16 种配置下 fine-tuned PPL 波动仅 0.6

涉及论文标题：
- Why Attend to Everything? Focus is the Key (Composing Sparse Attention via Learned Grouping)
