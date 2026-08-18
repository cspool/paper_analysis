## 分布式 Shampoo 优化器（Distributed Shampoo Optimizer）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Shampoo 是预条件随机张量优化器（Gupta et al., ICML 2018）：对每个参数张量按各维计算 Kronecker 因子（G_t = G_t + g g^T 的逐维累积），用矩阵逆根（Kronecker-factored preconditioner）做预条件更新，比 Adam 收敛更快但需存储矩阵平方根/逆根（内存与计算开销大）。分布式 Shampoo（Shi et al., arXiv:2309.06497）是 PyTorch 的分布式数据并行实现：优化阶段需 AllGather 交换预条件器/统计量。MTIA 300（ISCA'26）的生产 DLRM 训练用分布式 Shampoo 做 dense 组件优化（稀疏侧用 TBE/稀疏优化器），通信画像中 AllGather 入站 2.1 GB 即来自 Shampoo 优化阶段。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
一次 Shampoo 更新步骤的算法流程（MTIA 300 训练）：
```python
# 每参数张量 W ∈ R^{m×n}（如 dense 层权重）:
# 1. 统计累积: L_t = L_{t-1} + G W^T;  R_t = R_{t-1} + G^T W   (G 为梯度)
# 2. 预条件: L^{-1/4}, R^{-1/4}（矩阵特征分解/逆根）
#    → MTIA 300: 特征分解 offload 到 host CPU（1:1 架构，保数值精度;
#       若 1:8 或片上实现会损失 7.8%）
# 3. 更新: W_{t+1} = W_t - η · L^{-1/4} G R^{-1/4}
# 4. 分布式: 每步需 AllGather 预条件器/参数分片（2.1 GB 入站, 40 卡）
```
MTIA 300 上 Shampoo 特征分解是 host CPU offload 的代表算子（论文 3 项 co-design 之一）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：分布式 Shampoo 开源（PyTorch 参考实现，https://arxiv.org/abs/2309.06497）；MTIA 300 上特征分解 offload host CPU（1:1 host:加速器支撑）；H100 用 cuSOLVER 在 GPU 上算（8:1 时无碍）。使用场景：DLRM dense 组件训练（与稀疏侧 TBE 优化器并存）；通信开销（AllGather 2.1 GB）被 MTIA 300 的 ME/NMC 卸载消化。信息缺口：论文未给出 Shampoo 的预条件器更新频率与精度策略细节。

涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
