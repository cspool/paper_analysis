## Client Drift (客户端漂移) in Federated Learning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Client Drift（客户端漂移）是联邦学习中的核心挑战之一。当各 client 的数据分布为非 i.i.d.（non-identically and independently distributed）时，每个 client 在 local training 中优化的本地损失函数 F_i(w) 与全局损失函数 F(w) = Σ h_i F_i(w) 存在系统性偏差。经过 K 步 local SGD 后，各 client 的 Local Model Parameter (LMP) w_i 向各自局部最优方向偏离，而非向全局最优收敛。Server 聚合这些"漂移"的 LMPU Δw_i 后，得到的 Global Model Parameter (GMP) w_g 远离真正全局最优。形式化：E[Δw_i] 的方差在 non-i.i.d. 设置下显著增大，导致聚合后梯度的有效信噪比降低。FedWSQ 从梯度过滤视角分析 client drift：local gradient ∂L/∂w 包含 (a) 与当前 LMP w̃ 对齐的分量——模型过拟合本地数据的方向，(b) mini-batch 梯度均值分量——biased toward local data distribution。WS 通过双重投影 (I-P_1)(I-P_{w̃}) 过滤掉这两个分量，使梯度仅保留对全局收敛有益的方向。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Client drift 在 FedAvg 中的具体表现（无缓解措施的 baseline）：

```python
# Client i local training at round t
w_i = w_g^{t-1}                    # initialize from GMP
for k = 1 to K:                    # K local SGD steps
    batch = sample(D_i)            # sample from local (non-i.i.d.) data
    g_i = ∇f_i(w_i; batch)         # local gradient
    w_i = w_i - η * g_i            # SGD update
Δw_i = w_i - w_g^{t-1}             # LMPU

# Drift analysis:
# After K steps: w_i ≈ w_g - η Σ_k g_i^{(k)}
# Each g_i^{(k)} is biased: E_Di[g_i] ≠ E_D[g] (global gradient)
# Accumulation: ||w_i - w*|| >> ||w_g - w*|| (deviation from true optimum)
# Aggregation: w_g = Σ h_i w_i → still deviates from w*
```

**Annotations**: Drift 程度 ∝ K（local steps 越多，漂移越严重）× α（Dirichlet 参数越小/数据异质性越高，漂移越严重）。缓解方法：FedProx（近端项约束）、SCAFFOLD（control variate 修正）、FedDyn（动态正则化）、FedWSQ（WS 梯度过滤）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Client drift 是 FL 几乎所有改进方法的 motivation。检测 drift 的方法：监控各 client LMPU 之间的 cosine similarity、梯度方差、或 Hessian top eigenvalue（FedWSQ 使用 loss landscape 分析，较低的 Hessian eigenvalue 表示更平滑的收敛）。缓解 drift 的主流方法分四类：(1) **近端约束**（FedProx）：在 local loss 中加 μ/2·||w - w_g||² 项限制偏离幅度；(2) **Control variate**（SCAFFOLD）：在各 client 维护 control variate c_i 修正梯度方向；(3) **动态正则化**（FedDyn/ACG）：每轮调整正则化强度以对齐 local 和 global 目标；(4) **梯度过滤**（FedWSQ）：通过 WS 的前向/反向投影，在每步 local SGD 中隐式过滤导致 drift 的梯度分量，无需修改 loss 函数或维护额外状态。FedWSQ 的独特之处在于将 drift 缓解与量化通信压缩结合在一个统一框架中。

涉及论文标题：
- FedWSQ Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization
