## Block Coordinate Descent (BCD) for MoE Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block Coordinate Descent (BCD) 是一种用于大规模非凸优化的迭代算法，核心思想是每次迭代仅更新一个参数块（block），保持其他参数固定，从而大幅降低单步内存和计算开销。在 MoE 训练语境下，MoE-DisCo 将 BCD 应用于 expert 级：每个 training step 仅更新一个 expert 及其共享 backbone，其余 expert 全部冻结。每次训练仅需维护等价于单 expert 分支的 dense 子模型，内存需求从 O(E) 降至 O(1)，使 MoE 训练可在低内存设备（如 24GB RTX 4090）上进行。各 expert 的训练完全并行、零通信开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 传统 MoE 训练：全部参数同时加载和更新
Θ = (θ_shared, θ_1, ..., θ_E)    # O(E) 内存
for batch in D:
    loss = M(Θ, batch).forward()  # 前向经过 gating + Top-K experts
    loss.backward()               # 反向遍历所有 expert 路径
    optimizer.step()              # 全部参数同时更新

# BCD for MoE (MoE-DisCo)：分块独立训练
for k in 1..E:                    # 可完全并行，零通信
    Θ_k = (θ_shared^(k), θ_k)    # 仅 1/E expert + 1 份共享参数
    for batch in D_k:
        loss = M(Θ_k, batch).forward()  # 无 gating，固定 single expert
        loss.backward()                 # 反向仅遍历 expert k
        optimizer.step()               # 仅更新 Θ_k
```

训练阶段移除 gating 机制，子模型退化为标准 dense Transformer。关键优势：(1) 子模型参数量远小于完整 MoE；(2) 无跨设备通信开销（无 gradient/parameter 交换）；(3) 每个子模型可独立放入低成本 GPU。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 MoE-DisCo 中，BCD 通过四阶段 pipeline 落地：(1) Model Decoupling：完整 MoE 分解为 E 个 dense 子模型；(2) Data Decoupling：K-Means 聚类产生 E 个语义区分的数据子集；(3) Independent Parallel Training：各子模型在低成本 GPU 上独立训练；(4) Reintegration + Fine-Tune：expert 直接拼接，共享参数加权平均（WP-SGD），短期全局微调恢复 gating 协调性。实验验证：Qwen1.5-MoE-2.7B 上，BCD 策略将 69.5% 训练成本从 A100 移至 RTX 4090，且最终 PPL 和 downstream 性能不降反升。

涉及论文标题：
- MoE-DisCo: Low Economy Cost Training Mixture-of-Experts Models
