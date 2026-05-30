## Maximum Routing Imbalance (MRI, 最大路由不均衡度)

术语解释
Maximum Routing Imbalance (MRI) 是本文提出的 MoE 负载均衡度量指标：在给定 MoE 层 j 和 batch B 中，路由到单个 expert 的最大 token 占比。MRI 是 MoE 推理最坏情况延迟的代理（proxy）：由于 expert parallelism 下每个 accelerator 的计算量正比于其上的 expert 负载，MRI 越高意味着最忙的 accelerator 延迟越大，硬件利用率越低。

术语是什么？
对于 MoE 层 j，训练迭代步 t，MRI 定义为：
$$MRI(t, j) = \max_{i \in [1,\dots,E]} \left[ \frac{\sum_{x \in B} \mathbb{1}\{i \in I_k(x)\}}{|B|} \right]$$

其中 E 是 routed experts 数量，B 是 batch 中所有 token 集合，I_k(x) 是 token x 的 top-k 选中 expert 索引集合，𝟙 是指示函数。

MRI 取值范围：[1/E, 1]（从完美均衡到单 expert 垄断）。MRI 越高 → 最坏情况延迟越大 → 硬件利用率越低。

与延迟的关系：MRI 不直接报告延迟，而是作为延迟模型的输入。相比硬件和实现特定的延迟指标，MRI 是跨部署场景可比较的行为指标。

从算法pipeline角度拆解术语：
```python
def compute_mri_layer(layer, batch_tokens):
    """Compute MRI for a single MoE layer"""
    E = layer.num_experts
    expert_loads = zeros(E)
    for token in batch_tokens:
        topk_indices = layer.route(token)  # top-k selected experts
        for idx in topk_indices:
            expert_loads[idx] += 1
    expert_loads /= len(batch_tokens)  # normalize to proportions
    return expert_loads.max()          # MRI for this layer

# Layer-wise MRI analysis (as in paper):
def compute_mri_model(model, test_dataset, num_tokens=20_000_000):
    """Compute per-layer MRI on test set"""
    mri_per_layer = defaultdict(list)
    for batch in test_dataset.iterate(num_tokens):
        for j, layer in enumerate(model.moe_layers):
            mri_per_layer[j].append(compute_mri_layer(layer, batch))
    return {j: median(mris) for j, mris in mri_per_layer.items()}
```

术语一般如何实现？如何使用？
- **训练中监控**：MRI 在训练过程中实时计算（per step/per layer），用于检测分布偏移时的路由崩溃
- **最终 checkpoint 评估**：在 20M token 测试集上计算 per-layer MRI，评估最终模型的负载均衡质量
- **OOD 检测**：MRI 在 out-of-distribution 数据上显著升高（如非 German 模型在 German 数据上的 MRI），可作为分布偏移检测信号
- **Early layer 关注**：Switch MoE 中 early layers (0-6) 的 MRI 始终最高，是推理延迟的瓶颈层
- **推理预估算**：MRI × accelerator_count × expert_compute_time ≈ 最坏情况推理延迟（作为延迟模型的简化输入）

涉及论文标题：
- Continual Pre-training of MoEs How robust is your router

---
