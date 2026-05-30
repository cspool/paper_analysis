## Reference Set for Test-Time MoE Adaptation (MoE测试时自适应的参考集)

术语解释
Reference Set 是 C3PO 测试时自适应的数据基础：一个预先准备的数据集，包含模型输出正确的样本及其对应的 expert pathway。测试时，对每个新样本从参考集中检索最相似的 k 个邻居，利用其 successful pathway 来指导 routing weights 的优化。

术语是什么？
参考集构建要求：
- **正确性**: 样本在 base model 上的输出正确（f(x_i, ω_i) = y_i）
- **无重叠**: 参考集与测试 benchmark 不重叠（使用领域相关但不同的数据集），过滤问题相似度 > 0.95 的样本
- **覆盖度**: 覆盖多种任务类型，提供多样化 pathway 模式

C3PO benchmark-参考集配对：MMLU→BIG-Bench+SuperGLUE, HellaSwag/PIQA→CommonsenseQA+SocialIQA, ARC-C/E→OpenBookQA+SciQ, WinoGrande→KnowRef

从算法pipeline角度拆解术语：
```
def build_reference_set(base_model, reference_data, embedding_model):
    ref_set = []
    for (x_i, y_i) in reference_data:
        ω_i = base_model.get_routing_weights(x_i)
        if argmax(base_model.forward(x_i)) == y_i:
            emb_i = embedding_model(task_description(x_i))
            ref_set.append({'x': x_i, 'y': y_i, 'omega': ω_i, 'embedding': emb_i})
    return ref_set
```

术语一般如何实现？如何使用？
- 存储开销: |D_ref| × (|x| + |y| + L×E×4 bytes + |emb|)
- Embedding 模型质量是关键——更好的 embedding → 更相关的邻居 → 更好的优化
- 参考集可跨任务共享，离线构建后序列化存储

涉及论文标题：
- C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing
