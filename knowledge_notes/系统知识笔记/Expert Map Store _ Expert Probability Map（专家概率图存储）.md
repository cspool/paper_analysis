## Expert Map Store / Expert Probability Map（专家概率图存储）

术语是什么？
Expert Map Store是FineMoE系统核心数据结构：存储每个inference iteration、每个MoE layer上gate network对所有experts的完整probability distribution（expert probability map），而非仅记录request-level的expert激活计数。Semantic embedding（模型embedding layer输出的prompt语义向量）和expert trajectory（已过层的gate probability sequence）作为检索key。Expert Map Store位于CPU memory（默认1K maps < 200MB，32K maps < 200MB），支持cosine similarity检索最相似历史expert map。

从系统架构角度拆解术语：
```
Expert Map Store 结构：
{
  map_id: {
    semantic_embedding: tensor[d_model],       # prompt的语义向量（embedding layer输出）
    expert_maps: {                              # 每层的expert概率分布
      layer_0: tensor[num_experts],             # gate对所有expert的概率 [0.05, 0.3, 0.02, ...]
      layer_1: tensor[num_experts],
      ...
      layer_L: tensor[num_experts]
    },
    trajectory: tensor[L, num_experts],          # 已过层的gate概率序列（trajectory-based search用）
  }
}

# Search流程：
1. Semantic Search（前d层）：cosine_sim(prompt_embedding, store[i].semantic_embedding) → 取最相似map
2. Trajectory Search（d层之后）：cosine_sim(current_trajectory, store[i].trajectory) → 取最相似map
3. 从检索到的map中提取目标layer的expert probability distribution
4. 按similarity score动态设置delta=clip(1-score,0,1)选择预取expert
```
与MoE-Infinity的Expert Activation Matrix的关键区别：Activation Matrix只记录request级哪些expert被激活（binary/hit count），丢失了iteration级细粒度pattern和概率置信度信息。

术语一般如何实现？
用Python/PyTorch/NumPy实现：ndarray存储semantic embeddings和expert maps，PyTorch tensor做cosine similarity计算。去重机制：store满时计算semantic和trajectory redundancy score，保留覆盖性更好的maps。涉及PyTorch native ops做pairwise similarity和redundancy score。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading
