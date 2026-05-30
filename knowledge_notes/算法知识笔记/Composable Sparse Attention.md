## Composable Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Composable Sparse Attention 是 Focus 论文定义的一种新的高效注意力范式：稀疏注意力方法可以**叠加到任何预训练模型上**（类似插件），仅训练稀疏路由参数，所有原始权重保持冻结，且叠加后不退化任何下游 benchmark。这是与其他高效注意力方法的关键区别——结构化稀疏（Longformer）固定 pattern 无法适配、近似方法（Performer）误差累积、token selection（SparQ）退化 PPL 5-10 点——而 composable 方法要求零 benchmark 退化 + 改善或匹配 PPL。Focus 实现了这一性质的核心原因是 routing-attention separation：centroid 仅决定路由，预训练 QKV 注意力完整保留，因此模型不会"忘记"任何预训练知识。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Composability 的技术基础：
```
原始预训练模型: h → Q(h), K(h), V(h) → softmax(QK^T/√d)·V
Composable Focus: 
  h → W_g·h · C^T → Sinkhorn → g (group assignment, 仅 148K 新参数)
  h → Q(h), K(h), V(h) (原权重, 冻结)
  → group-gated attention = softmax(QK^T · gate(g))·V
```

Composability 的三层保证：
1. **权重冻结**：centroid 训练不修改 Q/K/V/O 权重，不破坏预训练表征
2. **精确 softmax 保留**：同组内仍使用标准 softmax（无近似、无重归一化），预训练计算模式不变
3. **Routing-content 分离**：路由（centroid）与内容（QKV attention）独立，互不干扰

验证范围：GPT-2 124M/774M、Mistral 7B、Qwen2.5 7B、OLMo-27B、LLaMA-2 13B/70B 七种模型五种 attention 架构，all benchmark 零退化（最差 -0.3%，噪声范围内）。

与 LoRA 对比：LoRA 修改权重矩阵（ΔW=AB），在相同参数预算下（147K vs 148K）退化所有 benchmark；Focus 仅添加路由不修改任何原权重，零退化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Composable 部署流程：
1. 加载任意预训练 HuggingFace 模型
2. 对每个 attention 层插入 centroid 参数（C, W_g）
3. 训练：仅 centroid 参数更新（AdamW, 4000 steps），所有原权重 requires_grad=False
4. 推理：使用 hard top-k assignment + FlashAttention 分解，获得加速
5. 跨模型可迁移：GPT-2 上训练的 centroid 思想可直接应用于 Mistral/Llama/Qwen，仅需重新训练 centroid 参数

限制：
- 训练时无加速（soft gate 计算全 O(n²)）
- ≤4K token 时路由开销（sort ~12ms）抵消加速
- 大模型（≥7B）上 PPL 收益递减（从 +1.1 降至 -0.7）

涉及论文标题：
- Why Attend to Everything? Focus is the Key (Composing Sparse Attention via Learned Grouping)
