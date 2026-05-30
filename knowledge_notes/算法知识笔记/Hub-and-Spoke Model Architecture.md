## Hub-and-Spoke Model Architecture

术语解释
Hub-and-Spoke（中心-辐条）架构是 BTS 采用的多模型合并组织模式：一个中心 Hub 模型（通常为 Seed 模型 $m_0$）通过 Stitch Layer 与多个 Spoke Expert 模型（$m_1, ..., m_n$）双向连接，Expert 之间无直接连接。Hub 的最终输出作为 BTS 模型的输出。

术语是什么？
设计原理：Seed 模型在通用数据上预训练，其表示空间与所有 Expert（均从 Seed 初始化）更对齐，因此 Seed 作为 Hub 可更有效地整合来自不同 Expert 的信息。消融实验证实：用 Math Expert 作 Hub 时平均分数从 28.1 降至 26.2（MMLU 35.8→33.9, GSM8K 20.2→15.6, MATH 10.6→5.73）。

Hub-and-Spoke 的替代方案（论文讨论但未采用）包括：
- 全连接：所有 Expert 之间两两连接 → 参数过多
- 仅 Hub→Expert 单向：非交替架构 → Cross-capability 退化（Ru-MGSM 16.0→11.6）

从算法pipeline角度拆解术语。
BTS Hub-and-Spoke 的数据流模式（以 4 Stitch Layer、3 Expert 为例）：

```
Input → [Hub L1-L4 || Expert_i L1-L4]  → Stitch1(Hub→Experts)
     → [Hub L5-L9 || Expert_i L5-L9]    → Stitch2(Experts→Hub)
     → [Hub L10-L14 || Expert_i L10-L14] → Stitch3(Hub→Experts)
     → [Hub L15-L19 || Expert_i L15-L19] → Stitch4(Experts→Hub)
     → Hub.L20 Output → LM Head → token
```

设计选择：最后一个 Stitch Layer 始终为 Experts-into-Hub 类型，确保最终输出来自 Hub（已融合所有 Expert 信息）。

术语一般如何实现？如何使用？
- 适用于 n 个 Expert + 1 个 Seed 模型的全冻结合并场景
- 推理时所有 Expert 均需前向传播（与 MoE 的稀疏激活不同）
- 可推广至其他 Hub 选择（论文验证了 Seed 作为 Hub 最优）
- 适用条件：要求 Hub 模型与 Expert 共享相同架构（层数、维度）

涉及论文标题：
- BTS Harmonizing Specialized Experts into a Generalist LLM

---
