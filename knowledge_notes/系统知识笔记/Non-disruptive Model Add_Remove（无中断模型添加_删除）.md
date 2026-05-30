## Non-disruptive Model Add/Remove（无中断模型添加/删除）

术语是什么？
Non-disruptive Model Add/Remove 是 MoEsaic 支持的动态模型实例管理能力：在 MoEsaic serving 过程中，可以添加新的 model instance（含其 experts 和 gates）或移除已有 instance，无需系统重启。原因是 MoEsaic 的 expert deduplication 使用 in-memory hash dictionary 进行动态查找——新 expert 加载时计算 hash 并与现有 dictionary 比对，命中的共享已有 tensor，未命中的分配新显存。独立 expert 表示（Independent Expert Representation）使增量添加/删除成为可能。

从系统架构角度拆解术语：
Non-disruptive Add/Remove 的工作约束和流程：

1. **添加新 model instance**：
   - 新 client 提交其 MoE model（experts + gates）。
   - MoEsaic 逐一加载新 expert，每 expert 完全填充后计算 hash → 查 dictionary。
   - 命中 → 引用已有 tensor；未命中 → 分配新 GPU 显存。
   - 新 gate 合并到 fused gate（更新 gate mapping table）。
   - 完成后 model instance 上线服务。
2. **移除已有 model instance**：
   - 标记目标 model instance 为待移除。
   - 等待该 instance 的所有 in-flight 请求完成。
   - 释放该 instance 独占的 non-MoE 参数（如 attention）和 unique experts（仅该 instance 使用的）。
   - 共享 experts 不释放（其他 instance 仍在使用），仅减少引用计数。
   - 从 fused gate 和 gate mapping table 中移除对应条目。
3. **关键约束**：添加/删除操作不可在活跃推理期间执行——MoEsaic 结构在集成期间暂时未定义。论文将此描述为"temporarily undefined structure of MoEsaic during integration"。

术语一般如何实现？如何使用？
- 实现依赖于 Independent Expert Representation（每个 expert 独立 nn.Parameter）和 in-memory hash dictionary 的持久维护。
- 与 S-LoRA 的 dynamic adapter loading 类似，但操作粒度不同——S-LoRA 管理 LoRA adapter，MoEsaic 管理完整 expert 参数。
- 服务提供商场景尤为关键：client churn（客户上下线）无需中断整个平台服务。论文指出加载全部 model instances 可能需要几十秒到几分钟，逐个重启不可接受。
- 局限：论文未详细说明"引用计数"机制的具体实现和 shared expert 的并发安全释放策略。

涉及论文标题：
- MoEsaic: Shared Mixture of Experts
