## Cross-Capability

术语解释
Cross-capability（交叉能力）是 BTS 论文引入的评估概念，指合并后的模型在多个 Expert 专长领域交集处展现的能力——超越任何单个 Expert 在交集任务上的表现。例如：Russian Expert（专精俄语）+ Math Expert（专精数学）→ 合并模型在 Russian Math（俄语数学题）上表现优于两者。

术语是什么？
论文通过实验定义和验证 Cross-capability：
- **设置**：仅合并 Russian Expert + Math Expert（移除 Code Expert 以避免语言污染），训练 2B tokens Russian Math 数据
- **评估**：Russian MGSM（GSM8K 俄语翻译子集，8-shot）
- **发现**：(1) 无 in-domain 训练数据时，所有合并方法无法产生 cross-capability（Russian MGSM 表现不超 Seed）; (2) 加入少量 in-domain 数据后，BTS 在 Expert Merging 方法中表现最佳（Russian MGSM 16.0 vs BAM Adapters 15.6 vs BTM 9.60）; (3) Expert Upcycling 方法（BAM 18.4, BTX 17.6）因更大训练容量略优于 BTS
- **关键消融**：交替 stitch 架构对 cross-capability 至关重要——全 Experts-into-Hub 架构使 Russian MGSM 从 16.0 降至 11.6

从算法pipeline角度拆解术语。
Cross-capability 的机制解释：Hub-into-Experts stitch layer 允许 Math Expert 的表示受 Russian Expert 信息影响（通过 Hub 作为中介），使得 Math Expert 在处理俄语数学题时能利用 Hub 中融合的俄语理解能力；Experts-into-Hub stitch layer 反之将融合结果回流到 Hub 输出。

术语一般如何实现？如何使用？
- 评估方法：选择两个正交 Expert 领域，构建交集任务的 benchmark（需 in-domain 训练和评估数据）
- 关键条件：需要少量 in-domain 交叉训练数据（无此数据则合并模型与 Seed 无显著差异）
- 适用场景：验证模型合并方法的表达能力和泛化边界

涉及论文标题：
- BTS Harmonizing Specialized Experts into a Generalist LLM

---
