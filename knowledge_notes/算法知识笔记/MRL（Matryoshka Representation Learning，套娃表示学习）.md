## MRL（Matryoshka Representation Learning，套娃表示学习）

术语解释
- MRL 是一种表示学习方法：用一个模型同时训练多个嵌套维度（如 768→512→256→128→64 维前缀均可用），使 embedding 的不同前缀保留不同粒度的语义信息，推理时可动态截断维度适配计算/带宽约束，无需额外推理成本。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MRL（Kusupati et al., NeurIPS 2022）在损失中加入嵌套维度的 softmax 损失，使向量前缀自相似地编码信息，可在 25-50% 维度处保持检索质量、全库检索提速达 14×（网络来源）；截断非 MRL 模型会无警告地掉 recall（"recall cliff"），MRL 训练则把下降点推迟到模型特定 knee 点之后。论文把 MRL 作为两阶段渐进式 ANN 的 reduced 向量来源之一（原生支持多分辨率向量），在 MRL 生成的 MS MARCO、20 Newsgroups、DBpedia 语料上验证 recall>98%。
- 从算法pipeline角度拆解术语：MRL 在训练 pipeline 中一次性产出多分辨率表示——embedding 的 512B 前缀作为 reduced 向量（粗筛用）、4KB 全长作为 full 向量（精排用），两者来自同一模型、无需额外推理；推理 pipeline 中粗筛阶段只取前缀、精排阶段取全长，构成论文 ANN 案例的向量供给侧。与 PCA/随机投影（需额外线性层）相比，MRL 无需变换层。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：训练时把同一 batch 的 logits 在多个嵌套维度分别计算损失并加权求和（MatryoshkaLoss，Sentence-Transformers 等提供实现）；推理时按需截断向量前缀。论文用途：在 SSD-resident 场景中 MRL 使"一份存储、两档精度"成为可能，与 Storage-Next 的 512B 高 IOPS 配合。论文仅在案例研究中使用 MRL 生成语料并报告 recall，未展开 MRL 训练细节。

涉及论文标题：
- Five-Minute Rule 40 Years Later A First-Principles Revisit for Modern Memory Hierarchy
