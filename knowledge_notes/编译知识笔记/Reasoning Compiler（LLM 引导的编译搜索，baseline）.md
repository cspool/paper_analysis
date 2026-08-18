## Reasoning Compiler（LLM 引导的编译搜索，baseline）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Reasoning Compiler（Georgia Tech / UC San Diego，NeurIPS 2025，commit 1390945）把 LLM 引导的 MCTS 引入深度学习编译：替换 TVM MetaSchedule 的进化式搜索后端，用 LLM 推理引导搜索方向，用于高效模型 serving 的算子优化。QiMeng-Tensify（ISCA'26）把它作为最接近的 exploration-based baseline（同为"LLM + MCTS 编译"路线）：但 Reasoning Compiler 继承 TVM 的专家设计图划分与融合策略（Policy space 受限，Table IV），仍无法在图级跨算子做无约束变换；QiMeng-Tensify 在相同搜索空间下 kernel 质量更高，且编译时间平均减少 3.06×（A100：Reasoning Compiler GatedMLP 4.53h/SelfAtten 5.23h/LoRA 5.19h/QKNorm 4.29h/nTrans 4.81h vs QiMeng-Tensify 1.37/1.83/1.92/1.17/1.69h）。

从编译框架角度拆解术语，比如术语所在编译框架的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
作为 baseline 的运转流程（GatedMLP）：输入计算图 → TVM 的 MetaSchedule 图划分（专家设计，切成预定义子图）→ 用 LLM-guided MCTS 替换进化搜索，在子图内的调度规则空间搜索 → 输出 kernel。局限：图划分与融合策略仍受限（只能优化 TVM 划分出的子图，无法跨子图融合）；论文对比时按其算法在 QiMeng-Tensify 的搜索空间里实现以公平比较（"We compare it by implementing its algorithm in our search space"）。子图级结果：QiMeng-Tensify 平均快 1.31×；NSA 子图上 1.18×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：基于 TVM/MetaSchedule 代码库改造（把 search backend 换成 LLM 引导 MCTS），commit 1390945；论文对比时按其算法在 QiMeng-Tensify 搜索空间实现。使用方式：作为"LLM+MCTS"路线的对照，证明"仅替换搜索算法、不放开图划分/策略空间"仍受限——QiMeng-Tensify 的 MDP 无约束变换 + 架构感知先验适配是额外增益来源。

涉及论文标题：
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
