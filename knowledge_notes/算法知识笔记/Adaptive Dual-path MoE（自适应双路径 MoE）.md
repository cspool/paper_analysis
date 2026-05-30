## Adaptive Dual-path MoE（自适应双路径 MoE）

术语是什么？
Adaptive Dual-path MoE 是 SpheroMoE Layer 中的双分支 expert 结构，利用 Checkpoint Recycling 继承的 dense 先验知识区分重要/非重要 token。Core path 包含少数（约占比 1/3）大型 expert（完整 hidden dim 4d'），处理高重要性 token；Universal path 包含多数（约占比 2/3）小型 expert（hidden dim ≈ d'，约 1/4 参数），处理低重要性 token。两路径在保持 FLOPs 不变的前提下优化计算资源分配。

从算法pipeline角度拆解术语：
SpheroMoE Routing 的 dispatch weights 自然区分了 token 的重要性（通过相似度 logits S 的 softmax 值）。dispatch 后的 slot 数组按重要性排序，前 core_num 个分配给 core experts，剩余分配给 universal experts。流程见 SpheroMoE Layer 术语中的伪代码 Step 5。

术语一般如何实现？如何使用？
在 MoE 层实现中，core_experts 和 univ_experts 是两个独立的 expert group，各有不同的 hidden dim（core: 4d', univ: d' 或更小）。router 输出的 dispatch weights 自然决定 slot 的重要性排序。Core expert 数量占总 expert 数的最优比例为 1/3（由消融实验确定）。

涉及论文标题：
- MoE Jetpack: From Dense Checkpoints to Adaptive Mixture of Experts for Vision Tasks
