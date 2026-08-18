## 平衡二叉树分层生成（BBT Hierarchical Generation，图态分层生成）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
平衡二叉树分层生成（balanced binary tree hierarchical generation）是 MemTree 编译框架采用的图态生成策略：把目标图态的生成过程建模为一棵平衡二叉树（BBT），树的中叶子是树编码逻辑量子比特组成的线性图，每个内部节点是两个子图经融合合并成的更大图态；同一层（同深度）的所有融合在一个时间步内并行执行，时间步从叶子层顺序推进到根（目标图态）。设计动机：naive 方案是在一个时间步内直接施加所有融合，但融合有错误——即使融合成功率极高（S_fusion=0.99），生成 100-qubit VQE 目标态需要 k>1000 次融合，整体成功率 S^1000~1e-5，不可行；分层方案中任意融合（BBT 的一个节点）失败只需恢复以该节点为根的子树。生成开销上界由临界路径（critical path，BBT 中从叶到根融合总数最大的一条路径）决定，因此编译优化的目标是最小化临界路径。BBT 保持平衡（两子树基数差距受约束）以最小化树高，同时下层的划分尽量少切边（更可能落在临界路径上）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
BBT 分层生成在编译框架中的运转流程（Fig.6(a)(c)）：
```
# 生成树构建（编译期）
root = G^l                          # 全部线性子图的集合
while not all leaf:
    (G^l_j, G^l_k) = min_cut_balanced_divide(G^l_i)   # MIP-2, 平衡 + 最小切割
# 时间方向执行（运行期, Fig.6(c) 流水线）
for t in time_steps:                # 自底向上按 BBT 层次
    for each node at layer t:       # 同一层并行
        sub = fusion(children_subgraphs)
        if fusion_failed(sub):      # 红箭头
            delay_sibling_to_next_step(sub)   # 绿箭头: 兄弟延迟
            # 下个时间步重试该子树（蓝箭头）
# 目标: 在有限时间周期内最大化成功生成的目标态数量(shots)
```
例子：36-qubit VQE 的图态被划分为若干线性子图后，经 BBT 约 log2(#subgraphs) 层融合完成；任一层失败只重做该子树，而 naively 全部融合在 1000+ 次融合后几乎必然失败。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：BBT 构建由第二个 MIP 模型实现（Gurobi 求解；变量 y_g^l 表示子图归属，平衡约束 |G^l_j|,|G^l_k|≥2^⌊log2|G^l_i|⌋，目标 min L=Σ|y_{g1}-y_{g2}|）；执行侧在模拟器/硬件流水线中按时间步驱动（融合失败 → feed-forward 延迟调度）。使用场景：任何"融合成功率 <1 且融合次数多"的图态生成问题；是 MemTree 相对 OneAdapt 执行时间指数级下降的关键机制之一（配合树编码融合的高单次成功率）。论文未开源。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion
