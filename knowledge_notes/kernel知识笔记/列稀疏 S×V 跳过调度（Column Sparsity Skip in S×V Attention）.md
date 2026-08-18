## 列稀疏 S×V 跳过调度（Column Sparsity Skip in S×V Attention）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
action 模态的 attention score 矩阵具有列稀疏性：注意力集中在少数列，约 52% 的列近零（论文在机器人任务集上统计的平均列稀疏度）。与 token 剪枝不同（action token 数仅 10^1–10^2，剪枝空间小），列稀疏跳过直接针对 S×V 计算：近零列对输出贡献可忽略，跳过对应列的多头矩阵乘，并进一步旁路产生这些列的 V 投影计算。这是 DiTPA 模型级冗余消除在运行时 kernel 层面的执行形态。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
运行时调度伪代码（对应 multimodal scheduler 的 column sparsity controller）：
```
for head h in heads:                          # 逐注意力头并行
    col_mask[h] = OR_over_rows(|S[h][:, c]| < eps for c in cols)   # 列比较器 + bit 寄存器 + OR 归约
queue[h] = [c for c in cols if col_mask[h][c] == 0]   # 非零列索引入队
# dispatcher：均衡各头负载
for h in heads:
    queue[h] = rebalance(queue, avg(len(queue[longest]), len(queue[shortest])))
for c in queue[h]:                            # 按队列顺序出列计算
    O[:, c] += S[:, c] * V[c, :]              # 仅非稀疏列参与 S×V
# 稀疏列对应行在 V 投影阶段直接旁路（如最后一个 head 的 V projection 跳过）
```
关键点：零列判定发生在 attention score 计算后、SoftMax 与 V 相乘前，因此 SoftMax 输出零列与 V 对应行一同省略；V 投影旁路属于跨算子级联收益（列稀疏从 S×V 反向传播到 V 的权重矩阵行）。负载均衡：不同头稀疏分布不同会拉长 straggler，dispatcher 平均最长/最短序列实现均衡。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：硬件列比较器直接对 S 的每个元素做阈值比较（近零判定），结果以 bit 位图存储，OR 树归约出列掩码，索引队列 + dispatcher 完成调度——全部在 multimodal scheduler 内完成，数据操作时延可忽略（GPU 端对应开销占 35.4%）。使用：与校准多模态近似组合后消除 91.74% 冗余 token 计算、其中动作模态贡献列稀疏跳过部分；能效维度 DRAM 权重访问保持不变时（从 16.67% 升至 67.68% 总能耗），列跳过是继续压缩片上计算的关键。通用性：列稀疏模式在 vision/language token 上不存在（其冗余来自重复而非稀疏），因此该调度仅施加于 action 模态注意力。

涉及论文标题：
- DiTPA A DiT-based Action Planner Accelerator Exploiting Action–Denoising–Multimodality Redundancy for Embodied Artificial Intelligence
