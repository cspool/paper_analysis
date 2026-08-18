## QEC Tree Code（QEC 树码 / 树簇态码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QEC 树码（tree cluster-state code，误差修正动物园收录于 https://errorcorrectionzoo.org/c/tree_cluster）是一类基于树形图态的量子纠错码，用于基于测量的量子计算（MBQC）中的光子丢失（loss）容错。核心思想：用树形纠缠结构冗余编码量子信息，使单个/少数光子丢失可以通过 stabilizer 测量模式被探测和纠正，而不破坏整个逻辑量子比特。树码的关键协议是 Varnava-Browne-Rudolph（PRL 97, 120501, 2006）提出的 loss tolerance：通过"间接 Z 测量"模式——对被擦除量子比特的相邻量子比特做 X 测量、对其余相邻量子比特做 Z 测量——确定性地读出被擦除量子比特的 Z 测量结果（基于 stabilizer X_i∏_{j∈E(i)}Z_j），从而无损移除被擦除的量子比特。Bell 等（PRX Quantum 4, 020328, 2023）系统优化了用于测量基丢失容错的图码。本论文的树编码融合直接受树码启发（引用 [1][7][68]），但把树码从"量子比特制备阶段的丢失容错"推广到"融合操作本身的失败 + 擦除容错"：每个融合分支配备 q_i^a/q_i^b 辅助量子比特以实现擦除时的间接 Z 测量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
树码的容错测量模式（论文 Fig.4(a)-(c) 借用的三条图态测量规则）：
```
# (a) 直接 Z 测量规则：Z 基测量移除目标 qubit 并断开其所有纠缠边
Z_measure(j)  ->  G' = G - {j}  （j 与所有邻居的边断裂）
# (b) 一对 X 测量规则：两个相邻 X 基测量移除 qubits 并在其邻居间建立直接连接
X_measure(j1); X_measure(j2)  ->  N(j1) 与 N(j2) 直接相连
# (c) 间接 Z 测量规则（loss tolerance 核心）：
#     目标 qubit j0 丢失 -> 选邻居 i：X_measure(i)；对 E(i) 中其它 qubit 做 Z_measure
#     stabilizer S_i = X_i ∏_{j∈E(i)} Z_j 确定性揭示 z_{j0}
```
树码在编码参数上给出"分支数 vs 丢失容错"的 trade-off；本论文把 b 分支树嵌入 caterpillar 态（b=4、b_prep=6），把树码的丢失容错思想转化为融合级擦除容错，成功率 S_tree=1-(1-(1-p_eras)^2+p_fail)^b。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：树码以"测量模式调度"实现——不需要额外的量子门，只需在测量阶段按 stabilizer 关系安排 X/Z 测量基（论文用 Perceval FFCircuitProvider 做前馈实现）。参考实现/工具：(1) 误差修正动物园收录树簇态码条目（errorcorrectionzoo.org/c/tree_cluster）；(2) Bell 等的图码优化（PRX Quantum 4, 020328）；(3) Varnava-Browne-Rudolph 的 loss tolerance 协议（PRL 97, 120501）。使用场景：光子 MBQC 中任何存在光子丢失的平台；本论文把它用于 spin memory PQC 的融合擦除容错，并指出同样的 loss-tolerant 逻辑融合思想可推广到 all-photonic 架构（把原融合单元替换为树编码逻辑量子比特即可）。论文未开源。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion
