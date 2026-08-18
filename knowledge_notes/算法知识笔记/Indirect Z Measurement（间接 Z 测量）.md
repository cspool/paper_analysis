## Indirect Z Measurement（间接 Z 测量）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
间接 Z 测量（indirect Z measurement）是图态测量的一种容错规则，源自 QEC 树码（Varnava-Browne-Rudolph 2006 的 loss tolerance 协议）与图态 stabilizer 结构：若目标量子比特 j_0 因光子丢失（擦除）无法直接测量，可以选一个与 j_0 相邻的量子比特 i，对它做 X 测量，再对连接到 i 的其它所有量子比特 j_1, j_2,... 做 Z 测量；基于图态 stabilizer 生成元 X_i∏_{j∈E(i)}Z_j，这组测量确定性地揭示"若对 j_0 直接做 Z 测量会得到的结果"。也就是说，被擦除的量子比特的 Z 测量结果可以从其邻居的测量中"间接读出"，从而无损地从图态中消除被擦除量子比特的影响——这是融合擦除容错的核心原语。关键前提：图态必须包含足够的辅助结构（邻居 + stabilizer 关系），这正是树编码（为每个叶子配备 q_i^a/q_i^b 辅助量子比特）与 OneAdapt-ET（利用 normalization 路径外的自由邻居）所构造的。本论文在融合擦除场景中的用法：q_i^c 经历擦除 → 对 q_i^b 做 X 测量 + 对 q_i^a 做 Z 测量 → 等价于间接得到 q_i^c 的 Z 测量结果，q_i^c 被无损消除，其余量子比特不受影响。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
间接 Z 测量的张量/stabilizer 计算过程（论文 Fig.4(c)，目标 qubit j_0 被擦除）：
```
# 图态 stabilizer：对每个顶点 i，S_i = X_i ∏_{j∈E(i)} Z_j 是 |G> 的保真生成元
# 目标：读出被擦除的 j_0 的 Z 测量结果
# 步骤：
# 1) 选相邻 qubit i，执行 X 测量 -> 得结果 x_i
# 2) 对连接到 i 的其它所有 qubit j_1, j_2, ... 执行 Z 测量 -> 得结果 z_{j1}, z_{j2}, ...
# 3) 由 stabilizer 约束：x_i * ∏ z_j = s_i  (s_i = stabilizer 特征值, 已知)
#    -> 读出 z_{j0} = s_i * x_i * ∏_{j≠j0} z_j
#    即：被擦除的 j_0 的 Z 测量结果被确定性揭示
```
树编码中的应用（分支 i 融合擦除）：q_i^c 被擦除 → X_measure(q_i^b) 得 x + Z_measure(q_i^a) 得 z → 由 stabilizer X_{q_i^b}·Z_{q_i^a}·Z_{q_i^c} 读出 q_i^c 的 Z 结果 → q_i^c 无损移除，q_root 与其余分支不受影响。这使"每分支擦除损失"从全逻辑量子比特崩溃降级为单分支失效，是 S_tree 相对 baseline 指数优势的来源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：间接 Z 测量在测量调度层面实现——编译器/控制器把"对 q_i^b 做 X、对 q_i^a 做 Z"的测量模式作为擦除结果的前馈动作下发（论文用 Perceval FFCircuitProvider 实现该条件前馈逻辑，feed-forward 延迟 <5 ns）。使用场景：(1) 本论文树编码融合——融合擦除恢复；(2) OneAdapt-ET——对 OneAdapt 中经历擦除的 qubit，在 normalization 路径外找相邻自由 qubit 做 X 测量 + 对其余相邻 qubit 做 Z 测量；(3) QEC 树码（tree cluster-state code，误差修正动物园收录）——一般性的光子丢失容错。注意：间接 Z 测量需要图态中预先存在相邻辅助 qubit（树编码的 q_i^a/q_i^b 就是为此设计的），这是编码开销的来源（光子源多 2.55×/1.63×）。论文未开源。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion
