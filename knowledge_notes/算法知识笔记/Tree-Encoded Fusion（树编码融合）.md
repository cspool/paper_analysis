## Tree-Encoded Fusion（树编码融合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
树编码融合（tree-encoded fusion）是本论文提出的核心新方案：一种同时容错融合失败与融合擦除的逻辑量子比特编码与融合方案，受 QEC 树码（tree cluster-state code）和冗余编码融合双重启发。方案要点：参与 Type-II fusion 的逻辑量子比特 A、B 被编码为树结构——根量子比特 q_root 连接 b 个分支，每个分支是 3 个量子比特的线性图 {q_i^a, q_i^b, q_i^c}；叶子 q_i^c 用于融合测量，q_i^a/q_i^b 是间接测量辅助量子比特。按融合结果执行不同测量模式：(1) 成功——对 q_i^a、q_i^b 做一对 X 测量，把成功融合的纠缠直接连到 q_root；(2) 失败——q_i^c 被测量掉，q_i^a/q_i^b 留在树中，对 q_i^b 做 Z 测量移除它，留 q_i^a 作备份；(3) 擦除——对 q_i^b 做 X 测量、q_i^a 做 Z 测量，实现 q_i^c 的间接 Z 测量（基于 stabilizer X_i∏Z_j），无损消除被擦除 qubit 的影响；(4) 全部分支失败/擦除的极端情况——用 (2) 留下的备份 q_i^a 再试一次。逻辑成功率 S_tree = 1-(1-(1-p_eras)^2+p_fail)^b。树结构可从 caterpillar 态高效组装（主路径 q_root + 叶 qubit + 经 Z 测量分离的 4-qubit 线性图融合），与 spin memory 架构天然契合。参数选择 b=4、b_prep=6（30-qubit caterpillar 限制下 photon 源与执行时间的 trade-off）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
树编码融合的完整算法流程（伪代码，含间接 Z 测量恢复）：
```
# 逻辑融合：逻辑量子比特 A、B（各编码为 b 分支树）
for branch i in 1..b:
    outcome = type2_fusion(q_i^c(A), q_i^c(B))
    if outcome == SUCCESS:
        X_measure(q_i^a); X_measure(q_i^b)   # 成功纠缠直接连到 q_root
    elif outcome == FAILURE:
        Z_measure(q_i^b)                      # 移除 q_i^b，留 q_i^a 作备份
    elif outcome == ERASURE:
        X_measure(q_i^b); Z_measure(q_i^a)    # 间接 Z 测量 q_i^c：stabilizer X_i∏_{j∈E(i)}Z_j
# 备份机制：若全部分支失败/擦除且存在备份
if no_branch_succeeded and backup_exists:
    fusion_retry_with(q_i^a)
# 制备侧（caterpillar 组装，b_prep 次并行尝试）：
for attempt in 1..b_prep:
    branch = prepare_branch_from_caterpillar()   # 4-qubit 线性图融合到叶 qubit
    if branch_fusion_failed: discard (自动测量掉)
    if branch_fusion_erased: indirect_Z_measure({q_i^b, q_i^e})   # 恢复
# 若成功分支 < b，下一 timestep 重试；参数 b=4, b_prep=6
```
成功率公式对比：S_redun=(1-p_fail^m)(1-p_eras)^(2m)、S_rus=1-Σp_fail^i·2p_eras-p_fail^m、S_tree=1-(1-(1-p_eras)^2+p_fail)^b。例：p_fail=0.25、p_eras=0.1、b=4：S_tree≈1-(0.44)^4≈96.3%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 逻辑量子比特制备——在 caterpillar 态上组装：caterpillar 主路径提供 q_root 与 b 个叶 qubit；另生成 b 个 4-qubit 线性图（从长线性图经 Z 测量分离），用融合拼到叶子上；制备参数 b_prep=6（b_prep>b），同 timestep 并行尝试，失败分支自动测量掉、擦除分支用间接测量恢复；(2) 真实硬件——Quandela 云平台 + Perceval 构建双轨编码光学电路（光子模式置换 + 相移 + 分束器），融合结果触发 FFCircuitProvider 的条件前馈（对 q_i^a/q_i^b 施加 X 或 Z 测量）。实验验证：83.3% 单 timestep 制备成功率、97.1% 两 timestep 内；相对 redundantly-encoded 执行时间 1.9×10^-3×、相对 RUS 1.7×10^-2×（光子源多 2.55×/1.63×，用空间换时间）。论文未开源。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion
