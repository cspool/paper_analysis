## Caterpillar State（毛毛虫态）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
毛毛虫态（caterpillar state）是量子自旋内存架构（spin memory PQC）生成的一种特殊图态：具有分支链（branched-chain）结构——一条线性纠缠的量子比特链作为主路径（main path），每个主路径量子比特额外连接若干叶量子比特（leaf qubits，每条叶子链只连一个主路径顶点，因形似毛毛虫而得名）。定义见 Pettersson-Sørensen-Paesani（PRX Quantum 6, 010305, 2025）；物理制备过程见 Huet 等（Nature Communications 16, 4337, 2025）。制备机制：在半导体量子点（QD，如 InGaAs）腔体上迭代施加纵向声学（LA）激发脉冲，可发射线性纠缠的光子图态；在激发脉冲间插入光学自旋旋转脉冲（OSRP），则发射出 caterpillar 结构——主路径链 + 叶量子比特。关键性质：(1) caterpillar 态已在实验中小规模演示（[29]），是 near-term 可实现的结构；(2) 其结构具有灵活性——主路径 + 叶子可直接作为树编码逻辑量子比特的骨架（q_root 在主路径上，叶 qubit 由 caterpillar 提供），还能通过 Z 测量从长线性图中分离 4-qubit 线性图再融合到叶子上组装树结构；(3) near-term 限制——单个 caterpillar 态最多 30 qubit（[30]），初始化 12 ns + 每 qubit 发射 0.6 ns。论文用 caterpillar 态作为图态生成的基本资源态：MemTree 编译器把目标图划分为可被 caterpillar 覆盖的线性子图。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
caterpillar 态在论文 pipeline 中的生成与使用（组装树编码逻辑量子比特，Fig.4(e)）：
```
# 阶段1：caterpillar 发射（spin memory 硬件）
for each qubit on main_path:
    LA(pi/2)          # 纵向声学激发脉冲 -> 发射与 QD 纠缠的光子
    OSRP(pi)          # 光学自旋旋转脉冲 -> 定义叶分支结构（间隔插入）
# 输出：caterpillar 态 = 主路径链 {m1-m2-...-mk} + 每主路径顶点挂 b 个叶 qubit
# 阶段2：组装树编码逻辑量子比特
q_root = main_path_qubit                        # 根：主路径上的量子比特
leaf_qubits = caterpillar_leaves(q_root)        # 叶子：caterpillar 提供的叶 qubit
for each branch i:
    linear4 = Z_measure_separate(long_linear)   # 从长线性图经 Z 测量分离 4-qubit 线性图
    fusion(linear4, leaf_qubits[i])             # 融合拼到叶子上 -> 形成 {q_i^a,q_i^b,q_i^c}
# 阶段3：融合拼接成目标图态（BBT 分层 + Type-II fusion）
```
关键参数：单个 caterpillar ≤30 qubit（near-term 硬件限制）、b=4 分支、b_prep=6 制备参数（30-qubit 限制下 b_prep>6 时光子源急剧增长）。caterpillar 态同时是"程序无关"的——其结构由目标图态按需确定，光子源利用率 ~10%（vs OneAdapt 0.03%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 硬件——硅基量子自旋内存（semiconductor quantum dot emitters）：InGaAs QD 腔体 + LA 激发脉冲 + OSRP 脉冲序列，实验演示见 Huet 等（Nat. Commun. 16, 4337, 2025；确定性、可重构图态生成）；(2) 模拟——论文自研 spin memory 模拟器按上述硬件配置（12 ns 初始化 + 0.6 ns/qubit、30-qubit 上限）模拟 caterpillar 发射；(3) 编译——MemTree 的 MIP-1 把目标图态划分为线性子图（后处理按 30-qubit caterpillar 上限细分）。使用场景：任何以量子点发射器为光源的 PQC；论文真实硬件实验在 Quandela 云平台（其平台基于自旋内存/QD 技术）验证。论文未开源。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion
