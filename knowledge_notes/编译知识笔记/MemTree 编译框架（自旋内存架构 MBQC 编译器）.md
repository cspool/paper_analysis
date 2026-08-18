## MemTree 编译框架（自旋内存架构 MBQC 编译器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MemTree 是本论文提出的 MBQC 编译框架，面向量子自旋内存（spin memory）PQC 架构，负责把量子程序编译为可高效、稳健执行的图态生成计划。它包含三个核心算法：(1) 目标图态划分——用混合整数规划（MIP，Gurobi 求解器）把程序图态 g_prog 划分为最小融合次数的线性子图集合 G^l（约束每个顶点度 deg(v)≤2 保证线性，目标函数 K=|E|-Σx_e 最小化切割边数，后处理切断环边并按 30-qubit caterpillar 上限细分）；(2) 平衡二叉树（BBT）构建——从根 G^l 递归二分 G^l_i 为 G^l_j、G^l_k（第二个 MIP 模型，变量 y_g^l∈{0,1} 表示子图分到左/右子树，平衡约束 |G^l_j|,|G^l_k|≥2^⌊log2|G^l_i|⌋，目标 L=Σ|y_{g1}-y_{g2}| 最小化跨分组切割边数），最小化临界路径（leaf→root 最大融合数）开销；(3) 生成流水线——类似 OneAdapt 的时间模型，光子源迭代生成 caterpillar 态，每时间步 BBT 一层各子图并行融合合并为父图，融合失败/擦除时兄弟子图延迟到下一时间步，最大化限定周期内成功执行的 shots。MemTree 与树编码融合（算法层）协同：树编码提供融合容错（抗失败 + 抗擦除），MemTree 提供资源高效的分层生成，两者共同实现相对 OneAdapt 执行时间指数级下降（1.5×10^-2×）、光子源 0.18×、编译时间 0.14×、fidelity 3.64×。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
MemTree 编译框架的运转流程（输入→输出，36-qubit VQE 例子）：
```
输入: 量子程序 (VQE 电路)
# 1) 程序图态化
g_prog = program_to_graph(VQE)          # 顶点=qubit, 边=CZ 纠缠
# 2) MIP-1: 线性子图划分 (Gurobi)
x_e ∈ {0,1} 每边保留/切割; 约束 Σ deg(v) ≤ 2, ∀v
min K = |E| - Σ x_e                      # 最小化融合数
G^l = postprocess(cut_cycles, cap_30qubit(G^l))   # 切环 + caterpillar 长度上限
# 3) MIP-2: 递归二分构建 BBT
build_BBT(root=G^l):
    if |G^l| == 1: return leaf
    y_g ∈ {0,1} 子图分左/右; 约束 |G^l_j|,|G^l_k| ≥ 2^⌊log2|G^l_i|⌋
    min L = Σ_{(v1,v2)∈E} |y_{g1} - y_{g2}|       # 最小化每层切割边数
    return node(children=build_BBT(G^l_j), build_BBT(G^l_k))
# 4) 流水线执行计划
for each BBT layer (time step):
    parallel_fusion(layer_subgraphs)     # 每层融合数 = 该步切割边数
    on_failure: delay_sibling_to_next_step()
输出: 分层融合计划 (每时间步的融合操作集合)
```
编译输出的意义：每层切割边数 = 该时间步需执行的融合操作数；失败只重试子树而非整图（对比一次性全部融合：100-qubit VQE 需 k>1000 次融合、成功率 S^1000~1e-5 不可行）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：MemTree 本体为论文全新自研（非修改开源编译框架），图划分用 Gurobi MIP 求解器；消费/对比的开源生态：OneAdapt（Zenodo artifact：https://zenodo.org/records/17065528，MICRO'25，Apache 2.0，Python 3.10 + Qiskit/PyZX/NetworkX）、Perceval（https://github.com/Quandela/Perceval，光子电路仿真/云平台）、Qiskit（真实硬件对比的转译 baseline）；论文新增 OneAdapt-ET（在 OneAdapt 中集成间接 Z 测量擦除容错）。使用方法：把量子程序图态 + 硬件约束（caterpillar 长度上限、融合错误率）作为输入，得到图态生成计划；该计划驱动光子源发射与融合执行。开源情况：论文未提供代码仓库（arXiv:2604.21475），联网搜索未找到，无法确认开源。两个 MIP 模型复杂度 O(|E|) 与 O(|G^l_i|) 二元变量数，编译时间相对 OneAdapt 平均 0.14×。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion
