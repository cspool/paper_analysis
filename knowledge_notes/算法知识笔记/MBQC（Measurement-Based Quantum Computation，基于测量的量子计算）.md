## MBQC（Measurement-Based Quantum Computation，基于测量的量子计算）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MBQC（测量基量子计算，又称 one-way quantum computation 单向量子计算）是与传统的门模型（gate-based）量子计算并列的一种量子计算模型：计算不通过执行量子门序列进行，而是先制备一个高度纠缠的多体量子态——图态（graph state），然后对图态上的量子比特执行按顺序的、依赖于先前测量结果的单比特测量（adaptive single-qubit measurements），测量结果即计算输出。关键事实：图态对 MBQC 是普适的（universal）——只要图态被制备出来，任何量子计算都只需单比特测量即可完成（Raussendorf-Briegel 2001）。核心逻辑链：(1) 程序被表示为图态；(2) 图态由小的资源态通过 fusion 操作拼接而成；(3) 测量图态上的量子比特完成计算——测量把纠缠"消耗"为计算结果；(4) 测量基的选择可以是 adaptively 的（feed-forward），把前面测量的结果作为后续测量的条件。在论文（MemTree）中，MBQC 是光子量子计算机（PQC）的计算模型：与超导、中性原子等门模型硬件不同，光子硬件天然适配 MBQC——光子是飞行比特，图态在发射过程中边生成边测量，无需在空间上驻留。PQC 编译的核心挑战因此变为"如何稳健高效地生成目标图态"。论文将 MBQC 作为背景提出：graph state 的定义 |G⟩=∏CZ_{(i,j)}|+⟩^⊗V，fusion 是拼接小图态生成大图态的关键操作。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MBQC 计算 pipeline（以论文中的 QAOA 程序为例，图态生成 → 测量执行）：
```
# 阶段1：图态生成（本论文的研究重点）
target_graph = compile_program_to_graph(QAOA_circuit)   # 程序 -> 图态 G=(V,E)
caterpillars = emit_from_spin_memory(target_graph)       # 光子源发射 caterpillar 态
big_graph = fuse(caterpillars)                            # Type-II 融合拼接成大图态
# 阶段2：单比特测量执行计算
for t in measurement_schedule:
    outcome_t = single_qubit_measure(q_t, basis_t)       # 按图态结构逐比特测量
    basis_next = feed_forward(outcome_t)                  # 测量基自适应更新
# 输出 = 测量结果序列（经典后处理得到程序结果）
```
张量/量子态计算：图态 |G⟩ = ∏_{e(i,j)∈E} CZ_{(i,j)} |+⟩^{⊗V}——每个顶点初始化为 X 本征态 |+⟩，每条边施加 CZ 门。一次 X 基测量把该量子比特从图态中移除并断开其所有纠缠边；一对相邻 X 测量把两个量子比特移除并在它们各自的邻居之间建立直接连接（即"测量即计算"——纠缠在测量下被传输和消耗）。论文在 spin memory 架构上实现此 pipeline：caterpillar 态是 MBQC 的资源态，融合完成后按时间方向测量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 硬件侧——光子量子计算机（PQC）天然实现 MBQC：PsiQuantum（全光子，线性光学 + SPDC 源）、Quandela（自旋内存架构，半导体量子点发射器 + 线性光学融合）、以及 emitter-based（量子发射器，理论确定性生成）三大架构都以"生成图态 → 融合拼接 → 测量"为执行模型；(2) 软件/编译侧——MBQC 编译器把门模型程序（如 Qiskit 电路）转译为图态生成 + 测量调度：OneQ（ISCA'23）、OnePerc（ASPLOS'24）、OneAdapt（MICRO'25）、RLGS（ISCA'25）、以及本论文的 MemTree。使用场景：光子平台（室温运行、退相干时间长、天然适合量子网络集成）；"图态生成"环节决定编译器的核心优化空间（融合次数、光子利用率、错误容错）。论文未开源；其真实硬件实验基于 Quandela 云平台 + Perceval 工具包。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion
