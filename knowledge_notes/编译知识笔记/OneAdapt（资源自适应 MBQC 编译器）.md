## OneAdapt（资源自适应 MBQC 编译器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
OneAdapt 是面向全光子（all-photonic）架构的 SOTA MBQC 编译器（Zhang, Ruan, Tullsen, Ding, Li, Humble，MICRO'25, pp.733-748, DOI 10.1145/3725843.3756100），本论文的主要对比 baseline。它扩展了 OnePerc 的 FlexLattice IR，加入两个硬件感知特性：(1) 有界时间类边长度（bounded temporal edge lengths）——建模有限光子延迟线的硬件约束；(2) 倾斜时间类边（skewed temporal edges）——连接不同层中邻近 2D 位置的节点。优化 pass：动态节点刷新（dynamic node refresh，相对 OnePerc 周期性刷新更细粒度）与 2D 有界时间类路由（2D-bounded temporal routing，利用倾斜边同时减小 2D 硬件尺寸与 1D 执行深度）。执行模型：迭代生成资源状态层（resource state layer, RSL），并用 normalization 方法把 RSL 规范化成有效的 2D 图态层（lattice graph），从而生成目标图态。论文指出的缺陷：(1) 只处理融合失败、忽视融合擦除——1% 擦除率下生成 84×84 RSL 需 >10^5 次融合，整个 RSL 不经历擦除的概率极低；(2) normalization 光子利用率极低——从 84×84 qubits 的 RSL 只 normalize 出 4×4 的 2D 层（利用率 ~0.03%）。评估设置（论文采纳）：时间类边长度限制 D_f=30 虚拟层、每虚拟层 PL=4 物理资源层、time-like 边最大 960 ns、RSL 尺寸 14n×14n、PsiQuantum 125 MHz 源泵浦、T2=2.04 μs。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
OneAdapt 编译/执行流程（36-qubit VQE 例子）：
```
输入: 量子程序 -> FlexLattice IR (含时间类边长度/倾斜边信息)
# 1) 迭代生成资源状态层 RSL (14n x 14n 2D)
RSL = generate_resource_state_layers(program, D_f=30, PL=4)
# 2) normalization: RSL -> 有效 2D 图态层 (lattice)
for layer in RSL:
    eff_2d = normalize(layer)      # 84x84 RSL -> 4x4 有效层 (利用率 ~0.03%)
    # 处理融合失败: 失败由 normalization 吸收/重排
    # 不处理融合擦除: 擦除使 2D 层结构不确定 -> 需丢弃
# 3) 时间模型执行: time-like edges 960ns, 虚拟层 D_f=30
输出: 2D 图态层序列 -> 后续 MBQC 测量
```
论文对照：OneAdapt-ET = OneAdapt + 间接 Z 测量擦除容错（对 normalization 路径外的相邻自由 qubit 做 X 测量 + 对其余相邻 qubit 做 Z 测量，Fig.7）；MemTree 相对 OneAdapt/OneAdapt-ET 执行时间指数级下降（36/64/100 qubit、p_eras 0~5%），光子源 0.18×、编译时间 0.14×、fidelity 3.64×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：OneAdapt 官方 artifact 公开于 Zenodo（https://zenodo.org/records/17065528，Apache License 2.0）：Python 3.10 + Jupyter Notebook，依赖 Qiskit、PyZX、NetworkX、Matplotlib、NumPy；conda 环境 `conda create -n OneAdapt python=3.10` + `pip install -r requirements.txt`；运行 `Evaluation.ipynb`（从预存数据生成图表）或 `run.sh`（重跑全部实验，标准 Intel CPU 服务器、128 GB RAM 可跑完整 benchmark）。使用场景：作为全光子 MBQC 编译的 SOTA baseline 与"资源自适应"范式参考（光子资源与执行深度 trade-off）；论文以其开源 artifact 复现对比，并扩展出 OneAdapt-ET。本论文（MemTree）未开源。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion
