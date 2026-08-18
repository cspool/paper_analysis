## 表面码（Surface Code，patch 抽象与容错操作）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 表面码（surface code）是最主流的量子纠错码：把数据 qubit（o）与两类 syndrome qubit（•，X 型/Z 型稳定子）交错排在二维格点上，每轮 syndrome extraction 测量稳定子检测错误；码距 d 的 patch 可纠正 ⌊(d−1)/2⌋ 个错误。逻辑 qubit 抽象为 d×d 的 patch，四个边界各为 X 型或 Z 型边界——X/Z 逻辑算子可表示为 patch 上的边缘算子。平面连通、阈值约 1%、高效解码，通过 lattice surgery、code deformation、magic-state distillation 实现通用容错计算。物理 qubit 数 ≈2d² 每逻辑 qubit（TACO 取 d=19 时 722/tile）。
- 论文用途（TACO）：surface code 是 FTQC 的硬件衬底——TACO 架构的 compute/memory 区域都用标准 surface-code patch（同一物理衬底），仅逻辑角色不同：compute 区域用扩展 patch（d×d → 约 2d×d，物理 qubit 约 2×）以暴露 X/Z 双边界支持任意 π/4 旋转，memory 区域用标准 patch。逻辑错误率模型 p_L=0.1(100p)^((d+1)/2)（p=物理错误率），code distance 由 total_tiles×total_cycles×d×p_L<0.01 确定。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 TACO 架构中的运转流程（18 比特 QFT，d=19）：
```
① 电路映射: 转译后 29 个数据逻辑 qubit 分配 tile——活跃的 π/4 旋转 qubit 进
   compute block（4 tile：2 个 π/4-site qubit + 2 ancilla，X/Z 边经 ancilla 暴露
   给 distillation），空闲 qubit 进 memory block（三行 1.5n 布局，中行 ancilla）
② 稳定子测量: 每 QEC cycle 一轮 syndrome extraction（d=19 的 2D 格点），
   检测错误并解码（纠错）
③ 门执行: T/Rx(π/4) 门 = 目标 qubit 的 Z/X 边与 magic state patch 做
   lattice surgery（2.5d+4 cycles 消耗）；CNOT = 两 patch 经共享 ancilla 合并
   （3d+4 cycles）；Hadamard = patch deformation（3d+4）
④ 数据传输: qubit 从 memory 移到 compute = 沿共享 ancilla 路径 expand 进
   compute block（1 cycle），patch rotation（换暴露边界）3 cycles
输出: 每 QEC cycle 一个逻辑门吞吐，总 595,604 cycles（MSC）
```
- 作用：把"逻辑电路层优化"落地为"物理 qubit 布局与门时序"，TACO 用 locality 感知的 compute/memory 分区把高并行电路（每层多个 T 门）高效映射到 surface-code 晶格，同时用空间-时间体积最小化决定 distillation 工厂数量/吞吐。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：物理上以稳定子测量电路实现（每轮 ancilla 与邻域数据 qubit CNOT 后测量，如 Google Willow d=7 演示每级约 16× 错误抑制，arXiv:2408.13687）；仿真上可用 Stim 生成 syndrome（`stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, ...)`）驱动解码器。TACO 的架构模拟不跑物理级仿真，而是用逻辑级资源模型（每逻辑门 QEC-cycle 成本 + 2d² 物理 qubit/逻辑 tile）估计空间-时间体积。使用场景：FTQC 资源估计（qubit 数、cycle 数、体积）、patch 布局/路由编译（O3LS、LSQECC）、以及本论文的 compute/memory 分区架构设计。注意：本知识库 算法pipeline 层另有一篇以 syndrome 压缩（IcePack）视角写的"表面码（Surface Code）"条目（A Streaming Architecture 论文），与本条目互补。

涉及论文标题：
- Transpiler-Architecture Co-Design to Curb Clifford Costs in Fault-Tolerant Quantum Computing
