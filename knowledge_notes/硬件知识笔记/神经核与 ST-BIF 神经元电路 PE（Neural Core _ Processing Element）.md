## 神经核与 ST-BIF 神经元电路 PE（Neural Core / Processing Element）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 神经核（neural core）是神经形态芯片的可扩展基本单元：一个核通常 = 若干 PE + 一个路由器，核间经 NoC 互联（TrueNorth 4096 核、PAICORE 1024 核、ELSA 6×6=36 核）。PE（processing element）是核内的计算单元，SNN 场景下执行 spike 驱动的加法型矩阵乘。ELSA 的 PE 针对 ST-BIF 神经元定制：每 PE 含 128 个 ST-BIF 神经元电路（16 输入加法树 + fire 组件 + update 组件）、N 路权重 buffer（4×102.4KB）、膜 buffer（4×307.2KB）、spike tracer buffer（4×102.4KB）与控制模块；每 PE 每周期执行 1024 次加法。路由器内含 SSoftmax/SLayerNorm 单元、im2col 单元、Local Input Reducer、Flit Generator/Decoder、FIFO、Output Scheduler、Routing Engine（见 BAER 术语）。
- 设计约束：一个神经核可承载多个 MM-sc（把 PE 的神经元电路与内存按 P 组划分，各组执行不同层的 MM-sc），因此"层内划分"（partition 阶段把整层放同核）可行。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- PE 内三个计算步（对应 ST-BIF 三步，ELSA Fig.8b）：
```
step-1 积分：控制模块收 spike {x,y_i,q_i} → 权重 buffer 按 y_i 取行（负 spike 二补码）
   → 加法树累加进膜行 V[x]（膜 buffer 单次读）
step-2 激发：fire 组件读 tracer S[x] + 膜 V[x] → Θ(V,V_thr,S) 判 y_t
step-3 更新：update 组件写回 V[x] 与 S[x]（soft reset + tracer 累积）
```
- 例：Tab.III 中单神经核（4 PE + 1 路由器）：PE 面积 2.59mm²（93.03% 的 ELSA 面积），其中 weight 0.487 + membrane 1.460 + tracer 0.487 mm²；路由器 0.19mm²（6.97%），其中 SSoftmax Unit 0.096 + SLayerNorm Unit 0.091 mm²。功耗上加法树 52%、权重 buffer 31.2% 主导（SNN = spike 加法 + 权重读取）。Fig.15 能量分解：adder tree 29%~39%、FIFO Queue 与 Membrane/Weight/Tracer buffer 次之，片外 DRAM 可忽略。
- Annotations：三类 SRAM（weight/membrane/tracer）是面积主导（93.97% PE 面积）；tracer 位宽小于膜但也是持久状态；每核 4 PE 是设计点（可扩展）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：RTL（Verilog）实现 PE/路由器，Synopsys Design Compiler 28nm 综合，VCS post-synthesis 仿真，PrimeTime PX 门级能量，SRAM 用商业 memory compiler，面积/功耗/延迟参数注入自研 cycle-level 模拟器。映射：贪心分区（Algorithm 2，约束核内存 A 与核神经元电路数 D）+ Hilbert 曲线放置 + 多路径路由（见编译框架层术语）。评价：ELSA 4135.4 GOPS / 25.55 TOPS/W / 41.26 GOPS/mm² / 0.032 pJ/Sop，相对 PAICORE 27.4× geomean 能量节省（mini-batch Gustavson）与 1.65× 速度（spine/token 流水）。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing
