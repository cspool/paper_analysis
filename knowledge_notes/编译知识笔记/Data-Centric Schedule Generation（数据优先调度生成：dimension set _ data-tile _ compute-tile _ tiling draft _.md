## Data-Centric Schedule Generation（数据优先调度生成：dimension set / data-tile / compute-tile / tiling draft / 剪枝，含 data-centric IR）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DCC 编译器的核心调度生成流程：与传统 ML 编译器"先 loop 变换切计算、再靠缓存层级提升局部性"相反，采用 data-first 策略——先枚举数据张量在 PIM group/core 上的所有候选分区（data-tile），再映射到计算循环的划分（compute-tile），形成 tiling draft；随后剪枝、逐 draft 优化并用预测器选最优。四阶段：(1) Dimension Set Construction：把张量维经 loop 变量映射为映射函数（如 A[b+1][i*2] → F(b)=b+1、F(i)=i*2），收集 reference set，共享同一 loop 变量的维度合并为 complex dimension set；(2) Data-Tile Construction：对每个 dimension set 穷举 [1,k] PIM 组 × [1,m] core 的分配（DFS + memoization，avail_resources_{i+1}=avail_i/allocated_i），每个 draft 每 part 记 [组数, 核数, representative mapping]；(3) Draft Pruning：三规则——等价 representative mapping 的冗余去重、SIMD 宽度对齐（d-way SIMD 下每 core 分配的维度尺寸须是 d 的倍数，否则剪）、执行性能等价（各组 worst per-core 性能相同则去重）；(4) Data-Tile→Compute-Tile 映射：core 级 data-tile 大小=dimension_size/(groups×cores)，反解映射函数得 loop 变量范围即 compute-tile，再导出全部张量数据索引（索引非连续时两策略：扩展为连续区间 vs 保持原索引，都生成由预测器裁决）。配套 data-centric IR：complex_dimset/data_tile/compute_tile/tensor_tile/rearrange[to_PIM_core|to_host]/parallel_PIM/loop 构造，编码最优分区与调度后 lowering 到后端 PIM 指令。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
例子（论文 Fig.5，kernel 含 A[b][i]、A[b+1][i*2] 引用）：维度 A_0 的 reference set {b,b+1}、A_1 为 {i,i*2}、C_0 为 {b}；A_0 与 C_0 共享变量 b → 合并为 complex dimension set (A_0={b,b+1}, C_0={b})；draft 第一 part [1,4,C_0^b] 表示 1 组 4 core、representative mapping C_0 用 F(b)=b；core K_0 的 data-tile = C_0/(1×4)=[0,3)，对应 compute-tile b∈[0,3)，再导出各张量索引（A_1 索引 {0,2} 非连续 → 生成扩展 [0:3) 与保持 [0],[2] 两个 mapping 变体）。剪枝效果：搜索空间平均减 9.01×、编译时间减 5.97×（ATTN 5 设备 batch 4：32186→4274 草案、9.57s→2.96s）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Python 前端解析 @DCC_kernel 标注的仿射 loop 嵌套 → DFS+memoization 枚举 → 剪枝 → 对每个 draft 生成 IR → XGBoost 预测器评端到端时间（15% 训练/85% 测试，全 workload 离线编译 ~67s）→ 最优 draft 入 lookup table（动态尺寸在线生成）。使用：替换模型层（DCC.Layer），preLoad 预载权重，运行时按张量尺寸查表执行。与 ATiM 的本质区别：数据分区与计算划分在一个统一搜索空间里联合枚举，而非先固定计算模板。开源：github.com/SPIN-Research-Group/DCC（MIT）。

涉及论文标题：
- DCC: Data-Centric Compilation of Machine Learning Kernels for Processing-In-Memory Architectures
