## 解耦流水（Decoupled Compute/Transfer/Memory Pipelines，PE 内多流水重叠）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 解耦流水是 MLX PE 的微架构组织：把每个 PE 拆成四条独立流水——内存移动（LD 流水，memory movement）、数据流传输（XFER 流水，dataflow transfer）与异构算术（COMP 流水，real/complex FMA + 超越函数），各自独立推进、由 tag 级仲裁协调，而非用单一指令级调度器管理。动机：MLX 折叠层执行要求每个 PE 同时推进多个层（加载未来层输入、计算当前层、转发前层输出），且混合算子混杂实数/复数算术与激活/归一化函数（Fig.9(c)/(d)）；单指令级调度器管理这些异构单元会带来大量面积与控制复杂度。解耦后每个 PE 以层粒度调度、天然重叠各层相位（active-window 内多层 time-share 同一 PE 资源），控制轻量、利用率高。
- 与 tagged block 的关系：tagged block 的固定指令布局（LD 头/COMP 中/XFER 尾）正是为解耦流水设计的——各流水从同一 block 中抽取自己负责的指令段，跨层重叠时"一层加载、一层计算、一层转发"同时进行；memory 流水偶尔在中间层空闲，但占主导的 compute 流水保持连续占用。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（active-window 内三折叠层）：层 i 的 LD 流水从 SIMD-striped scratchpad 读输入（列向 lane 对齐序列轴 N 做 BSMM、行向流式隐藏轴 D 做 chunk FFT）→ 层 i+1 的 COMP 流水在 FU 上执行 FMA（蝴蝶混合/GEMM psum）→ 层 i+2 的 XFER 流水把部分和经 skip-hop 网格发给消费 PE——三条流水并行推进、由 tag 就绪位与仲裁器协调（T_compute(block) ≥ max(T_load, T_xfer) 保证覆盖）。效果：BSMM 与 FFT 计算利用率约 90%（load/store/transfer 归并的数据供应流水延迟行为一致），SWA 的 FMA 利用率 43%-75%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：每 PE 内四条独立流水 + 共享 register file/scratchpad 接口 + tag 仲裁器（frontier 指令 inst_i 粒度仲裁、tag ID 编码层间偏序）；面积分布（Table II）：FU(SIMD32) 0.298 mm²/252.4 mW（占 70%）、Data Network 0.092 mm²/56.2 mW、Register File 0.044 mm²/28.7 mW、Tag Buffer 0.019 mm²/9.3 mW。使用：编译器按 LD-COMP-XFER 布局生成 tagged block，硬件以层粒度（而非指令粒度）调度多折叠层重叠；scalability 时靠增大块计算预算 C 或活跃 tag 数 B_T 维持覆盖（B_T·C ≥ T_load+T_xfer）。该解耦是"轻量 tagged-block 编排替代细粒度操作调度"的关键，以有限硬件复杂度维持高利用率。
- 涉及论文标题：MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures
