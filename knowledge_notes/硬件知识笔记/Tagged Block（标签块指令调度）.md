## Tagged Block（标签块指令调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Tagged Block 是 MLX PE 的指令组织与调度单位：把每个逻辑层编码为一个紧凑的 tagged block——短的静态指令序列 + loop trip count（n），块内指令固定布局（LD 加载在开头、COMP 计算在中间、XFER 传输在结尾），携带 tag（层身份）与可重放性（同一 block 跨 CDC 实例/折叠窗口滑动反复重放）。它解决"执行很多 MLX 层需要巨大指令缓冲 Θ(K·I_layer)"的问题：由于每层是内部顺序不变的可复用指令模板，指令存储与层数解耦，PE 只需驻留活跃窗口内少量 block（4×4 网格每 PE 32 条指令即满足覆盖条件）。
- 调度语义：(1) 层对齐——tagged block 把硬件调度单元与 MLX 层边界对齐，按层粒度跟踪就绪/进度/完成，避免逐指令簿记、大依赖表与细粒度 hazard 元数据；(2) 活跃窗口重叠——block 作为 decoupled 流水中的可调度条目，同一窗口内不同折叠层同时处于不同流水阶段（一层 LD、一层 COMP、一层 XFER），tag 标识所属活跃层；(3) 粗粒度仲裁——层完成 LD 后置 ready 位使能后续 stage，各流水按资源可用性在 ready tags 中选择（多 ready 时 round-robin、以 Tag ID 作平局裁决、优先小 tag 保证依赖序正确）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（Fig.9(d) 仲裁例子）：两个折叠层 tag1 与 tag2 的指令同时到达 PE 的 compute 流水（tag1 的 add 与 tag2 的 mul 争用）→ 仲裁器按"小 tag 优先"给 tag1、stall tag2 的 block（在 block/tag 粒度做决策，而非逐指令）→ tag1 的 XFER 完成后置 ready 位 → 下一轮 tag2 获得资源。该机制把层内确定性（编译器发静态指令序列）与跨层弹性（硬件仅协调 tag 级传输/转发事件）分离——"hybridized scheduling"：不依赖细粒度动态 wakeup 也不要求全局周期级规划。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：每 PE 的 layer-encoded instruction store 持有固定数量 tagged blocks（tag + 缓冲指令序列 + loop trip count）；tag buffer 面积 0.019 mm²/9.3 mW（Table II）。使用：编译器把每个 CDC/层编译成 tagged block（指令布局按 LD-COMP-XFER 分组）；运行时 PE 按 tag 依赖重放（C_i ↦ loop(k) tagged_kernel_i(k)）；软件侧用 LLVM-based C 编译器或 dataflow 汇编 + spatial assembler 生成配置。效果：BSMM/FFT 计算利用率约 90%、kernel launch 开销 17%→<12%；跨层重叠让 FMA 主导的蝴蝶 kernel 保持高 affinity（相对 FLOP 削减的加速比）。
- 涉及论文标题：MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures
