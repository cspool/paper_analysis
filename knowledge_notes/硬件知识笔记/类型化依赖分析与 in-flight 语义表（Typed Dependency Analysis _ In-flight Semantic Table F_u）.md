## 类型化依赖分析与 in-flight 语义表（Typed Dependency Analysis / In-flight Semantic Table F_u）

术语解释
动态 tile 调度器的核心依赖追踪机制：每个执行单元 u 维护一张 in-flight 语义表 F_u = {(idx, start_addr, end_addr, access_type, unit, inst_ptr)}，用区间重叠 + 内存 scope 隔离 + OpType 语义兼容 + 资源可行性四类规则判定 Hazard，把 CPU 式 register/内存保守依赖跟踪升级为 tile 级语义感知的 RAW/WAR/WAW 检测。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 论文（Section V-A）：Hazard(I, F_u) = ∃r ∈ F_u : SemanticConflict(I, r)，其中 SemanticConflict 由类型化规则解析：① 数据依赖（RAW/WAR/WAW）用 TileMem 区间重叠测试，区间不相交视为独立；② 内存 scope 隔离——目标不同内存层级/bank（L1 vs L2、不同 HBM channel）的非别名访问可安全并行；③ 语义兼容——OpType 兼容（如 GEMM vs Softmax）且数据区间不冲突且单元类不同者可重叠；④ 资源可行性——当前单元容量（本地 buffer 大小/带宽）不足则推迟发射。
- 依赖类型 Deps = {(src, type, condition)} 由 TileMem 区间重叠分析自动导出（如两 tile 地址区间重叠且一写一读 → RAW，writer commit 前强制）；condition 字段表达部分/条件就绪（partial-tile availability），允许依赖 tile 在其所需子区域有效时提前发射。interval 分析对非连续/strided 访问保守（可能标记跨 stride 间隙的假冲突，但绝不错过真冲突；常规非连续访问经 transpose 等在高层编译为连续访问）。
- 与传统机制对比：CPU scoreboard 跟踪寄存器 tag、Tomasulo 做 O(N²) 全局比较；F_u 编码语义与空间上下文（idx/start_addr/end_addr/access_type/unit/inst_ptr），冲突检测 O(|F_u|) 每候选，支持部分完成追踪（子 tile/内存区域就绪即唤醒依赖 tile）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件实现：F_u 是每执行单元的硬件状态表（类似语义化 scoreboard），RTL 中用 CAM 结构支撑窗口匹配。Algorithm 2 伪代码：对每个 r ∈ F_u——不同 scope 直接 continue（可并行）；OpType 语义兼容则 continue（可重叠）；仅当内存区间重叠且为真依赖且不可安全重排才返回冲突；否则安全并行。
- 运转流程例子（FA3）：M0^i 写 s_P（区间 P），S^i 读 s_P → 区间重叠且一写一读 → RAW，S^i 在 M0^i commit 前不发射；M0^{i+1} 读 s_Q、写 s_S_next，与 S^i（读 s_P、写 s_S）区间不重叠、OpType 不同（GEMM vs Softmax）、单元类不同（ME vs VE）→ 判定独立并发发射；S^i 完成时 F_ve 对应条目退休、M1^i（读 s_S）被唤醒。相比静态 bar.sync 锁定的固定屏障，调度器可区分"两个单元都忙"的真冲突与"一个单元在等"的可恢复停顿。
- 指标：每调度周期 O(U·W·|F|max)，典型 W≤8、|F_u|≤16 时近似 O(U)；综合到 Epoch 每核硬件，仅 7~9 cycles/dispatch。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：依赖由编译器（FC/TISA generator）通过 TileMem 区间分析标注进 TISA 指令；硬件调度器运行时用 F_u 表 + 规则检测执行。TileMem 目前不支持原生 strided 数组访问，需编译器转置列向量为连续地址或发射保守边界（产生假依赖）；论文称密集 LLM/CNN 天然操作大连续块，精度损失可忽略，扩展 stride 元数据为未来工作。
- 使用：每单元独立 F_u 提供分布式逐单元仲裁，防无关阻塞；与集中式 ILP 调度器（Tomasulo O(N²)）相比在更大规模下可扩展。Epoch 真硅片验证：类型化依赖使 S_i∥M0_{i+1}、M1_i∥S_{i+1} 跨迭代紧凑重叠，消除隐式迭代屏障，Dynamic vs Static 再高 1.14–1.63×。
- 开源情况：论文未给出实现代码/开源链接。

涉及论文标题：
- Dynamic Scheduling for AI Accelerators via TISA
