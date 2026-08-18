## GH-Scope（IR 级模拟与验证框架）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- GH-Scope 是 Graph.hls 的快速验证框架，运行于 Graph.hls-IR 级别（而非 HLS C 级或硬件仿真），含四层能力：①静态结构验证——类型检查（顶点/边/属性数据类型全程一致）+ 循环依赖检查（生成的 DAG 是否合法，IR 级提前发现非法 pipeline 反馈环，替代传统运行时死锁检测）；②IR 级模拟——自定义解释器直接在 IR 数据流 DAG 上执行：每次迭代处理所有边经 map/reduce/filter 节点（用真实属性值）、更新顶点状态、重复至收敛或最大迭代数；③运行时检查——overflow 检测（所有算术操作超出属性位宽立即报告，如 16-bit 距离回绕）与无限循环检测（跟踪迭代数识别不收敛算法）；④硬件级对比——把用户实现与预验证 golden reference（框架内维护的可信 HLS 实现）自动对比图特定架构指标：cross-SLR 连接数与顶点分区大小（静态代码分析提取）、SLR 利用率分布与各模块 LUT/FF 占用（Vitis 综合报告提取）、FIFO 深度分配（暴露过度缓冲）。
- 与 C 级模拟的本质区别：标准 HLS 把图模型编译为 C 数组+for 循环，抹掉图结构语义（gather 变成嵌套 for + 指针冲突）；C 级模拟器只报"某数组地址的指针冲突"，开发者需人工反推回图拓扑并重写 C。GH-Scope 在保留图语义的 IR 级模拟，直接指出冲突的源/目的顶点。
- 论文量化（Pitfall II 的解决）：R24+PR 模拟 1,779s（C-Sim）→ 8.29s（GH-Scope），215×；全图 PR/CC/SSSP 平均 301.6×；调试时间（32K 节点/512K 边 SSSP）——算法失败 HLS 仿真 ~6h → ~0.04s（~455,000×）、流类型不匹配 73m40s → ~0.02s（~186,000×）、参数不匹配 13m13s → ~0.02s（~33,000×）。对比的 C-Sim 是 Vitis C-Sim（论文引用的 cycle-accurate C 模拟器 [16] 未开源、作者报告其速度与 C-Sim 相当 ~1×）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在编译框架中的位置：GH-Architect 之后、Vitis 综合之前/并行的验证通道；输入 GH-Architect 产出的优化 IR，输出验证结论。运转流程例（16-bit SSSP bug 场景）：开发者把 32-bit 改 16-bit 后，GH-Scope 在 LiveJournal（4.8M 顶点、直径>65535）上 IR 模拟→overflow 检测在距离 65540 回绕为 4 时即时报告具体源/目的顶点与属性→与 32-bit golden reference 逐顶点对比定位发散点→静态结构检查 + 架构指标对比（如新分区策略引入 40% 更多 cross-SLR 传输）给出纠错建议——全程亚秒级，替代"注释 pipeline stage→重跑 Co-Sim→二分"的数小时循环。
- 编译框架视角要点：把"验证"从硬件仿真提升为 IR 语义解释 + golden 对比，是图专属（结构属性相关）验证而非通用 HLS 验证；`--simulate-json <dsl> <graph.json> [max_iters]` 即其 CLI。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Rust 解释器 + 静态分析器，GH-Scope 部分无需 FPGA（任意 x86-64 Linux、≥16GB RAM、Rust toolchain 即可复现）；复现脚本 ae_fig10.sh（模拟速度）与 ae_tab3.sh（调试时间）。开源：https://github.com/pku-lemonade/Graph.hls（MIT）。
- 使用：开发者在 DSL/IR 改配置后先跑 GH-Scope（类型检查+循环依赖+IR 模拟+baseline 对比）获得亚秒级反馈，再决定是否进入 4–6h 的 Vitis 综合；对算法正确性验证与跨 SLR/资源类架构问题在综合前拦截。价值：把"综合后才发现错误"的 synthesize-and-hope 流程变为"综合前正确"的快速迭代。

涉及论文标题：
- Graph.hls: A Compiler Framework for Composable Graph Accelerator Design
