## 数据流调度（mapping engine + operator-level + system-level 三层调度，DAGS 依赖感知贪心调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 数据流调度指把算子图按依赖与硬件资源编排成可执行计划的过程，覆盖"算子如何映射到硬件单元、每个算子何时在哪个单元执行、数据如何搬移"。NeRArch-Sim 的模块化数据流调度器把神经渲染加速器的调度问题分解为三层：**mapping**（算子-硬件绑定，mapping engine 按统一分类学匹配算子与硬件模块，多候选取最高吞吐、多实例均衡避免瓶颈，不匹配报 mismatch，输出 mapped IR）、**operator-level scheduling**（各硬件模块内的局部优化：查优化库选 domain-specific 优化策略，输出带 start_cycle/duration 的 operator-scheduled IR）、**system-level scheduling**（跨模块全局编排：Dependency-Aware Greedy Scheduler，DAGS，消费 operator-scheduled IR 生成最终执行计划与 PPA）。本库既有"分层静态-动态协同调度（Hierarchical Scheduling）"条目是静态/动态分层概念，与此处的"三层调度器"是不同上下文。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 算子级时长模型（式 1）：duration(op) = max(n_op/Θ_hw · s_comp, v_off/B_hw · r_bytes)，n_op 处理元素数、v_off 通信量、Θ_hw 硬件吞吐、B_hw 有效带宽（来自 mapped IR）；s_comp/r_bytes∈(0,1] 是优化库带来的计算/访存削减因子（如 tile culling 优化用 active Gaussian ratio 得 s_comp）。DAGS 伪代码（Algorithm 1）：
```
Q ← 所有就绪源算子
while Q ≠ ∅:
    for op in Q:
        d = 后继算子数(op)          # 启发式1：优先解锁整阶段
        c = 关键资源影响(op, G, S)   # 启发式2：异构资源需求(n_op, v_off)
        score(op) = α·d + β·c        # α/β 可配置建模不同加速器
    sel = argmax score(Q)
    st = FindEarliestSlot(sel)       # 尊重依赖/带宽/SRAM bank-端口冲突
    SS[sel] = (st, S[sel].duration)
    更新就绪队列
```
- 例子（GSCore 3DGS）：CCU→(排序 QSU/BSU)→VRU 链；mapping 把 culling 算子绑 CCU、排序绑 QSU/BSU、混合绑 VRU；operator-level 给各算子算 duration（表 VI：CCU 128/128、BSU 4/4、QSU 64/64、VRU 192/192 cycle），system-level DAGS 按依赖与资源约束排全局计划。三种代表性 dataflow 策略都能被同一调度算法捕获：ICARUS weight-stationary、NeuRex pipelined encoding-MLP overlap、GSCore tile 化 sorting-rasterization overlap——通过算子图结构+硬件配置+memory binding 表达，不改调度算法。DAGS 目标建模保真而非最优性：依赖与硬件约束相同则关键路径不变，故端到端延迟/能耗在误差界内。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：调度器 C++/Python（NeRArch-Sim 的 Scheduler/ 目录），CLI 分步 `./nerarch_sim map <dag.pkl> <hw.json> -o mapped_ir.json` → `./nerarch_sim schedule mapped_ir.json --hardware <hw.json> -o scheduled_ir.json` → `./nerarch_sim report scheduled_ir.json --format html`；调度延迟（表 XI）：mapping 2.0~5.3s、op-level 7.1~24.1s（ICARUS/NeuRex 算子多、GSCore/GBU/Uni-Render op-level 高因 inter-operator 调度复杂）、sys-level 1.0~21.1s（随算子数）。优化库按三维分类（优化类型 reuse/skip/low-bit × 作用域 element/region/frame × 决策准则 boundary/threshold），如 element-level Gaussian skipping（GSCore/GS Processor）、per-ray early termination（NeuRex/GSCore）、region-level restricted hashing（NeuRex）、tile-based Gaussian processing 与 bitmap culling（GSCore）、frame-level sparse radiance warping（CICERO）、threshold 触发的 sensitivity 精度降低（SRender）。使用价值：秒级端到端调度支持快速 DSE、决策可解释、接口可换更复杂调度算法（扩展新加速器只需新增算子与其调度器，<300 行 C++）。

涉及论文标题：
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators
