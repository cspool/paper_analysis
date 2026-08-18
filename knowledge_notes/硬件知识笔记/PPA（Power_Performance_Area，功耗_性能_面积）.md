## PPA（Power/Performance/Area，功耗/性能/面积）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PPA 是芯片设计三要素——功耗、性能（延迟/吞吐）、面积——的设计指标三元组，加速器评估与 DSE 的核心输出。NeRArch-Sim 的模块化硬件加速器"reports power, performance, and area (PPA) metrics"，调度器输出端到端 PPA（延迟→FPS、面积、功耗），并支持按 DSE 目标（如 energy-delay product 与面积）搜索配置。表 VI 给出单模块 PPA 对比（NeRArch-Sim HLS 估计 vs 全 ASIC post-layout：如 ICARUS MLP Engine 面积 5.9/6.3×10⁶ µm²、功率 4.0/4.2×10⁵ µW；CICERO NPU(24×24) 面积 3.1×10⁵ µm²、功率 7.6×10⁴ µW），17 模块面积/功率误差 4.72%~9.33%、延迟全一致。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- PPA 在 NeRArch-Sim 中的产出链路：算子图（execution_dag.pkl）+ 硬件 JSON → 映射/算子级/系统级调度 → 按式 1（duration = max(n_op/Θ_hw·s_comp, v_off/B_hw·r_bytes)）与 memory-aware 模型（SRAM+Ramulator DRAM 时序）累加周期 → 得 FPS；硬件模块经 HLS/ASIC 流程得面积与功耗 → 汇总为端到端 PPA 报告（HTML dashboard：PPA 卡片、调度 Gantt、per-SRAM 流量）。性能指标实例（表 VII，Lego 场景）：ICARUS 面积 6.9mm²/FPS 0.02、NeuRex 20.3mm²/18.6、CICERO 0.33mm²/326.4、GSCore 3.6mm²/182.2、GBU 1.9mm²/172、Uni-Render 15.66mm²/63；PSNR 复现 23.5~34.4。跨加速器对比（表 X，Unbounded360）：除 3DGS 外多数 pipeline 无法达 30 FPS，CICERO 用最少资源获较高 FPS，CICERO+（放大）比 GSCore-（缩到相似 FPS）更省资源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：面积/功耗来自 HLS（Catapult）估计或全 ASIC 流程（Synopsys Fusion Compiler + PrimePower）、SRAM compiler、DRAM 用 Ramulator 时序 + 28nm HPCP 节点经 DeepScaleTool 归一化基线；性能来自调度器的周期模型。使用：DSE 时改硬件 JSON（buffer/PE/并行因子/精度）秒级重算 PPA；表 XI 显示单设计点端到端编译 47.7~79.2 秒（instrumentation 22.8~45.2s、mapping 2~5.3s、op-level 7~24.1s、sys-level 1~21.1s），硬件特性离线预计算成查找表，硬件-only DSE 只需调度器部分（约 1 分钟/点）。相关：知识库硬件笔记的 NMP Logic Die 条目亦以 PPA 评估逻辑 die 面积/能效。

涉及论文标题：
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators
