## 硬件性能计数器（Hardware Performance Counters：BIPS / CPI / MCPI / CCPI）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 硬件性能计数器是 CPU 内部的可编程计数寄存器，统计微架构事件（执行指令数、周期数、缓存未命中、分支误预测、uop 数、访存 stall 等），是软件透明的运行时观测接口（Linux perf_event、Intel PEBS/固定计数器、AMD 类似机制）。PowerGrad（ISCA'26）用六个计数器在线刻画工作负载并构建功率/性能模型：instruction-count（指令数）、cycle-count（非空闲周期）、uops.executed、cache-misses、branch-misses、ldm_stalls_pending（内存 stall 周期）；Accelerated（Emerald Rapids）额外读 exe.amx_busy（AMX 单元忙碌周期）与 fp_arith_inst_retired.vector（向量指令数）。
- 由计数器导出的核心指标（论文公式 1-2，每秒粒度）：BIPS（billions of instructions per second，指令吞吐）、BCPS=f×util（每秒活跃周期）、CPI=BCPS/BIPS（每指令周期数）、MCPI=ldm_stalls/BIPS（内存导致的 CPI 分量）、CCPI=CPI−MCPI（计算导致的 CPI 分量）。这些指标把工作负载分解为"计算密集 vs 内存密集"，是性能梯度 ∂BIPS/∂P 估计的输入。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 计数器→模型→梯度的运转流程（每 100ms，per-core）：
```
读计数器: BIPS, util(=非空闲周期/总周期), ldm_stalls, 频率 f^(t)
BCPS ← f^(t)·util;  CPI ← BCPS/BIPS
MCPI ← ldm_stalls/BIPS;  CCPI ← CPI − MCPI
CPI(f) ← CCPI + MCPI·f/f^(t)        # 性能模型：MCPI 随频率缩放、CCPI 不变
P_active(V,E) ← Σ_i(w_i E_i^(t))·(V^γ+V)   # 功率模型：计数器事件加权
∂BIPS/∂P ← 微分上述模型 (8)-(13)
```
- 例子（Llama-3.1-8b 在 Haswell 上）：prefill 阶段指令密集、ldm_stalls 占比低 → CPI 低、CCPI 主导（计算密集）→ 梯度高；decode 阶段读模型权重导致 cache-misses/ldm_stalls 飙升 → MCPI 主导（内存密集）→ 梯度低。控制器据此把功率从 decode 阶段处理器转给 prefill 阶段处理器。AMX 计数特殊处理：1 个 AMX busy 周期计为 16 条指令（等价 16 条向量指令的操作量 [19]），使 BIPS 度量对 AMX 加速的 Emerald Rapids 公平。
- 关键硬件约束：需要"kernel 执行期间动态读计数器"的能力——NVIDIA profiling 工具只能在 kernel 完成后查询计数器，故 PowerGrad 无法用于当前 GPU/加速器，只能在 CPU 上评估。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：CPU 微架构内嵌计数器阵列 + 性能监控单元（PMU，含 MSR/固定计数器与可编程计数器）；软件经 perf_event_open/Linux perf 采样、PAPI、RAPL 能量计数读取。使用：性能剖析（瓶颈定位）、在线自适应系统（PowerGrad 的梯度估计、PPEP 的在线功率/性能预测）、调度器与监控。训练用途：PowerGrad 按 PPEP 工作流用六个计数器离线拟合功率模型系数（Legacy 用 PARSEC 3.0、Accelerated 用 TorchBench 触发 AMX/向量指令）。精度：计数器事件与 BIPS 成比例假设（E_i=e_i·BIPS）是梯度估计的三个近似之一，短 kernel 上 R² 较低（平均 0.501）但迭代优化只需近似梯度。
- SPEC CPU2026 的 PMC 使用（PMC Characterization，论文 §VII-A 与附录 B/C）：在 AMD EPYC 9755（Zen 5，2.7 GHz/Boost 4.1 GHz，GCC 15.2 -O3，Ubuntu 24.04）上以单 copy 运行全部 SPECrate benchmark、以 128 线程运行 SPECspeed benchmark，采集 IPC 与 stall 分布做 Top-down 特征化，产出每 benchmark 的 Frontend/Backend/Lost/Retiring 四类占比表（Table V-VIII），用于 benchmark 选择与瓶颈画像（如 709.cactus 前端压力高、750.sealcrypto 高 IPC 5.23 且 lost≈0）。IPC 与 stall 分布还配合 perf 时间序列图与 BBV recurrence 图做相位分析。

- TimeGaps 补充视角（ISCA'26，halted/unhalted 判定）：本文用两个 PMC 事件 CPU_CLK_UNHALTED.THREAD（目标核执行指令的周期数）与 CPU_CLK_UNHALTED.REF_TSC 把 timestamp jump 分类为 halted（CPU 挂起、不执行指令）与 unhalted（中断/SMM 等，CPU 仍在执行）：在 rdtscp 检测到跳变的间隔内若 unhalted 计数器不推进 → halted jump（TimeGap）。i7-9750H 上 94.3% 的跳变是 halted，时长多 >30k cycles，而 unhalted 多 <10k cycles。分类实验用 isolcpus+tickless 隔离核消除中断得到无噪声环境；攻击侧则用 SegScope（GS 段寄存器）做非特权过滤。PMC 还用于验证 iGPU 频率切换与 CPU P-state 产生相似的 halted 时长分布（Obs. 7），并支持经 perf_event_open 读 PMU 事件 power/energy-gpu 监测 iGPU 功耗。

涉及论文标题：
- PowerGrad: Hierarchical Power Management for Power-Limited ML Inference Clusters
- SPEC CPU: The Next Generation
- TimeGaps Channels: Exploiting CPU Halted Time for Fun and Profit
