## DVFS（Dynamic Voltage and Frequency Scaling，动态电压频率调节）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- DVFS 是芯片动态调节工作电压与时钟频率的电源管理技术：在 TDP（热设计功耗）约束下，通过降/升 V 与 f 保证可靠执行、提升性能或节省能耗。功耗近似 P = α·V²·f（动态功耗），因此小幅降频可显著省电，而升频提升吞吐。GPU 上 DVFS 由设备自身独立管理（per-device），即使同型号同节点内每颗 GPU 的温度/频率策略独立执行，造成跨 GPU 差异——这正是热致掉队的来源。
- Lit Silicon 论文（ISCA'26）用 amd-smi 实测：8×MI300X 节点内频率最高/最低相差 1.062×，且与温度排名（1.155× 差异）几乎一致；还观察到个别 GPU 的 DVFS 在温度超阈值后过度降频（非最热但频率最低）。论文的功率模型把主动功耗简化为 P_active = M·f（M=αV²，假设电压/温度变化可忽略），并假设 f = ρ/t（运行时与频率反比），据此推导功率与吞吐收益。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程：温度传感器上报 → DVFS 控制器（在 TDP 内）决定 f,V → 时钟发生器更新核心时钟 → GPU 吞吐变化。节点级视角：每 GPU 独立 DVFS → 频率分布不均 → straggler（低频）拖慢通信同步点 → C3 重叠下的 leader 也被拖慢（Lit Silicon）。论文 Insight 5（性能模型推导）证明：由于 straggler 在 varying-overlap kernel 上反而是最快的，提升 leader 上的重叠 kernel 速度无法解决 Lit Silicon——性能只受跨 GPU 频率差影响，对齐频率（让所有 GPU 同频）即解决。
- 例子：GPU-A 温度 95°C 被降频到 1.5GHz，GPU-B 温度 82°C 保持 1.6GHz；同一 AllGather kernel 在 GPU-A 上耗时更长，GPU-B 早开始却要等 GPU-A；若通过功率上限给 GPU-A +15W、GPU-B -15W/8，两者频率拉齐，lead 归零，节点吞吐提升。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：芯片内置的电源管理单元（PMU/CMU）+ 温度传感器 + 时钟树；软件接口：amd-smi / nvidia-smi 查询与设置（如 amd-smi set 功率上限，nvidia-smi -pl 功率限制、-lgc 锁频）。使用：数据中心运维把 DVFS 作为节点级功率管理的一部分——Lit Silicon 论文用"功率上限调整"（GPU-Red/GPU-Realloc/CPU-Slosh 三种用例）间接控制频率：功率上限比直接锁频更可预测（论文引 [49]）。注意事项：DVFS 的过度降频行为（温度超阈值后的保守策略）是硬件固有的，软件只能通过功率分配间接影响。
- Power Sloshing 补充视角（ISCA'26 推理场景）：DVFS 是“模块级功率预算”的执行器而非管理目标——GPU 频率缩放（NVML 等接口）与 CPU 频率缩放/功率封顶共同把软件设定的功率预算落到硬件。论文发现生产 AI 推理服务器默认行为：CPU 恒在最大频率 f_CM，GPU 仅在功耗逼近 TDP 时被 driver 内置 DVFS 被动降频（最低 55% f_GM）；模型加载后即使无查询 GPU 也回到 f_GM（空闲浪费）。论文用频率作为直接控制杠杆（不直接设功率上限，以隔离 vendor 内置电源管理干扰），把 DVFS 重新定义为“动态服务器级功率预算的执行机构”：u_G 低于目标区间时逐档降频（收获功率）、高于时单步拉满（防突发 SLO 违反）。频率可调范围：GPU 53%-100% f_GM（driver 离散档），CPU 71%-100% f_CM；频率迁移硬件耗时 100µs-数 ms，故控制环延迟由 100ms 采样周期主导。协调 DVFS（f_C=F(f_G) 线性映射）用于非功率受限的能效优化模式，避免单独监控 CPU 的开销。
- PowerGrad 补充视角（ISCA'26，CPU 集群场景）：DVFS 作为 RAPL 功率封顶的底层执行机制——PowerGrad 不直接设置频率，而是由 RAPL 在功率上限内自主选择 V-f 组合，DVFS 是"功率上限→实际运行状态"的硬件转换层。梯度估计的三个近似之一即"核心电压是频率的二次多项式 V(f)=c₀+c₁f+c₂f² [3]"，使模型可沿 f 用链式法则微分（∂V/∂BIPS=∂V/∂f·(∂BIPS/∂f)^(-1)，其中 ∂BIPS/∂f(f^(t))=(util·CCPI)/CPI²−util(1−util)/CPI）；性能模型 CPI(f)=CCPI+MCPI·f/f^(t) 假设 MCPI 随频率线性缩放（内存访问延迟相对固定、时间戳随频率变化）。频率范围例子：Legacy（Haswell E5-2660 v3）梯度精度实验在 1.2–2.6 GHz 间以 100MHz 随机游走采样各 V-f 工作点；fmin 防饿死保护取 cpuinfo 报告的最低频率。注意 DVFS 功耗超线性（P_active∝(V^γ+V)，γ 由不同电压下行为拟合）——这解释了"核心频率已高时计算密集负载也可能低梯度"。
- PowerWeave 补充视角（ISCA'26，空间 DVFS）：DVFS 的空间粒度是核心论题——现代 GPU（B200）只暴露设备级单一频率域，而 PowerWeave 论证应把 DVFS 细化到 GPU 内部空间域（per-TPC 直至 per-SM 148 域），每域由独立 on-die DLDO 稳压器供电、可独立调频。论文把"设备级 DVFS"作为 baseline 缺陷：一个全局频率被迫跟随最苛刻的 prefill 内核或租户，decode/低负载区域被过度供给（动态功耗 ~f³，浪费显著）。硬件侧：B200 默认最大频率运行、功率超 TDP 时固件被动降频（热节流）；Blackwell 频率切换延迟 ≈10–100µs（Hopper 为 ≈10–100ms）。实现层面用 NVML 下发每 GPU 频率、DCGM 4.2.2 测能量；因真实 GPU 无细粒度域，用多 GPU+TPC 分配仿真空间 DVFS（隔离/同卡分区/MIG 三种争用建模，SLO 按争用比例保守放大）。Blackwell 更快的频率切换使 10-100µs 级控制成为可能，是 PowerWeave 可行的硬件前提之一。
- TimeGaps 补充视角（ISCA'26，CPU P-state 切换的 halted 时间）：DVFS 的 P-state 切换代价在本文被提升为侧信道观测源——切换期间 CPU 核因电压不稳定停止执行指令（Intel Power Management Guide 文档化，P-state 切换延迟约 12µs，第 3 代 Xeon 起优化到近 0），这段 halted 时间（TimeGap）在 Skylake 系微架构上跨所有核同步（起止差平均 46.21 cycles/0.69%），且在空闲系统上占比 >1% 总时间（i7-9750H 10s 中 1.53%）。不同切换对产生特征性时长（4.0→4.1GHz 约 35k cycles，4.4→4.3GHz 约 52k cycles），攻击者用 rdtscp 阈值跳变即可感知，无需 cpufreq 或 MSR 特权。固定 CPU 频率可消除 CPU 型 TimeGaps，但 iGPU DVFS 经包级 PMU 协调仍产生 halted 时段（见"iGPU 动态频率切换（iGPU DVFS / GT frequency transition）"条目）。

涉及论文标题：
- Lit Silicon: A Case Where Thermal Imbalance Couples Concurrent Execution in Multiple GPUs
- Power Sloshing in Compound Servers for Large-Scale AI Inference Workloads
- PowerGrad: Hierarchical Power Management for Power-Limited ML Inference Clusters
- PowerWeave: Unlocking Energy-Efficient ML on GPUs with OS-Level Spatial Power Management
- TimeGaps Channels: Exploiting CPU Halted Time for Fun and Profit
