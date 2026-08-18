## Power Sloshing（功率舀取 / 模块级功率预算，Module-Level Power Budgeting）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Power Sloshing 是服务器级（server/module-level）动态电源管理机制：把一台 compound AI inference server（CPU+多 GPU）视作功率受限模块（可配置功率上限 P_M），在运行中把功率预算从"有闲置功率（slack）的组件"舀取（harvest）出来、重新分配给"当前功率受限/性能瓶颈的组件"，从而在固定模块功率包络下最大化 Performance/Watt。论文（Meta，ISCA'26 "Power Sloshing in Compound Servers"）实现为闭环控制器（Algorithm 1）：每 100ms 采样 GPU 利用率 u_G（应用无关信号，作为负载与"GPU 是否功率受限"的代理），与目标区间 [u_min, u_max] 比较——u_G < u_min 时把 GPU 频率 f_G 逐档下调（收获功率、降低功耗），u_G > u_max 时单步拉满 f_G=f_GM（保护突发负载的 P99 SLO）；功率受限模式下用 GPU 功率模型 P_G(f_G, u_G) 估算 GPU 稳态功率，反推 CPU 功率上限 P_C = max{0, P_M - ΣP_Gi} 并施加 CPU power cap；非受限（energy optimization）模式下用线性映射 f_C = F(f_G) 协调 CPU DVFS。实现手段：GPU 频率缩放（NVML 等）与 CPU power capping 等现有生产执行器，无应用代码修改。效果：1 小时动态负载 trace 下 SLO-Optimized 变体省电 11%、Power-Optimized 省 24%（贴近 Theoretical 下界 30%）；功率上限 sweep 下 Performance/Watt 最高 1.83×。与 Lit Silicon（ISCA'26）的 CPU-Slosh 区分：后者面向训练、由热致掉队驱动、用功率上限间接调频；本文面向推理、由利用率驱动、以频率缩放为主要执行器。
- 别名/近义：power harvesting（功率收割，从松弛组件回收闲置功率）、power budget reallocation（功率预算再分配）、server-level power budgeting（服务器级功率预算抽象 P_C + ΣP_Gi ≤ P_M）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（Algorithm 1 伪代码）：
```
输入: 模块功率上限 P_M; GPU 利用率目标区间 [u_min, u_max];
     GPU 频率档位集合 F_G（最大 f_GM）; GPU 功率模型 P_G(f_G, u_G); CPU-GPU 频率映射 F(·)
while 系统运行 do
    读取当前 GPU 利用率 u_G
    if u_G < u_min then  f_G ← Lower(F_G, f_G)          # 逐档降频，收获功率
    else if u_G > u_max then f_G ← f_GM                # 单步拉满，保护突发
    if 功率受限 then                                     # 预算再分配模式
        P̂_G ← P_G(f_G, u_G)                            # 估算 GPU 稳态功率
        P_C ← max{0, P_M - P̂_G}                        # 残差给 CPU
        SetCpuPowerLimit(P_C)                          # CPU 功率封顶
    else                                               # 能效优化模式
        f_C ← F(f_G); SetCpuFreq(f_C)                  # 协调 CPU DVFS
```
- 系统层面要点：GPU 是主要功耗源与性能瓶颈，故以 GPU 为控制锚点、CPU 预算由残差导出（降低监控开销）；每 GPU 独立控制（per-GPU，~100µs/100ms 开销），多 GPU colocate 时把松弛 GPU 的功率转给受限 GPU；tensor-parallel 组内建议统一频率防 straggler；100ms 控制周期匹配 P99 SLO（数十-数百 ms）量级；非对称调整（快升慢降）保证稳定性。负载代理依据：实测 QPS 与 u_G、功率线性相关（R² 0.71-0.94），且 f_G·u_G ≈ Q_w（吞吐守恒）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：host 上的轻量电源管理控制器（运行时 Python/守护进程），仅依赖标准 GPU 利用率计数器与频率控制接口（NVIDIA NVML 或类似 API，如 nvidia-smi -lgc 锁频 / -pl 功率上限）和 CPU cpufreq/power-cap 接口；要求无应用代码修改、无应用级指标（QPS/SLO）访问。目标区间按模型历史利用率百分位离线选取（SLO-Optimized=P75、Power-Optimized=P90），新模型无需重新 profiling（TABLE I：与 profile-based / RL 方法对比，无 per-model setup、即时适配、per-GPU 独立、部署开销最小）。论文未开源（Meta 生产系统，fleet 数据/负载不可公开），但机制可基于公开 NVML/cpufreq 接口按 Algorithm 1 在任何 NVIDIA 平台重实现；论文 PDF 公开于 Edinburgh Research Explorer。相关系统：PowerGrad（层次化功率管理，polynomial power model + 梯度分配）、Lit Silicon（节点级功率上限 GPU-Red/GPU-Realloc/CPU-Slosh）、PowerWeave（空间 DVFS）。

涉及论文标题：
- Power Sloshing in Compound Servers for Large-Scale AI Inference Workloads
