## 空间 DVFS（Spatial DVFS，GPU 内多电压-频率域）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 空间 DVFS 指把芯片/GPU 划分为多个独立的电压-频率（V-f）域，每个域可独立调节工作电压与时钟频率的电源管理方式，与传统的"设备级/整卡单一频率域 DVFS"相对。现代 GPU（如 NVIDIA B200）只暴露一个 device-wide DVFS 域：整卡所有 SM 共享同一 V/f 状态。空间 DVFS 的目标是把频率控制粒度细化到 GPU 内部空间（per-GPC、per-TPC 直至 per-SM 级频率域），使每个域按自己的算力/内存需求运行在合适的频率。PowerWeave（ISCA'26）是该方向的代表工作：提出首个基于 OS 级电源管理控制平面的空间 DVFS 系统，软件层用 Online Kernel Profiler + Frequency-Latency Scaling + DVFS Controller + Governor 协调每域频率；硬件层给出可行性成本模型（on-die DLDO 稳压器 + 电压域边界同步 + per-domain PLL，per-SM 148 域面积开销 <0.5% die）。
- 动机链：LLM 推理 prefill 是 compute-bound（对频率敏感）、decode 是 memory-bound（对频率不敏感）；多租户/agentic 流水线把计算画像差异巨大的模型堆叠在同一 GPU；多 die GPU（B200 双 die、MI350X 八 die）使这种空间异构不可避免。单一全局频率被迫跟随最苛刻的内核/租户，其余区域被过度供给，浪费动态功耗。相关先前机制：PCSTALL（ASPLOS'23，warp 级微架构信号的细粒度 DVFS）证明 per-compute-unit 频率控制可行但缺乏应用语义与 SLO 可见性。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件运转流程：GPU 的 SMs 按 GPC/TPC 组织（B200：9 GPC、74 TPC、148 SM），空间 DVFS 把 SM 群划分为 N 个独立电压-频率域，每域配一个 on-die 数字低压差稳压器（DLDO，输入 1.15V、输出 0.8–1.1V、256 级分辨率）独立供电、一个时钟域（每域可复用 PLL 或独立 PLL）。频率变化时：域内先 quiesce 流量（AXI-like VALID/READY 去断言）→ 电压域边界电平转换/隔离/同步 FIFO 保证跨域数据完整性 → 更新该域 V/f → 该域 SM 按新频率继续执行。软件侧（PowerWeave）每控制周期：kernel 完成时 Interposer 记录延迟 → DVFS Controller 按 Governor 给的 performance slack 为该域选频率 → 经 NVML 写入。示例：B200 上 disaggregated prefill，prefill 域保持高频率（1965MHz）满足 TTFT SLO，decode 域降到 915MHz 级利用 TPOT slack，整卡功耗显著下降（论文平均节能 28%、最高 38%）。
- 面积/功耗模型：ΔA_tot(N) = ΔA_reg(N) + ΔA_LS(N) + ΔA_PLL(N)（稳压器控制逻辑 + 边界同步 + 时钟生成）；per-SM 148 域总开销 <0.5% 的 1600 mm² die 面积；每额外域 boundary sync 0.0359 mm² 是主导项（比 regulator+clock 合起来大约一个数量级）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：真实 GPU 尚无细粒度频率域暴露给软件，PowerWeave 用"多 GPU + TPC 分配使总分配等于一个整卡"仿真空间 DVFS，并对比 isolated / same-GPU compute partitioning / MIG 三种争用模型（争用使 TTFT/TPOT 平均 +3%、最差 <+7%，SLO 按其保守放大）；频率经 NVML 设置，能量经 DCGM 4.2.2 测量（总能量=各 GPU 之和−按未分配 TPC 份额比例扣除的 idle 能量，B200 idle ≈140W）。硬件实施方面，业界已有多域 DVFS 先例：Intel FIVR（14nm 图形核心 per-block DVFS）、Zeppelin（AMD）等；论文用 mflowgen + OpenFASoC（DLDO 生成）+ ASAP7 7nm/130nm PDK 综合缩放得到面积证据。使用价值：在 SLO 不违反前提下按域差异化调频，把"整卡被最苛刻负载绑架"变成"各域各取所需"，同时消除跨租户热节流干扰。

涉及论文标题：
- PowerWeave: Unlocking Energy-Efficient ML on GPUs with OS-Level Spatial Power Management
