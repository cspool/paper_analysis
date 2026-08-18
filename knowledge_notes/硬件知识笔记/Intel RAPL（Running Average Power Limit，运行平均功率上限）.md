## Intel RAPL（Running Average Power Limit，运行平均功率上限）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- RAPL 是 Intel CPU 内置的硬件功率管理接口：提供 per-package/per-core 的功率测量（energy counter，可换算平均功率）与功率封顶（power capping）能力，通过 MSR（Model-Specific Register，如 MSR_PKG_POWER_LIMIT、MSR_PKG_ENERGY_STATUS）暴露给软件。软件写入功率上限后，RAPL 硬件控制器在 package 内部通过动态调节电压-频率状态（DVFS）与功耗分配，保证处理器功耗保持在设定上限以下（存在毫秒级 window 的滑窗平均语义，故采样周期低于 50ms 时测量不可靠）。Linux 侧访问路径：/sys/class/powercap/intel-rapl/（sysfs 接口）或 msr 驱动直接读 MSR；常用工具 powercap 子系统、turbostat、rapl-read。
- PowerGrad（ISCA'26）把 RAPL 作为功率分配的最终执行器：Local/Hierarchical Controller 算出的每处理器功率上限经 RAPL 强制到处理器，RAPL 通过控制 V-f 状态使实际功耗低于上限。论文还依赖 RAPL 的功率测量作为功率模型验证的 ground truth（每 100ms 比较模型预测与实际 RAPL 功率，Legacy AAE 4.1%）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- RAPL 在 PowerGrad 控制环中的运转流程（每 100ms 周期）：
```
① Gradient Estimator 读 RAPL 能量计数 → 换算实测功率 P[i]（模型验证/状态上报）
② Local Controller 跑 Algorithm 1 → 新功率上限 PL'[i]（梯度驱动 + 节点上限校正 + fmin 防饿死）
③ 把 PL'[i] 写入 RAPL MSR/sysfs → RAPL 硬件在 package 内调 V-f 状态使功耗 ≤ PL'[i]
④ 处理器以新 V-f 状态继续执行 ML 推理 → 下周期①重新采样形成闭环
```
- 例子（Legacy 双 CPU 节点，55W 节点上限）：Local Controller 决定把处理器 1（Llama-high）上限设为 35W、处理器 2（Llama-low）设 20W → RAPL 对每个 package 封顶 → Haswell E5-2660 v3 的核自动降频/升频维持上限 → Llama-high 因功率更足 prefill 更快，Llama-low 内存密集损失小 → 平均/P95 延迟下降。硬件约束：RAPL 采样周期快于 50ms 不可靠，故 Local Controller 周期取 100ms。
- 与 Power Sloshing 的差异：Power Sloshing 用 GPU 频率缩放（NVML）+ CPU power cap 接口在服务器内部分配；PowerGrad 用 RAPL 对 CPU package 直接封顶（GPU/加速器因 kernel 执行中不可动态读计数器，论文未支持）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Intel 固件 + 片上电源控制单元（PCU）实现；软件接口三层——MSR（最底层）、Linux powercap 内核子系统（/sys/class/powercap/intel-rapl/rapl:N 的 constraint_0_power_limit_uw 等属性）、用户态工具/库（turbostat、likwid、PAPI）。使用：数据中心功率管理（把 RAPL 上限设为节点功率预算的一部分）、DPC（Demand Response）/超订场景的功率封顶、研究系统（PowerGrad 的功率分配执行器、功率测量 ground truth）。写上限需 root；RAPL 封顶粒度通常到 package（含核+uncore），部分平台支持 per-core（PP0/PP1）。注意：RAPL 不直接暴露"设某个频率"，而是让硬件在功率上限内自主选择 V-f 组合——这对梯度驱动功率管理恰好合适（上限即分配变量）。

涉及论文标题：
- PowerGrad: Hierarchical Power Management for Power-Limited ML Inference Clusters
