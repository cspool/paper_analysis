## Frequency-Latency Scaling（频率-延迟缩放模型：kernel 级曲线合成 per-application 缩放函数）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Frequency-Latency Scaling 模块把 Online Kernel Profiler 收集的 per-kernel 频率-延迟曲线，合并成 per-application（每租户/每域）的频率-性能模型，回答"降到某个频率会让应用端到端延迟慢多少"。核心是一阶 Taylor 近似启发式：对目标性能退化 k，调整后频率 f(k) = f_max / S，其中 S = 1 + k / Σ(s·w)：w 是每 kernel 占应用总运行时间的权重（在线更新，跟踪 prefill/decode 比例漂移），s 是每 kernel 的频率敏感度因子（曲线斜率，刻画"降频 1% 延迟涨多少"，由指令混合决定、profiling 后固定）。
- 直觉：权重 w 平衡应用内 compute-bound（高敏感）与 memory-bound（低敏感）工作的比例；敏感度 s 防止低敏感 kernel 把目标频率拖到频率敏感 kernel 无法接受的低点。per-application 独立建模使每租户/每阶段可独立缩放频率，且不绑定固定 slowdown 假设，能适应负载波动与 SLO 变化。论文消融（Fig. 12）：完整系统平均偏差 1.7%（最坏 5.2%，且平均低于目标——不会高估性能损失）；去掉 live weight updates 平均偏差升到 4%、最坏 75%（高负载下预测严重失准）；去掉 sensitivity 平均 4% 但 10%/20% slack 处过度估计损失导致 SLO 违反（最坏 10.6%）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算过程（伪代码级）：
```
# 对每个应用(域)：
# 从 profiler 得到每 kernel 的 {latency(f_max), latency(f_i)} 曲线
s_k = 平均每降频 1% 的延迟增幅(曲线斜率)        # 固定
w_k = kernel_k 运行时间 / 应用总运行时间          # 在线更新
S(k) = 1 + k / sum(s_k * w_k)                    # k = governor 允许的性能退化
f_target = f_max / S(k)                          # DVFS Controller 下发的频率
```
- 例子（disaggregated prefill，Llama-3.1-8B）：prefill 应用里 GEMM 类 kernel 权重 w 大、敏感度 s 高 → 允许 20% 退化时 S≈1.15，频率仅从 1965MHz 降到 ~1710MHz（TTFT 仍满足 SLO）；decode 应用里 memory-bound kernel 权重 w 大但 s≈0（降频不加速）→ 同样 20% slack 可把频率降到很低（~915MHz 级），TPOT 几乎不变。Governor 每监控窗口更新 slack：s₂ = ((1−s₁)×l₁)/SLO（由当前延迟反推最大频率下理论延迟再除以 SLO）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：模块位于 PowerWeave Interposer 地址空间（Rust），输入 per-kernel 画像与 governor 的 slack 指令，输出每域目标频率经 NVML 下发。使用：profiling 阶段结束后进入 operating 阶段，持续用 live weight 更新跟踪 workload 构成；每 kernel 完成后对照预测延迟，偏离超 5%（profiling-threshold）重启 profiling。该模型与 serving 框架、应用解耦：应用只通过 governor 声明 SLO 与上报指标，频率策略在控制平面内实现——同一套机制可支撑 latency-driven / per-tenant / throughput-balancing 等多种策略。

涉及论文标题：
- PowerWeave: Unlocking Energy-Efficient ML on GPUs with OS-Level Spatial Power Management
