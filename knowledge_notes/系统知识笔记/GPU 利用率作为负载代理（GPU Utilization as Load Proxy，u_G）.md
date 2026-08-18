## GPU 利用率作为负载代理（GPU Utilization as Load Proxy，u_G）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- GPU 利用率（GPU utilization，u_G）指 GPU 计算单元实际忙碌时间占比（0-100%），是最普遍可得的 GPU 硬件计数器之一（NVML 等 API 直接暴露）。作为"负载代理"（load proxy），是指在不访问应用级指标（每服务 QPS、SLO 目标）的情况下，用 u_G 推断服务器当前负载强度与资源受限状态。论文（Power Sloshing）实测：AI 推理负载下 QPS 与 u_G、GPU 功率三者强线性相关（三模型 QPS-util R² = 0.71-0.79、QPS-power R² = 0.84-0.94），因此 u_G 是低开销、应用无关、fleet 通用的负载信号。更进一步，u_G 与频率的关系满足吞吐守恒近似 f_G·u_G ≈ Q_w（固定 QPS 下提高频率→处理更快→利用率下降），故 u_G 能反映"GPU 是否功率/频率受限"：持续高 u_G 表明 GPU 需要更多功率（更高频率）换取性能，低 u_G 表明有功率 slack 可收割。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在电源管理闭环中的角色（Algorithm 1 的输入信号）：① 采样：每 100ms 读取 u_G（NVML）；② 判断：与目标区间 [u_min, u_max] 比较（按模型历史 P75/P90 选取）；③ 决策：u_G < u_min → 降 GPU 频率（收割功率）；u_G > u_max → 拉满频率（防 SLO 违反）；④ 功率预算导出：功率受限模式下 GPU 预算由 P_G(f_G, u_G) 决定，CPU 预算为残差。设计动机：数据中心 host 按设计不知道跑哪个服务、SLO 是多少、QPS 多少（编排层抽象掉应用指标），而 u_G 通用可得且与负载强相关，故作为 service-agnostic 的控制器唯一输入。注意区分：u_G 是"频率耦合"的信号——降频后同负载 u_G 反而升高（处理变慢），控制器必须结合 f_G 解读；这也解释了 Fig.9 中功率受限时 u_G 非线性陡升。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：从 GPU 管理库读取（NVML nvmlDeviceGetUtilizationRates / nvidia-smi --query-gpu=utilization.gpu），CPU 侧从 /proc/stat 或 cpufreq 统计。使用场景：除电源管理外，u_G 广泛用于容量规划、调度（如 Bullet 用 GPU 利用率衡量 prefill/decode 协同、μShare 用 SM 利用率衡量 kernel 共置）、监控告警。局限：u_G 是粗粒度（通常 100ms-1s 窗口平均）且不区分计算/访存瓶颈，论文仅用它做阈值控制而非精确建模。论文未开源相关实现；控制循环可与 NVML 示例代码（NVIDIA 官方 nvidia-ml-py）对照复现。

涉及论文标题：
- Power Sloshing in Compound Servers for Large-Scale AI Inference Workloads
