## 空闲感知内存调度与 IWE（Idleness-aware Scheduling, Idle Window Estimator）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
内存控制器在 CPU 优先原则下，分析 CPU 请求队列预测每个 DRAM bank 与内存总线的空闲时间窗口，把 PIM 命令精确插入窗口——既保证 CPU 延迟又最大化 PIM 吞吐。IWE（空闲窗口估计器）输出 window_bank[b]（每 bank 最早服务周期）与 window_bus（全部请求的最小服务周期）。窗口有两类来源：(1) 应用级请求稀疏造成的 bank 队列空闲间隔（Chopim 的 CPU-first 已利用）；(2) 多 bank 命令在共享总线串行化导致的"ACT 已发、数据访问未到"间隔——bank 行已打开但总线忙，内部带宽浪费（Fig.3c③，CPU-first 未利用）。COSM 把第二类窗口通过"推迟过早的 ACT"转化为 PIM 执行时间。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm 1（IWE 最早访问周期估计）：
```
REQ[] = 每 bank 最早到达请求
ready_cycles = [get_ready_cycle(r) for r in REQ]  # bank 三态：Row-Closed 需 ACT(+tRCD)、
                                                  # Opened-to-target-row、Opened-to-different-row 需 PRE+ACT(+tRP+tRCD)
t = cur_tick(); cr = cur_rank(); service_time = {}
while REQ.size():
    if 存在 r.rank==cr 且 r.ready<=t:      # FR-FCFS 特性1：同 rank 行命中请求连续处理
        r = 最早就绪(同 rank 请求); t += tBL
    else:                                   # 特性2：无就绪才跨 rank 切换（防 tRT_RS 惩罚）
        r = 最早就绪(全部); t = max(r.ready, t); cr = r.rank
    service_time[r] = t; REQ.remove(r)
window_bank[b] = service_time[该 bank 请求]; window_bus = min(service_time)
```
Annotations：利用 FR-FCFS 的两个特性（行命中连续处理、同 rank 分组）使估计近真实调度序且开销小（可适配其他调度策略，改估计逻辑即可）。Command Arbiter 的用法：CPU 调度器发 ACT 时，若该 bank 窗口 ≥ tRP+tRCD+至少一列 PIM 执行则推迟 ACT 先跑 PIM；PIM 执行中 CPU 队列非空时不立即 Pause，延迟到"不推迟下次 CPU 访问的最后周期"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：内存控制器内硬件模块（COSM 面积 0.0085 mm²，占 LPDDR5 控制器 7.4% 开销的一部分）；效果较 CPU-first：平均 PIM 性能 1.21×、多利用 37.0% 可用带宽（剩余 <1% 未用）。使用：任何共享内存 CPU-PIM 并发调度；对 SIMD PIM 需扩展为"多 bank 同时空闲"的窗口预测（未来工作）。与可抢占命令、解耦传输正交组合（消融的 All 配置）。

涉及论文标题：
- COSM: A Cooperative Scheduling Framework for Concurrent PIM and CPU Execution on Mobile Devices
