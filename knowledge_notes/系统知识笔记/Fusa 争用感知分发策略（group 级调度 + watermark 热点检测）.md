## Fusa 争用感知分发策略（group 级调度 + watermark 热点检测）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Fusa 的核心请求调度策略：把 RNIC 锁定表的每个槽按地址多取 g 位细分为 group（默认 8,192 组，槽粒度下再分 2^g 组），每组维护 64-bit 请求计数器 + 1-bit 分发位（1=卸载到服务端 CPU，0=走 RNIC 硬件）。每 stage（1 s）以全组平均请求数为 watermark 判定争用：把组按请求数降序扫描，最热组标记为卸载，直到剩余组均低于 watermark 或累计卸载量超过服务端 CPU 处理容量 C。动机：锁定表槽争用集中在极少数槽（论文热力图：Zipfian θ=0.99 时少数槽承担绝大部分请求与高延迟），组级调度只卸载热点子集，其余请求保留 RNIC 快路径。Web 证据：工业热 key 探测（京东 JD-hotkey 的自适应阈值 + 滑动窗口）、MIDAS/DualMap 的热点阈值迁移与 watermark 思路同源。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 每 stage 的决策伪代码（据 §IV-B）：
  ```
  r_i = counters[group_i].per_second   # 每秒请求数
  watermark = mean(r_i over all groups)
  sorted = sort groups by r_i desc
  onload = 0
  for g in sorted:
      if r_g < watermark: break        # 剩余组无争用
      strategy[g] = 1                  # 卸载到 CPU
      onload += r_g
      if onload > C: break             # 达 CPU 容量上限
  ```
- 数据路径：group_id = address % 8192；strategy[group_id]==1 → Fusa-RPC 到 server CPU；==0 → RNIC 锁定表。容量约束 C 源自 server 线程处理能力（单线程约 2.5 Mops/s），防止过度卸载引发排队延迟。论文图 7 示例：4 组、C=15 时卸载请求数最大的两组。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 计数器 64-bit 拆为 32-bit 每秒请求数 + 32-bit 在途请求数，每 stage 归零每秒字段；组数 8,192 是元数据开销（65 KB Fusa-SHM）与调度粒度的平衡点。低争用静默性：YCSB-B（5% 更新）不触发卸载，Fusa 吞吐为 RNIC-Only 的 97.5%；均匀分布下 Fusa 与 RNIC-Only 持平，θ=0.99 时 4.8×。动态性：热点转移时 48 µs 内完成策略切换（Exp#4），吞吐在 1 s 内恢复。类似阈值/水位线机制可推广到任何"热点子集卸载"型异构调度（如 KV 热 key 路由、LLM 请求热点迁移）。

涉及论文标题：
- Breaking Barriers in Atomic Scaling: A Hardware–Software-Collaborated Framework to Deconstruct RDMA Atomic
