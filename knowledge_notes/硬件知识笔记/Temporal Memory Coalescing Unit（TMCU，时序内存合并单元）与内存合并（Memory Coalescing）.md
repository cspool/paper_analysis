## Temporal Memory Coalescing Unit（TMCU，时序内存合并单元）与内存合并（Memory Coalescing）

术语解释
Memory coalescing：把多个窄访存请求合并为更少、更宽的 cache line 事务，提升带宽利用率、降延迟与能耗。GPGPU 依赖 warp 内 32 线程同拍发请求的"空间同时性"完成合并；DICE 流水派发丢失该同时性，TMCU 以"时间相邻请求"的缓冲合并 + 超时 flush 恢复合并能力。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GPGPU coalescing：同一 warp 内线程访问同一 128B line 的连续 4B 元素 → coalescer 合并为一次 128B 事务；合并条件 = 请求类型相同 + 地址对齐（落在同一 sector/line）。MSHR（Kroft 1981）不是合并机制——它只跟踪未完成 miss 以支持非阻塞缓存，不合并邻接请求。TMCU（Fig.7f）：位于 Dispatcher 与 L1 之间的合并缓冲 + max_interval 定时器，逐周期输出命令；请求可合并（can_coalesce：类型与地址对齐）则并入缓冲命令，否则先弹出旧命令再以新请求为基；超时强制弹出保证前向进展。DICE 配置 max_interval=8（= 32B L1 sector / 4B 访问）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
逐周期逻辑（论文 Algorithm 1）：
```
function OUTPUTCOMMAND(in_req)     // 每周期执行
  if coalesce_buffer.is_valid() then timer.decrement()
  if timer.timeout() then buffer.pop(); timer.reset(max_interval)
  if in_req.is_valid() then
    if not buffer.is_valid() then buffer.initial(in_req)
    elif buffer.can_coalesce(in_req) then buffer.coalesce(in_req)
    else buffer.pop(); timer.reset(max_interval); buffer.initial(in_req)
```
Annotations：timer 仅在缓冲有效时递减，超时即弹出入 L1；can_coalesce 检查请求类型与地址对齐。效果：连续线程访问连续地址时合并效果等价传统 coalescer；写通缓存策略下合并写请求还显著减少互连流量与拥塞（NN/BFS-2/BPNN-1/2 上 TMCU 单独带来 1.41–2.45× speedup，L1 能耗与基线相当）。局限：in-order 时序处理下，thread 0 与 2 可合并而 thread 1 不可时漏合并（优化良好的程序中罕见）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：小合并缓冲 + 定时器 + 类型/对齐比较逻辑（DICE 有 RTL 实现，占 CP 面积 A3 部分）；GPGPU 侧用空间合并（同拍请求），DICE 侧用时序合并。使用场景：一切流水/时序化发访存的 SIMT/CGRA 后端，保持与 cache 体系（sector 粒度）的匹配。Web sources：NVIDIA CUDA C Best Practices（coalesced access）；Kroft, ISCA'81（MSHR，论文引用 [24]）。

涉及论文标题：
- DICE: Enabling Efficient General-Purpose SIMT Execution with Statically Scheduled Coarse-Grained Reconfigurable Arrays
