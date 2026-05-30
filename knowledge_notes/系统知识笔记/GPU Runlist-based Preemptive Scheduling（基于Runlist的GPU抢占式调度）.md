## GPU Runlist-based Preemptive Scheduling（基于Runlist的GPU抢占式调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

基于 Runlist 的 GPU 抢占式调度是 Capodieci et al. (RTSS 2018) 提出的一类 GPU 实时调度方法。核心机制：将 GPU 的硬件 runlist 作为调度接口——CPU 端 scheduler 维护 task 的就绪队列，按调度策略（如 EDF, Earliest Deadline First）选择最高优先级 task，将其 channel 插入 runlist 使 GPU HW scheduler 开始执行该 task。抢占通过重置 runlist 实现：当更高优先级 task 到达时，CPU scheduler 清空 runlist 上的当前 task channel，插入新 task 的 channel。该方法的优势是 GPU native scheduling 保证执行效率（无需 CPU 端排队），且 runlist 重置提供硬件级别的抢占能力。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

基于 Capodieci et al. 方法和 Bakita & Anderson 的修正：

```
原Capodieci et al.方法 (基于错误假设——所有GPU仅一个runlist):
  CPU Scheduler (EDF):
    while True:
      next_task = select_highest_priority_EDF(ready_queue)
      reset_runlist()                        # 清空runlist
      insert_channels(next_task, runlist)    # 插入新task的channel
      // 假设: 抢占compute runlist后task不再执行任何GPU操作

Bakita & Anderson修正 (基于R5, R7):
  CPU Scheduler (EDF + multi-runlist aware):
    while True:
      next_task = select_highest_priority_EDF(ready_queue)
      for each runlist in GPU.runlists:       # 遍历所有runlist
        reset_runlist(runlist)                # 必须抢占所有runlist
        if next_task.uses_engine(runlist.engines):
          insert_channels_for_engines(next_task, runlist)
      // 修正: Jetson Xavier有独立compute runlist + copy runlist
      // 仅重置compute runlist → copy runlist上仍有前task的copy 
      // → copy干扰新task的compute → deadline miss
```

Counter-example（Bakita & Anderson §VI-B）：
- Jetson Xavier: compute runlist + copy runlist（独立，R5）
- Task1 (compute+copy, period 3s, deadline 2s) + Task2 (compute-only, period 3s, deadline 3s)
- Task2 以更高 EDF 优先级抢占 Task1 → 只重置 compute runlist → Task1 的 copy 在 copy runlist 上继续 → copy 严重延迟 Task2 的 compute（Olmedo et al. [7] 在 Jetson Xavier 上证明）→ Task2 deadline miss

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

正确的实现必须：(i) 通过 nvdebug 的 device_info 接口获取 GPU 的完整 engine-runlist 拓扑；(ii) 为每个 task 记录其使用的 engine 类型；(iii) 抢占时重置所有涉及 engine 的 runlist；(iv) 考虑 runlist 的多 engine 绑定（R6, Runlist 0 同时含 compute 和 GRCE）可能导致的 side effect。在高 end GPU（如 RTX 6000 Ada, 17 runlists）上，runlist 管理比嵌入式平台更复杂但也提供更细粒度的隔离能力。

涉及论文标题：
- Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management
