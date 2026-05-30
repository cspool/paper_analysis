## Dependency-aware Request Scheduling (依赖感知请求调度)

术语解释
Dependency-aware Request Scheduling 是 CoServe 提出的 CoE 推理请求调度策略，利用 CoE 系统中 expert 依赖关系（多请求依赖同 expert + expert 间依赖链）来减少不必要的 expert switching，从而在有限内存设备上提升 CoE 推理吞吐量。

术语是什么？
该调度包含三个子机制：
1. **Prediction（延迟预测）**：预测新请求加入各 executor 队列后的额外推理延迟 = 执行延迟 + 专家切换延迟。执行延迟建模为 K × batch_requests + B（K 和 B 通过 offline profiling 获得），切换延迟在 expert 已在 model pool 或队列中已有同 expert 请求时为零。
2. **Assigning（请求分配）**：选择使当前各队列最大总推理时间最小化的 executor；平局时选择额外延迟最小的队列。
3. **Arranging（请求排列）**：将新请求排在队列中同 expert 请求之后，实现同 expert 请求成组，确保一次加载服务多个请求。
4. **Splitting（请求拆分）**：根据当前可用内存和最大 batch size 将同 expert 请求组拆分为多个 batch。

从系统架构角度拆解术语：
CoServe 中 Dependency-aware Request Scheduling 的运转流程：
```
新请求 R (需要 Expert_E):
1. For each executor queue Q_i:
     if Expert_E in Q_i's model pool 或 Q_i 中已有 Expert_E 请求:
       switching_latency = 0
     else:
       switching_latency = Expert_E loading time
     exec_latency = K × (Q_i.current_batch_size + 1) + B
     additional_latency_i = switching_latency + exec_latency

2. Assigning: 选择使 max(Q.total_time) 最小化的 executor
   平局时选 min(additional_latency) 的 executor

3. Arranging: 将 R 排在 Q 中最后一个 Expert_E 请求之后

4. Splitting: 若连续 Expert_E 请求数 > max_batch:
    拆分为 ceil(N/max_batch) 个 batch
```

术语一般如何实现？如何使用？
- 实现于 CoServe 的 Request Scheduler 模块，CPU 执行（与 GPU 推理并行）
- 依赖 Offline Profiler 提供的 K、B 参数和 expert 加载延迟
- 适用于 CoE 系统（路由规则预定义，expert 依赖可提前获知），不适用于 MoE（路由动态不确定）
- 效果：请求调度延迟 < 推理延迟，调度开销 < 3%

涉及论文标题：
- CoServe: Efficient Collaboration-of-Experts (CoE) Model Inference with Limited Memory
