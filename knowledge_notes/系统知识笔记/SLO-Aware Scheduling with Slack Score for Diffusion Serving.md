## SLO-Aware Scheduling with Slack Score for Diffusion Serving

术语是什么？通过联网搜索让回答具体和精准。

SLO-Aware Scheduling（服务水平目标感知调度）是一种面向延迟敏感型推理serving系统的请求调度策略，核心目标是在满足每个请求的SLO (Service Level Objective) deadline约束下最大化系统throughput。Slack Score是一种量化请求紧急度的启发式metric，定义为 `Slack_i = (DDL_i - C_i - P_i) / SA_i`，其中DDL_i是请求i的SLO deadline、C_i是已消耗时间、P_i是剩余stage的预测时间、SA_i是请求i的standalone model latency。Slack score越低表示请求越紧急，应优先调度执行。MixFusion论文引入此机制是因为扩散模型中混合分辨率场景下，不同分辨率的请求具有差异极大的执行延迟（SD3中高分辨率请求执行时间是低分辨率的2.4×），纯FCFS调度无法在tight SLO下高效利用GPU。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

MixFusion中SLO-Aware Scheduler的workflow（Algorithm 1）：

```
1. while True:
2.     cur_task = get_least_slack_task(wait_queue)     // 从等待队列取slack最小的请求
3.     act_task = get_least_slack_task(act_queue)       // 从active队列取slack最小的请求
4.     pred_latency = predictor(cur_task, active_queue) // MLP Throughput Analyzer预测加入后的batch延迟
5.     // SLO Violation Analyze: 检查cur_task是否已无法满足deadline
6.     if time_out(cur_task, pred_latency):
7.         discards(cur_task)    // 直接丢弃（prior work convention）
8.         continue
9.     // Schedule Mode Decision: 若最紧急请求slack仍较宽松
10.    if switch_mode(cur_task, pred_latency):
11.        cur_task = update_task()  // 切换到throughput-optimized模式，选最大吞吐提升的请求
12.        pred_latency = update_latency(cur_task, act_queue)
13.    // Schedulability test: 检查新请求加入是否导致active中请求超时
14.    if time_out(act_task, pred_latency):
15.        break   // 不再加入新请求
16.    else:
17.        act_queue.enqueue(cur_task)
```

关键设计点：(1) dual-mode scheduling——当最紧急请求仍有充足slack时切换到throughput-optimized模式（选择能最大提升当前batch吞吐的请求），防止因过度保守而浪费GPU；(2) online latency prediction——通过MLP预测合并batch的实际延迟（非简单求和），避免高估延迟导致拒绝本可服务的请求；(3) 调度决策与denoising computation并行执行，不阻塞推理。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Slack score计算依赖准确的latency prediction。MixFusion使用MLP Throughput Analyzer（3层MLP, 输入为各resolution的task数量/ongoing resolution数/total patch数，训练集200个resolution组合，error <3.7%）。调度器在每轮denoising step之间执行，选择下一个加入active batch的请求，直到schedulability test失败或无可调度请求。SLO constraint按Clockwork惯例设为5× standalone latency per resolution。被判定无法满足deadline的请求直接丢弃。该调度器被验证在SD3上SLO satisfaction相对Mixed-Cache（FCFS变体）提升11.4%，goodput提升1.1×。

涉及论文标题：
- MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models
