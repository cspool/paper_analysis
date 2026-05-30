## Defragging Scheduler（碎片整理调度器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Defragging Scheduler 是 AMoE 系统中用于在多个 colocated layers 之间选优执行顺序的调度算法。在 AEP 中，每个 GPU 托管多个 block 的多个 expert 层（colocated），GPU 空闲时需要决定下一个执行哪个 layer。两个 strawman 策略各有缺陷：Most-Token-First-Serve (MTFS) 优先执行 token 数最多的 layer——导致各 layer 留下孤立的"最后一小片" token（batch fragmentation），形成 disorganized batch；First-Layer-First-Serve (FLFS) 优先执行 block# 最低的 layer——aggressively defragments 但新请求的 token 会打断较高 block 的执行（live-lock）。Defragging Scheduler 在两者间折中：通过 lookahead 加权下游 token 密度来鼓励 defragmentation，同时纳入当前队列 token 数来避免过度忽视频繁到达的 token。

从系统架构角度拆解术语，给出术语在系统架构中运转流程的具体例子。
Algorithm 1（简化版）：
```
Input: N_B: NumBlocks, N_E: NumExperts (on this GPU), Q[l,g]: tokens in µ-queue, δ: decay factor
Output: (b*, e*): optimal (block, expert) to schedule

Scores[N_B][N_E] ← 0
for b ← 0 to N_B - 1:
    LScore ← 0
    for k ← 1 to K:                    // lookahead K blocks ahead
        b' ← (b + k) mod N_B
        TotalTokens ← sum_{e'} Q[b'][e']  // all experts' tokens in future block
        LScore ← LScore + (TotalTokens / N_E) × δ^k  // decayed by distance
    for e ← 0 to N_E - 1:
        if Q[b][e] > 0:
            Scores[b][e] ← LScore + Q[b][e]  // lookahead + current queue

return argmax_{b,e} Scores[b][e]
```

效果：Scheduler 偏好执行其下游 block 已有密集 token 的 layer（通过 LScore），鼓励 token wave 在连续 block 中前进，自然合并碎片化 batch。同时保留 MTFS 的 queue occupancy 感知（Q[b][e] 项），避免 FLFS 的 live-lock。

术语一般如何实现？如何使用？
在 AMoE 中 Defragging Scheduler 在 main Python thread 中运行（与 Executor 同线程），使用 C++ 实现以最小化 scheduling overhead（Figure 13 显示 scheduling stage 仅占执行时间小部分）。参数：K（lookahead 窗口，e.g., K=3）、δ（衰减因子，e.g., 0.5）。Decay 因子控制 defragmentation 激进程度——δ 越大越接近 FLFS，δ 越小越接近 MTFS。

涉及论文标题：
- Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony
