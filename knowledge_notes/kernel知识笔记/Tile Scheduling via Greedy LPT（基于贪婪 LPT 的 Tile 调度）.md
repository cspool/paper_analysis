## Tile Scheduling via Greedy LPT（基于贪婪 LPT 的 Tile 调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Tile Scheduling via Greedy LPT 是 MxMoE 用于将混合精度 Group-GEMM 的 tile 分配到 GPU SM 的调度策略。这是经典的 makespan minimization 问题（P||C_max）：给定 M 个 SM 和 N 个 tile（每个 tile 的预估执行时间 c_t 不同），目标是最小化最慢 SM 的完成时间。MxMoE 使用 LPT（Longest Processing Time first）greedy 启发式：按 tile 执行时间 c_t 降序排列，依次将每个 tile 分配给当前累积负载最小的 SM。Graham (1966) 证明 LPT 在 tile 数远大于 SM 数时实现近最优性能（makespan ≤ (4/3 - 1/(3P)) × OPT），且调度开销远低于动态规划精确解。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
输入: tiles = [(c_1, SM_id=None), ..., (c_N, SM_id=None)]
      P = number of SMs

LPT 调度:
  sort tiles by c_t descending
  SM_load = [0] * P  // 每个 SM 的累积负载

  for tile (c_t, _) in tiles:
      # 找到当前负载最小的 SM
      min_sm = argmin(SM_load)
      assign tile to SM min_sm
      SM_load[min_sm] += c_t

  makespan = max(SM_load)
  # 接近最优: ≤ (4/3 - 1/(3P)) × OPT
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 MxMoE 运行时，tile scheduler 在 kernel launch 前执行：收集所有 expert 的 tile 列表 → greedy LPT 分配 → 编译进 kernel grid 配置。tile 数量通常远大于 SM 数（Qwen1.5-MoE 的 60+ expert 可产生数千 tile vs RTX 4090 的 128 SM），因此 LPT 近最优。调度开销 O(N log N)（排序），远低于 DP 的 O(N P^N)。

涉及论文标题：
- MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design

---
