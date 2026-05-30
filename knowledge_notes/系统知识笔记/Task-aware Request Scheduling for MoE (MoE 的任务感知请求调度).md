## Task-aware Request Scheduling for MoE (MoE 的任务感知请求调度)

术语解释
Task-aware Request Scheduling 是一种针对 MoE 推理的请求调度算法，它在调度决策中联合考虑三个因素：(1) 用户施加的 latency SLO，(2) profiled 任务特定 token 生成长度，(3) expert loading latency（∆E）。与现有调度器（vLLM、Orca、Sarathi-Serve）仅考虑 token 计数或 SLO 不同，该调度器显式建模 expert loading 引入的额外延迟，并利用任务间 output length 差异做出更优调度决策。

术语是什么？
eMoE 的 Task-aware Request Scheduler 采用贪心迭代算法（Algorithm 1）：

1. **SLO Stringiness 排序**：按 first-token 生成 deadline 的紧迫程度排序等待请求
2. **Token Budget 检查**：确保调度后总 token 数不超过推理引擎处理上限 Tmax
3. **延迟预估**：对每个已调度请求计算 expected latency `t_i = ΔE + (W + n_i · G_i) · c + r_i`：
   - ΔE：profiled expert loading latency（取决于 load set 大小）
   - W：新请求的 input tokens
   - n_i：在新请求的 tokens 生成完之前会完成的 running requests 数
   - G_i：请求 i 所属 task 的 profiled 平均 output length（运行时递减）
   - c：per-token 平均 expert computation + communication latency
   - r_i：请求 i 已运行时间
4. **SLO 约束**：仅当 `t_i < SLO_i, ∀i ∈ scheduled` 时才调度新请求
5. **G_i 递减与 Reset**：每轮 token generation 后 G_i 递减 1；若 G_i=0 但请求未完成，reset 为 initial value 的 5%

从系统架构角度拆解术语：

```
=== Task-aware Request Scheduling Algorithm ===
见论文 Algorithm 1：

Input: Waiting queue Qw, Scheduled queue Qs, Max tokens Tmax
1  T ← Qs.length  # total input tokens of scheduled requests
2  Qw ← Sort Qw by SLO stringiness  # 升序：最紧迫的优先
3  for R ∈ Qw do
4      if R.inputTokens + T < Tmax then
5          # 预估：调度 R 后所有已调度请求的新延迟
6          expected_ok = True
7          for each S ∈ Qs:
8              t_S = ΔE + (R.inputTokens + n_S · G_S) · c + r_S
9              if t_S ≥ S.SLO:
10                 expected_ok = False; break
11         if expected_ok:
12             Qs ← Qs ∪ {R}
13             T ← T + R.inputTokens
14 return Qs

Notation:
  ΔE: profiled expert loading latency (changes when new experts loaded)
  n_S: # of requests that will complete before S
  G_S: profiled output length (task-specific, decrements per generation step)
  c: per-token latency constant
  r_S: request S's current runtime

Scheduler 触发条件：
  - 新请求到达
  - 运行中请求完成 token generation
```

术语一般如何实现？如何使用？
- Scheduler 运行在 CPU 上，不产生额外 GPU 开销
- Task-specific G_i 通过 offline profiling 获得：对每个 task type 收集 output token 分布，取平均值或分位数
- Expert loading latency ∆E 也通过 profiling 获得
- 贪心策略的局限：局部最优可能非全局最优（但 overhead 低，适合在线调度）
- 与 Task-aware Expert Loading 协同：scheduler 优先调度 routing-sensitive 任务 → expert loader 集中为 sensitive tasks 精确加载 expert

涉及论文标题：
- eMoE: Task-aware Memory Efficient Mixture-of-Experts-Based (MoE) Model Inference
