## Least-Loaded Assignment (LLA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Least-Loaded Assignment (LLA) 是 LLEP 论文提出的贪心负载均衡算法，用于在 Expert Parallelism (EP) 框架下，根据全局 per-expert token 负载，预先计算一个"超载 GPU → 欠载 GPU"的 token 和 expert 权重分配计划。LLA 按 expert 负载降序处理（先分配最大负载的 expert），对每个 expert 判断其原生 GPU 是否有足够容量容纳所有 token。若容量不足，则通过 LLAS（Least-Loaded Assignment Spill）子程序将多余 token 溢出到当前负载最轻的 GPU。同时输出权重传输计划 W（将 expert 权重从原生 GPU P2P 传输到计算 GPU）。

LLA 是"精确 MoE 计算"（exact computation）算法 —— 不改变模型的 gating 输出或 FFN 数学计算，仅在系统层面改变 token-to-GPU 的物理分配。关键参数：(1) 容量因子 α 控制每 GPU token 容量 m_α = α × Σl_i / P；(2) 最小 GEMM token 数 m（低于 m 的溢出量强制本地计算以避免低效微小 GEMM）；(3) 自适应阈值 λ（当 max(l)/mean(l) < λ 时回退到标准 EP）。

从系统架构角度拆解术语：

LLA 在 EP dispatch 前于 CPU 侧执行（纯 Python），核心流程（Alg. 2-3）：

```
Input: expert loads l[], M experts/GPU, α, m
m_α = α × Σl/P;  g_a = [0]*P;  g_p = per-GPU native loads
for each expert e (descending load):
    ng = native GPU;  na = m_α - g_a[ng] - g_p[ng]
    if na >= e:  all to ng                      // Case 1
    elif na > 0:  na to ng, LLAS spill r=e-na   // Case 2
    else:  LLAS spill r=e                        // Case 3

LLAS(r): while r > 0:
    sort other GPUs by g_a+g_p ascending → pick least-loaded
    c = min(r, remaining capacity); if c ≥ m: assign; r -= c
Output: A (per-expert token-to-GPU assignment), W (weight transfer plan)
```

设计要点：优先原生 GPU 计算最小化权重传输（W 传输 D×H 远大于 token 传输）；容量硬限制预防 OOM；m 约束摊销权重传输开销。

术语一般如何实现？如何使用？

GPU 数为 P 时，为纯 Python CPU 模块，N≤256 下计算开销微秒级。产生的 A 和 W 用于构建 All-to-All buffer 和 NCCL P2P 权重传输。不同于 EPLB（expert 复制方案增加显存）——LLA 运行时动态迁移 token+权重不增显存。不同于 Load Balance Loss（Switch Transformer，修改模型行为）——LLA 是纯系统级方案。

涉及论文标题：
- Least-Loaded Expert Parallelism: Load Balancing An Imbalanced Mixture-of-Experts
