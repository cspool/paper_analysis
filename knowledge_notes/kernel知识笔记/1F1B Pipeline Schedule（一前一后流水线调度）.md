## 1F1B Pipeline Schedule（一前一后流水线调度）

术语是什么？
1F1B (One-Forward-One-Backward) 是 Pipeline Parallelism 中最常用的 micro-batch 调度策略。由 PipeDream (Harlap et al., 2018) 提出。调度分三阶段：(1) Warmup——各 stage 依次执行 forward 填充 pipeline；(2) Steady——各 stage 交替执行 1 个 forward 和 1 个 backward；(3) Cooldown——各 stage 依次完成剩余的 backward。1F1B 相比 GPipe（全 forward 后全 backward）将 activation memory 峰值从 O(M×P) 降至 O(M)，其中 M 为 micro-batch 数，P 为 pipeline stage 数。

从kernel调度角度拆解术语：
4-stage pipeline (P=4), M=8 micro-batches 的 1F1B 调度：
```
Device 0: F0 F1 F2 F3 F4 B0 F5 B1 F6 B2 F7 B3    B4 B5 B6 B7
Device 1:    F0 F1 F2 F3 F4 B0 F5 B1 F6 B2 F7 B3    B4 B5 B6 B7
Device 2:       F0 F1 F2 F3 F4 B0 F5 B1 F6 B2 F7 B3    B4 B5 B6 B7
Device 3:          F0 F1 F2 F3 F4 B0 F5 B1 F6 B2 F7 B3    B4 B5 B6 B7
```
Bubble（空闲时间）占总时间的比例 = (P-1)/(P-1+M)。

在 PPMoE 中，1F1B 与 MoE 层无缝配合——MoE 层的 forward/backward 通信模式与非 MoE FFN 相同（均为 all-reduce），因此 1F1B 调度无需修改即可用于 PPMoE。

术语一般如何实现？如何使用？
Megatron-LM 默认使用 1F1B 调度（可通过 `--num-layers-per-virtual-pipeline-stage` 进一步减少 bubble）。DeepSpeed 的 PipeDream 也支持 1F1B。PPMoE 实验中 PP=4（小规模）和 PP=16（大规模）。bubble overhead 在小模型/micro-batch 少时显著——PPMoE 小规模（6.7B）backbone 仅达 81.4% throughput（vs 90.7% for 143B），部分因 PP bubble。

涉及论文标题：
- Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism
