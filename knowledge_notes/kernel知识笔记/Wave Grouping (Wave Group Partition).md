## Wave Grouping (Wave Group Partition)

术语是什么？

Wave grouping（wave group partition）是 FlashOverlap 中将 GEMM 的 T 个 wave 划分为 P 个 group（P ≤ T），以 group 为单位触发 NCCL 通信的调度策略。每个 group G_j 包含 |G_j| ≥ 1 个连续 wave，group 内所有 tile 完成后统一执行通信。核心 motivation：tile-wise 通信（每 tile 立即通信）带宽利用率极低——RTX 4090 上单个 tile (192KB) 的 AllReduce 仅 13% 带宽利用率。通过将多个 wave 合并为 group 以稍延迟通信换取大幅提升的带宽利用率。

从kernel调度角度拆解术语：

设计空间与 performance trade-off：

```
给定 T=5 waves，搜索空间 = 2^{T-1} = 16 种 partition

二进制编码（每位表示第i个wave后是否通信，"1"=通信,"0"=不通信）:
"1000" → partition: G1={W1}, G2={W2,W3,W4,W5}, P=2
"1010" → partition: G1={W1}, G2={W2,W3}, G3={W4,W5}, P=3

剪枝约束（FlashOverlap）:
  |G_1| ≤ 2  (first group ≤ 2 waves, 避免 cold start)
  |G_P| ≤ 4  (last group ≤ 4 waves, 避免 long tail)

剪枝后搜索空间: O(2^{T-2})
```

**Annotations**: P=1（单 group）等价于 no overlap。P=T（每 wave 一个 group）max overlap opportunity 但 bandwidth utilization 最低。RTX 4090 + AllReduce 测试中仅 4% case 的最优 partition 为 P=T，使用 baseline partition (P=T) 导致平均 17.34% 性能退化。

术语一般如何实现？如何使用？

FlashOverlap 通过 predictive search 离线搜索最优 wave group partition。Predictor 基于 GEMM config 和 bandwidth curve 预测每种 partition 的 overlap 后延迟，选择延迟最小的 partition。对于 GEMM size 有限变化的任务（LLM training），tuning 在 runtime 前完成；对于动态任务（LLM inference），pre-search 代表性 GEMM size 后 nearest-neighbor matching。

涉及论文标题：
- Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering
