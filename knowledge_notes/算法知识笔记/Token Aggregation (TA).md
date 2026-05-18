## Token Aggregation (TA)

术语是什么？
Token Aggregation (TA) 是 VAR-Turbo 提出的面向浅层 Transformer 的 attention 压缩算法。核心设计：将每层的 token sequence 划分为 non-overlapped local window，每个 window 内的 token 先经 Small Attention 聚合成一个 representative token（通过 attention score 加权求和），再将所有 window 的 representative token 拼接后送入 Big Attention 做全局建模。TA 基于 attention 作为低通滤波的 insight：浅层（Learning Region）attention map 中局部 token 高度相关但未完全退化，通过两级 attention 在不固定稀疏 pattern 的情况下压缩计算量。

从算法pipeline角度拆解术语：
TA 的伪代码（以单层浅层 Transformer 为例）：
```
输入：token sequence X (shape: [N, D]), local window size W
1. 划分：将 X 分为 N/W 个 non-overlapped local window
2. for each window w:
     Small Attention: Q_w, K_w, V_w = Linear(X_w)
     Score_w = softmax(Q_w × K_w^T / sqrt(d))
     Coeff = mean(Score_w, axis=0)  // 平均所有 query 的 attention
     Rep_w = Coeff × V_w  // representative token
3. Rep = concat(Rep_1, ..., Rep_{N/W})  // shape: [N/W, D]
4. Big Attention: Q_rep, K_rep, V_rep = Linear(Rep)
   Output_rep = softmax(Q_rep × K_rep^T / sqrt(d)) × V_rep
5. 将 Output_rep 映射回原 token 维度用于后续层
```
关键参数：local window size —— 低分辨率用 2，高分辨率用 2/4 混合；size ≥ 8 时质量明显下降。TA 论文声称减少约 41% attention MAC，质量下降 <0.5%。

术语一般如何实现？如何使用？
TA 应用于浅层（0-15 层 Learning Region）。硬件上需 Unified Attention Core 同时支持 Small Attention 的 OP dataflow 和 Big Attention 的 Row dataflow。TA 与 DB 互补：TA 处理浅层仍活跃的 Learning Region，DB 处理深层趋于惯性的 Inert Region。

涉及论文标题：
- VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

