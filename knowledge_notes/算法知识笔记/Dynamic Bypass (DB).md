## Dynamic Bypass (DB)

术语是什么？
Dynamic Bypass (DB) 是 VAR-Turbo 提出的面向深层 Transformer 的 token 级计算跳过算法。核心设计：在深层（Inert Region），先用轻量 MLP 为每个 token 计算 importance score，再通过 TopK 选出最重要的 K 个 token 进入完整 Transformer（attention + FFN），其余 token 绕过（bypass）Transformer 层并通过 token restoration 在下一层补回原有信息。DB 基于深层 attention map 趋于相似（低通滤波效果）的 insight：深层 token 的重要性高度分化，大部分 token 对输出的信息增量低，可安全跳过。

从算法pipeline角度拆解术语：
DB 的伪代码（以单层深层 Transformer 为例）：
```
输入：token sequence X (shape: [N, D]), schedule parameter α, β
1. Importance scoring：Score_i = LightMLP(X_i)  for i=1..N  // 轻量单层 MLP
2. Compute skip threshold: s(l) = min(α×l + β, max_threshold)  // l=layer index
3. K = N × (1 - s(l))  // 保留的 token 数
4. TopK selection：选出 Score 最高的 K 个 token indices
5. Split：
   - X_keep = X[indices]  // K 个重要 token → 进入 attention + FFN
   - X_bypass = X[others]  // N-K 个不重要 token → bypass
6. 仅 X_keep 经过完整 Transformer 层（attention + FFN）→ Y_keep
7. Token restoration for bypass tokens:
   Y_bypass[i] = X_bypass[i] × JudgeWeight[i] + X_bypass[i]  // 原有信息保留
8. 合并 Y_keep 和 Y_bypass 回到原排列 → 输出 Y
```
关键超参数：α=0.3、β=-0.4、max skip threshold=0.55。DB 额外减少约 58% MAC（覆盖 attention 和 FFN），且 token restoration 避免信息永久丢失。

术语一般如何实现？如何使用？
DB 应用于深层（Inert Region），与 TA 互补。需配合 schedule function 控制逐层 skip rate，skip rate 随层数加深逐步提高。硬件上依赖 Radix Sort Core 执行大 K TopK selection。

涉及论文标题：
- VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

