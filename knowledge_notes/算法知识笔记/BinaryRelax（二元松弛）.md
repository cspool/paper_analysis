## BinaryRelax（二元松弛）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BinaryRelax 是 Yin et al.（2018, SIAM Journal on Imaging Sciences）提出的 QAT 方法，使用 W 形非凸正则化的 relaxed proximal map 替代硬量化。其 proximal map 在量化值附近为倾斜段，斜率随时间逐渐减小至 0。与 STE 全程硬量化不同，BinaryRelax 允许训练初期权重不完全量化，通过放松的 regularized proximal map 稳定训练。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
BinaryRelax 使用 Moreau envelope of indicator function: Ψ(w)=min_{v∈Q^d} ‖v-w‖_2²（平滑近似）。其 proximal map（图 9b）slanted segment 的斜率随时间递减到 0。与 PARQ 的关键区别：PAR 是凸的（max of linear functions），BinaryRelax 的 W 形正则化是非凸的（有中间 hill），gradient-based 优化可能因初始点被困在"错误山谷"而达不到全局最优。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PARQ 论文将其作为对比 baseline。差异方向：BinaryRelax 的 slanted slope → 0（放松量化），PARQ 的 slanted slope → ∞（收紧量化到硬量化）。实验显示 PARQ 训练更稳定（无 sudden accuracy drops），尤其在极低位宽（1-bit/ternary）场景。

涉及论文标题：
- PARQ Piecewise-Affine Regularized Quantization
