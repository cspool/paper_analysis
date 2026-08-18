## Dynamic Upscaling（动态系数放大，Ising 硬件 embedding）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dynamic Upscaling 是 SATIC++ 的硬件 embedding 技巧，增强问题系数缩放到机器系数范围的常见做法（文献[46]）：自适应决定缩放因子，有两种策略——按最大系数缩放，或按次大系数缩放（并把最大系数截断到上限）。后者用于最大与次大系数差距大的情况（避免单个离群大系数迫使整体缩放因子过小、浪费系数精度）。作用：把 QUBO 系数推向/略超硬件上限，让硬件在更展开的系数范围上工作（假设硬件在分布更广的系数上表现更好）。评估：对小 3SAT（UF20）有用、对 UF75 影响小、对大 4SAT（Batch-4-50-500）有害（系数已超限时再放大加剧截断）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
硬件 embedding 运转流程（子问题 → 芯片前）：
```
# 输入：子 QUBO 系数集合 {c_i}，硬件系数上限 R=14
max_c ← max|c_i|; second_c ← 次大|c_i|
if gap(max_c, second_c) 大:        # 策略二（论文推荐场景）
    scale ← R / second_c            # 按次大缩放
    系数 ← round(c_i * scale); 最大系数截断到 R
else:                               # 策略一
    scale ← R / max_c; 系数 ← round(c_i * scale)
# 输出：全部落在 [-R, R] 且尽量展开的系数 → 送芯片
```
与 Adaptive Spin Merging 组合：合并消除个别超限项后，Upscaling 处理整体精度展开。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：embedding 例程按系数统计自适应选策略（跟踪最大/次大系数）并缩放取整；复杂度 O(子问题系数数)。使用：配合 Formula Mix/Spin Merging 的系数后处理链（Formulation → 混合/摊平 → 缩放/合并 → 芯片），是 SATIC++ Machine Embedding 块的两个技巧之一；对小系数问题（3SAT/UF20）收益明确。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)
