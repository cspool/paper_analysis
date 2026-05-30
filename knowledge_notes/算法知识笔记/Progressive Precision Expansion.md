## Progressive Precision Expansion

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Progressive Precision Expansion（渐进式精度扩展）是 AnyBCQ 中实现多精度模型的核心机制。从基础精度 p_L（如 2-bit）开始逐步扩展到 p_H（如 4-bit），每次增加 1 bit。核心原则：(1) 冻结已分配比特平面 B_1...B_{p-1}（低精度模型是高精度模型的严格子集）；(2) 从残差提取新比特平面 B_p = sign(W - Ŵ^{(p-1)})（捕获之前未表达的信息）；(3) 仅优化缩放因子 α（最小二乘闭式解，不修改 B）。与 Any-Precision LLM 的 Incremental Upscaling（分裂 centroid，refinement 型）不同，这是"添加新信息"型（additive），保证单调精度改善：p-bit ≥ (p-1)-bit。AnyBCQ 2-bit MMLU=35.32（vs Any-Precision LLM=24.66），验证了该方法在极低比特的有效性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AnyBCQ Progressive Precision Expansion（p_L=2, p_H=4）：

```
# 阶段 1: 基础精度 p_L=2
B_1, B_2, α_1, α_2 = GREEDY(W)
for t in 1..T:  # T=20
    α = least_squares([B_1,B_2], W)
    B_1, B_2 = binary_search(α, W)

# 阶段 2: p=3 扩展
α_3 = 0; B_3 = zeros_like(W)
for t in 1..T:
    R = W - (α_1*B_1 + α_2*B_2 + α_3*B_3)
    B_3 = sign(R)  # 从残差提取新比特平面
    {α_i} = least_squares([B_1,B_2,B_3], W)  # B_1,B_2 冻结

# 阶段 3: p=4 扩展 (类似)
α_4 = 0; B_4 = zeros_like(W)
for t in 1..T:
    R = W - (α_1*B_1 + α_2*B_2 + α_3*B_3 + α_4*B_4)
    B_4 = sign(R)
    {α_i} = least_squares([B_1,B_2,B_3,B_4], W)
```

B_{p-1} 不能修改的原因：如果允许修改，低精度模型的权重表达会改变，破坏"多精度统一模型"的前提。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) p_L 越低省内存越多但优化空间越紧，AnyBCQ 选 2；(2) B_p=0 初始化使第一次 sign(R) 直接捕获最大残差方向；(3) α 优化用最小二乘闭式解，搭配 block-wise MRE 微调；(4) 每增 1 bit 表达能力翻倍但收益递减（3→4 < 2→3）。限制：高精度下 Fixed-Precision 略优（因 B_i 在低精度优化时未考虑高精度需求），AnyBCQ Multi-prec. 4-bit MMLU=63.15 vs Fixed-prec.=63.90。

涉及论文标题：
- AnyBCQ Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs

---
