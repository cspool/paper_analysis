## Structured Salient Weight Selection（结构化显著权重列选择）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Structured Salient Weight Selection（结构化显著权重列选择）是 BiLLM 提出的基于 Hessian 矩阵的结构化重要权重识别策略。传统方法（如 PB-LLM）通过逐元素 Hessian 重要性排序非结构化选取 salient 权重，但需要额外 1-bit bitmap index 记录每个元素位置，增加平均 bit-width。BiLLM 观察到在 LLM 的多头注意力机制中，salient 权重的 Hessian 值高度聚集在特定列（尤其在 Q/K/V 投影层），因此采用按列结构化选择——将权重矩阵 W ∈ R^{n×m} 按列聚合显著性 S_j = Σ_i w_{ij}²/[H^{-1}]_{ii}²，选择 top-k 显著性最高的列作为 salient 组，以极小 bitmap 开销（1/b_size per weight，b_size=128 时仅 ~0.008 bit）覆盖绝大多数重要权重。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# Step 1: 计算逐元素显著性矩阵（Equation 3）
S_{ij} = w_{ij}² / [H^{-1}]_{ii}²           # Hessian 加权的元素级显著性

# Step 2: 按列聚合显著性
col_salience = sum(|S|, dim=0)               # 每列的总显著性

# Step 3: 排序 + 搜索最优列数 n*
row_s = argsort(col_salience, descending)    # 按显著性降序排序列索引
for i = 1, 2, ..., len(row_s):
    B_1 = binary(W_{:, row_s[:i]})            # salient 列二值化
    B_2 = binary(W_{:, not in row_s[:i]})     # 非 salient 列二值化
    e_i = ||W - (B_1 ∪ B_2)||²               # 整体重构误差
n* = argmin_i e_i                             # 选最小误差的列数
salient_cols = row_s[:n*]                     # 通常 3-30 列（per block）
```
搜索范围通常限定在 3-30 列以控制 bit-width。BiLLM Table 1 显示该方法在 OPT/LLaMA/LLaMA2 系列上仅引入 0.07-0.13 bit 额外开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
BiLLM 代码中 salient 函数实现 (binary.py)：`S = W**2 / H_chol_inv**2` → `row_s = torch.topk(torch.sum(torch.abs(S), dim=0), k=max_cols)` → 循环搜索 n*。结构化选择 vs 非结构化选择的权衡：(1) 结构化（按列）：bitmap 开销极小但可能漏掉列内分散的 salient 元素；(2) 非结构化（逐元素）：覆盖所有 salient 元素但每条需要 1-bit bitmap。BiLLM 论文（Appendix G, Figure 15）验证了 OPT 系列中 Q/K/V 层 salient 权重高度聚集于特定列，证实结构化策略的有效性。对于 FFN 层（FC1/FC2）salient 分布较分散的场景，搜索过程会自动选择更多列以覆盖，但平均 bit-width 仍控制在 ~1.1 bit。

涉及论文标题：
- BiLLM Pushing the Limit of Post-Training Quantization for LLMs

---
