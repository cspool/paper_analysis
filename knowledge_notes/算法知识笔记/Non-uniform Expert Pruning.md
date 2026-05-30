## Non-uniform Expert Pruning

术语解释
非均匀专家剪枝是指在不同 MoE 层使用不同剪枝比例的专家剪枝策略。与传统的 uniform pruning（每层删除相同数量专家）不同，非均匀剪枝根据各层专家冗余程度自适应调整。例如在 Mixtral 8×7B 中，浅层（1-15）专家重要性高（处理多样化的低层语言特征如词性标注、局部词序），应保留更多专家；深层（16-32）处理全局语义信息，冗余度更高，可剪去更多专家。

术语是什么？
MoE 模型中不同层的专家冗余程度存在显著差异：CKA 可视化分析显示，浅层 expert-pair similarity 矩阵呈现更复杂的块状结构（专家间分工明确），深层相似度更高（功能趋于同质化）。Uniform pruning 忽略这种差异，对所有层应用相同的 expert sparsity ratio，导致浅层剪枝过度（丢失关键语言特征处理能力）或深层剪枝不足（浪费参数）。

非均匀剪枝的核心挑战是搜索空间巨大：L 层每层 N 个专家，搜索每层保留不同数量专家的组合数呈指数增长。DiEP 通过 differentiable optimization 解决了这个问题：学习 per-layer 重要性 β^(l) 和 per-expert 重要性 α_i^(l)，全局排序后自然产生非均匀剪枝分布。

从算法pipeline角度拆解术语：
```
# Uniform vs Non-uniform Pruning Comparison

# Uniform Pruning (e.g., NAEE, M-SMoE)
for each layer l:
  experts[l] = top-k_experts_in_layer(l, k = N×(1-r))
# 每层保留相同数量专家，忽略跨层差异

# Non-uniform Pruning (DiEP)
scores = []  # global list
for layer l:
  for expert i:
    scores.append((l, i, α_i^(l) × β^(l)))
sorted_scores = sort(scores, by=score, descending=True)
keep = sorted_scores[:N_total×(1-r)]  # 全局选择
# 结果：layer 1 可能保留 7/8 experts, layer 30 可能仅保留 3/8
```
实验结果：Mixtral 8×7B 50% sparsity 下，DiEP 的 MMLU avg 57.9 vs uniform baseline 47.3-54.6，提升 3.3-10.6 个百分点。验证了非均匀剪枝在保持模型性能方面的关键作用。

术语一般如何实现？如何使用？
- DiEP 方法：通过 differentiable search 自动学习非均匀分布
- 也可采用 heuristic：人工设定浅层高 β、深层低 β（如 β=2 for layers 1-16, β=1 for 17-32），但论文表明这种方式无法泛化到不同 MoE 架构
- 剪枝后模型直接运行，无需特殊 runtime 支持
- 可与其他压缩方法（merging, quantization）正交组合

涉及论文标题：
- DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning

---
