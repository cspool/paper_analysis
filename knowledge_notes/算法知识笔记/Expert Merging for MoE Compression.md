## Expert Merging for MoE Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Merging（专家合并）是一种后训练 MoE 模型压缩技术，通过将多个功能相似或冗余的 expert 合并为更少的 expert 来减少 MoE 模型的总参数量和内存占用。核心原理：MoE 层中 N 个独立 expert（每个含 SwiGLU FFN 的三组权重矩阵 W_D, W_U, W_G）被聚类为 M 个组（M < N），每组内的 expert 通过某种合并策略融合为一个新的 expert，路由权重相应聚合。最早由 M-SMoE（Li et al. 2023, ICLR 2024）提出——基于 expert 使用频率识别 dominant experts、按路由 logits 相似度聚类、簇内按使用频率加权平均参数、可选 low-rank decomposition + structural pruning 进一步压缩。MergeMoE（Miao et al. 2025, arXiv 2510.14436）从理论上改进：将合并重新解释为"输出合并"视角下的优化问题，通过最小二乘法优化维度缩减矩阵 T1（Moore-Penrose 伪逆闭式解），并严格证明了使用频率作为合并权重的最优性。

从算法pipeline角度拆解术语：
```
// MergeMoE 完整压缩流程 (N→M experts)
// 1. 频率统计：calibration 数据上前向推理，统计 f_i = count(expert_i 被 top-K 选中)/total
// 2. 聚类：top-M frequency 为 center，其余按 ||[W_Uj||W_Gj] - [W_Uc||W_Gc]|| 分配
// 3. 簇内权重：B_{ji} = f_j / Σ_{k∈C_i} f_k (Theorem 1 证明最优)
// 4. 扩展参数+维度缩减：W'_{Di}=[B_{1i}W_{D1},...]; T2,T3=[B_{1i}I,...] (式4); T1=Q·P^† (式6)
// 5. 最终权重：W^final_Di=W'_{Di}·T1; W^final_Gi=T2·W'_{Gi}; W^final_Ui=T3·W'_{Ui}
// 6. 路由更新：merged_routing = A · original_routing (求和)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- M-SMoE 开源：https://github.com/pppp/M-SMoE；MergeMoE 论文未提供公开代码仓库。
- 实现：PyTorch + HuggingFace Transformers，torch hooks 获取中间激活，BFloat16 精度，逐层反向遍历压缩，每层 <1 分钟。
- 适用场景：将大 MoE 模型压缩到资源受限设备；减少 expert 数量降低推理内存带宽需求。
- 局限：合并后无法恢复原始结构；聚类策略影响最终性能；routing discriminative power 下降（REAP 指出可能导致 functional subspace collapse）。

涉及论文标题：
- MergeMoE: Efficient Compression of MoE Models via Expert Output Merging

---
