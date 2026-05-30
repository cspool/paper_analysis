## Hierarchical Weight Control in LoRA（LoRA 层级权重控制）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical Weight Control（层级权重控制）是 MOLE 的核心设计理念：不同于 NLA 对所有 transformer 层使用统一组合权重，MOLE 在每层独立学习一组 gating 分布，实现"不同层对不同 LoRA 赋予不同权重"。其理论基础是 MOLE Observation 2：单个 LoRA 的不同层编码了不同特征——V&L 域中浅层控制"毛色/耳朵形状"而深层控制"背景风格"；NLP 域中 0%-20% 层擅长 QNLI，80%-100% 层擅长 ANLI-R1。因此理想的 LoRA 组合应在不同层给不同 LoRA 不同权重。

从算法pipeline角度拆解术语：
MOLE 的 coarse-to-fine gating 层级划分（Table 9）：
```
n-MoLE (network-wise):   1 个 gating，全局统一权重  → 最粗粒度
b-MoLE (block-wise):     每 transformer block 1 个 gating
l-MoLE (layer-wise):     每 transformer sub-layer 1 个 gating
m-MoLE (matrix-wise):    每个参数矩阵 1 个 gating  → 最细粒度

V&L text-alignment 实验结果:
n-MoLE: 0.722 → 粒度过粗，无法区分层间差异
m-MoLE: 0.731 → 粒度过细，过度控制破坏 LoRA 参数间的内在关联
l-MoLE: 0.760
b-MoLE: 0.766 → 最佳，block 级在灵活性和稳定性间取得平衡
```
论文默认使用 b-MoLE（= "MoLE"），即每 transformer block 独立 gating。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：在选定层级的每个位置插入 gating 模块（concat → normalize → flatten → dot-product → softmax）。对于 L 层、N 个 LoRA，b-MoLE 有 L 个独立 gating。
- 可视化验证（Fig. 7）：MOLE 确实学习到非均匀的层间权重分布——LoRA A 在浅层权重 45%、深层 12%；LoRA C 相反（浅层 8%、深层 52%），自动复现 Observation 2 的层特异性。
- 粒度选择是性能-灵活性的 trade-off：更细粒度 = 更强表达能力但可能过拟合。

涉及论文标题：
- Mixture of LoRA Experts
