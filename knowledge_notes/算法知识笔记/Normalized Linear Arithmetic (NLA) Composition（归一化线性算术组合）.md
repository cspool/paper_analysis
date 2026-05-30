## Normalized Linear Arithmetic (NLA) Composition（归一化线性算术组合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NLA (Normalized Linear Arithmetic) Composition 是 LoRA 组合中最简单、最广泛使用的 baseline 方法（MOLE Eq.2）。对 N 个预训练 LoRA adapter 的增量权重做加权求和：$\hat{W} = W + \sum_{i=1}^{N} w_i \Delta W_i$，其中约束 $\sum w_i = 1$。归一化约束防止了直接叠加（Eq.1: $\hat{W} = W + \sum \Delta W_i$）在 N 增大时导致的权重膨胀和生成能力退化。但代价是每个 LoRA 的有效贡献被压缩到约 1/N，导致个体 LoRA 的区分性特征被稀释。

从算法pipeline角度拆解术语：
```
# NLA 的权重合并（推理前一次性操作）:
for each linear layer with weight W ∈ R^{d×k}:
    ΔW_merged = zeros(d, k)
    for i in 1..N:
        ΔW_merged += w_i * (B_i @ A_i)   # w_i ∈ [0,1], Σw_i=1
    W_hat = W + ΔW_merged

# 推理时: y = W_hat @ x  (与标准推理完全相同，零额外开销)
```

NLA 的关键局限（MOLE Observation 1 & 2）：
1. 所有层共享同一组 {w_i}：忽略了不同层编码不同特征
2. w_i 由人工指定或 heuristic 搜索，缺乏数据驱动的逐层自适应
3. 当 N≥3 时稀释效应显著，个体 LoRA 特征被噪声淹没

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现最简单：加载所有 LoRA weights → 按 w_i 缩放 → merge_to_base_model。PEMs 和 LoRAHub 均以 NLA 为基础。
- 适用：快速原型、LoRA 数量少（≤2）的场景。常见于 Stable Diffusion WebUI 中的 LoRA weight slider。

涉及论文标题：
- Mixture of LoRA Experts
- LoRAHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition
- PEMs: Composing Parameter-Efficient Modules with Arithmetic Operations
