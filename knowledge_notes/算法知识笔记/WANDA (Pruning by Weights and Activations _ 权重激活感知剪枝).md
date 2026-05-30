## WANDA (Pruning by Weights and Activations / 权重激活感知剪枝)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
WANDA (Pruning by Weights and activations) 由 Sun et al. (CMU & Meta AI, ICML 2023) 提出，是一种简单高效的 one-shot LLM 剪枝方法。核心创新：剪枝度量 S_ij = |W_ij| · ||X_j||₂（权重绝对值 × 对应输入激活 ℓ₂ 范数），比纯幅度剪枝更准确。每个输出神经元内独立剪枝（行内），保证各输出特征稀疏度平衡。无需反向传播、无需 Hessian、无需权重更新，仅需单次前向。复杂度 O(d²) vs SparseGPT O(d³)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# WANDA 逐行激活感知剪枝
输入: W ∈ R^{C_out×C_in}, 校准激活 X ∈ R^{C_in×L}, sparsity s
输出: M ∈ {0,1}^{C_out×C_in}

for i in range(C_out):
    scores = |W[i,:]| ⊙ ||X||₂              // row j → |W_ij|×||X_j||₂
    k = floor(C_in × s)
    threshold = top_k(scores, k)             // 保留 top-(1-s)
    M[i,:] = (scores >= threshold)
    W_pruned[i,:] = W[i,:] ⊙ M[i,:]
```

OBR 以 WANDA 生成的 mask 为输入（WANDA 负责选定剪枝位置，OBR 负责补偿剪枝损失）。多种 mask 兼容性测试：WANDA > SparseGPT > Magnitude > Random。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/locuslab/wanda。128 calibration samples 单次前向收集 ||X_j||₂ → 逐行剪枝 → 无需 finetuning。支持 unstructured 和 N:M semi-structured。作为 OBR 的默认 mask generator。

涉及论文标题：
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs
