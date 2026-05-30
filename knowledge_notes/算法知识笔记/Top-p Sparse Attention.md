## Top-p Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Top-p Sparse Attention（top-p 稀疏注意力）是将 LLM text generation 中的 nucleus sampling (top-p sampling) 引入 attention sparsity 的技术。核心思想：用累积概率阈值 p 替代固定 token 数量 k 来决定 sparse attention 中保留多少 KV cache token。具体而言，给定估计的 attention weights W ∈ R^N，选择最小的 token 子集 I 使得 Σ_{i∈I} W[i] ≥ p（而非选择固定数量 B 个 token）。这使得稀疏 attention 的 budget（被选 token 数）可以自适应不同 attention head、不同 layer、不同 query 下 attention weight 分布的动态性——对 focused attention（权重集中）自动选少量 token，对 diffuse attention（权重平坦）自动选更多 token。理论误差界：||o - ô|| ≤ (1-p) · ||V||_F（来自 Frobenius norm 的 sub-multiplicative 性质）。

从算法pipeline角度拆解术语，给出具体例子。
```
// Top-k Sparse Attention (baseline):
I = argmax_I Σ_{i∈I} W[i]  s.t. |I| = B  // 固定budget B
// 问题: B无法适应不同分布——focused分布下B过大(浪费), diffuse分布下B过小(精度不足)

// Top-p Sparse Attention:
I = argmin_I |I|  s.t. Σ_{i∈I} W[i] ≥ p  // 固定累积概率p
// 优势: budget自适应——分布决定B, 而非预设B
```

Twilight中的实现：Token Selector用保守大budget B0≈N/4预选token → INT4 SpGEMV估计attention weights → Top-p binary search精筛到B1（累积概率≥p的最小token集）。p值选择比k更鲁棒——p代表累积概率，对不同分布head/layer/query的敏感度远低于k。p=0.85-0.95 typically。

术语一般如何实现？如何使用？
基于FlashInfer，使用4-bit quantized K cache做SpGEMV估计attention weights，GPU上tensorized binary search找满足ΣW[i]≥p的最小token子集。p值通过PG-19等小数据集calibration确定。适用场景：任何使用top-k sparse attention的LLM推理系统，可作为drop-in optimizer叠加到现有算法上。

涉及论文标题：
- Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning
