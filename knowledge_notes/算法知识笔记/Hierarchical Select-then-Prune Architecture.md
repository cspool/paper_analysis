## Hierarchical Select-then-Prune Architecture

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical Select-then-Prune Architecture是Twilight提出的统一sparse attention优化框架。采用两阶段设计：(1) Token Selector——将现有top-k sparse attention算法作为黑盒，使用保守的大budget（如1/4 sparsity）预选token子集，保证高recall；(2) Twilight Pruner——在预选子集上用INT4 K cache估计精确attention weights，然后通过top-p thresholding进一步剪枝到最小token子集。最终sparse attention kernel仅对top-p token执行精确计算。

从算法pipeline角度拆解术语，给出具体例子。
以Quest作为base algorithm为例：
```
// Stage 1: Token Selector (Quest with conservative budget B0=N/4)
page_scores = q @ max_pool(K, page_size=16)^T
top_pages = TopK(page_scores, k=B0/16)
I0 = expand_pages_to_tokens(top_pages)    // |I0| = B0

// Stage 2: Twilight Pruner
W_approx = q @ K_int4[I0]^T               // INT4 SpGEMV
W_norm = softmax(W_approx)                // normalize
I1 = top_p_binary_search(W_norm, p=0.85)  // |I1| = B1 << B0

// Stage 3: Sparse Attention
O = FlashAttention(q, K[I1], V[I1])       // only B1 tokens
```
关键优势：任何top-k sparse attention算法都可被"升级"——只需将其结果作为Token Selector的输出。

术语一般如何实现？如何使用？
基于FlashInfer实现。Token Selector复用原算法kernel，Pruner使用自研INT4 SpGEMV + top-p binary search kernel，Sparse Attention使用FlashInfer的varlen attention kernel。额外开销：INT4 K cache（1/8 FP16 KV cache）。适用于LLM serving。

涉及论文标题：
- Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning
