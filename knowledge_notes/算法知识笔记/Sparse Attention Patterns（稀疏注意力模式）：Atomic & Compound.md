## Sparse Attention Patterns（稀疏注意力模式）：Atomic & Compound

术语是什么？通过联网搜索让回答具体和精准。

Sparse Attention Patterns是Transformer模型中通过mask layer引入的注意力稀疏化模式，用于减少MHA (Multi-Head Attention) 中Q×K^T score matrix的计算量。STOF论文系统分类了4种Atomic（原子）pattern和2种Compound（复合）pattern。Atomic patterns是基础构建块：(a) Causal Attention（因果注意力）——token只能attend到之前的token，mask矩阵呈下三角，sparsity=50%；(b) Global Attention（全局注意力）——某些"global"节点接收所有token信息（对应行全有效）并发送给所有token（对应列全有效），sparsity取决于global节点占比；(c) Sliding Window Attention（滑动窗口注意力）——每个query仅关注窗口大小w内的邻近token，mask矩阵呈带状（banded pattern），w=32时sparsity=93.8%；(d) Random Attention（随机注意力）——query随机关联前后token，通过filling rate控制密度。Compound patterns由atomic patterns组合而成：(e) Longformer——sliding window + global attention组合，w=32时sparsity=88.8%；(f) Bigbird——sliding window + global + random三组合，w=32, filling rate=10%时sparsity=80.8%。不同pattern在element distribution上存在关键差异：causal/sliding window的元素分布在row和column上均连续（structured sparsity），Longformer元素discrete但column仍结构化，Bigbird由于random pattern引入unstructured sparsity。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Sparse attention pattern在Transformer推理MHA计算中的位置和计算流程：

```
// ===== MHA with Sparse Mask in Transformer Inference =====
// Input: Q, K, V ∈ R^{seq_len × head_dim}
// Mask M ∈ {0, -∞}^{seq_len × seq_len}  (sparse pattern defined)
// 
// Step 1: QK^T GEMM
S = Q × K^T          // S ∈ R^{seq_len × seq_len}, O(seq_len²) compute
//
// Step 2: Scale
S = S / sqrt(d_k)    // d_k = head_dim
//
// Step 3: Apply Sparse Mask  ← 关键步骤，不同pattern差异在此
if mask[i][j] == 1:   // valid attention
    S[i][j] = S[i][j]
else:                  // masked out
    S[i][j] = -inf     // after softmax: attention weight ≈ 0
//
// Step 4: Softmax (row-wise)
P = softmax(S)        // P[i][j] ≈ 0 for masked positions
//
// Step 5: PV GEMM
O = P × V             // O ∈ R^{seq_len × head_dim}
```

不同pattern的mask示例（seq_len=8）：
```
// Causal mask (sparsity 50%):
//   0 1 2 3 4 5 6 7
// 0 1 0 0 0 0 0 0 0
// 1 1 1 0 0 0 0 0 0
// ...

// Sliding window (band width w=2, sparsity ~75%):
//   0 1 2 3 4 5 6 7
// 0 1 1 0 0 0 0 0 0
// 1 1 1 1 0 0 0 0 0
// ...

// Bigbird (sliding w=2 + global col 0/row 0 + random):
//   0 1 2 3 4 5 6 7
// 0 1 1 1 1 1 1 1 1   ← global row
// 1 1 1 1 0 1 0 1 0   ← sliding + random
// ...
```

关键性质：mask引入的sparsity使大量S[i,j]计算可以跳过——这是STOF论文优化的核心opportunity。但不同pattern的sparsity structure（连续vs离散、structured vs unstructured）要求不同的数据结构和kernel实现策略。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实践中，sparse attention pattern通过以下方式实现：(1) Mask矩阵直接定义——PyTorch等框架中mask作为binary tensor传入attention函数，最通用但最低效（仍需完整QK^T GEMM）；(2) Fused kernel直接利用sparsity——FA2支持causal mask作为内置参数，FlashMask扩展支持column-continuous mask，STOF用two-level BSR+bitmap格式表示任意mask；(3) Compound pattern分解——Bigbird等可通过多个atomic kernel组合实现（sliding window kernel + global kernel + random kernel，输出scatter-add合并）。

涉及论文标题：
- Accelerating Sparse Transformer Inference on GPU

