## HAP (Harmonic-Aligned Permutation / 谐波对齐排列)

术语是什么？

HAP是GyRot提出的基于Hadamard矩阵harmonic行结构的outlier channel排列策略。Hadamard矩阵递归构造（Sylvester's method）产生"harmonic rows"——长度为2^k的全+1或全-1向量（k<n）。HAP将全局选出的high-magnitude outlier channel permute到这些harmonic rows上，使post-rotation后每个group内的outlier乘以一致符号（全+1或全-1），从而tightly bound per-group range。

从算法pipeline角度拆解术语：

```
// HAP算法流程 (G=8, R=32, g=2, 即R=4G)
Input: activation X of shape [batch, Nch], G=8, R=32

Step 1: 识别全局outlier channels
  // 按per-channel magnitude排序
  O = topk_outlier_channels(X, k)  // 选top-k高magnitude channel
  // 例: O = {O1, O2, O3, O4} (4个outlier channels for R=32=4G)

Step 2: 识别harmonic rows
  // Hadamard H_32有harmonic rows at positions: G, 2G, 3G, 4G (=8, 16, 24, 32)
  // Harmonic row at position k·G (k=1..2^g): 长度为G的重复+1/-1 pattern
  // 例: row 8 = [+1×8, -1×8] within each block, row 16 = [+1×16, -1×16] etc.

Step 3: Permute outlier channels → harmonic rows
  // Permute O1→row G, O2→row 2G, O3→row 3G, O4→row 4G
  X_perm = apply_permutation(X, O, harmonic_rows)

Step 4: Apply Hadamard rotation (CoRFiG, R=32)
  X_rot = FHT_32(X_perm)

// Post-rotation effect (以group 0为例):
// Without HAP: outlier O1混入group 0 with 随机符号 (mix of +1/-1)
//   → group range = [-O1-O2-O3-O4, +O1+O2+O3+O4] → wide spread
// With HAP: 每个outlier×同符号 within its group
//   例: row G→[+1,..,+1] → outlier O1所有元素在group 0内×(+1)
//   → group range = [O1+O2+O3+O4, ...] → shifted bias, tighter bound

// Effect on Scale Factor precision (Table IV):
// G32 without HAP: INT8 SF → PPL 364.17 (catastrophic)
// G32 with HAP: INT8 SF → PPL 6.80 (near FP16 6.80)
```

术语一般如何实现？如何使用？

HAP的关键特性：(1) Permutation可fuse进weight矩阵（permutation-invariant property: 非线性和element-wise ops对排列不变），无runtime overhead——只需在offline阶段permute weight output channels，activation自然被permuted。(2) Harmonic rows的选择：Hadamard矩阵大小为R时，有log₂(R)个harmonic row位置（2^k, k=1..log₂(R)），每个位置对应一个特定stride的全+1/-1 pattern。(3) HAP与CoRFiG配合：CoRFiG确保rotation scope R=2^g·G，使所有harmonic rows的stride与group size对齐，每个group恰好包含一个harmonic row的完整segment。

涉及论文标题：
- GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

