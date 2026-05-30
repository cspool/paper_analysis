## Data Preprocessing Operators in Recommendation Systems

术语是什么？
Data Preprocessing Operators是推荐系统推理pipeline中的前置数据转换算子族，负责将raw features转换为model-ready inputs，在model inference的latency-critical path上同步执行。KernelEvolve论文识别的三类核心preprocessing transformation：(1) Dense normalization——BoxCox/Logit统计变换、one-hot encoding、linear scaling (shift + multiplication)；(2) Sparse processing——top-k selection with truncation、cryptographic hashing mapping IDs to embedding table indices、type downcasting (int64 → int32)；(3) Feature derivation——bucketizing continuous values to categorical bins、set operations across multiple sparse lists、n-gram hashing for text features。

从算法pipeline角度拆解术语：
论文给出了三个具体preprocessing kernel的算法流程：

**MapIdTransform**：将sparse high-cardinality categorical IDs映射为dense consecutive integers for embedding lookup。
```
Algorithm:
Input: values V, sorted mapping M
For each v in V:
  1. Binary search: idx = bucketize(v, M)  # find insertion index
  2. Clamp: idx = min(idx, |M| - 1)
  3. Validate: if M[idx] == v → output idx + 1; else → output 0 (unknown)
Example: V=[100,300,500,200,999], M=[100,200,300,400,500] → output=[1,3,5,2,0]
```

**MBDT (MergeBucketizedDense Transform)**：将continuous features映射到discrete bin indices for embedding lookup。
```
Algorithm:
Input: X ∈ R^{F×B}, border lists {B_f} for each feature f
For each feature f, batch element i:
  1. For each border value b_k in B_f:
      if X_{f,i} < b_k → bin = k (first match)
  Output: Y_{f,i} = bin_index + feature_offset
Example: feature 0 borders [0.3, 0.6], values [0.1, 0.4, 0.8] → bins [0, 1, 2]
```

**Batch Event Truncate**：截断sequence learning中的event-based features (EBF)。
```
Input: nested jagged tensors — outer_lengths, inner_lengths, values — for multiple features
Operation: per-user, per-feature, truncate to top-N events (coordinate across all features simultaneously)
Output: truncated nested tensors preserving multi-feature consistency
```

术语一般如何实现？如何使用？
Preprocessing operators在训练中由distributed workers执行，在推理中嵌入model module内同步执行（preprocessing latency直接影响end-to-end inference response time）。KernelEvolve通过自动生成融合Triton kernel（将多个PyTorch operator融合为单个accelerator launch，消除intermediate tensor materialization和host-device synchronization）优化这些operators。MapIdTransform在MTIA v2i上实现3.28-4.07× speedup；MBDT实现2.94-9.25× speedup；Batch Event Truncate batched kernel实现1.4-14.5× speedup（vs per-feature sequential baseline）。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---
