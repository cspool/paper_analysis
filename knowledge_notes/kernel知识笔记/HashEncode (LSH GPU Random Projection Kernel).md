## HashEncode (LSH GPU Random Projection Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

HashEncode是MagicPIG在GPU上执行的LSH哈希码计算kernel：对于每个decode step产生的query向量q∈R^{1×d}，将其与共享随机投影矩阵W∈R^{d×(K×L)}相乘后取符号，得到K×L bit的哈希码q_code = Sign(q @ W)。该kernel是compute-bound的（不是memory-bound），因为W仅400KB~825KB，所有attention head共享。计算开销CO（random projection FLOPs / 原始模型线性投影FLOPs）仅1.8%~8.5%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
// HashEncode GPU Kernel (per query)
// Input: q ∈ R^{1×d}  (broadcast to all heads or per-head)
//        W ∈ R^{d×(K×L)}  (shared across all heads, fixed during decoding)
// Output: q_code ∈ {0,1}^{K×L}  (bits packed)

// Matrix multiply + sign extraction
q_code_float = q @ W  // Shape: [1, K*L], matmul on GPU
q_code = (q_code_float > 0).to(torch.int8)  // Sign → bit
// Pack K bits per byte for efficient storage/transmission

// 传输到CPU: q_code (K×L bits ≈ few KB)
```

**GPU kernel调度特点**：
- compute-bound kernel（小矩阵乘），GPU利用率高
- 所有head共享W → 仅需1次matmul（GQA下head数少的额外优势）
- 内存开销：W仅384KB (K=10,L=150,d=128) 到 825KB (K=11,L=300)
- 计算开销CO=3.8% (8B, K=10,L=150) 到 8.5% (8B, K=11,L=300)

术语一般如何实现？如何使用？

在PyTorch中实现为`q_code = (q @ W).sign()`，属于标准matmul kernel（cuBLAS）。由于LLM decoding是memory-bandwidth-bound的（主要时间花在加载参数/KV cache而非计算），3.8%~8.5%的额外compute-bound计算对wall-clock时间影响极小。bit packing后q_code通过PCIe以极低带宽传输到CPU。

涉及论文标题：
- MagicPIG: LSH Sampling for Efficient LLM Generation

---
