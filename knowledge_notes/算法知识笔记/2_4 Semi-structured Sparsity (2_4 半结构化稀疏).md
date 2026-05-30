## 2:4 Semi-structured Sparsity (2:4 半结构化稀疏)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
2:4 semi-structured sparsity：每组连续 4 个权重中恰好 2 个为零，保证 50% 恒定稀疏率 + 硬件友好规则性。NVIDIA Ampere 起通过 Sparse Tensor Cores 原生支持，约 2× 理论加速。vs unstructured sparsity（灵活但无硬件加速），vs block sparsity（粗粒度但精度损失大）。可扩展为 4:8 等变体。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// W ∈ R^{M×K}, 按列分组为 K/4 组
// 每组 [w₀,w₁,w₂,w₃] → 保留 top-2 幅度, 清零 bottom-2
// Metadata: 2-bit index 编码非零位置

// Sparse Tensor Core MMA:
for tile in [M×K]:
    load W_tile(50% bandwidth) + metadata
    load X_tile
    Y += mma.sp.sync(W_tile, X_tile, metadata)  // 硬件跳过零
```

OBR W4A4KV4+2:4 sparse → perplexity 13.32 (vs 34.76 for SparseGPT+GPTQ)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
NVIDIA CUTLASS 2:4 sparse GEMM API；PyTorch `torch.sparse.semi_structured`；TensorRT-LLM 内置。剪枝算法 ASP/SparseGPT/WANDA 均支持 2:4。需 A100+ GPU。

涉及论文标题：
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs
