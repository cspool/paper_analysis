## Kernel Fusion for Hash Encoding (Attention中哈希编码的CUDA Kernel融合)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Kernel Fusion for Hash Encoding 是HATA中将hash编码阶段的连续GPU操作（MatMul→Sign→BitPack→Cache Update）融合为单个CUDA kernel的技术。在PyTorch原生实现中，这四个操作各自需要独立的GPU kernel launch——每个kernel在GPU上仅需数微秒，但CPU需要数十微秒来dispatch，导致GPU计算单元处于空闲等待（kernel launch overhead > kernel execution time）。通过kernel fusion，将四次CPU-GPU同步合并为一次，减少了端到端延迟。

在HATA中，该优化贡献了约7.6%的端到端延迟减少——是三项硬件优化中最小的一项，但对消除"death by a thousand cuts"式的微小kernel launch开销至关重要。

从kernel调度角度拆解术语：

```
# Before (PyTorch native, 4 kernel launches):
K_H_float = torch.matmul(K, W_H)         # Kernel 1: cuBLAS MatMul
K_H_sign  = torch.sign(K_H_float)         # Kernel 2: element-wise sign
K_H_packed = BitPack(K_H_sign)            # Kernel 3: bit packing
K_H_cache  = torch.cat([K_H_cache, K_H_packed])  # Kernel 4: cache append

# After (Fused CUDA kernel, 1 kernel launch):
# Single CUDA kernel:
# Grid: (num_heads, ceil(s/block_size))
# Each thread block:
K_H_fused = FusedHashEncode(K_tile, W_H, K_H_cache_ptr)
# Inside fused kernel:
#   1. Load K_tile[d] from global memory → shared memory
#   2. Load W_H[d, rbit] from global memory → registers
#   3. Compute K @ W_H via tiled matmul (in registers)
#   4. Apply sign() inline (register-level, no write-back)
#   5. BitPack 128 bits → 4 INT32 (register-level, using bit shifts)
#   6. Write packed result directly to K_H_cache global memory
#   7. No intermediate DRAM write-backs between steps
```

术语一般如何实现？如何使用？

HATA实现（https://github.com/gpzlx1/HATA）包含1470行C++/CUDA代码。Fused Hash Encode kernel定义为自定义CUDA kernel，集成到PyTorch via torch.utils.cpp_extension或custom op。与FlashInfer框架兼容——在FlashInfer的attention pipeline中替换标准KVCache update为fused hash encode + cache update。适用场景：任何需要在每decode step做hash encoding后更新code cache的长上下文LLM推理。

涉及论文标题：
- HATA__Trainable_and_Hardware-Efficient_Hash-Aware_Top-k_Attention_for_Scalable_Large_Model_Inference

---
