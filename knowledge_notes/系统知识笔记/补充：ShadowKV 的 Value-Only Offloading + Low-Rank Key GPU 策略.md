## 补充：ShadowKV 的 Value-Only Offloading + Low-Rank Key GPU 策略

ShadowKV 为 KV Cache Offloading 增加了 Value-Only 卸载的新策略——与 InfiniGen（offload 完整 KV 对到 CPU，每次 decoding 取回稀疏 KV 对）不同，ShadowKV 仅将 value cache 卸载到 CPU，GPU 上保留低秩 key 投影（A, B）、compressed landmarks（L）和 outlier KV 对。

具体优势：
(1) 仅取回 value（非完整 KV 对），PCIe 数据传输减半；
(2) Key cache 通过低秩投影在 GPU 上重建（Tensor Core GEMM），与 value CPU→GPU 取回（cudaMemcpy H2D）通过 CUDA multi-stream 重叠执行，net latency = max(compute, transfer) 而非 sum；
(3) GPU 存储降至原始的 1/6-1/7，支持 6× 更大 batch size。

```
// ShadowKV 异构存储布局
GPU (HBM):
  - A, B: 低秩 key 投影 [s×r + h_kv×r×d]
  - L: landmarks [h_kv×s/c×d]
  - K_outlier, V_outlier: o 个 outlier chunk 的完整 KV
  - Model weights (resident)

CPU (Pinned Memory):
  - V_CPU: 全部 value cache [h_kv×s×d]（非 outlier 部分）

// 解码时数据传输
Stream 1 (GPU compute): K_sparse = Gather(A, I) @ B  // GPU Tensor Core
Stream 2 (PCIe H2D):   V_sparse = cudaMemcpy(V_CPU[I]) // PCIe 4.0 x16
// 两者通过 CUDA event 同步后拼接执行 FlashAttention
```

涉及论文标题：
- ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference
