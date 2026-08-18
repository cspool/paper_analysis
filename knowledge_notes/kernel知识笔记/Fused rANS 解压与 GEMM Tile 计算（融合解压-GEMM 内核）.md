## Fused rANS 解压与 GEMM Tile 计算（融合解压-GEMM 内核）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
把熵解码直接融入 GEMM 内核的"压缩权重执行原语"：权重以压缩 bitstream 常驻全局内存，解码 warp 按 GEMM tiling 序把 tile 解压进 shared memory，GEMM warp 立即用 tensor core 消费，解码与矩阵乘在同一线程块内以生产者-消费者方式流水重叠，解压后的权重从不写回全局内存。对比层粒度 decompress-store-compute（NeuZip/DFloat11：整层解压写回全局内存再 GEMM），融合内核消除层同步屏障、冗余全局内存流量与额外解压缓冲。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm V.1 伪代码（论文）：
```
Shared: T~ (decode table, 从全局拷贝), A(0),A(1) in R^{M×K} (双缓冲), ready[2] (原子标志)
for k = 0 .. K_tiles-1:
    b = k % 2; p = 1 - b
    # 生产者：warp 0 只跑 rANS 解码
    Warp0: RansDecodeTile(A(b), stream[k], T~); ready[b] = 1
    # 消费者：warp 1..W 只跑 tensor-core GEMM
    if k == 0: wait(ready[b]==1); GemmTile(A(b), B(k), C); ready[b] = 0
    else:      wait(ready[p]==1); GemmTile(A(p), B(k), C); ready[p] = 0
    __syncthreads()
```
Annotations：每个权重只解码一次（解码与计算共享 on-chip footprint，不再从 HBM 重读）；ready 用 cuda::atomic_ref<int, thread_scope_block> 实现；权重全局内存流量从 V_B = (M/M_t)·K·N 降为 (M/M_t + α − 1)·K·N（每权重只取一次压缩形式）。性能：tile 对齐（vs DietGPU 解码 + CUTLASS GEMM 分离两段）3.3–8.2×，再加双缓冲总计 4.0–10.1×（Qwen-1.5B 4.41×、DeepSeek-67B 6.71×、Llama-405B 10.06×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：以 CUTLASS plugin 式投影算子 override 库级 GEMM 实现 + DietGPU ANS 内核扩展 + PyTorch wrapper；tile 几何由 profiling 选定（128×32/256×64/128×128；A100 32×128、H200 64×256）。批大小行为：小 batch 时 tensor 管线先耗尽、内核 decoder-bound（退化为 decode-then-GEMM）；大 batch 时 GEMM 主导、解码完全隐藏。效果：A100 大 batch 达 CUTLASS 的 1.0–1.1× 内、H200 最高 1.2× 超越 CUTLASS；vs NeuZip 最高 ~10×、vs DFloat11 ~6–7×。使用：压缩权重模型推理、显存受限的 serving（batch 放大 1.3–4.8×）。

涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression
