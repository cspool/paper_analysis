## Tile-Based Communication (Multi-GPU)

术语是什么？
Tile-Based Communication是Iris提出的多GPU通信编程模型，将跨GPU数据传输操作对齐到与Triton tile计算模型相同的粒度（BLOCK_SIZE_M × BLOCK_SIZE_N）。与传统的bulk-synchronous通信（整个tensor完成计算后才开始跨GPU传输）不同，tile-based communication允许每个tile产出后立即通信——例如在GEMM kernel的main loop中，每个K-iteration产出一个BLOCK_M×BLOCK_N的C_tile后，立刻通过iris.store将其scatter到所有remote GPU。通信原语（load/store/get/put/copy/atomic_*）均操作于tile粒度，与Triton的tl.load/tl.store/tl.dot处于同一语义空间和同一kernel内。

从kernel调度角度拆解术语：
Tile-Based Communication在Fused Sequential GEMM+All-Scatter中的执行流程：
```
@triton.jit
def fused_gemm_all_scatter(A, B, C, heap_bases, ...):
    pid = tl.program_id(0)
    total_tiles = ceil(M/BLOCK_M) * ceil(N/BLOCK_N)
    for tile_id in range(pid, total_tiles, NUM_SMS):
        # Compute Phase (standard GEMM loop)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(A + offsets_a)
            b = tl.load(B + offsets_b)
            acc += tl.dot(a, b)  # Tensor Core MMA
        # Communication Phase (same kernel, immediately after tile compute)
        c = acc.to(C.dtype.element_ty)
        for remote_rank in range(world_size):
            iris.store(C + offset, c, cur_rank, remote_rank, heap_bases, mask=mask)
```
关键特征：(1) 同一kernel内无launch/teardown overhead；(2) 通信单位为tile而非整个tensor；(3) 值语义直接从register写remote GPU memory；(4) Triton编译器看到通信操作，可联合调度计算与通信。

术语一般如何实现？如何使用？
Iris提供两类device-side API——值语义(load/store从register到remote memory)和指针语义(get/put/copy做buffer间拷贝)。所有API均需heap_bases参数做指针翻译，翻译开销近乎为零(heap_bases 64 bytes常驻L1 cache)。

涉及论文标题：
- Iris: First-Class Multi-GPU Programming Experience in Triton
