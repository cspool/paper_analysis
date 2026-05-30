## Distributed Offset Calculation for Paged KV Cache (分布式偏移量计算)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Distributed Offset Calculation（分布式偏移量计算）是一种 GPU kernel 优化技术，用于加速 Paged KV cache 场景下的注意力解码。PagedAttention（Kwon et al., 2023）将 KV cache 存储为非连续的 page，每次访问需通过 page table 间接寻址——从 page table 读取 page index，计算 global memory 地址，再加载 KV 数据。该地址计算使用 64-bit 整数索引（每条指令需多个 32-bit 整数乘法模拟），当 page size 小（如 page size = 1，RadixAttention prefix caching 所需）时，地址计算开销超过数据加载本身，使 kernel 性能严重退化。

该论文的核心洞察：将地址计算的负载分布到同一 warp 内多个线程——每个线程仅计算少量地址，通过 warp shuffle 在组内共享结果，大幅减少每线程的地址寄存器压力和指令数。使 page size 1 的 kernel 速度匹配 page size 64（1.2-1.5× speedup）。

从kernel调度角度拆解术语，给出伪代码。

```
# 128 threads 加载 128×128 block，page_size 任意（含 1）
# 分组：8 groups × 16 threads/group

for t in 0..127 (thread index in warp):
    g = floor(t / 16)                  # group ID: 0..7
    local_t = t % 16                   # within-group thread: 0..15

    # Step 1: 每线程读取其负责的 page table entry
    row = g + local_t * 8              # thread t 负责的 row
    page_idx = page_table[row]         # 从 page table 读 page index
    # 计算该 row 的 global memory 地址（64-bit 整数运算）
    addr = compute_global_addr(page_idx, row, head_dim)

    # Step 2: 通过 warp shuffle 共享地址
    # 对于分配给 group g 的 8 行（g, g+8, ..., g+120）
    for r in g, g+8, ..., g+120:
        # 找到负责该行的线程
        src_thread = g*16 + (r - g) / 8
        load_addr = __shfl_sync(0xFFFFFFFF, addr, src_thread)
        # 使用 cp.async 加载 KV 元素
        cp.async(shared_mem[r], load_addr)
```

关键优化点：
1. 每线程仅存储 **1 行**的地址（而非 16 行），降低寄存器压力
2. Warp shuffle（__shfl_sync）实现组内地址共享，延迟约 1 cycle
3. 8 组的 16 线程各自独立执行，无组间同步开销
4. 消除 page size 对速度的影响——page size 1 无减速

术语一般如何实现？如何使用？

实现在 GLA CUDA kernel 中（https://github.com/Dao-AILab/grouped-latent-attention），使用 PTX 内联汇编。评估结果（H100, GLA-2 kernel）：
- 无优化：page size 1 比 page size 64 慢 1.3×
- 启用优化：page size 1 匹配 page size 64 的速度（1.5× speedup for page size 1）
- 即使 page size 64 也获得 1.2× speedup

适用场景：使用 PagedAttention 且需小 page size 的场景（如 RadixAttention prefix caching 的 page size 1）。与 TMA 互补：TMA 用于 contiguous block（大 page），cp.async + distributed offset 用于非连续 paged access（小 page）。

涉及论文标题：
- Hardware-Efficient_Attention_for_Fast_Decoding

---
