## Weight Layout Transformation for Low-Precision Loading (低精度权重layout变换)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight Layout Transformation是Tilus用来高效加载低精度权重的预处理技术。核心思想：将原始低精度权重tensor的global memory layout变换为标准类型（uint8）兼容的紧凑格式，从而利用硬件友好的coalesced memory access和pipelined async copy，避免低精度bitwise extraction在加载时的开销。例如，将i6[K,N]权重变换为u8[K/BK, N/BN, ceil(BK*BN*6/8)]，每个tile内的bits连续排列为u8字节序列。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
变换过程（Figure 9）：
```
# 输入: i6[BK, BN] weight tile, 32 threads
# 输出: u8[ceil(BK*BN*6/8)] compact bytes
transform_kernel:
    b_in = ViewGlobal(w_ptr, dtype=i6, shape=[K, N])
    b_out = ViewGlobal(t_ptr, dtype=u8, shape=[K/BK, N/BN, ceil(BK*BN*6/8)])
    for each tile [bk, bj]:
        b_reg = LoadGlobal(b_in, dtype=i6, layout=原layout, offset=[bk*BK:, bj*BN:])
        b_reg = View(b_reg, dtype=u8, layout=local(n_bytes_per_thread).spatial(32))
        StoreGlobal(b_reg, b_out, offset=[bk, bj, 0:])
```
变换后的权重使kernel内LoadGlobal可以使用标准u8类型、连续内存访问和pipelined async copy（CopyAsync），然后通过View零开销reinterpret恢复低精度类型。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
变换的关键参数计算：给定每线程n bytes和T线程，使用u8 dtype和layout local(n2).spatial(T).local(n1)，其中n1=gcd(n,16), n2=n/gcd(n,16)。该变换在kernel启动前作为预处理执行一次，变换后的权重在各次推理中复用。Tilus的artifact中，此变换作为模型加载的一部分自动完成。

涉及论文标题：
- Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation
