## Tile Swizzle 与确定性驱逐（swizzle 布局 + 静态替换序）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Swizzle 是 GEMM kernel 对 shared memory 中 tile 的布局重排（XOR 布局、K-strided interleave 等），用于消除 bank conflict、使 tensor core 从 shared memory 无冲突取片段；threadblock swizzle 则把 blockIdx 映射重排以改善 L2 locality。本论文扩展出一个"确定性驱逐"机制：大维度 GEMM 按固定 tile-swizzle 遍历序取 tile，当活跃 tile 工作集暂时超过 shared memory 容量时，按该静态序确定驱逐顺序（recency 由 swizzle 访问序静态定义，无需运行时记账），被逐 tile 的解压形式暂存小 decompression buffer 以避免近期重复解压，压缩形式保留在全局内存。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
访问序 = fixed_swizzle_order(tile grid)        # 由 GEMM tiling schedule 静态确定
for tile in 访问序:
    if tile not in shared_mem and shared_mem full:
        victim = next_to_evict(访问序)          # 按静态序选 victim（无 LRU 记账）
        stash(victim, decompress_buffer)        # 暂存解压形式防近期重访
    decode(tile -> shared_mem)                  # 压缩形式常驻全局内存
    gemm(shared_mem tile)
```
Annotations：驱逐顺序离线可知 → 零额外 bookkeeping；与"解压一次、不再从 HBM 重读"原则结合：重复访问命中 shared memory 或 decompression buffer，避免重复 rANS 解码。论文用于压缩 tile 工作集管理（decoded tile 超 shared memory 容量时）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CUTLASS 提供 hgemm_swizzle.h / igemm_swizzle.h（16-bit/8-bit tile 转置布局）与 identity_block_swizzle.h（threadblock→GEMM 分块映射，L2 局部性）；CuTe 用 Layout 代数表达 swizzle。本论文的确定性驱逐是 swizzle 序的延伸应用（静态替换而非 LRU）。使用：大维度 GEMM 的 shared memory 管理、压缩/稀疏 tile 的替换策略；与 double buffering 正交（双缓冲管流水、驱逐管容量）。

涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression
