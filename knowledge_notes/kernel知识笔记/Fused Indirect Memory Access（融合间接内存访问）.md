## Fused Indirect Memory Access（融合间接内存访问）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused Indirect Memory Access 是 FlexAttention 中将 BlockMask 的稀疏跳过（sparsity skip）与 PagedAttention 的 page table 映射合并为一次间接内存访问的技术。核心思路是：BlockMask 已经通过 kv_indices 实现了一层间接内存访问（跳过 oblivious block），而 PagedAttention 的 page table 也引入了一层间接访问（逻辑 KV index → 物理 KV index）。FlexAttention 将两层间接访问融合——在编译时将 page table 的逻辑-物理映射应用到 kv_indices 上，使 kv_indices 直接指向物理 KV cache 中的 block 位置，从而实现单层间接访问，无需修改 attention kernel。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
传统方法（vLLM）：手写 CUDA kernel，在 attention kernel 内部先查 page table 获取物理地址，再执行注意力计算。这增加了 20-26% kernel overhead，且每种 attention 变体需要独立的手写支持。

FlexAttention 方法：
```
# 编译时：将 page table 映射融入 kv_indices
for row in range(num_rows):
    for i in range(kv_num_blocks[row]):
        logical_block = kv_indices_original[row, i]
        physical_block = page_table[batch, logical_block]
        kv_indices_fused[row, i] = physical_block

# 运行时：单层间接访问
for i in range(kv_num_blocks[row]):
    phys_col = kv_indices_fused[row, i]
    K_tile = load_K_physical(phys_col * BS)  # 直接从物理 KV cache 加载
    V_tile = load_V_physical(phys_col * BS)
    # ... 标准 attention 计算 ...
```

同时，mask_mod 和 score_mod 也通过 `converted_mask_mod` 和 `converted_score_mod` 自动适应：维护一个物理 block → 逻辑 block 的映射向量（O(1) overhead），在调用用户定义的 mask_mod/score_mod 前将物理 KV index 转换回逻辑 KV index。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FlexAttention 的 paged attention 支持无需用户修改 mask_mod/score_mod 定义，仅需在调用 `flex_attention` 时传入 page table 信息。系统自动完成 BlockMask 的物理-逻辑转换。实测开销 <1%（远低于 vLLM 的 20-26% overhead），原因是不引入任何 kernel 代码修改，仅依赖 fused indirect memory access。

涉及论文标题：
- Flex Attention: A Programming Model for Generating Optimized Attention Kernels
