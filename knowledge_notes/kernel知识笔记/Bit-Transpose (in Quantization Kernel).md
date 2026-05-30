## Bit-Transpose (in Quantization Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bit-Transpose（比特转置）是 Any-Precision LLM 等非均匀多精度量化 kernel 中的一种数据重排操作。在非均匀量化中，每个权重存储为 centroid index（如 4-bit 模型的 0-15），推理时以 p 个独立比特平面格式存储和加载（每平面 1 bit/元素）。为了将 p 个独立的 1-bit 平面重组为一个 p-bit 的 centroid index 用于 table lookup，需要将 p 个平面"转置"——即对每个权重位置，从 p 个比特平面中分别取出对应 bit，合并为 p-bit 整数索引。在 GPU 上，这个操作涉及大量的 bitwise shift + OR 运算和跨比特平面的不规则内存访问。AnyBCQ 论文（Table 7）的 kernel 延迟分解显示：bit-transpose 是 Any-Precision LLM kernel 的最大开销来源，占 kernel 总延迟的 35-58%（取决于矩阵形状和比特宽度），远高于 centroid table lookup 的 9-17%。AnyBCQ 通过使用 BCQ 二值格式彻底消除了 bit-transpose——BCQ 的比特平面直接是可计算操作数（{-1,+1}），无需转置为 index。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Any-Precision LLM kernel 中 bit-transpose 的计算过程：

```
# 输入: p 个比特平面 BP_0, BP_1, ..., BP_{p-1}
# 每个 BP_i 是 M×K 的 packed binary tensor (1 bit/元素)
# 输出: index_matrix ∈ {0,...,2^p-1}^{M×K}

for m in range(M):
    for k in range(K):
        index = 0
        for i in range(p):  # p=2/3/4
            bit = extract_bit(BP_i, m, k)  # 从第 i 个平面取第 (m,k) 位置的 bit
            index |= (bit << i)             # 移位合并
        index_matrix[m, k] = index  # 0~2^p-1 的 centroid index

# 然后用 index_matrix 查 centroid table:
# weight_deq[m,k] = centroid_table[index_matrix[m,k]]
```

GPU 实现中的瓶颈：(1) 每个权重的 p 个 bit 来自 p 个不同比特平面，访问内存位置不同（跨平面、非连续）；(2) 提取并合并 p 个 bit 需要 p 次 loaded bit + p-1 次 shift + p-1 次 OR（或等效的 bitwise 操作）；(3) 输出 index 后续还要用于 shared memory / global memory 中的 centroid table lookup，形成两阶段依赖。Table 7 数据显示，bit-transpose 占比随 K 增大而增加（K=14336 时最高 57.71%），说明不规则内存访问是主要瓶颈而非纯计算。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Bit-transpose 的优化方案：(1) SIMD 加速——使用 CUDA 的 `__byte_perm()` 或 warp shuffle 指令批量执行 bit 重组；(2) 预转置存储——在模型加载时预先将比特平面存为"按权重交织"的格式（每个 byte 存某权重的 p bits），消除推理时的 transpose 开销，但代价是存储格式与"按需只加载 p 个平面"的带宽节约目标冲突；(3) 消除法——AnyBCQ 的方法：不使用 centroid index 格式，直接用 BCQ 二值平面操作，从根源消除 transpose 需求。BCQ 的比特平面本身就是操作对象（±1 乘激活），不经过 "index → centroid" 的中间表示。这也是 BCQ 在多精度场景下硬件效率优于非均匀量化的根本原因。

涉及论文标题：
- AnyBCQ Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs

---
