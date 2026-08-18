## MPA（Multi-Precision Arithmetic，多精度算术流水线）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MPA 是 FHE NTT/多项式乘法在无原生宽整数乘法的硬件（如 INT8/FP16/FP64 Tensor Core）上模拟 32×32-bit（或更宽）模乘的技术：把宽乘法拆成 bit-splitting（按位拆分成多段）、多次窄乘法、bit-merging（分段结果合并回宽整数）三步。INT8 TCU 方案（TensorFHE/WarpDrive）需要把 32-bit 乘法拆成 16 次 8×8-bit 子乘法，MPA 流水线开销占 INT8-TCU-NTT 总计算时间的 26%~28%（图 3）。
- HyperDrive 用 FP64 TCU（53-bit 尾数精度）实现轻量 MPA：单次 32-bit 乘法把一个乘数拆成两个 16-bit 部分（高/低），分别与另一个 32-bit 乘数做 FP64 乘法，乘积 ≤48-bit，MMA 累加 8 个乘积后仍不超 FP64 精度，最后用 INT64 位合并——单次 32-bit 乘法仅需 2 次 FP64 乘法（对比 INT8 方案的 16 次）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- FP64-TCU 上的 32-bit 模乘（配合 Inner-NTT 的 MMA 执行）：
```
# 输入：a, b ∈ Z_q (32-bit)，q < 2^32；FP64 MMA 累加 8 个乘积
a_hi = a >> 16;  a_lo = a & 0xFFFF        # bit-splitting：高/低 16 位
# MMA1/2：数据矩阵 A 分别乘 TFM 的高/低 16-bit 分量（同一数据两次 MMA）
P_hi = a_hi * b;  P_lo = a_lo * b          # 各 ≤48-bit，FP64 精确表示
acc = Σ (P_hi<<16 + P_lo)                  # 8 个乘积累加 ≤53-bit
r   = BitMerge(acc) mod q                  # INT64 位合并 + 模约减（ModRed）
```
- Annotations：radix-64 Inner-NTT 中每个 warp 的 4 次 MMA（MMA1/2 乘 TFM 高/低 16-bit 分量、MMA3/4 第二级 radix-8）+ 中间的 Bit-Merge/ModRed/EWMult 即一条完整 MPA 流水；相对 INT8 方案，子乘法数从 16 降到 2，MPA 占 NTT 总延迟从 26%~28% 降到 5.0%~7.3%（相对 -84%~-91%，图 16）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：把 NTT 的 32-bit 乘法改写为"数据矩阵 × twiddle factor matrix 高低位分解"的两次 MMA + 寄存器内位合并/约减；INT8 方案（WarpDrive/TensorFHE）则用 8×8-bit 子乘法链 + 进位处理。使用场景：任何用 TCU 加速高精度整数/模运算的 kernel（FHE NTT、多项式乘）；HyperDrive 用它支撑 radix-64 Inner-NTT 的 FP64 TCU 映射，是"FP64 低 MPA 成本 vs INT8 高 MPA 成本"设计取舍的核心（Table III）。

涉及论文标题：
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration
