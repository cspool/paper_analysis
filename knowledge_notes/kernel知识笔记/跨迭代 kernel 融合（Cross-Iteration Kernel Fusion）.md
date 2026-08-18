## 跨迭代 kernel 融合（Cross-Iteration Kernel Fusion）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 跨迭代 kernel 融合是 MNEMOS 针对 FFT/IFFT 的 kernel 级融合技术：盲旋转主循环每迭代含一次 FFT（正变换）与一次 IFFT（逆变换），而 FFT 与 IFFT 使用同一组系数的共轭版本——twiddle factors（旋转因子）与 precomputation factors（预处理因子，如 Tangent FFT 的 ω^j）。标准实现中每迭代都从全局内存重载这两套系数；MNEMOS 构造跨迭代边界的融合 kernel，把迭代 i 的尾部（IFFT）与迭代 i+1 的头部（FFT）作为单个 workload 执行，使两套系数直接从片上（寄存器/共享内存）跨迭代复用，完全消除主循环内对这些系数的冗余全局载入。收益随分解层数 ℓ 线性增长（ℓ 越大，融合窗口覆盖更多迭代）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 融合 kernel 的结构（图 6，盲旋转循环展开 2 次示意）：
```
# 原（未融合）：每迭代 IFFT 输出写回全局，下次 FFT 再读；系数每迭代重载
for i in 0..n-1:
    c_f = FFT_i(decomp(rot))              # 用 twiddle/precomp 系数（全局读）
    acc = IFFT_i(c_f ⊙ BSK)               # 用共轭系数（全局读）
    acc += prev

# MNEMOS（融合）：迭代边界内一次 kernel 同时处理 IFFT(i) 尾部 + FFT(i+1) 头部
fused_kernel(i):
    c_f = FFT_i(decomp(rot_i))            # 系数已在片上（上一轮融合 kernel 复用）
    acc = IFFT_i(c_f ⊙ BSK)               # 尾部：IFFT 输出留在片上
    # ——迭代边界（无 kernel 启动、无全局往返）——
    c_f2 = FFT_{i+1}(decomp(rot_{i+1}))   # 头部：复用同源共轭系数（片上）
    acc2 = IFFT_{i+1}(c_f2 ⊙ BSK)
    ...
```
- Annotations：`⊙` 为 BSK Hadamard 乘；跨迭代融合消除的是"系数集（twiddle/precomputation）"的重复全局载入，而非 BSK（BSK 复用由分块机制解决）；ℓ 增大 → 每迭代 FFT/IFFT 次数增多 → 系数复用收益线性放大（图 16(b)）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：把相邻迭代的 FFT/IFFT 段合并进单个 CUDA kernel（循环体内跨迭代连续执行），系数表预载入共享内存/寄存器并跨迭代保留。与一般 kernel fusion（消除中间张量 HBM 往返）的区别：此处消除的是"每迭代重载的系数表"访存。使用场景：任何逐迭代重复使用同源系数的变换循环（FFT/IFFT、NTT、负循环卷积）；在 TFHE 盲旋转中与 BSK 分块、Tensor Core FFT 组合，共同把 PBS 从 memory-bound 转为 compute-bound（Para-D 最高 3.01× A100）。

涉及论文标题：
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization
