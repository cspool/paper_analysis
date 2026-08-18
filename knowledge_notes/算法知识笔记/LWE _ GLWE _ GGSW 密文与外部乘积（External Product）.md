## LWE / GLWE / GGSW 密文与外部乘积（External Product）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 这是 TFHE 的三种内部密文类型与核心计算原语。Torus（环面）T 概念上是 [0,1) 上的实数，实现为 w-bit（通常 32/64）离散定点小数。(1) LWE 密文：加密客户端消息的最小密文，参数化 LWE 维数 n（通常 500–1000），含 n 个 mask 元素 + 1 个 body 元素；(2) GLWE 密文：把 LWE 的每个 torus 标量换成 degree-N 多项式（N 为 2 的幂），一个 GLWE 含 k+1 个多项式（k 为 GLWE 维数），用于编码 LUT 与存 PBS 中间结果；(3) GGSW 密文：构成 bootstrapping key（BSK）的密文类型，每个 BSK 含 n 个 GGSW，每个 GGSW 是 (1+k)^2×l_b 的多项式矩阵（l_b 为 gadget 分解深度），支持外部乘积。外部乘积（External Product，GGSW □ GLWE → GLWE）是自举（盲旋转）的核心运算：本质上是"向量-矩阵乘"，每个元素都是 degree-N 多项式乘法（用 FFT/IFFT 加速）。
- 三者的角色：LWE 在"明文侧"（用户消息），GLWE 在"函数侧"（LUT/test vector），GGSW 在"密钥侧"（BSK）。一次 PBS 中三者反复交互（见盲旋转条目）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 外部乘积的计算过程（论文 Figure 4）：对 GGSW 密文做 gadget 分解得到 l_b 层整数向量，与 GLWE 多项式的各次幂做多项式乘加，最后累加。硬件里每个多项式乘法用 FFT/IFFT：FFT(GGLWE) 与 FFT(分解后的 GGSW chunk) 逐点乘 → IFFT。FlashTFHE 的数据通路（Figure 10/11 伪代码）：
```
for bsk_chunk in BSK:                 # 外层：载入一个 BSK chunk（≤0.8MB）片内复用
    for i, decomp_glwe in rr_ctxts:   # 内层：round-robin 遍历所有在飞 ciphertext
        fft_out = FFT(decomp_glwe)    # FFT-A/FFT-B 每周期产 chunk
        acc += VecMAC(fft_out, bsk_chunk)   # 与 BSK subchunk 做 tiled 乘累加
    # 满 (k+1)*l_b 次累计后 I-FFT → sample extraction
```
- 关键量：一次盲旋转要做 n 次外部乘积迭代；每迭代含 (k+1) 个多项式点积（k=1 时仅 2× 复用）与 l_b 层分解。BSK 总量 O(n·N·l_b)，10-bit 时数 GB——每个 BSK 系数在一次自举中恰好用一次，算术强度极低，是"流式大 key"问题的来源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件：TFHE-rs/Concrete 的 external product 用浮点或近似 FFT/NTT 实现；硬件：所有 TFHE 加速器（MATCHA、Morphling、Strix、FlashTFHE、Trinity/UFC）的核心 MAC/FFT 流水线都围绕外部乘积设计。FlashTFHE 用 VecMAC（512 coefficients/cycle/core）+ 48-bit 定点双实数 FFT 集群实现，支持 N 至 2^16。实现选择要点：FFT 位宽（32-bit 会损失正确性，FlashTFHE 用 48-bit 保证 TFHE-rs 与 Concrete 全参数集正确）、分解深度 l_b（越大噪声越小但 BSK 越大）、k（multi-bit 实用参数强制 k=1 以控制 O(k²) 自举成本）。

MNEMOS 补充视角（ISCA'26，GPU 上外部乘积的 BSK 分块复用）：在 GPU 上执行一次外部乘积（MAC）时，对单个 GLWE 做 MAC 需取 (k+1) 倍于 GLWE 体积的 BSK 数据（BSK 形状 (k+1)ℓ×(k+1)，含 (k+1)ℓ×(k+1)×N×n 个元素），且 BSK 预计算后跨一批 PBS 复用。朴素"整 BSK 缓存进共享内存"不可行（部分参数集 BSK 超 A100 每 SM 192KB 合并 L1/SPM 上限，且过度分配共享内存会蚕食 L1 容量）。MNEMOS 利用 BSK 与傅里叶系数之间是逐元素 Hadamard 积（非一般矩阵乘）的性质做分块（tiling）：单个线程块只需处理一块 TBSK 对一块 TGLWE，同一 BSK 分块被一批中多个 PBS 实例（同卷积层共享参数）并发复用，把复用层级从 L2 提升到 SM 级；分块几何取 8 个连续复数 FP64 元素（128B）对齐内存事务粒度保证合并访问。参数 k 增大时（安全级别依赖 kN，Concrete 常用大 k）BSK 足迹占比上升，该复用收益随之增大（消融：+MAC 单独 1.10×~1.77×，k 大时最显著）。

CASCADE 补充视角（ISCA'26，BSK 的 GGSW 结构与外积的硬件数据流）：CASCADE 中 BSK 是 GGSW 密文（L×(k+1)×(k+1) 多项式矩阵，每个元素是 N 阶多项式），中间密文 ACC 是 RLWE（(k+1) 向量，N 阶多项式），外积 = 矩阵-向量乘。硬件上外积映射到 HC 的 VMA（Vector Multiplication-Add）单元：FFT 域里 BSK 多项式与 ACC 多项式变成逐系数相乘，VMA 由向量乘法单元（逐系数乘）+ 累加器（逐系数加）组成；FFT 单元做 log2N 级 butterfly（BU 个并行 butterfly 单元，2·BU 系数/级，总约 log2N·N/(2·BU) cycle），IFFT 单元因 Decomposition 单元使 FFT 侧多项式数更多而分配更少资源。BSK 总量 126 MB（参数集 III 需 112 MB、IV 需 90 MB）全部驻留分布式 SRAM——对比 CKKS 需 GB 级密钥，TFHE 的 10s-100s MB 量级正是"可全部片上驻留"的前提（见"Bootstrapping Key（BSK，自举密钥）"条目）。

涉及论文标题：
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator
