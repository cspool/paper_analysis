## Compute-based Gaussian Codes for Trellis Quantization（基于计算的高斯码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Compute-based Gaussian Codes 是 QTIP 提出的新型 codebook 设计，以计算代替查找表来生成伪随机近似高斯分布的 codebook 值。目标：在 bitshift trellis 中，相邻权重组共享大量比特位，若 codebook 值顺序排列会产生强相关性（Figure 3 far left），须通过伪随机 codebook 去相关。传统 RPTC 存储/应用随机排列开销过大，QTIP 的 compute-based codes 用极少量 GPU ALU 指令即时生成伪随机高斯值。三种 codes：(1) **1MAD** (2 instr)——LCG (ax+b mod 2^32) → 求和 4 个 8-bit unsigned ints（近似高斯）→ scale/shift；(2) **3INST** (3 instr)——LCG → XOR bottom 16 bits 修改 magic FP16 数的尾数/指数/符号位 → XOR top 16 bits 同操作 → m1+m2 近似高斯（两镜像指数分布之和）；(3) **HYB** (摊销 2 instr/weight)——x²+x mod 2^32 hash → 取 bits 作为 2^Q×2 LUT 索引 → XOR bit 15 翻转第二分量符号。HYB codebook 仅 2KiB (Q=9)，可放入 GPU L1 cache。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
1MAD 码伪代码（Algorithm 1）：
```
输入: L-bit 0-left-padded integer x, uint32 a=34038481, b=76625530.
x ← (a·x + b) mod 2^32          // LCG, 1 MAD 指令
x ← (x & 255) + ((x>>8)&255) + ((x>>16)&255) + ((x>>24)&255)  // 求和 4×8-bit, 1 vabsdiff4
x ← (x - 510) / 147.8           // scale/shift, 复用 MAD 或单独 FMA
输出: 伪随机近似高斯 x ~ N(0,1).
```
3INST 码伪代码（Algorithm 2）：
```
输入: x, a=89226354, b=64248484, m=0.922 (FP16).
x ← (a·x + b) mod 2^32          // LCG, MAD
// 复制 m 到 32-bit 寄存器两半: m32 = (m<<16) | m
x ← x XOR (m32 & mask)          // lop3: XOR + mask, 修改 FP16 尾数/指数
m1 = low16(x) as FP16; m2 = high16(x) as FP16
输出: m1 + m2  // FADD, 近似高斯 (两镜像指数分布之和)
```
HYB 码伪代码（Algorithm 3）：
```
输入: x, codebook C ∈ R^{2^Q × 2} (Q=9 → 2KiB).
x ← (x·x + x) mod 2^32          // hash, MAD
idx ← (x >> (15-Q)) & (2^Q - 1) // LUT index
v ← C[idx]                       // L1 cache lookup, 2×FP16
v[1] ← v[1] XOR (x & (1<<15))   // sign flip via lop3
输出: v (2D 伪随机近似高斯向量).
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
这些 codes 在 GPU 上以 CUDA/PTX 实现，嵌入 GEMV kernel 的权重解码阶段。Codebook 值在寄存器中产生后直接用于矩阵乘累加。HYB codebook 可 fine-tune（类似 QuIP# 的微调流程）。ARM CPU 可用 NEON vqtbl4q_u8 实现 6-bit 1D HYB code（Q=6, V=1）。关键约束：LCG 参数 (a,b) 和 magic FP16 数 m 须精心选择以避免强相关（论文通过遍历搜索确定）。1MAD 和 3INST 不含可训练参数，完全 lookup-free。

涉及论文标题：
- QTIP: Quantization with Trellises and Incoherence Processing
