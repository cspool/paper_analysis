## LUT-based GEMM（查表矩阵乘法 / LUT 加速的混合精度 GEMM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LUT-based GEMM（查表矩阵乘法）是把量化（尤其低比特权重/二值 bit-plane）GEMM 中"高精度激活 × 低比特量化值"的点积预计算并存入查找表，运行期用量化值作为索引直接查表、把乘法换成查表+累加的加速方法。它避免两种低效：(1) 商业加速器不支持 FP-INT mpGEMM 而需先 dequant 权重；(2) 逐元素浮点乘法的高能耗。运作（LUT-GEMM [53]，Fig.2）：把激活分组（通常 4 个一组），该组激活与一组低比特量化值的全部组合点积预计算成 2^4=16 个高精度表项；每 cycle 该组按量化矩阵一列的 binary 值查表，查表次数∝量化位宽（2-bit 权重需 2 次查表）；LUT 在激活 tile × 量化矩阵列间复用，partial sum 累加得结果；新激活组重新生成 LUT。Omni-LUT 在此之上：LGU 把 row-wise 缩放与 zero-point 补偿内嵌进表生成（scale-aware），使 AA-GEMM（Key/Value 量化）也能走 LUT datapath；PE 每 4 激活组共享一个 LUT、32 个 binary weight 并行 Read-and-Accumulate。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 张量计算例子（μ=4 分组，A∈R^{1×K} 激活、B∈{-1,+1}^{K×N} 量化矩阵）：对每 4 元素组 g 构建 LUT：for pattern in 0..15：LUT[pattern]=Σ_{i=0..3} sign_i·A[g+i]（sign 由 pattern 的 bit 决定）；计算时 for n：w_bits=B[g:g+4,n]（4-bit 模式）；output[n]+=LUT[w_bits]。在 Omni-LUT 中进一步按 bit-plane 展开：每个量化操作数（权重 W4 或 KV3/KV4）有 q 个 bit-plane，对每个 plane 做上述查表+累加，最后 Σ_i α_i·partial_i 得点积；LGU 生成表时先对组内 4 个激活各乘 row-wise scale、并在首 plane 内嵌 zero-point 补偿（把激活组×zero-point 向量的点积加进每个表项），从而支持精度最优的 row-wise 量化方向。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现方式：软件侧 NAVER LUT-GEMM（github.com/naver-aics/lut-gemm，首次在 GPU 实现 BCQ 格式 LUT 计算）、FLUTE（LUT 向量化 + 跨 bank 复制消 conflict，2-4×）、T-MAC（CPU in-register LUT，ARM TBL/x86 PSHUF）；硬件侧 FIGLUT（HPCA 2025，custom Read-Accumulate 单元替代 MAC）、LUT Tensor Core（ISCA 2025）、Omni-LUT（ISCA 2026，AW+AA 全覆盖）。硬件实现要点：每 PE 存 LUT（4 激活组 = 16 项 FP16）、32 个量化权重并行查表、half-LUT 符号对称省一半表项（T-MAC 思想）、LUT 生成器与 PE 并行避免 stall。用途：低比特 LLM 推理（W4A16/KV4A16 等 mpGEMM），decode（batch=1、memory-bound）与 prefill 都受益；Omni-LUT 等峰值吞吐下能效比 FIGLUT 高 1.25×-1.91×。

涉及论文标题：
- Omni-LUT: Energy-Efficient LUT-based Accelerator with Hardware-Aware KV Cache Quantization
