## Vectorized Casting with PRMT and LOP3 (向量化类型转换)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Vectorized Casting是Tilus将低精度register tensor高效转换为标准浮点类型（如float16）的机制，使用CUDA PTX的PRMT（Permute Bytes）和LOP3（三输入逻辑操作）指令在registers内完成，无需shared memory或inter-thread通信。PRMT指令从两个32-bit源寄存器（视为8个连续字节）中按4-bit selector抽取并重新排列4个输出字节；LOP3指令对三个32-bit输入执行任意布尔逻辑操作（通过8-bit truth table编码）。结合bitwise AND/SHIFT/OR指令，这些操作可在registers内实现高效的向量化dequantization（如int6→float16）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。
低精度INT6→FP16的寄存器内向量化casting流程（单线程视角）：
```
# 输入：32-bit寄存器 r 包含4个packed int6值（bits布局: [6:0, 14:8, 22:16, 30:24]）
# 输出：4×FP16 values (每个占16 bits)

# Step 1: 提取各int6值并符号扩展到8-bit
# 使用PRMT配合selector使各byte独立提取；或使用SHIFT+AND提取4个值
v0 = (r >> 0)  & 0x3F;  v0 = v0 << 26 >> 26  # sign extend int6 to int32
v1 = (r >> 8)  & 0x3F;  v1 = v1 << 26 >> 26
v2 = (r >> 16) & 0x3F;  v2 = v2 << 26 >> 26
v3 = (r >> 24) & 0x3F;  v3 = v3 << 26 >> 26

# Step 2: INT32 → FP16 conversion
# (使用CUDA intrinsic __float2half或PTX cvt指令)
f0 = int2float(v0) * dequant_scale  # dequantize
f1 = int2float(v1) * dequant_scale
f2 = int2float(v2) * dequant_scale
f3 = int2float(v3) * dequant_scale

# PRMT优化版：一次指令处理2个32-bit寄存器对的4字节重排
# 可加速Step 1的byte extraction和rearrangement
```

Tilus编译器在每个低精度类型的Cast指令code emitting时自动选择最优的PRMT/LOP3/bitwise指令序列，实现vectorized casting。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在Tilus的编译流程Step 3（低精度类型降低）中，Cast操作被展开为针对目标硬件（CUDA）的指令序列。开发者只需写`b = Cast(a, dtype=f16)`，编译器自动根据源dtype选择PRMT selector pattern和转换逻辑。在CUDA C中，PRMT通过`__byte_perm(a, b, selector)` intrinsic使用（selector=0x7777时编译器自动mask sign-extend bit），需要sign extension模式时则用inline PTX assembly。LOP3通过`__lop3(a, b, c, truth_table)`或PTX inline assembly访问。

涉及论文标题：
- Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation
