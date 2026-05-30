## Arbitrary Bit-Width Low-Precision Data Types (任意位宽低精度数据类型，1-8 bit)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
任意位宽低精度数据类型是指bit width在1到8之间（非powers-of-two的任意值）的数值表示格式，用于LLM推理中的模型权重量化。传统量化方法（如INT8、INT4）使用powers-of-two位宽以对齐GPU的byte边界处理，但4-bit可能过于激进（精度损失大）、8-bit相对浪费（带宽节省少）。5-bit、6-bit、7-bit等中间位宽可在精度与效率间取得更优trade-off，但因GPU架构和软件栈以byte为最小处理单元而缺乏高效支持。

Tilus支持三类family共21种低精度类型：(1) 有符号整数int2-int8；(2) 无符号整数uint1-uint8；(3) 浮点数float3-float8（含任意exponent/mantissa分布如e4m3, e3m3, e3m2, e2m2, e2m1, e1m1）。所有类型可在同一参数化程序模板中支持，无需为每种位宽编写单独kernel。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
低精度推理pipeline（以FP16×INT6矩阵乘法为例）：
```
# 预处理：权重layout变换
B_transformed = rearrange(B[i6][K,N] → u8[K/BK, N/BN, ceil(BK*BN*6/8)])

# Kernel内：低精度计算流程
for k in 0..K step BK:
    a_tile = LoadGlobal(A_f16, layout=m16n8k16_compat, offset)     # FP16 activation
    b_tile = LoadGlobal(B_transformed, dtype=u8, layout=local(3).spatial(32), offset)  # u8紧凑加载
    b_tile = View(b_tile, dtype=i6, layout=tensor_core_compat)     # 零开销reinterpret
    b_tile = Cast(b_tile, dtype=f16)                                 # PRMT/LOP3向量化casting
    C_accum = Dot(a_tile, b_tile, C_accum)                          # Tensor Core mma
C_out = Cast(C_accum, f16)
StoreGlobal(C_out)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现基于两点关键洞察：(1) 紧凑存储（bit packing）将低精度元素连续打包进u8字节，可能跨字节边界；(2) layout变换使紧凑存储的字节序列可通过标准u8加载指令高效读取，再在registers内通过View reinterpret恢复到低精度表示。Tilus的Cast操作使用CUDA的PRMT（permute bytes）、LOP3（三输入逻辑操作）和bitwise指令在registers内完成向量化类型转换，无需shared memory往返。通过参数化的程序模板（约200配置per operator, auto-tuning tile大小），所有1-8 bit类型共享同一kernel模板。

涉及论文标题：
- Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation
