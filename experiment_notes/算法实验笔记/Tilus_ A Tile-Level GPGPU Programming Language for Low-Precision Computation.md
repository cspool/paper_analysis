## Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是Tilus生成的任意位宽（1-8 bit）低精度矩阵乘法kernel，支持三种数据类型族：有符号整数（int2-int8）、无符号整数（uint1-uint8）、浮点数（float3-float8，含任意exponent/mantissa分布如e4m3、e3m3、e3m2、e2m2、e2m1、e1m1）。低精度计算流程：权重在kernel启动前做global memory layout变换（如i6[K,N] → u8[K/BK, N/BN, BK*BN*6/8]）实现连续内存访问；kernel内通过LoadGlobal加载为标准u8类型，View指令零开销将register tensor reinterpret为低精度类型和layout，Cast指令使用PRMT/LOP3/bitwise指令做vectorized casting到float16；最后Dot指令调用Tensor Core mma.m16n8k16完成矩阵乘累加。

  实验比较：vs cuBLAS FP16 kernel（标准精度baseline），覆盖uint1-uint8、int2-int8、float3-float8共21种低精度数据类型，矩阵乘法维度BS=16, K=8192, N=57344（来自Llama-3.3-70B的典型matmul）。同时与Triton、Ladder、QuantLLM、Marlin在uint8、f6e3m2、int4、uint4、uint2、uint1上比较speedup。

- 硬件平台是什么，配置是什么。
  NVIDIA L40S GPU (48 GiB)，GPU driver 565.57.01，CUDA 12.6.3。跨硬件验证：NVIDIA A100 (Ampere)、H100 (Hopper)。

- 模型是什么。数据集和bench分别是什么。
  三个LLM：Gemma-2-9B、QWen2.5-32B、Llama-3.3-70B-Instruct。使用dummy inputs和weights（系统性能不依赖具体输入内容权重内容），模型metadata从Hugging Face Hub自动获取。Benchmark使用CUDA Events测量latency，每次kernel执行50次取median，L2 cache每次执行前清除。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/NVIDIA/tilus，artifact: https://github.com/yaoyaoding/tilus-artifacts

  低精度FP16×INT6矩阵乘法pipeline（图2）：
  ```
  # 预处理（kernel启动前执行transform_b kernel）
  # 将权重tensor从 i6[K, N] 变换为 u8[K/BK, N/BN, ceil(BK*BN*6/8)]
  # 每个tile [BK, BN] 内的 16*8*6=768 bits 以连续 u8 字节存储 (96 bytes)

  # Kernel内部 (Tilus VM program)
  C_accum = AllocateRegister(f32, [BM, BN])        # 累加器
  for k in range(0, K, BK):
      # Step 1: 加载activation tile (标准f16)
      a_tile = LoadGlobal(A_view, layout=m16n8k16_A, offset=[bi*BM:, k:])
      # Step 2: 加载weight tile (低精度→标准类型通过layout变换)
      b_tile = LoadGlobal(B_transformed_view, dtype=u8,
                          layout=local(3).spatial(32),
                          offset=[k/BK:, bj*BN:, 0:])  # u8加载，连续内存访问
      # Step 3: 零开销reinterpretation
      # 32 threads × 24 bits/thread → 原始为 4×i6, reinterpret为 3×u8
      b_tile = View(b_tile, dtype=i6,
                    layout=local(2,1).column_spatial(4,8).local(2,1))
      # Step 4: 向量化casting (i6 → f16)
      b_tile = Cast(b_tile, f16)  # PRMT + LOP3 + bitwise, 全在registers内完成
      # Step 5: Tensor Core矩阵乘累加
      C_accum = Dot(a_tile, b_tile, C_accum)  # mma.m16n8k16

  # 结果写回
  C_accum = Cast(C_accum, f16)
  StoreGlobal(C_accum, C_view, offset=[bi*BM:, bj*BN:])
  ```

  关键优化：weight loading pipeline（图1c）避免了Triton的shared memory layout conversion和Ladder的缺少pipelining问题。所有低精度类型（1-8bit）可在同一个参数化程序模板中通过改变tile size超参数支持，约200个配置per operator，编译时间~1分钟。
