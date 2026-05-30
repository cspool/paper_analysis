## BitBLAS

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BitBLAS 是 Microsoft 开发的高性能低比特 LLM 推理算子库，为 INT4/INT3/INT2 等低比特量化模型提供 GPU kernel 实现。全称 "Bit-BLAS"（结合 bit-level 操作和 BLAS 接口设计）。BitBLAS 是 Ladder 项目（OSDI 2024）的子项目，专注于低精度张量运算的硬件感知优化。在 EfficientQAT 中使用 BitBLAS 评估量化模型的实际推理加速——通过将 FP16 矩阵向量乘法替换为 BitBLAS 的 INT2 kernel，获得 2.9x-4.4x 的前向加速。BitBLAS 通过对不同位宽自动生成优化的 CUDA kernel（利用 Tensor Core 或 CUDA Core），将低比特打包权重直接送入硬件算术单元，避免运行时解量化开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
BitBLAS 的 INT2 矩阵向量乘法（Matrix-Vector Multiplication, GEMV）kernel 执行流程：
```
# BitBLAS INT2 GEMV kernel (简化)
KERNEL int2_gemv(Weight_packed_int2, Input_fp16, Output_fp16):
    smem_w = load_and_depack(Weight_packed_int2, tile_id)  # 解包INT2 weight tile到共享内存
    smem_x = load_input_tile(Input_fp16, tile_id)
    accum = 0
    for k in range(K / TILE_K):
        w_tile = smem_w[:, k*k_step : (k+1)*k_step]       # INT2 weight tile
        x_tile = smem_x[k*k_step : (k+1)*k_step]           # FP16 input tile
        # 低精度MAC + 反量化缩放
        mac_partial = int_mad(w_tile, x_tile)               # INT2 * FP16 → FP32累加
        accum += mac_partial * scale + zero_point_adjust    # 反量化
    output[tile_id] = accum
    return
```
EfficientQAT 在 A100-80GB 上测试的 BitBLAS INT2 加速比（Table 10）：Llama-2-7B size=4096x4096: 3.1x, 11008x4096: 2.9x; Llama-2-13B: 3.6x/3.5x; Llama-2-70B: 3.9x/4.4x。加速比随矩阵尺寸增大而提高，因解包开销在更大矩阵中被摊销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
BitBLAS 通过硬件感知的张量变换（Tensor Transformation）实现低比特加速：核心策略是将不同位宽的量化权重通过 ladder 变换映射到 GPU 硬件原生支持的数据格式和指令（如利用 Tensor Core 的 INT8 mma 指令模拟 INT2/INT4 计算，或使用 CUDA Core 的 bit-serial 执行）。使用方式：(1) 环境安装：`pip install bitblas`；(2) 模型集成：替换 HuggingFace 模型的 Linear 层为 BitBLAS 低比特算子；(3) 代码调用：`bitblas.matmul(weight_packed, input, bit=N, group_size=g)`。BitBLAS 支持 INT2/INT3/INT4 等多种位宽，与 GPTQ、AWQ、EfficientQAT 等量化方法的输出格式兼容。可替代 MLC-LLM 和 Marlin kernel 作为低比特推理后端。

涉及论文标题：
- EfficientQAT Efficient Quantization-Aware Training for Large Language Models

---
