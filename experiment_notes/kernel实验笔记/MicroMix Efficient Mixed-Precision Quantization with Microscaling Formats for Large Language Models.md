## MicroMix Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - MicroMix 在 Blackwell GPU 上实现了基于 CUTLASS 的混合精度 GEMM kernel 和 fused reorder-and-quantize kernel。GEMM kernel 支持单一 kernel 内任意混合 MXFP4/MXFP6/MXFP8 通道比例，利用 Blackwell Tensor Core 的 MMA 指令（原生支持 FP4/FP6/FP8 + 融合 scale 反量化），输出 BFloat16。Fused reorder-and-quantize 将通道重排和 block-wise MX 量化合并为一个 kernel，避免 irregular memory access 的开销。实验比较了：(1) 单 kernel 延迟 vs TensorRT FP16/FP8/W4A16；(2) 自定义 GEMM kernel vs CUTLASS 的 TFLOPS 和加速比；(3) prefill 延迟和 decode 吞吐 vs Atom/QuaRot/FP16/INT8；(4) 峰值内存占用。
- 后端平台是什么，配置是什么。
  - NVIDIA RTX 5070Ti Laptop GPU（Blackwell）、RTX 5090（Blackwell）、RTX PRO 6000（Blackwell）。Blackwell Tensor Core 支持 FP4 (E2M1) MMA，FP4 吞吐为 FP16 的 4×、FP8/INT8 的 2×，MMA 指令原生融合 scale factor 反量化。
- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 CUTLASS 实现自定义 MXFP GEMM kernel，支持 MXFP4/MXFP6/MXFP8 的混合精度矩阵乘法。baseline 使用 TensorRT FP16、TensorRT FP8 (per-tensor)、TensorRT W4A16 (per-token)、HuggingFace FP16、Bitsandbytes INT8、Atom/QuaRot 的 INT4 kernel。修改：(1) GEMM kernel 按精度分组分别调用对应的 CUTLASS MXFP GEMM 实例；(2) 实现 fused reorder-and-quantize kernel（将通道重排 + block-wise 量化 + scale 计算融合）；(3) 集成 FlashInfer 进行 KV cache INT4 量化以进一步减少内存占用。
- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源地址：https://github.com/lwy2020/MicroMix
  - Kernel 评估原理与流程：
    ```
    1. 输入：FP16 激活 X ∈ R^{M×K}，已量化权重 W ∈ MXFP_format（包含 scale）
    2. 配置加载：读取该层的 p4/p6/p8 比例和排列 σ
    3. Fused Reorder-and-Quantize Kernel（GPU kernel）：
       - 从 global memory 读取 FP16 X
       - 按 σ 对通道索引重排（在 shared memory / register 中完成）
       - 分组：前 p4*K 通道 → G4, 中 p6*K 通道 → G6, 后 p8*K 通道 → G8
       - 对每组内每 32 个元素 (block_size=32)：
         s = 2^{floor(log2(max(|block|))) - b}  (E8M0)
         Q(x) = round(clip(x/s, -q_max, q_max))
       - 输出三组 MX format 张量（含 element + shared scale）
    4. GEMM Kernel（CUTLASS-based MMA）：
       - 对每组精度分别：加载 A_tile (MXFP) + B_tile (MXFP) + scales → Tensor Core
       - MMA.884 或类似指令：每次操作 = A·B + scale_dequant → FP32 accum
       - 累加到 BFloat16 输出 tile，写回 global memory
    5. 输出：三组结果按 σ^{-1} 恢复原通道序，得到 BF16 Y ∈ R^{M×N}
    ```
  - 性能评估：对 M={1,2,4,8,16,32,64,128}, N=K=4096 测量 TFLOPS 并与 CUTLASS 对比。对 sequence length {128,256,512,1024,2048,4096} 测量单 kernel 延迟并与 TensorRT 对比。
  - 关键结果：RTX 5070Ti laptop 上 MicroMix kernel 2.45-2.93× vs TensorRT-FP16, up to 1.45× vs TensorRT-FP8。RTX 5090 上 2.29-3.38× vs TensorRT-FP16, up to 1.74× vs TensorRT-FP8。自定义 GEMM vs CUTLASS：W6A6 在 M=32 时最大 5.0× 加速。Fused reorder-and-quantize 仅占总 kernel 时间的 <20%。RTX PRO 6000 上 MicroMix prefill 延迟约 Atom/QuaRot 的 15%，decode 吞吐约 Atom 的 1.82-3.02×。memory 减少：vs FP16 减少 2.29-2.84×，vs INT8 减少 1.60-2.01×。
