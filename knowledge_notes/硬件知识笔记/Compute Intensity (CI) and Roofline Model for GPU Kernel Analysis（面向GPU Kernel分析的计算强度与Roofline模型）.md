## Compute Intensity (CI) and Roofline Model for GPU Kernel Analysis（面向GPU Kernel分析的计算强度与Roofline模型）

术语是什么？通过联网搜索让回答具体和精准。
Compute Intensity (CI) = FLOPs / Bytes Transferred，是Roofline模型的X轴参数，用于判断GPU kernel是compute-bound（高CI，受限于峰值算力）还是memory-bound（低CI，受限于内存带宽）。Roofline模型在log-log或线性坐标中画出两条上界：(1) 斜线：performance = bandwidth × CI（memory-bound ceiling）；(2) 水平线：peak compute throughput（compute-bound ceiling）。两线交点为ridge point。ZipServ论文用Roofline模型分析lossless compression在推理中的效应：标准GEMM decode阶段CI = MNK/(MK+KN+MN)，decoupled pipeline下CI下降约62%（M=K=4096时），因为解压后的中间权重buffer增加memory traffic；fused ZipGEMM下CI反而比标准GEMM高约50%，因为直接从DRAM读压缩权重（减少25-30% bytes）。这定量解释了为何ZipGEMM在memory-bound decode阶段有效。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
GPU硬件上CI的计算依赖于具体memory hierarchy level：
1. HBM-level CI（最常用）：FLOPs / DRAM bytes read，反映kernel的global memory efficiency
2. L2/L1-level CI：FLOPs / cache bytes，反映data reuse on chip
3. Register-level CI：FLOPs / register file bandwidth，反映compute unit utilization
ZipServ的Roofline分析：
- RTX4090: peak BF16 Tensor Core 165 TFLOPS, HBM bandwidth 1008 GB/s
- Ridge point: CI_ridge = 165 TFLOPS / 1008 GB/s ≈ 164 FLOPs/Byte
- Standard GEMM decode (M=K=4096, N=32): CI ≈ 32×4096×4096 / (4096² + 32×4096 + 32×4096) ≈ 2048 FLOPs/Byte → 已远超ridge point，但实际上是memory-bound因为decode的batch size小导致实际FLOPs受限于weight loading
- 更精确的decode CI公式（考虑GEMM中权重读取占主导）：CI ≈ N×K / K = N（当M≈K），batch size 32→CI≈32 FLOPs/Byte → 明显memory-bound

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
工具链：
- NVIDIA Nsight Compute：直接提供Roofline chart和Speed-of-Light metrics，可逐kernel分析
- AMD Omniperf：对应AMD GPU的Roofline分析
- 手动计算：根据kernel的FLOPs count（从matrix dimensions推导）和profiled DRAM bytes（从Nsight或nvprof获取）计算CI并绘制Roofline图
使用方式：
1. Profile目标kernel获取execution time和DRAM read volume
2. 计算achieved FLOPs/s和CI
3. 在Roofline图上定位kernel → 判断优化方向（减少memory traffic或增加compute utilization）
4. 对于memory-bound kernel：优先减少data movement（如compression, kernel fusion, data reuse）
5. 对于compute-bound kernel：优先提高compute efficiency（如Tensor Core utilization, instruction mix optimization）

涉及论文标题：
- ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression
- Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems
- QuCo: Efficient and Flexible Hardware-Driven Automatic Configuration of Tile Transfers in GPUs

