## BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出BitDecoding，一个利用Tensor Cores加速低比特KV Cache解码的GPU kernel系统。核心实现：
  (1) Residual Kernel：fuse computation、quantization、packing进单kernel——用ldmatrix将FP16 KV tensor加载到register（Tensor Cores interleaved layout）→执行矩阵运算（QK^T或PV）→每个thread在register内量化和pack→低比特packed data直接写global memory更新low-bit KV cache。利用ldmatrix的thread-to-register mapping自动induce与Tensor Cores兼容的packed layout，无需global reshape。
  (2) Packing Kernel：fuse dequantization与Tensor Cores计算——mirror Residual Kernel的ldmatrix/mma variant配置→ldmatrix加载packed low-bit data→lop3-based layout remapping (75316420 pattern)→INT4/INT2→FP16高效转换→直接参与mma。硬件指令配置（ldmatrix + mma variant）由GPU架构自动确定，residual block size Nr=Pn×Wn×R根据量化bit-width自动计算。
  (3) Warp-level parallelization：新颖warp layout——Wm=1（decode query length<16）→Wn增大→多个warps并行做dequantization→SM warp scheduler overlap dequantization与Tensor Cores mma。Cooperative softmax：利用shared memory buffer (sTMP + sAcc) 做cross-warp reduction和P矩阵重载，仅0.5% overhead。
  (4) 异步pipeline：register-level软件定义pipeline——ldmatrix加载+Dequant (CUDA Cores) 与mma (Tensor Cores) 异步重叠：第i个slice在Tensor Cores上做mma时，第(i+1)个slice同时从shared memory加载并dequantize。
  (5) 架构专用优化：Hopper上使用STSM PTX指令将dequantized FP16写入shared memory→支持wgmma_SS（B矩阵必须在shared memory）。Blackwell上绕过lop3 remapping，直接使用原生mxfp4/nvfp4 mma指令执行packed 4-bit GEMM。

  实验比较：kernel-level在Blackwell（RTX 5090, RTX PRO 6000）、Hopper（H100）、Ada（RTX 4090）、Ampere（A100）上评估，三种workload setting：Single (bs=1)、Batches (大batch)、Page (page management)。对比baselines：FlashDecoding-v2 (FP16)、Kivi（非fused low-bit kernel）、Atom和QServe（fused CUDA Cores-only kernel）。量化配置：4-bit/2-bit Key, tensor-wise (KT) + channel-wise (KC) scaling。

- 后端平台是什么，配置是什么。
  NVIDIA Blackwell: RTX 5090, RTX PRO 6000（原生MXFP4/NVFP4低精度格式支持，up to 20 PFLOPS）。NVIDIA Hopper: H100（WGMMA指令、warp-specialized pipeline、TMA异步数据加载、STSM指令）。NVIDIA Ada: RTX 4090（带宽受限，DRAM瓶颈显著）。NVIDIA Ampere: A100（高带宽，compute-bound场景）。多GPU: 8×A100 for LLaMA-3.1-70B。

- 评估性能的软件/脚本是什么。修改了什么。
  自研CUDA kernel实现BitDecoding，基于FlashAttention/FlashDecoding架构扩展。修改：(1) 新增Residual Kernel（fuse FP16 computation+quantization+packing）；(2) 新增Packing Kernel（fuse dequantization+Tensor Cores mma with async pipeline）；(3) 修改warp partitioning策略（Wm=1, 增大Wn）；(4) 新增cooperative softmax（shared memory cross-warp reduction）；(5) Hopper版本基于FA-3，用wgmma和tma.copy；Blackwell版本直接用原生mxfp4 mma。使用Nsight Compute profiling分析Tensor Cores utilization、memory throughput、stall cycles。baselines包括FlashDecoding v2/v3（开源，CUDA实现）、Kivi（Triton实现，非fused）、QServe和Atom（CUDA Cores-only fused实现）。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源: https://github.com/OpenBitSys/BitDecoding。BitDecoding kernel执行流程（以GQA、4-bit channel-wise量化、seq_len=128K、Hopper H100为例）：
  1. Query Transformation：Q tensor从[1, (gq, hkv)] reshape为[gq, hkv]，形成更大Q tile完全填充Tensor Cores fragment。
  2. Prefill阶段：Residual Kernel——用TMA/ldmatrix加载FP16 K/V→执行QK^T和PV mma (Tensor Cores)→每thread在register内用__shfl_xor_sync warp-reduce计算scale/zero→in-register INT4量化+pack→packed data写global memory更新low-bit KV cache。前Np = L - (L mod Nr) entries存packed low-bit cache，剩余res_len = L mod Nr存FP16 residual cache。
  3. Decode阶段（逐token）：
     a. 新生成K/V FP16 tensor追加到residual cache→residual cache≤Nr（通常<256）
     b. Packing Kernel: cp.async异步加载Q (global→shared mem)、加载Kpack/Vpack low-bit data和Kp/Vp scale/zero metadata→ldmatrix加载packed data到register→lop3-based 75316420 remapping→INT4→FP16 dequant (CUDA Cores)→异步overlap：slice i做mma (Tensor Cores)同时slice i+1做ldmatrix+dequant→softmax (cooperative cross-warp)→sAcc重载P via ldmatrix→PV mma→output writeback
     c. 每当residual cache达Nr时触发Residual Kernel将其量化写入packed cache
  4. Hopper优化：用STSM将dequantized FP16写入shared memory→wgmma_SS直接访问shared memory (B矩阵)。Blackwell：跳过dequant直接用mxfp4 mma。性能：vs FP16 FlashDecoding-v2: Blackwell 8.6×, Hopper 8.0×, Ada 7.5×, Ampere 4.8× speedup。

