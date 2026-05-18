## BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

- baseline方法是什么？
  Baseline有两种：(1) **Non-fused low-bit KV cache attention**（如Kivi）：将mixed-precision attention分解为多个standalone kernel（量化kernel、dequantization kernel、attention kernel分离），每个kernel各自load/store intermediate data到global memory，无on-chip data reuse，导致高kernel launch overhead和global memory traffic。(2) **CUDA Cores-only fused attention**（如Atom、QServe）：将量化/dequantization和attention融合到单kernel，但dequantization和矩阵乘法（GEMV/GEMM）全部在CUDA Cores上通过FMA执行，Tensor Cores完全闲置。CUDA Cores承担dequantization（INT4→FP16）、scaling、element-wise运算等memory-bound任务，消耗instruction slots、register bandwidth、L1/L2 capacity，降低occupancy和tile sizes，剩余给compute-heavy matmul的资源极少。

  全栈执行例子（以QServe on A100, LLaMA-3.1-8B, GQA, 4-bit KV cache, 32K context, decoding step为例）：
  - 算法层：4-bit KV cache量化。K/V在prefill后量化为INT4，decoding时新token也即时量化。GQA下h_q=32, h_k=8，4个query heads共享1组K/V。
  - 系统框架层：QServe在FlashAttention kernel内fuse量化/dequantization操作。Page management管理KV cache内存分配。
  - 编译框架层：论文未明确说明。QServe使用CUDA实现，无编译框架自动代码生成。
  - kernel调度层：FlashAttention-style block-wise tiling (Tm×Tn tiles) + CUDA Cores FMA。Q tile(M=1) × K tile (N=128)→dequantization (INT4→FP16) + QK^T GEMV全在CUDA Cores→该warp沿N维串行处理每个tile，每tile都需dequant→频繁CUDA Cores stall。Nsight分析：dequantization overhead占近半数kernel时间（Fig 15a），CUDA Cores FMA consume 72.24% pipe utilization（Fig 15b），Tensor Cores utilization 0%。GQA下K/V tile需为32 query heads服务但CUDA Cores FMA compute-bound，GQA speedup仅1.4×（RTX4090，Fig 10）。
  - 硬件架构层：A100 GPU。Tensor Cores (312 TFLOPS FP16) 闲置，CUDA Cores (19.5 TFLOPS FP32) 成为瓶颈。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出BitDecoding，cooperative use of Tensor Cores + CUDA Cores for low-bit KV cache decoding。针对baseline三大缺陷：

  **缺陷1：Tensor Cores布局不匹配 → 已有系统无法在Tensor Cores上执行低比特数据**
  → **BitDecoding方案**：(a) Hardware instruction-induced layout：ldmatrix的thread-to-register mapping自动将低比特packed data隐式保留FP16 interleaved layout→Residual Kernel内每thread在register中做完计算→量化→pack→写global memory→Packing Kernel用相同ldmatrix/mma配置加载时layout天然正确。无需global reshape或手动layout transform。(b) Remapping for fast dequant：lop3-based 75316420 pattern→高效INT4/INT2→FP16转换（对比naive static_cast极慢）。(c) Residual block alignment：Nr=Pn×Wn×R确保packed tile精确填充Tensor Cores warp-tiling，消除fragment underfill。

  **缺陷2：Dequantization频繁stall Tensor Cores → 低GPU occupancy**
  → **BitDecoding方案**：(a) 新颖warp layout：Wm=1（decode Q len<16通常仅1 token），将资源重新分配给Wn→多个warp并发做dequantization→SM warp scheduler overlap dequantization与Tensor Cores mma，消除serialization bottleneck。(b) Asynchronous pipeline：register级软件pipeline——slice i做mma (Tensor Cores)同时slice i+1做ldmatrix+dequant (CUDA Cores)→producer-consumer持续流动。(c) Cooperative softmax：引入sTMP和sAcc shared memory buffer→跨warp reduction→sAcc通过ldmatrix重载P→保证后续PV mma layout正确。仅0.5% overhead换取correctness。

  **缺陷3：系统不支持多样的量化算法和attention变体**
  → **BitDecoding方案**：(a) Residual Kernel统一支持tensor-wise和channel-wise scaling：通过residual block内按seq_len维度做channel-wise、按hidden维度做tensor-wise reduction。Warp-level __shfl_xor_sync reduction + shared memory cross-warp aggregation计算scale/zero→half2 compact存储。(b) Query Transformation：Q reshape [1,(gq,hkv)]→[gq,hkv]，使MHA/MQA/GQA统一在Tensor Cores上高效执行，GQA的grouped queries形成大GEMM block。(c) 架构可移植性：Hopper用STSM+wgmma_SS；Blackwell直接用原生mxfp4 mma；layout-agnostic设计自动适配不同GPU世代的fragment layout。

  论文方法全栈执行例子（以LLaMA-3.1-8B, GQA (gq=4), 4-bit KC, 32K context, H100, decode step为例）：
  - 算法层：4-bit channel-wise KV cache + FP16 Q/score。Scale/zero compact为half2。Residual block size Nr = 8 × 4 × 4 = 128 (Pn=8 for mma.m16n8k16, Wn=4, R=4 for INT4→INT16)。
  - 系统框架层：BitDecoding提供Residual Kernel + Packing Kernel双kernel设计。Prefill: Residual Kernel量化→packed KV cache + residual FP16 cache。Decode: Packing Kernel做fused attention with low-bit data。
  - 编译框架层：论文未明确说明。CUDA手写PTX指令级实现，无编译框架。
  - kernel调度层（核心创新）：Query Transformation reshape Q [1, (4, 8)] → [4, 8]→4个query heads并行形成4×8 GEMM→full Tensor Cores occupancy。Packing Kernel: cp.async异步加载Q到shared mem (cp.async.cg)→加载Kpack (low-bit packed) 和Kp (scale/zero, half2)→ldmatrix加载packed data→lop3 75316420 remap→INT4→FP16 dequant (CUDA Cores, 4 warps并行)→异步overlap: slice i做QK^T mma (Tensor Cores wgmma_SS) 同时slice i+1做ldmatrix+dequant→cooperative softmax (sTMP跨warp reduction)→sAcc storer2s P→ldmatrix s2r重载→PV mma (wgmma_SS)→output。Hopper: STSM将dequantized K写入shared memory→wgmma_SS直接从shared memory读B矩阵。Tensor Cores utilization从baseline的0%提升至19.66%（Table III），dequantization overhead降至<15% (4-bit)，整体speedup 8.0× vs FP16 FlashDecoding-v2。
  - 硬件架构层：H100 (SM90, wgmma, TMA, STSM)。Blackwell B200 (SM100, native mxfp4 mma) 绕过所有dequant——Q (FP16) × K_packed (mxfp4) 直接硬件mma，speedup 8.6×。

  Baseline缺陷→BitDecoding方案映射：
  | Baseline缺陷 | BitDecoding方案 | 效果 |
  |-------------|----------------|------|
  | CUDA Cores-only FMA执行→Tensor Cores完全闲置 | Cooperative CUDA+Tensor Cores: ldmatrix自动induce layout + lop3快速dequant→Tensor Cores mma | TCs utilization 0%→19.66%, speedup up to 8.6× |
  | Dequant频繁stall→低GPU occupancy | 新颖warp layout (Wm=1, ↑Wn) + async pipeline: dequant与mma重叠 | dequant overhead <15% (4-bit), <35% (2-bit) |
  | 低比特layout不匹配Tensor Cores→无法正确执行 | Hardware instruction-induced layout + residual block alignment (Nr=Pn×Wn×R) | 零overhead layout transform, kernel correctness verified |
  | 不支持多样化量化算法和attention变体 | Residual Kernel统一channel-wise+tensor-wise scaling + Query Transformation支持MHA/MQA/GQA | GQA下speedup 3× vs QServe仅1.4× |
  | weight-optimized mpGEMM (Marlin/Ladder) 不适用于动态KV cache | Online fused quantization+packing in Residual Kernel, overhead 0.008ms/decode step (vs Marlin 0.41ms) | 2个量级quantization overhead降低 |
