## NVIDIA H800 GPU (Hopper Architecture for LLM Serving)

术语是什么？
NVIDIA H800是NVIDIA Hopper架构的GPU变体，专为中国市场设计，符合美国出口管制要求。与H100的关键区别在于NVLINK带宽和互连限制。LiquidGEMM论文使用H800 (80GB)作为实验平台。H800支持Hopper架构的全部计算特性：WGMMA指令（INT8 MMA peak ~990 TFLOPS）、TMA异步数据搬运、DPX指令、Transformer Engine（FP8支持）、mbarrier异步同步。H800的Tensor Core INT8吞吐与H100相同，CUDA Cores吞吐约60 TFLOPS，HBM带宽约3.35 TB/s (80GB HBM3)。16.5×的Tensor Core/CUDA Core吞吐比使W4A8 dequantization的CUDA Core瓶颈在H800上同样突出。

从硬件架构角度拆解术语：
H800 GEMM关键性能指标（LiquidGEMM Figure 1, H100数据，H800类似）：
- Tensor Core INT8: ~990 TFLOPS
- CUDA Core FP32: ~60 TFLOPS
- HBM Bandwidth: ~3.35 TB/s (80GB)
- Shared Memory per SM: 228KB (Hopper, 比A100的192KB大)
- L2 Cache: 50MB
- TMA: 支持1D-5D tensor传输, multicast across thread block cluster
- WGMMA: m64nNk32/m64nNk64 (INT8), m64nNk16 (FP16)

LiquidGEMM的H800 kernel设计约束：
- 需最小化CUDA Cores负载（每元素α≤5指令才能与memory/MMA重叠）
- 利用TMA + WGMMA + mbarrier实现异步pipeline
- 利用Hopper的更大shared memory (228KB vs A100 192KB)实现更大tile
- Dual-MMA packed layout利用LDS.128充分利用SMEM带宽

术语一般如何实现？如何使用？
H800是当前中国云服务提供商的主流LLM推理GPU（与H100/H20并列）。CUDA 12.4+, CUTLASS 3.x完整支持Hopper特性。H800的核心限制不在单卡计算能力（与H100相同），而在多卡互连（NVLink带宽受限）。对单卡推理场景（如LiquidGEMM的evaluation），H800 ≈ H100，所有Hopper优化技术直接适用。

涉及论文标题：
- LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving
