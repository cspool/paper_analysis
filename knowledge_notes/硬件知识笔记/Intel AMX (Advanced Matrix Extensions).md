## Intel AMX (Advanced Matrix Extensions)

术语是什么？

Intel AMX (Advanced Matrix Extensions) 是Intel在2022年Sapphire Rapids (SPR)起引入的x86 ISA扩展，为CPU每物理核集成专用矩阵乘法加速器单元。AMX核心由两部分组成：(1) **TILECFG寄存器**：每核8个二维寄存器，每个1KB，可配置为不同tile尺寸；(2) **TMUL (Tile Matrix Multiply Unit)**：硬件矩阵乘法加速器，每个周期执行1024次BF16运算（1024 BF16 ops/cycle），计算 C[M][N] += A[M][K] × B[K][N]，其中M≤16, N≤64。从SPR (BF16) → GNR (FP16) → DMR (FP8) 代际提升精度和算力（AVX-512 25.6→32 TFLOPS, AMX 206.4→344 TFLOPS）。

从硬件架构角度拆解术语：

AMX在CPU流水线中的运作方式：
1. **指令流共享**：AMX与ALU、FPU、AVX等传统功能单元共享同一物理核的fetch/decode/rename/scheduler前端，不独立于CPU流水线
2. **TILECFG配置流程**：软件通过`TILECFG`指令配置8个tile寄存器的行列尺寸→数据通过`TILELOADD`从内存/缓存加载到tile寄存器→`TDPBF16PS`（BF16 dot-product）指令触发TMUL计算→`TILESTORED`写回
3. **SMT限制**：AMX资源（tile寄存器和TMUL）不跨hyperthread共享——SMT sibling thread竞争同一物理核的AMX，限制了并发AU使用
4. **功耗影响**：AMX启用后的高功耗触发core频率下降（prefill→2.5 GHz, decode→3.1 GHz, base 3.2 GHz），由TDP限制导致，频率降幅与AMX使用率正相关

术语一般如何实现？如何使用？

AMX编程通过Intel oneDNN或直接使用AMX intrinsics：
- **oneDNN方式**：xft/xFasterTransformer调用oneDNN GEMM API→oneDNN内部根据矩阵维度判定是否使用AMX
- **编译器支持**：GCC/LLVM支持AMX intrinsics (`_tile_loadd`, `_tile_dpbf16ps`, `_tile_stored`)，需`-march=sapphirerapids`编译标志
- **性能**：prefill GEMM (8192×4096×22016, batch=16) 达40.57 TFLOPS；小矩阵GEMV (16×4096×22016) 仅3.87 TFLOPS（AMX tile register配置overhead > 计算收益，此时AVX更高效）
- AMX单核算力：SPR 206.4 TFLOPS (2.7 GHz base), GNR 344 TFLOPS (2.8 GHz base)
- **LLM推理实测** (SLINFER, 第4代Xeon 6462C @3.3GHz, OpenVINO backend, BF16)：Llama-2-7B 1K-input TTFT=567ms, TPOT(1-batch 1K-length)=71ms, 32-batch 1K TPOT=196ms, 32-batch 4K TPOT=459ms；Llama-2-13B 1K-input TTFT=650ms, 1-batch TPOT=83ms。4th Gen (AMX) vs 3rd Gen (无AMX): TTFT speedup 6.7-7.3×, TPOT speedup 1.4-1.7×。可独立serve ≤13B LLM在moderate SLO下（TTFT ≤4s, TPOT ≤250ms），但tight SLO (TPOT 100ms)下仅7B可行

涉及论文标题：
- AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving
- Towards Resource-Efficient Serverless LLM Inference with SLINFER

