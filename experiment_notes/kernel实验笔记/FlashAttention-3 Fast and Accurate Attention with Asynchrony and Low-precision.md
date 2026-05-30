## FlashAttention-3 Fast and Accurate Attention with Asynchrony and Low-precision

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现基于Hopper H100 GPU的warp-specialized异步attention kernel，使用CUTLASS primitives（WGMMA、TMA、setmaxnreg）构建。核心kernel实现包括：(i) **Producer-consumer warp specialization**：producer warpgroup使用TMA从HBM异步加载Q/K/V tiles到circular SMEM buffer，consumer warpgroup使用WGMMA执行QK^T和PV矩阵乘法；(ii) **setmaxnreg动态寄存器分配**：producer warp释放registers（仅需1 thread for TMA），consumer warp获取更多registers用于WGMMA；(iii) **2-stage GEMM-softmax pipelining**：在consumer warpgroup内通过寄存器缓冲$\mathbf{S}_{\text{next}}$实现跨迭代流水线——WGMMA(QK^T) of iter j+1 与 softmax of iter j 重叠，WGMMA(PV) of iter j 与 softmax of iter j+1 重叠；(iv) **FP8 WGMMA支持**：Q/K以k-major布局存储（contiguous in head dim），V需in-kernel transpose（LDSM + byte_perm + STSM）转为m-major布局，FP32 accumulator通过byte_perm + shfl_sync转换为FP8 operand register layout以conform to FP8 WGMMA要求；(v) **Persistent kernel**：launch与SM数量相等的threadblocks（132 on H100），每个threadblock处理多个tiles，重叠epilogue和prologue以减少tensor core空闲。
  实验比较：(i) Forward speed (TFLOPs/s) vs FlashAttention-2 (CUDA)、FlashAttention-2 in Triton、cuDNN attention、standard PyTorch attention；(ii) Backward speed vs FlashAttention-2、FlashAttention-2 in Triton；(iii) FP8 forward speed vs BF16 baselines；(iv) 消融：warp-specialization vs GEMM-softmax pipelining各自贡献（batch=4, seqlen=8448, nheads=16, hdim=128 forward）；(v) 2-stage vs 3-stage pipelining效果（3-stage理论上更多重叠但register pressure导致更小block size）。

- 后端平台是什么，配置是什么。
  - NVIDIA H100 80GB SXM5 GPU (Hopper architecture, 700W)：132 SMs，80 GiB HBM @ 3.35 TB/s，228 KiB SMEM per SM，boost clock 1830 MHz
  - FP16/BF16 Tensor Core理论峰值：989 TFLOPs/s；FP8 Tensor Core理论峰值：~1978 TFLOPs/s (2× FP16)
  - Special functions throughput：3.9 TFLOPs/s（16 ops/SM/clock × 132 SMs × 1830 MHz）
  - 软件栈：CUDA 12.3, CUTLASS 3.6 (WGMMA and TMA abstractions), cuDNN 9.5.0.50, Triton 3.1, PyTorch 2.5.0

- 评估性能的软件/脚本是什么。修改了什么。
  - 自编CUDA C++ kernel，基于CUTLASS primitives实现：(i) WGMMA for asynchronous tensor core matrix multiplication（SS prefix: first operand from SMEM; RS prefix: first operand from register file）；(ii) TMA for asynchronous HBM↔SMEM data movement；(iii) setmaxnreg for dynamic register (de)allocation between warpgroup roles；(iv) bar.sync for inter-warpgroup synchronization and barrier-based pipeline management；(v) circular SMEM buffer (s-stage) with producer-consumer commit/wait protocol。
  - 修改：(i) 替换FlashAttention-2的同步模型——原FlashAttention-2内循环中BMM1→wait→softmax→BMM2→wait为全同步，FlashAttention-3改为2-stage流水线：BMM1(iter j+1, commit no wait) → softmax(iter j, overlapping with BMM1) → BMM2(iter j, commit no wait) → wait both；(ii) Warp-specialization替代原统一warp模型——producer warp仅执行TMA load + commit，consumer warpgroup仅执行WGMMA + softmax；(iii) FP8 precision——添加FP32→FP8 operand layout转换（byte_perm + shfl_sync），in-kernel V transpose（LDSM→byte_perm→STSM），k-major/m-major layout constraints处理；(iv) Inference优化——split-KV（Flash-Decoding）+ GQA packing + PagedAttention with TMA block table。
  - 评估脚本：CUDA event timing测量kernel wall-clock time，重复10次取平均，固定GPU clock 1830MHz。FLOPs计算：forward = 4 × seqlen² × head_dim × nheads（causal时/2），backward = forward × 2.5（2 forward matmuls + 5 backward matmuls via recomputation）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  已开源：https://github.com/Dao-AILab/flash-attention（BSD license），集成到PyTorch。
  
  评估原理与流程（以单卡H100 SXM5 BF16 forward benchmark, seqlen=8192, head_dim=128, nheads=16为例）：
  1. **Input准备**：Q/K/V tensors in BF16/FP16 `[batch=2, seqlen=8192, nheads=16, head_dim=128]`（total tokens=16K），位于H100 HBM。
  2. **Kernel launch configuration**：Grid = 132 threadblocks（persistent kernel，对应132 SMs）。每个threadblock处理一个Q tile ($B_r$)。CTA内3个warpgroups：(a) producer warpgroup（1 warp, register-deallocated）执行TMA loads；(b) consumer warpgroup 1（2 warps, register-reallocated）；(c) consumer warpgroup 2（2 warps）用于pingpong scheduling。
  3. **Kernel执行（per CTA, single Q tile $\mathbf{Q}_i \in \mathbb{R}^{B_r \times 128}$）**：
     a. Producer warpgroup TMA load $\mathbf{Q}_i$ → SMEM → commit。进入主循环：for j=0..T_c-1，wait for stage consumed → TMA load $\mathbf{K}_j$, $\mathbf{V}_j$ → SMEM at stage j%s → commit。
     b. Consumer warpgroup 1 执行 Algorithm 2 主循环（2-stage GEMM-softmax pipelining）：
        - SS-WGMMA: $\mathbf{S}_{ij} = \alpha \mathbf{Q}_i \mathbf{K}_j^T$ (64×128×128, BF16 accum in FP32, ~128K MACs per tile)
        - softmax: FMNMX.FTZ (rowmax) → SHFL.BFLY (warp-level reduction) → MUFU.EX2 (exponential) → FADD + FMUL (rowsum + rescale)
        - RS-WGMMA: $\mathbf{O}_i \mathrel{+}= \tilde{\mathbf{P}}_{ij} \mathbf{V}_j$ (64×128 tensor core MMA)
        - 关键：WGMMA(QK^T) of iter j+1 异步发射后，softmax of iter j 在另一 warpgroup 的 tensor core 空闲间隙被调度执行
     c. Epilogue：$\mathbf{O}_i = \operatorname{diag}(\ell_i)^{-1} \mathbf{O}_i$ → TMA store $\mathbf{O}_i$, $L_i$ → HBM。
  4. **Performance measurement**：CUDA event记录kernel launch→completion时间。FLOPs = 4 × 8192² × 128 × 16 = ~550B FLOPs (forward)。TFLOPs/s = FLOPs / runtime。达到840 TFLOPs/s即85% H100 FP16理论峰值利用。
  5. **FP8 variant额外步骤**：(a) Q/K量化为FP8 e4m3 with per-block scaling；(b) in-kernel V transpose via LDSM + byte_perm + STSM转置in SMEM；(c) FP32 accumulator → FP8 operand register exchange via byte_perm + shfl_sync；(d) FP8 WGMMA: 2× throughput of BF16 WGMMA，达到1.3 PFLOPs/s。
