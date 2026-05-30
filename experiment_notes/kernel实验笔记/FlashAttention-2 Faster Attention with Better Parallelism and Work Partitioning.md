## FlashAttention-2 Faster Attention with Better Parallelism and Work Partitioning

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  基于FlashAttention v1，重写CUDA kernel实现三方面改进：(i) **算法tweak**——前向不再对output两项都做`diag(ℓ)^{-1}` rescale，改维护"un-scaled" output并在最终一次性rescale；反向只存储logsumexp L = m + log(ℓ)而非同时存m和ℓ，减少non-matmul FLOPs；(ii) **并行化**——除batch和head维度外，增加sequence length维度的并行。前向：外循环（over K/V blocks）embarrassingly parallel，不同thread block处理不同row block，无需通信；反向：不同thread block处理不同column block，用atomic add更新dQ；(iii) **Warp间工作划分**——前向改为split Q across warps（保持K/V所有warp可访问），避免FlashAttention v1的"split-K"方案（split K/V across warps）带来的shared memory通信开销；反向同样避免split-K。实验比较：(a) Forward+backward runtime vs FlashAttention v1、FlashAttention Triton、xformers cutlass实现、PyTorch标准attention，seq length 512-16K，head dim 64/128，causal/non-causal，A100 80GB SXM4；(b) Decoding阶段attention kernel vs PyTorch naive、FasterTransformer，MQA setting，batch size 1-1024；(c) End-to-end GPT训练吞吐（1.3B/2.7B参数，2k/8k context，8×A100）vs 无FlashAttention baseline和FlashAttention v1。

- 后端平台是什么，配置是什么。
  - NVIDIA A100 80GB SXM4 GPU：108 SMs，每SM 192KB on-chip SRAM，HBM带宽1.5-2.0TB/s，SRAM带宽约19TB/s，FP16/BF16 matmul峰值312 TFLOPs/s，non-matmul FP32峰值19.5 TFLOPs/s
  - NVIDIA H100 GPU（仅forward+backward runtime benchmark，未使用TMA和4th-gen Tensor Cores特殊指令）
  - 端到端训练：8×A100 80GB SXM4
  - Benchmark：seq length 512/1k/2k/4k/8k/16k，batch size使总tokens=16k，hidden dim=2048，head dim=64（32 heads）或128（16 heads），FP16/BF16

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于CUTLASS 3.x库手写CUDA kernel实现FlashAttention-2 forward和backward pass
  - 修改：(i) Forward kernel——外循环并行化：对每个row block（Q block i），启动一个thread block独立处理所有KV blocks；thread block内部warp split Q而非K/V，每个warp持有Q的一个slice，K/V由所有warp共享，每个warp计算其Q slice对应的局部S=QK^T，直接乘V得到output slice，无需warp间通信。最终一次性rescale output by `diag(ℓ)^{-1}`并存储L=m+log(ℓ)。causal mask优化：跳过row indices < column indices的blocks（约一半blocks），每行仅需对1个block施加causal mask。(ii) Backward kernel——列block并行化：每个column block j由一个thread block处理，遍历所有row blocks i，在SRAM中重计算S_ij和P_ij，累加dK_j和dV_j，用atomicAdd更新dQ_i。同样避免split-K warp划分。(iii) Decoding kernel——将KV cache加载split到不同thread blocks以saturate HBM bandwidth，写中间结果到HBM后再用separate reduce kernel合并。
  - Benchmark方法：CUDA event timing测kernel wall-clock time。FLOPs计算：forward = 4·seqlen²·head_dim·num_heads（causal mask时÷2），backward = forward FLOPs × 2.5。TFLOPs/s = FLOPs / runtime。
  - 端到端训练FLOPs公式（Megatron-LM）：6·seqlen·num_params + 12·num_layers·hidden_dim·seqlen²

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  已开源：https://github.com/Dao-AILab/flash-attention（BSD许可证）。安装：`pip install flash-attn`。Python接口兼容FlashAttention v1：`from flash_attn import flash_attn_func; output = flash_attn_func(q, k, v, causal=True)`，内部自动使用FlashAttention-2 kernel。底层CUTLASS 3.x实现，build system基于PyTorch extension。

  **Kernel输入到性能输出全过程**（以A100 forward pass, N=8192, d=128, 16 heads, batch=2, causal=False为例）：

  1. **Tensor分配**：Q/K/V tensors `[2, 8192, 16, 128]` FP16/BF16 in HBM。Block sizes: `B_r=128, B_c=128`（tuned for head_dim=128）。划分：`T_r = ceil(8192/128) = 64` row blocks，`T_c = 64` column blocks。Output O `[2, 8192, 16, 128]`，logsumexp L `[2, 8192, 16]`。

  2. **Thread block调度**（forward，parallelism over sequence length）：对每个(head, row_block_i)组合，launch 1个thread block。共batch×heads×T_r = 2×16×64 = 2048个thread blocks。每个thread block独立处理其row block的所有64个KV column blocks，无需与其他thread block通信。2048 >> 108 SMs，occupancy接近满载。

  3. **Thread block内部forward执行**（单个row block i, B_r=128 rows）：
     - 从HBM加载Q_i `[128, 128]` 到SRAM（~32KB FP16）。初始化 `O_i = 0 [128, 128]`, `ℓ_i = 0 [128]`, `m_i = -inf [128]`。
     - **外循环** j=1..64（KV column blocks）：
       a. 从HBM加载K_j `[128, 128]`、V_j `[128, 128]` 到SRAM（~32KB each, 64KB total）。
       b. **Warp内计算**（4 warps per thread block, split Q row-wise, 各warp持有Q_i的32 rows）：
          - Each warp: `S_warp = Q_warp × K_j^T` (32×128, Tensor Core MMA, FP16→FP32 accumulate)
          - `m_new = max(m_old, rowmax(S_warp))` (CUDA core reduction, 128 elements per row × 32 rows)
          - `P_warp = exp(S_warp - m_new)` (MUFU.EX2 instruction)
          - `ℓ_new = exp(m_old - m_new)·ℓ_old + rowsum(P_warp)` (CUDA core)
          - `O_warp = diag(exp(m_old - m_new))·O_old + P_warp × V_j` (Tensor Core MMA, 32×128 × 128×128)
          - **无需warp间通信**：K_j和V_j由所有warp共享，每个warp独立计算其output slice。
       c. SRAM中仅保留最新O_i、ℓ_i、m_i。中间S_warp (32×128×2B=8KB/warp) 和P_warp存于register/SRAM，不写HBM。
     - **循环结束**：`O_i = diag(ℓ_final)^{-1}·O_final` (一次性rescale)，`L_i = m_final + log(ℓ_final)`。
     - 写O_i和L_i到HBM对应位置。

  4. **Backward kernel**（column-parallel）：对每个(head, col_block_j) launch 1个thread block（共batch×heads×T_c=2048 blocks）。Load K_j, V_j to SRAM。Initialize dK_j=0, dV_j=0。内循环i遍历T_r row blocks：load Q_i, O_i, dO_i, L_i, D_i → recompute S_ij, P_ij in SRAM → compute dV_j += P_ij^T·dO_i, dS_ij → dK_j += dS_ij^T·Q_i → atomicAdd dQ_i += dS_ij·K_j to HBM。Write dK_j, dV_j to HBM。

  5. **性能评估**：CUDA event timing测量kernel wall-clock time（ns）。TFLOPs/s = (4×8192²×128×16) / 1e12 / runtime_seconds ≈ 5.5e11 FLOPs / runtime。A100 FP16 matmul peak: 312 TFLOPs/s。FlashAttention-2 forward实测~210 TFLOPs/s（~67% peak），FlashAttention v1实测~105 TFLOPs/s（~34% peak），standard PyTorch attention ~21 TFLOPs/s（~7% peak）。
