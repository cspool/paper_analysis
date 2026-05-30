## LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是LiquidGEMM——一个硬件高效的W4A8 GEMM kernel，包含两项关键技术：(1) LiquidQuant (LQQ)——仅需两条32-bit硬件指令（IMAD + XOR）处理四个元素的overflow-safe dequantization算法；(2) Implicit Fine-Grained Pipeline (ImFP)——single-producer multiple-consumer执行模型，Load WG通过TMA加载weight到SMEM后切分为fine-grained tasks，多个Compute WG竞争获取task并各自完成dequantization+CUDA Core MMA，跨Compute WG实现dequantization与MMA的自然重叠，消除SMEM↔RF round-trip数据搬运和软件同步开销。还包含Dual-MMA packed layout——将两个连续MMA操作所需元素打包存储，每个线程用单条LDS.128指令加载32个UINT4元素。实现使用CUTLASS和Cute编程原语，WGMMA/barrier/TMA等用PTX包装，dequantization逻辑直接用CUDA实现。计算Y=(WX^T)^T替代Y=XW^T以利用WGMMA的m=64固定维度。

  实验比较的baseline kernels：QServe（W4A8，QoQ dequantization算法）、TRT-W4A16、TRT-W8A8、TRT-FP8、TRT-FP16。评估方式：(1) 系统级——LiquidServe vs QServe/TRT吞吐量和延迟；(2) kernel级——使用统一CUDA benchmark框架从各系统抽取GEMM kernel，隔离对比单层transformer所有GEMM（fused QKV projection、output projection、两个FFN GEMM）延迟，batch size 4-256。消融实验：逐步启用LQQ、ExCP（显式粗粒度pipeline）、ImFP（隐式细粒度pipeline），对比各组件贡献。

- 后端平台是什么，配置是什么。
  NVIDIA H800 GPU（80GB HBM, Hopper架构）。WGMMA指令支持INT8 MMA（m64nNk32/m64nNk64, N∈[8,256]）。TMA用于异步数据搬运。软件：PyTorch 2.4.0，CUDA 12.4，CUTLASS/Cute。

- 评估性能的软件/脚本是什么。修改了什么。
  使用内部统一CUDA benchmark框架（"An internal benchmarking tool used to evaluate GPU kernel performance before deployment"）对各系统抽取的GEMM kernel进行公平对比，支持灵活配置矩阵形状以模拟各种模型场景。每次测量5次取平均。修改：(1) 自研LiquidGEMM kernel——基于CUTLASS/Cute编程原语构建warp-specialized ping-pong kernel，fuse dequantization到MMA mainloop，实现Dual-MMA packed layout数据加载，ImFP pipeline替代ExCP；(2) kernel计算改写——从Y=XW^T改写为Y=(WX^T)^T以更好地利用WGMMA指令。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源情况：LiquidGEMM未提供开源代码。论文明确说明"LiquidGEMM is currently deployed as the primary GEMM kernel in our production LLM serving infrastructure"。评估kernel性能使用内部benchmark工具，外部不可获取。

  评估原理：
  1. 统一benchmark框架从各系统（LiquidServe, QServe, TRT）中抽取纯GEMM kernel，去除系统级因素（attention、KV cache管理等）干扰。
  2. 对每个模型（LLaMA2-7B/13B/70B, LLaMA3-8B, Mistral-7B, Mixtral-8×7B）的单层transformer所有GEMM（QKV fusion projection、output projection、两个FFN GEMM）分别测量延迟。
  3. Batch size从4到256遍历，每次5次运行取平均。
  4. 消融实验：从baseline（无LQQ/无pipeline）开始，逐步启用LQQ → ExCP/ImFP，测量每步加速比。

  全过程（以LiquidGEMM处理FFN层W4A8 GEMM为例，M=batch_size, N=hidden_dim×intermediate_factor, K=hidden_dim）：
  ```
  Host: 启动LiquidGEMM kernel(grid=(m×n thread blocks), block=(384 threads=3 WGs))
  
  Per Thread Block (处理Mt×Nt输出tile, 在K维度迭代):
    // ImFP: 1 Load WG + 2 Compute WGs, 共3 WGs
    
    Load WG (4 warps, TMA + CUDA Cores):
      for k_iter in 0..K/Kt:
        // 异步weight加载
        cp.async.bulk (TMA): GMEM[weight_tile_u4] → SMEM[buffer_ping]  // Dual-MMA packed layout
        cp.async.bulk.commit_group
        cp.async.bulk.wait_group
        
        // 将weight tile切分为fine-grained tasks写入SMEM task queue
        // 每个task = 一个WGMMA fragment所需weight（64×32 UINT4 elements）
        smem_task_queue.push(task_metadata)
        // 切换到pong buffer

    Compute WG_0 (4 warps, CUDA Cores + Tensor Cores):
      for task = smem_task_queue.pop():  // hardware-managed scheduling, no software sync
        // Step 1: 从SMEM加载weight到RF
        LDS.128: RF[0:31] = SMEM[task.weight_addr]  // 32 UINT4 elements, 1 instruction
        
        // Step 2: Unpack 4-bit → 8-bit (QServe method)
        // 8 × 4-bit elements in reg → 2 × 32-bit regs with 8-bit elements
        unpack_lo(reg_w0, w_packed)
        unpack_hi(reg_w1, w_packed)
        // (repeat for all 4 packed regs → 8 regs of 8-bit)
        
        // Step 3: Dequantization with LQQ (CUDA Cores)
        // Equation 12: Q_i8 = (Q_u4 * s_u8 + a) XOR 0x80
        // 2 instructions per 4 elements:
        r0 = IMAD(r0, s_broadcast, a_broadcast)  // multiply-add
        r0 = XOR(r0, 0x80808080)                 // flip MSB of each byte
        // (total: 7 instructions for 8 elements incl. unpack)
        
        // Step 4: MMA (Tensor Cores)
        warpgroup.mma.fence  // ensure dequantization results visible
        WGMMA.m64nNk32: C_frag += A_frag(INT8) × W_frag(INT8)
        // 使用dequantized weight作为INT8 MMA输入
        
    Compute WG_1: 同时竞争获取不同task，dequantization与MMA自然与WG_0重叠
      // WG_0做dequantization时, WG_1可能在做MMA, 反之亦然
      // 无需软件同步——由硬件task scheduling管理

    // Epilogue: 第一级dequantization (INT8→FP16) + 写回GMEM
    C_fp16 = C_int32 * s_i8 (per-channel scale)
    store GMEM[output_tile] = C_fp16
  ```

  性能输出（以LLaMA2-7B FFN GEMM为例，batch=256）：
  - LiquidGEMM: 2.90x speedup vs QServe W4A8 kernel
  - LQQ alone (memory-bound): limited benefit; LQQ alone (compute-bound): up to 1.29x speedup
  - ImFP vs ExCP: ImFP consistently better across all batch sizes;
    ExCP degrades at small batch due to round-trip traffic + sync overhead
  - LiquidGEMM vs TRT-FP8 on LLaMA2-7B: 1.12-1.58x speedup
  - LiquidGEMM vs TRT-W4A16 on Mixtral-8×7B: 1.12-2.53x speedup
