## Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是五套针对Monarch和BLAST矩阵乘法的自定义Triton kernel优化，解决多token推理场景下BLR方法因中间数据移动和内存访问模式差导致的memory-bound瓶颈：
  
  **Monarch优化①②③（联合使用，提供累加效率增益）：**
  - ① V矩阵重排布（Re-layout of V）：将V的存储格式从(b₁, r'b₂, p)改为r'先连续再b₂（原来b₂先连续再r'），消除推理时的r'↔b₂ permutation。此优化在离线阶段对静态权重执行一次。
  - ② 排列融合（Permutation fusion）：将b₂↔b₁ permutation与第一个bmm融合为单个Triton kernel。Kernel计算b₁×t_n×t_r个输出tiles，permutation通过计算b₂索引→调整r'偏移→用swapped indices写出来实现（Fig. 5 pseudo-code）。
  - ③ 避免最终permutation：当Monarch线性层输出立即与静态权重相乘时，离线pre-permute该权重的行，消除推理时的(b₂,n,q)→(n,q,b₂) kernel launch。
  
  **BLAST优化④和⑤（分别应用，代表不同策略）：**
  - ④ bmm部分融合（Partial fusion of bmm）：消除V和S之间的中间permutation和第一个bmm输出在global memory中的物化。每个thread block在内部循环b₁维度，加载S的(b₂,t_r) tile并广播与第一个bmm的(1,t_n,t_r)输出做累加batched outer product。牺牲tensor core利用率（第二个bmm跑CUDA cores），但避免了大中间张量(b₁,b₂,n,r)。
  - ⑤ 仅排列融合+Tensor Core优化（Permutation-only fusion with tensor core optimization）：转置S和U的第一和最后一维（S^T, U^T），从左侧乘，在每个kernel内transpose中间输出tiles。保持n连续，r/b₁/b₂依次作为三个kernel的batch维度，每个实现transposed bmm with outer-dimension reordering。消除permutation开销同时保持高tensor core利用率（via Triton dot()）。

  实验比较的baseline：BLAST repository (Lee et al. 2024)和Monarch repository (Dao et al. 2022)的PyTorch实现，均使用Triton auto-tuner和torch.compile()。评估方式：(1) layer-wise speedup——对每个模型的所有(B)LR替换层单独benchmark延迟；(2) end-to-end throughput——整个模型用torch.compile() + CUDA graph后测量prefill throughput（语言模型）、单步inference（扩散模型）、标准前向（视觉模型）。消融实验：BLAST ④ vs ⑤ 的tradeoff（④用CUDA cores做第二个bmm牺牲tensor core throughput，⑤用transpose保持tensor core但引入额外transpose开销）。

- 后端平台是什么，配置是什么。
  NVIDIA A40（40GB显存，6MB L2 cache，BF16 tensor core支持，HBM2e带宽约696 GB/s）。NVIDIA Jetson Orin Nano 8GB（边缘GPU，4-6MB L2 cache，DDR DRAM带宽约68 GB/s，2048 CUDA cores + 64 tensor cores）。软件：A40用Python 3.12.8、PyTorch 2.8.0、Triton 3.4.0、CUDA 12.6.3；Jetson用JetPack 6.2、L4T 36.4.3、CUDA 12.6.11、PyTorch 2.6.0、Triton 3.2.0。Triton autotuner sweep tile sizes（32-256 powers of two）、threads per block、pipelining stages。

- 评估性能的软件/脚本是什么。修改了什么。
  使用Triton do_bench() utility对每个layer单独benchmark：多次执行取平均延迟，warm-up消除cold-start（kernel compilation + cache population），torch.no_grad() + torch.cuda.synchronize()确保异步CUDA完成。端到端用torch.utils.benchmark.Timer()多次迭代forward pass。所有benchmark使用torch.compile() + CUDA graph capture。Baseline实现来自BLAST和Monarch开源repository，已包含Triton autotuner和torch.compile()优化。
  
  修改：(1) Monarch kernel——重写V存储layout + 融合permutation到bmm + 可选pre-permute下游权重；(2) BLAST kernel——两个变体（④partial fusion用CUDA cores batched outer product、⑤permutation-only fusion用tensor core transpose）；(3) 全部使用Triton编写，利用dot()算子做tensor core MMA，自定义tile sizes和内存布局。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源链接：https://github.com/pabillam/mem-efficient-blr

  评估原理：
  1. Layer-wise benchmark：对每个模型的每个线性层类型（QKVproj、Oproj、gate/upproj、downproj、c_attn、c_fc、c_proj、fc1、fc2等），用各方法（Dense/LR/Monarch/BLAST × baseline/optimized kernel）分别执行多token前向。Triton do_bench()自动多次warmup+计时取平均。输入为随机生成的tensor，shape由模型层配置（Table 3）和序列长度n决定（语言模型n=1024-2048, DiT n=16K, ViT n=197）。
  2. End-to-end benchmark：整个模型加载压缩权重后执行完整前向推理，torch.compile() wrapping整个网络以启用CUDA graph，torch.utils.benchmark.Timer()计数迭代取平均。语音模型报告prefill throughput（tokens/s），扩散模型报告单步latency，视觉模型报告标准forward latency。
  3. 消融：BLAST ④ vs ⑤分别对每种层和模型benchmark，分析tensor core vs CUDA core的tradeoff。

  全过程（以Llama-7B QKVproj层，n=1024, i=o=4096, r=1024, b=16在A40上，Monarch ②优化为例）：

  ```
  Host: 调用 Monarch 线性层 forward(X: [1024, 4096])
    # V ∈ R^{16 × 1024 × 256}（已重排布为 r'=64 先连续）
    # U ∈ R^{16 × 256 × 1024}

  Step 1: 输入reshape
    X_blocks = X.view(1024, 16, 256)   # [n, b₁, p]

  Step 2: Fused perm+bmm kernel (优化②, Fig. 5 pseudo-code)
    Triton Kernel Launch: grid=(b₁, ceil(n/t_n), ceil(r/t_r))
    # b₁=16 块, t_n=64, t_r=128 (通过autotuner选择)

    For each thread block (b_1, n_tile, r_tile):
      # 计算permutation target indices
      b_2 = (r_tile * t_r + [0:t_r-1]) // r'   # ★ target b_2 index
      r'_offset = (r_tile * t_r + [0:t_r-1]) % r' + b_1 * r'  # ★ adjusted r' offset

      acc = zeros(t_n, t_r)   # accumulator tile in registers
      for p_tile in range(0, p, t_p):   # p=256, t_p=64
        x = X_blocks[b_1, n_tile*t_n:(n_tile+1)*t_n, p_tile*t_p:(p_tile+1)*t_p]
        v = V[b_1, p_tile*t_p:(p_tile+1)*t_p, r_tile*t_r:(r_tile+1)*t_r]
        acc += triton.dot(x, v)   # Tensor Core MMA, t_n×t_p @ t_p×t_r → t_n×t_r

      # 写入输出，使用swapped indices完成permutation ★
      Z_out[n_tile*t_n:(n_tile+1)*t_n, b_2 * n * r' + r'_offset] = acc

  Step 3: 第二批bmm（U）
    # Z_out shape after kernel: [n, b₂·b₁·r'] effectively [n, b₂, b₁·r']
    Triton Kernel Launch for batch matmul: Z_out × U
    # 产生Y: (b₂, n, q) → 若输出连residual/add，则需最终permutation

  Step 4 (可选, 优化③): 若Y随后与静态W_down相乘
    # 离线已pre-permute W_down行 → 跳过在线permutation kernel

  性能输出（A40上）：
    - Monarch ② vs baseline Monarch: Qproj 1.46× speedup, Kproj 1.58×, Vproj 1.62×, Oproj 1.37×
    - Monarch ①+②+③ vs baseline: 综合 1.46-2.37× speedup across layers
    - BLAST ⑤ vs baseline BLAST: DiT-XL/2 QKVproj up to 7.15× on Jetson
    - BLAST ⑤ vs dense: up to 3.76× (GPT2-S c_fc layer on Jetson)
    - End-to-end BLAST ⑤: 1.13-1.48× over dense across all models
    - BLAST ④ < BLAST ⑤ consistently（因为④的CUDA core batched outer product < tensor core throughput 16×）
  ```
