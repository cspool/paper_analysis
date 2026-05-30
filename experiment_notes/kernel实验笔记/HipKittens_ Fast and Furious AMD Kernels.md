## HipKittens: Fast and Furious AMD Kernels

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是基于ThunderKittens的C++ embedded tile-based编程原语，移植并重新设计AMD GPU上的高性能AI kernel框架HipKittens (HK)。核心实现包括：(1) 8-WAVE PING-PONG和4-WAVE INTERLEAVE两种wave调度模式，替代NVIDIA的wave specialization模式；(2) 开发者可控的寄存器分配（pinned register tiles），绕过HIPCC编译器对AGPR寄存器使用的限制；(3) 针对AMD CDNA异构MFMA指令形状的共享内存swizzle优化，解决bank conflict；(4) chiplet感知的L2/LLC两级缓存grid调度算法（Algorithm 1: XCD swizzle for cache reuse）。实验比较的baseline包括：AMD AITER（手写汇编）、Composable Kernel (CK)、PyTorch SDPA/torch.compile、HipBLASLT、ROCm Triton、Mojo。评估的workload包括：BF16 GEMM、FP8 GEMM、FP6 GEMM（初步）、MHA/GQA Attention forward/backward（causal/non-causal, d=64/128）、fused dropout-residual-layernorm、RoPE。

- 后端平台是什么，配置是什么。
  AMD CDNA4 MI355X OAM GPU（8 XCD chiplet，256 CU，BF16 2.5 PFLOPs，288GB HBM，8.0 TB/s带宽）。AMD CDNA3 MI325X GPU。AMD MI350X GPU。对比平台：NVIDIA B200 SXM5（2.2 PFLOPs BF16，180GB，8.0 TB/s）。软件环境：ROCm 7.0 Docker (rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta)。

- 评估性能的软件/脚本是什么。修改了什么。
  自研HK C++ kernel通过Python bindings在Python脚本中benchmark。每个kernel 500次warmup + 100次测量取平均TFLOPs/s，输入为N(0,1)随机张量。AITER通过aiter.flash_attn_func调用，PyTorch通过torch.nn.functional.scaled_dot_product_attention，CK通过编译tile_example_gemm_basic/tile_example_fmha_fwd/tile_example_fmha_bwd二进制运行，HipBLASLT通过hipblaslt-bench命令行。修改：基于ThunderKittens框架，重写所有tile原语以包装AMD CDNA assembly/HIP（替代NVIDIA PTX/CUDA），新增pinned register tile接口、8-wave/4-wave调度模板、chiplet swizzle算法。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源链接：https://github.com/HazyResearch/HipKittens

  评估原理：
  1. HK以C++ header-only库形式提供，开发者使用tile原语（rt_bf/rt_fl/st_bf等）编写kernel，通过Python pybind11绑定调用。
  2. BF16 GEMM kernel（Fig. 21）以256×256输出tile/thread block、16×16×32 MFMA指令为基本单元，采用8-wave ping-pong调度：每SIMD 2个wave交替执行compute cluster（MFMA矩阵乘）和memory cluster（buffer_load_dword从HBM到LDS，ds_read从LDS到register）。
  3. Attention forward kernel（Fig. 23）以32×128 tile/wave为输出单元，同样采用8-wave ping-pong，在compute cluster内交替执行online-softmax vector ops（max/subtract/exp2/accumulate）和MFMA指令，通过sched_barrier hints指导LLVM编译器精确调度vector和matrix指令的交错。
  4. Cache优化：Algorithm 1在kernel启动前remap block indices，将连续C个block分配给同一XCD（L2复用），以W高度的垂直窗口遍历输出矩阵（LLC复用）。

  全过程（以BF16 GEMM为例）：
  ```
  用户调用HK GEMM kernel(D=AB+C, M=N=K=8192, dtype=BF16)
    → Algorithm 1: 根据M/N tile数、XCD数(8)、W和C参数，计算remap后的block坐标(row, col)
    → 每个thread block负责256×256输出子矩阵
    → Prologue: 8 waves协作preload A/B tile从HBM到shared memory (buffer_load_dword)
    → Conditional barrier: 4个leader wave继续preload，4个follower wave等待
    → Hotloop: leader和follower交替执行
        Cluster 0: load B_tile_0从shared到register (ds_read_b128) → load A_tile → G::load next As → s_barrier
        Cluster 1: __builtin_amdgcn_s_setprio(1) → mma_ABt(C[0][0], A, B_tile_0) → s_setprio(0) → s_barrier
        Cluster 2: load B_tile_1 → G::load next Bs → s_barrier
        Cluster 3: mma_ABt(C[0][1], A, B_tile_1) → s_barrier
        Cluster 4-7: 对称处理C[1][0]/C[1][1]，完成compute和memory的overlap
    → Epilogue: store C_accum tiles回HBM
    → 测量：rocm-profiler或wall-clock计时，500 warmup + 100 iters → 报告TFLOPs/s
  ```
