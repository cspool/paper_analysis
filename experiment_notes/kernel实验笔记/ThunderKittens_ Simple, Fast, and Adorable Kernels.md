## ThunderKittens: Simple, Fast, and Adorable Kernels

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是ThunderKittens (TK)，一个C++ embedded AI kernel编程框架，通过三层GPU抽象简化高性能kernel开发：(1) warp级——16×16矩阵tile作为基本数据结构，提供PyTorch风格的操作（mma, exp, cumsum等），自动选择最小化bank conflict的shared memory布局（32/64/128字节swizzle）；(2) block级——LCSF (Load-Compute-Store-Finish) 异步编程模板，基于生产者-消费者范式协调load/store worker与compute worker的异步overlap执行，支持多级pipeline buffer隐藏HBM延迟；(3) grid级——persistent grid减少block launch/setup开销，block launch order调度提升L2 cache复用率。实验比较：(a) GEMM vs CuBLAS、CUTLASS；(b) Attention forward/backward（causal/non-causal, d=64/128）vs FlashAttention-3；(c) Linear attention（polynomial-based特征图和learned特征图）vs Flash Linear Attention (FLA, Triton)；(d) State space models long convolution (FFT-based) vs FlashFFTConv；(e) Mamba-2 vs Triton kernels from Dao & Gu 2024；(f) Rotary positional encoding、fused dropout-residual-layernorm vs popular Triton kernels。

- 后端平台是什么，配置是什么。
  NVIDIA H100 80GB SXM5 GPU（Hopper架构，132 SM，tensor cores支持wgmma指令，TMA异步数据搬运）。扩展测试：NVIDIA RTX 4090（consumer GPU），Apple M2 Pro（personal hardware，Metal API）。CUDA 12.6，Triton 3.00，PyTorch 2.4。

- 评估性能的软件/脚本是什么。修改了什么。
  使用NVIDIA Nsight Compute (NCU) 进行kernel profiling，分析tensor core利用率、issue slot利用率、HBM bandwidth/stalls、shared memory stalls。性能测量：10次warmup + 10次benchmark iteration取平均，通过cudaEvents计时。TK本身不修改现有软件，而是提供一个全新的C++ embedded框架替代CUTLASS/CuTe进行kernel开发。baseline GEMM通过CuBLASLt的auto-tuning获取最优性能，Triton kernel通过triton.autotune调优。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/HazyResearch/ThunderKittens。TK以header-only C++ library形式发布（include/目录<1MB），用户编写kernel时include TK头文件，使用kittens命名空间。评估原理：NCU通过硬件性能计数器采集SM内部的tensor core pipelines利用率、issue slot active cycles、HBM读/写bytes及stall cycles、shared memory bank conflict计数。以attention kernel为例：(1) 用户用TK的tile类型(rt_bf, st_bf)声明register/shared memory tiles和global layout descriptors (gl<bf16, -1, -1, -1, D>)描述HBM tensor；(2) 在LCSF模板中编写load函数（TMA异步load K、V tiles到shared memory pipeline buffer）、compute函数（warpgroup::mm_ABt计算Q@K^T → softmax via sub_row/exp/row_sum/div_row → copy转为bf16 → warpgroup::mma_AB计算att@V）、store函数（TMA异步写回output）、finish函数；(3) TK自动选择shared memory swizzle布局消除bank conflict，自动生成TMA descriptor，管理barrier同步；(4) 编译为CUDA binary后在H100上执行，NCU monitor采集性能计数器。TK GEMM kernel仅40行device code即与CuBLAS竞争。整个TK attention kernel约217行LoC，对比FlashAttention-3的CUTLASS实现约2325行LoC。
