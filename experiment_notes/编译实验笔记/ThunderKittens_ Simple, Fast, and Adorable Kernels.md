## ThunderKittens: Simple, Fast, and Adorable Kernels

  **备注：部分匹配编译框架。TK本质是C++ embedded kernel开发框架（非编译器），但其tile抽象层提供了编译器级别的高级接口——自动管理shared memory布局（编译时选择最优swizzle避免bank conflict）、自动生成TMA tensor map descriptor、编译期静态检查布局与操作兼容性（如mma_AB要求A为row-major、B为col-major，不匹配则编译报错）。属于编译框架的"邻近层次"。**

- 属于编译框架的实现是什么？实验比较什么？
  ThunderKittens提供一个C++ embedded编程框架，通过三层抽象桥接高层ML算子语义与底层GPU硬件指令：(1) warp级tile数据结构——16×16矩阵tile（rt、st、gl类型），支持BF16/FP16/FP32/FP8多种精度，自动管理register/shared memory布局，提供PyTorch风格操作原语(mma、exp、cumsum、sub_row、div_row等)；(2) block级LCSF编程模板——将kernel分解为Load/Compute/Store/Finish四个阶段，框架自动管理multi-stage pipeline buffer、同步barriers（arrive函数）、异步I/O（统一包装cp.async和TMA指令）；(3) grid级persistent launch和block order调度。实验比较GEMM vs CuBLAS/CUTLASS，Attention vs FlashAttention-3，以及Linear Attention、SSM、FFT Convolution、Mamba-2、RoPE、fused layernorm等广泛AI workload的kernel对比。

- 硬件平台是什么，配置是什么。
  NVIDIA H100 80GB SXM5 GPU（CUDA 12.6），扩展测试NVIDIA RTX 4090和Apple M2 Pro。

- 开源编译框架是什么。修改了什么。
  TK是一个全新的独立C++ embedded框架（开源地址：https://github.com/HazyResearch/ThunderKittens），而非修改现有编译框架。它定位于CUTLASS（过于复杂，nested templates，手动管理bank conflict）和Triton（编译器自动但无法使用特殊硬件指令、难以管理异步执行和register）之间的平衡点。TK的include/目录<1MB（对比CUTLASS 22MB、Triton 12.6MB）。

- 开源情况。基于开源文档和论文，使用例子解释编译框架如何使用？作用是什么？至少具体到编译框架输入到输出的全过程。
  开源地址：https://github.com/HazyResearch/ThunderKittens。使用流程：(1) 用户include TK头文件，using namespace kittens；(2) 定义tile类型——rt_bf<16, 64>声明16×64 BF16 register tile，st_bf<64, 64>声明shared memory tile，gl<bf16, -1, -1, -1, D>声明global memory layout descriptor（-1表示runtime维度，D为compile-time维度节省register）；(3) 编写kernel函数，使用PyTorch风格的操作：load(tile, smem)加载数据、mma_ABt(accum, A, B, accum)调用tensor core矩阵乘法、exp(tile, tile)逐元素指数、copy(dst, src)类型转换；(4) 在LCSF模板中定义producer（load worker：TMA异步加载HBM→SMEM→arrive barrier通知）和consumer（compute worker：从SMEM tile执行mma→softmax→mma→arrive barrier）的struct；(5) 编译时TK自动根据tile width选择最优shared memory swizzle布局（≤32列用32B swizzle/4-way conflict，≤64列用64B swizzle/2-way conflict，>64列用128B swizzle/0 conflict），自动生成TMA tensor map descriptor，编译期检查layout兼容性；(6) nvcc编译生成CUDA binary，在GPU上执行。整个process从用户的算子语义描述（<200行C++代码）到硬件指令执行，TK作为中间层管理了数据布局、同步和异步执行的复杂性。
