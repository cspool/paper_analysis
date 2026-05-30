## Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是Tilus的thread-block级VM指令集（Table 1）对低精度tensor计算和内存传输的调度。核心kernel调度机制包括：(1) 软件流水线（software pipelining）——通过CopyAsync/CopyAsyncCommitGroup/CopyAsyncWaitGroup指令实现异步global→shared memory拷贝与计算的overlap，在decode stage batch>1时显著优于Ladder；(2) 显式内存层次调度——LoadGlobal/StoreGlobal操作global memory，LoadShared/StoreShared操作shared memory，CopyAsync异步global→shared拷贝，AllocateRegister管理register分配，开发者精确控制数据placement和movement；(3) 低精度weight loading pipeline——global memory layout预变换 + u8高效加载 + 零开销View reinterpret + 寄存器内PRMT/LOP3 vectorized casting，消除Triton的shared memory layout conversion瓶颈和Ladder的pipelining缺失；(4) auto-tuning——200配置per operator，auto-tune tile大小（BM, BN, BK参数）；(5) k-dimension parallelization支持。

  实验比较：operator级——vs Triton v3.1.0、Ladder (bitblas v0.0.1.dev15)、QuantLLM (commit 9802c5a)、Marlin v0.1.1，在batch size 1和16下评估低精度matmul（uint8, f6, uint4, int4, uint2, uint1）vs cuBLAS FP16的speedup。end-to-end级——将Tilus kernel集成至vLLM v0.5.3，在Gemma-2-9B、QWen2.5-32B、Llama-3.3-70B上评估prefill（2048 tokens）和decode（1/16 batch）延迟。跨batch size评估覆盖decode BS=1,4,8,16和prefill BS=4096,8192,12288。

- 后端平台是什么，配置是什么。
  NVIDIA L40S GPU (48 GiB, Ada Lovelace)，driver 565.57.01，CUDA 12.6.3。跨架构验证：NVIDIA A100 (Ampere, compute capability 8.0)、NVIDIA H100 (Hopper)。H100上Ladder产生非法指令（ERR），vLLM FP16在L40S上OOM（qwen2.5-30B超过48GB）。

- 评估性能的软件/脚本是什么。修改了什么。
  Tilus kernel通过单一参数化Python程序模板生成，支持所有低精度类型。集成至vLLM v0.5.3做end-to-end（artifact使用vLLM 0.7.3）。实验脚本bash run.sh自动拉取Docker镜像并顺序运行所有实验。Docker镜像预装PyTorch v2.5.1、Triton v3.1.0、BitBLAS v0.0.1.dev15、Marlin v0.1.1。kernel性能测量使用CUDA Events，50次执行取median latency，每次执行前清除L2 cache。低精度类型支持通过预处理kernel（图9）将权重变换为标准类型兼容的layout实现。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/NVIDIA/tilus
  Artifact: https://github.com/yaoyaoding/tilus-artifacts (DOI: 10.5281/zenodo.16756859)
  Docker image (~21 GiB)，所有依赖预装。

  评估原理：Tilus将GPU kernel执行建模为thread-block级VM指令序列。每条指令（LoadGlobal, CopyAsync, View, Cast, Dot, StoreGlobal等）操作于整个thread block，编译时逐条生成低级GPU代码。性能优势来自：(1) 通过flattened layout预变换避免低精度加载的非连续内存访问；(2) 零开销View reinterpret消除Triton的shared memory layout conversion；(3) CopyAsync软件流水线解决Ladder的pipelining缺失；(4) 寄存器内PRMT/LOP3向量化casting避免shared memory往返。

  Kernel输入到性能输出全过程（FP16×INT4 decode matmul, BS=1, K=8192, N=57344）：
  1. 输入：A f16[BS, K] in global memory (activation cache)；B transformed u8[BK*BN*4/8 per tile] in global memory (权重预变换后的连续u8字节)
  2. 异步预取：CopyAsync将下一K-iteration的B tile从global memory拷贝到shared memory（pipelined with 当前iteration的computation）
  3. 同步：CopyAsyncCommitGroup() + CopyAsyncWaitGroup(0)确保预取完成
  4. 加载activation：LoadGlobal A tile [BM, BK] from global memory → registers, layout=m16n8k16 compatible
  5. 加载weight：LoadShared B tile from shared memory → registers, dtype=u8, layout=local(3).spatial(32)
  6. 零开销reinterpret：View(b_tile, dtype=i4, layout=spatial(8,4).repeat(1,4)) — 32 threads × (4×i4=16 bits) → reinterpret到Tensor Core兼容layout
  7. 向量化casting：Cast(b_tile, f16) — 使用PRMT permute bytes + LOP3 logical ops + bitwise指令在registers内完成
  8. Tensor Core计算：Dot(a_tile, b_tile, C_accum) → PTX mma.m16n8k16, 累加到f32 accumulator
  9. 循环K维：重复Steps 2-8直至K维完成
  10. 输出：Cast(C_accum, f16) → StoreGlobal → global memory 写出结果tile
  11. 测量：CUDA Event记录kernel start/stop → latency = stop - start。50次执行，clear L2 cache between runs，取median
  12. Speedup = cuBLAS FP16 kernel latency / Tilus kernel latency
