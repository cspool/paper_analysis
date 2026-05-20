## RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

- 属于Serving调度的实现是什么？实验比较什么？
  提出RoMeo LLM serving系统，集成RTMPQ算法到SGLang serving框架。核心系统级实现：(1) 异步并发执行（Asynchronous Concurrent Execution）：将activation量化分解为outlier token量化和normal token量化两个独立任务，四个cross-precision乘法kernel（W4A4/W4A8/W8A4/W8A8）各自仅依赖一个量化任务，通过fine-grained task dependency graph在多个CUDA stream上异步并发执行。使用CUDA events确保仅有真实依赖关系处才同步。(2) CUDA Graph工作流捕获：使用CUDA Graph捕获整个serving工作流，消除kernel launch overhead、memory allocation cost和PyTorch framework overhead，捕获后的graph无同步重复执行确保GPU连续运行。(3) SGLang集成：将RoMeo quantized modules集成到SGLang v0.5.5的离线benchmarking pipeline，测量prefill throughput（tokens per second），支持tensor parallelism（TP=2 for 14B, TP=4 for 32B）。实验比较prefill throughput和end-to-end serving speedup，对比BF16 baseline和QuaRot。Qwen3-8B (TP=1, batch=64) 上RoMeo prefill throughput 20073.60 tok/s vs BF16 10545.13 tok/s (1.90× speedup)。

- 硬件平台是什么，配置是什么。
  NVIDIA GeForce RTX 4090 GPU (24GB memory)。多GPU serving: Qwen3-14B使用2×RTX 4090 TP，Qwen3-32B使用4×RTX 4090 TP。软件: Python 3.12, PyTorch 2.8.0, CUDA 12.8, SGLang v0.5.5。

- 开源Serving框架是什么。修改了什么。
  基于SGLang v0.5.5修改。核心修改：(1) 将RoMeo的PyTorch nn.Module quantized linear layer替换SGLang原生FP16 linear layer；(2) 集成RTMPQ算法的online outlier detection + mixed precision quantization + cross-precision GEMM到SGLang的prefill pipeline；(3) 使用CUDA Graph捕获包括quantization→GEMM→dequantization→post-overwrite的完整workflow；(4) 异步并发执行使用多个CUDA streams管理quantization和GEMM kernel的并行执行。所有evaluation通过SGLang官方offline benchmarking scripts完成，固定input sequence length=128。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：https://github.com/thu-pacman/RoMeo。Serving使用流程（以Qwen3-8B在单RTX 4090上serving为例）：
  1. 部署：加载Qwen3-8B PyTorch模型→RoMeo替换所有Linear为RTMPQ quantized modules→Hadamard rotation online融入FWT→offline weight outlier identification and quantization→JIT编译cross-precision CUDA kernels并缓存
  2. 请求到达（SGLang offline benchmark，batch size 8-64，seq_len=128）：Activation Hadamard Rotation (FWT)→per-token max value计算→top-k outlier token selection→online quantization（outlier→INT8, normal→INT4, 5% outlier ratio）→asynchronous concurrent GEMM execution across CUDA streams→dequantization with per-token scaling→post-mul overwrite of outlier positions
  3. CUDA Graph模式：graph捕获quantization→GEMM→dequantization→overwrite的完整执行图→重复replay无同步开销
  4. 多GPU场景：Qwen3-14B用TP=2，每GPU加载一半模型→RoMeo替换→NCCL all-reduce通信开销部分抵消量化加速
  5. 在batch=64时RoMeo prefill throughput达20073 tok/s (1.90× over BF16 10545 tok/s)；Qwen3-14B TP=2上batch=64达9064 tok/s (1.32× over BF16 6848 tok/s)

