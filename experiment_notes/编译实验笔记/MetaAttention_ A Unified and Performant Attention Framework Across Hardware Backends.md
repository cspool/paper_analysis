## MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends

- 属于编译框架的实现是什么？实验比较什么？
  提出MetaAttention，一个从用户定义attention模板自动生成跨硬件后端优化kernel的编译框架。核心设计：(1) Attention Abstraction：将多种attention机制抽象为relevance scoring + aggregation两个核心操作，实例化为Parallel Pattern（全局K/V context matmul）和Recurrent Pattern（迭代序列维护hidden state）；(2) Programming Interface：用户通过input tensor shapes、attention pattern、customizable functions（Mod函数支持elementwise scaling/masking，RowNorm函数支持row-wise normalization，RowNorm online接口支持online softmax/RetNet normalization等分块在线归一化）定义attention变体；(3) IntermediateTensor-based Scheduling：定义IntermediateTensor（包含tile、memory location、pipeline stage属性）建模所有中间张量，DeviceConfig（包含basetile、memoryInfo）抽象硬件约束，两层调度策略——外层TileConfigScheduling枚举output tensor tile size并传播到所有中间张量，内层TileResourceScheduling确定内存位置和pipeline stage并逐步降级tensor memory tier以满足约束；(4) 编译Lowering：frontend将customizable functions trace成tensor DAG（elementwise节点SIMT融合，row-reduce节点intra-warp parallel reduction），runtime将优化调度计划lowering为TileLang/CUTE后端kernel，custom logic直接inline到attention loop中消除额外kernel launch。实验比较：operator-level normalized latency对比FlashAttention-2/3、FlashSigmoid、FlashMLA、Mamba2 chunk kernel、Flash-Linear-Attention、FlexAttention、FlashInfer、PyTorch Inductor等baseline。Compilation time（Table 4）：H100上Softmax Attention 46秒、Mamba2 SSM 82秒，MI250上Softmax Attention 64秒、Mamba2 SSM 89秒。Table 5显示开发效率：Softmax Attention 87 LoC vs 2.7k CUDA（FlashAttention-3），MLA 90 LoC vs 1.7k CUDA（FlashMLA）。端到端H100推理平均1.4× speedup，训练平均1.4× speedup。

- 硬件平台是什么，配置是什么。
  NVIDIA H100 SXM5（CUDA 12.4, Triton 2.3.1）、AMD Instinct MI250（ROCm 6.2.4, Triton 3.1.0）。NVIDIA后端使用Tensor Memory Accelerator (TMA)异步加载+Tensor Cores MMA+TileLang/CUTE backend。AMD后端使用Matrix Cores+asynchronous copy units+TileLang backend。

- 开源编译框架是什么。修改了什么。
  基于TileLang（composable tiled programming model）和CUTE（CUDA Templates for Linear Algebra）两种high-level backend实现kernel templates。未修改TileLang/CUTE本身，而是在其之上实现：(a) attention pattern kernel templates（parallel/recurrent两种模式，包含global↔shared↔register数据搬移、输入在shared/register时的matrix multiplication）；(b) scheduling policy engine（两层调度，从attention computation graph搜索最优IntermediateTensor配置）；(c) customizable function lowering（trace→DAG→elementwise SIMT fusion + row-reduce intra-warp reduction→inline到attention loop）；(d) DeviceConfig hardware abstraction（base tile shape、register/shared/global memory容量约束）。整体实现约7.3k行C++和Python。

- 开源情况。编译框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：GitHub https://github.com/SJTU-IPADS/MetaAttention，Zenodo archive DOI: 10.5281/zenodo.17701680。使用流程：
  1. 选择Pattern：Parallel Pattern（全局K/V context）或Recurrent Pattern（迭代hidden state）
  2. 定义input shapes：声明Query/Key/Value的batch/head/seq_len/dim等维度
  3. 定义customizable functions：Mod（如`def scores_Mod(scores): return scores * mask`）、RowNorm（如RetNet的reduceAbsSum-based row normalization）、RowNorm online（如online softmax的prologue/forward/epilogue三段式）
  4. MetaAttention trace functions→tensor DAG→scheduler搜索最优tile/memory/pipeline配置→runtime生成CUDA/ROCm kernel
  5. 例如RetNet attention：选择Parallel Pattern→定义Q[batch,head,seq_len,256]/K[batch,head,seq_len_kv,256]/V[batch,head,seq_len_kv,512]→scores_Mod缩放+scores_RowNorm归一化→MetaAttention自动生成带online normalization的fused kernel

