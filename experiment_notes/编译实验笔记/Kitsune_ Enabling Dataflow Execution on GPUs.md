## Kitsune: Enabling Dataflow Execution on GPUs

- 属于编译框架的实现是什么？实验比较什么？
  实现是基于PyTorch Dynamo的端到端编译后端，将DL应用自动lowering到dataflow执行（spatial pipeline）。核心包括三部分：(1) Subgraph Selection（§5.1）——从PyTorch Dynamo提取的完整计算图中，通过启发式模式匹配（pattern-matching）选取候选子图（sf-node），排除bulk-sync友好的节点和跨全部数据index/gather的节点（如embedding），在拓扑序中搜索预定义的算子链模式；(2) Pipeline Design（§5.2）——在sf-node节点间插入软件队列（ring buffer queue），对简单1-1 producer-consumer关系直接插入queue节点，对复杂pattern（如attention、back-propagation的reduction）构建并行reduce归约树；代码生成阶段将CUDA kernel改写为从queue读写而非global memory，kernel改写约8人时/个、10-40行代码修改；(3) Load Balance（§5.3）——通过integer linear program (ILP)将CTA分配到各pipeline stage，最大化子图吞吐量，约束包括DRAM bandwidth、L2 bandwidth、每个stage的CTA分配数，并支持将CTAs超额订阅到SM以实现SIMT-heavy和TensorCore-heavy CTAs的overlap colocation。

  实验比较：(1) Kitsune vs baseline（unmodified PyTorch BSP执行）；(2) Kitsune vs 垂直融合（vertical fusion，综合TensorRT + AStitch + Welder的技术）；(3) 硬件敏感性分析——2× SM count、2× L2 bandwidth、2× DRAM bandwidth下的加速比变化。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 GPU（108 SMs, 192 KB shared memory/SM），通过NVArchSim (NVAS)混合trace/execution-driven GPU simulator评估，simulator已针对NVIDIA Ampere GPU验证。队列性能微基准在真实A100硅片上测量（100 M atomics/sec/CTA）。硬件敏感性实验改变SM数量、DRAM带宽、L2/crossbar带宽。

- 开源编译框架是什么。修改了什么。
  基于PyTorch 2.0 Dynamo开源的编译框架。Kitsune修改/新增：(1) 自定义PyTorch Dynamo后端——消费PyTorch Dynamo提取的前向+反向图，构建spatial pipeline；(2) 模式库（pattern library）——一组正则表达式描述可融合的子图拓扑模式；(3) ILP求解器——用于负载均衡的整数线性规划最优CTA分配；(4) 软件队列库——基于CUDA atomics的L2-resident ring buffer queue，提供acquire/release API；(5) CUDA kernel改写——手动将DL算子kernel改写为从queue读写tiled数据（每个kernel约8人时）。

- 开源情况。基于开源文档和论文，使用例子解释编译框架如何使用？作用是什么？至少具体到编译框架输入到输出的全过程。
  论文为NVIDIA研究团队发表，论文未提供开源链接。系统基于PyTorch Dynamo开源框架构建。

  作用：Kitsune编译器的核心理念是"用spatial pipeline替代temporal multiplexing"——将DL计算图的不同算子映射到不同CTA，通过on-chip queues传递tile级中间数据，使多算子在不同SM上同时执行，避免BSP的global barrier和垂直融合的temporal multiplexing限制。

  全过程（以MeshGraphNets forward pass中的MLP子图为例）：
  ```
  输入：PyTorch application (MeshGraphNets训练脚本)
    → PyTorch Dynamo: 提取完整前向+反向计算图（FX graph）

  Step 1 - Subgraph Selection (§5.1):
    → Pattern-matching在拓扑排序后的FX graph中搜索预定义模式
    → 规则1: 排除bulk-sync友好的节点（如large GEMM that already achieves >50% peak compute）
    → 规则2: 排除gather节点（跨所有数据index，如embedding lookup）
    → 模式匹配：识别MLP链（Linear → Elementwise → Linear），选为sf-node
    → 输出：标注了sf-node的labeled graph

  Step 2 - Pipeline Design (§5.2):
    → 遍历sf-node内的每个节点n_i:
      - 若n_i是Reduction → SplitReduction()为partial_reduce + final_reduce
      - 若n_i产生intermediate → CreateQueue()创建queue节点
      - 为n_i的每个后继consumer j: 设置queue的producer/consumer关系
    → 对trivially fusable的相邻节点pair使用epilogue fusion（垂直融合）
    → 代码生成:
      - Queue节点 → lowering到§4.1的ring buffer queue（L2 resident, atomics-based）
      - 改写CUDA kernel: 原读写global memory → 读写queue
        例：GEMM kernel原来load A from global → 改为acquire from input queue
            原来store C to global → 改为release to output queue
      - GEMM算子已有tiling，queue payload size选择与tile size匹配（64-256KB）
    → 输出：pipelined graph (含queue节点)

  Step 3 - Load Balance (§5.3):
    → 对sf-node内每个stage i:
      - t_i = 该算子的bulk-sync throughput（实测获得）
      - u_i = 最大资源利用率（SIMT或TensorCore pipeline），Speedup(a_i) = 1/u_i
      - ResourceScale(a_i) = 估计随CTA分配数a_i的scaling因子
    → ILP formulation (Algorithm 2):
      maximize Throughput
      subject to:
        Throughput < t_i × ResourceScale(a_i) × Speedup(a_i)  (对每个stage i)
        Throughput × DRAM_Bytes < DRAM_Bandwidth
        Throughput × L2_Bytes < L2_Bandwidth
        1 ≤ a_i ≤ #SMs
        Σ IsSimt_i × a_i = #SMs  (SIMT和Tensor CTAs可overlap在同一SM)
        Σ IsTensor_i × a_i = #SMs
    → 求解ILP → 输出CTA分配方案
      例：MLP sf-node: Linear_1: 64 CTAs (Tensor), ReLU: 44 CTAs (SIMT), Linear_2: 44 CTAs (Tensor)
    → 通过cudaPipeline API启动spatial pipeline

  输出：可执行spatial pipeline
    → GPU上执行: Linear_1 CTAs → queue → ReLU CTAs → queue → Linear_2 CTAs
      不同operator的CTAs并发执行，通过queue传递tile级中间数据
      Modified grid scheduler确保SIMT和Tensor CTAs在SM上co-locate
  ```
