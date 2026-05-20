## FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出基于DSM的fused kernel dataflow调度，在NVIDIA H100 cluster架构上实现GEMM chain/Gated FFN/conv chain的跨SM数据流编排。核心kernel调度设计：(1) Cluster-level tiling：定义clsi（沿维度i的cluster内并行block数）和blki（block沿维度i计算的数据粒度）。cluster size四维(clsm, clsn, clsk, clsl)决定inter-block dataflow模式。(2) dsm_comm原语调度：GEMM0阶段K维spatial partition产生partial sum→dsm_all_exchange在cluster内沿K维All-Reduce聚合完整中间tile；GEMM1阶段dsm_shuffle在Shuffle Group内交换C矩阵切片供不同block计算不同E partial sum；Store阶段dsm_reduce_scatter cluster内scatter-reduce+inter_cluster_reduce TMA跨cluster原子聚合。对Gated FFN：dsm_all_exchange从Add变为Mul操作；两种spatial策略——cls_k=2空间划分两GEMM branch最大化并行（但增加DSM通信）vs 单block内顺序执行两GEMM最小化DSM通信。(3) Loop scheduling：统一codependent loop dimensions为独立集合X={x0,...,xJ-1}，通过permutation设置nesting order，partition为spatial(并行处理)或temporal(单processor顺序执行)维度。MNLK vs MLNK order影响需cached tensor大小和spilling策略。(4) Resource mapping：heuristic-driven贪心分配——按getFootprint确定单tile数据量→greedy分配到reg→SMEM→DSM→global层次→容量不足时逐层spill。(5) Kernel codegen调度：扩展CUTLASS prologue→mainloop→epilogue结构——prologue初始化DSM semaphore；mainloop插入DSM通信原语（生产者accumulation done→DSM mul/exchange→barrier；消费者循环→DSM shuffle ring communication）；epilogue执行DSM reduce后store global memory。实验比较GEMM chains/conv chains/Gated FFN kernel latency vs BOLT/Chimera/Relay/TASO/TensorRT/PyTorch；DSM primitive bandwidth utilization（shuffle/reduce/mul在不同cluster size下）；ablation隔离SE/DA/DC贡献；cost model搜索accuracy和top-K选择；brute-force vs search engine编译时间。

- 后端平台是什么，配置是什么。
  NVIDIA H100 SXM GPU（Hopper架构），CUDA 12.4。利用H100 Thread Block Cluster机制（最多16个SMs per cluster，DSM通过SM-to-SM NoC连接）。单SM SMEM上限约227KB。DSM bandwidth随cluster size增加逐渐降低（cluster size=2时最高，cluster size=16时最低但仍优于global memory bandwidth）。DSM latency始终低于global memory。

- 评估性能的软件/脚本是什么。修改了什么。
  FlashFuser自研Python search engine + CUTLASS CUDA codegen。评估使用NVIDIA Nsight Compute 2025.2.0 profiling memory access；PyTorch 2.6作为non-fusion baseline；BOLT/Chimera/Relay/TASO作为compiler baseline；SGLang用于端到端evaluation。修改：在CUTLASS kernel结构中嵌入DSM communication primitives；Python前端实现Dataflow Analyzer和Fusion Search Engine；内核扩展TMA数据移动+mbarrier many-to-many同步替代原有all-to-one cluster-sync。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  论文未明确开源。Kernel调度使用流程：
  1. 子图定义：输入GEMM chain维度(M,N,K,L)或conv chain(IC,H,W,OC1,OC2,K1,K2)或Gated FFN参数
  2. Search space构建：枚举loop schedule（MNKL/MNLK等×spatial/temporal partition 24+12+4+1=41种组合）×tile size（block-level以16×16×16 MMA为最小粒度）×cluster size（{1,2,4,8,16}^4=54种）→初始约2.75×10^13候选
  3. Pruning：5条规则逐级过滤→Rule1可整除tile→Rule2 cluster product≤16→Rule3 accumulation dim innermost→Rule4 L维度不能spatial→Rule5 capacity≤lowest cache→约1.15×10^6候选
  4. Dataflow Analyzer：对每候选生成spilling plan和data movement volume→cost model C_l=V_l/B_l minimax评估→保留top-11
  5. H100 profiling：11个候选编译为CUDA kernel→on-device测实际latency→选最优
  6. 例如GPT-6.7B FFN fusion：cluster size(2,4,2,4)→blkm=128, blkk0=128, blkn=128, blkk1=64, blkl=128→GEMM0 M=128沿M spatial partition到2平行block→K=4096沿K partition到2平行block产生partial C→dsm_all_exchange沿clsk=2 All-Reduce→GEMM1 dsm_shuffle clsshuffle=2交换C→dsm_reduce_scatter clsreduce=2聚合→inter_cluster_reduce→store E
  7. Runtime：M维在inference时动态变化→binning策略按M范围预编译多kernel→查表选择

