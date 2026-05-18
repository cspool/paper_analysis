## FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection

- baseline方法是什么？
  Baseline是现代GPU上compute-intensive operator chain的传统kernel fusion方法。以GPT-6.7B FFN（两个连续GEMM：M×K×N→M×N×L）为例的全栈执行：
  - **算法层**：标准Transformer FFN，两个GEMM间有一个激活函数（ReLU/SiLU），权重为预训练固定值。M=128(batch token), N=16384(intermediate), K=4096(hidden→intermediate), L=4096(intermediate→hidden)
  - **系统框架/Serving层**：PyTorch 2.6 + torch.compile（减少kernel launch overhead），或SGLang。cuBLAS将每个GEMM作为独立kernel launch，BOLT/Chimera尝试融合，TensorRT做graph optimization
  - **编译框架层**：cuBLAS/CUTLASS将两个GEMM编译为两个独立kernel（或BOLT使用CUTLASS模板融合、Chimera做block order探索）。中间结果tensor C [M×N=128×16384=~2M floats≈8MB] 必须存到某处。单SM SMEM上限约227KB，远小于8MB C matrix
  - **kernel调度层**：cuBLAS中第一个GEMM kernel将完整结果C写回HBM global memory；第二个GEMM kernel再从HBM读取C作为输入。形成"write-then-read" round-trip。BOLT尝试在SMEM内fusion但受限于模板固定block执行顺序；Chimera可在reg/SMEM中保留部分C tile但大模型下SMEM capacity不够→fusion failure，回退到global memory round-trip
  - **硬件架构层**：NVIDIA H100 GPU，HBM带宽3TB/s，FP16算力~1000 TFLOPS。H100具备DSM（SM-to-SM NoC, cluster内最多16 SMs），但baseline软件栈不使用DSM——中间数据只能走global memory路径

  **Baseline缺陷**：(1) 单SM SMEM约227KB的严格容量限制，当中间tensor超过阈值时fusion直接失败(图5)；(2) 即使partial fusion可能，fixed block order（BOLT）和忽略DSM导致中间结果必须round-trip HBM；(3) 全局memory访问约为FlashFuser的2.4倍(图11)，FFN占LLM推理40%–60%时间且memory-bound

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出FlashFuser编译框架，通过DSM扩展on-chip memory pool，实现大中间tensor的cross-SM fusion。以同一GPT-6.7B FFN为例：
  - **算法层**：与baseline相同（标准FFN/Gated FFN/conv chain的GEMM运算），无算法修改
  - **系统框架/Serving层**：FlashFuser作为offline compiler预编译多版本fused kernel→运行时通过binning/table lookup按动态变化的M选kernel。与SGLang集成实现端到端加速（平均1.32× speedup for small models, 1.24× overall）
  - **编译框架层**：核心创新——三个层次化设计。
    (a) dsm_comm primitives：定义DSM-aware通信原语（dsm_all_exchange→cluster内All-Reduce聚合partial C tile；dsm_shuffle→ring communication交换C切片；dsm_reduce_scatter→scatter-reduce聚合partial E；inter_cluster_reduce→TMA cp.reduce.async.bulk跨cluster）。编码cluster size四维参数(clsm,clsn,clsk,clsl)和派生变量clsshuffle/clsreduce
    (b) Dataflow Analyzer：将loop schedule/tile/resource mapping扩展到reg→SMEM→DSM→global四层。贪心策略优先放高层缓存，容量不足逐层spill——关键：当C matrix tile超出SMEM 227KB时，不再直接fusion fail，而是spill到DSM（cluster内多SM SMEM聚合可达3.6MB=16×227KB）。分析各层数据搬移量供cost model评估
    (c) Fusion Search Engine：用C_l=V_l/B_l minimax cost model + 5条DSM-aware pruning rules在2.75×10^13→1.15×10^6搜索空间中找到最优plan。Top-K=11候选实测选最佳
  - **kernel调度层**：执行分三阶段——
    GEMM0: K维spatial partition(clsk=2)两平行block计算C partial→dsm_all_exchange沿K维All-Reduce得完整C tile→C tile留在DSM不写global memory（避免Chimera的fusion failure和cuBLAS的HBM round-trip）
    GEMM1: dsm_shuffle在Shuffle Group内ring communication交换C切片→各block与D tile乘得E partial
    Store: dsm_reduce_scatter cluster内scatter-reduce + inter_cluster_reduce TMA原子聚合→最终E写global memory
    关键trade-off：DSM bandwidth随cluster size变化（越大cluster带宽越低），需cost model平衡；spatial partition最大化并行vs sequential执行最小化DSM通信
  - **硬件架构层**：NVIDIA H100 GPU。FlashFuser利用DSM（SM-to-SM NoC on-chip path）替代global memory round-trip——DSM latency始终低于global memory，DSM bandwidth（除最大cluster size 16外）高于HBM bandwidth。dsm_comm通过TMA数据移动+mbarrier many-to-many synchronization实现，替代默认all-to-one cluster-sync（如CUTLASS），支持仅同步必要的CTA子集以构建higher-level collectives

  **解决效果**：通过DSM将on-chip memory从单SM SMEM 227KB扩展到cluster级~3.6MB，使8MB C matrix的tile可以留在片上。减少58% global memory access，GEMM chain kernel speedup 3.1×-5.4× over baselines，整体1.24× end-to-end。关键不是简单"多一层缓存"，而是把DSM变成可搜索、可建模、可codegen的编译器抽象层次
