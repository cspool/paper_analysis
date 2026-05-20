## FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection

- 属于编译框架的实现是什么？实验比较什么？
  提出FlashFuser，首个利用现代GPU inter-core Distributed Shared Memory (DSM)进行compute-intensive operator chain kernel fusion的编译框架。核心实现：(1) dsm_comm原语抽象：定义dsm_all_exchange（cluster内沿K维accumulation）、dsm_shuffle（shuffle group内交换中间tensor切片）、dsm_reduce_scatter（cluster内scatter-reduce）、inter_cluster_reduce（基于Hopper TMA cp.reduce.async.bulk跨cluster原子reduce）和dsm_mul四种primitive，统一编码cluster-level SM划分和inter-SM dataflow。通过clsshuffle=clsl/clsk和clsreduce=clsn/clsshuffle两个派生变量精确描述shuffle/reduce的group配置。(2) Dataflow Analyzer：将loop schedule、tile size、resource mapping扩展到register/SMEM/DSM/global memory层次化空间。对复用tensor采用贪心spilling策略——优先放到最高层缓存，容量不足时逐层spill到更低层级（reg→SMEM→DSM→global），计算每层数据搬移量。DSM带宽随cluster size变化（越大cluster带宽越低延迟越高），dataflow分析考虑这一特性。(3) Fusion Search Engine：前端Python engine枚举LoopSchedule（MNKL/MNLK/MLNK等顺序+spatial/temporal partition）、TilingSize（block-level tile+cluster-level tile）和ResourceMapping，调用Dataflow Analyzer得每配置数据搬移量，用C_l=V_l(T_l)/B_l的minimax优化cost model和5条pruning rules（Rule1硬件aware可整除tile；Rule2 cluster size≤16乘积约束；Rule3 activation约束accumulation dim必须innermost loop；Rule4 dependency约束L dim不能spatial；Rule5 memory capacity limit）过滤搜索空间。GPT-6.7B时搜索空间从2.75×10^13 prune至1.15×10^6。Top-K(K=11)候选传给hardware profiling选最优kernel。Backend基于CUTLASS扩展prologue/mainloop/epilogue结构生成CUDA code，dsm_comm通过TMA+mbarrier many-to-many synchronization实现。实验比较：GEMM chain上vs BOLT/Chimera/Relay/TASO/TensorRT/PyTorch平均5.4×/4.6×/4.7×/3.4×/2.4×/3.1× speedup；convolution chain上平均6.3×/6.4×/5.6×/4.3×/3.3×/3.9× speedup；Gated FFN上额外vs Mirage/PipeThreader；SGLang端到端平均1.32× speedup（含大模型Llama3-70B/Qwen2.5-14B/32B时1.16×-1.24×）。Ablation：All(SE+DA+DC) 3.29× vs DC+DA(random) 2.11× vs DA(仅SMEM/global) 1.52× vs no-fusion baseline。Nsight Compute显示平均减少58% global memory access。

- 硬件平台是什么，配置是什么。
  NVIDIA H100 SXM GPU（Hopper架构）。主机双路Intel Xeon Platinum 8468（96 cores, 2.10GHz）。软件栈：CUDA 12.4, PyTorch 2.6, TVM 0.9, Triton 3.2, Nsight Compute 2025.2.0。

- 开源编译框架是什么。修改了什么。
  基于NVIDIA CUTLASS构建代码生成框架。未修改CUTLASS本身，在其之上新增：(a) dsm_comm primitive实现——SHUFFLE使用TMA数据移动+mbarrier实现ring communication；MUL和REDUCE使用TMA+mbarrier实现cluster内collective操作；inter-cluster reduce使用TMA cp.reduce.async.bulk。(b) 扩展CUTLASS kernel结构：prologue初始化DSM semaphore，mainloop插入DSM mul/shuffle操作（生产者accumulation完成→DSM mul/exchange；消费者accumulation循环→DSM shuffle ring communication），epilogue执行DSM scatter-reduce+inter-cluster reduce后写global memory。(c) Python前端search engine——枚举loop schedule/tile/cluster配置→Dataflow Analyzer分析→pruning+cost model筛选→top-K profiling。(d) Runtime kernel selection：通过binning/table lookup针对inference时动态变化的M维度选择预编译kernel（N/K/L固定）。

- 开源情况。编译框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  论文未明确提供开源仓库链接。基于CUTLASS构建，使用流程：
  1. 输入：DNN子图描述（GEMM chain/conv chain/Gated FFN的维度参数M/N/K/L）→search engine枚举loop schedule和tile配置
  2. dsm_comm primitive声明：根据模型类型选择Standard FFN或Gated FFN communication pattern——定义cluster size(clsm,clsn,clsk,clsl)和block tile(blkm,blkn,blkk,blkl)参数
  3. Dataflow Analyzer分析：对每配置计算loop schedule指定执行顺序→GetFootprint确定单tile访问量→I/O tensors计算global memory搬运量→reused tensors贪心分配到reg→SMEM→DSM→global层次→计算DSM traffic和总data movement
  4. Cost model+pruning：C_l=V_l/B_l计算每层搬移成本→minimax优化避免单一层成为瓶颈→5条pruning rules过滤→保留top-11 candidates
  5. Backend code gen：CUTLASS模板+dsm_comm primitive代码→编译为CUDA kernel→H100实测选最优
  6. Runtime：针对不同M值预编译多版本kernel→inference时查表选择对应kernel
  7. 例如GPT-6.7B FFN (M=128, N=16384, K=4096, L=4096)：cluster size=(2,4,2,4)→GEMM0沿K=2 partition→dsm_all_exchange聚合C tile→GEMM1 dsm_shuffle交换C切片→Store phase dsm_reduce_scatter+inter_cluster_reduce产生E→全程中间tensor不写global memory

FlashFuser编译器的作用：通过将DSM纳入编译器的自动搜索、分析和代码生成，突破传统单SM SMEM约227KB的融合限制，使原本因中间结果过大而无法融合的compute-intensive operator chain（如LLM FFN/conv block）可以fusion，减少global memory round-trip，实现kernel级3.3×-4.1× speedup和端到端1.24× speedup。

