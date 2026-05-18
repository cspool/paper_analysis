## Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems

- baseline方法是什么？
  Baseline是SpecPIM类PIM-enabled heterogeneous speculative decoding系统（PIM-SD）。PIM-SD在HBM-PIM+GPU异构系统上运行speculative decoding：(1) DLM (如OPT-1.3B) 在PIM/GPU上自回归生成固定长度d=8的draft tokens；(2) TLM (如OPT-66B) 并行验证所有draft tokens；(3) operator mapping通过离线design-space exploration基于初始batch size和fixed draft length确定，推理中不改变；(4) DLM prediction和TLM verification严格串行执行。

  全栈执行例子（以OPT-66B+OPT-1.3B, Dolly dataset, batch_size=64, d=8为例）：
  - 算法层：standard speculative decoding with fixed draft length d=8。DLM (OPT-1.3B) autoregressive生成8个draft tokens→TLM (OPT-66B) parallel verification with rejection sampling。当d=8时acceptance rate从d=4的~0.6降至~0.4(Fig.4a)，大量draft token被拒绝后丢弃，浪费DLM生成和TLM验证的计算。
  - 系统框架层：PIM-SD采用静态operator mapping，DLM prediction→TLM verification串行执行。每轮speculative iteration：所有请求先等DLM生成d=8 tokens（micro-batch同步屏障），再统一TLM验证。batch内请求间draft长度相同无bubble，但固定长度导致整体吞吐在BS=64时反而低于autoregressive baseline (Fig.3a)。
  - 编译框架层：论文未明确说明。
  - kernel调度层：PIM-SD离线分析后固定映射：DLM attention→PIM, TLM FC→GPU。当effective batch size和draft length变化时映射不变。例如当batch内部分请求提前完成drafting后effective batch size降低，DLM FC算术强度降低（从GPU带宽限制转为PIM计算限制，Fig.6），但operator仍固定映射在GPU上执行，导致suboptimal utilization。
  - 硬件架构层：AttAcc风格HBM-PIM架构：每bank 1 PE，bank-level并行。PIM-SD的Manager无runtime adaptive control，仅执行离线确定的mapping schedule。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出SADDLE，针对PIM-SD的三大缺陷分别设计解决方案：

  **缺陷1：固定draft length → 生成大量被TM拒绝的无效token，浪费计算和带宽**
  → **SADDLE方案**：运行时自适应draft length。Controller每生成draft token时读取采样概率p_t→维护累计接受概率H_t=∏p_i→当H_t<阈值τ（离线用验证集校准，选20%区间内平均draft length最高且≥90%验证成功率的τ）时停止该请求drafting。简单请求（高p_t token）自动获得更长draft，复杂请求（低p_t token）更早停止。运行时可根据系统负载动态调节τ：轻负载降低τ允许更长draft→提升并行度。

  **缺陷2：DLM-TLM串行执行 + 自适应draft长度引入的同步bubble → 请求间等待加剧延迟**
  → **SADDLE方案**：prediction-verification解耦异步pipeline + Shared Pool + Eager Pool。(a) Shared Pool跨micro-batch聚合draft tokens：各micro-batch不再单独等待所有请求完成DLM prediction，draft tokens逐生成即存入Shared Pool，当token数达GPU capacity C(=512)或GPU空闲时触发TLM并行验证；(b) Eager Pool乐观执行：TLM验证Shared Pool时，DLM基于"当前token将被接受"假设继续生成后续tokens暂存Eager Pool，验证通过后迁入Shared Pool，被拒绝则丢弃；(c) 异步重叠prediction和verification，消除串行pipeline的idle time。

  **缺陷3：算术强度动态变化 → 静态operator mapping suboptimal**
  → **SADDLE方案**：arithmetic intensity-aware operator scheduler。(a) predication后根据仍活跃请求数估算DLM FC有效micro-batch size→计算arithmetic intensity→与预标定PIM/GPU阈值比较动态remap；(b) verification前根据Shared Pool每请求token数估算TLM attention arithmetic intensity→同理动态remap。初始固定映射：DLM attention→PIM (低强度)、TLM FC→GPU (高强度的GEMM)。动态remap使SADDLE中14.89% ops在PIM、85.11%在GPU执行（vs 无scheduling时的9.51%/90.49%），吞吐提升1.21×。

  论文方法全栈执行例子（以OPT-66B+OPT-1.3B, Dolly, BS=64为例）：
  - 算法层（核心创新）：自适应draft length。每请求dynamically调整draft长度：DLM生成token x_t时获取p_t=DLM(x_t|x_{<t})→更新H_t=H_{t-1}·p_t→若H_t<τ=θ则停止该请求drafting。H_t基于DLM自身采样概率，无需额外训练或分类器。相比baseline固定d=8时acceptance rate ~0.4，SADDLE自适应停止在H_t低于阈值时，每个请求的draft length在[1, optimal]区间动态变化，减少无效token生成。
  - 系统框架层（核心创新）：异步pipeline。batch切成micro-batches→每micro-batch有独立Draft Generator。请求#0 (simple task) H_t始终>τ→持续draft→tokens入Shared Pool。请求#1 (complex task) H_t在第3 token后<τ→停止drafting。不等待请求#1继续：Shared Pool累计token数达C→TLM并行验证所有已存tokens。同时请求#0在TLM验证期间继续生成新tokens→Eager Pool暂存→验证通过后migrate到Shared Pool。
  - 编译框架层：论文未明确说明。
  - kernel调度层（核心创新）：动态operator mapping。prediction后Scheduler统计仍活跃请求数→估算DLM FC effective batch size→与roofline阈值比较→决定FC在PIM或GPU执行。verification前Scheduler统计Shared Pool每请求token数→估算TLM attention arithmetic intensity→同样动态remap。例如当请求#1停止drafting后effective batch size降低→DLM FC arithmetic intensity降到PIM compute-bound区→Scheduler将DLM FC从GPU remap到PIM执行。Operator mapping随每speculative iteration动态调整。
  - 硬件架构层（核心创新）：SADDLE Manager硬件。Controller以专用硬件（softmax unit + multipliers + comparators）低延迟计算H_t并比较τ（仅占end-to-end latency 0.83%）。Shared Pool (1KB CAM)和Eager Pool (1KB) 的token migration为lightweight on-chip memory operation，每verification iteration后刷新无容量压力。SFU在buffer die加速softmax/layer norm等非矩阵运算。PE沿用HBM-PIM design (16 FP16 MACs/bank)，面积overhead仅13.4% DRAM die。

- baseline方法是什么？
  Baseline是ServerlessLLM (sllm, OSDI'24)，一个面向LLM serverless部署的系统。sllm的核心设计：(1) 每GPU独占分配给单个model instance，GPU资源不可共享；(2) event-driven分配：请求到达时若无运行中instance则在空闲GPU上启动新instance（cold-start经过fast model loading优化到~1s），否则排队等待；(3) 使用vLLM作为底层inference engine（paged-attention KV-cache管理+continuous batching），但vLLM默认将全部GPU memory分配给单一instance。

  全栈执行例子（以serverless场景下serving 64个7B LLM请求为例）：
  - 算法层：standard transformer autoregressive decoding，Llama-2-7B FP16，prefill+decode两阶段
  - 系统框架层：ServerlessLLM分配一个7B instance独占A100-80GB GPU → vLLM continuous batching处理该instance的in-flight requests → KV-cache静态预分配整个GPU memory → 其他model排队等待GPU释放。即使该instance仅用23% GPU memory（图5），剩余~60GB闲置。
  - 编译框架层：论文未明确说明（vLLM使用PyTorch CUDA backend，无编译框架自动生成）
  - kernel调度层：vLLM默认scheduler按batch内request到达时间执行continuous batching，无token-level精细化调度。GPU kernel为standard FlashAttention+GEMM，CPU核心仅用1 core（图10），其余31 core闲置。
  - 硬件架构层：NVIDIA A100-80GB GPU + Intel Xeon CPU (AMX idle)。GPU独占导致大量GPU memory over-provisioning，CPU matrix accelerator完全闲置。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出SLINFER，通过三个核心机制解决ServerlessLLM的三大缺陷：

  **缺陷1：GPU独占+over-provisioning → serving capacity低**
  → **SLINFER方案**：异构硬件抽象+弹性资源共享。将CPU/GPU统一为resource pool，instance不再独占整节点。CPU通过OpenVINO+AMX独立serve ≤13B LLM（第4代Xeon TTFT 567ms for 7B-1K），GPU上多model共享memory。实验证明64个7B model下SLINFER仅用0.9 GPU vs sllm的3.2 GPU。

  **缺陷2：无token-level compute调度 → SLO violation无法精细化预防**
  → **SLINFER方案**：Headroom-Driven Compute Subsystem。每scheduling cycle选最短headroom instance执行一个iteration，shadow validation在添加请求前虚拟模拟future compute探索三种SLO violation case（新请求prefill超时、现有请求被delay、aggregate decode超TPOT SLO）。performance quantification用linear/2D interpolation profiling，estimator偏差仅5.9%(TTFT)/3.9%(TPOT)。

  **缺陷3：静态内存+无协调 → OOM风险和fragmentation**
  → **SLINFER方案**：Hazard-Aware Memory Subsystem (watermark w=25% early scale-up + lazy scale-down, optimistic budgeting + pessimistic reservation station协调并发memory操作避免OOM) + Efficiency-Oriented Consolidator (proactive preemption让大batch instance抢占小邻居来scale-up, reactive bin-packing优先路由到大batch instance加速碎片回收)。

  论文方法全栈执行例子（以serving 64个7B LLM，一个Llama-2-7B请求到达为例）：
  - 算法层：standard transformer Llama-2-7B FP16 autoregressive decoding，同baseline
  - 系统框架层（核心创新）：SLINFER proxy收到请求→优先尝试CPU instance（通过OpenVINO backend）→compute subsystem对候选instance执行shadow validation：(a) 线性interpolation估计新请求prefill time，(b) 虚拟添加后仿真所有in-flight request headroom，(c) 检查三种SLO violation case均不发生→通过验证。memory subsystem检查node可用memory是否够容纳新请求的KV-cache（Mrequire=C·Σ(Ir+max(Or,Ō))），若需scale-up则检查optimistic budget→若不足则尝试compromise降级为Mrequire→若仍不足则evict最长headroom请求。请求加入后token-level调度器按headroom轮转instance执行iteration：每cycle选最短headroom instance执行一次decode/prefill→更新headroom→重复。
  - 编译框架层：论文未明确说明（vLLM PyTorch CUDA backend + OpenVINO CPU backend，无编译框架修改）
  - kernel调度层：SLINFER的compute subsystem替代vLLM默认scheduler，实现token-level跨instance scheduling。CPU instance用OpenVINO后端（AMX-accelerated matmul），GPU instance用vLLM CUDA backend（FlashAttention+GEMM）。对比baseline每instance独占GPU、batch size较小且CPU闲置，SLINFER实现instance sharing使average batch size提升74%→sub-linear compute growth特性带来更高吞吐。
  - 硬件架构层：NVIDIA A100-80GB GPU + Intel 4th Gen Xeon (AMX)。SLINFER充分发挥CPU的AMX matrix accelerator（7B 1K-input TPOT仅71ms vs SLO 250ms），CPU可独立serve ≤13B model短输入请求。GPU memory utilization接近1.0（vs sllm的three-tier阶梯分布），KV-cache scaling watermark机制使scaling overhead仅1.4%。

  Baseline缺陷→SLINFER方案映射：
  | Baseline缺陷 | SLINFER方案 | 效果 |
  |-------------|-----------|------|
  | GPU独占每instance仅用23% memory → 大量over-provisioning | 异构硬件抽象+弹性资源共享（CPU独立serve+GPU多model共享） | 64×7B: 0.9 GPU vs 3.2 GPU (sllm)，serving capacity +86-154% |
  | 无token-level调度 → SLO violation不可控 | Headroom-driven token-level scheduling + shadow validation | 128 models下SLO-met rate显著高于baseline，TTFT sub-second CDF |
  | 静态KV-cache分配 → memory resizing overhead大且无协调 | Watermark-based scaling (w=25%) + optimistic budget/pessimistic reservation | Scaling overhead从11.3%降至1.4%，无OOM |
  | 碎片化instance → 重复weight loading + 小batch | Proactive preemption + reactive bin-packing consolidation | Batch size +74% vs sllm, decode throughput +0-88% on GPU |
  | CPU完全闲置 → 浪费AMX加速能力 | OpenVINO + AMX backend, CPU优先调度, SLO不满足时fallback GPU | CPU可独立serve 7B/13B，3-4 CPU node ≈ 1 GPU node serving capacity |

- baseline方法是什么？
  Baseline是现有的GPU SpMSpV write-back策略：(1) **Atomic write-back**：每个partial product直接用global atomicAdd写入output vector y[ind]。this method在many-to-one scatter pattern下产生严重address contention，uncoalesced memory stores导致带宽利用率极低。在A100上sparsity=0.1时atomic write-back仅270 GB/s (peak 1555 GB/s的17%)，write-back占overall runtime的>30%，stall cycles中>45%是long scoreboard waits。(2) **Sort-based write-back**：buffer所有(row_idx, val) pairs→global sort by row index→sequential reduce duplicates→每个row仅写一次。sort阶段带宽仅~43.3 GB/s，write-back占overall runtime的>70%，需要large temporary buffers。
  
  全栈执行例子（以it-2004 web graph, sparsity=0.1, NVIDIA A100为例）：
  - 算法层：SpMSpV y=A*x，CSC格式矩阵→vector-driven (column-major) paradigm: 对x中每个nonzero遍历A对应column→生成partial products→write-back到y
  - 系统框架层：GPU graph frameworks (Gunrock/GraphBLAST)提供atomic-based SpMSpV kernel，或Adaptive SpMSpV根据matrix statistics在4种column-major kernel+2种row-major kernel间选择
  - 编译框架层：论文未明确说明（手工CUDA kernel，无编译框架自动生成）
  - kernel调度层：BlockAtomic (Gunrock-like): 多short column聚合到一个CTA→CTA内threads各自计算partial products→global atomicAdd写入y。GlobalAtomic: global prefix scan计算total NNZ→均匀分配每个CTA→CTA内threads global atomicAdd。BlockSort (FastSpMSpV-like): CTA内生成(row_idx, val) pairs→sort→reduce→coalesced global write。GlobalSort: global均匀分配→sort-reduce→write。Load balancing策略：Block-mapped按column数分配CTA（skewed分布下poor balance），Global-mapped按NNZ均匀分配CTA（较好balance但prefix-scan overhead）
  - 硬件架构层：NVIDIA A100 GPU。global memory bandwidth 1555 GB/s，L2 cache 40MB，per-SM 168KB shared memory。Atomic units处理global atomicAdd，uncoalesced scatter导致cache line浪费、L2 hit rate低、bandwidth saturation差

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**VDHA (Vector-Driven Hash Aggregation)**：通过shared-memory hash table做local aggregation减少global write conflicts，column decomposition+reordering增强locality，fetch-compute-writeback pipeline隐藏hash overhead。

  论文方法全栈执行例子（以it-2004, sparsity=0.1, A100为例）：
  - 算法层：SpMSpV y=A*x，CSC格式→vector-driven paradigm，hash-based write-back替代atomic/sort write-back
  - 系统框架层：VDHA CUDA kernel可集成到adaptive SpMSpV框架中（配合预测模型在VDHA和baseline间选择，best-of-7 fallback实现1.22× speedup）
  - 编译框架层：论文未明确说明（手工CUDA kernel，使用cp.async异步copy指令、atomicCAS等PTX级操作，无编译框架自动生成）
  - kernel调度层（核心创新）：
    (1) **Shared-memory hash aggregation**（解决baseline缺陷：global atomic scatter→low bandwidth utilization）：
    每个CTA维护2048-entry shared-memory hash table→partial products插入hash table做local accumulation→hash table满时bucket-order flush到global memory。对比baseline：BlockAtomic每个update都global atomic→>30% runtime in write-back with ~270 GB/s bandwidth；VDHA大部分updates在shared memory local aggregation→仅flush和fallback才global write→global atomic conflicts显著减少（atomic-unit utilization从22.99%降至12.82%）。hash table还提供partial ordering：entries按bucket order (0→2047) flush→warp内threads访问相对连续地址→改善memory coalescing（γ从0.744提升至2.607）。
    (2) **Column decomposition with reordering**（解决baseline缺陷：skewed column lengths→workload imbalance + poor hash locality）：
    长列按SPLIT_SIZE=256切分为segments→segments metadata按首row index排序（仅排序segment metadata而非nonzeros本身，O(S log S) cost where S<<N）→cross-column segment overlap增强hash table reuse。对比baseline：Global/Block mapping均不exploit long-column内部缺乏locality的特点→ρ仅51.0% (T=2048, density=100%)→hash table insufficient aggregation；VDHA切分+重排序后ρ提升至89.8%→更多updates在共享内存完成aggregation。local overlap ratio ρ从0.510→0.898，coalescing factor γ从0.744→2.607。
    (3) **Fetch-compute-writeback pipeline**（解决baseline缺陷：high memory latency→warp stall无法被occupancy掩盖）：
    double buffering + cp.async异步fetch下个segment→当前segment做hash aggregation→next segment ready时swap buffer。对比baseline：45%+ stall cycles是long scoreboard waits (pending global memory)→即使高occupancy也无法隐藏latency；VDHA将hash computation叠加到memory fetch latency上→stall ratio从>45%降至~15%→hash computation cost从16.7%降至12.3%。
  - 硬件架构层：NVIDIA A100 GPU。利用shared memory（per-SM 168KB）做hash table显式scratchpad（vs L1 cache implicit caching），2048-entry hash table=16KB per CTA→8 CTAs/SM→256 threads/CTA平衡occupancy。cp.async (Ampere+)实现asynchronous global→shared copy。atomicCAS支持intra-CTA shared memory atomic操作。FALLBACK_ITER机制控制linear probing worst-case latency避免warp divergence。

  Baseline缺陷→VDHA方案映射：
  | Baseline缺陷 | VDHA方案 | 效果 |
  |-------------|---------|------|
  | Global atomic scatter→low bandwidth (~270 GB/s) | Shared-memory hash local aggregation + coalesced flush | Atomic-unit utilization ↓ 22.99%→12.82% |
  | Skewed column lengths→imbalance + poor hash locality | Column split (SPLIT_SIZE=256) + segment reorder by first row index | Local overlap ρ ↑ 0.510→0.898, γ ↑ 0.744→2.607 |
  | Memory stall dominates (>45% long scoreboard) | cp.async double-buffering pipeline: fetch↔compute overlap | Stall ratio ↓ >45%→~15%, hash cost ↓ 16.7%→12.3% |
  | Sort-based→O(N log N) sort overhead (~43 GB/s sort bandwidth) | Hash aggregation O(N) with hash table (O(1) amortized insertion) | No global sort needed, hash computation mostly hidden |
  | No basis for kernel selection→suboptimal choice | Decision tree predictor (5 features, 91.3% accuracy) → fallback to best-of-7 | Adaptive geomean speedup 1.13×→1.22× on SuiteSparse |
