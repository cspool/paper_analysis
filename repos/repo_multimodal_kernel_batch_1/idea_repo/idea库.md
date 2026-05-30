## DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training

- baseline方法是什么？
  Baseline采用Vanilla 3D Full Attention（基于FlashAttention-2实现），对视频latent space中所有token进行完整的全对全attention计算。全栈执行例子（2.7B video DiT, latent 32×32×32 = 32K tokens, 32 GPUs, 数据并行）：
  
  - **算法Pipeline层**：Flow matching训练范式，每步对完整latent tokens做3D full self-attention（计算QK^T矩阵→softmax→乘V），复杂度O(S²d)。单个attention head的QK^T矩阵大小为32K×32K=1B个元素，16个heads共16B个元素。
  - **系统框架层**：PyTorch FSDP做数据并行（DP=8），Context Parallelism（CP=4）切分序列到4个GPU。使用head-wise CP优先（通信性能更优）：每GPU初始持有[S/4, 16 heads]的QKV chunks→All-to-All重分布→每GPU持有完整[S, 4 heads]→独立计算attention→All-to-All恢复布局。FlashAttention-2做I/O-aware融合计算。
  - **编译框架层**：论文未明确说明。使用PyTorch原生编译路径，Triton kernel通过`torch.compile`或直接调用。
  - **Kernel调度层**：FlashAttention-2 kernel在H100 GPU上执行标准tiled attention——将QKV分块加载到SRAM→on-chip softmax计算→写回HBM。每个token的query需访问全部32K个KV对，计算量O(S²)。forward pass中92%-93%的时间消耗在self-attention（200K token长度时）。
  - **硬件架构层**：8×H100 SXM节点，900 GB/s NVLink互联，InfiniBand跨节点。HBM容量80GB/GPU。Tensor cores执行FP16/BF16矩阵乘法。

  Baseline缺陷：
  (1) **计算浪费**：attention score呈power-law分布，top 10% KV pairs贡献>90% attention scores，但baseline对所有KV对均做完整计算，大量低score的KV对计算纯属浪费。
  (2) **无法利用动态稀疏性**：attention稀疏度随训练进行而加剧（从50K iter的0.81 median sparsity到300K iter的0.92），baseline无法利用此演变。
  (3) **固定窗口方法的局限**：Window Attention（WA）虽能减少计算但使用固定局部窗口，而DSV发现critical KV pairs在视频DiT中并无局部性模式（仅15.1%在5-token半径内），导致WA-M无法收敛、WA-L性能有限（仅约70%固定sparsity）。
  (4) **通信冗余**：标准CP传输所有KV pairs，在稀疏场景下大量非critical KV被无意义传输。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  DSV通过三个协同组件解决baseline缺陷。全栈执行例子（同配置2.7B model, 32K tokens, 32 GPUs）：

  **(1) Two-Stage Training with Sparsity Predictors**（对应缺陷1/2/3）：
  Stage 1训练低秩预测器（W_Q^lr, W_K^lr with d_lr≪d_k）学习approximate attention score distribution。每个attention module仅增加<10M参数（3B model）。预测器不依赖局部性假设，通过CosLoss+NormLoss直接学习QK^T的相对分布，因此不受critical KV无局部性模式的限制（Observation 2）。Stage 2激活稀疏计算后，仅对预测器识别的critical KV pairs（贡献>90% attention score）进行完整attention计算，将计算量从O(S²)降至O(S·K)，K≈0.1S at 90% sparsity。

  **(2) Fused Critical KV Estimation + Query Grouping Kernels**（对应缺陷1/4的核心实现）：
  - Fused kernel将Q_lr·K_lr^T矩阵乘法与top-K选择融合，避免物化完整[S,S] attention score矩阵（288GB for H=16, S=300K），将空间复杂度从O(S²)降至O(SK)。
  - Query grouping利用相邻tokens critical KV重叠率>92.4%（Observation 5），按3D voxel（如2×2×2）分组共享critical KV indices，减少estimation overhead和提升memory coalescing。
  最终实现forward 2.2-5.7× speedup、backward 3.3-4.0× speedup（相对FlashAttention-2）。

  **(3) Hybrid Sparsity-Aware Context Parallelism**（对应缺陷4）：
  - 建模HCP和SCP在稀疏场景的trade-off：HCP需解决head-wise sparsity heterogeneity导致的load imbalance（最长处理时间算法优化head分配），SCP仅传输critical KV（通信量∝(1-α)·σ）。
  - 形式化为min-max优化问题后用Gurobi求解最优HCP/SCP group size组合，HCP优先节点内（高带宽All-to-All），SCP用于跨节点（低通信量）。
  - 端到端实现2.73-3.02×训练吞吐加速（32 GPUs, up to 130K tokens）和2.41-2.93×加速（64 GPUs, up to 260K tokens），同时FVD和VBench指标与full attention持平。

  全栈对比：
  | 层次 | Baseline (FA) | DSV |
  |------|--------------|-----|
  | 算法Pipeline | Flow matching + full self-attention O(S²d) | Flow matching + 预测稀疏attention O(SKd), K≪S |
  | 系统框架 | FSDP + Head-wise CP | FSDP + Hybrid HCP/SCP (Gurobi优化group sizes) |
  | 编译框架 | PyTorch eager / torch.compile | 论文未明确说明 |
  | Kernel调度 | FlashAttention-2 fused tiled attention | Fused MM+TopK (CUDA cores, Bitonic Select) + Sparse Attention with Query Grouping (Triton) |
  | 硬件架构 | H100, NVLink 900GB/s, InfiniBand | 同上，额外CPU offloading for KV indices |

## Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management

- baseline方法是什么？
  已有三种主要的实时GPU管理/分析方法，均因对NVIDIA GPU硬件调度行为的理解不完整而存在结构性缺陷：

  **(1) Management-Free Analysis（Yang et al. [4]）**：通过限制编程模型（每job仅一个kernel、单CUDA stream）来简化分析，给出不需要GPU管理中间件的响应时间bound。基于Amert et al. [5]的简化调度规则：(i) kernel launch时入EE FIFO队列；(ii) head-of-queue kernel被fully dispatched后dequeue；(iii) block在资源满足时eligible for assignment。全栈执行例子（10个OpenVX DAG任务，每任务含1个kernel in 1 stream, NVIDIA Jetson GPU）：
  - **算法层**：OpenVX vision graph → 每节点编译为单个CUDA kernel → 无跨kernel依赖的任务模型。论文未明确说明具体算法层修改。
  - **系统框架层**：无GPU管理中间件——task直接通过CUDA API launch kernel到各自的stream，依赖GPU原生调度。OpenVX runtime负责任务间依赖管理。
  - **编译框架层**：论文未明确说明。OpenVX graph编译器→CUDA kernel，无编译器级优化。
  - **kernel调度层**：Kernel通过CUDA stream提交到GPU → GPU HW scheduler按channel→runlist→engine dispatch chain处理 → Yang et al.假设所有kernel按FIFO顺序dequeued from EE queue → 实际上当stream数>channel数时（R2），后续kernel的channel assignment非FIFO（Corollary 2），导致later-released kernel可cut-ahead先执行。
  - **硬件架构层**：NVIDIA Jetson TX2/Xavier等嵌入式GPU。默认每个context仅2-4个compute channel（R2）——无法同时支持超此数目的独立stream。Yang et al.的分析未限制co-running job数量。

  Baseline缺陷：当co-running job数（即stream数）> GPU compute channel数时，(i) 额外stream产生false dependency（R2/Corollary 1），kernel需等任意channel释放而非FIFO调度；(ii) channel assignment非FIFO（Corollary 2），later-released kernel可能先于earlier-released kernel执行 → 响应时间bound proof（Theorem 3）假设的FIFO dequeue顺序被打破 → 未计入的额外delay导致deadline miss。

  **(2) Preemptive EDF via Runlist Management（Capodieci et al. [3]）**：通过只将一个应用的channel放在runlist上实现任务调度，通过重置runlist实现抢占。基于TX2特有属性：仅一个runlist，并错误地将此概括为所有NVIDIA GPU的通用规则。全栈执行例子（2 GPU-using tasks on Jetson Xavier, Task1=compute+copy, Task2=compute-only, EDF调度）：
  - **算法层**：EDF（Earliest Deadline First）调度算法 → 根据absolute deadline确定优先级。Task1: period=3s, deadline=2s, cost=2s；Task2: period=3s, deadline=3s, cost=1s。
  - **系统框架层**：Capodieci et al.的runlist管理机制——CPU端维护task队列，运行时按EDF选择最高优先级task将其channel插入runlist → 抢占通过重置runlist实现。假设仅一个runlist存在（因此抢占compute runlist后task不再执行任何GPU工作）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：Runlist管理假设抢占compute runlist → task停止所有GPU活动 → 但Jetson Xavier有第二个runlist（copy-only）（R5）→ Task1的copy操作在第二个runlist上不受抢占影响继续执行 → copy操作干扰Task2的compute（已在Jetson Xavier上被Olmedo et al. [7]证明严重延迟compute）。
  - **硬件架构层**：NVIDIA Jetson Xavier（Volta embedded）——含有至少2个独立runlist（R5, R7），而非Capodieci et al.假设的1个。

  Baseline缺陷：(i) 将TX2的单runlist配置错误泛化为所有GPU的通用规则；(ii) 未考虑其他非compute engine（copy engine等）需要独立runlist访问（R7）；(iii) 抢占仅作用于一个runlist，其他runlist上task的未完成操作继续执行并干扰被抢占到的task → 响应时间bound失效、deadline miss。

  **(3) Granular GPU Locking（Elliott et al./GPUSync [2, 14]）**：将每个compute engine和copy engine视为独立可锁资源，应用mutual-exclusion lock，使已有资源管理分析可直接应用于GPU-using task。全栈执行例子（RTX 6000 Ada, 2 GPU-using tasks: Task1=CPU→GPU copy+graphics, Task2=GPU→CPU copy）：
  - **算法层**：实时资源管理analysis（如suspension-based locking protocol）→ treat GPU engines as lockable resources。
  - **系统框架层**：GPUSync框架——每个engine对应一个lock → task需acquire对应engine的lock才能使用 → release后下一task可acquire → 假设不同方向copy engine（CPU→GPU vs GPU→CPU）为独立资源→创建Lock1(CPU→GPU copy), Lock2(GPU→CPU copy), Lock3(compute)。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：Lock1和Lock2被视为非竞争资源 → Task1和Task2可同时分别持有Lock1和Lock2并行执行 → 但实际上，RTX 6000 Ada的GRCE0和GRCE1均映射到同一LCE→同一PCE（R8）→ 两task的copy操作竞争同一个物理copy engine→ copy时间翻倍。
  - **硬件架构层**：NVIDIA RTX 6000 Ada——CUDA报告2个copy engine但底层LCE→PCE映射（Fig.11）显示GRCE共享PCE配置 → 实际只有1个独立的物理copy path可用。

  Baseline缺陷：(i) 假设多个copy engine提供独立copy path（R8推翻——GRCE通过共享PCE破坏独立性）；(ii) 假设copy engine和compute engine总是独立调度（R6推翻——可共享runlist）；(iii) 未考虑硬件LCE→PCE间接映射层（需通过nvdebug的lce_for_pce和shared_lce_for_grce接口检查实际配置）→ execution time bound被打破→ unreliably-met deadlines。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法是通过**深度硬件逆向工程**实验性推导出**8条NVIDIA GPU调度规则**（R1-R8），覆盖从GPU library到on-engine-dispatch的全链路调度行为。不提出新的GPU管理方法，而是提供所有safe GPU管理方法必须遵守的硬件调度ground truth。

  核心贡献对应关系：

  **(a) 实验方法学：组合硬件内省+微基准测试** → 解决baseline对GPU硬件行为的不完整/错误假设
  - **nvdebug内核模块**：绕过GPU驱动，通过MMIO直接读写GPU寄存器，解析GPU页表访问runlist物理内存。暴露PTOP寄存器（engine-runlist拓扑）、PCE-LCE映射寄存器、channel状态等此前仅存在于NVIDIA内部文档的硬件信息。
  - **gpu-microbench（exec_logger + copy_monitor）**：精密计时（微秒级）的定向微基准测试，精确测量特定engine的调度时间线。通过组合测试（如同时运行compute-only和copy-only任务）交叉验证硬件行为。
  - **跨9 GPU验证**：在Pascal到Ada Lovelace的5代架构、9款GPU（含嵌入式Tegra和离散GPU）上重复所有实验，确保规则的普适性。

  **(b) 规则体系：3层调度规则→纠正baseline假设**
  - **Channel层规则（R1-R2）**→ 纠正Yang et al. [4]的假设：
    - R1: 所有GPU操作必须经过channel → 提出disable_channel验证方法
    - R2: Channel数限制并行度（x86_64默认8, Jetson默认2-4）→ Fig.5实验证明了超过channel数时false dependency和non-FIFO channel assignment（Corollary 1, 2）→ 在Evaluation §VI-A中给出违反R2的counter-example：10 jobs → 8个先执行 → kernels 9和10竞争释放的channel → 非FIFO → Yang et al.的response-time bound proof被打破

  - **Runlist层规则（R3-R5）**→ 纠正Capodieci et al. [3]的假设：
    - R3: Channel必须在runlist中 → 与Capodieci et al.一致
    - R4: 每runlist最多一个task per engine active → 既支持又挑战Capodieci et al.——单engine task互斥（正确），但多engine task可在单runlist上co-run（他们未考虑）
    - R5: Runlist数量限制独立inter-task并行度 → 多runlist支持独立调度（Fig.8），单runlist导致跨引擎干扰（Fig.9, Jetson TX2）→ 在Evaluation §VI-B中给出counter-example：Jetson Xavier有2个runlist → 抢占compute runlist不能阻止copy在另一个runlist上继续 → 被抢占task的copy干扰EDF高优先级task的compute → deadline miss

  - **Engine映射层规则（R6-R8）**→ 纠正Elliott et al./GPUSync [2, 14]的假设：
    - R6: 一个runlist可绑定多engine → copy和compute可能共享runlist（Table II-IV都显示Runlist 0同时包含Compute和GRCE）→ 非独立的copy和compute调度
    - R7: 每个engine只绑定一个runlist → PTOP寄存器约束 → 确保引擎不会在多个runlist上被重复调度
    - R8: LCE→PCE硬件间接映射层（GRCE可共享PCE）→ Fig.10实验（RTX 6000 Ada上OpenGL texture upload使CUDA GPU→CPU copy减速2×）+ Fig.11 LCE/PCE/GRCE映射图 → 在Evaluation §VI-C中给出counter-example：GPUSync为2个copy engine创建2个lock → 但实际GRCE共享PCE → 一个物理PCE被竞争 → copy时间翻倍

  **(c) 工具开源** → 使其他研究者可以验证和扩展规则
  - nvdebug开源（http://rtsrv.cs.unc.edu/cgit/cgit.cgi/nvdebug.git/）：支持Kepler(2011)至Ada(2022)全系列GPU，同时支持x86_64和aarch64，零依赖，无需配置
  - gpu-microbench开源（https://www.cs.unc.edu/~jbakita/rtas24-ae/）：经多年专家级调优和bug修复的微基准测试套件

  全栈执行例子（基于论文8条规则的correct GPU management, Jetson Xavier, 2 tasks, EDF）：
  - **算法层**：同baseline EDF调度算法 → 但调度决策基于正确硬件模型
  - **系统框架层**：改进的runlist管理——不仅管理compute runlist（对应Compute/Graphics engine），也必须管理copy runlist（对应Copy Engine）。抢占时需要同时重置所有相关runlist，确保被抢占task的所有engine操作停止
  - **编译框架层**：论文未明确说明
  - **kernel调度层**：基于R5→在Jetson Xavier上，compute和copy的独立runlist意味着抢占compute runlist后copy继续 → 正确的管理必须同时preempt copy runlist → 消除copy对compute的干扰
  - **硬件架构层**：通过nvdebug的device_info获取正确的engine-runlist拓扑→根据R7（每engine单runlist）和R8（检查GRCE→PCE映射）确定引擎独立性→创建正确数量的lock（而非盲目按CUDA报告的copy engine数创建lock）

## Characterizing Concurrency Mechanisms for NVIDIA GPUs under Deep Learning Workloads

- baseline方法是什么？
  Baseline是三种NVIDIA GPU现有的application-level并发机制，各自存在结构性缺陷：

  **(1) Priority Streams**：两task在同一进程的不同CUDA stream上运行，stream可设三级优先级（-2/1/0），thread block scheduler优先从高优先级stream取blocks。但**不抢占已执行blocks**。全栈执行例子：
  - **算法层**：PyTorch/TensorFlow模型编译为CUDA kernel序列（convolution、SGEMM、batch norm等），inference请求按MLPerf server mode（Poisson arrival）或single-stream mode（连续）到达。
  - **系统框架层**：CUDA runtime创建高优先级stream（inference）和低优先级stream（training），同一进程内异步提交kernel。无Serving框架，直接CUDA API。
  - **编译框架层**：论文未明确说明。使用PyTorch/TensorFlow默认compilation pipeline（torch.compile或TF XLA，论文中未描述具体编译过程）。
  - **kernel调度层**：Thread block scheduler采用leftover policy（最近到达kernel的所有blocks优先调度完）和most-room placement（选剩余资源最多的SM放置blocks）。Warp scheduler可能使用greedy-then-oldest或loose round-robin，且官方文档未说明priority streams如何与warp scheduling交互——可能导致warp scheduler实际"去优先级化"高优先级blocks。
  - **硬件架构层**：NVIDIA GeForce RTX 3090（Ampere），82 SMs，每SM 4 warp scheduler单元（每两周期发射一条warp指令），fixed resources per SM。
  - **Baseline缺陷**：Compounded delay——inference kernel完成后、下一inference kernel到达前的窗口期，training kernel抢占GPU所有SM。下一inference kernel到达后必须等待已执行training blocks完成，造成2-4× turnarround time增加且variance大。

  **(2) Time-Slicing**：两task独立进程，CUDA application-level scheduler以约2ms固定时间片轮转分配整个GPU。**不支持spatial sharing**。全栈执行例子：
  - **kernel调度层**：GPU交替专属于单进程。时间片切换约145μs开销。Blocks可被coarse-grained抢占（整个GPU清空），但无partial preemption。Register和shared memory不传输（推测为避免高开销），导致两进程资源需求总和不能超硬件上限。
  - **Baseline缺陷**：无法spatial sharing——资源在时间片内空闲时无法被另一进程使用。Utilization最差（training time可比baseline多100+秒）。Memory transfer contention跨进程干扰（如ResNet-34 case）。时间片长度/频率不可配置。

  **(3) MPS**：MPS server调度不同CUDA context的kernels，允许SM-level spatial sharing（blocks colocation）。但**无优先级**且采用FCFS+leftover policy。全栈执行例子：
  - **kernel调度层**：MPS server接受多client kernel dispatch请求，FCFS顺序处理。Leftover policy导致后到达kernel需等待当前kernel所有blocks调度完。Per-client thread limit可设（实验中100%）。
  - **Baseline缺陷**：无优先级概念——inference和training task被同等对待。Load-balancing behavior使得training task受益多于inference task（inference degradation显著）。Compounded delay在100% thread limit时也影响MPS。大kernel（grid size超SM容量）长时间独占GPU时，inference被starved。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法是**Fine-Grained Block-Level Preemption**（细粒度thread block抢占），即thread block scheduler能在任意时刻中断任意subset of thread blocks的执行，并在之后恢复。论文未实现此机制（需要NVIDIA硬件修改），但通过10个Observations系统论证了其必要性和可行性。解决Baseline缺陷的机制：

  **解决Priority Streams的compound delay**：Inference kernel到达时立即抢占部分training blocks腾出空间，消除等待。全栈执行例子对比如下：
  - **系统框架层**：仍使用CUDA streams + priority，但thread block scheduler支持fine-grained preemption。论文未明确说明用户态API设计。
  - **kernel调度层（核心变化）**：Thread block scheduler在inference kernel到达时：(i) 选择被抢占的training blocks（基于contention-aware placement policy）；(ii) 保存被抢占blocks的context（register file、shared memory、warp state）到global memory；(iii) 将inference blocks调度到腾出的SM空间；(iv) inference kernel完成后恢复training blocks。抢占开销估算：per-SM约448KB context（128KB L1/shared + 256KB register file + 64KB constant），按11.4 GB/s per-SM带宽约37μs；或基于time-slicing实测（145μs/2切换=73μs per save）。开销可被隐藏（见下）。
  - **硬件架构层**：需在SM内增加context save/restore hardware state machine。论文建议复用现有time-slicing context-switching硬件。

  **解决MPS的FCFS无优先级问题**：Fine-grained preemption + MPS可实现"minimum resource guarantee" + "priority over-alloc"——为inference设定最小资源预留，training用剩余资源。Inference kernel到达时若资源不足，抢占training blocks达到预留阈值。

  **抢占开销隐藏策略（论文O8-O9核心贡献）**：
  (a) **利用memory transfer latency**：Preemption state save可在host-to-device memory transfer期间并行执行（DMA引擎独立于SM计算）。
  (b) **利用kernel序列特性**：当已知大kernel紧跟小kernel时，在小kernel执行期间预抢占training blocks为后继大kernel腾空间。举例（ResNet-152 Figure 8 Region B）：第一个kernel仅32 blocks × 64 threads（只占32 SM），执行时间137μs，后续512-block kernel仅需2μs。137μs足够完成抢占操作（~73μs）。
  (c) **保持空间开放**：小kernel完成后不立即用training blocks填充，直接留给下一kernel（如Region A：136-block kernel 400μs → 112-block kernel 6μs，没必要在6μs kernel执行前后做抢占）。

  **Utilization度量优化（O10）**：提出用best-effort training task execution time作为utilization proxy（优于简单的thread occupancy或SM occupancy metrics），因为同是100% thread usage的两个kernel，实际register/shared memory利用率可差异很大（如49152 vs 61440 registers per SM）。

## A Survey of Resource-efficient LLM and Multimodal Foundation Models

- baseline方法是什么？
  本文为综述论文，不提出新的baseline或方法，而是对已有资源高效大型基础模型研究的系统性分类与梳理。Baseline即综述前文献中各独立研究点的分散状态——各方向（高效架构、训练算法、推理算法、模型压缩、分布式系统、端侧部署）彼此独立发表，缺乏统一的分类体系和纵向联系。论文识别的核心痛点：(i) 模型规模持续增长（scaling law），训练LLaMA-2-70B需1.7×10^6 GPU hours，碳排放291吨CO2；(ii) 资源需求集中在计算、存储、带宽、能耗；(iii) 资源壁垒阻碍模型民主化，仅少数巨头可训练部署SOTA模型；(iv) 各子方向成果分散，缺乏全栈视角。

  全栈执行例子（以LLaMA-7B推理为例，综述整合的全栈视角）：
  - **模型推理算法层**：自回归解码 + KV Cache（每个token需O(Td + d²)计算，KV cache占用2×B×S×D×L×4 bytes）；可通过Speculative Decoding加速2-3×、稀疏注意力降至O(Td)、量化降至4bit。
  - **系统框架层**：vLLM PagedAttention按需分配KV cache（block级管理消除碎片）、Orca迭代级批处理、SARATHI chunked-prefill与decode混合调度。
  - **编译框架层**：FlashAttention/CUDA kernel手写（nvcc编译）、MLC-LLM通过TVM编译加速多平台部署。论文未详细覆盖此层。
  - **kernel调度层**：FlashAttention-2 IO-aware tiling（HBM→SRAM分块计算避免N×N矩阵）、Flash-Decoding针对小seqlen大batch优化、FlashDecoding++优化softmax+flat GEMM。
  - **硬件架构层**：论文明确排除硬件设计（§1: "exclude hardware design"）。
  - **芯片设计层**：论文明确排除（同上）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  本综述的核心贡献是提出一个**多维度的分类框架**，将资源高效基础模型研究按三个维度组织：
  (1) **高效架构**（§3）——注意力机制（稀疏/近似/无注意力）、动态网络（MoE/Early Exiting）、扩散模型优化、ViT优化；
  (2) **高效算法**（§4）——按模型生命周期：预训练→微调→推理→压缩；
  (3) **高效系统**（§5）——分布式训练、联邦学习、云侧Serving、端侧Serving。
  这一分类框架解决了分散研究之间的纵向关联缺失问题，使得从算法到系统到部署的全栈优化路径可被系统性地追踪和对比。

  论文方法全栈执行例子（综述的分类导航能力）：
  - **模型推理算法层**：论文提供各类方法的统一cost分析框架——如用flops-profiler对GPT-2和Stable Diffusion 2.1各模块（Embedding/Attention/FFN/Im_head）的FLOPs和存储开销进行定量分解（图4-7），揭示Attention在长序列时的O(T²D)瓶颈和FFN在D增大时的O(TD²)瓶颈，为优化选择提供决策依据。
  - **系统框架层**：表5提供17个主流开源训练/推理框架的统一对比（DeepSpeed、Megatron-LM、vLLM、llama.cpp、MLC-LLM等），按Cloud/Edge/Training/Inference维度分类，为工程选型提供指导。
  - **编译框架层**：论文在该层覆盖有限，主要提及MLC-LLM的编译器加速部署方案。
  - **kernel调度层**：整合FlashAttention家族（FlashAttention/FA-2/Flash-Decoding/FlashDecoding++）的发展脉络和各自适用场景，明确prefill vs decode的不同kernel优化策略。
  - **硬件架构层**：论文明确排除（引用[192,185]为已有综述覆盖）。
  - **芯片设计层**：论文明确排除（同上）。

  对比baseline（分散研究），论文方法的独特价值在于：(i) 首次将LLM、ViT、扩散模型、多模态模型的资源高效技术统一在同一框架下；(ii) 跨越算法到系统到部署的全生命周期；(iii) 提供定量cost分析（flops-profiler）辅助技术选型；(iv) 指出6个未来方向（cloud-edge hybrid、model sparsity、FMaaS、agent optimization、privacy-preserving FM、scaling law understanding）。

## HLX: A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models

- baseline方法是什么？
  **GPU 上运行 Hybrid Transformer-Mamba 模型的 CUDA 优化 kernel（FA-2 + unfused SSD）**：Hybrid 模型在 GPU（A100/H100）上通过交替执行注意力层和 Mamba-2 层进行推理。注意力层使用 FlashAttention-2（FA-2）kernel——将 QK^T + softmax + PV 融合为单 kernel，按 block tiling 减少 DRAM 访问，沿 sequence length 维度并行处理 Q block。Mamba-2 使用 5 个分离的 SSD kernel（chunk cumsum → chunk state → state passing → BMM chunk → chunk scan），每个 kernel 独立执行后中间数据通过 DRAM 传递。虽然 H100 上的 FA-3 通过异步 warp specialization 实现 2-stage 流水线来重叠数据搬运和计算，但仍然受限于 pipeline-agnostic hardware 和 register pressure。SSD 因大量 element-wise 操作和 Einsum 导致内存密集型特征，中间数据不重用。

  全栈执行例子（Hybrid-2.7B, seqlen=128K, A100 GPU）：
  - **模型推理算法层**：Hybrid 模型交替 6 层 attention + 58 层 Mamba-2。Attention: QK^T → softmax → PV → O。Mamba-2: input projection → conv1D → SiLU → SSD (dt/A/x/B/C → state equations → Y) → z-gating → output projection。
  - **系统框架层**：PyTorch + CUDA。FA-2 kernel 通过 torch 调用，SSD 的 5 kernel 通过 PyTorch 逐个 launch。CPU-GPU kernel launch overhead 和 DRAM 中间数据传输开销存在。
  - **编译框架层**：无编译器参与。FA-2 为手写 CUDA kernel（nvcc 编译），SSD 各 kernel 基于 PyTorch 的 Einsum 操作。
  - **kernel调度层**：FA-2: Q block 间无依赖，沿 seq_len 并行，每 block 内同步顺序执行 QK^T → local softmax → PV → update O。非 MatMul（softmax, update O）无法与 MatMul 重叠，compute utilization 饱和于 61%。FA-3 on H100: 2-stage warp-specialized 异步流水线，但 register pressure 限制效果，utilization 约 61%。SSD: 5 kernel 串行，中间数据经 DRAM 传递，Einsum 多维张量操作 memory-bound，utilization 仅 26.9%（A100）/ 38%（H100）。Fused SSD 即使实现（SSD-fr），因中间数据 642KB/block 超 SM 寄存器/共享内存容量（A100 256KB RF + 164KB SMEM, H100 256KB RF + 224KB SMEM）导致 register spilling 和 occupancy 下降，延迟反而恶化 1.74×。
  - **硬件架构层**：Nvidia A100/H100 GPU。SM 内 SIMT 模型要求 warp 执行统一指令，warp-specialized pipeline 引入的异构性（producer/consumer warp 不同资源需求）导致调度开销和资源竞争。H100 TMA 针对粗粒度 tile 移动优化，对细粒度 streaming/gather 内存访问模式支持不足。

  Baseline 缺陷：
  - (a) FA-2 同步执行限制：非 MatMul（softmax, update O）与 MatMul（QK^T, PV）间存在依赖，无法 overlap，compute utilization 饱和于 ~61%（A100）/ ~49%（H100）。FA-3 虽有异步改善但仍饱和于 ~61%
  - (b) SSD 极低 compute utilization：大量 element-wise 操作 + Einsum 多维张量运算 + 中间数据无立即重用 → memory-bound，utilization 仅 26.9%（A100）/ 38%（H100）
  - (c) Fused SSD 不可行：虽然融合减少 DRAM 流量，但中间数据量（642KB/block）是 FA-2（321KB）的 2×，超出 GPU SM 内存容量 → register spilling → occupancy 下降 → 性能退化
  - (d) GPU 架构不支持细粒度流水线：SIMT 模型假设统一 warp 执行，warp-specialized pipeline 异构性导致调度开销；TMA 不支持细粒度内存访问模式

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **HLX：统一流水线加速器架构**，通过 PipeFlash（FA-2 细粒度流水线数据流）+ PipeSSD（SSD 融合三阶段流水线数据流）+ URSC 统一硬件架构实现高 compute utilization。

  对应关系：
  - (a) → **PipeFlash 细粒度流水线**：将 FA-2 块级计算改为每次处理 Q block 内 2 行的粒度，QK^T (DPE#0) → local softmax (RVPE) → PV (DPE#1) → update O (UpE) 四阶段流水线并发执行，非 MatMul（softmax, update O）延迟被 MatMul（QK^T, PV）完全隐藏。score/probability 矩阵仅 1KB（vs FA-2 128KB），中间数据减少 4.8×。compute utilization 达 97.5%@128K。
  - (b) → **PipeSSD 融合三阶段流水线**：将 SSD 5 kernel 融合为单 kernel 三阶段流水线：预处理（dA-related in RVPE）→ Y_Diag（CB^T in DPE#0 → CB^TLdt in RVPE → Y_Diag MatMul in DPE#1）→ Y_Off∥states_N + Y_Final∥update states（dBdt^T/dC_Off in RVPE → Y_Off MatMul in DPE#0 ∥ states_N MatMul in DPE#1 → add+update in UpE）。Y_Off 和 states_N 的计算因无依赖而并发执行。DRAM 流量减少 6.8×，中间数据 642KB→58.5KB（11×），compute utilization 达 78.4%。
  - (c) → **URSC 硬件直连数据转发**：DPE→RVPE→DPE→UpE 之间通过 NoC 直接转发数据，不需要经过 DRAM 或大容量 SRAM 暂存。58.5KB 中间数据可全部片上存储（HLX^60 仅需 30.4MB on-chip SRAM，3.4× 小于 H100）。无 register spilling 问题。
  - (d) → **URSC 异构引擎专用流水线**：每个引擎（DPE/RVPE/UpE）为专用硬件而非 warp 通用处理器，天然支持异构流水线阶段。流水线平衡策略通过控制每引擎处理行数（而非固定 warp 资源分配）实现：当 block_size=d_head=d_state 时可达近 100% utilization，不同维度时通过调整行数最小化 inefficiency。

  全栈执行例子（HLX^60, Hybrid-2.7B, seqlen=128K）：
  - **模型推理算法层**：与 baseline 相同的 Hybrid 模型计算图。Attention 层 → PipeFlash 数据流映射；Mamba-2 层 → PipeSSD 数据流映射。
  - **系统框架层**：无传统 CPU-GPU 框架。Top controller 解析 Hybrid 模型计算图，将每层 dispatch 到 URSC，配置 DPE/RVPE/UpE 的操作模式和数据流路径。
  - **编译框架层**：论文未明确说明。推测为离线将 Hybrid 模型层映射为 URSC 配置序列（PipeFlash 模式：DPE#0 MatMul → RVPE softmax mode → DPE#1 MatMul → UpE update O mode；PipeSSD 模式：RVPE pre-processing mode → DPE#0 MatMul → RVPE element-wise mul mode → DPE#1 MatMul → RVPE dBdt/dC_Off mode → DPE#0/DPE#1 MatMul → UpE Y_Final+update states mode）。
  - **kernel调度层**：PipeFlash 流水线——DPE#0 执行 QK^T（⌈128/16⌉×⌈(128×256)/256⌉ cycles），同时 RVPE 处理上一行 softmax，DPE#1 计算上一行 PV，UpE 更新上一行 O。数据流：Q,K,V 从 GS 广播至 DPE#0；score 从 DPE#0 → RVPE（行级转发）；probability 从 RVPE → DPE#1 + UpE（用于 rescale）；PV 从 DPE#1 → UpE（累加）。PipeSSD 流水线——第 1 阶段 RVPE 完成 dA 预处理 → 第 2 阶段 DPE#0 算 CB^T → RVPE 算 CB^TLdt → DPE#1 算 Y_Diag（存 GS）；第 3 阶段 RVPE 同时算 dBdt^T 和 dC_Off → 通过 mux/demux 切换数据方向：dC_Off→DPE#0 算 Y_Off（×states_(i-1)），dBdt^T→DPE#1 算 states_N（×x）→ UpE 算 Y_Final（Y_Diag+Y_Off）∥ update states（states_(i-1)×exp(dA_CS[-1])+states_N）。最终 Y_Final 和 states 存 OMEM→DRAM。
  - **硬件架构层**：HLX^60 = 60 个 URSC + 30.4MB GS + HBM2E 2000 GB/s DRAM + NoC。每个 URSC 含 DPE#0（32 lanes × 8 DPU × 16 FP16 MAC）+ RVPE（2 RVPU + VMEM）+ DPE#1（同 DPE#0）+ UpE（2 UpU + OMEM）。单 core 14nm 面积 7.89mm² / 5.39W，60 core 经缩放至 7nm = 169mm² / 201.8W。vs H100 面积 20.8%，功耗 57.5%，SRAM 29.3%，但 FA-2 compute utilization 从 61% → 97.5%（1.6×），SSD 从 38% → 78.4%（2.06×），端到端加速 2.08×。

- baseline方法是什么？
  **手写 attention kernel（如 FlashAttention-2/3）+ SDPA fallback**：现有 attention 生态系统中，高性能 attention kernel（FlashAttention-2/FAv2、FlashAttention-3/FAv3、FlashDecoding/FAKV）均为手工优化的 CUDA/Triton kernel，仅支持有限的 attention 变体（如 causal mask、sliding window、alibi_bias、soft_cap）。对于不支持的变体（如 prefix_lm、neighborhood attention、soft_cap on FAv3 等），用户被迫 fallback 到 SDPA（PyTorch 原生 scaled_dot_product_attention），SDPA 使用 itemized mask（预计算完整 B×N×N 布尔 mask 矩阵），导致 O(N²) 内存开销和显著的性能退化。每次出现新 attention 变体都需要手写新 kernel，形成"软件抽奖"（software lottery）——变体是否高效取决于是否有恰好匹配的手写 kernel。

  全栈执行例子（FlashAttention-2 causal mask, training, QKV_LEN=16k, head_dim=64, A100）：
  - **模型推理算法层**：Standard scaled dot-product attention with causal mask。公式：O = softmax(QK^T/√d_k + mask)V。FlashAttention 是唯一被支持的 mask 变体；若需 prefix_lm mask，需要 fallback 到 SDPA itemized mask。
  - **系统框架层**：PyTorch SDPA API（torch.nn.functional.scaled_dot_product_attention）根据 backend 优先级自动选择：优先 FAv2 → mem_efficient → cuDNN → math。gpt-fast 和 torchtune 通过 SDPA 调用 attention。对不支持的 mask，框架必须 precompute N×N mask 矩阵作为额外输入。
  - **编译框架层**：无编译器参与。FlashAttention 为手写 CUDA kernel，nvcc 编译。SDPA fallback 使用 cuBLAS GEMM + 手写 softmax kernel。torch.compile 无法自动融合 QK^T + mask + softmax + PV 的 chain（缺少 online softmax 支持，mask 实现需 N×N tensor）。
  - **kernel调度层**：FAv2 kernel 内手写调度——warpgroup 沿 Q_LEN tile 和 KV_LEN tile 两维迭代，手写 online softmax rescaling logic。causal mask 通过手动指定迭代起止索引实现（仅计算上三角）。新增 prefix_lm mask 需要手写新的迭代逻辑，无法复用。
  - **硬件架构层**：Nvidia A100/H100 GPU。FAv2 利用 SRAM 做 tiled computation（online softmax 避免 N×N intermediate 写回 HBM）。SDPA itemized mask fallback 则需从 HBM 加载 N×N mask 矩阵（16k² × 2B = 512MB），完全抵消 flash attention 的内存优势。

  Baseline 缺陷：
  - (a) **灵活性不足（software lottery）**：手写 kernel 仅支持有限变体（FAv2 支持 5/8 种测试变体），不支持的变体 fallback 到 SDPA itemized mask，性能退化 5.49×-8.00×
  - (b) **无法组合（combinatorial explosion）**：每种变体组合（如 sliding window + ALiBI + document mask）需要新的手写 kernel，导致 kernel 种类爆炸
  - (c) **编译器的 failure**：现有 ML 编译器（torch.compile/TVM/Mirage）无法生成 competitive fused attention kernel，因为缺乏 online softmax 支持和双 GEMM fusion（QK^T + PV）
  - (d) **SDPA itemized mask 的内存瓶颈**：预计算 B×N×N 布尔 mask 的内存开销随序列长度二次增长（torchtune 实验中序列长度从 2k 到 8k 时训练吞吐下降 25%）

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FlexAttention：编译器驱动的编程模型 + 模板化 lowering + BlockMask block sparsity**。

  核心设计一（解决缺陷 a,c）：**统一抽象 + 模板化 lowering**。将 attention 变体统一为 score_mod（修改 score 值，如 Alibi bias）和 mask_mod（指定哪些 score 为 -inf，如 causal mask）两个 callable。用户用 PyTorch 编写这两个函数，编译器（TorchDynamo + TorchInductor）自动捕获计算图并翻译为 Triton 代码块，动态注入到手写的 Triton attention kernel 模板（forward/backward/decoding）中。模板内包含 online softmax、GQA 支持、GPU occupancy 管理等经过手工优化的技术，而 score_mod/mask_mod 仅影响 element-wise 的点操作。结果：用户编写 5-10 行 PyTorch 代码即可获得与手写 kernel 竞争的性能（0.68×-1.43× vs FAv2），对不支持的变体获得 5.49×-8.00× speedup vs SDPA fallback。

  核心设计二（解决缺陷 b）：**逻辑融合（Logical Fusion）**。通过 and_mask 和 or_mask 自动组合多个 mask_mod，支持 attention 变体的自由组合（如 PrefixLM = causal_mask OR prefix_mask），无需为每种组合手写新 kernel。

  核心设计三（解决缺陷 d）：**BlockMask block sparsity**。将 score 矩阵按 block（默认 128）分割，编译时通过 torch.vmap 自动评估 mask_mod 生成 block-level sparsity 信息（kv_num_blocks + kv_indices 两个紧凑张量）。区分 Full Blocks（跳过 mask_mod，~15% 额外提升）、Partial Blocks（逐元素 mask_mod）和 Oblivious Blocks（完全跳过）。内存开销 O(⌈N/BS⌉²) vs itemized mask 的 O(N²)。在 torchtune 端到端实验中，BlockMask + document_id tensor of size B×N 替代 SDPA 的 B×N×N mask，从 2k 到 8k 序列长度无性能退化（SDPA 退化 25%）。

  核心设计四（附加创新）：**Paged Attention via BlockMask 转换**。将 page table 的间接内存访问与 BlockMask 的稀疏跳过合并（kv_indices 同时编码物理 KV 位置和稀疏掩码），实现 zero-kernel-change paged attention 支持，overhead <1%（远低于 vLLM 的 20-26% overhead）。

  全栈执行例子（FlexAttention causal mask + document mask, torchtune fine-tuning LLaMa3-8B, H100）：
  - **模型推理算法层**：用户定义 mask_mod = and_masks(causal_mask, document_mask)，其中 causal_mask: q_idx >= kv_idx，document_mask: doc_id[q_idx] == doc_id[kv_idx]。FlexAttention 自动组合二者，公式：O = softmax(QK^T/√d_k + combined_mask)V。
  - **系统框架层**：FlexAttention 替代 SDPA 在 gpt-fast 和 torchtune 中的调用。与 PyTorch 框架原生兼容（支持 CUDA graphs、parameter freezing、kernel fusion 等 torch.compile 优化）。API：flex_attention(query, key, value, block_mask=block_mask)。
  - **编译框架层**：TorchDynamo 捕获 combined_mask 的 PyTorch 图 → TorchInductor 翻译为 Triton 代码块 → 注入手写 Triton forward/backward attention 模板 → 生成最终 fused Triton kernel。同时通过 torch.autograd 自动生成 backward pass 中的 mask_mod 计算。
  - **kernel调度层**：create_block_mask 编译时生成 BlockMask（causal mask 约 50% sparsity + document mask 额外 block-level sparsity）→ SM 沿 Q_LEN tile 并行处理 → 每 SM 通过 kv_indices 间接访问非 oblivious block → Full blocks 跳过 mask_mod，Partial blocks 逐元素执行 → 数据预取 pipeline 隐藏 HBM 延迟。BlockMask 消除了 itemized mask 的 B×N×N 内存开销，用 B×N 的 doc_id 张量替代。
  - **硬件架构层**：Nvidia H100 GPU（功率限制 650W，2.4TB/s 带宽）。BlockMask（~16KB for 16k seq_len, block_size=128）完全驻留 SRAM，替代 itemized mask（512MB at 16k），恢复 flash attention 的 IO 优势。端到端训练 2.4× speedup vs SDPA。

## FlashAttention-T: Towards Fully Tensorized Attention by Exploiting Tensor-Vector Parallelism

- baseline方法是什么？
  **FlashAttention-2（Ampere）和FlashAttention-3（Hopper）fused attention kernels**：将QK^T GEMM（tensor core MMA）、softmax（vector unit）、PV GEMM（tensor core MMA）融合为单个GPU kernel，通过tiling with safe online softmax避免完整N×N attention矩阵的HBM materialization。FlashAttention-2采用sequential scheduling——warpgroup先执行同步QK^T MMA → vector softmax → PV MMA；FlashAttention-3利用Hopper异步WGMMA实现pipelined scheduling——两个warpgroup的GEMM和softmax部分overlap。但两者核心缺陷相同：**softmax计算完全依赖vector unit (CUDA cores)**，而GEMM用tensor unit (Tensor Cores)，由于vector unit吞吐远低于tensor unit，导致"vector interval"——tensor unit在等待vector unit完成softmax期间idle，严重浪费tensor unit算力。

  全栈执行例子（FlashAttention-3 on H100, h=128, s=4096, FP8-FP32）：
  - **模型推理算法层**：Tiled safe online softmax。每个warpgroup处理O的一个row block，在s/b_N个iteration中：QK^T GEMM → rowmax m_i → exp(S-m_i) → rowsum l_i → rescale previous O and l with exp(m_old-m) → P̃V GEMM。Algorithm flow维持逐行maximum以保证数值稳定性。
  - **系统框架层**：PyTorch、vLLM、SGLang等LLM inference/serving框架集成了FlashAttention系列作为默认attention backend。框架通过torch接口调用fused attention kernel。
  - **编译框架层**：论文未明确说明。FlashAttention-2/3为手写CUDA kernel，nvcc编译。Triton fused attention kernel作为compiler-generated baseline比较。
  - **kernel调度层（关键瓶颈）**：FlashAttention-2 sequential scheduling——warpgroup内严格顺序执行：tensor MMA(QK^T) → vector(softmax: max, exp, mul, add, rowsum) → tensor MMA(PV)。tensor unit在vector softmax期间idle（t_vec=924 cycles on A100, 29.8% of iteration）。FlashAttention-3 pipelined scheduling——WG1执行异步QK^T+PV WGMMA后signal WG2，WG2的WGMMA与WG1的vector softmax并行，但未overlap的softmax部分仍产生t_vec=1126 cycles on H100 FP8-FP32（36.3% of iteration）。
  - **硬件架构层**：NVIDIA Ampere A100和Hopper H100 GPU。Tensor Core吞吐远高于CUDA Core——A100 FP16 GEMM 312 TFLOPS vs vector throughput ~16 elements/cycle；H100 FP8 GEMM ~989 TFLOPS。c/k ratio（tensor/vector吞吐比）持续增大（Hopper vs Ampere doubled for FP16, quadrupled for FP8），加剧vector interval bottleneck。

  Baseline缺陷：
  - (a) **Vector interval bottleneck**：softmax完全依赖vector unit，tensor unit idle时间占iteration 29.8%（A100 FP16）至36.3%（H100 FP8），且随硬件tensor/vector吞吐比增大而恶化
  - (b) **计算解耦**：GEMM和softmax分别绑定tensor unit和vector unit，无法利用二者并行能力；所有tensorizable softmax原语（scaling, FMA, row-sum reduction）放在vector unit上浪费了tensor unit吞吐
  - (c) **Softmax time fraction不可忽略**：由T_softmax/(T_softmax+T_GEMM)公式，给定head dim h和c/k ratio，softmax占attention computation的显著比例，不随sequence length改变

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FlashAttention-T：通过repurposing tensor MMA指令 + tensorized online softmax + tensor-vector parallelism scheduling实现向fully tensorized attention的推进**。
  
  核心设计一（解决缺陷a,b）：**Repurpose tensor MMA instructions for softmax primitives**。通过特殊operand value assignment方法，使HMMA.1688.F32.TF32 / HGMMA.64x8x8.F32.TF32指令执行element-wise scaling（D=α·A）、fused multiply-add（D=α·A+C）、row-sum reduction（sum over rows of matrix）——原本仅用于GEMM的tensor MMA指令被改用于softmax计算。关键是repurposed操作直接在GEMM output fragments上执行（同一register space），零copy overhead。例如scaling：设置fragment B含α值pattern，使D(v,t)=α·A(σ(v),t)，swaps开销由minimizing Cayley distance of permutation σ控制。
  
  核心设计二（解决约束）：**Tensorized online softmax with surrogate maximum**。tensor MMA scaling instruction要求scaling factor α对所有行uniform，但original safe online softmax的O rescaling factor exp(m_old[i]-m[i])逐行不同。提出X-row tile surrogate maximum m̂[i]=max over X consecutive rows（X=16 for HMMA, X=64 for HGMMA），使scaling factor在X行内uniform。Numerical safety: (1) m̂≥m → no overflow; (2) all-underflow概率对典型分布asymptotically small; (3) 极端情况fallback to vectorized rescaling。

  核心设计三（解决缺陷c）：**Architecture-aware scheduling for tensor-vector parallelism**。
  - **Ampere ILP**: Horizontal/vertical split将softmax计算分为tensorized（scaling, FMA, row-sum reduction via repurposed MMA）和vectorized（max, exp）部分，在warp内均匀interleave tensor和vector指令，利用ILP使vector指令在tensor指令issue bubble中并行执行。将softmax时间从t_vec压缩至t'_softmax，vector interval降至t'_vec。
  - **Hopper TLP**: 仅tensorize P̃ row-summation（leaf-stage最小化cross-stage register dependency避免WGMMA serialization），将repurposed WGMMA row-sum加入下一iteration的QK^T+PV WGMMA batch，与另一warpgroup的vector S/O rescaling并行，实现跨warpgroup的tensor-vector overlap。Vector interval ratio降至2.7%。

  全栈执行例子（FlashAttention-T ILP on A100, same config, h=128, s=4096, FP16-FP32）：
  - **模型推理算法层**：Tensorized online softmax (Algorithm 1)。16-row surrogate m̂ via warp REDUX → tensorized O rescaling (repurposed HMMA) and S rescaling (repurposed HMMA FMA with const log₂(e)) → vector exp₂ → tensorized P̃ row-sum (repurposed HMMA row-sum reduction)。Softmax primitives（scaling, FMA, row-sum）从vector unit移至tensor unit。
  - **系统框架层**：直接替换FlashAttention-2/3的PyTorch集成，API兼容，generative benchmark（HumanEval, MMLU）功能正确性无退化（ratio ≈ 1.0000），无numerical failures。
  - **编译框架层**：论文未明确说明。手写CUDA kernel修改自FlashAttention-2/3代码库，nvcc编译。
  - **kernel调度层（关键创新）**：
    - Horizontal/vertical split: 将每个warp处理的16-row tile的S和O rescaling操作分为tensorized和vectorized部分
    - ILP interleaving: warp内均匀交叉插入repurposed tensor MMA指令和vector指令，tensor MMA latency被vector指令利用（issue bubble填充）
    - 结果: t'_softmax < t_vec (original all-vector softmax time)，t'_vec = t'_softmax - (t_vec - t'_softmax)（saved部分反映tensor unit被利用），vector interval ratio 1.17-2.18× lower than baseline
  - **硬件架构层**：A100/H100 GPU。Tensor unit吞吐≈16 elements/cycle via repurposed HMMA（与vector throughput持平说明当前all-tensorized scheme不优，需要tensor-vector parallelism）。Future hardware with faster tensor units将使tensorized softmax更有利。

- baseline方法是什么？
  **标准attention实现（Algorithm 0）**：在GPU上计算self-attention $\mathbf{O} = \text{softmax}(\mathbf{QK}^\top)\mathbf{V}$，分三步：(1) 加载Q/K from HBM，计算$\mathbf{S}=\mathbf{QK}^\top \in \mathbb{R}^{N \times N}$，写回HBM；(2) 从HBM读取S，计算$\mathbf{P}=\text{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N}$，写回HBM；(3) 从HBM读取P和V，计算$\mathbf{O}=\mathbf{PV}$，写回HBM。中间矩阵S和P（各$N \times N$）完全materialize在HBM中，导致$O(N^2)$内存消耗和大量HBM访问。大多数操作（softmax、elementwise）是memory-bound。

  全栈执行例子（GPT-2 small, N=1024, d=64, 16 heads, batch=64, A100 GPU）：
  - **模型推理算法层**：标准scaled dot-product attention。Q/K/V通过线性投影得到，计算$\text{softmax}(QK^T/\sqrt{d})V$。训练时前向计算S和P写入HBM，反向再从HBM读取S和P求梯度。
  - **系统框架层**：PyTorch eager execution。`torch.nn.functional.scaled_dot_product_attention`或手写QK^T + softmax + PV。HuggingFace和Megatron-LM等训练框架调用标准attention实现。框架对GPU memory访问无细粒度控制，中间tensor必须完整分配并写入HBM。
  - **编译框架层**：论文未明确说明。cuBLAS提供优化的GEMM kernel（QK^T和PV），但无法跨独立kernel调用进行融合。XLA可融合部分elementwise操作（如masking+softmax），但受限于反向需要中间值。
  - **kernel调度层**：三个独立kernel调用：(1) cuBLAS GEMM: QK^T → write S to HBM（$1024^2$×2B = 2MB/head, 16 heads = 32MB, batch 64 = 2GB）；(2) softmax kernel（含masking/dropout）: HBM read S → compute P → HBM write P（another 2GB）；(3) cuBLAS GEMM: PV → write O to HBM。HBM读写总量：forward约$2Nd + 2N^2$ elements ≈ 35.3GB。反向还需再从HBM读取S和P。
  - **硬件架构层**：NVIDIA A100 GPU。HBM带宽1.5-2.0TB/s vs SRAM带宽~19TB/s（~10× faster）。SRAM仅192KB/SM（108 SMs = ~20MB总量），远小于S+P的~4GB。数据在HBM↔SRAM间频繁搬移，memory bandwidth成为瓶颈。

  Baseline缺陷：
  - (a) **Memory bottleneck**：$O(N^2)$的S和P矩阵完全materialize在HBM，需大量HBM读写。Attention是memory-bound操作，HBM带宽限制wall-clock速度。
  - (b) **不可扩展**：内存$O(N^2)$。N=16K时16 heads × 64 batch × (16K)² ≈ 4.2B elements ≈ 8.4GB仅S+P（FP16），超出单卡A100 40GB。Path-256 (N=64K)完全OOM。
  - (c) **无细粒度内存控制**：PyTorch/TensorFlow高层框架不允许kernel内部的内存访问调度，每个独立kernel强制数据经过HBM往返。
  - (d) **Kernel fusion不足**：即使融合masking+softmax，仍无法解决核心问题——$N \times N$中间矩阵必须保存用于反向传播。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FLASHATTENTION：IO-aware exact attention algorithm with tiling and recomputation**。核心设计：(i) **Tiling with online softmax**——将Q/K/V分块加载到SRAM，逐block计算softmax，通过维护running max $m_i$和running sum $\ell_i$实现增量softmax，永不materialize完整$N \times N$ attention矩阵到HBM；(ii) **Recomputation for backward**——前向仅保存输出O和stats $(m,\ell)$（$O(N)$额外内存，vs $O(N^2)$），反向在SRAM中从Q/K/V块重计算S和P求梯度；(iii) **Single fused CUDA kernel**——所有操作（BMM1 → online softmax → BMM2）融合为单个kernel，消除所有中间HBM读写。

  全栈执行例子（同样GPT-2 small, N=1024, d=64, 16 heads, batch=64, A100 GPU）：
  - **模型推理算法层**：Attention数学定义不变（exact softmax(QK^T/√d)V），计算被重组为block-wise streaming算法。Algorithm 1通过tiling + online softmax rescale保证数值等价。Theorem 1证明返回与标准attention完全相同的O。
  - **系统框架层**：`flash_attn_func(q, k, v)`作为PyTorch兼容的drop-in replacement，直接替换标准attention。HuggingFace/Megatron-LM通过替换attention模块集成，框架无需其他修改。
  - **编译框架层**：论文未明确说明编译框架修改。CUDA kernel手写实现（非编译器自动生成）。论文在Limitations中讨论了未来方向：high-level language compiling to IO-aware CUDA（类似Halide in image processing）。
  - **kernel调度层（关键创新）**：单个fused CUDA kernel替代原来3+个独立kernel调用。内部结构：
    - 划分: $B_c \approx M/(4d)$（KV列block），$B_r = \min(B_c, d)$（Q行block，受head_dim bound）。对N=1024, d=64, M=192KB: $B_c=384, B_r=64$，$T_c=3$ KV blocks，$T_r=16$ Q blocks。
    - 外循环j=1..3（KV blocks）：从HBM加载$K_j$(384×64), $V_j$(384×64)到SRAM（~49KB each）。
    - 内循环i=1..16（Q blocks）：加载$Q_i$(64×64, 8KB)到SRAM。On-chip: $S_{ij}=Q_iK_j^T$（Tensor core FP16 GEMM）→ rowmax → EXP(MUFU.EX2) → rowsum → online rescale: $m_i^{\text{new}}=\max(m_i,\tilde{m}_{ij})$, $\ell_i^{\text{new}}=e^{m_i-m_i^{\text{new}}}\ell_i+e^{\tilde{m}_{ij}-m_i^{\text{new}}}\tilde{\ell}_{ij}$ → $\tilde{P}_{ij}V_j$（Tensor core MMA）→ rescale+accumulate $O_i$ → write $O_i,\ell_i,m_i$ to HBM。
    - 关键：中间$S_{ij}$(64×384×2B=49KB)和$\tilde{P}_{ij}$(49KB)仅驻留SRAM，永不写入HBM。Total SRAM使用: 2×49KB(K,V) + 8KB(Q) + 49KB(S) + 49KB(P̃) + 8KB(O_acc) ≈ 163KB, fits in 192KB。
    - IO复杂度: $\Theta(N^2d^2M^{-1})$ vs standard $\Theta(Nd+N^2)$。实测: GFLOPs 75.2（略增因recomputation）vs 66.6，但HBM R/W从35.3GB降至4.4GB（8× reduction），Runtime从35.1ms降至11.7ms（3× faster）。
    - 反向pass: 从$O(N)$ stats重计算S/P in SRAM。虽然FLOPs增加（recomputation），但消除$O(N^2)$的HBM reads，总体加速。
    - Block-sparse variant: 对预定义稀疏mask中$M_{ij}=0$的block跳过softmax+PV计算，IO复杂度进一步乘sparsity ratio s。
  - **硬件架构层**：同一A100 GPU，无自定义硬件修改。通过tiling使working set fit in SRAM，提升arithmetic intensity。Block size消融（Figure 2 middle）：增大$B_c$→减少HBM passes→减少HBM accesses→runtime下降，直到$B_c\geq 256$后compute-bound。Block-sparse通过跳过零值block进一步减少计算。

  关键设计选择与baseline缺陷的对应：
  - **defect (a): Memory bottleneck** → 方案：Tiling + kernel fusion。将Q/K/V分块，通过online softmax保证跨block正确性。中间结果仅驻留SRAM。HBM读写减少8×（实测35.3GB → 4.4GB）。
  - **defect (b): 不可扩展（长序列OOM）** → 方案：$O(N)$额外内存（仅保存$m,\ell$，每head $2\times N$ floats）。Memory从$O(N^2)$降至线性，使Path-X(16K)和Path-256(64K)训练成为可能。
  - **defect (c): 无细粒度内存控制** → 方案：手写CUDA kernel，完全控制何时从HBM加载/写入，中间tensor不离开SRAM。
  - **defect (d): Kernel fusion不足（反向需中间值）** → 方案：Recomputation代替HBM读取。前向仅保存compressed stats，反向在SRAM中重计算S和P。关键洞察：HBM带宽是真正瓶颈而非FLOPs——即使FLOPs增加13%，runtime仍因HBM访问减少而加速3×。
  - **额外理论贡献：IO复杂度下界** → Proposition 3证明tiling-based方法已达理论最优：不存在exact attention算法能在所有SRAM大小M上用$o(N^2d^2M^{-1})$次HBM访问计算attention。
  - **额外扩展：Block-sparse FLASHATTENTION** → 利用tiling自然地支持block-sparsity mask，仅计算非零block。使用butterfly sparsity pattern在LRA上2.8× speedup且accuracy持平。

## Flash Multi-Head Feed-Forward Network

- baseline方法是什么？
  **标准Llama-like SwiGLU FFN**：在Transformer block中，FFN层使用单一全连接SwiGLU结构——SwiGLU(X) = (SiLU(X·W_gate) ⊙ (X·W_up)) · W_down。输入X ∈ R^{L×d_model}先通过W_gate和W_up投影到d_ff维（d_ff ≈ 8/3·d_model），计算element-wise gating后通过W_down投影回d_model。FFN被视为"单head"的对参数attention——Q attend over W_1 retrieve from W_2。没有多head分解，所有hidden dimension共享同一套参数。

  全栈执行例子（Llama-like SwiGLU FFN, 370M, d_model=1024, d_ff=2752, L=4096, H100 GPU）：
  - **模型推理算法层**：标准SwiGLU FFN。gate = SiLU(X·W_gate^T) ∈ R^{4096×2752}，up = X·W_up^T ∈ R^{4096×2752}，hidden = gate ⊙ up（element-wise），output = hidden·W_down ∈ R^{4096×1024}。单路径推理（"implicit thinking"的"greedy search"——Chen et al. 2025类比）。
  - **系统框架层**：PyTorch eager execution或torch.compile。FFN层作为nn.Module，由transformers库调用。论文未说明特定serving框架修改。
  - **编译框架层**：论文未明确说明。cuBLAS为GEMM提供高度优化的kernel。
  - **kernel调度层**：cuBLAS GEMM kernel执行三次矩阵乘法：(1) X×W_gate^T (4096×1024×2752)，(2) X×W_up^T (4096×1024×2752)，(3) (gate⊙up)×W_down (4096×2752×1024)。中间激活(gate⊙up) ∈ R^{4096×2752} vollständig materialized in HBM——需先写后读，成为I/O瓶颈。cuBLAS对此有高度优化的数据reuse策略和cache hit。
  - **硬件架构层**：NVIDIA H100 GPU。Tensor core执行FP16/BF16 GEMM。HBM↔SRAM间传输tile。

  Baseline缺陷：
  - (a) **单路径推理限制表达力**：FFN的单一d_ff维中间表示可视为"implicit thinking"的单路径搜索（Chen et al. 2025类比），缺少多路径并行探索的representational diversity（类似multi-head attention中不同subspace的收益）。
  - (b) **中间激活的HBM materialization**：gate⊙up ∈ R^{L×d_ff}必须先写入HBM再读取，成为I/O瓶颈。大序列和大模型下（如L=16K, d_ff=5504），单层中间激活占用HBM约L·d_ff·2 bytes（bf16），大模型下单层可达~180MB。
  - (c) **Scaling ratio固定**：标准设计d_ff/d_model ≈ 8/3，这是单路径下的经验最优。若想增加表达力（如增加"head数"）需要额外参数和计算。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FlashMHF（Flash Multi-Head FFN）：Parallel FFN Sub-Networks + I/O-aware Fused Kernel**。核心设计：(1) 将FFN分解为H个head（multi-head design增强表达力），每个head的d_h=d_model/H；(2) 每head内进一步分解为E个并行sub-network（类似dense MoE但不做sparse top-k路由），每sub-network的d_e ≈ 8/3·d_h维持balanced ratio；(3) learned sigmoid gating动态加权聚合sub-network输出；(4) I/O-aware fused kernel (SRAMFFN)避免HBM中materialize中间激活——沿d_ff维度blockwise计算，所有中间结果仅在SRAM中temporary存在。

  全栈执行例子（FlashMHF, 370M, d_model=1024, H=8, d_h=128, E=7, d_e≈342, L=4096, H100 GPU）：
  - **模型推理算法层**：FlashMHF替代SwiGLU FFN。流程变为：(1) Q = split_H(X·W_in) ∈ R^{4096×8×128}；(2) 每head h内计算gating weights R^h = sigmoid(Q_h·W^h)/Σsigmoid ∈ R^{4096×7}；(3) 每sub-network e内独立执行FFÑ(Q_h; K_e^h, U_e^h, V_e^h) = (SiLU(Q_h·K_e^{hT}) ⊙ (Q_h·U_e^{hT}))·V_e^h——注意每sub-network的中间维仅d_e≈342而非d_ff=2752；(4) S_h = Σ_e R^h[:,e] ⊙ FFÑ_output ∈ R^{4096×128}；(5) O = concat_H(S)·W_out。这是"implicit thinking"的"beam search"——多个parallel pathway（H×E个path）并行探索，每path维度小而balanced。本质上类似dense MoE但每head有独立私有sub-network参数。
  - **系统框架层**：PyTorch层替换FFN module为FlashMHF module。论文未说明特定serving框架修改。
  - **编译框架层**：论文未明确说明编译框架修改。Triton和ThunderKittens kernel为手动实现（非编译器自动生成）。
  - **kernel调度层（关键创新）**：SRAMFFN fused kernel（参见Algorithm 1-5）。核心思想——沿d_ff维度（更准确地说，E·d_e维度）blockwise迭代计算，避免materialize intermediate activation：
    - Forward: O ← 0; for block m=1..M: O += (SiLU(Q·K_m^T)⊙(Q·U_m^T))·V_m。每block的结果直接累加到output accumulator（in SRAM），无需保存完整的(gate⊙up) ∈ R^{L×d_ff}。最后一次性将output写入HBM。
    - Hopper实现：warp-group specialization——producer异步prefetch tiles到SRAM ring buffer，CON_WARPGRPS个consumer独立处理不同x-block（sequence partition），在完成一个sub-network的router同步后各自并行计算。
    - Memory从O((d_ff+d_model)·L)降至O(d_model·L)，甚至比标准SwiGLU更小。Memory reduction 3-5x。
    - 通过在SRAM中驻留中间结果+消除HBM round-trip，即使H的总中间维与传统d_ff相等，实际memory和I/O开销大幅降低。Latency speedup 1.00-1.08x（得益于消除I/O瓶颈，但cuBLAS已高度优化GEMM）。
  - **硬件架构层**：NVIDIA H100 GPU。FlashMHF通过降低HBM traffic（消除中间activation的读写）提升memory efficiency。Hopper TMA用于异步prefetch。Warp-group specialization利用SM内并行性。

  关键设计选择与baseline缺陷的对应：
  - **defect (a): 单路径限制表达力** → 方案：Multi-Head设计——FFN分解为H个head，每head有独立private sub-network参数。这是类比multi-head attention的representational subspace specialization。实验验证：FlashMHF-128hdim在370M上比baseline loss低0.016（3.014 vs 3.030），1.3B上低0.050（2.793 vs 2.843）。下游任务平均分370M: 40.48 vs 39.92（+0.56），1.3B: 43.35 vs 41.75（+1.60）。消融实验：Dense-MoE (H=1)甚至差于baseline（3.062 vs 3.030），证明多head分解（而非仅是parallel sub-network）是gain的主要来源。
  - **defect (b): 中间激活HBM materialization** → 方案：SRAMFFN I/O-aware fused kernel。核心trick——沿d_ff维度blockwise计算（每block size = BLOCK_INTER），每block的(SiLU(QK^T)⊙(QU^T))·V直接在SRAM中计算并累加到output accumulator，中间tensor永不写入HBM。结果：peak memory reduction 3-5x（L=4096时FlashMHF≈866MB vs SwiGLU≈2592MB；L=16128时3016MB vs 9966MB）。Memory footprint甚至小于标准SwiGLU（因multi-head design将大中间激活分解为H个窄head的小块accumulation）。
  - **defect (c): Scaling ratio固定** → 方案：Parallel FFN Sub-Networks。Naïve multi-head FFN在模型scale up时d_ff/d_h ratio爆炸（128M: 16, 370M: 21, 1.3B: 45），超过Kaplan et al. 2020的optimal range导致性能退化。FlashMHF通过将d_ff分解为E个d_e维sub-network（每d_e ≈ 8/3·d_h），维持每sub-network internal ratio balanced。关键消融：Naïve MH-FFN在128M上优于baseline但在370M上失效（Table 1: 3.031 vs 3.030，无gain），而FlashMHF-128hdim持续gain（3.014），证明parallel sub-network是scaling成功的关键。
  - **额外设计：Gated aggregation代替简单平均** → Sigmoid-based learned gating（公式11-12）而非softmax-based sparse routing（如标准MoE）。好处：(1) 每token所有sub-network都参与计算（dense activation），避免load imbalance问题；(2) sigmoid + normalization给每token独立的per-sub-network权重，提供更细粒度的动态组合能力；(3) 计算开销可控（仅E维gating vs 全d_ff维FFN操作）。

## FastTree Optimizing Attention Kernel and Runtime for Tree-Structured LLM Inference

- baseline方法是什么？
  **SGLang 的 radix tree KV cache 共享 + 传统 per-query 分离 attention kernel**：SGLang 用 radix tree 组织全局 KV cache 实现 multi-level prefix sharing，减少 GPU 内存占用以服务更多并发请求。但在 computation 层面，SGLang 仍使用传统 attention kernel（FlashAttention/FlashInfer），将每个 query 的 attention 计算分配到独立的 GPU thread block，各 query 间无数据复用。

  全栈执行例子（Llama-2-7B, batch=128, multi-level system prompt + few-shot learning, H100 GPU）：
  - **模型推理算法层**：标准 scaled dot-product attention。Decoder 每步处理 batch 中所有 query 的最后一个 token（Q 为 vector），attention weight = softmax(QK^T/√d)V。各 query 独立计算，无交互。论文未明确说明算法层修改。
  - **系统框架层**：SGLang v0.2.13，radix tree 组织 multi-level shared KV cache。Key 创新在 memory layout，computational part 只是将其分派到 FlashAttention/FlashInfer 等标准 kernel。论文未明确说明框架层额外修改。
  - **编译框架层**：论文未明确说明。Triton/CUDA kernel 通过 PyTorch 调用链执行。
  - **kernel调度层**：FlashAttention decode kernel：每个 query 单独分配 thread block → 每个 block 独立从 HBM 加载 KV cache → Q·K^T 为 GEMV（matrix-vector）→ softmax → P·V（GEMV）。问题：(i) 共享 KV cache 被不同 query 的 thread block 从 HBM 重复加载，shared memory 无法跨 block 复用（HBM bandwidth ≈ 1/10 shared memory bandwidth）；(ii) decode 阶段 Q 为 vector，attention 退化为 GEMV，无法有效使用 tensor core（FlashAttention 仅 <1% effective computation after padding）；(iii) 每个 query 单独 launch 增加 kernel launch overhead。
  - **硬件架构层**：NVIDIA H100 GPU，无自定义硬件修改。Shared memory 在 SM 内各 thread block 间不共享，导致 KV 冗余加载。

  Baseline 缺陷：
  - (a) **Memory-computation gap**：radix tree 优化了 memory layout（内存复用），但 computation 仍 per-query 分离，无法利用 tree 结构隐式的 query-context 共享关系聚合计算。
  - (b) **Redundant HBM loads**：共享 KV cache 被每个 query 从 HBM 独立加载，浪费 memory bandwidth。
  - (c) **Tensor core underutilization**：decode 阶段 GEMV 无法填满 tensor core 的最小输入 shape 要求（需 padding → wasted computation）。
  - (d) **无 group 级优化**：不同 queries 共享不同长度的 prefix，如何分组才能在 padding overhead、intermediate result overhead、parallelism 之间取得平衡——baseline 不考虑此问题。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FastTree：tree-structured attention kernel + tree structure-adaptive runtime optimization**。核心设计：(i) greedy heuristic 将 radix tree 转换为最优 context-queries grouping plan——把 KV cache 以 tree 结构为 guid 分区为 shared contexts，将共享同一 context prefix 的 queries 聚合为一个 group；(ii) tree-structured attention kernel 以 Flash-Attn 风格 tile-by-tile 处理各 group——Q tile 在 query dim 并行化（跨 block）、KV tile 在 context dim 串行迭代（block 内），Q tile 聚合后 GEMV→GEMM 使能 tensor core，KV tile 在 shared memory 中被 group 内所有 query 共享复用（消除 HBM 重复加载）；(iii) multi-phase tiling 根据 node 在 tree 中的位置（near-root vs near-leaf）自适应选择 tile size；(iv) long context splitting 在 GPU SM 欠饱和时 split 超长 context 提升并行度。

  全栈执行例子（同样 batch=128, multi-level system prompt, H100 GPU）：
  - **模型推理算法层**：attention 计算逻辑不变（scaled dot-product），但 execution 从 per-query 独立计算变为 group-aggregated 计算。Group 内各 query 共享同一 attention context prefix 部分的 K/V tile，仅在 unique suffix 部分各自计算。论文未明确说明算法层修改。
  - **系统框架层**：FastTree 作为 SGLang plugin——读取 SGLang 维护的 radix tree → runtime 生成 grouping plan → 替换 attention backend。每次 radix tree 结构变化时（新请求 arrival/completion）重新执行 runtime search。CPU preprocessing overhead 被 SGLang 的多步 continuous decoding 摊销。论文未明确说明框架层额外修改。
  - **编译框架层**：论文未明确说明。FastTree 的 attention kernel 用 Triton 实现（Python DSL），无需编译框架修改。
  - **kernel调度层**：核心理念——**从 memory layout-guided computation optimization**。
    Step 1 - Runtime greedy heuristic：BFS 遍历 radix tree → 对每条 parent→child 边比较 SplitKVCost（分离：padding cost + intermediate result cost）和 SplitQCost（拼接：padding cost）→ 贪心选开销更小的边赋值 → 生成 virtual tree → node-centric query aggregation 得出 (context, {queries}) grouping plan。
    Step 2 - Tree-structured attention kernel launch：单 kernel 处理所有 groups。每个 group 内：Q 矩阵 tiles 沿 query dim 分派到不同 block（并行）→ 每个 block 循环 KV tiles（串行，因 softmax 跨 context dim 有 inter-tile dependency）→ BMM1(Q_tile·K_tile^T) on tensor core (GEMM) → online softmax (shared memory) → BMM2(P·V_tile) on tensor core (GEMM) → 写 partial O 和 L 到 HBM。
    Step 3 - Reduce kernel：利用 LogSumExp vectors rescale 各 group 的 partial O_i 后累加得到 final output。
    Step 4 - 优化：靠近 root 的 node query 多 → 大 tile size（如 64）最大化 KV 复用；靠近 leaf query 少 → 小 tile size（如 16）避免 shared memory 浪费；long context 若导致 SM 欠饱和 → split context 增加 parallelism。
  - **硬件架构层**：同一 NVIDIA H100 GPU，无自定义硬件。FastTree 通过 query aggregation + shared memory KV reuse + GEMM tensor core utilization，不再受制于 HBM bandwidth 和 GEMV 的低效。未使用 H100-specific features（TMA 等），可移植到其他 GPU。

  关键设计选择与 baseline 缺陷的对应：
  - **defect (a): Memory-computation gap** → 方案：tree structure-adaptive runtime 将 radix tree 的 memory layout 作为 grouping plan 的输入，使 computation 直接受益于 tree 结构的共享关系。radix tree 边 → binary assignment → virtual tree → grouping plan，memory layout 和 computation 统一在同一 tree representation 下。
  - **defect (b): Redundant HBM loads** → 方案：query aggregation 后，同一 group 内的 K/V tile 只需从 HBM 加载一次到 shared memory，被 Q tile 内所有 query 复用。shared memory bandwidth >> HBM bandwidth，大幅减少 memory transaction。特别在 root node（聚合 query 最多）处效果最显著。
  - **defect (c): Tensor core underutilization** → 方案：query aggregation 使 Q 从 vector 变为 matrix（batch of queries），attention 从 GEMV 变为 GEMM，满足 tensor core 的最小 tile shape 要求。无需 padding 或仅少量 padding。FlashAttention 在 decode 阶段 <1% effective computation → FastTree 显著提升 tensor core utilization。
  - **defect (d): 无 group 级优化** → 方案：greedy heuristic 的 cost model 同时考虑 padding overhead（C_P,q + C_P,c）和 intermediate result overhead（SplitKVCost_R），在 query splitting（更多聚合/更大 tile → 可能更多 padding）和 context concatenation（更少 groups/更少 intermediate results → 可能 query splitting 导致 padding 恶化）之间做 trade-off。实验结果：复杂 deep tree 中 greedy heuristic 比 direct aggregation 快 up to 2.2×。
  - **额外设计：multi-phase tiling** → 根据 node 在 tree 中的层级自适应 tile size，解决 uniform tile size 在 heterogeneous tree（不同层级 node 的 query 数差异大）下的 shared memory waste 问题。DeFT（concurrent work）使用 fixed tile size 导致大量 shared memory 浪费，FastTree 在这方面明显胜出。

## Fast-dLLM v2: Efficient Block-Diffusion LLM

- baseline方法是什么？
  **标准自回归（AR）LLM推理**：以Qwen2.5-Instruct为backbone，使用causal attention mask + next-token prediction loss训练，推理时逐token自回归生成。每个forward step仅生成1个token，需要response_length次forward才能完成生成（如256 tokens需要256次forward）。自回归模型的sequential decoding限制了推理并行度和吞吐量。

  全栈执行例子（Qwen2.5-7B-Instruct，GSM8K 5-shot推理，A100 GPU，gen_len=256）：
  - **模型推理算法层**：Qwen2.5-7B-Instruct使用causal self-attention，每个token只能attend到自身及之前的token。逐token自回归生成：p(x_i | x_{<i})。训练loss为cross-entropy over next-token prediction。推理时从prompt开始，逐个生成token直到[EOS]或max_length。256 tokens需要256次sequential forward passes。
  - **系统框架层**：使用标准PyTorch推理，可搭配vLLM等serving框架使用continuous batching + PagedAttention管理KV cache。每次forward迭代生成1个token → 更新KV cache → 继续下一token。prefill阶段一次处理prompt的KV cache计算。论文未明确说明特定serving框架修改。
  - **编译框架层**：论文未明确说明。使用标准PyTorch/HuggingFace Transformers推理路径。
  - **kernel调度层**：Decode阶段为memory-bound的GEMV操作（batch_size × 1 token），每次forward处理小矩阵向量乘，GPU计算利用率低。KV cache存储在HBM中，每步加载完整cache参与attention计算。
  - **硬件架构层**：NVIDIA A100/H100 GPU，无自定义硬件修改。AR decode吞吐量约39.1 tok/s（GSM8K, A100）。

  Baseline缺陷：
  - (a) **Sequential decoding限制并行度**：逐token生成，256 tokens需256次forward，GPU在decode阶段利用率低（memory-bound GEMV）。
  - (b) **无法利用块内token间的双向依赖**：causal attention仅允许单向（left-to-right）attention，块内token无法互相condition以提升预测质量。
  - (c) **吞吐量扩展受限**：增加batch size虽能提升吞吐，但AR模型的每token延迟不变，总延迟与生成长度线性相关。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Fast-dLLM v2：将预训练AR模型适配为block diffusion LLM**。核心设计：(i) Block diffusion训练——序列拆分为block size=32的块，块内使用bidirectional attention + masked token prediction（diffusion），块间保持AR causal conditioning；(ii) Complementary masking + token shift——每个训练样本产生两个互补mask view，masked位置使用i-1位置的hidden state预测token i，保留AR模型的representation quality；(iii) Hierarchical caching推理——block级KV cache（跨block复用历史上下文）+ sub-block DualCache（块内高效并行refinement）+ confidence-aware parallel decoding。

  全栈执行例子（Fast-dLLM v2 7B，GSM8K 5-shot推理，A100 GPU，gen_len=256，threshold=0.9）：
  - **模型推理算法层**：序列被组织为K=⌈256/32⌉=8个block，每个block内通过masked diffusion并行refine（bidirectional attention），block间通过causal conditioning自回归生成。子块大小=8，每次forward在sub-block内基于confidence阈值0.9并行解码多个高置信token。训练时complementary masking保证所有token都接收masked和unmasked上下文的监督。Token shift通过position i-1的hidden state预测position i的token，使dLLM保持AR-like temporal representations。总解码步数远小于256（约40-80步），throughput从39.1→101.7 tok/s（2.6×加速）。
  - **系统框架层**：推理pipeline实现block-level KV cache管理：已解码block的K/V被缓存为read-only prefix，当前block仅需计算自身attention + 对prefix的cross-attention。无需serving框架修改，在PyTorch层实现block-wise decoding loop + cache管理。batch decoding通过右填充[MASK]对齐block边界，所有序列同步逐block推进。
  - **编译框架层**：论文未明确说明。使用PyTorch flex-attention实现自定义block-wise attention mask的高效计算。
  - **kernel调度层**：Block-level cache将attention计算量从O(T·(|p|+L)²·d)降至O(K·T'·(B²+|p|·B)·d)，其中B=block size，K=block数，T'为每块内步数（远小于原始T=256）。DualCache进一步缓存sub-block的prefix/suffix KV，将块内attention降至O(S²·d)（S=sub-block size=8）。这使得GPU compute利用率大幅提升，尤其在batch size较大时（compute-bound regime）。
  - **硬件架构层**：NVIDIA A100/H100 GPU，无自定义硬件修改。通过降低每token所需forward次数和复用KV cache减少总计算量，在A100上达2.5×加速（batch=1），H100上batch=64时达1.8×加速。

  关键设计选择与baseline缺陷的对应：
  - **defect: Sequential decoding限制并行度 (a)** → 方案：Block diffusion——序列分为block，块内bidirectional attention允许并行生成多token。配合confidence-aware parallel decoding（来自Fast-dLLM v1），threshold=0.9时仅轻微准确率下降即达2.6×加速（GSM8K）。块间仍保持AR因果依赖，保证全局语义连贯性。
  - **defect: 无法利用块内双向依赖 (b)** → 方案：Block-wise attention mask设计——M_BD（块内双向自注意力）使block内token可互相condition，M_OBC（offset block-causal）保持对历史clean context的单向访问，M_BC（block-causal）保持clean token间的AR-like progression。这种hybrid attention使模型在块内获得更丰富的context modeling能力，同时保留AR模型的预测质量。
  - **defect: 吞吐量扩展受限 (c)** → 方案：Block diffusion的并行特性使throughput随batch size增长优于AR模型（Figure 5）。H100上batch=64时diffusion比AR快1.8×，因为diffusion的forward pass计算更密集（多token并行），能更好利用H100的更高算力。Sub-block cache在compute-bound regime（batch=32）下提供额外加速。
  - **额外设计：数据效率** → 通过复用预训练AR模型的权重和AR-friendly的block-wise attention设计（接近原始causal attention结构），Fast-dLLM v2仅需~1B tokens微调（vs Dream的~580B tokens），实现500×数据减少。训练仅需64×A100约8-12小时，使block diffusion适配变得实际可行。
  - **额外设计：训练-推理一致性** → 通过引入sub-block解码策略（推理时使用sub-block size=8，训练时block size=32），在保持训练block结构与推理一致的前提下（避免Table 4中的mismatch性能退化），灵活控制推理粒度以优化accuracy-throughput trade-off。

## Composing Distributed Computations Through Task and Kernel Fusion

- baseline方法是什么？
  **标准分布式 task-based runtime 执行模型（不进行跨 task 融合）**：高层分布式库（如 cuPyNumeric, Legate Sparse）将每个库操作分解为独立的 index task 序列，每个 task 内部执行嵌套循环（kernel），task 通过 runtime 管理的分布式数据 collection（region）进行通信。每个中间操作结果分配为独立的分布式数组。不同库操作之间的 task 隔离开，不进行跨 task 融合。

  全栈执行例子（cuPyNumeric 5-point stencil，4 nodes 4 GPUs，Figure 1）：
  - **模型推理算法层**：论文未涉及 ML 推理（科学计算场景）。5-point stencil: avg = center + north + east + west + south; work = 0.2 * avg; center[:] = work。
  - **系统框架层**：cuPyNumeric 将每个 NumPy 操作（ADD, MULT, COPY）映射为独立 index task launch。每次调用 np.add/np.multiply 时，cuPyNumeric 创建临时 distributed array（如 t1, t2, t3, avg），然后发射独立的 index task 计算该操作。task 按程序顺序发射到 Legion runtime，Legion 负责动态发现 task 间的依赖关系并计算所需通信（data coherence）。
  - **编译框架层**：无明显编译框架层优化。cuPyNumeric 的每个操作独立 target Legion，不进行跨操作的编译优化。Legion 收到 task stream 后按序执行。
  - **kernel调度层**：每个 index task 内部为 element-wise 嵌套循环（Figure 1e: ADD 含一对 for 循环，MULT 含一对 for 循环，COPY 含一对 for 循环），各 kernel 独立执行。5-point stencil 一次迭代产生 6 个独立 kernel（4 ADD + 1 MULT + 1 COPY），每个 kernel 单独 launch，中间结果通过 HBM 传递，需要 6 次 pass over data（或至少需要 5 个临时数组的 HBM read/write）。
  - **硬件架构层**：NVIDIA A100 GPU，无自定义硬件修改。

  Baseline 缺陷：
  - (a) **临时分布式数据膨胀**：每个中间操作产生一个分布式临时数组（t1, t2, t3, avg, work），占用大量 GPU HBM 并产生额外 memory traffic。
  - (b) **数据局部性差**：多个 element-wise kernel 依次执行，每个 kernel 从其输入读取、计算、写入临时输出，下一个 kernel 再从临时输出读取。数据无法在 on-chip memory（register/SRAM）中复用。
  - (c) **kernel launch overhead**：大量小 kernel 的 launch overhead 累积（Black-Scholes: 67 tasks/iteration）。
  - (d) **跨库边界无法优化**：不同库（cuPyNumeric + Legate Sparse）的 task 互相独立，无法跨库进行融合。
  - (e) **分布式数据 aliasing 导致融合复杂**：aliasing views（center, north, east, west, south 均为 grid 的切片）使简单融合可能违反依赖关系。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Diffuse：通过 task fusion + kernel fusion 在分布式 runtime 上自动组合分布式计算**。核心设计：(i) 一个 scale-free IR，使分布式程序的表示大小与机器规模无关；(ii) 基于 4 个 fusion constraint 的动态 task fusion 算法；(iii) 基于 MLIR polyhedral compilation 的 kernel fusion，在融合 task 内部融合循环、消除临时分配；(iv) 在高层库（cuPyNumeric, Legate Sparse）和底层 runtime（Legion）之间作为中间层，对用户透明。

  全栈执行例子（同样 5-point stencil，Diffuse enabled）：
  - **模型推理算法层**：同一 5-point stencil 计算逻辑，用户代码不变。
  - **系统框架层**：修改后的 cuPyNumeric 动态生成 Diffuse IR 而非直接 target Legion。Diffuse 缓冲 task stream 到 window（window size=5），运行 fusion constraints 数据流分析：(a) launch-domain-equivalence 验证同一 launch domain；(b) true-dependence 检查 write→read/write 仅在相同 partition 时安全；(c) anti-dependence 检查 read→write 仅在相同 partition 时安全；(d) reduction 约束读写冲突。由于 aliasing（center/north/east/west/south 均为 grid 的不同 partition），COPY(work, center) 不满足 true-dependence constraint（work 写入的 partition 不同于各个 aliasing view 读取的 partition），因此 COPY 无法融入 fused task。而 4 ADD + MULT 操作使用相同的 partition 读写中间结果，满足所有 constraints → 融合为 FUSED_ADD_MULT。
  - **编译框架层**：Diffuse 的 MLIR JIT compiler（Section 6）将 FUSED_ADD_MULT 中的 5 个 task body 组合：(a) 顺序调用各 task 的 MLIR generator 生成 fragment → 组合为初始 fused kernel body；(b) temporary store elimination（Section 5.1）通过 dataflow 分析发现 avg 由 fused task 完全产生且不被 fused task 外的 task 或应用引用，降级为 task-local memref.alloca；(c) polyhedral fusion pass 将 4 个 ADD + 1 MULT 的 5 个独立 affine.for 循环融合为单个 affine.par 循环；(d) memref.alloca 消除，因为 temporary 被完全 inlined；(e) 最优 single-pass kernel（Figure 8d）：一次 pass 完成 5-way scaled add。
  - **kernel调度层**：单一 CUDA kernel launch 替代原来的 6 个 kernel。每次 GPU thread 处理一个 element 的全部 5 次加法和 1 次乘法，中间结果存储在 register 中（而非 HBM temporary array）。Arithmetic intensity 大幅提升，memory traffic 减少约 5×。
  - **硬件架构层**：标准 GPU，无自定义硬件修改。Memoization（Section 5.2）通过 canonical De-Bruijn index 表示检测 isomorphic task stream 并复用分析+编译结果，使编译开销可摊销（25–119 次迭代即可 breakeven，Figure 13）。

  关键设计选择与 baseline 缺陷的对应：
  - **defect: 临时分布式数据膨胀** → 方案：temporary store elimination（Section 5.1）。通过 dataflow 分析在 fused task 内识别仅被 fused task 内部产生和消费、且无外部引用的 store（满足 Definition 4 三约束），将其从分布式分配降级为 task-local allocation。MLIR 后续 pass 进一步 inlining 消除临时 allocation。
  - **defect: 数据局部性差** → 方案：kernel fusion（Section 6）。MLIR polyhedral compilation 将多个独立循环融合为单一循环，使中间结果保留在 register/SRAM 中，最大化数据复用。Black-Scholes 中 67 个 element-wise 操作融合为单 kernel 一 pass 计算，10.7× speedup。
  - **defect: kernel launch overhead** → 方案：task fusion 将多个 index task 合并为单个 index task，kernel fusion 将多个 kernel body 合并为单个 kernel。Black-Scholes: 67 tasks → 1 task（Figure 9）。即使 task granularity 已大于 Legion 的最小有效粒度（1ms/task），kernel fusion 通过提升 arithmetic intensity 产生实际加速。
  - **defect: 跨库边界无法优化** → 方案：Diffuse 的 IR 和 fusion analysis 基于 task 的 privilege 信息（R/W/Rd/RW）和 partition 结构（None/Tiling），而非任何特定库的语义。cuPyNumeric 和 Legate Sparse 的 task 在 Diffuse IR 中被统一表示，fusion constraints 在统一的 IR 层面运行，实现跨库（cross-library）融合。
  - **defect: aliasing 导致融合复杂** → 方案：Diffuse 的 fusion constraints 通过 partition equality check（而非计算 sub-store intersection）来检测 aliasing。由于 partition 的结构化表示（Tiling(shape, offset, proj)），equality check 为 O(1) 操作，且 scale-free（不随 processor 数增长）。constraints 的设计使得 aliasing 视图中非冲突的读写（如同时读不同的 aliasing view）可以安全融合，而冲突的读写（如 COPY 写 center 同时 ADD 通过不同 partition 读 center）被正确阻止融合。
  - **defect: 分布式依赖分析复杂度随规模增长** → 方案：scale-free IR。partition 的映射是隐式的（通过 Tiling 公式 sub-store-bounds(Tiling(shape, offset, proj), p) = [proj(p)*shape, proj(p+1)*shape) + offset），IR 大小仅由 task 数和 partition 数决定，与 GPU 数量无关。这使得 fusion analysis 可以在任意规模机器上执行。对比 Legion 直接表示 partition（显式存储每个 sub-store 边界），别名查询复杂度随 processor 数增长。
  - **额外设计：analysis 复用** → Memoization 将 alpha-equivalence 问题应用于 task stream 匹配（Figure 7），通过 canonical De-Bruijn index 表示消除 store ID 重命名的影响，使 isomorphic task stream 的 fusion decision 和编译结果可被复用。在循环中特别有效（同一 pattern 的 task stream 重复出现在每次迭代中）。

## BLASST Dynamic BLocked Attention Sparsity via Softmax Thresholding

- baseline方法是什么？
  **Dense FlashAttention（FlashAttention-3/4）**：在GPU上通过tiled block-wise online softmax算法计算完整attention矩阵Attention(Q,K,V)=softmax(QK^T/√d_k)·V。所有QK block（T_r×T_c个）都完整计算softmax指数和PV矩阵乘法，所有Value block都从HBM加载。

  全栈执行例子（Llama-3.1-8B，prefill batch=1，128K sequence length，B200 GPU）：
  - **模型推理算法层**：标准scaled dot-product attention，Q/K/V线性投影后计算全量QK^T ∈ R^{128K×128K}，softmax归一化后乘以V。MHA: 32个head独立计算；GQA: 8个KV head共享。
  - **系统框架层**：TensorRT-LLM或FlashInfer调用FlashAttention-3 kernel。In-flight batching（concurrency 64）混合调度prefill和decode请求。用户不可见attention内部sparsity pattern，所有token平等对待。
  - **编译框架层**：论文未明确说明编译框架层修改。TensorRT-LLM的graph optimization pass选择FlashAttention kernel作为attention实现。
  - **kernel调度层**：FlashAttention-3 prefill kernel pipeline：BMM1(QK^T) → softmax(EX2+rowsum) → BMM2(PV)，在warpgroup级别重叠tensor core MMA和CUDA core softmax。每轮mainloop迭代处理一个KV block。Decode kernel：V load from HBM → BMM1(单query QK^T) → BMM2(PV)，memory-bound于HBM带宽。Kernel内部无sparsity判断——所有B_c个score都参与softmax和PV乘法。
  - **硬件架构层**：标准NVIDIA B200/H200 GPU。Tensor core执行FP16/BF16 MMA（QK^T和PV）；CUDA core执行MUFU.EX2指数、FMUL乘法、FADD加法（softmax）；HBM→SRAM的tiled加载（Q/K/V分块）。论文未明确说明硬件架构层自定义修改。

  Baseline缺陷：(1) 计算浪费：长序列中大量attention score接近零（attention分布稀疏），但仍消耗CUDA core和tensor core计算softmax和PV乘法。(2) 内存带宽浪费：decode阶段所有V block从HBM加载，但很多block的attention权重近零。(3) 无法利用attention稀疏性：现有sparse attention方法（MInference/XAttention等）依赖pre-computation、proxy scores或额外训练，引入overhead抵消理论加速。(4) 缺乏统一的prefill+decode方案：大多数方法仅优化单一阶段。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **BLASST（BLocked Attention Sparsity via Softmax Thresholding）——基于online softmax统计量的动态block级attention稀疏化**：在FlashAttention的tiled online softmax过程中，复用已计算的running maximum和block local maximum统计量，通过简单阈值比较（m̃_i^{(j)} - m_i^{(j)} < ln(λ)）识别可跳过的block，跳过其softmax计算、PV乘法和V加载。**零额外推理开销**——skip decision复用的统计量在FlashAttention中本来就要计算。

  全栈执行例子（同样Llama-3.1-8B，prefill batch=1，128K sequence length，B200 GPU，target sparsity=50%）：
  - **模型推理算法层**：attention计算逻辑变为——对每个QV block i和KV block j，(a) BMM1计算S_ij = Q_iK_j^T（不变），(b) rowmax(S_ij)得到local max（本来就计算），(c) 更新running max（本来就计算），(d) 阈值判断：if local_max - running_max < ln(λ) → skip，否则正常softmax+PV。通过跳过50%约block，减少50%的CUDA core EXP计算和tensor core MMA计算。**无需训练、无需pre-computation、无需proxy scores**。
  - **系统框架层**：BLASST已集成到TensorRT-LLM和FlashInfer的attention接口中，作为FlashAttention的drop-in replacement，仅增加一个scalar threshold参数λ。用户通过calibration确定λ = α·exp(β·S)/L，其中S为目标sparsity。In-flight batching正常工作，prefill和decode请求均可受益。端到端TTFT和TPOT均有1.1×加速（Qwen3-30B，LongBench V1）。
  - **编译框架层**：论文未明确说明编译框架层修改。BLASST kernel直接替换框架中的attention kernel调用，接口兼容。
  - **kernel调度层**：
    - Prefill kernel（compute-bound优化）：BMM1照常计算所有block → skip check（predicate+VOTE+ATOMIC，隐藏于BMM1后）→ 被跳过block省略EX2 softmax和BMM2(PV)，tensor core直接进入下一轮BMM1。Pipeline从18 time units压缩到14 time units（50% sparsity, 4轮迭代）。Speedup从1.25×（39% sparsity）到1.77×（94% sparsity）on B200。
    - Decode kernel（memory-bound优化，batched load scheduling）：连续发射多个K^TQ的BMM1 → 批量skip check → 仅发射通过检查的V_j loads → 再执行对应的BMM2。消除naive sequential pipeline的scoreboard dependency bubble。Speedup从1.18×（37% sparsity）到1.79×（92% sparsity）on B200。
  - **硬件架构层**：标准GPU硬件，无自定义硬件修改。skip check实现仅需warp-level VOTE指令 + 单线程ATOMIC to shared memory，均由现有GPU指令集原生支持。论文未明确说明硬件架构层自定义修改。

  关键设计选择与baseline缺陷的对应：
  - **defect: 计算浪费（大量近零attention scores仍被计算）** → 方案：复用online softmax已有的running maximum统计量进行block级阈值判断（m̃_i^{(j)} - m_i^{(j)} < ln(λ)），跳过被剪枝block的softmax和PV乘法。Skip decision仅添加predicate+VOTE+ATOMIC几条指令，被pipeline隐藏。
  - **defect: 内存带宽浪费（decode阶段加载全量V blocks）** → 方案：decode kernel的batched load scheduling——先背靠背计算多个K^TQ确定skip pattern，再仅加载需要的V blocks，直接按sparsity比例减少HBM traffic。
  - **defect: 现有sparse attention方法引入overhead（pre-computation/proxy scores/training/fine-tuning）** → 方案：完全training-free和pre-computation-free。所有sparsity decision基于online softmax内部已有统计量，是"免费"的byproduct。Table 1对比了所有方法的特性——BLASST是唯一同时支持prefill+decode加速且无需training和pre-computation的方法。
  - **defect: 缺乏统一的prefill+decode方案** → 方案：同一算法框架下设计了两套specialized kernel——prefill kernel优化compute-bound场景（跳过softmax+MMA），decode kernel优化memory-bound场景（跳过V loading+softmax）。两者共享相同的skip判断逻辑，使用同一套阈值校准参数。
  - **defect: 固定阈值在不同context length下sparsity不稳定** → 方案：校准pipeline（Algorithm 2）发现λ·L = α·exp(β·s)的指数关系，其中λ与L成反比。用户仅需指定目标sparsity S，kernel自动按context length适配阈值，sparsity偏差仅~1.2%（Table 6）。
  - **defect: 超高sparsity下accuracy退化** → 方案：sparsity-aware training（fine-tuning forward pass中应用BLASST），模型在训练中学到将重要信息集中在高attention score block中，accuracy退化降低至1.7×（Figure 6）。

## ACS Concurrent Kernel Execution on Irregular, Input-Dependent Computational Graphs

- baseline方法是什么？
  **单 CUDA Stream 串行执行**：所有 GPU kernel 被发射到同一个 CUDA stream 中。CUDA runtime 保证同一 stream 上的 kernel 按发射顺序串行执行，consumer kernel 必须等待 producer kernel 完全结束后才能开始。这是 PyTorch、TensorFlow 等主流 DL 框架的默认行为。

  全栈执行例子（以 Deep RL Brax Ant 物理仿真的一次 training batch 数据生成，RTX 3060 28 SM）：
  - **模型推理算法层**：物理仿真计算刚体碰撞检测、关节力矩、接触力等的多个小 kernel（每个 kernel 通常 < 200 CTA）。Brax/JAX 实现。
  - **系统框架层**：Brax/JAX 将所有仿真 kernel 发射到单一 CUDA stream，无并发调度。程序员无法提前知道完整的计算图（每次仿真 input 不同导致不同的接触/碰撞计算路径）。
  - **编译框架层**：论文未明确说明编译框架层修改。使用标准 nvcc/JAX 编译器。
  - **kernel调度层**：大量小 kernel 在单一 stream 中串行执行。例如 Ant 环境一次 batch 生成需要数百个 kernel launch，每个 kernel 仅有少量 CTA（中位数 < 200 CTA），远不能填满 28 SM。GPU 实际达到的 occupancy 仅约 34%（Ant 环境，RTX 3060），即约 66% 的 SM 计算资源被浪费。每个 kernel 执行时间短，kernel launch 延迟（~5-20μs）相对于 kernel 执行时间不可忽略。
  - **硬件架构层**：标准 NVIDIA GPU 命令处理器按序从命令队列中取 kernel 发射。无 inter-kernel dependency 信息，同一队列内 kernel 严格串行。论文未明确说明硬件架构层自定义修改。

  Baseline 问题两重：(a) **Input-dependent 计算图**：kernel 间依赖关系随每次 input 变化，无法提前构建完整的 kernel 依赖 DAG。CUDA Graph 虽能消除 launch/sync 开销，但 DAG 构建耗时达执行时间的 47%（Brax），不适合每 input 重新构建。(b) **不规则依赖导致细粒度调度**：计算图不规则（非简单独立 partition），使用多 stream + cudaStreamWaitEvent 细粒度调度会产生大量 CPU-GPU 同步开销（每次同步 5-20μs）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **ACS（Automatic Concurrent Scheduling）——运行时乱序 kernel 调度**：在固定大小的调度窗口内（类似 CPU 乱序指令调度），对顺序发射的 kernel 进行运行时依赖检查和乱序并发调度。kernel 在调度窗口内被标记为 ready/pending/executing 三种状态，当 upstream kernel 全部完成时标记为 ready，由 scheduler 并发发射到多个 CUDA stream 或 hardware dispatch unit。

  全栈执行例子（同样 Brax Ant 仿真，RTX 3060 28 SM，ACS-SW 窗口大小 32）：
  - **模型推理算法层**：物理仿真计算逻辑不变。程序员通过 ACS_wrapper 为每个 kernel 标注 read_segments 和 write_segments（起始虚拟地址+大小）。对于常见 kernel（矩阵乘、卷积、加法等），segments 从函数原型直接可得；对于间接内存访问的 kernel，标注为访问全部 GPU memory（保守但保证正确）。
  - **系统框架层**：ACS-SW 作为用户态运行时系统替代 JAX/Brax 的单 stream 执行。应用线程调用 kernel 时，先调用 `get_addresses()` 将 kernel 参数解析为实际虚拟地址范围，然后将 kernel + segments metadata 送入输入 FIFO 队列。CPU 上的 window module 线程维护调度窗口和依赖关系，scheduler module 线程（可配置数量，每个绑定一个 CUDA stream）轮询窗口获取 ready kernel 并发发射。
  - **编译框架层**：论文未明确说明编译框架层修改。ACS_wrapper 的 `get_addresses()` 函数可由程序员手写或通过 GPUOcelot 等二进制分析工具自动提取。依赖检查算法为 O(segments²) 遍历 read+write segments 对检查地址范围重叠。
  - **kernel调度层**：调度窗口（大小 N=32）中，每个 kernel 维护一个 upstream kernel 列表（依赖的 kernel ID）。当 kernel A 的 write segments 与 kernel B 的 read/write segments 有重叠时，A 被加入 B 的 upstream list。kernel 完成时，window module 遍历窗口中所有 kernel 的 upstream list 移除已完成 kernel。Upstream list 变为空的 kernel 被标记为 ready，由 scheduler 发射到空闲的 CUDA stream。多个 CUDA stream 上的 kernel 在 GPU 上真正并发执行，原本少量 CTA 的小 kernel 现在可以并行填满 GPU SM。ACS-SW 达到平均 1.56×（最高 1.87×）加速，occupancy 从 34% 提升至接近满载。
  - **硬件架构层（ACS-HW）**：在 GPU 命令处理器中增加硬件调度窗口（1KB SRAM for N=32）：每个 slot 包含 8-bit kernel ID 及 (N-1) 个 8-bit upstream kernel ID（全关联存储），2-bit 状态（ready/pending/executing）。upstream load module 修正 CPU 端可能 stale 的 scheduled_list（移除已完成 kernel，阻塞超过 M 个新 kernel 插入以防遗漏）。kernel 完成时硬件在 N-1 cycle 内更新所有 slot 的 upstream list。Ready kernel 直接被 hardware dispatch unit 发射。消除 CPU-GPU 同步和 kernel launch 开销（原本 5-20μs/次）。ACS-HW 达到平均 1.79×（最高 2.19×）加速。端到端 Deep RL 训练加速 1.42×（ACS-HW）和 1.30×（ACS-SW）。

  关键设计选择与 baseline 缺陷的对应：
  - **defect: input-dependent 计算图导致无法提前静态调度** → 方案：运行时滑动窗口调度（类似乱序执行），每次仅检查和调度窗口内有限数量 kernel（N=32），延迟低（依赖检查 410ns~1640ns），不依赖完整 DAG 预构建。
  - **defect: 多 stream 细粒度调度的 CPU-GPU 同步开销大（5-20μs/次）** → 方案：ACS-SW 通过固定数量 scheduler 线程 + stream 复用减少同步次数；ACS-HW 将调度完全移到 GPU 硬件内，消除 CPU-GPU 往返。
  - **defect: persistent threads 无法支持异构 kernel** → 方案：ACS 通过 CUDA stream 发射原生 kernel，每个 kernel 保持自身的寄存器/shared memory 配置，无 PT 的同质性限制。
  - **defect: CUDA Dynamic Parallelism 仅支持父子依赖** → 方案：ACS 依赖检查支持任意 kernel 间的多对多依赖关系（通过 read/write segments 重叠检测）。

## A Framework for Fine-Grained Synchronization of Dependent GPU Kernels

- baseline方法是什么？
  **Stream Synchronization（CUDA Stream 同步）**：将两个有依赖关系的 CUDA kernel 发射到同一个 CUDA stream 上。CUDA runtime 保证同一 stream 上的操作按发射顺序执行，因此 consumer kernel 的所有 thread block 必须在 producer kernel 的**所有** thread block 完成后才能开始执行。

  全栈执行例子（以 MegatronLM GPT-3 MLP 的两个依赖 GeMM 为例，Batch=256，V100 80 SM）：
  - **模型推理算法层**：MLP 执行 XW₁ = GeLU(X × W₁)，然后 XW₁₂ = XW₁ × W₂。两个 GeMM 串行依赖。
  - **系统框架层**：PyTorch 调用 CUTLASS GeMM kernel，两个 kernel 在同一 CUDA stream 上发射。论文未明确说明上层 serving 框架修改。
  - **编译框架层**：论文未明确说明编译框架层修改，使用标准 nvcc 编译。
  - **kernel调度层**：Producer GeMM 的 grid=[1,48,4]（192 thread blocks），Consumer GeMM 的 grid=[1,96,2]（192 thread blocks）。两者均需 ceil(192/80)=3 wave。Stream 同步要求 producer 的 3 个 wave 全部完成后，consumer 的 3 个 wave 才能开始。最后每个 kernel 的 partial wave（第 3 波只执行 32 个 thread block）仅利用 40% SM，两个 kernel 共浪费 2×48=96 SM 时隙。
  - **硬件架构层**：80 个 SM 上，每个 thread block 占 1 个 SM。每 wave 中空闲的 SM 产生 bubbles。论文未明确说明硬件架构层自定义修改。

  Baseline 问题：GPU 利用率低（该例仅 60%），因为 partial wave 中 thread block 数不是 SM 数（× occupancy）的整数倍，且 stream 同步强制 kernel 间完全串行。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **cuSync 细粒度 tile 级同步**：将依赖关系从 kernel 级别下推到 tile 级别。Producer 和 consumer kernel 发射到**不同** CUDA stream 上，通过 global memory semaphore 仅在依赖的 tile 之间同步。Independent tiles 可以并发执行。

  全栈执行例子（同样 MLP，Batch=256，V100 80 SM，使用 TileSync+WRT 策略）：
  - **模型推理算法层**：同上 MLP 计算逻辑不变。
  - **系统框架层**：cuSync header-only 库替代 stream 同步，cuSyncGen DSL 描述 tile 间依赖（如 consumer tile (x,y) 依赖所有 producer 同行 col tile），生成 policy 代码。论文未明确说明上层 serving 框架修改。
  - **编译框架层**：cuSyncGen 编译器将 DSL 依赖描述编译为 CUDA policy 代码（sem/value 方法 + tile order 函数）。自动生成 TileSync（每 tile 独立 semaphore）和 RowSync（同行 tile 共享 semaphore）两种 policy，以及优化的 tile 处理顺序。自动应用 reorder tile load（重叠 wait 与无关 tile load）等优化。
  - **kernel调度层**：Producer 和 consumer 同时发射到不同 stream。Producer wait-kernel 先占位确保 producer 先获得 SM。Producer tile 计算完后 `post()` 对 semaphore atomicAdd。Consumer tile 加载前 `wait()` busy-wait 对应 semaphore。由于 consumer tile E(x,y) 只依赖同行 producer tile C(x,0),...,C(x,N-1)，consumer 无需等 producer 全部完成。两个 kernel 的 independent tiles 可在同一 wave 中混合执行。从原来的 3+3=6 wave（StreamSync）降为约 2.4 wave（cuSync），消除 partial wave 浪费。
  - **硬件架构层**：80 SM 被 producer 和 consumer 的 independent thread block 混合填充，每 wave 的 SM 利用率接近 100%。论文未明确说明硬件架构层自定义修改。

  关键设计选择与 baseline 缺陷的对应：
  - **defect: stream 同步强制 kernel 完全串行** → 方案：不同 stream + semaphore 实现 tile 级依赖，independent tile 可跨 kernel 并发。
  - **defect: partial wave SM 空闲** → 方案：tile 级混合调度，consumer tile 一旦依赖满足即可执行，填充原本空闲的 SM。
  - **defect: 通用性不足（Stream-K 仅支持 GeMM）** → 方案：cuSync 适用于所有 tile-based kernel（GeMM、Conv2D、Dropout、Softmax），仅需少量代码修改（0.5%-1%）。

## AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization

- baseline方法是什么？
  **人工/专家驱动的 AI 加速器 kernel 优化**：对于新兴 AI 加速器（如 Amazon Trainium），kernel 开发者依赖有限的性能直觉和优化经验手动调优 NKI kernel。初始 kernel 由 Neuron Compiler 自动生成，但其性能远低于硬件理论峰值（Trainium 1 上平均仅 49% peak throughput）。开发者需通过反复试验探索 loop ordering、tiling、memory layout、recomputation trade-off 等优化空间，过程耗时且依赖专家知识。

  全栈执行例子（以 NKIBench BatchMatmul+Softmax kernel 为例，Trainium 1 单核）：
  - **模型推理算法层**：Transformer 中 BatchMatmul 后接 Softmax 的标准算子链。Falcon-40B config: K=64, M=4096, N=4096。
  - **系统框架层**：Neuron Compiler 将高层 ML operator 图编译为 NKI kernel 调用序列。没有运行时调度框架的优化——kernel 优化完全依赖编译器自动生成的初始版本或人工手写的 NKI 代码。
  - **编译框架层**：Neuron Compiler 自动生成的 baseline kernel 分配 tiles 时，tile v 和 p 跨越两个嵌套循环存活，导致 SBUF 容量不足，触发 memory spilling（spill 到 HBM）。此外编译器不做跨循环的全局分析（如循环不变量外提），性能受限。
  - **kernel调度层**：Baseline kernel 使用固定 tile size（256 elements）和朴素 loop nest ordering（如 LHS transpose 在 i1 循环内重复执行 16 次）。HFU（Hardware FLOPs Utilization）仅 7.78%，memory write 达 1.07 GB。无自动的代数简化（如 θ-γλθ 仍为两次乘法和一次减法）或 intrinsic fusion。
  - **硬件架构层**：Trainium 1 单核 PeakMM 23.75 TFLOPS, PeakBW 440.2 GB/s, PeakVec 286.8 GFLOPS。Tensor/Vector/Scalar engine 并发运行，SBUF 每 partition 限 192KB，PSUM free dim 限 512。Baseline kernel 未充分利用 tensor engine（HFU 低），且大量 SBUF spilling 导致 HBM bandwidth 成为瓶颈。

  Baseline 核心缺陷：
  - (a) **优化空间巨大但探索效率低**：NKI kernel 需探索 memory layout、parallelization scheme、tiling、loop ordering 等多维度空间，人类需逐一尝试，时间成本高。
  - (b) **缺乏跨 iteration 的经验积累**：每次人工优化 kernel 从零开始，之前的优化经验无法系统性地迁移到新 kernel。
  - (c) **性能反馈周期长**：编译-运行-分析 loop 依赖人工介入，无法自动化规模化。
  - (d) **缺乏绝对性能标准**：传统 benchmark 只衡量相对 speedup，无法判断 kernel 是否已接近硬件理论峰值，优化方向不明确。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **AccelOpt：自改进 LLM agentic 系统**，通过 Planner-Executor-Summarizer 三代理工作流 + Beam Search + Optimization Memory，在无人工专家知识输入的情况下自主探索 NKI kernel 优化空间，迭代提升 kernel 性能。

  全栈执行例子（同样 BatchMatmul+Softmax kernel，Trainium 1 单核，AccelOpt B=6, N=12, K=2, T=16）：
  - **模型推理算法层**：同一 BatchMatmul+Softmax 算子链不变。AccelOpt 通过 Agent 理解算子语义（如识别 LHS transpose 在 i1 循环中不变），无需修改算法本身。
  - **系统框架层**：AccelOpt 在 Neuron Compiler 之上增加一层 meta-optimization 框架。Agentic workflow 不修改编译器，而是生成更优的 NKI kernel 源码作为编译器输入。分布式 profiling service 利用 Trainium core-level 和 machine-level 并行度批量测评 kernel。Roofline 模型计算每个 kernel 的 peak throughput percentage（T = max(Traffic_Min/BW, FLOPs_MM/Peak_MM, FLOPs_Vec/Peak_Vec)），提供绝对性能坐标系。
  - **编译框架层**：Agent 取代了人类专家的优化决策。Planner 分析 Neuron Profile 数据（HFU=7.78%, HBM write=1.07GB）识别 memory-bound 瓶颈 → 提出 "Hoist LHS Transpose Out of Reduction Loop" 优化计划 → Executor 实现 loop reordering（将 v7/v8/v9 transpose 外提到 i1 循环外，存入 global buffer v9_global，消除 16 次冗余计算）→ 进一步发现 recomputation 可消除 spilling（图 8: v 和 p tile 跨越两个循环导致 spill，通过 recompute v' 消除 spill，再通过移除额外 m loop 消除 recomputation 引入的额外 matmul）→ Summarizer 将 slow-fast kernel pair 提炼为通用经验 "Loop Invariant Code Motion for LHS Matrix Transposition"。
  - **kernel调度层**：AccelOpt 发现多种 kernel 级优化：(a) Peephole: θ-γλθ → (1-γλ)θ (代数简化), reciprocal(sqrt) → rsqrt (intrinsic fusion), x/(1+e^(-x)) → x·sigmoid(x) (利用 NKI 专用指令); (b) Loop 优化: tile size 256→512 (匹配 hardware optimal 128×512 moving 配置), loop fusion (将 i1 和 i5 循环融合为 16 次迭代的单个 i1 循环，减少嵌套开销)。优化后的 Mamba kernel 达到 54.6% peak throughput，超过人类专家最优版本 52.7%。
  - **硬件架构层**：同一 Trainium 硬件上，优化后的 kernel 通过降低 spilling (减少 HBM 访问)、提升 tensor engine utilization (HFU 从 7.78% 上升)、减少冗余计算，使 peak throughput 从 49% 均值提升至 61% (Trainium 1)。

  关键设计选择与 baseline 缺陷的对应：
  - **defect: 优化空间巨大但探索效率低** → 方案: Beam Search 每轮生成 B×N×K 个 kernel，从 B+N×K 个候选中选择 Top-B 进入下一轮。B=6, N=12, K=2 → 每轮 144 个 kernel，T=16 轮共约 2304 个 kernel 被探索。比人类手动逐一尝试效率高数个量级。
  - **defect: 缺乏跨 iteration 经验积累** → 方案: Optimization Memory（容量 ExpN=16，每轮追加 TopK=8 items）存储 slow-fast kernel pairs + LLM-summarized 通用优化策略。正负样本（positive/negative rewrites with tpos=1.04, tneg=1.15 阈值）均收录。Memory 在候选组内做 diversity 过滤（group by candidate and plan，仅取异常值）。实验证明 memory 使达到相同 speedup 的迭代数减少 16-17%。
  - **defect: 性能反馈周期长** → 方案: 分布式 profiling service 利用 Trainium core/machine 级并行，B×N×K 个 kernel 可同时 profiling。Neuron Profile 提供详细硬件级指标（HBM read/write bytes, spill, engine utilization 等），直接输入 Planner 作优化依据。自动 correctness check（||output - cpuref|| < tol × ||cpuref||）过滤错误 kernel。
  - **defect: 缺乏绝对性能标准** → 方案: NKIBench 基于 Roofline 模型计算每个 kernel 的 hardware peak throughput percentage。Traffic_Min 为所有 I/O tensor 的 byte 总量，FLOPs_MM 和 FLOPs_Vec 分别计算 matmul 和非 matmul 操作的理论算力上限。绝对坐标系使系统能判断 kernel 是否已接近硬件极限（如 82% peak 后速度 plateau 是因为已接近理论峰值而非探索停止）。
  - **额外设计: cost efficiency** → Open-source 模型（gpt-oss-120b + Qwen3-Coder-480B）实现与 Claude Sonnet 4 相当的性能提升（61% vs 61% peak throughput on Trainium 1），但成本仅为其 1/26。Beam search 比重复采样更有效（图 13），optimization memory 提升 cost efficiency 但不过度影响最终最优 kernel 性能。

## Cornserve Efficiently Serving Any-to-Any Multimodal Models (Cornfigurator)

- baseline方法是什么？
  **手动/专家驱动固定部署策略**：现有系统（vLLM, vLLM-Omni, SGLang-Omni, ModServe, EPD）使用预定义的固定部署策略：monolithic（全组件 colocation）、encoder-disaggregated（仅 encoder 解耦）、encoder-prefill-decode disaggregation（EPD）、或 fully disaggregated（全部组件解耦）。executor 级别配置（batch size, tensor parallelism degree, 实例数）需要人类专家手动调优。这些系统要么仅针对 A2A 的特例（如 ModServe 仅 MLLM, EPD 仅 encoder-prefill-decode），要么只提供解耦机制但不提供自动规划器。

  全栈执行例子（Qwen 3 Omni 30B on 16×A100-80GB，1/3 audio output，baseline=vLLM-Omni 专家调优方案）：
  - **模型推理算法层**：Qwen 3 Omni 的 DAG 组件图：E_img + E_vid + E_aud（多模态 encoder） → L_th（thinker LLM，自回归 text 生成） → L_ta（talker LLM，自回归 audio token 生成） → G_aud（vocoder，audio waveform 生成）。不同 request type 遍历不同子图（如有 audio output 则需 L_ta+G_aud，无则仅需 L_th）。
  - **系统框架层**：vLLM-Omni 采用固定解耦策略——预定义的组件分组方式，人工指定各 executor 的 GPU 分配、batch size、tensor parallel degree。无法根据 workload 中不同 request type 的比例自动调整分组或资源分配。例如 audio output 请求占 1/3 时，talker+voco 需要较多 GPU 资源，但专家调优的固定方案可能未能充分分配。
  - **编译框架层**：论文未明确说明编译框架层修改。使用标准 PyTorch/CUDA 编译路径。
  - **kernel调度层**：各 executor 独立运行，组件间通过 NCD（network collective communication）传输中间 tensor（~10ms 中位延迟）。vLLM-Omni 的 fixed plan 下，encoder executor 和 LLM executor 之间的数据传输 latency 固定，无自适应调度。
  - **硬件架构层**：16×NVIDIA A100-80GB GPU，NVSwitch 互联。GPU 分配固定，某些 GPU 可能因 colocation/batching 不当而利用率低。

  Baseline 核心缺陷：
  - (a) **固定策略无法适应模型和 workload 变化**：专家方案的 colocation/disaggregation 决策固定，无法根据 request type 分布（如 audio output 比例）、GPU budget 变化自动调整。图 3 显示即使对简单的单 encoder MLLM（InternVL 3 38B），不同 workload 下最优策略也显著不同——无 silver bullet。
  - (b) **全局延迟约束导致轻量 request type 不受保护**：使用单一全局延迟目标时，仅最重的 request type（如 audio output）约束生效，轻量 type（如 text output）可被无限制降级（图 5）。
  - (c) **缺乏 per-request-type 精细化推理**：将所有 request type 混在一起优化，无法为不同类型定制专用 subplan（如为 heavy video-input 请求准备 disaggregated video encoder 分支）。
  - (d) **搜索空间巨大（~500M candidate plans）但没有高效搜索机制**：手动探索不可行（Qwen 3 Omni + 16 GPU 产生 483M candidate plans），无粗到细评估剪枝 pipeline。
  - (e) **解耦非普遍有利**：将组件放到独立 GPU 上虽允许独立扩展，但消耗的 GPU 资源无法被其他组件使用（如 encoder 专用 GPU 无法存 LLM 的 KV cache），在某些场景下 monolithic 反而更优。固定策略无法根据具体情况权衡。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Cornfigurator：通用 Any-to-Any 多模态模型推理 Serving 的自动化部署规划器**。核心设计：(i) per-request-type 独立延迟约束和 goodput 优化——每种 request type 有独立延迟目标 L_t，最大化各 type goodput 之和；(ii) 系统化探索 colocation/disaggregation 组合——从 model DAG 枚举 simple→compound subplans→logical plans→physical plans，而非预设固定策略；(iii) 粗到细三阶段统计评估——Network flow（吞吐量上界）→ Monte Carlo（延迟估计）→ Request-level simulator（精确队列建模），每阶段剪枝劣化方案；(iv) 支持 subplan specialization 和 composition——允许不同 subplan 为不同 request type 子集优化，组合为完整 logical plan。

  全栈执行例子（同样 Qwen 3 Omni 30B on 16×A100-80GB，1/3 audio output，Cornfigurator 自动生成的 plan）：
  - **模型推理算法层**：同一 Qwen 3 Omni 模型定义不变。
  - **系统框架层**：Cornfigurator 接收 model definition（DAG）、configuration space（executor types）、workload（8 request types 及其比例）、GPU budget=16、per-type latency targets。Profiler 先对各 component 在目标硬件上 sweep batch size 和 parallelism degree，记录稳态吞吐+延迟。Planner 枚举后生成 physical plan: 1×(E_aud) + 4×(E_img+E_vid+L_th) + 11×(L_ta+G_aud)——将 audio encoder 解耦到独立 GPU，其余 encoder 与 thinker LLM colocation 为 4 个 executor，talker LLM 和 vocoder 共享 11 个 executor。Cornserve runtime 按此方案在 16 GPU 上部署 executor 实例。
  - **编译框架层**：论文未明确说明编译框架层修改。
  - **kernel调度层**：Plan 中 routing probabilities 决定各 request type 在各 parallel path 间的流量分配。Network flow phase 识别 bottleneck node，确保各 node 的 executor capacity 按 request type 比例加权后不超过 node 总容量。Monte Carlo 和 Simulator 进一步精细化 per-type goodput 估计，考虑 CPU-GPU overlap（encoder 的 CPU preprocessing 与 GPU 执行流水线化）、inter-type contention at shared nodes、occupancy-aware latency scaling（非 bottleneck node 按有效 batch size 缩放延迟）。
  - **硬件架构层**：同一 16×A100 GPU。Cornfigurator 的 plan 通过自动化的 colocation 决策（audio encoder 独立 GPU 以匹配其低吞吐特性，image+video encoder 与 thinker LLM colocation 以共享 GPU 资源），比 Fixed Disaggregation 减少 GPU 碎片。

  关键设计选择与 baseline 缺陷的对应：
  - **defect: 固定策略无法适应模型和 workload 变化** → 方案：Plan enumeration（Algorithm 1）从 model graph 系统化探索所有 colocation/disaggregation 组合（simple subplans → compound subplans → logical plans），不预设策略。图 18 显示当 InternVL 3 的 image input 概率从 25% 升至 75%，planner 自动从 monolithic 过渡到 encoder-disaggregated 再增大 batch size。
  - **defect: 全局延迟约束导致轻量 type 不受保护** → 方案：Per-type latency targets（§3.2）。Appendix A 证明当 L_t ∝ compute cost of type t 时，所有 type 的延迟约束 equally tight（scale factor ℓ_t 被约去，CDF 均在相同参数 L/ℓ_max 评估）。确保 Planner 不能以牺牲 text output 延迟为代价提升 audio output 吞吐。
  - **defect: 缺乏 per-request-type 精细化推理** → 方案：Subplan specialization（§4.2）。每个 simple/compound subplan 可仅覆盖部分 request type，在 logical plan 中组合（k_s=2）。图 10 的 Qwen 3 Omni on 16 GPU plan 展示了 compound subplan——一个分支用 disaggregated video encoder 服务 heavy video-input 请求，另一个分支用 monolithic 配置服务其余请求，共享 talker+vocoder executor。
  - **defect: 搜索空间巨大无高效探索** → 方案：Coarse-to-fine 三阶段评估+精确剪枝（§4.3, Algorithm 2）。Network flow（3.48s, 483M→1.95M candidates）剪枝冗余 GPU 配置，Monte Carlo（34.23s, 1.95M→25 candidates）剪枝 Pareto-suboptimal per-type goodput，Simulator（0.83s, 25→5 candidates）精确建模排队。总计 < 2 分钟完成，若全用 simulator 需 4400+ 小时。剪枝规则精确（仅丢弃保证冗余/劣化的 plan）。
  - **defect: 解耦非普遍有利** → 方案：Planner 在枚举阶段同时考虑 colocation（MERGE edge）和 disaggregation（KEEP edge），对每种组合评估 goodput。Qwen-Image（§6.4）上 planner 正确识别 monolithic 为最优（2-component model, LLM prefill 轻量且解耦会浪费 GPU），而 Full Disaggregation baseline 因强制解耦导致 GPU 碎片。
  - **额外设计: workload drift 自适应** → 当 request type 比例变化时，仅需 re-weight profiling 样本（无需重新 profiling），planner 重新规划耗时仅 single-digit seconds（§6.6）。GPU budget 变化也仅需重新运行规划（profiling 有效）。
  - **额外设计: 规划器 runtime-agnostic** → Cornfigurator 的 plan space 是 vLLM-Omni, ModServe, EPD, vLLM, Full Disaggregation 等所有 baseline 的严格超集，所有 baseline 方案可表达为 Cornfigurator plan。实验中将所有方案部署在同一 Cornserve runtime 上，消除框架实现差异的影响。

## Demystifying the Placement Policies of the NVIDIA GPU Thread Block Scheduler for Concurrent Kernels

- baseline方法是什么？
  **Round-robin scheduling assumption（先前的共识假设）**：在并发 kernel 执行场景下，GPU thread block scheduler 使用 round-robin policy 将 thread block 分配到 SM。此前的研究（GPGPU-Sim、Accel-Sim 等 GPU 模拟器，以及 Naghibijouybari et al. [11]、Amert et al. [2]、Li et al. [10] 等工作）均假设或观察到 thread block scheduler 使用 round-robin 策略轮询地将 block 分配到各个 SM。

  全栈执行例子（以 Turing GPU 上两个并发 kernel 的 round-robin 假设执行流程为例）：
  - **模型推理算法层**：论文未涉及 ML 推理。使用 purpose-built kernel 类别（L1-cache-dependent、compute-intensive、memory-intensive、transfer-bandwidth-dependent）。
  - **系统框架层**：标准 CUDA programming model。kernel 从不同 CUDA stream 发射，thread block scheduler 负责将 block 分配到 SM。Round-robin 假设下，scheduler 依次将 block 分配到 SM0, SM1, SM2, ...
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：Round-robin policy 假设：当 Kernel X 的 5 个 block 先占满 5 个 SM 后，Kernel Y 的第一个 block 释放时，scheduler 将 Kernel Y 的 block 依次轮询分配到 SM0 → SM1 → SM2 → SM3 → SM4 → SM0 → ...，按固定循环顺序，**不考虑各 SM 当前的资源可用性差异**。在单 kernel 场景中，由于所有 block 大小相同且行为相似，round-robin 与 most-room 的表现难以区分（各 SM 资源可用性基本相同）。
  - **硬件架构层**：标准 NVIDIA GPU（Pascal/Volta/Turing），thread block scheduler 是 NVIDIA 闭源硬件实现的 black-box。

  Baseline 缺陷（round-robin assumption 的问题）：
  - (a) **Round-robin 假设在并发 kernel 场景下是错误的**：当存在多个不同尺寸的 kernel 并发执行时，各 SM 的资源可用性因已resident 的 block 大小不同而异，round-robin 无法解释实际的 placement 行为。
  - (b) **GPU 模拟器精度受损**：GPGPU-Sim、Accel-Sim 等模拟器若假设 round-robin，在模拟并发 kernel workload 时会得出错误的 block distribution，导致性能预测不准。
  - (c) **无法预测"反直觉"的性能退化**：例如减少 1 个 thread/block 导致 3.58X execution time 增加（transfer-bandwidth-dependent kernel），这种看似矛盾的现象无法用 round-robin 解释。
  - (d) **缺乏对 concurrent kernel 性能的系统性理解**：现有调度研究关注 time-multiplexing（preemption）或 space-multiplexing（resource sharing），但均基于对硬件 scheduler 的不完全认知。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **通过实证测量推导 Most-Room Policy 并表征其性能影响**：论文通过设计精确控制 block 执行时间和 resource dimension 的两 kernel 实验，从真实硬件的 behavior 推导出 thread block scheduler 的 "most-room policy"。核心发现：scheduler 选择能容纳当前 kernel 最多 block 数量的 SM 来放置下一个 block，按 pre-defined device-specific ordering 打破平票。论文进一步设计了四类 purpose-built kernel 来表征 most-room policy 在不同 kernel 类别下的性能影响。

  全栈执行例子（以 most-room policy 推导实验，Pascal GPU，Figure 2 为例）：
  - **模型推理算法层**：论文未涉及 ML 推理。
  - **系统框架层**：标准 CUDA stream concurrency。Kernel X 从 stream 1 发射（5 blocks × 256 threads），Kernel Y 从 stream 2 延迟发射（3 blocks × 160 threads）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：
    1. **时序控制**：通过 `globaltimer` spin 使 Kernel X 的 block 按 SM id 顺序完成（B0 on SM0 先完成 → B4 on SM4 最后完成），在 SM0 空但 SM1-4 仍 busy 时发射 Kernel Y。
    2. **Most-room 决策**：Scheduler 计算每个 SM 可容纳的 Kernel Y block 数 = floor(SM 剩余 threads / Y.threads_per_block)。SM0 空（2048 free = 12 blocks），SM1-4 各含一个 X block（1792 free = 11 blocks）→ scheduler 选 SM0（most room）→ Y0→SM0。
    3. **资源重算**：Y0 占 160 threads 后，SM0 剩余 1888 = 11 blocks，与 SM1-SM4 平票 → tie-breaking order → Y1→SM0。
    4. **再次重算**：Y1 占后 SM0 剩余 1728 = 10 blocks，SM1-4 各有 1792 = 11 blocks → Y2→SM1。
    5. **结果**：Y0,Y1→SM0, Y2→SM1（most-room），而非 round-robin 的 Y0→SM0, Y1→SM1, Y2→SM2。
  - **硬件架构层**：通过 `smid` 寄存器读取 SM id，`blockIdx` 识别 block。实验揭示 limiting resources 包括 threads、shared memory、blocks/SM、warps/SM（论文声明可能还有其他未识别的因素）。

  性能影响表征（以 Turing GPU, L1-cache-dependent kernel 为例）：
  - **kernel调度层**：通过改变 Kernel B 的 threads/block（33 vs 32），触发 different limiting resource：33 threads → limiting=threads → all B blocks 分配到唯一的空 SM67（concurrent-isolated）；32 threads → limiting=blocks/SM → B blocks 分布到 8 个 SM，部分与 A 的 blocks colocate（concurrent-colocated）。
  - **性能结果**：1 thread/block 的不同导致 colocation，L1 cache contention 使 Kernel A 从 85ms→105ms (1.24X)、Kernel B 从 79ms→105ms (1.33X)，total time 从 85ms→105ms。

  关键设计选择与 baseline 缺陷的对应：
  - **defect: round-robin 假设错误** → 方案：通过 `globaltimer` 时序控制 + `smid` 位置追踪的实证实验设计，精确揭示 most-room policy。当 SM 间的资源可用性因已 resident block 尺寸不同产生差异时，most-room 与 round-robin 的行为明显不同（Figure 2），而单 kernel 场景两者几乎无区别（Section 4.4 解释 why round-robin 被长期误用）。
  - **defect: GPU 模拟器精度受损** → 方案：提供三种微架构（Pascal/Volta/Turing）的 most-room policy 详细参数，包括 limiting resource 识别（threads、shared memory、blocks/SM、warps/SM）和 device-specific tie-breaking ordering（Pascal=ascending, Turing=even-then-odds, Volta=device-specific），可直接用于改进模拟器中的 thread block scheduler 实现。
  - **defect: 无法预测反直觉性能退化** → 方案：通过四类 purpose-built kernel 的系统性实验（Section 5），展示 most-room policy 如何将微小的 kernel 参数变化（如 1 thread/block）放大为 significant performance degradation（1.33X-3.58X for L1-cache/transfer kernels），并解释 root cause（colocation → resource contention → specific resource type matters）。
  - **defect: 缺乏 concurrent kernel 性能的系统化理解** → 方案：识别出影响 concurrent kernel 性能的三个关键因素：(i) thread block scheduler 的 scheduling policy；(ii) 多种硬件资源（L1 cache、functional units、global memory bandwidth、PCIe bandwidth、TLB）的潜在竞争；(iii) kernel launch timing 等可能不可预测的因素。Section 5 按 kernel 类别分别分析这些因素的影响。
  - **额外发现: leftover policy 与 most-room 配合**：Leftover policy（Section 6）定义 when/which block 被调度（只有队列头 kernel 的 block 可被调度），most-room policy 定义 where 放置该 block。两者共同决定了 concurrent kernel 的调度行为空间。

## Dimple Discrete Diffusion Multimodal Large Language Model with Parallel Decoding

- baseline方法是什么？
  **自回归多模态大语言模型（AR MLLM）**，以LLaVA-NEXT为代表。训练使用causal attention mask + next-token prediction loss，逐token自回归生成。推理时每个forward step生成1个token，LLaVA-NEXT-7B训练数据约1.2M样本。

  全栈执行例子（LLaVA-NEXT回答"What is the common item in the two images?"）：
  - **模型推理算法层**：Vision encoder (CLIP ViT-L/14) 将两幅图像编码为visual tokens，经过projector映射到LLM embedding空间。Prompt tokens + visual tokens拼接为输入序列。使用causal attention：每个token只能attend到自身及之前的token。逐token自回归生成回答：先输出"In..."，再"the..."，依次生成直到[EOS]或max_length。total forward次数 = response token数（如64 tokens需要64次forward）。
  - **系统框架层**：自回归MLLM serving（如vLLM/SGLang）使用continuous batching + PagedAttention管理KV cache。每次forward迭代：所有batch中的请求各生成1个token → 更新KV cache → 检查是否[EOS] → 移出完成的请求。Prefill阶段一次处理prompt的KV cache计算。
  - **编译框架层**：论文未明确说明。通常使用PyTorch + FlashAttention-2进行高效attention计算。
  - **kernel调度层**：论文未明确说明。自回归decode阶段为memory-bound的GEMV操作（batch_size × 1 token），每次forward处理小矩阵向量乘，GPU利用率低。
  - **硬件架构层**：NVIDIA H100 GPU，无自定义硬件修改。

  Baseline缺陷：
  - (a) **逐token生成低效**：每个forward step仅生成1个token，response length 64需要64次forward，GPU在decode阶段利用率低（memory-bound GEMV）。
  - (b) **无法并行解码**：causal attention要求严格的左到右生成顺序，无法利用token间的独立性。
  - (c) **输出控制困难**：无法精确控制输出格式和长度。自回归模型依赖[EOS]终止，无法预先指定response长度；控制输出结构需要依赖instruction prompt或CoT的间接引导。
  - (d) **无法提前给出答案**：自回归模型必须按序生成全部tokens后才能到达最终答案，无法在处理中间推理步骤时提前给出结论。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Dimple：首个离散扩散多模态大语言模型（DMLLM）**。核心设计：(i) Autoregressive-then-Diffusion混合训练范式——先AR训练学习多模态能力，再Diffusion训练恢复并行解码；(ii) Confident Decoding——基于置信度阈值γ动态决定每步解码token数；(iii) Prefilling——复用prompt KV cache降低注意力复杂度；(iv) Structure Prior——预置特定位置token实现输出结构和长度精确控制。

  全栈执行例子（Dimple回答同样问题，使用Confident Decoding + Structure Prior）：
  - **模型推理算法层**：Vision encoder (Qwen2.5-VL ViT, 冻结) 编码图像 → 2层MLP projector → Dream DLM embedding空间。初始化：设定response_length=64，所有answer位置初始化为[MASK] token。Structure Prior预置特定位置："In the first image, there "、"In the second image, there "、"The common item in the two images is"（标记为已确定，不参与mask）。Forward #1（首个迭代）：bidirectional attention over全部L_prompt+L_answer个位置，所有未确定的[MASK]位置预测token分布。计算confidence c_t^(i) = max(softmax(logits)^(i))（使用pre-revision概率，不受temperature/top-p影响）。设置阈值γ：筛选c_t^(i)≥γ的位置，一次性更新多个高置信token（如同时解码9个token）。Fallback：若无位置超过阈值，随机选K个最不确定位置采样更新。Forward #2-#7：每次forward基于上一步结果，继续在未确定位置预测+筛选。第10步（共30步）：token "scissors"（最终答案）已完成解码——答案出现在完整response之前。最终30步完成（vs baseline 64步），实际迭代数仅为response_length的~1/2至1/3。
  - **系统框架层**：Prefilling实现：首次forward计算完整attention并保存prompt tokens的K/V。后续迭代复用保存的prompt K/V（仅计算answer部分的attention），复杂度从O((L_prompt+L_answer)²)降至O(L_answer²)。注意：由于DMLLM使用full bidirectional attention，answer token会attend到prompt tokens，prefilling理论上不是lossless（但实验证明性能下降仅平均0.8%）。生成不需要[EOS]终止——通过Padding token填充到预定义response_length。
  - **编译框架层**：论文未明确说明。使用PyTorch标准框架。
  - **kernel调度层**：论文未明确说明。Confident Decoding将多次memory-bound GEMV合并为更大的矩阵运算（每步同时处理多个token），提升了decode阶段的算术强度。
  - **硬件架构层**：NVIDIA H100 GPU集群（训练约100 GPU hours），单H100用于Prefilling消融实验。无自定义硬件修改。

  关键设计选择与baseline缺陷的对应：
  - **defect: 逐token生成低效 (a)** → 方案：Confident Decoding在高置信度时一次更新多个token。Table 5展示22个token仅需7次迭代（~1/3）；Table 6展示55个token需37次迭代。配合Prefilling，batch=32下TPS加速达7×（Table 3），batch=1下加速1.5×-2×。对比自回归需64步完成64-token response，Dimple仅需~30步。
  - **defect: 无法并行解码 (b)** → 方案：离散扩散模型使用full bidirectional attention，所有位置可同时attend和预测。训练时Phase II使用masked language modeling + absorbing-state diffusion，推理时所有[MASK]位置并行预测、基于置信度选择性更新。
  - **defect: 输出控制困难 (c)** → 方案：(i) Structure Prior允许预置任意位置的token，如"{date:"、"time:"、"}"定义JSON格式输出（Table 5）。(ii) Length Control——通过response_length参数精确控制输出长度，并在指定位置放置结束标记（如"Thus, the answer is \box{"在position[-12:-4]），模型自动调整推理跨度填满token预算（Table 6：length=16和32的两种配置）。
  - **defect: 无法提前给出答案 (d)** → 方案：扩散模型可在任意位置先解码出高置信token，不限于从左到右。Table 4展示正确答案"scissors"在第10步解码，而整个response在第30步才完成——答案先于完整推理步骤出现，这是自回归模型无法实现的。
  - **额外设计：训练不稳定性** → 方案：Autoregressive-then-Diffusion训练。纯扩散训练存在两个低效：(i) masked language modeling仅对masked token计算loss，监督信号覆盖率低于next-token prediction；(ii) 每个样本仅提供一个timestep的监督。AR-then-Diffusion先用AR训练建立多模态能力（更高监督信号利用率），再用Diffusion训练恢复并行解码。Table 2证明AT+DT在所有9个benchmark上优于纯DT，且缓解了Length Bias（纯DT在ChartQA上accuracy从42.7%→8.6%随response_length增加）。

## Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering

- baseline方法是什么？
  两种现有 computation-communication overlap 方法：(i) **Decomposition-based**（CoCoNet、Async-TP、Domino、Centauri 等）——将 GEMM 输出 tensor 分解为多个 subtensor，异步交替执行第 i 个 subtensor 的通信和第 i+1 个 subtensor 的计算。为确保数据地址连续以直接调用通信 API（如 NCCL），分解仅限于单一维度，与 GEMM tile 的二维分块范式不对齐，无法实现 tile 级细粒度重叠；且小 GEMM shape 时碎片化计算无法充分利用 GPU 资源。(ii) **Fusion-based**（FLUX、Comet、TileLink、cuBLASMp 等）——将通信原语直接融合进 GEMM kernel 内部，通过指令调度实现 tile 级重叠。但需要为每种通信原语手动实现定制融合 kernel（AllReduce、ReduceScatter、All-to-All 各需独立实现），且融合时因协调计算和通信 pipeline 可能需修改 tiling 策略或计算逻辑导致性能退化。

  全栈执行例子（以 Llama3-70B TP=8 推理中 GEMM+AllReduce，A800 GPUs，使用 Decomposition-based Async-TP baseline 为例）：
  - **模型推理算法层**：TP=8 下每 GPU 计算 GEMM 部分结果 → AllReduce 求和得到完整结果。
  - **系统框架层**：PyTorch 调用 Async-TP，将 GEMM 输出沿单维度分解为多个 subtensor。框架层负责管理 subtensor 的通信调度和 CUDA stream 同步。论文未明确说明上层 serving 框架修改。
  - **编译框架层**：论文未明确说明。使用标准 cuBLAS + NCCL API 调用，无编译优化。
  - **kernel调度层**：GEMM 被分解为多个独立的小 GEMM kernel（fragmented GEMMs），每个小 GEMM 调用 cuBLAS 执行。Subtensor 通信通过 NCCL API。第 i 个 subtensor 的 NCCL 通信与第 i+1 个 subtensor 的 cuBLAS GEMM 并发执行。但分解为 subtensor 时仅沿单一维度（保证地址连续），而 GEMM tile 是二维分块——tile 完成顺序（wave pattern）与分解维度不匹配，先完成的 tile 不能立即通信，必须等整个 subtensor 完成。Tile 级重叠机会丧失。小 K 值时碎片化 GEMM 无法填满 SM。
  - **硬件架构层**：A800 GPU，NVLink 互联。GEMM 在 Tensor Core 上执行，通信通过 NVLink。Decomposition-based 的 subtensor 通信数据量较小，可能导致带宽利用不足。

  Baseline 核心缺陷：
  - (a) Decomposition-based **无 tile-wise overlapping**：限于单维分解，与 tile 的 2D 分区不对齐，无法利用 tile 是最小并行数据单元的事实。已完成 tile 无法立即触发通信。
  - (b) Decomposition-based **interfere 计算**：GEMM 被碎片化为多个小 kernel，小 K 值时 GPU 利用率不足，overlap 带来的通信隐藏收益被计算性能损失抵消。
  - (c) Fusion-based **通信不通用（无 communication agnosticism）**：每种通信原语需要定制融合实现（AllReduce、ReduceScatter、All-to-All 各不同），重复开发成本高。
  - (d) Fusion-based **干扰计算**：融合时协调计算-通信 pipeline 可能需要改变 tiling 策略或计算逻辑，引入额外调优需求，可能导致性能退化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FlashOverlap —— 基于 signaling 的 computation-communication overlap**：核心思路是在 GEMM kernel 中嵌入轻量信号机制，当 tile 完成时发送信号触发通信，同时 GEMM 继续执行剩余 tile（interference-free computation）。在此基础上：(i) wave-wise signaling timing —— 利用 GEMM 执行的 wave pattern（多个 tile 几乎同时完成，差异 <5% wave 时长），以 wave 而非单个 tile 为信号单位提升通信带宽利用率；(ii) tunable wave grouping —— 将连续 wave 组合为 group，在重叠机会和通信分段之间优化；(iii) pre/post-communication reordering —— pre-reordering 将非连续地址的 tile 按执行顺序重排为连续地址以直接调用 NCCL API（communication agnosticism），post-reordering 在通信后恢复正确数据顺序；(iv) predictive search —— 剪枝设计空间并通过延迟预测器实时搜索最优 wave group partition。

  全栈执行例子（同样 Llama3-70B TP=8 GEMM+AllReduce，A800 GPUs，FlashOverlap）：
  - **模型推理算法层**：同一 TP=8 GEMM+AllReduce 计算逻辑不变。不修改模型结构和计算语义。
  - **系统框架层**：FlashOverlap 替换 vLLM/Megatron-LM/xDiT 中的原始 linear layer + 通信对。端到端 LLM 推理（vLLM）直接调用 FlashOverlap 的 GEMM+overlap 实现。论文未明确说明框架层调度修改。
  - **编译框架层**：论文未明确说明。GEMM kernel 基于 CUTLASS 模板，使用标准 CUDA 编译路径。
  - **kernel调度层**：
    1. **GEMM kernel（Stream A）**：单一 GEMM kernel 完整执行（不碎片化），main loop 不变。每个 tile 完成时，epilogue 中执行 pre-communication reordering（将 tile 数据按执行顺序散射到连续地址的通信 buffer），同时 atomicAdd 更新 counting table 中对应 group 的计数。
    2. **Wave pattern 利用**：T 个 wave 依次完成（T = tile_num / SM_num），counting table 记录每个 group 的完成 tile 数。
    3. **Signaling kernel（Stream B）**：周期性查询 counting table。当 group G_j 计数达到 |G_j|（该 group 的 wave 数 × wave 内 tile 数）时，调用 NCCL API 对重排后的连续 buffer 执行通信。同时 Stream A 中 GEMM 的后续 wave 继续执行（interference-free）。
    4. **Post-communication reordering**：通信完成后，fused 到后续 RMSNorm kernel 中根据 mapping table 恢复数据原始顺序。
    5. **Tuning**：predictive search 在已知 GEMM size 和 bandwidth curve 后，搜索使 overlap 延迟最小的 wave group partition，选择如 (1, 2, 2) 的 partition（分别在第 1/3/5 个 wave 后触发通信）。
  - **硬件架构层**：同一 A800 GPU。GEMM 在 Tensor Core 上执行（main loop 不受干扰），通信通过 NVLink。Two CUDA streams 实现 concurrency：while GEMM 的 wave 2-3 在 Tensor Core 上计算，wave 1 的数据通过 NVLink 通信。

  关键设计选择与 baseline 缺陷的对应：
  - **defect: Decomposition-based 无 tile-wise overlapping (a)** → 方案：Wave-wise signaling —— tile 完成后通过 counting table 信号立即识别（tile-wise），但以 wave 为单位触发通信以保证带宽利用率（兼顾 overlapping opportunity 和通信效率）。对比 decomposition-based 必须等整个 subtensor 完成才能触发通信。
  - **defect: Decomposition-based interfere 计算 (b)** → 方案：Signaling 机制在主 GEMM 外部运行（另一 CUDA stream），GEMM main loop 完整保留不变。Counting table 的 atomicAdd 在 epilogue 中仅增加 ~0.07% GEMM 开销（A800 tile-level）。GEMM 不碎片化，GPU 利用率与无 overlap 时相同。
  - **defect: Fusion-based 通信不通用 (c)** → 方案：Pre-communication reordering —— 将按执行顺序的非连续 tile 重排为连续地址，无需修改通信库即可直接调用 NCCL API。所有通信原语（AllReduce、ReduceScatter、All-to-All）复用同一套 signaling + reordering 机制，仅 reordering pattern 不同（tile 级/subtile 级/subtoken 级）。Communication agnosticism 得以实现。
  - **defect: Fusion-based 干扰计算（tiling 策略修改）(d)** → 方案：Signaling 不改变 GEMM 的 tiling 策略或计算逻辑。Main loop 完全由 CUTLASS profiler 最优配置驱动，epilogue 中仅增加 reordering scattering 操作（0.07%-0.68% 开销）。Post-communication reordering 融合到后续必须执行的 element-wise kernel 中（RMSNorm 开销 7.46%-9.63%）。
  - **defect: 固定 overlap 策略在变 workload 下非最优** → 方案：Predictive search —— 根据 GEMM size 和 bandwidth curve 自动搜索最优 wave group partition。搜索空间从 2^{T-1} 通过剪枝约束 |G_1|≤2, |G_P|≤4 降低，延迟预测器误差 <5%（平均 3.4%），搜索 partition 达到穷举 >99% 性能。RTX 4090 + AllReduce 上仅 4% 的 case 最优为单 wave group（即 baseline partition），平均 17.34% 性能差距证明了 tuning 的必要性。

## EPD-Serve A Flexible Multimodal EPD Disaggregation Inference Serving System On Ascend

- baseline方法是什么？
  多模态推理的 **monolithic 架构**（以 vLLM v0.11.0 为代表）：将 Encode（视觉编码器处理图像/视频）、Prefill（LLM 首次前向生成首 token 及 KVCache）、Decode（自回归逐 token 生成）三个阶段串行绑定在同一硬件资源上执行，不存在阶段级别的逻辑隔离或物理资源划分。

  全栈执行例子（openPangu-7B-VL on Ascend Atlas 800I A2，处理一条含图像的多模态请求）：
  - **模型推理算法层**：ViT (0.7B) 编码图像 → 特征 token 序列 V_m；文本 prompt token 化与 V_m 拼接 → 输入 LLM (7B) 执行 Prefill（生成首 token O_1 + 构建 KVCache KV1）→ 自回归 Decode（基于 O_i 和 KV_i 生成 O_{i+1} 直至 max_length 或 <eos>）。
  - **系统框架层**：vLLM v0.11.0 monolithic——E/P/D 三阶段在同一 NPU 上串行执行。PagedAttention 管理 KV cache。请求间通过 continuous batching 共享 Decode 阶段。没有按模态或阶段分拆的调度路径（多模态和纯文本请求混在同一队列）。
  - **编译框架层**：论文未明确说明。使用 Ascend CANN（Compute Architecture for Neural Networks）编译框架，标准 PyTorch 前端。
  - **kernel调度层**：Ascend NPU 上 MatMul、Attention 等算子由 CANN 运行时调度到 AI Core、AI Vector 等计算单元，无阶段级算子系统调度优化。论文未明确说明 kernel 级细节。
  - **硬件架构层**：华为 Ascend Atlas 800I A2，单 NPU 64 GB HBM。AI Core（矩阵/向量密集计算）+ AI Vector（AllReduce 等通信算子）。E/P/D 全部串行占据同一 NPU 资源，无跨 NPU 通信优化。

  Baseline 缺陷：
  - (a) **阶段耦合导致执行干扰**：视觉编码（compute-heavy Encode）和文本 Prefill 共享同一 NPU，无隔离机制。多模态请求可能阻塞纯文本请求，推高 TTFT，并打乱 Decode 调度节奏，恶化 TPOT 和整体吞吐。
  - (b) **统一并行策略不匹配异构阶段需求**：Encode 偏好数据并行或序列并行，Decode 偏好张量并行以降低延迟。Monolithic 的统一并行策略无法为每个阶段单独优化，限制扩展性。
  - (c) **串行执行阻止资源复用**：E/P/D 严格串行，尽管三阶段在计算-访存特征上互补（Encode compute-heavy + Decode memory-heavy），大量 NPU 计算资源在阶段切换时闲置。
  - (d) **跨阶段 tensor 传输无优化**：Monolithic 无跨 NPU 的 E-P 特征传输和 P-D KV 传输需求（同一 NPU 内存共享），但扩展到解耦场景时缺少传输优化机制。
  - (e) **无模态感知路由**：多模态请求和纯文本请求无差别的混合调度，高负载多模态请求会抢占资源影响纯文本请求的延迟。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **EPD-Serve：将多模态推理 pipeline 按 E/P/D 三阶段解耦为独立可调度实例，配合异步跨阶段 tensor 传输优化和灵活物理共置策略**。核心设计：(i) E-P-D 阶段级解耦——将 Encode/Prefill/Decode 拆分为独立实例进程，通过 Proxy 统一路由；(ii) E-P 异步特征预取——基于 Mooncake Store 构建 MM Store（hash→feature 缓存池），事件驱动只传 hash 不传全量特征，异步预取隐藏通信延迟；(iii) P-D 分层分组 KV 传输——按 Transformer 层分组打包 KV cache，延迟调度对齐通信与 Prefill 计算；(iv) 模态感知多路径调度——按请求模态分派到 E-P-D 完整管道或 P-D 纯文本管道；(v) 灵活物理共置——逻辑隔离 + 硬件空间复用（AI Core/AI Vector 算子级并行）。

  全栈执行例子（openPangu-7B-VL on Ascend Atlas 800I A2, (E-P)-D 部署, 2 NPU, ShareGPT-4o, 10 req/s）：
  - **模型推理算法层**：同一 ViT + LLM 推理 pipeline。Encode 生成 V_m ∈ R^{n×3584}，Prefill 计算 KV cache，Decode 自回归生成。不修改模型结构与计算逻辑。
  - **系统框架层**：EPD-Serve 将 vLLM monolithic 替换为三实例架构——(E-P) 实例处理 Encode+Prefill，D 实例独立处理 Decode。Proxy 接收请求 → 模态感知路由（多模态→E-P-D 管道，纯文本→P-D 管道）→ 实例级最少负载优先调度。MM Store 缓存已编码的多模态特征（key=hash(input)，value=feature vector）。E-P 异步预取：Encode 完成后仅发 hash 事件；Prefill listener 收到后从 MM Store 检索并加载特征。P-D 分层传输：Prefill 计算 L+1 层时异步传输 L 层 KV cache 至 Decode 实例。分组打包减少握手频率，延迟调度避开通信峰值。
  - **编译框架层**：论文未明确说明。使用 Ascend CANN 编译框架，未修改底层编译路径。
  - **kernel调度层**：NPU 上的算子调度由 Ascend CANN 运行时管理。EPD-Serve 在算子层面的关键优化是**物理共置空间复用**：(E-P) 共置 NPU 上，当 Prefill 的 MatMul（AI Core）完成而该阶段等待 P-D 传输时，Encode 的 AllReduce（AI Vector）可利用空闲 AI Core 执行；反之亦然。通过算子级硬件资源互补（operator complementarity 详见 Figure 6），减少 NPU 的 idle 周期。论文未明确说明自定义 kernel。
  - **硬件架构层**：Ascend Atlas 800I A2，每 NPU 64 GB。利用 AI Core（矩阵乘）和 AI Vector（规约/通信）的异构计算单元实现算子级并行复用。(E-P)-D 中 Encode+Prefill 共置 1 NPU，Decode 独占 1 NPU。跨 NPU 通信通过 Mooncake Store 的异步传输接口实现。

  关键设计选择与 baseline 缺陷的对应：
  - **defect: 阶段耦合导致执行干扰 (a)** → 方案：E/P/D 拆分为独立实例进程，通过 Proxy 统一路由和负载均衡。多模态请求和纯文本请求按模态分派到不同管道，隔离跨模态的调度干扰。
  - **defect: 统一并行策略不匹配异构阶段需求 (b)** → 方案：每个阶段实例独立配置并行策略和资源分配，可按需弹性伸缩（如 Encode 实例多副本、Decode 实例张量并行）。
  - **defect: 串行执行阻止资源复用 (c)** → 方案：物理共置 + 空间复用——逻辑层独立调度，物理层共享 NPU。通过 operator-level co-location（Figure 6），将硬件资源需求差异大的算子（MatMul vs AllReduce）在时间线上交错复用，提升 NPU 利用率。
  - **defect: 跨阶段 tensor 传输无优化 (d)** → 方案：(i) E-P 异步特征预取：MM Store 缓存 + 仅传 hash → Prefill 异步检索本地缓存，transmission overlap ratio 接近 100%（主流分辨率下）；(ii) P-D 分层分组 KV 传输：overlap ratio 从 15.27%→98.78%（seq_len=1024），bandwidth utilization 提升 58%。
  - **defect: 无模态感知路由 (e)** → 方案：多路径调度——按请求模态分派到 E-P-D（多模态）或 P-D（纯文本）管道，配合实例级最少负载调度，实现异构流量分离和动态负载均衡。
  - **额外设计：灵活部署拓扑**：支持 E-P-D / EP-D / ED-P / E-PD / (E-P)-D / (E-D)-P / (E-PD) 等拓扑按 SLO 需求切换：(E-P)-D 兼顾低 TTFT+低 TPOT（balancing）；(E-D)-P 优化 TTFT（first-token priority）；(E-PD) 最大化吞吐（throughput priority under relaxed SLO）。

## Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding

- baseline方法是什么？
  **Vanilla Masked Diffusion Model (MDM) with τ-leaping 独立并行解码**：当前开源的Diffusion LLM（LLaDA、Dream）使用基于τ-leaping近似的掩码扩散模型进行序列生成。在推理时，模型从全[MASK]序列开始，通过多步迭代逐步将[MASK]token替换为真实token。默认最优策略是每步解码1个token（顺序解码），因为τ-leaping虽然允许一次解码多个token，但存在conditional independence assumption问题——多token从独立边际分布中采样，破坏了token间真实联合分布中的依赖关系（如"high card" vs "high house"的不合理组合）。同时，由于Diffusion LLM使用full bidirectional attention，无法像自回归模型那样使用KV Cache复用之前的attention计算结果，每步都需要对全序列重新计算attention。

  全栈执行例子（LLaDA-Instruct GSM8K 5-shot, gen_len=256, A100 GPU）：
  - **模型推理算法层**：MDM使用absorbing-state离散扩散（Equation 1: q_{t|0} = Cat(x_t^i; (1-t)δ_{x_0^i} + tδ[MASK])），loss为MDM ELBO（Equation 2）。推理时使用τ-leaping近似反向过程（Equation 3），每步选择置信度top-1的token解码（1 token/step），共需约256步完成生成长度256。
  - **系统框架层**：直接使用LLaDA官方inference脚本，不使用任何serving框架（无vLLM/TensorRT-LLM）。batch size=1单请求推理。
  - **编译框架层**：论文未明确说明。使用标准PyTorch forward pass，无自定义编译优化。
  - **kernel调度层**：标准PyTorch attention实现（full bidirectional attention），每步执行Q·K^T/V计算于全序列矩阵（尺寸(|p|+L)×d），无KV cache无法复用前缀attention结果。256 step各自独立执行全注意力计算。
  - **硬件架构层**：NVIDIA A100 80GB GPU，无自定义硬件。吞吐量约6.7 tok/s（GSM8K 5-shot gen_len=256）。
  
  Baseline缺陷：
  - (a) **无KV Cache导致重复全注意力计算**：每步都需对prompt + 已生成 + 未生成的全序列重新计算Q·K^T和softmax，计算量O(T·(|p|+L)²·d)，T为总解码步数。当prompt较长（如8-shot）和生成长度较大（512/1024）时开销巨大。
  - (b) **τ-leaping并行解码破坏token依赖**：多token同时从p(X|E)=Π_i p_j(X_{i_j}|E)独立采样，忽略了真实联合分布p(X|E)中包含的token间条件依赖。导致生成不合理的token组合，质量随每步并行token数增加而显著下降。
  - (c) **无动态并行控制机制**：LLaDA的baseline要么全顺序（1 token/step，慢但准确），要么固定top-K解码（K token/step，快但质量下降），缺少根据模型置信度自适应调节并行粒度的机制。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Fast-dLLM：通过Block-wise Approximate KV Cache + Confidence-Aware Parallel Decoding 两种互补技术，训练无关地加速Diffusion LLM推理**。核心设计：(i) 分块生成（block-wise generation）+ 近似KV Cache（PrefixCache/DualCache），利用相邻步KV激活的高余弦相似度实现cache复用；(ii) 置信度感知并行解码（threshold/factor策略），通过理论保证（Theorem 1）仅在高置信度时并行解码多token，平衡速度与质量。

  全栈执行例子（LLaDA-Instruct GSM8K 5-shot, gen_len=256, A100 GPU, Fast-dLLM PrefixCache+Threshold τ=0.9, B=32）：
  - **模型推理算法层**：同一LLaDA MDM模型结构和权重，不修改模型参数（训练无关）。流程变为：
    1. 首步全序列forward pass，缓存prefix（prompt部分）的K/V矩阵
    2. 将生成拆分为K=⌈256/32⌉=8个块，每块最多T步解码
    3. 块k内：复用prefix K/V cache，仅对块内token执行attention→置信度计算→阈值过滤→多token并行解码
    4. 块k完成后：全序列forward pass，更新prefix K/V cache（与标准forward融合，无额外开销）
    5. 总步数显著减少（平均约40步 vs baseline 256步），吞吐量达到54.4 tok/s（vs baseline 6.7 tok/s，8.1×加速）
  - **系统框架层**：Fast-dLLM v1代码在PyTorch层实现，修改LLaDA推理loop：添加cache管理（存储/复用prefix K/V或prefix+suffix K/V），替换逐token顺序解码为块内自适应并行解码。不依赖serving框架。
  - **编译框架层**：论文未明确说明。标准PyTorch eager execution，无编译优化修改。
  - **kernel调度层**：attention计算量从O(T·(|p|+L)²·d)降至近似O(|p|²·d + K·T'·(B²+|p|·B)·d)，其中B为块大小（32），T'为每块内步数（远小于原始T），K为块数。DualCache进一步消除suffix attention计算，仅保留B×B块内自注意力。
  - **硬件架构层**：同一NVIDIA A100 80GB GPU，无自定义硬件。Fast-dLLM吞吐量54.4 tok/s vs baseline 6.7 tok/s（8.1×加速，GSM8K gen_len=256），8-shot gen_len=1024时DualCache达27.6×加速（0.7→19.3 tok/s）。

  关键设计选择与baseline缺陷的对应：
  - **defect (a): 无KV Cache → 重复全注意力计算** → 方案：Block-wise Approximate KV Cache。利用观察——相邻步KV激活余弦相似度接近1（Figure 3），在块内复用prefix/suffix K/V。块完成时更新cache（与解码forward融合，无额外计算）。块大小32在速度-精度间取得最佳折中（Figure 4）。DualCache变体进一步缓存suffix（全[MASK]），消除交叉注意力计算。
  - **defect (b): τ-leaping独立采样破坏token依赖** → 方案：Confidence-Aware Parallel Decoding。理论分析（Theorem 1）证明：当每token置信度>1-ε且(n+1)ε≤1时，argmax的乘积边际分布等价于argmax的真实联合分布。基于此，设计threshold策略（仅解码c_i>τ的token）和factor策略（动态选择满足(n+1)(1-c^(n))<f的最大n个token），在高置信度时安全并行解码多token，低置信度时保守解码。
  - **defect (c): 无动态并行控制** → 方案：threshold和factor两种策略都根据当前步的模型置信度水平动态决定并行解码的token数量。Threshold策略自适应在1到B个token之间调节；Factor策略通过理论绑定量(n+1)(1-c^(n))进一步精确控制并行度。对比固定token-per-step baseline，动态策略在相同accuracy水平下decodes significantly fewer NFEs（Figure 5c, Figure 8c）。
  - **额外设计：理论与实践的桥梁**：Theorem 1不仅给出argmax等价条件，还给出L_p距离和KL散度的上界（D_TV < (3n-1)ε/2, D_KL < (n-1)[H_b(ε)+ε·ln(|V|-1)]），量化了乘积分布对真实联合分布的逼近程度，为实际部署中选择阈值/因子提供了理论基础。
  - **额外设计：Prefill长度和生成长度的加速放大**：由于cache复用与序列长度成正比，更长的prefill（8-shot vs 5-shot）和更长的生成长度（1024 vs 256）带来更大的加速比（DualCache: 27.6× at 8-shot gen_len=1024 vs 19.6× at 5-shot），使方法在few-shot和长文本生成场景中价值更大。

## FlashAttention-2 Faster Attention with Better Parallelism and Work Partitioning

- baseline方法是什么？
  **FlashAttention v1（Dao et al., 2022）**：IO-aware exact attention，通过tiling + online softmax将attention计算融合为单个CUDA kernel，避免$N\times N$ attention矩阵在HBM的materialization。前向在block-wise计算中每次内迭代都做`diag(ℓ)^{-1}` rescale维护正确输出，后向存储row-wise max m和row-wise sum ℓ用于softmax梯度重计算。

  全栈执行例子（GPT-style training, N=8K, d=128, 32 heads, batch=2, A100 80GB）：
  - **模型推理算法层**：Exact softmax(QK^T/√d)V。FlashAttention v1通过tiling + online softmax保证数值等价，Algorithm 1的forward loop结构：外循环over KV blocks, 内循环over Q blocks。
  - **系统框架层**：PyTorch extension (`flash_attn_func`)，替换HuggingFace/Megatron-LM中标准attention调用。框架无需修改。
  - **编译框架层**：论文未明确说明。手写CUDA kernel，非编译器自动生成。
  - **kernel调度层（关键缺陷）**：
    (a) **并行度不足**：仅parallel over batch和head维度（1 thread block per head），对长序列场景（batch小, head少）occupancy低——例如N=8K, head_dim=128, 32 heads, batch=2时仅64 thread blocks，远低于A100的108 SMs，GPU利用率不足。
    (b) **Warp划分低效（split-K）**：在一个thread block内，FlashAttention v1将K和V split到4个warp，Q对所有warp可访问。每个warp计算部分QK^T后需将各自的partial results写入shared memory、同步、然后累加——额外的shared memory reads/writes成为瓶颈。
    (c) **Non-matmul FLOPs开销**：每次内迭代都对已累积的output做`diag(ℓ)^{-1}` rescale（elementwise multiply, non-matmul），且后向需同时存储m和ℓ（2×N个scalars per head）。Non-matmul FLOP虽占比小但耗时（A100上non-matmul吞吐仅19.5 TFLOPs/s vs matmul 312 TFLOPs/s，贵16×），拖累整体吞吐，forward仅达30-50% peak，backward仅25-35% peak。
  - **硬件架构层**：A100 GPU，108 SMs，192KB SRAM/SM，HBM带宽1.5-2.0TB/s。FlashAttention v1 block size受SRAM限制（B_c ≈ M/(4d), B_r = min(B_c, d)），长序列下每个thread block循环次数T_c = ceil(N/B_c)增大。

  Baseline缺陷：
  - (a) **Occupancy不足**：仅batch×heads个thread blocks，长序列场景下远少于SM数量，GPU compute units空闲。
  - (b) **Shared memory通信开销**："split-K" warp划分导致warp间需通过shared memory同步和累加partial results。
  - (c) **Non-matmul FLOPs过多**：每次迭代rescale output，减少可用于matmul的时间比例（matmul:non-matmul吞吐比16:1）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FlashAttention-2**：三种改进分别对应三个缺陷：
  (i) **Algorithm tweak（对应缺陷c）**：前向改维护"un-scaled" output，所有KV blocks处理完后一次性`diag(ℓ)^{-1}` rescale，消除每次迭代的output rescale non-matmul操作。后向仅存logsumexp `L = m + log(ℓ)` 替代 (m, ℓ)，减少register使用和non-matmul计算。效果：non-matmul FLOPs减少，更多时间花在matmul上。
  (ii) **Sequence length parallelism（对应缺陷a）**：前向外循环embarrassingly parallel，不同thread block处理不同row block（对应不同sequence position的output chunk），无需同步。后向不同thread block处理不同column block，仅dQ更新需atomicAdd。thread block数从batch×heads增至batch×heads×T_r（或T_c），例如N=8K, B_r=128时T_r=64，总thread blocks = 2×32×64 = 4096 >> 108 SMs，occupancy大幅提升。
  (iii) **Avoid split-K warp partitioning（对应缺陷b）**：前向改为split Q across 4 warps（K/V所有warp共享），每个warp计算其Q slice的完整output，无需warp间通信。后向同样避免split-K。消除shared memory读写瓶颈。

  全栈执行例子（同样GPT-style, N=8K, d=128, 32 heads, batch=2, A100 80GB）：
  - **模型推理算法层**：Exact softmax(QK^T/√d)V定义不变。Algorithm 1 tweak：un-scaled output维护 + final rescale + 仅存logsumexp L，数学上等价（Theorem 1, Dao et al. 2022）。
  - **系统框架层**：同FlashAttention v1，`flash_attn_func(q,k,v,causal=True)`作为drop-in replacement。新增GPT端到端训练验证（HuggingFace GPT-1.3B/2.7B），框架无需修改。
  - **编译框架层**：论文未明确说明。CUTLASS 3.x提供底层building blocks（TileIterator, Collective，etc），手写kernel而非编译器自动生成。论文讨论了未来方向：让compiler自动做这些优化。
  - **kernel调度层（三层改进的集中体现）**：
    - **Thread block调度**：Forward: 4096 thread blocks（batch×heads×T_r = 2×32×64），每个处理1个row block的所有KV blocks（j=1..64）。Backward: 4096 thread blocks（batch×heads×T_c），每个处理1个column block的所有row blocks（i=1..64），dQ通过atomicAdd合并。
    - **Warp内划分**：Forward时4 warps per thread block，Q按row split到4 warps（各32 rows of B_r=128），K_j/V_j在shared memory所有warp可见。每个warp: `S_warp = Q_warp @ K_j^T`（32×128, Tensor Core）→ rowmax/exp/rowsum（CUDA core）→ `O_warp = diag(exp(m_old-m_new))·O_warp + P_warp @ V_j`（Tensor Core）。**零warp间通信**——每个warp独立产出output slice。vs FlashAttention v1 split-K需写partial results到shared memory、同步、累加。
    - **Memory**：SRAM layout: Q_i slice per warp in registers, K_j[128,128] (32KB) + V_j[128,128] (32KB) in shared memory, O_tilde[128,128] (32KB) across registers, ℓ[128]/m[128] in registers。S_warp[32,128]和P_warp[32,128]在registers中。Total ~96KB shared memory + registers，fit 192KB SRAM。
    - **解码阶段**：KV cache loading split到多个thread blocks并行加载以saturate HBM bandwidth。写partial results到HBM后通过separate reduce kernel合并（因为thread blocks间无法直接通信）。
  - **硬件架构层**：相同A100。改进不在硬件，而在kernel调度充分利用已有硬件：通过更多thread blocks填满108 SMs，通过避免split-K减少shared memory traffic，通过减少non-matmul FLOPs增加Tensor Core利用率。

  效果量化（forward pass, A100, head_dim=128, causal mask）：
  - FlashAttention v1: ~30-50% peak FLOPs/s（~94-156 TFLOPs/s）
  - FlashAttention-2: ~50-73% peak FLOPs/s（~156-228 TFLOPs/s），约2× speedup
  - End-to-end GPT-2.7B 8k context: 225 TFLOPs/s (72% model FLOPs utilization) vs FlashAttention v1 175 TFLOPs/s vs 无FlashAttention 80 TFLOPs/s

## FlashAttention-3 Fast and Accurate Attention with Asynchrony and Low-precision

- baseline方法是什么？
  **FlashAttention-2 (Dao, 2023)**：同步执行的tiled block-wise exact attention kernel。核心设计：(i) 将Q沿seqlen维度分块以增加并行度（vs FlashAttention-1仅并行化batch和heads）；(ii) 内循环沿KV blocks迭代，每一步串行执行QK^T GEMM → wait → softmax → PV GEMM → wait；(iii) 所有warps统一角色，同时执行数据搬运和计算；(iv) 仅支持FP16/BF16精度；(v) 基于Ampere架构设计，未利用Hopper特有的TMA、WGMMA异步、FP8 tensor core、setmaxnreg等能力。
  
  全栈执行例子（H100 GPU, BF16 forward, N=8192, d=128, 16 heads）：
  - **模型推理算法层**：标准scaled dot-product attention $\mathbf{O} = \text{softmax}(\mathbf{QK}^T/\sqrt{d})\mathbf{V}$，通过tiled block-wise online softmax实现exact computation。FlashAttention-2并行化策略：outer loop over Q blocks（$T_r$路并行，不同CTA处理不同Q tiles），inner loop over KV blocks（sequential per CTA）。
  - **系统框架层**：PyTorch集成，`flash_attn_func(q,k,v)`作为drop-in replacement。HuggingFace Transformers、GPT-NeoX等框架通过替换attention模块调用。框架对底层kernel执行无细粒度控制——kernel内同步/异步调度对框架透明。
  - **编译框架层**：论文未明确说明。CUDA C++手写kernel（基于CUTLASS或自编），非编译器自动生成。FlashAttention-2 in Triton版本利用Triton compiler自动tile和调度，但未使用Hopper-specific指令（TMA/WGMMA异步）。
  - **kernel调度层（核心缺陷）**：单个同步CUDA kernel。内循环迭代j：（1）从HBM加载$\mathbf{K}_j$, $\mathbf{V}_j$到SMEM；（2）Tensor core MMA: $\mathbf{S}_{ij}=\mathbf{Q}_i\mathbf{K}_j^T$ → warp-level同步等待 → （3）CUDA core softmax: rowmax + EX2 + rowsum + rescale → 等待$\mathbf{V}_j$加载完成 → （4）Tensor core MMA: $\mathbf{O}_i += \tilde{\mathbf{P}}_{ij}\mathbf{V}_j$ → 同步等待。问题：(a) Tensor core算完BMM1后进入idle等待softmax完成（softmax throughput仅3.9 TFLOPs/s vs matmul 989 TFLOPs/s）；(b) 数据搬运与计算串行——HBM→SMEM的加载不能与当前迭代的GEMM重叠；(c) 所有warps统一角色，register分配不优化；(d) H100仅35%利用率的根源：同步模型使tensor core大量空闲等待non-matmul操作和memory操作。
  - **硬件架构层**：NVIDIA H100 SXM5 GPU (Hopper)。FP16 Tensor Core: 989 TFLOPs/s。TMA硬件单元支持异步HBM↔SMEM拷贝但未使用。WGMMA指令支持异步tensor core操作但未使用。FP8 Tensor Core（2× throughput）未使用。setmaxnreg动态register分配未使用。FlashAttention-2仅在H100上达到~350 TFLOPs/s（35% utilization），而optimized GEMM达到800+ TFLOPs/s（80-85%）。

  Baseline缺陷：
  - (a) **同步模型导致tensor core利用率低**：FlashAttention-2的内循环是同步的——BMM1完成后需等待softmax（3.9 TFLOPs/s特殊函数）完成才能发射BMM2，tensor core在此期间空闲。FP16 head_dim=128时，matmul FLOPs:exponential FLOPs比=512:1，但exponential throughput比matmul低256×，exponential可占用50% cycle time。
  - (b) **数据搬运与计算无重叠**：HBM→SMEM的K/V加载与tensor core计算串行，TMA硬件单元的异步能力未利用。
  - (c) **Register分配不优化**：所有warps均分register，数据搬运仅需1 thread/warp但持有满额register，tensor core计算需要大量register但受限。
  - (d) **未利用FP8低精度**：Hopper FP8 tensor core提供2× throughput，但FlashAttention-2不支持。直接使用FP8 per-tensor量化导致高数值误差（RMSE 2.4× worse），尤其在有outlier features的LLM中。
  - (e) **Kernel launch overhead**：非persistent kernel，每次launch的prologue（Q加载）和epilogue（O写回）期间tensor core空闲。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FlashAttention-3：asynchronous warp-specialized attention kernel exploiting Hopper hardware features**。三个核心技术创新：

  **(1) Producer-Consumer asynchrony via warp-specialization + pingpong scheduling**
  解决缺陷(a)(b)(c)：将CTA内warps划分为producer（仅发射TMA）和consumer（仅发射WGMMA+softmax）。Producer通过TMA异步加载Q/K/V到circular SMEM buffer，不阻塞consumer的GEMM执行。两个consumer warpgroups通过bar.sync实现pingpong——当warpgroup 1执行softmax时，warpgroup 2执行GEMM，tensor core持续被占用。setmaxnreg让consumer获更多register（用于WGMMA），producer释放register（TMA只需1 thread）。
  
  **(2) Intra-warpgroup GEMM-softmax overlapping (2-stage pipeline)**
  解决缺陷(a)：通过寄存器缓冲$\mathbf{S}_{\text{next}}$打破迭代间依赖。迭代j：发射WGMMA(QK^T) of iter j+1（异步，不等待）→ 发射WGMMA(PV) of iter j（异步，不等待）→ 等待WGMMA(QK^T)完成 → softmax on iter j+1（与WGMMA(PV)重叠）→ 等待WGMMA(PV)完成 → rescale $\mathbf{O}_i$。SASS分析验证：softmax指令被compiler重排到第一个WGMMA之前，第一个WGMMA与softmax的FP32→FP16转换交错执行。
  
  **(3) Hardware-accelerated FP8 with block quantization + incoherent processing**
  解决缺陷(d)：Q/K量化为FP8 e4m3 with per-block scaling。Incoherent processing：Q和K先乘随机正交矩阵M（Hadamard × random sign diagonal），$\mathbf{M}\mathbf{M}^\top = I$不改变attention结果，但将outlier"摊平"到所有维度——每个$(\mathbf{QM})$和$(\mathbf{KM})$的元素是Q/K元素的随机线性组合，消除孤立大值对量化的影响。Block quantization fused with rotary embedding（memory-bound操作，零开销）。解决FP8 WGMMA的layout constraints：k-major Q/K（TMA load不变），in-kernel V transpose via LDSM/STSM + byte_perm，FP32→FP8 register exchange via byte_perm + shfl_sync。

  全栈执行例子（H100 GPU, BF16/FP8 forward, N=8192, d=128, 16 heads）：
  - **模型推理算法层**：Attention数学定义不变($\mathbf{O}=\text{softmax}(\mathbf{QK}^T/\sqrt{d})\mathbf{V}$)，计算重组为warp-specialized异步流水线。FP8 variant：$\mathbf{O} = \text{softmax}((\mathbf{QM})(\mathbf{KM})^\top/\sqrt{d})\mathbf{V}$ with per-block quantization of QM, KM, V。Algorithm 1+2提供完整的CTA-view和consumer warpgroup-view伪代码。
  - **系统框架层**：与FlashAttention-2相同的PyTorch接口（`flash_attn_func(q,k,v)`），drop-in replacement。计划集成到PyTorch core。开源：https://github.com/Dao-AILab/flash-attention。
  - **编译框架层**：论文未明确说明。使用CUTLASS primitives（WGMMA, TMA, setmaxnreg, pipeline barriers）手写CUDA C++ kernel。NVCC compiler重排指令以实现overlap——SASS分析验证compiler正确调度了WGMMA与softmax的交错执行。
  - **kernel调度层（关键创新）**：
    - **Warp-specialization**：CTA = 1 producer warp（仅TMA loads）+ 2 consumer warpgroups（各2 warps, WGMMA+softmax）。Producer使用s-stage circular SMEM buffer pingpong调度——异步加载Q/K/V tiles，commit到pipeline barrier通知consumer。Consumer warpgroups交替执行：warpgroup 1 GEMMs while warpgroup 2 softmax，反之亦然。
    - **2-stage pipelining per consumer warpgroup**：prologue: WGMMA(QK₀, sync) → softmax → mainloop: for j=1..T_c-1: WGMMA(QKⱼ, async) → WGMMA(PVⱼ₋₁, async) → wait QK → softmax on Sⱼ → wait PV → rescale O → copy S_next→S_cur。Tensor core和CUDA core通过异步执行实现了GEMM与softmax的重叠。
    - **FP8 support**：per-block quantize Q,K,V → in-kernel V transpose (LDSM→byte_perm→STSM) → SS-WGMMA FP8 QK^T → softmax → RS-WGMMA FP8 PV with register layout conversion (byte_perm→shfl_sync→byte_perm)。
    - **Persistent kernel**：132 threadblocks（=132 SMs），每个处理多个Q tiles，重叠后一个tile的prologue与当前tile的epilogue。
    - **Inference优化**：split-KV (Flash-Decoding) + GQA packing + PagedAttention with TMA block table。
  - **硬件架构层**：NVIDIA H100 SXM5 (Hopper)。TMA硬件单元：异步HBM↔SMEM拷贝，producer warp独占使用。FP8 Tensor Core：2× BF16 throughput。WGMMA异步指令：warpgroup-level tensor core操作，可异步发射不阻塞CUDA core执行。setmaxnreg：动态register重分配。bar.sync：warpgroup间同步屏障。达到：BF16 forward 840 TFLOPs/s (85% utilization, 2.4× vs FlashAttention-2的35%)，FP8 forward 1.3 PFLOPs/s，FP8 RMSE 2.6× better than per-tensor quantization baseline。

## FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection

- baseline方法是什么？
  **现有 kernel fusion 编译器和库（Chimera, BOLT, Welder, MC-Fuser）**：这些方法仅利用单个 SM 的 register 和 shared memory (SMEM) 来存储 kernel fusion 的中间结果。当融合的 GEMM 链中间 tensor 大小超出 SMEM 容量上限（H100 每 SM 227KB）时——例如 LLM 的 FFN 层中间 activation 通常远超此值——fusion 失败并回退到将中间结果经 global memory round-trip 的 low-efficiency 执行方式。这些方法采用单一的 block-level tiling hierarchy，不考虑 cluster-level 的数据分布和 inter-SM data exchange。

  全栈执行例子（Chimera on H100, GPT-6.7B FFN GEMM chain, M=128, N=16384, K=4096, L=4096）：
  - **模型推理算法层**：Standard FFN 的两个连续 GEMM: C = A×B (128×16384), E = C×D (128×4096)。中间 C ∈ R^{128×16384} ≈ 4.2MB (FP16)，远超单 SM 的 SMEM 上限 227KB。
  - **系统框架层**：PyTorch 调用 cuBLAS GEMM kernel。Chimera 尝试融合但失败于 SMEM capacity limitation。框架回退到 2 次独立 GEMM kernel launch——kernel1 写 C to HBM, kernel2 从 HBM 读 C。
  - **编译框架层**：Chimera 的 analytical model 仅分析 reg 和 SMEM 两级 cache 的数据 reuse，当中间 tensor 超出 SMEM 容量时直接判定 infeasible 并跳过。BOLT 使用 CUTLASS 模板但受限于固定 block execution order，也未考虑 DSM。Welder 分析 reg/SMEM data reuse 但同样无 DSM 支持。
  - **kernel调度层**：两独立 GEMM kernel——GEMM0: 加载 A/B tiles → Tensor core WGMMA → write C to HBM（~4.2MB per batch）；GEMM1: 从 HBM 读取 C + 加载 D tiles → Tensor core WGMMA → write E to HBM。中间 C 的 HBM read/write 产生了 2×4.2MB = 8.4MB 额外 global memory traffic per batch element。全局显存访问量约 2.4× more than FlashFuser。
  - **硬件架构层**：NVIDIA H100 GPU。SMEM 227KB/SM 是 fusion 的硬限制。DSM 硬件特性存在（inter-core connection via Crossbar），但被现有 software 完全忽略。Global memory bandwidth 3.35 TB/s，DSM bandwidth 在 cluster size 2 时约 8 TB/s（2.4× higher）。

  Baseline 缺陷：
  - (a) **SMEM capacity bottleneck 限制 fusion scope**：227KB/SM 上限导致中间 tensor 超过此大小的 GEMM chain 无法融合，只能 resort to costly HBM round-trip。大量 compute-intensive operators（FFN 占模型总执行时间 40-60%）受此制约。
  - (b) **完全忽略 DSM 硬件能力**：H100 的 inter-core connected architecture 提供了 cluster 内 SM 间高带宽低延迟的 inter-SMEM data path（DSM），但现有 compiler 和 library 无一利用。DSM bandwidth (up to ~8TB/s) 远高于 global memory (3.35TB/s)，DSM latency (~20ns) 远低于 global memory (~280ns)。
  - (c) **单一 tiling hierarchy 限制**：现有方法只考虑 block-level tiling（一个 SM 内的 tile 划分），缺少 cluster-level tiling 概念。引入 DSM 后需同时处理 spatial partition across SMs 和 temporal scheduling within SMs 的两级 hierarchy。
  - (d) **搜索空间爆炸但无处理方案**：引入 DSM 使 infeasible fusion 变 feasible，搜索空间从 ~10^4 膨胀至 ~10^6（GPT-6.7B），现有 pruning 策略不足以应对。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FlashFuser：首个利用 DSM 进行 kernel fusion 的 DL compiler**。核心设计：(1) dsm_comm primitive——抽象 cluster 内 inter-SM data exchange 模式为形式化原语；(2) Dataflow Analyzer——在 reg→SMEM→DSM 三级 hierarchy 上量化数据搬移并生成 spill plan；(3) Fusion Search Engine——用解析 cost model + DSM-aware pruning 从 ~10^13 的搜索空间中找出最优 plan。

  全栈执行例子（FlashFuser on same GPT-6.7B FFN, H100 SXM, cluster size=(2,4,2,4)）：
  - **模型推理算法层**：同 Standard FFN 计算逻辑不变。FlashFuser 将两个 GEMM 融合为单个 fused CUDA kernel，中间 C 永远不离开 on-chip memory hierarchy (reg→SMEM→DSM)。
  - **系统框架层**：FlashFuser 作为 compiler 生成 CUTLASS-based fused CUDA kernel。离线搜索结果预编译，运行时通过 M-dimension binning + table lookup 选择最优 kernel。可直接嵌入 PyTorch/SGLang 替换 FFN layer 计算。
  - **编译框架层（核心创新——FlashFuser compiler）**：
    - **dsm_comm primitive**：定义三种通信原语统一描述 inter-SM dataflow——dsm_all_exchange（cluster 内 AllReduce/Mul for accumulation）, dsm_shuffle（ring communication within Shuffle Group for data redistribution）, dsm_reduce_scatter（hierarchical reduction for final output aggregation）。通过 cls_m/cls_n/cls_k/cls_l 参数导出 cls_shuffle 和 cls_reduce，精确控制 shuffle group 大小和 reduce group 数量。
    - **Dataflow Analyzer (Algorithm 1)**：遍历 graph 中所有 tensor——IO tensor 按 loop schedule 反序遍历计算 global memory data movement volume；reused tensor 用贪心 heuristic 从 reg→SMEM→DSM 逐级 spill，每级计算 data movement volume（重点分析 DSM traffic，因 DSM bandwidth 低于 SMEM）。最终输出总 D_V 和 placement plan。
    - **Fusion Search Engine (Algorithm 2)**：枚举 LoopSchedule (41 种 S/T 组合) × TilingSize (cluster-level 5^4 + block-level) × ResourceMapping → 5 条 pruning 规则（Divisible Tile, Cluster Size≤16, Activation=innermost loop, Dependency≠spatial L, Memory Capacity）→ Cost model C = max_l(V_l/B_l) → Top-11 硬件 profiling → 最优 plan。
  - **kernel调度层**：单 fused CUDA kernel，内部分为三阶段：
    - **GEMM0 Phase**: Block 内 Tensor core WGMMA 计算 partial C → cls_k=2 表示 K-dim spatial partition, dsm_all_exchange 执行 intra-cluster AllReduce 获得完整 C tile
    - **GEMM1 Phase**: dsm_shuffle ring communication in Shuffle Group (cls_shuffle=2 Blocks) 交换 C tile slices → 各 Block 获得所需 C slice → Tensor core WGMMA 计算 partial E
    - **Store Phase**: dsm_reduce_scatter 两次级归约 (intra-cluster + inter-cluster via TMA cp.reduce.async.bulk) → write final E to HBM
    - 中间 C tile 驻留 DSM（>227KB, 超出 SMEM 但 fit in DSM of multiple SMs），永不写入 HBM
  - **硬件架构层**：同一 H100 GPU。FlashFuser 将 intermediate data path 从 "SMEM → HBM → HBM → SMEM"（traditional round-trip）改为 "SMEM → DSM → SMEM"（direct on-chip path）。DSM bandwidth 约为 global memory 的 1.2-2.4×，latency 约 1/14。全局显存访问减少 58%（Nsight Compute 实测）。dsm_comm primitives 基于 TMA（data movement）+ mbarrier（many-to-many sync）实现，无自定义硬件修改。

  关键设计选择与 baseline 缺陷的对应：
  - **defect (a): SMEM capacity bottleneck** → 方案：DSM 作为 expanded on-chip memory pool——通过 inter-SM communication 将多个 SM 的 SMEM 聚合为虚拟大容量 memory。中间 tensor 超出单 SMEM 容量时 spill to DSM（而非 HBM），在 cluster 内通过 dsm_comm primitives 完成数据复用的通信。以 GPT-6.7B FFN 为例，中间 C tile 在 SMEM 中仅放 128×128×2B≈32KB per Block，但完整 C 需要 128×128×cls_n×cls_k×2B（cluster context），经 DSM exchange 后各 Block 持有完整 row。
  - **defect (b): 完全忽略 DSM 硬件能力** → 方案：dsm_comm primitive 是首个形式化的 DSM-based communication abstraction for kernel fusion。通过 cls_m/cls_n/cls_k/cls_l 参数化 cluster size，导出 cls_shuffle 和 cls_reduce 以精确控制数据交换模式。两种 Gated FFN mapping 策略——spatial partitioning (maximize parallelism) vs sequential execution (minimize DSM overhead)——展示了 DSM-based dataflow 的配置灵活性。
  - **defect (c): 单一 tiling hierarchy** → 方案：两级 hierarchical tiling——cluster-level tile 决定 work distribution across clusters 和 inter-block data exchange patterns，block-level tile 决定单个 Block 内 reg vs SMEM 分配。Loop Scheduling 中 Spatial dimensions 由多个 SM 并行处理（利用 DSM 同步），Temporal dimensions 由单 SM 串行处理。
  - **defect (d): 搜索空间爆炸** → 方案：4 条新增 DSM-aware pruning rules：Cluster Size Constraint (product ≤ 16 hardware limit, consecutive GEMMs' cluster dims must match), Activation Constraint (innermost accumulation dim), Dependency Constraint (L dim can't be spatial), Memory Capacity Limit (tensor ≤ lowest cache capacity)。Pruning 实现 >99.99% 缩减（2.75×10^13 → 1.15×10^6）。Cost model 的 minmax formulation (min max C_l) 精确识别 bottleneck stage 并选 Top-K=11 硬件 profiling，搜索比 brute-force 快 12-68×。
  - **额外贡献：Topology-agnostic design**：dsm_comm 在设计层面是 topology-agnostic 的 collective communication 概念。对 crossbar interconnects (H100, Graphcore IPU) 直接适用；对 mesh architectures (Cerebras WSE) 可通过将 shuffle groups 映射到邻近 core 实现。实现层面基于 CUTLASS + TMA + mbarrier，可移植到其他提供 inter-core connection 的硬件平台。

## FlashInfer Efficient and Customizable Attention Engine for LLM Inference Serving

- baseline方法是什么？
  **各LLM serving框架各自实现专用的attention kernel + 标准FlashAttention-2/3 dense kernel**。当前LLM serving生态中，attention计算面临三个割裂的痛点：(i) **KV-cache存储异构**——vLLM用PageAttention (non-contiguous pages)、SGLang用RadixAttention (tree-structured)、prefix-caching用多级共享前缀，每种存储格式需要专用的attention kernel，不同框架各自实现缺乏统一抽象；(ii) **attention变体手工优化**——现代LLM大量使用标准attention的变体（GQA/MQA、RoPE、sliding window、logits soft-cap、FlashSigmoid等），每种变体需要手工编写专用的CUDA kernel以获得最佳性能，维护成本随变体数量快速增长；(iii) **workload动态性处理不足**——LLM serving中query/KV长度随请求不同而变化，标准FlashAttention使用固定tile size无法适应变化（decode场景中短query使用大tile浪费SM occupancy，prefill中长query需不同配置），且variable-length batch内各请求的计算量不均导致SM idle (wave quantization)。

  全栈执行例子（SGLang + FlashAttention-2 baseline, Llama 8B decode, batch 16, varying seqlen 512-2048, A100）：
  - **模型推理算法层**：标准scaled dot-product attention with GQA (group=4)。各请求attention计算独立：$O_i = \text{softmax}(Q_i K_i^T/\sqrt{d}) V_i$，共享KV-cache时需逐query重复加载同一KV page。
  - **系统框架层**：SGLang维护radix tree管理prefix-caching的KV-cache pages（非连续存储），但attention backend (Triton或FlashInfer-0.1) 暴露的是dense/连续tensor接口。框架需要将radix tree结构"展平"为连续tensor才能调用attention kernel（存在memory mapping gap），或使用per-query GEMV逐个计算（无法利用GQA批量tensor core优势）。Prefix-caching在内存层面减少重复存储，但计算层面仍重复计算shared prefix的attention。
  - **编译框架层**：无JIT编译机制。每种新attention变体（如Streaming-LLM的RoPE+attention融合）需要手工编写CUDA kernel——改写FlashAttention-2/3源码，nvcc编译，验证正确性。RoPE和attention作为两个独立kernel执行时，中间结果需经HBM round-trip（Q after RoPE → write HBM → read HBM for attention）。
  - **kernel调度层**：FlashAttention-2使用固定tile sizes——默认(128,64)针对prefill优化，在decode工作时$l_{qo}=1$，但tile size 128远大于实际query长度：128 rows of Q tile中仅1 row有实际数据其余为padding，严重浪费tensor core throughput和SM occupancy。Variable-length batch（seqlen 512-2048）下，CTA间按固定tile size分配workload——处理长KV的CTA耗时远大于短KV的CTA（load imbalance），导致fast CTA等待slow CTA (wave quantization)。
  - **硬件架构层**：NVIDIA A100/H100 GPU。Shared memory复用受限——同block内queries可复用shared memory中的KV tile，但不同block的queries无法访问彼此的shared memory，即使它们共享相同KV-cache prefix。Global memory / L2 cache是唯一跨block通信手段，带宽远低于SMEM。固定$B_r$限制：大$B_r$使block内更多queries共享KV-cache于SMEM（高带宽复用）但增加fragmentation（请求不在同一block时无法共享），小$B_r$减少fragmentation但失去SMEM复用。

  Baseline缺陷：
  - (a) **KV-cache存储格式与attention kernel耦合**：每种page table/radix tree格式需要专用kernel（或通过unnecessary flattening overhead调用通用kernel），缺乏统一的storage abstraction→compute kernel映射
  - (b) **Attention变体手工优化不可持续**：变体数量快速增长，手工为每种变体编写专用CUDA kernel不可扩展。RoPE+attention分离→extra HBM round-trip→bandwidth waste
  - (c) **固定tile size不适应动态workload**：decode (short Q) vs prefill (long Q) 需要不同tile配置，但现有kernel编译时固定tile size
  - (d) **Variable-length batch load imbalance**：固定tile分配下CTA workload不均导致SM idle (wave quantization)

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FlashInfer：以BSR统一存储抽象 + JIT编译变体生成 + load-balanced动态调度 + composable formats内存优化四层设计实现可定制高效attention engine**。

  核心设计一（解决缺陷a）：**BSR block-sparse format作为KV-cache统一抽象**。将page table (PageAttention) 和radix tree (RadixAttention) 统一表示为BSR矩阵：KV-cache blocks = non-zero BSR blocks，page/radix table = BSR indices arrays。FlashInfer kernel接收BSR metadata (`kv_indptr`, `kv_indices`) 直接索引sparse KV-cache，无需upper framework做展平转换。支持任意$(B_r, B_c)$ block sizes，$B_r$匹配query tile size (SMEM复用粒度)，$B_c$由KV-cache管理算法指定（page/block size）。从BSR格式加载sparse KV tile：compute block address from indices→`cp.async` LDGSTS从分散global memory gather到contiguous SMEM→dense tensor core MMA处理。vector-sparsity ($B_c$=1) 支持fine-grained page-level sparsity (Quest等token重要性稀疏)。

  核心设计二（解决缺陷b）：**JIT compiler for attention variants**。定义$f_{epilogue}(scan(f_{logits}(f_q(Q) \cdot f_k(K))) \cdot f_v(V))$形式的general attention template，通过6个可替换functor slot (QueryTransform/KeyTransform/ValueTransform/OutputTransform/LogitsTransform/LogitsMask) 覆盖绝大多数attention变体。用户以CUDA code定义variant class→template population→PyTorch JIT compiler编译→register custom operator。支持advanced PTX instructions甚至user libraries。相比FlexAttention：生成CUDA code（非Triton，性能更优），支持Q/K变换（FlexAttention不支持），支持vector-sparsity和load-balancing（LLM serving特定优化）。

  核心设计三（解决缺陷c,d）：**Dynamic load-balanced scheduling**。Compile-time选择tile sizes（根据硬件资源和预期workload特征的heuristic），runtime调度：CPU端每generation step执行Algorithm 1——计算max KV chunk size $L_{kv}$→split query tiles into KV chunks→sort by length descending→greedy min-cost assignment to CTAs→output plan info（work queue + partial/final output index mapping）→async copy to GPU workspace buffer。GPU端persistent kernel按plan执行：attention kernel处理allocated KV chunks→partial attention states $(O_{partial}, LSE_{partial})$→contraction kernel用$\oplus$ compose操作合并为final O。Multiple tile sizes $(1,16,32,64,128) \times (32,64,128)$：FA2, $(64,128,...) \times (32,64,128)$: FA3，$T_q=1$用CUDA core（decode短query），$T_q \geq 16$用Tensor Core。CUDAGraph兼容：persistent kernel + fixed grid size + workspace buffer fixed offset。

  核心设计四（额外贡献，解决缺陷d中prefix-caching计算浪费）：**Composable formats for prefix-caching**。利用prior knowledge将KV-cache BSR matrix分解为多个不同$B_r$的sub-matrices：shared prefix → 大$B_r$ block（多个queries share loaded KV-cache tile in SMEM, high bandwidth），unique suffix → 小$B_r$ block（per-query处理，tolerate L2/global latency）。无需data movement:仅compute different `kv_indptr`/`kv_indices` arrays。

  全栈执行例子（FlashInfer + SGLang, Llama 8B decode, batch 16, varying seqlen, H100）：
  - **模型推理算法层**：Attention数学定义不变。BSR format将attention计算重组为block-sparse: $O_i = \oplus_{j \in \text{non-zero}(row_i)} \text{AttentionState}(Q_i, K_j, V_j)$，其中$\oplus$是associative/commutative的attention compose operator。
  - **系统框架层**：SGLang radix tree still manages KV-cache pages。FlashInfer作为attention backend，直接接收SGLang的page table metadata作为BSR indices输入，无需展平conversion。Composable formats: framework告知FlashInfer shared prefix的query ranges→FlashInfer创建两个AttentionWrapper（大$B_r$ for prefix, 小$B_r$ for suffix）→各自捕获不同CUDAGraph→runtime根据KV-cache配置select optimal graph。无需修改SGLang的scheduler或memory manager。
  - **编译框架层**：JIT compiler pipeline。以Streaming-LLM RoPE fusion为例：~20行CUDA code定义`RoPEFusionVariant` (QueryTransform+KeyTransform) → template populated with RoPE functors injected before QK^T GEMM → `torch.utils.cpp_extension.load_inline` → shared library → PyTorch custom operator → CUDAGraph captured。生成单一fused kernel: Q/K in SMEM → RoPE rotation (CUDA core) → QK^T GEMM (tensor core) → online softmax → PV GEMM → O，消除RoPE kernel + attention kernel间的HBM round-trip。
  - **kernel调度层**：CPU scheduler读$\{l_{qo}=1\text{ (decode)}, l_{kv}\text{ varying 512-2048}\}_{i=1}^{16}$→cost model→$L_{kv}$ computed→chunk assignments计划→GPU persistent kernel per-plan execution。$T_q=1$ (decode specific) CUDA core path避免tensor core min row=16的限制。Sorted descending assignment ensures 长KV的CTA工作量均衡（大chunks先分配，小chunks填空隙）。CUDAGraph capture: persistent kernel grid size compile-time constant→workspace buffer中partial O区域和plan info区域at fixed offset→CUDAGraph valid for all generation steps（仅plan info content changes, pointers fixed）。
  - **硬件架构层**：H100 GPU。BSR sparse loading: `cp.async` LDGSTS (128B width) → SMEM → dense WGMMA。TMA仅用于dense contiguous KV-cache (Hopper)；sparse pattern回退Ampere-style LDGSTS。Composable formats: shared prefix submatrix $B_r=3$→3 queries' attention share same K/V tile in SMEM→3× less global memory traffic for shared prefix。Unique suffix $B_r=1$→per-query access via L2。SMEM→register和L2→register hierarchy区分利用。

  关键设计选择与Baseline缺陷的对应：
  - **defect (a): KV-cache存储格式与kernel耦合** → 方案：BSR作为统一sparse abstraction。Page table/radix tree→BSR indices arrays。任意$(B_r,B_c)$使kernel适应多种page/block sizes。上层framework仅需提供BSR metadata，无需修改kernel。
  - **defect (b): 变体手工优化不可持续** → 方案：JIT compiler with 6 functor slots。~20行CUDA code per variant（vs 全手工CUDA kernel 数百行）。Cover $f_{epilogue}(scan(f_{logits}(f_q(Q) \cdot f_k(K))) \cdot f_v(V))$空间含MLA、FlashSigmoid、sliding window、soft-cap等。Compiled once, cached on disk for reuse。
  - **defect (c): 固定tile size** → 方案：Multi-tile-size microkernel heuristics。Compile-time从$(1,16,32,64,128) \times (32,64,128)$中选择optimizing SM occupancy → 不同workload (decode/prefill, short/long query) 适用不同tile → compile到不同CUDAGraph → runtime select best。
  - **defect (d): Variable-length load imbalance + prefix-caching计算浪费** → 方案：Algorithm 1 greedy scheduling (Stream-K inspired, deterministic) for load balance。Composable formats: 多BSR分解capitalize prefix structure，大$B_r$共享SMEM节约bandwidth。两者combined: balanced scheduling确保prefix和suffix处理均匀分布在CTAs间。

## FlashMoE: Fast Distributed MoE in a Single Kernel

- baseline方法是什么？
  **标准分布式MoE执行（以Megatron-LM + DeepEP / DeepSpeedMoE / FasterMoE / COMET为代表）**：MoE layer分为多个独立kernel——Gate kernel（计算routing）+ Dispatch AlltoAll collective（token重排）+ Expert FFN kernels（逐expert GEMM，2层）+ Combine AlltoAll collective（token恢复原始顺序）。所有kernel由CPU逐个launch，AlltoAll为NCCL同步collective。关键性能瓶颈：
  - (a) **同步AlltoAll collective**：所有GPU必须同时参与AlltoAll，straggler GPU卡住全部GPU的进展（P95 delay 1.32× on supercomputer, 11.4× on VM），通信占MoE layer总运行时间最高68%
  - (b) **大量kernel launch overhead**：单层MoE forward pass需33-550个GPU kernel（Table 1），每次launch产生CPU-GPU同步、CUDA API overhead、kernel start time non-determinism，造成GPU idle gap
  - (c) **CPU-managed调度缺乏work-conserving**：CPU串行launch kernel，无法根据GPU内部readiness动态分配计算任务
  - (d) **Token padding浪费通信带宽**：不对称routing导致GPU收到的token数不足expert capacity时，DeepSpeed等框架在通信buffer中补零传输，浪费带宽和算力

  全栈执行例子（Megatron-LM + DeepEP, 8×H100, S=16K tokens, E=128 experts, top-2 routing, FP16）：
  - **模型推理算法层**：Standard top-2 gating。Gate function G(x) = softmax(x·W_g) → select top-2 experts per token → 每个token需routing到2个expert。routing table T_φ[e][c] = (token_idx, combine_weight)。Expert capacity C = 256（假定uniform distribution），超过capacity的token丢弃（capacity factor=1.0）。
  - **系统框架层**：Megatron-LM distributed MoE layer实现。使用PyTorch distributed + NCCL backend。MoE layer forward: Gate → token permutation (AlltoAll dispatch) → expert FFN (各GPU上的local experts的serial或batched GEMM via cuBLAS/Transformer Engine) → token unpermutation (AlltoAll combine)。DeepEP提供NCCL + NVSHMEM混合的优化AlltoAll实现。
  - **编译框架层**：论文未明确说明。Megatron-LM使用nvcc编译手写CUDA kernel，Transformer Engine（TE）使用fp8精度GEMM。
  - **kernel调度层（关键瓶颈）**：
    - Gate kernel (1 launch): CUDA kernel计算routing
    - Dispatch AlltoAll (NCCL collective): 同步barrier，所有GPU等待最慢GPU的token buffer准备好 → GPU在等待期间SM idle。AlltoAll底层为多次P2P send/recv或NVLink copy + NVSwitch routing
    - Expert FFN (expert_cnt × 2 launches): 每个expert的GEMM0和GEMM1各一次cuBLAS kernel launch（或TE fused kernel），若16 local experts=32次launch。每次launch：CPU enqueue → CUDA API overhead → kernel start → global memory load weights → GEMM → store
    - Combine AlltoAll (NCCL collective): 再次同步barrier
    - 总kernel launch数：432（Megatron-LM+DeepEP，Table 1）。CUDA API trace（Figure 4b）显示大量small CUDA API calls，kernel间显著stall gap。SM utilization仅13.55%（DeepEP+Megatron-LM）、9.67%（FasterMoE）。
    - Overlap方案（COMET、FasterMoE）：将部分expert computation与AlltoAll pipeline重叠，但受限于collective的同步barrier——先dispatch完成才能开始全部FFN，先全部FFN完成才能开始combine，overlap有限
  - **硬件架构层**：8×H100 GPU NVSwitch全互联。NVLink 900 GB/s unidirectional per GPU。NCCL AlltoAll通过NVSwitch可实现full bisection bandwidth，但同步barrier导致collective latency由最慢参与者决定。Tensor core算力充足但大量时间idle等data。

  Baseline缺陷：
  - (a) **同步通信straggler问题**：AlltoAll collective barrier要求全部GPU参与，最慢GPU决定整体延迟。straggler cause包括kernel scheduling jitter、OS interference、NVLink congestion——尤其expert分布不均衡时更严重
  - (b) **Kernel launch overhead**：432个kernel launch per MoE layer（vs FlashMoE 1个），产生~90% GPU idle time（Figure 4a）。每次launch对应CUDA API call (~10μs)、CPU-GPU synchronization、kernel cold-start L1/L2 cache miss、global memory round-trip for intermediate data
  - (c) **Non-work-conserving execution**：CPU按静态顺序launch kernel（Gate→AlltoAll→expert0_GEMM0→...→AlltoAll），无法根据dynamic availability（某GPU已完成expert计算但另一GPU的FFN仍在执行）进行动态调度。GPU SM在等待远端token期间完全空闲
  - (d) **通信payload冗余**：expert capacity padding——若某些expert收到少于C个token，剩余slot补零，这些零值通过NVLink传输再被GEMM计算（结果为零），浪费通信带宽和计算资源

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FlashMoE：单个持久化GPU kernel融合全部分布式MoE计算与通信**。核心创新是将MoE operator从CPU协调的多kernel架构彻底重构为GPU-resident的单kernel actor-based并发系统。

  核心设计一（解决缺陷b,c）：**Actor-based单kernel架构 + Warp Specialization**。将N个thread block特化为N-1个Processor + 1个OS block。OS block内：1个Scheduler warp（work-conserving多线程调度器，Algorithm 3）+ 3个Subscriber warps（解码远端tile packet，Algorithm 4）。Processor持续loop等待Scheduler signal分配task（Algorithm 2）。整个MoE layer生命周期内仅1次kernel launch，消除所有kernel launch overhead（从432→1）。Scheduler根据task readiness动态分配——只要有Processor idle且task queue非空→立即schedule——实现work-conserving。

  核心设计二（解决缺陷a）：**Device-initiated one-sided (R)DMA替代同步AlltoAll collective**。使用NVSHMEM PGAS编程模型——GPU kernel内直接通过nvshmem_putmem发起跨GPU内存写——无需远端GPU CPU参与，无需collective barrier。关键insight：token dispatch从"pull model"（AlltoAll，所有GPU同步exchange）变为"push model"（Processor主动push tile到远端GPU memory，Subscriber被动接收）。远端GPU的Subscriber通过检测NVSHMEM signal flag异步消费数据。配合Symmetric Tensor Layout L（Theorem 3.1 write-write conflict-free），所有one-sided write无需锁或同步——每个(p_s, r, b=1, e, c) index唯一确定目标位置。

  核心设计三（解决缺陷d）：**In-place padding + Payload-efficient communication**。Token在dispatch前在本地symmetric tensor buffer L内padding到expert capacity对齐（divisible by tile height bM=128），网络仅传输实际token tile——无null payload在网络传输。相比DeepSpeed/DeepEP在通信buffer中补零传输的做法，节省通信带宽（极端不对称routing case下可节省数倍payload）。

  核心设计四（实现fine-grained overlap）：**Tile-level parallelism with (128, 64) tile**。将MoE计算和通信分解为tile粒度——每个tile映射为task descriptor t = (M, ⋆, φ)。GEMM0、GEMM1、combine、通信均为独立task，可并发执行：Processor A执行expert_i的GEMM0时，Processor B可同时dispatch expert_j token到远端，Subscriber可解码已到达tile的GEMM1 task，Scheduler并行调度。三个actor通过shared memory（Subscriber↔Scheduler）和global memory（Scheduler↔Processor，inter-GPU signal flag）异步通信，形成reactive、非阻塞的pipeline（Figure 6）。

  全栈执行例子（FlashMoE, 8×H100, same config S=16K, E=128, FP32）：
  - **模型推理算法层**：Same top-2 gating, standard FFN, expert-combine（Equation 2,3）。算法本身不变——创新在执行方式。FusedGate in-kernel（Algorithm 1 line 2）——Gate计算与routing table构建在一个函数内完成，无需写回HBM。
  - **系统框架层**：FlashMoE Python/C++ library（6820行代码）。`flashmoe::forward(A, X, O)` 一次调用完成全部MoE layer。无PyTorch NCCL backend依赖——通信全部通过NVSHMEM kernel内完成。框架不参与dispatch/combine调度——CPU仅做一次CUDA kernel launch。
  - **编译框架层**：CMake + nvcc编译。CUTLASS作为submodule提供device-side BLAS。NVSHMEM提供device-side通信API。无JIT compilation——tile size (128,64) compile-time固定。Binary size 29 MB, compilation time 53s。
  - **kernel调度层（关键创新，全覆盖）**：
    - FusedGate（Algorithm 1 line 2）: 所有block并行计算Gate → T_φ, G_φ写入shared memory + global memory
    - 角色分化（Algorithm 1 lines 6-11）: N-1 blocks → Processor (processor::start()); 1 OS block → warp 0 Scheduler + warps 1-3 Subscriber
    - Dispatch: Processor按T_φ将(128,64) tile通过NVSHMEM put写入远端GPU的L → write flag通知远端Subscriber。In-place padding确保无null传输
    - Subscriber poll flags → atomic retrieve → memory fence → 解码tile为GEMM0 task → write tQ → doorbell Scheduler
    - Scheduler sweep doorbells → WarpInclusiveSum → 从ready queue取idle Processor → signal。Work-conserving: 任一Processor空闲即分配，任一tile就绪即调度
    - Processor: awaitTaskFromScheduler → warp broadcast task → switch(type): GEMM0(fused GEMM+epilogue via CUTLASS device-side) → notify completion → GEMM1(same, result可能NVSHMEM put到远端combine buffer) → Subscriber解combine signal → Scheduler调度combine → Processor: Hadamard product + accumulate to O
    - Kill: Scheduler counted taskBound → interrupt subscribers → interrupt processors → kernel return
    - 结果: 93.17% SM utilization (9× higher than FasterMoE), 6× latency speedup, 4× overlap efficiency vs baselines (at 8 GPUs, 128 experts, 16K tokens)
    - 注意: FlashMoE在FP32下取得这些结果，baseline在FP16——通信量（4B vs 2B per element）和计算量（FP32 vs FP16 GEMM）均为double
  - **硬件架构层**：H100 GPU + NVLink。NVSHMEM nvshmem_putmem通过NVLink RDMA直接写远端HBM——利用H100的NVLink interconnect和NVSwitch实现cross-GPU data path。Symmetric tensor layout L overprovision 4× memory（2 rounds × 2 staging buffers = 4× token buffer），H100 80GB完全可容纳（≤2% overhead for popular models）。CUTLASS device-side GEMM utilize Tensor Cores through MMA instructions called from within persistent kernel。

  关键设计选择与Baseline缺陷的对应：
  - **defect (a): 同步AlltoAll straggler** → 方案：Device-initiated push-model one-sided (R)DMA替代pull-model synchronous collective。每个GPU独立push token到目标GPU，无需等待该GPU也完成同一round的push——消除barrier和straggler effect。Theorem 3.1证明write-write conflict-free保证无需同步。Temporal buffering (2 staging buffers) 隔离dispatch和combine的并发访问。
  - **defect (b): 432次kernel launch** → 方案：单persistent kernel融合全部操作。Actor-based concurrency模型允许同一kernel内执行gate、FFN GEMM0、FFN GEMM1、combine、dispatch通信、combine通信、调度——所有逻辑在while(!interrupt) loop的CUDA thread内完成。Kernel launch overhead从~3.6ms（432次，估算）降至~1μs（1次launch）。
  - **defect (c): Non-work-conserving CPU调度** → 方案：In-kernel work-conserving Scheduler（Algorithm 3）。多线程（1 warp=32 threads）并行sweep doorbells→aggregate→schedule，保证只要Processor空闲且task就绪→立即分配。Scheduler持续atomic poll taskBound确保不漏调度。Processor-Subscriber-Scheduler三者通过shared memory/global memory异步通信，形成reactive event-driven pipeline。
  - **defect (d): Token padding浪费带宽** → 方案：In-place padding + payload-efficient通信。Padding在本地L buffer内完成，网络传输仅包含实际token tile——NVSHMEM put的size = actual_tokens × H × sizeof(float)，而非 padded_capacity × H × sizeof(float)。尤其expert分布高度skewed时收益最大。

## FlowMM Cross-Modal Information Flow Guided KV Cache Merging for Efficient Multimodal Context Inference

- baseline方法是什么？
  **现有KV cache压缩方法（eviction-based和merging-based）直接应用于多模态场景**：主要包括两类——(1) Eviction方法（StreamingLLM、H2O、D2O），基于attention scores评估token重要性后丢弃低重要性token，但不可逆的信息丢失导致context fragmentation和hallucination（Jiang et al., 2025）；(2) Merging方法（KVMerge）将所有eviction候选token合并到保留token中以保留更丰富的上下文信息，以及multimodal-specific方法（LOOK-M）针对多模态设计的KV cache合并。但这些方法均为text-based或仅简单适配multimodal，未能考虑MLLM中跨模态信息流的层间差异性——浅层以intra-modal交互为主（提取低层特征），深层以inter-modal交互为主（跨模态融合和高层语义抽象）。对所有层使用统一合并策略导致：浅层跨模态合并造成模态信息混淆（modality information confusion），深层仅做模态内合并导致跨模态语义融合不充分（insufficient cross-modal fusion）。

  全栈执行例子（以Qwen2.5-VL-7B处理ALFRED任务的一个多模态样本，单A100 GPU，使用KVMerge baseline）：
  - **模型推理算法层**：MLLM将visual tokens（来自ViT编码器的patch embeddings）和text tokens拼接为输入序列X = {X_1^T, X_1^I, ..., X_N^T, X_M^I} ∈ R^{L_p×d}。在prompt encoding阶段计算K_0 = XW^K, V_0 = XW^V（公式2）。生成阶段逐token更新KV cache: K_t = [K_{t-1}, k_t], V_t = [V_{t-1}, v_t]（公式3）。Attention输出: o_t = Softmax(q_t K_t^T / √d) V_t（公式4）。KVMerge baseline在所有L层使用统一策略——基于token相似度将non-pivot tokens合并到pivot tokens，不论该层是以intra-modal还是inter-modal交互为主。
  - **系统框架层**：论文未明确说明使用特定serving框架。KV cache合并算法作为HuggingFace Transformers推理pipeline的插件式KV cache后处理模块，在每层attention计算后对KV cache进行压缩。
  - **编译框架层**：论文未明确说明。使用标准PyTorch推理路径，KV cache合并操作为纯PyTorch tensor操作（cosine similarity + weighted averaging）。
  - **kernel调度层**：论文未明确说明。KV cache合并操作（token similarity计算、top-k selection、weighted averaging）在PyTorch层通过GPU kernel执行，无自定义CUDA kernel。
  - **硬件架构层**：单张NVIDIA A100 80GB GPU。Baseline KVMerge在20% cache budget下GPU memory从2.06 GiB降至0.44 GiB，但ALFRED accuracy从36.92%降至27.94%（Qwen2.5-VL-7B, Table 1），说明统一合并策略在multimodal场景下造成显著信息损失。

  Baseline缺陷：
  - (a) **忽略跨模态信息流的层间差异性**：MLLM的浅层（layers 1-N/2）以intra-modal attention为主（visual→visual, text→text），深层（layers N/2+1到L）以inter-modal attention为主（visual↔text）。统一合并策略在浅层做跨模态合并导致modal information confusion（Figure 3b: misaligned merging仅达full cache的~50% accuracy），在深层做模态内合并导致cross-modal fusion不充分。
  - (b) **无token敏感度保护**：所有token在合并决策中被平等对待（仅基于相似度），导致高敏感度task-critical token被合并后信息被稀释（dilution），尤其在TextNeedle等需要精确保留特定token信息的任务中表现明显。
  - (c) **合并策略与任务无关**：统一合并策略不考虑当前推理任务的语义需求，仅基于底层token表示相似度决策。
  - (d) **多模态分布偏移**：visual tokens和text tokens存在显著的distributional divergence，indiscriminate merging（不区分模态的合并）可能导致语义扭曲（semantic distortion）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FlowMM：Cross-Modal Information Flow Guided KV Cache Merging + Sensitivity-Adaptive Token Matching**。核心设计一（解决缺陷a,c）：通过分析每层的cross-modal attention interaction ratio ρ^l（公式6-7），动态判断每层应以intra-modal还是inter-modal方式合并——浅层（ρ^l<θ）做intra-modal保留各模态的低层特征完整性，深层（ρ^l≥θ）做cross-modal促进跨模态语义融合。核心设计二（解决缺陷b）：在token匹配时引入sensitivity threshold τ（公式10），仅允许将non-pivot tokens合并到低敏感度的pivot tokens（I_j ≤ τ），保护高敏感度task-critical tokens不被合并稀释。核心设计三（解决缺陷d）：intra-modal merging策略将visual和text tokens分别聚类合并，避免跨模态分布偏移导致的语义混淆。

  全栈执行例子（同样Qwen2.5-VL-7B, ALFRED任务, FlowMM, cache budget=20%, θ=0.2, τ=0.3, 单A100 GPU）：
  - **模型推理算法层**：同一MLLM前向推理流程。FlowMM在每层attention计算后进行两阶段处理：(Phase 1 离线) 在校准样本上计算每层ρ^l确定merge_strategy——浅层4层ρ^l<0.2→intra-modal merging，深层28层ρ^l≥0.2→cross-modal merging；(Phase 2 在线) 每层KV cache merge时：(1) 用proxy tokens（最后~16个prompt tokens）聚合计算每个token的重要性I(i) = Σ_{j∈P} α_{j→i}；(2) 选top-20% tokens作为pivot set K^p；(3) 对每个non-pivot token i，在K^p中找cosine similarity最高且sensitivity I_j ≤ τ的pivot j，将(K_i, V_i)加权平均合并到(K_j, V_j)；(4) 搜索空间受merge_strategy约束——intra-modal层仅在同类模态token内搜索。ALFRED accuracy从KVMerge的27.94%提升至35.43%（Table 1）。
  - **系统框架层**：论文未明确说明使用特定serving框架。FlowMM在HuggingFace Transformers推理pipeline中作为KV cache后处理插件，替换原有KV cache compression逻辑。
  - **编译框架层**：论文未明确说明。FlowMM的KV cache合并操作为纯PyTorch实现，无自定义编译框架修改。
  - **kernel调度层**：论文未明确说明。Cross-modal ratio计算、cosine similarity矩阵计算、weighted averaging合并等操作使用PyTorch GPU kernel（如torch.matmul, torch.cosine_similarity），无自定义CUDA kernel优化。
  - **硬件架构层**：同一A100 GPU上，FlowMM 20% cache budget: GPU memory 0.44 GiB（~80% reduction），decoding latency 17.35 ms/token（vs full cache 29.08 ms/token，~1.7×加速）。在ALFRED上accuracy从KVMerge的27.94%恢复至35.43%（接近full cache的36.92%），在TextNeedle上accuracy从KVMerge的9.69%提升至10.00%（Table 1）。消融实验（Table 4）：移除information flow guidance后ALFRED降至33.58%（-1.85%），移除sensitivity-adaptive matching后降至33.75%（-1.68%），两者都移除降至31.01%（-4.42%）。

  关键设计选择与Baseline缺陷的对应：
  - **defect (a): 忽略跨模态信息流的层间差异性** → 方案：Cross-modal information flow analysis。通过公式ρ^l = (1/H)·Σ_h(A_{v→t}^{l,h} + A_{t→v}^{l,h})/A^{l,h}量化每层的跨模态交互强度。实证发现（Figure 3a）：浅层cross-modal attention比例低（<0.2），深层高（>0.2），且该pattern在ALFRED/MMCoQA/TextNeedle三个不同任务上一致。由此设定threshold θ=0.2（由Table 3验证θ在0.2-0.3区间最优），浅层做intra-modal merging避免模态混淆，深层做cross-modal merging促进跨模态融合。与align vs misalign实验（Figure 3b）相呼应——aligned merging接近full cache性能，misaligned merging大幅退化。
  - **defect (b): 无token敏感度保护** → 方案：Sensitivity-Adaptive Token Matching。定义token sensitivity为其对模型输出保真度的贡献——高敏感度token合并后对模型准确度有显著负面影响。使用attention scores作为sensitivity的零开销近似度量（near-zero-overhead approximation），设置threshold τ过滤高敏感度pivot tokens（公式10: I_j ≤ τ）。消融实验（Table 4）显示：移除sensitivity protection后TextNeedle从10.00%降至6.32%（-3.68%），ALFRED从35.43%降至33.75%（-1.68%），证明sensitivity保护在需要精确保留特定token信息的任务中尤为关键。
  - **defect (c): 合并策略与任务无关** → 方案：Proxy token-based重要性评估。使用prompt末尾少量proxy tokens（capture task-specific contextual information）聚合的attention scores作为token重要性度量（公式8: I^{l,h}(i) = Σ_{j∈P} α_{j→i}^{l,h}），使pivot selection偏向当前任务相关的关键token。相比统一使用累积attention的biased评估，proxy tokens提供更公平（equitable）的token重要性估计。
  - **defect (d): 多模态分布偏移** → 方案：Intra-modal merging in shallow layers。浅层ρ^l<θ时，intra-modal merging将visual tokens和text tokens分别聚类合并——visual tokens仅在visual token内部搜索最近邻合并，text tokens仅在text token内部搜索，避免浅层的cross-modal merging造成visual-text embedding分布偏移导致的语义混淆。深层ρ^l≥θ且cross-modal interactions已充分建立后，才允许跨模态合并。
  - **额外设计：无fine-tuning和plug-and-play** → FlowMM无需fine-tuning，作为plug-and-play KV cache压缩模块直接应用于已有MLLM。所有合并策略由离线校准的ρ^l pattern和运行时动态计算的token importance/sensitivity决定，无需修改模型权重。
  - **性能结果**：80%-95% KV cache memory reduction，1.3×-1.8× decoding latency reduction。在InternVL2.5-8B上20% cache budget下平均accuracy degradation仅0.12%（vs full cache, Table 1）。在低cache budget（<10%）时优势尤其显著（Figure 4），在40% budget时已达full cache相当性能。

## FoundationMotion: Auto-Labeling and Reasoning about Spatial Movement in Videos

- baseline方法是什么？
  现有VLM（Gemini、Qwen-VL、PerceptionLM、NVILA）使用通用视频-文本数据预训练，在motion understanding上表现不足。这些模型的训练数据主要回答"what is this motion"（如识别"pouring water"），但缺乏"how this motion happens"的细粒度运动数据（如"pouring water from a bottle into a glass"的具体轨迹和空间关系）。手工标注motion数据极其昂贵：一个标注员需要几分钟标注3秒视频，10人团队需约100天完成100K视频。此外，现有motion benchmarks（MotionBench、FAVOR-Bench）关注细粒度motion recognition但忽视spatial reasoning（运动交互、相对轨迹、几何约束）。
  
  全栈执行例子（Gemini-2.5 Flash on MotionBench, 给定视频clip）：
  - **模型推理算法层**：VLM接收video frames作为多模态输入，通过visual encoder提取帧级特征，经过cross-attention/projection注入LLM，LLM自回归生成QA答案。模型可识别视频中"a car is moving"但无法正确回答"which direction is the car turning"（Gemini在MotionBench上仅55.6%）。
  - **系统框架层**：VLM inference通过standard transformer serving pipeline执行——视觉编码器处理frames、LLM处理多模态tokens。无特殊motion-centric pre/post-processing。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明。
  
  Baseline缺陷：
  - (a) 训练数据缺乏fine-grained motion annotations（只有"what"没有"how"），导致VLM的motion reasoning能力弱
  - (b) 手工标注motion数据成本过高，无法规模化
  - (c) 现有自动标注pipeline未提供结构化spatial signals（bbox轨迹、tracking信息），LLM仅从raw video生成QA质量差
  - (d) 缺乏覆盖多领域（驾驶、机器人、日常手部运动）的"how" motion benchmarks

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FoundationMotion全自动数据标注pipeline + motion-centric VLM fine-tuning**：
  
  **解决缺陷(a)(b)——automated pipeline替代人工标注**：设计四阶段全自动pipeline：(1) Video Preprocessing（temporal cropping + 相机运动过滤）；(2) Object Detection & Tracking（Qwen2.5-VL+GroundingDINO做开放词汇检测，Cascade Mask R-CNN+ViTPose++Hands23做人体/手部检测，SAM2做两阶段时序tracking）；(3) Caption Generation（将tracking JSON+bbox overlay+frames输入GPT-4o-mini，按7维度生成motion caption）；(4) QA Generation（从caption生成5类QA）。该pipeline在46.7K InternVid视频上自动生成467K QA pairs。
  
  **解决缺陷(c)——structured spatial signals注入LLM**：关键创新是在LLM生成caption/QA时，不仅输入raw video frames，还输入归一化bounding box trajectory JSON（包含每帧每个object的bbox坐标、object_type、interactions关系），以及color-coded bbox visual overlay。Ablation（Table 2）证明video+bbox JSON vs video-only在Fine-grained Action Accuracy上+2.6，Motion Detail+2.6，Temporal Coherence+2.4，Overall QA Quality从6.3提升到8.6（GPT-4评分，0-10）。
  
  **解决缺陷(d)——自建zero-shot "how" motion benchmarks**：手动标注四个跨领域benchmarks——AV-Car（NuScenes car motion, 1968 QAs）、AV-Hand（NuScenes hand motion, 108 QAs）、Daily（100 Days of Hands, 832 QAs）、Robotics（YouTube robot videos, 102 QAs）。
  
  全栈执行例子（FoundationMotion fine-tuned NVILA-Video-15B on AV-Car benchmark）：
  - **模型推理算法层**：fine-tuned VLM接收"the car is turning right"类QA输入。模型在FoundationMotion 467K training pairs上学到了bbox级spatial reasoning（通过tracking JSON中归一化bbox轨迹学习motion方向/距离/速度模式），可在AV-Car上从84.4%提升到91.5%（+7.1%），超越Gemini-2.5 Flash（84.1%）和Qwen2.5-VL-72B（83.3%）。
  - **系统框架层**：使用llamafactory（Qwen）和NVILA official training code做SFT fine-tuning。Training在8x A100 GPUs上进行，lr=1e-5(Qwen)/1.5e-5(NVILA)，cosine annealing，Adam optimizer。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明。

## HyTiS: Hybrid Tile Scheduling for GPU GEMM with Enhanced Wave Utilization and Cache Locality

- baseline方法是什么？
  Baseline 是 cuBLAS（NVIDIA 闭源 GEMM 库，使用 homogeneous tile scheduling——所有 output tiles 采用统一的 tile size，通过 precompiled kernel specializations 执行）。以 H100 GPU 为例，给定 GEMM 问题 M×N×K=1672×1024×4096：

  **全栈执行例子**：
  - **模型推理算法层**：以 GPT/LLM 中 GEMM 为例，输入为 activation tensor A (M×K=1672×4096) 和 weight matrix B (K×N=4096×1024)，需计算 C=A×B。cuBLAS 选择预编译的 fixed-size tile kernel（如 bM×bN×bK=128×128×64）。
  - **系统框架层**：PyTorch 调用 torch.matmul → dispatch 到 cuBLAS cublasGemmEx。cuBLAS 的 tile 调度是 homogeneous 的——所有 output tiles 使用相同 micro-kernel。
  - **编译框架层**：cuBLAS 使用预编译 kernel，无运行时编译。Inductor-Triton baseline 使用 Triton compiler 生成约 20 种候选 tile 配置的 kernel，但 tile layout 固定为 group-M with group_size=8。
  - **kernel调度层**：H100 有 132 个 SM。1672×1024 GEMM 在 bM×bN=128×128 下产生 ceil(1672/128)×ceil(1024/128) = 14×8 = 112 个 output tiles。112 tiles 在 132 SM 上形成 0 个 full wave（112 < 132），全部成为 1 个 partial wave，SM 利用率仅 112/132 ≈ 85%，剩余 20 个 SM 闲置——这是 wave quantization 问题。cuBLAS 的 homogeneous tile 无法应对此情况，性能随 M 变化剧烈波动（在 M=640→704 和 M=1664→1728 处出现 36% 和 21% 的性能陡降，Figure 1）。
  - **硬件架构层**：H100 SM 内 Tensor Core 执行 wgmma 指令进行 MMA 计算，SM 通过 L1/SMEM 做 local data staging，所有 132 SM 共享 50MB L2 cache。Partial wave 中部分 SM 空闲导致 Tensor Core 利用率不足。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  HyTiS 提出两层级联的混合 tile 调度 + 自适应 tile layout 选择，在 Triton 上实现。以同一 1672×1024×4096 GEMM 为例：

  **全栈执行例子（对比 baseline）**：
  - **模型推理算法层**：同一 GEMM 问题，HyTiS 不改变算法——仍是 dense FP16 matmul with FP32 accumulation。关键区别在于 scheduling 层。
  - **系统框架层**：PyTorch 调用 hytis.matmul(a, b) → HyTiScheduler 接收 (M,N,K)=(1672,1024,4096) → 检查 tuning cache → 执行 auto-tuning 在 TO×LO 搜索空间中选择最优 (K1, K2, layout)。
  - **编译框架层**：HyTiS 使用 Triton 作为后端 compiler，不改动 Triton 编译流程。Triton 负责 intra-tile 优化（memory coalescing、thread swizzling、shared memory allocation、TMA instruction emission on H100）。
  - **kernel调度层（核心创新）**：
    1. Offline profiling：目标 GPU 上预先对所有候选 micro-kernel 执行 profiling（H100 约 19 min），构建 S^TO（吞吐量导向，large tiles，在 full waves 中最大化 compute/memory ratio）和 S^LO（延迟导向，fine-grained tiles，在 partial wave 中最小化 per-wave latency）两个候选集。按资源约束（SMEM≤SMEM_0, REG_spill==0）、指令约束（H100 wgmma 要求 bM%64==0）、SMEM 利用约束（one-tile-per-SM，确保 tile 足够大以有效利用 shared memory）过滤，再按 threshold l1/l2 保留性能接近最优的候选。
    2. 两层级联调度：第一级用 TO micro-kernel K1（如 128×128×64）处理 full waves。1672×1024 问题没有 full wave（112 tiles < 132 SM），退化为 LO-only scheduling。此时第二级用 LO micro-kernel K2（如 64×64×32）处理所有 tiles。更典型的例子是 M=1800, N=1024 → ~144 tiles：第一级 TO kernel 产生 1 个 full wave（132 tiles），第二级 LO kernel 处理剩余 12 tiles，fine-grained tiling 使 partial wave 执行时间显著缩短。
    3. 对比 Split-K/Stream-K：Split-K 和 Stream-K 沿 K 维度分割 partial wave 的 workload 到更多 SM，但引入 reduction sync 和额外 workspace（Stream-K 比 cuBLAS 多消耗 70%+ device memory）。HyTiS 无需沿 K 拆分——通过两级不同 tile size scheduling 直接避免 partial wave 的 SM 浪费，零同步开销。
    4. 自适应 tile layout：在 tile size 确定后，通过分析模型计算第一 wave 的 DRAM→L2 流量 V_1 选择最优 group size s_opt（s_opt^GM = min(ceil(sqrt(N_SM·bN/bM)), ceil(M/bM))）；计算所有 wave 的总流量 V_tol = ΣV_i，在 GM 和 GN 布局中选择 V_tol 更小的。实测 DRAM read 量减少（H100 上 low region 从 HyTiS(STL) 的 46% 降至 20%，high region 从 15% 升至 28%）。
  - **硬件架构层**：SM 内 Tensor Core（H100 wgmma）执行 micro-kernel 的 MMA 计算。TMA 指令做 asynchronous global→shared memory 数据搬运（H100 only）；A100 上使用传统 ldmatrix 指令 + data-parallel launch。L2 cache 数据复用受益于 layout 调度：选取最优 (GM/GN, s) 后，同一 wave 内相邻 SM 对矩阵 A/B 的访问在 L2 中命中率更高。

  关键设计选择映射到 baseline 缺陷：
  - wave quantization → SM 利用率低：两级别联调度，full waves 用大 tile 保吞吐，partial wave 用小 tile 降延迟。
  - fixed tile layout → 次优 L2 cache affinity：分析模型自适应选择 GM/GN 布局和 group size，最小化 wave 粒度 DRAM→L2 流量。
  - Split-K/Stream-K 的同步开销：HyTiS 不沿 K 维度拆分，免 reduction sync。
  - Inductor-Triton fixed search space：offline profiling 构建 architecture-aware 候选集 + runtime adaptive search space（l1/l2 阈值动态调整）。

## MetaAttention: A Unified and Performant Attention Framework across Hardware Backends

- baseline方法是什么？
  **手写 CUDA/Triton attention kernel（FlashAttention/FlashMLA/Mamba2 chunk kernel）+ compiler fallback（PyTorch SDPA/TorchInductor）**：现有 attention 生态中，高性能 attention kernel 均为手工优化实现，每种 attention 变体（Softmax/Sigmoid/ReLU/MLA/Mamba2/RetNet/Gated Retention 等）需要独立手写完整的 CUDA 或 Triton kernel（如 FlashMLA 1000+ 行 CUDA, Mamba2 3000+ 行 Triton）。Kernel 内部执行策略（tiling scheme, memory placement, pipeline stages, warp specialization）全部 hardcode 并针对特定 attention pattern 和特定 GPU（如 H100）优化。对于不支持的变体（如 ReLU Attention, RetNet Parallel），用户被迫 fallback 到 PyTorch native implementation——每个操作（matmul + normalization + mask）作为独立 kernel launch，中间 tensors 全部经 HBM round-trip，性能极差。Compiler-based 方案（TorchInductor, TVM, TensorRT）虽减少开发量，但无法理解 attention 的语义（如 online softmax），将 attention 视为离散 opaque 操作序列，无法生成 fused attention kernel。

  全栈执行例子（FlashAttention-3 on H100, LLAMA-3.1-8B Softmax Attention, seqlen=8K, head=32, dim=128, bf16）：
  - **模型推理算法层**：Standard scaled dot-product attention: O = softmax(QK^T/√d)V。FlashAttention-3 使用 tiled online softmax + warp-specialized asynchronous pipeline。
  - **系统框架层**：PyTorch v2.5.0 通过 SDPA API 调用 FA3 CUDA kernel。若遇到不支持变体（如 ReLU Attention），fallback 到 `torch.nn.functional.scaled_dot_product_attention` → PyTorch decomposes 为独立 cuBLAS matmul + softmax kernel + matmul → 3 次 kernel launch，中间 N×N attention matrix 经 HBM round-trip。
  - **编译框架层**：FlashAttention-3 为手写 CUDA C++ kernel，nvcc 编译。无自动 lowering pipeline——修改 attention pattern（如从 softmax 改 sigmoid）需重写 kernel。TorchInductor 无法自动生成 fused attention kernel（缺乏 online normalization 语义理解和双 GEMM fusion 能力）。
  - **kernel调度层**：FA3 kernel 内 hardcoded 调度——producer warpgroup (TMA load K/V tiles from HBM→SMEM) ∥ consumer warpgroup (wgmma QK^T → CUDA core online softmax: FMNMX + MUFU.EX2 + rowsum → wgmma PV → rescale O)。Tile size B_r/B_c 根据 head_dim=128 手写固定；若 head_dim 改为 192（DeepSeek-V2-Lite），FA3 需 padding 到 256 对齐 MMA tile，浪费 compute。Pipeline stages=2 (Pingpong scheduling)，register allocation 通过 setmaxnreg 手动分配。Strategy 硬编码：任何 attention 变体或 hardware 变更（如移植到 AMD MI250）需重写全部 execution strategy。
  - **硬件架构层**：NVIDIA H100 SXM5 (132 SMs, Tensor Core wgmma, TMA, 228KB SMEM/SM)。FA3 forward 达到 ~740 TFLOPs（~75% peak），但仅限 H100+Softmax Attention+causal mask。移植到 AMD MI250 需重写全部 kernel（ROCm Matrix Core, async copy → non-trivial porting）。

  Baseline 缺陷：
  - (a) **"Software lottery"——注意力变体性能取决于是否有手写 kernel**：支持的变体（Softmax+causal）性能优异，不支持变体（Sigmoid/ReLU/RetNet parallel）fallback 到 PyTorch native，性能差 5-10×
  - (b) **Hardcoded scheduling 不适应非标准 shapes**：FA3 固定 B_r/B_c 基于 head_dim=128 优化；Diff-Transformer-3B (dimqk=128≠dimv=256) 或 DeepSeek-V2-Lite (dimqk=192) 需 padding 对标对齐，浪费 compute 和 memory
  - (c) **Hardware lock-in——每 GPU 重新手写 kernel**：FA2 在 A100 上达 70% peak throughput，但移植到 H100 仅 30% peak；需引入 register-level pipelining 和 ping-pong kernel design 才能达到 H100 peak；移植到 AMD GPU 更困难
  - (d) **Recurrent attention（Mamba2/RetNet）缺乏 fused kernel**：手写 Triton kernel（Flash-Linear-Attention）有优化，但仍有大量 HBM intermediate traffic；无法像 FA3 那样利用 online computation + hardware-specific asynchrony
  - (e) **Compiler 无法理解 attention 语义**：TorchInductor/TVM/TensorRT 将 attention 视为 opaque 操作序列，无法实现 online softmax/sigmoid/norm 等 attention-specific fusion

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **MetaAttention：统一 attention 抽象 + 可定制模板 + 跨硬件自动调度框架**。

  对应关系：
  - (a) → **统一 abstraction + customizable templates**：将 attention 抽象为 relevance scoring (QK^T) + aggregation (PV) 两个固定操作，通过 Parallel Pattern（全局上下文，matmul-based）和 Recurrent Pattern（压缩 state，iterative update）两种模式实例化。Customizable functions（Mod 元素级变换 + RowNorm 行归一化 + RowNorm online interface）提供任意 attention 变体的表达能力——用户仅需 22-90 行 Python 定义 template+functions 即可获得高性能 kernel。支持 10+ attention variants（Softmax/Sigmoid/ReLU/MLA/Mamba2/RetNet/YOCO/RFA/Sparse GQA/Sliding Window），消除 "software lottery"。
  - (b) → **IntermediateTensor-based scheduling 自适应 shape**：不 hardcode tile size——外层 Tile Config Scheduling 枚举所有合法 output tile sizes（对齐 basetile 但不受限于固定 head_dim），通过 computation graph 自动传播 tile shape 到所有 IntermediateTensors。对 dimqk≠dimv（如 Diff-Transformer-3B dimqk=128, dimv=256），scheduler 自动选择非等长 tile sizes 避免 padding waste，在 Diff-Transformer-3B forward 实现 1.61× speedup over FA3。
  - (c) → **DeviceConfig + multi-backend runtime**：硬件约束抽象为 DeviceConfig（BaseTileShape + MemoryInfo）→ 同一套 scheduling policy 适配不同 hardware。NVIDIA backend 通过 TileLang 和 CUTE 使用 TMA + Tensor Core；AMD backend 通过 TileLang 使用 Matrix Core + async copy。MI250 上平均 3.3× forward speedup over baselines，无需 per-GPU 重写 kernel。
  - (d) → **Recurrent Pattern 统一支持**：Recurrent pattern 将 Mamba2/RetNet Recurrent 等 stateful attention 统一为 "matmul(Q, h) + h = h + matmul(K^T, V)" 的固定模板 + h_mod customizable function。Attention runtime 对 recurrent pattern 实现 chunk parallelism 技术 [32] 最大化 hardware utilization。Mamba2 forward 1.66×/backward 1.78× vs Flash-Linear-Attention。
  - (e) → **Two-layer scheduling policy 理解 attention 语义**：IntermediateTensor 建模揭示 attention computation graph 的 dataflow——tile propagation 确保 online normalization 的 tile dependency 正确传播；RowNorm online interface 将 online softmax/sigmoid/L2-norm 等标准化为统一接口供 scheduler 推理；code inlining 将 customizable functions fused 到 attention mainloop，实现与 handcrafted kernel 同等的 memory-efficient pipelining。

  全栈执行例子（MetaAttention Diff-Transformer-3B Softmax Attn, H100, seqlen=8K, dimqk=128≠dimv=256, bf16）：
  - **模型推理算法层**：用户定义 Parallel Pattern + RowNorm online softmax (scores_RowNorm_Online) + scores_Mod (causal mask) + Q_mod (scale by 1/√d)，约 87 行 Python。
  - **系统框架层**：用户调用 MetaAttention Python API → 生成 scheduling plan → attention runtime 生成 kernel → 替换 PyTorch Transformers 中的 attention 调用。无 PyTorch decomposition，单次 kernel launch 完成全部 attention computation。
  - **编译框架层**：
    1. Customizable Function Lowering: trace scores_Mod (Mul with mask) + scores_RowNorm_Online (ReduceMax+Exp+ReduceSum+Div chain) → elementwise + row-reduce DAG → hardware-mapped code snippets
    2. Scheduling Space: IntermediateTensors = {Q, K, scores, weights, V, output} + customizable function internal tensors
    3. Tile Config Scheduling: Enumerate output tiles for dimv=256（vs FA3 固定 pad 到 256 对齐 dimqk=128→256 padding）→ 可自然支持 dimqk≠dimv 的 tile shapes
    4. Tile Resource Scheduling: 分配 memory——Q(128×128×2B=32KB)→SMEM, K(128×128×2B=32KB)→SMEM, scores(128×128×2B=32KB)→SMEM, V(128×256×2B=64KB)→SMEM, weights(128×128×2B=32KB)→RF, output(128×256×2B=64KB)→RF accum then SMEM；检查 SMEM total=32+32+32+64+64=224KB ≤ 228KB → valid
    5. Profiling 选最优 plan
  - **kernel调度层**：Attention runtime 根据 plan: TMA async load Q tile → SMEM, TMA load K,V tiles → SMEM (pipeline stage=2: prefetch next tile while computing current) → wgmma QK^T [128,128]×[128,128]^T → CUDA core online softmax (ReduceMax + MUFU.EX2 + ReduceSum + rescale, all in RF) → wgmma PV [128,128]×[128,256] → rescale output → TMA store。Customizable functions (mask, scale) inline 到 CUDA core region，零额外 launch overhead。
  - **硬件架构层**：NVIDIA H100 SXM5 (TMA + wgmma + async pipeline)。与 FA3 不同：tile size 自适应 dimqk=128/dimv=256 无需 padding，tiling scheme 由 scheduler 而非手写决定。同时同一套 template 可通过 DeviceConfig(MI250) 移植到 AMD MI250——BaseTileShape 适应 Matrix Core (64×64)，MemoryInfo 适应 MI250 hierarchy，无需改任何 Python 代码。

  关键设计选择映射到 baseline 缺陷：
  - attention variants 不统一 → relevance scoring + aggregation abstraction，两种 pattern 覆盖全部变体
  - hardcoded scheduling → IntermediateTensor + DeviceConfig + two-layer scheduling policy 自动化
  - hardware lock-in → DeviceConfig 抽象 + TileLang/CUTE multi-backend mapping
  - compiler 不理解 attention → RowNorm online interface 标准化 online normalization，scheduler 传播 tile dependency
  - 开发成本高 → 22-90 LoC Python vs 400-3000 LoC CUDA/Triton

## UltraAttn: Efficiently Parallelizing Attention through Hierarchical Context-Tiling

- baseline方法是什么？
  **Ring-based Context Parallelism（Ring Attention, ZigZag-Ring Attention, Striped Attention）**：现有的 context parallelism 系统沿单一维度（通常 $c_q$）用 stripe-like partition 将 attention workload 划分为带状分配给各 GPU，并通过 ring-based communication pattern 在 GPU 间轮转 KV。具体执行过程：在 ring attention 中，每个 GPU 持有部分 Q（沿 $c_q$ 维划分）和部分 KV（沿 $c_{kv}$ 维划分），通过 send/recv 在环中轮转所有设备持有的 KV，执行 per-step 的 attention 计算（每次一个 step：本地 Q × 当前持有的 KV → 部分 attention → 将 KV 传给下一个 rank → 从上一个 rank 接收新的 KV）。zigzag ring attention 在此基础上调整 stripe 顺序以获得更好的 causal attention 负载均衡；striped attention 对 Q 和 KV 做交织划分。

  全栈执行例子（Llama2-7B causal attention training, CP=64, S=512K, 8 nodes × 8 H100）：
  - **模型推理算法层**：causal attention 计算 $O = \text{Softmax}(\text{Mask}(QK^T/\sqrt{d_k}))V$。64 GPUs 在 $c_q$ 维度并行，每个 GPU 负责约 512K/64 = 8K tokens 的 Q 块。KV 在所有 GPU 间通过 ring 轮转。
  - **系统框架层**：PyTorch + Megatron-LM（或类似框架）管理 context parallel group。每个 GPU 发射 FlashAttn kernel 对本地 Q × 当前持有的 KV 做 attention。通过 NCCL send/recv 在 ring 中传输 KV。
  - **编译框架层**：论文未明确说明。使用 PyTorch 原生框架，无自定义编译 pass。
  - **kernel调度层**：
    1. Stripe-like partition：attention workload 沿 $c_q$ 维划分为 64 个 stripe（每个 GPU 1 条）。Stripe 形状为 $1 \times N$（1 维划分），沿 Q 和 KV 的 projection lengths 均为 O(N)。
    2. Ring-based communication：每个 step，每 GPU 同时执行三个任务——本地 attention block 计算（FlashAttn kernel）、发送当前 KV 到下一个 rank、从上一个 rank 接收新的 KV。Step 数 = CP = 64（对于 ring attention, step=CP 意味着 KV 轮转完整一圈）。
    3. Fine-grained kernel split：每个 step 的计算量极小（约 8K × 8K attention），以最大化 computation-communication overlap，但导致极低的单 kernel device utilization（SM occupancy 低）。
    4. Inter-node bottleneck：跨节点的 ring 连接在 8 节点时仅利用 2 个 NIC 的单向带宽（图 3a，ring 经过每个 node 时仅使用出/入各 1 NIC），浪费 75% 的 NIC 带宽。
    5. Redundant communication：zigzag ring attention 约 25% KV 传输是浪费的（对应当前 GPU 不需要的部分 KV blocks），标准 ring attention 接近 50% 浪费。
  - **硬件架构层**：8 节点 × 每节点 8× H100-NVLink-80GB（共 64 GPU）。节点内 NVLink 450GB/s 双向。节点间 8× 400Gb/s InfiniBand EDR，每 GPU 与 1 NIC 有 PCIe-5.0 affinity。Ring 通信未利用网络拓扑异构性——NVLink（intra-node）和 InfiniBand（inter-node）带宽差异巨大，但 ring pattern 将它们同等对待。

  Baseline 缺陷：
  - (a) **High Communication Traffic**：stripe-like partition（$1 \times N$ 形状）的 workload projection sum 是 O(N)，而 ideal curled-up partition（$\sqrt{N} \times \sqrt{N}$）的 projection sum 是 $O(\sqrt{N})$，相差一个数量级。stripe 瘦长形状导致每 GPU 需要接收大量不必要的 Q/KV 投影。
  - (b) **Inflexible Kernel Granularity**：ring-based 系统为最大化 computation-communication overlap 将 attention 拆分为极细粒度 kernel（每 step 一个小 kernel），但过度细粒度导致单 kernel SM utilization 极低。kernel granularity 存在 U 型性能曲线——太细则设备利用率低，太粗则重叠机会少。
  - (c) **Bandwidth Waste of Ring Communication**：ring 模式将所有 KV 轮转到每个设备，但在 block sparse attention 中许多设备只需部分 KV，导致 ~25%（zigzag ring）到 ~50%（ring）的冗余通信。跨节点时 ring 仅使用 2 NIC 单向带宽（75% NIC 带宽浪费）。
  - (d) **Poor Strong Scalability**：随着 CP 增加（固定 context length），每 GPU 的 computation volume 反比下降（$O(1/CP)$），但 ring pattern 的 communication volume 几乎不变，导致逐渐 communication-bound。
  - (e) **No Support for Irregular/Block Sparse Attention**：ring-based 系统的 step-by-step structure 在 block sparse attention 下遭遇 severe in-step load imbalance——每 step 只有部分 GPU 需要计算 attention block，其余 GPU 空等。
  - (f) **Sub-optimal Kernel Scheduling**：现有系统（FlexFlow、Tofu 等）使用 BFS-based 方法寻找 feasible topological order，但无法保证最优调度顺序。通信 contention（共享同一带宽的 kernel 重叠执行会延长各自执行时间）未被避免。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **UltraAttn：Hierarchical Context-Tiling System**，通过三层 context-tiling（node-level, device-level, kernel-level）+ ILP-based runtime 实现高效的 context parallelism for irregular attention。

  对应关系：
  - (a) → **Device-Level 2D Context-Tiling**：将 attention workload 沿 Q 和 KV 两个维度同时划分（而非仅一维），形成 curled-up ($\sqrt{N} \times \sqrt{N}$) 形状的 tile。通过 ILP 在 $P \times P$ grid 上分配每个 block $B_{r,c}$ 到设备 $U_g$，最小化 MCV（Maximum Communication Volume）。形式上：定义 partition degree P，将 workload 划分为 $P \times P$ grid，ILP 变量 $x_{r,c,g}$ 控制分配，约束 Allocate Uniqueness（每 block 到唯一设备）+ Computation Balance（$\sum_{FB} x \times 1 + \sum_{CB} x \times 0.5 \le \tau$），最小化 MCV。从 stripe-like 的 O(N) projection sum 降为 curled-up 的 $O(\sqrt{N})$，通信量降低一个数量级。
  - (b) → **Kernel-Level Context-Tiling**：在 parallel dependency graph（DAG）上通过三种 transformation（computation batching, peer-to-peer comm batching, collective comm batching）进行图变换，使用贪心策略自适应选择最优 kernel granularity。三种 substitution 的 transformation candidates 按 gain（融合后减少的时间）降序排序，贪心应用到 DAG → ILP runtime 评估执行时间 → 保留改善的变换。从而在 kernel overlap（灵活性）和单 kernel device utilization（效率）之间找到最优平衡点。对 dense attention 特别有效（kernel fusion 机会多），小 $\frac{S}{CP}$ 和 Nh 时增益更大。
  - (c) → **Node-Level Context-Tiling + Groupwise Peer-to-Peer**：将 context-tiling 解耦为 node-level 和 device-level 两层。Node-level tiling 将每个 node 视为集成设备，仅在 node 间执行 minimized 通信。节点间使用 groupwise peer-to-peer（图 3b），每个 node pair 通过多个 NIC 并行通信，充分利用全部 NIC 带宽（vs ring 的 75% 浪费）。Device-level tiling 在 node 内使用 NVLink peer-to-peer。两层建模类比统一——workload block ↔ node 的 distributed attention computation，peer-to-peer ↔ groupwise peer-to-peer，profiling 方式对应。
  - (d) → **2D Partition + Communication Minimization ILP**：当 CP 增加时，2D curled-up partition 的 communication volume 也随 $1/\sqrt{CP}$ 下降（vs ring 的常数），保持 communication 与 computation 的比例。ILP 显式最小化 MCV 确保 communication 不成为 bottleneck。实验验证 UltraAttn 在 CP=16→64 时实现 near-linear strong scalability（图 10）。
  - (e) → **ILP 自适配不规则 Workload**：ILP formulation 通用处理任意 FB/CB/EB 集合定义的 attention pattern（causal, full, strided, global+local, star, streaming），自动计算最优 workload-to-device 分配。Device-level tiling 打破 ring-based 的 step-by-step 结构，允许更灵活的通信模式（不限于顺序轮转），从而消除 in-step load imbalance。对于 block sparse attention，UltraAttn 在 CP=64 时获 $10.2\times$（strided）到 $5.7\times$（global+local）加速。
  - (f) → **ILP-based Runtime Kernel Scheduling**：将 DAG 的 kernel 调度建模为 ILP——按共享带宽分组 kernel 到不同 CUDA stream（避免 contention），对每 stream 求解 ILP（变量 $S_v$ + $Order_{uv}$，约束 Stream Exclusivity + Dependency，目标 minimize $End\_Time$），获得理论最优执行顺序。对比 FlexFlow BFS-based scheduling，ILP 方案在涉及复杂 kernel 依赖和 comparable computation-communication duration 的场景下有显著优势。

  全栈执行例子（UltraAttn strided attention training, CP=64, S=512K, Nh=1, 64 GPU）：
  - **模型推理算法层**：strided attention（Figure 2c diagonal stripes pattern），$O = \text{Softmax}(\text{Mask}(QK^T/\sqrt{d_k}))V$。与 baseline 相同的 attention 计算逻辑，但 mask pattern 为 stride-based sparse。
  - **系统框架层**：UltraAttn 作为 PyTorch 库（~10K LoC Python）。运行时：读取 attention pattern → 执行 hierarchical context-tiling → 生成 parallel dependency graph → ILP runtime 调度 → CUDA graph 编译 → GPU 执行。Context remap 预处理（$\phi(t_i) = \lfloor i \cdot 16/S \rfloor \mod 4$）增强 locality。
  - **编译框架层**：论文未明确说明。UltraAttn 直接使用 FlashAttn 2.5.7 作为 computation backend，NCCL 2.21 C-level API 作为 communication backend。无自定义编译 pass。
  - **kernel调度层**：
    1. Adaptive Workload Partition：计算 P 使 $DLI_{P,CP} \le \theta_{DLI}$，将 strided pattern 划分为 $P \times P$ grid
    2. Node-Level ILP（$CP_{node}=8$）：将 $P \times P$ grid 的 blocks 分配到 8 nodes，minimize MCV → node 间 groupwise peer-to-peer 通信计划
    3. Device-Level ILP（$CP_{device}=8$）：每个 node 内的 blocks 分配到 8 GPU，minimize MCV → GPU 间 peer-to-peer 通信计划
    4. DAG 构建：computation kernel（FlashAttn, 矩形节点）+ recv kernel（NCCL recv, 椭圆节点）+ send kernel（NCCL send, 菱形节点）构成 DAG
    5. Kernel-Level Tiling：贪心选择变换（computation batching/comm batching）应用到 DAG
    6. ILP Runtime：按共享带宽分组 kernel → 各 stream 内 ILP 求解最优顺序 → CUDA stream graph
    7. CUDA Graph 执行：各 stream 并行执行，stream 内串行，computation（FlashAttn forward）与 communication（NCCL send/recv）按 ILP 最优 schedule 交错
  - **硬件架构层**：8 nodes × 8 H100-NVLink-80GB。Node-level tiling 使用 groupwise peer-to-peer（每 node pair 利用 8 NIC 并行），device-level tiling 使用 NVLink peer-to-peer。最终 64 GPU 的 distributed attention 模块强可扩展性接近线性（图 10a）。

  关键设计选择与 baseline 缺陷的对应：
  - **defect (a): stripe-like high traffic** → 2D context-tiling：从 $1 \times N$ partition 变为 $\sqrt{N} \times \sqrt{N}$ curled-up partition，projection sum 从 O(N) 降至 $O(\sqrt{N})$
  - **defect (b): inflexible kernel granularity** → 贪心 kernel-level tiling + ILP runtime 评估：遍历三种 substitution 的 bounded 搜索空间，贪心选择，通过 ILP runtime 准确评估执行时间
  - **defect (c): ring bandwidth waste** → node-level tiling + groupwise peer-to-peer：仅在 node 间传必需数据，充分利用所有 NIC
  - **defect (d): poor strong scalability** → 2D tiling 的 communication 也随 CP 增加而下降：$1/\sqrt{CP}$ communication scaling vs ring 的 constant
  - **defect (e): no support for irregular attention** → ILP formulation 通用处理任意 FB/CB/EB pattern：打破 step-by-step 结构消除 in-step load imbalance
  - **defect (f): sub-optimal kernel scheduling** → ILP runtime 形式化 kernel scheduling：stream exclusivity + dependency constraints，理论上最小化 $End\_Time$
