## 9-H2-LLM- Hardware-Dataflow Co-Exploration for Heterogeneous Hybrid-Bonding-based Low-Batch LLM Inference.pdf

- baseline方法是什么？
  Baseline是现有In-Die NMP-based异构LLM加速器设计，以Samsung LPDDR5-PIM [39]、SK-Hynix GDDR6-AiM [37]和NeuPIMs/AttAcc [24,60]为代表。全栈执行例子（LLaMA3 8B, batch size 4, 8 memory channels, decoding stage）：
  - **算法pipeline**：标准transformer-based LLM（MHA/GQA/MQA），FP16精度，prefill和decoding两阶段。无量化/稀疏优化。
  - **系统框架/Serving调度**：论文未明确说明Serving框架（硬件架构层面研究）。
  - **编译框架**：论文未明确说明（无自动化编译框架，operator mapping由designer手动固定或SpecPIM [47]的compute-centric exploration决定）。
  - **kernel调度**：现有数据流设计固定operator mapping——GPU+HBM-PIM [39] 固定offload所有FC到NMP（FC only）、GPU+GDDR6-AiM [37] 固定offload单batch FC+Attention到NMP、NeuPIMs [24] 固定offload Attention only、AttAcc [60] 固定offload Attention + Fixed Fission of FFN。SpecPIM [47]的compute-centric abstraction先分配computation engine（centralized processor或NMP PEs）再决定operator placement——operator被约束到单一类型channel（normal或NMP），导致外部memory bandwidth被限制，compute-bound prefill operators可能变为memory-bound。
  - **硬件架构**：In-die NMP——PIM处理引擎嵌入DRAM die内部，受DRAM工艺限制（transistor比CMOS慢3×，逻辑密度低10×，metal layers少）。Samsung LPDDR5-PIM: 1.2TFLOPS/cube (9.6GFLOPS/PE), 1 FLOP/Byte computation-bandwidth ratio。SK-Hynix GDDR6-AiM: 512GFLOPS/ch (32GFLOPS/PE), 1 FLOP/Byte。**根本缺陷**：
    - **Computation Capacity不足**：in-die NMP极低computation capacity（computation-bandwidth ratio仅1-2 FLOP/Byte），batch size增加时speedup急剧衰减。当batch size≥8或KV head≤4时in-die NMP完全无法提供性能提升。
    - **Fixed Operator Mapping无法适应低batch场景**：固定mapping在大batch cloud inference有效（operators arithmetic intensity差异大），但在低batch inference无法充分利用NMP加速能力——所有operators（含attention）均memory-intensive，固定mapping限制了可被NMP加速的operator范围。
    - **Prefill-unaware dataflow exploration**：SpecPIM的compute-centric mapping忽略prefill性能——operator被约束到单一channel类型后减少了centralized processor可用external bandwidth，compute-bound prefill operators因bandwidth不足降级为memory-bound，损害end-to-end性能。
    - **HB computation-bandwidth trade-off未被探索**：Hybrid Bonding虽有高bandwidth（3μm pitch, 110,000 I/O/mm²）和低power的优势，但HB controller占logic die面积大（40% @1024 pins 40nm），computation和bandwidth的trade-off缺乏系统性design space exploration。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出H2-LLM，通过三个核心技术解决baseline缺陷：
  (1) **HB-NMP异构架构设计空间**：首次系统性地探索HB固有computation-bandwidth trade-off——通过参数化HB I/O bandwidth（6.4-51.2GB/s, 128-1024 pins）、FPU数(1-8)/频率(0.4-1GHz)、buffer sizes，在给定PE area(6.76mm²)和HB controller area占比约束下搜索最优配置。解决了baseline in-die NMP computation capacity不足的缺陷。
  (2) **Data-Centric Dataflow Abstraction**：替代SpecPIM的compute-centric mapping——先bind memory channels到operators（而非先分配computation engine），三阶段binding：MAG partition（split operator graph到Memory Access Groups）→ Coarse-grain binding（GCMap分配channel subsets）→ Fine-grain binding（OCMap分配operator-tier到具体channels）。支持flexible operator fission（operator可同时分配到normal和NMP channels沿output feature dim N分割），保持full external bandwidth对centralized processor可用。解决了baseline fixed mapping无法适应低batch场景、prefill-unaware导致性能下降的缺陷。
  (3) **Hardware-Dataflow Co-Exploration DSE Framework**：Genetic algorithm-based自动搜索architecture-design co-design space——Population Generator随机采样→Capacity Checker过滤非法配置→Ramulator2+Tileflow Evaluator评估latency/energy→Selector选Top-K→Genetic operators (Re-sample/Mutate/Crossover)进化→迭代至最优。支持多workload联合优化。

  全栈执行例子（LLaMA3 8B, batch size 4, 8 memory channels含4 HB-NMP+4 normal, decoding-heavy ShareGPT scenario）：
  - **算法pipeline**：标准LLM decoding/prefill，FP16。H2-LLM不修改算法。
  - **系统框架/编译框架**：Model parser从ONNX提取operator graph和tensor shapes → DSE engine用genetic algorithm搜最优architecture-dataflow co-design → Model compilation生成execution flow（NMP operators用Eq.1求解optimal tiling并匹配NMP operator templates，centralized processor operators用Tileflow生成）。
  - **kernel调度/数据流**：Data-centric dataflow abstraction的三阶段binding：(1) MAG Partition将parallel transformer layer split为A0(Attention block ops)和A1(FFN block ops)两MAG；(2) Coarse-grain binding每个MAG内partition weakly connected components为MPGs并分配channel subsets；(3) Fine-grain binding按operator dependency stratify tiers并分配具体channels。例如：Q assigned to 4 NMP channels，K assigned to 4 normal channels（两者并发执行）→ QK operator fission分配到NMP channels处理部分特征+normal channels处理其余部分→ 同步merge结果→SV到NMP→O fission继续。Prefill阶段operators全部在centralized processor执行（利用batch中hundreds-thousands tokens的high parallelism），避免prefill被NMP bandwidth限制。
  - **硬件架构**：HB-NMP PE——每个PE含FPUs（如8 FPU@0.6GHz, 25.6GB/s HB I/O, balanced设计），output-stationary execution flow（Input tile→input global buffer→各PE weight tiles+MAC→output tile write-back→循环），无inter-PE NoC。4 HB-NMP channels + 4 normal channels。HB controller占logic die area约19.7%（25.6GB/s对应512 pins）。

  核心设计映射：
  - 缺陷①（in-die NMP computation capacity不足）→ H2-LLM的HB-NMP architecture：通过HB在logic die上customize PEs引入更高computation capacity（FPU@0.6-1GHz vs in-die NMP@200MHz）→ 2.72× geomean speedup vs in-die NMP+。
  - 缺陷②（fixed operator mapping不适应低batch）→ Data-centric dataflow abstraction：flexible operator-channel binding + operator fission支持任意operator分配到任意channel组合→ 1.37× speedup vs FC-NMP。
  - 缺陷③（prefill-unaware exploration）→ Prefill-aware data-centric abstraction：prefill operators全部在centralized processor执行保持full external bandwidth→ prefill latency 1.27× geomean speedup vs CC-NMP。
  - 缺陷④（HB trade-off未被探索）→ Architecture DSE在computation-bandwidth trade-off空间搜索balanced设计→ 1.38× speedup和1.74× energy efficiency vs fixed design。

## 87-MHE-TPE- Multi-Operand High-Radix Encoder for Mixed-Precision Fixed-Point Tensor Processing Engines.pdf

- baseline方法是什么？
  Baseline是典型的空间GEMM加速器架构（OS/WS systolic array、multiplier-adder tree、bit-serial架构），使用MBE（Modified Booth Encoder）乘法器进行定点GEMM计算。全栈执行例子（INT8 GEMM, OS systolic array @1GHz）：
  - **算法pipeline**：标准GEMM C=A·B，A和B均为定点整数，MBE编码将乘数的PP数量减半，compressor tree + full adder完成PP归约。
  - **系统框架/Serving调度/编译框架/kernel调度**：论文未明确说明（直接硬件层面设计，不涉及上层软件框架）。
  - **硬件架构**：典型OS systolic array PE微架构（Fig.1a）：(1) MBE Encoder读入3-bit multiplicand窗口(a_{2i+1}, a_{2i}, a_{2i-1})，查表生成系数∈{-2,-1,0,1,2}；(2) Candidate Partial Product Generator (CPPG)生成PP（INT4: 2 PP, INT8: 4 PP）；(3) Compressor Tree (4-2 compressor / Wallace Tree)归约多operand PP；(4) Full Adder产生最终乘积；(5) Accumulator跨cycle累加。**两个根本缺陷**:
    - **冗余PP归约（Redundant PPs Reduction）**：❶Temporal维度：同一PE内不同cycle的MBE编码系数因MBE输出域对称性({-2,-1,0,1,2})产生inverse-signed identical reductions（如2B1,T+B1,T+1 vs -(2B1,T+B1,T+1)），独立bit-slice分布却产生等价归约；❷Spatial维度：同一column PEs因systolic传播同一B值产生跨PE的重复归约模式。根因是传统scalar dataflow并行下的isolated computation — 硬件不支持跨PE协作。
    - **混合精度计算密度失衡（Compute Density Imbalance）**：低bit-width乘法器的reduction dimension不足。INT4乘法器仅用1个full adder（1级归约），INT8乘法器用4-2 compressor tree（更高效），因此4个INT4拼成INT8时仅达57% compute density；NVIDIA A100 INT4×INT4理论上应达4×INT8吞吐，实际仅2×（1248 vs 624 TOPS）。根因是乘法器的乘法逻辑和归约逻辑物理解耦 → compressor tree无法适配multi-precision weighted PP → accumulator位宽无法动态适配。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出MHE-TPE架构，核心设计理念：**将乘法器内部的bit-slice归约维度与向量内积的空间归约维度融合**，通过dual-operand joint encoding在半数PP基础上再压缩一半。

  **全栈执行例子**（INT4 GEMM, MHE-TPE Array, 配置: MHD fanout=8, Tree=16, M=32, K=32, NT=64, fast clock 4GHz / slow clock 1GHz, UMC 22nm）：

  - **算法pipeline**：改造后GEMM算法（Algorithm 2）— 将C=A·B分解为：(1) 对B矩阵每2列(B2k, B2k+1)配对，预计算8个reduced linear combinations存入VPP LUT；(2) A矩阵4-bit双操作数切片(Am,2k⟨i⟩, Am,2k+1⟨i⟩)经MHE编码为选择信号(CE, S)；(3) MHD根据(CE, S)查VPP LUT直接生成VPP（跳过传统乘法步骤）；(4) Compressor Tree归约K/2个VPP（传统需K个PP，减少一半）；(5) 混合精度mapping：A矩阵精度→temporal mapping（⌈LA/2⌉ cycles遍历bit-slices），B矩阵精度→spatial mapping（4-bit sliced tiles分配）。

  - **硬件架构**：**Preprocessing Phase**：B矩阵的每2列经VPP Controller用1个adder serially生成6个派生项（B2k+B2k+1, B2k-B2k+1, ...），与2个原始项（B2k, B2k+1）共8项存入VPP LUT（DFF存储，深度8，位宽LB+2），实现TPU-like WS dataflow。**Computation Phase**：(1) MHE编码：双MBE同步读入Am,2k⟨i⟩和Am,2k+1⟨i⟩的4-bit（各2-bit），通过VPP Select Encoder生成CE和5-bit S[4:0]；(2) VPP生成：MHD根据(CE,S)查VPP LUT输出VPP，其中Map单元将8个基本状态扩展为24个有效状态（信号扩展+补码计算）；(3) PP归约：各行的Compressor Tree（K/2 inports）归约同列VPP → Full Adder → PS累加；(4) 跨tile归约：LRM通过pipeline buffer对齐不同tile的bit-slice partial product权重，执行bit-shifted累加；(5) 混合精度支持：A矩阵变化仅需扩展LRM accumulator bit-width；B矩阵精度通过spatial tile分配缩放（INT4: 1 tile/col, INT8: 2 tiles/col, INT16: 4 tiles/col, INT32: 8 tiles/col）；(6) 双时钟域：MHE/MHD @4GHz快速填充 → 时间复用使MHD单元数降为M/4，单bit valid握手同步到compressor tree @1GHz。

  **核心设计→缺陷映射**：
  - **缺陷①（冗余PP归约）→ MHE dual-operand encoding + VPP LUT共享**：通过将相邻2列B的MBE编码结果合并为8个线性组合的lookup table，使VPP归约维度从K减至K/2。MHE的selection signals可沿M维度广播复用，一个VPP LUT驱动多个MHD，硬件overhead分摊至可忽略。
  - **缺陷②（计算密度失衡）→ 三阶段计算范式 + spatiotemporal mapping**：Bit-slice encoding → VPP generation → unified reduction消除了传统乘法器的multiplication/reduction物理解耦。Compressor tree参数仅由VPP LUT位宽决定，与操作数精度解耦。A精度通过temporal mapping（cycle数×⌈LA/2⌉），B精度通过spatial tile mapping，4A4B天然实现4×4A8B的4× compute density（vs baseline仅2×），实现理论scaling（halving both operands → 4× throughput; halving one operand → 2× throughput）。

- baseline方法是什么？
  Baseline是现有稀疏加速器的tiling策略，以Tailors[38]/DRT[25]/HARP[19]为代表。三类baseline各有缺陷：
  - **Tailors[38]**：纯静态方法，用10% overbooking heuristic固定tile size，tile shape固定沿k→j→i顺序扩展，inter-tile order固定j first，buffer allocation固定不分。全栈执行例子（SpMSpM with Gust dataflow, kron_g500-logn18矩阵, 32 PEs @1GHz, 4MB SRAM, 4 DDR4 channels）：
    - **算法pipeline**：标准SpMSpM (Einsum Cij = Aik × Bkj)，Gustavson dataflow (i⊲k⊲j)
    - **系统框架/编译框架**：论文未明确说明
    - **kernel调度**：离线预先用10% overbooking决定tile size，运行时对所有tile使用相同size/shape，不随局部稀疏度变化调整。对mouse_gene矩阵这种变化显著的pattern，10% overbooking远非最优（实测最优overflow ratio大于2×）
    - **硬件架构**：固定buffer allocation，不支持flexible inter-tile order，tile shape固定k优先扩展。当矩阵pattern需要不同tile shape时（如TSOPF_FS_b300_c3需主要tile k，kron_g500-logn18需主要tile j），Tailors sub-optimal
  - **DRT[25]**：纯动态方法，tile size精确fit buffer，tile shape按k→i→j greedy在线生成，inter-tile order固定i first。但需要昂贵预处理（将tensor预先切为32×32 micro-tiles），metadata开销大；tile shape greedy算法倾向产生类立方体shape而非最优shape
  - **HARP[19]**：纯动态方法，仅沿i维度tiling（pseudo-tiles→super-tiles），inter-tile order固定i first，专为OP dataflow设计。缺乏flexibility
  三者共同的缺陷：(1) 未探索完整tiling设计空间（tile size/shape/inter-tile order/buffer allocation的组合优化）；(2) 纯静态不adaptive，纯动态实现复杂或受限；(3) 未有效管理tiling metadata开销

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出HYTE，混合静态-动态框架，通过四个关键设计解决baseline缺陷：
  (1) **完整设计空间探索**：静态调度器搜索tile size（含overflow允许）、tile shape（各维度独立power-of-two）、inter-tile order（3种选择=innermost loop choice）、buffer allocation（inter-tile reuse vs intra-tile reuse only）的组合空间（≤4×log²N个scheme），用cost model评估选最优→解决baseline固定参数导致sub-optimal的问题（Takeaway 1,2,4）
  (2) **轻量采样与估计**：以sp=1/√N比例采样，用hash-based方法（Flajolet-Martin变体）估计effMAC和nnzCTk，避免完整扫描输入tensor→解决稀疏场景下输出tensor size未知导致搜索困难的问题
  (3) **硬件metadata协同管理**：off-chip memory维护inter-tile位置metadata（fiber segment begin positions），on-chip buffer中data/metadata从两端协同增长（circular buffer for streaming, buffering for reuse），允许buffer space在data和metadata间灵活共享→解决metadata可能高达3.2×于data的开销问题（Takeaway 5）
  (4) **动态tile shape调优**：runtime硬件用4个象限计数器统计局部非零分布，估计9种调整后shape的hit rate，若超5%+则延迟到下一inter-tile iteration应用→补偿静态调度器的estimation error（平均15%, 最高43%）

  全栈执行例子（同硬件, mouse_gene矩阵, Gust dataflow）：
  - **算法pipeline**：同上SpMSpM，但HYTE调度器发现mouse_gene nnzC=18×输入，tiling k会导致nnzCTk暴涨→决定不对k维度tiling
  - **系统框架/编译框架**：CPU上静态调度器（Algorithm 1-2）采样sp=0.005, 估计effMAC/nnzCTk误差仅5-10%→搜索pruned空间→输出tiling scheme（tile shape, inter-tile order, buffer allocation）
  - **kernel调度/硬件架构**：Tiling controller加载scheme→按inter-tile order协调tile执行→accessor fetch fiber segments按buffer allocation（streaming/buffering模式）→data/metadata协同增长→runtime计数器监控局部非零分布→可选动态调整tile shape（若hit rate改善>5%）。当静态估计偏差大时（如mouse_gene），dynamic tuning提供额外1.9×加速
  - **硬件架构**：RTL实现tiling controller (0.6% area)和accessors (3.1% area)，总area overhead仅3.7%（13.78mm² in TSMC 28nm）

  核心设计映射：Baseline缺陷（搜索空间不完整+纯静态/纯动态各有缺陷+metadata开销未管理）→ HYTE解决方案（4维完整搜索空间探索+采样估计+静态生成near-optimal初始scheme+硬件动态微调补偿误差+data/metadata协同buffer管理）

## 83-PAPI- Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System .pdf

- baseline方法是什么？
  Baseline是现有PIM-enabled heterogeneous LLM accelerator design，以A100+AttAcc为代表[74]：采用static scheduling将FC kernel固定调度到GPU PUs、attention kernel固定调度到AttAcc PIM (1P1B) units。全栈执行例子（GPT-3 175B, batch size在runtime从64降到4变化, speculation length=4）：
  - **算法pipeline**：LLM decoding包含FC (GEMV)和attention (GEMV)两类kernel。Baseline使用标准decoding算法，无量化/剪枝/稀疏优化。
  - **系统框架/Serving调度**：采用static batching或mixed continuous batching。FC kernel始终在6×A100 GPU上执行（312 TFLOPS FP16），attention kernel始终在AttAcc PIM（1P1B HBM3, 1 FPU/bank）上执行。Kernel-to-hardware mapping在design time固定，不随runtime parallelism变化调整。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：Static mapping——无论batch size=4（FC memory-bound）或batch size=64（FC compute-bound），FC kernel永远在A100 GPU上执行，attention kernel永远在PIM上执行。当batch size小时（如RLP=4, TLP=2），FC kernel为memory-bound，A100 GPU因HBM bandwidth瓶颈性能不如PIM（PIM 1P1B latency更低）；当batch size大时（如RLP=64），FC kernel变为compute-bound，PIM compute throughput不足导致性能远不如GPU。
  - **硬件架构**：单一类型PIM device（1P1B AttAcc），固定computation throughput和memory bandwidth。Power budget 116W (HBM3)。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出PAPI，通过三个关键技术解决baseline的缺陷：
  (1) **Dynamic Parallelism-Aware Scheduling**：在线监测RLP和TLP变化，用轻量级公式 AI_est = RLP×TLP 估算FC kernel arithmetic intensity，与offline确定的threshold α比较，动态决定FC kernel去GPU（compute-bound）还是FC-PIM（memory-bound）。解决了baseline static scheduling无法适应runtime parallelism变化导致kernel-to-hardware mismatch的问题（Shortcoming 1）。
  (2) **Hybrid PIM Architecture**：设计两种PIM device——FC-PIM（4P1B, high compute parallelism, 12GB）和Attn-PIM（1P2B, high memory capacity, 16GB）。FC-PIM利用data reuse降低DRAM access功耗，在power budget内提供4× compute throughput；Attn-PIM优化memory capacity以容纳KV cache长序列需求。解决了baseline单一PIM type无法满足FC和attention kernel之间4.5× arithmetic intensity差异的问题（Shortcoming 2）。
  (3) **Disaggregated Attn-PIM**：Attn-PIM通过PCIe/CXL与high-performance processor物理分离，可灵活扩展以应对长序列下KV cache memory capacity线性增长。FC-PIM通过NVLink高速互联与GPU PUs保持低延迟weight parameter access。

  全栈执行例子（同硬件同模型，batch size从64→7 runtime变化）：
  - **算法pipeline**：与baseline相同的LLM decoding算法，无accuracy loss。
  - **系统框架/Serving调度**：Host CPU运行runtime scheduler。Initial：RLP=64, TLP=4 → AI_est=256>α → FC到GPU PUs。Runtime evolution：3 iterations后3个request完成→RLP=61→AI_est=244>α→仍GPU。后续requests逐渐完成→RLP降至7→AI_est=28<α→reschedule FC到FC-PIM（4P1B, 4 FPU/bank并行GEMV）。Attention kernel始终在Attn-PIM（1P2B, 单FPU serving 2 banks）。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：Token-level scheduling——每次decoding后收集<|eos|> tokens，若count>0则更新RLP，重新估算AI并与α比较，仅在AI跨越α阈值时触发reschedule。Reschedule开销极低（乘法+比较+token counting），远小于kernel执行时间。对比baseline静态调度（无论batch大小FC始终在GPU），PAPI在memory-bound场景将FC从高功耗GPU移至低功耗高带宽FC-PIM（3.4× energy efficiency），在compute-bound场景利用GPU高FP16 throughput（1.8× speedup over PIM-only）。
  - **硬件架构**：FC-PIM 4P1B (96 banks, 12GB, 满足121mm² die area和116W power budget) vs Attn-PIM 1P2B (16GB, 满足power budget) vs GPU PUs (A100 tensor cores)。FC-PIM data reuse≥4时DRAM access energy从96.7%降至33.1%，使得在power budget内支持4 FPU/bank的并行计算。

  核心设计映射：Baseline缺陷（静态调度导致kernel-hardware mismatch + 单一PIM type无法适配FC/attention的差异化需求）→ PAPI解决方案（在线AI估算动态调度 + hybrid FC-PIM/Attn-PIM分别优化compute throughput和memory capacity + disaggregated Attn-PIM实现灵活扩展）。

## 82-COMET- Towards Practical W4A4KV4 LLMs Serving.pdf

- baseline方法是什么？
  Baseline 是现有 LLM 推理系统（TensorRT-LLM、Qserve、cuBLAS）采用 weight-only (W4A16) 或 weight-activation (W8A8) 量化方案。全栈执行例子（LLaMA-3-70B on A100-80G-SXM4, batch=16）：
  - **算法pipeline**：W4A16（如 Omniquant）：权重 INT4 量化，激活保持 FP16。运行时 GEMM 需将 INT4 权重 dequantize 到 FP16，再与 FP16 激活在 FP16 tensor core (312 TFLOPS) 计算。W8A8（如 SmoothQuant）：权重和激活均为 INT8，利用 INT8 tensor core (624 TOPS)。但两者均未利用 A100 最高吞吐的 INT4 tensor core (1248 TOPS)，且未量化 KV cache。
  - **Serving 调度**：TensorRT-LLM 使用标准 GEMM kernel (cuBLAS/cuTLASS) 执行 FP16/INT8 矩阵乘法。KV cache 以 FP16 存储，长序列下 KV cache 占显存主导（如 seq_len=128K 时 LLaMA-7B KV cache 占 72% 总显存），限制 batch size 扩展。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：cuBLAS W16A16 GEMM kernel 仅支持同精度（FP16/INT8）tensor core 计算。W4A16 kernel 需先 dequantize INT4→FP16 再计算，dequantization 开销显著（CUDA core 78 TFLOPS vs INT4 tensor core 1248 TOPS，差距 16×）。W8A8 kernel 虽用 INT8 tensor core，但数据加载成为瓶颈（memory-bound for small batch）。
  - **硬件架构**：论文未明确说明。使用标准 NVIDIA A100 GPU。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 COMET，通过 FMPQ 算法 + COMET-W4Ax kernel 实现首个实用的 W4A4KV4 LLM Serving。全栈执行例子（同硬件同模型）：
  - **算法pipeline**：FMPQ 实现 W4A4KV4 量化——block-wise (k=128) 混合精度激活量化，84%+ GEMM 以 W4A4 执行（利用 INT4 tensor core 1248 TOPS），约 16% outlier block 以 W4A8 执行（INT8 tensor core 624 TOPS）；KV cache 以 channel-wise INT4 量化（4× 压缩）。PPL 仅增加 0.05，zero-shot 精度下降 <0.75%。直接解决 baseline 的 (a) 未利用 INT4 tensor core 和 (b) KV cache 未量化问题。
  - **Serving 调度**：COMET 框架集成 W4Ax kernel + PagedAttention 4-bit KV cache 管理。W4Ax kernel 编译为独立 .so 动态库，通过 pybind 提供 Python 接口，可无缝集成 TensorRT-LLM、llama.cpp、DeepSpeed。4-bit KV cache 使支持更大 batch（如 LLaMA-3-70B 可在 A100 80GB 上 BS≥64），而 FP16 KV cache baseline 在类似配置下 OOM。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：COMET-W4Ax kernel 的核心创新：(a) SIMT-enhanced software pipeline（双缓冲 shared memory + async_copy_barrier）隐藏 dequantization/permutation 开销在计算中；(b) Weight interleaving 消除 INT4 权重加载时的 shared memory conflict，ldmatrix 指令减半；(c) Fast INT4→INT8 conversion（location switch + zero extension）将转换开销从 10 inst/value 降至 2 inst/value；(d) Fine-grained SM scheduling（tile remapping + task-stealing + one-to-many tile-SM binding）均衡 INT4/INT8 混合计算负载，使 COMET-W4Ax 达到 Oracle W4A4 kernel 的 ~96% 性能。直接解决 baseline 的 dequantization 开销和 INT4 tensor core 利用率低问题。
  - **硬件架构**：论文未明确说明。

  核心设计映射：Baseline 缺陷（浪费 INT4 tensor core + dequantization 开销大 + KV cache 显存瓶颈 + 混合精度 SM 负载不均）→ COMET 解决方案（FMPQ 实现 W4A4KV4 利用 INT4 TC + SIMT pipeline 隐藏转换开销 + 4-bit KV cache 释放显存 + fine-grained SM scheduling 均衡负载）。

- baseline方法是什么？
  Baseline 是现有 offloading-based MoE 推理方法（FlexGen、MoE-Infinity、Fiddler 等），它们采用 single-batch pipeline 或 naive multi-batch pipeline 进行推理。全栈执行例子（Mixtral-8×7B on RTX 3090, batch size=16）：
  - **算法pipeline**：FlexGen 的 zig-zag block schedule 将多 batch 计算组织为逐 batch 竖向执行，prefetch 整个下一层（包括所有 8 个 experts）以 overlap I/O 和计算。然而 MoE layer 包含多个 FFN experts，prefetch 整个 MoE layer 需要满足 `n * tc_A >= tIO_MoE`，需要很大的 n，引入 KV cache 膨胀。即使满足条件，许多 inactive experts 也被加载，造成不必要的 I/O。
  - **Serving 调度**：FlexGen 通过 solve linear programming 在计算图中分配 tensor placement，但 MoE 特有的 sparse activation 未被考虑——inactive experts 也占用 transfer 带宽。MoE-Infinity 通过 activation-aware expert prefetching + caching 减少不必要的 expert transfer，但仍是 single-batch 模式，无法消除 inter-layer bubbles。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：使用 PyTorch 默认 CUDA kernel，无特殊 kernel 优化。四个 CUDA stream 仅用于基本的异步 I/O 和计算 overlap。
  - **硬件架构**：论文未明确说明。使用标准的 NVIDIA GPU + CPU + Disk 三级异构内存层次。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 Klotski，通过 expert-aware multi-batch pipeline paradigm 同时消除 inter-layer 和 intra-layer bubbles。全栈执行例子（同硬件同模型）：
  - **算法pipeline**：将 MoE layer 的计算从 batch 维度切换为 expert 维度。Attention layer 执行多 batch 竖向计算时，仅 prefetch gate + K hot experts（基于 hot-expert 现象：少数 experts 覆盖多数 tokens），满足 `n * tc_A >= tIO_G + K * tIO_E`，比 baseline 的全层 prefetch 所需 n 更小。然后 gate 确定各 token 的 expert assignment 后，按 expert 维度组织计算：同一 expert 的所有 token（跨 batch）一起计算。这直接解决了 baseline 中 inactive experts 被无效 prefetch 的问题。
  - **Serving 调度**：Correlation-aware expert prefetcher 基于 offline 预跑建立的 expert correlation table 动态确定 hot experts；constraint-sensitive I/O-compute planner 测量硬件环境后通过不等式组自动求解最优 n；adaptive tensor placement 支持 layer 粒度的三层存储分配，充分利用剩余 GPU memory 缓存高频 experts。对比 baseline 的 naive prefetch-all 策略，Klotski 的 expert-aware prefetch 仅在 gate + hot experts 上产生 I/O 开销，cold experts 按需 transfer 且其 I/O 被 hot expert 的计算时间覆盖。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：与 baseline 相同，使用 PyTorch 默认 CUDA kernel，但通过四个 CUDA stream（weight prefetch / expert transfer / KV cache prefetch / KV cache store）实现更精细的异步操作编排，使得 expert transfer 可以在 attention 计算和 hot expert 计算的背景中完成。
  - **硬件架构**：论文未明确说明。与 baseline 使用相同的 GPU-CPU-Disk 异构存储层次。

  核心设计映射：Baseline 缺陷（inter-layer bubbles 来自全 MoE layer I/O > attention 计算时间 → intra-layer bubbles 来自多个 experts 的 I/O 远大于每个 expert 的计算时间）→ Klotski 解决方案（multi-batch 扩大 attention 计算时间覆盖 hot expert I/O + expert-wise 计算顺序重组让 hot expert 计算覆盖 cold expert I/O）。

## 7-A_Mess_of_Memory_System_Benchmarking_Simulation_and_Application_Profiling.pdf

- baseline方法是什么？
  Baseline是当前内存系统benchmarking、simulation和application profiling使用各自独立、解耦的工具：
  **(a) 内存benchmark**：测量单一指标（最大可持续带宽如STREAM，或unloaded延迟如LMbench/Google multichase）。Intel MLC能测loaded latency但仅提供稀疏数据点，不覆盖完整读写比范围。缺乏统一的bandwidth-latency视角。
  **(b) 内存simulator**：依赖详细DRAM时序模型（DRAMsim3, Ramulator, Ramulator2）或简化模型（gem5 Simple memory/Internal DDR, ZSim fixed-latency/M/D/1 queue）。这些模拟器虽经过JEDEC timing verification，但与实际系统性能严重不符：模拟延迟低至4-25 ns（实际≥85 ns），带宽超过理论最大1.8-2.7×，STREAM/LMbench模拟误差达数十个百分点。根源：(1) CPU simulator到外部memory simulator的接口误差；(2) DRAMsim3/Ramulator的row-buffer利用率模型不准确。
  **(c) 应用profiling**：基于内存延迟阈值（PerfMemPlus）、Roofline模型位置、或CPI stack中memory相关部分判断memory-bound程度，未利用完整bandwidth-latency关系。
  **(d) 新兴内存技术支持滞后**：DDR5量产后3年公版simulator才支持（Ramulator2, 2023）；CXL memory expander在论文发表时没有公开模拟器支持。

  Baseline全栈执行例子（以gem5+Ramulator2模拟Graviton 3 DDR5为例）：
  - **算法pipeline**: 论文未明确说明（非ML领域）。
  - **系统框架**: gem5 cycle-accurate full-system simulator，配置64核Neoverse N1、8×DDR5-4800。
  - **编译框架**: 论文未明确说明。
  - **kernel调度**: Ramulator2接收gem5通过外部接口发出的trace（含内存操作地址和non-memory指令数间隔）→ 执行逐cycle的DDR5状态机模拟：precharge→activate→read/write→refresh cycles→返回延迟。Ramulator2经Verilog model验证无JEDEC时序违规，但模拟延迟仅4-25 ns（实际≥129 ns），最大带宽仅126 GB/s（实际可达292 GB/s）。
  - **硬件架构**: Ramulator2的DDR5时序模型未能正确建模row-buffer利用率与bandwidth的关系。模拟中row-buffer hit rate异常高，无法复现实际系统在负载增加时row-buffer hit rate下降→miss增加→带宽效率降低的动态过程。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出**Mess (Memory stress) framework**，将内存系统benchmarking、simulation和application profiling统一在一套bandwidth-latency曲线模型上：
  **(a) Mess benchmark** (解决baseline(a)缺陷)：用汇编实现pointer-chase（测延迟）+ multi-core traffic generator（测带宽），生成完整的bandwidth-latency曲线族（数百个测量点，覆盖全带宽范围和多种读写比）。部署于x86/ARM/Power/RISC-V CPU和NVIDIA GPU，刻画DDR4/DDR5/HBM2/HBM2E内存系统。发现了前人未报道的行为：读写流量对性能的影响差异、bandwidth-waveform现象（带宽随请求率增加反而下降，因row-buffer miss率激增）。
  **(b) Mess simulator** (解决baseline(b)(d)缺陷)：放弃详细DRAM时序模拟，将bandwidth-latency曲线直接作为内存性能模型。通过PI feedback controller（比例-积分控制器）在每个simulation window末比较模拟带宽cpuBW与曲线预测messBW，自动调整应用在曲线上的位置以消除不一致。模拟误差仅0.4%-6%（vs baseline的数十个百分点），速度比DRAMsim3/Ramulator快13-15×。因为只需换用不同bandwidth-latency曲线，Mess成为首个支持CXL memory expander的模拟器（曲线由Micron SystemC model提供）。
  **(c) Mess application profiling** (解决baseline(c)缺陷)：集成到Extrae/Paraver HPC性能分析工具套件中，将应用执行的每10ms片段定位到bandwidth-latency曲线上，计算memory stress score（0=unloaded, 1=fully saturated），并与timeline和source code关联。

  论文方法全栈执行例子（以ZSim+Mess模拟Intel Skylake DDR4为例）：
  - **算法pipeline**: 论文未明确说明（非ML领域）。
  - **系统框架**: ZSim event-based simulator，配置24核Intel Skylake、6×DDR4-2666。
  - **编译框架**: 论文未明确说明。
  - **kernel调度**: Mess benchmark汇编kernel在Intel Skylake上执行→ uncore硬件计数器测得bandwidth-latency曲线族（输入到Mess simulator）。Mess simulator不执行DRAM状态机，而是：CPU simulator产生内存操作→每1000次操作采样带宽cpuBW→PI controller: `messBW_{new} = messBW + convFactor × (cpuBW - messBW)`→从曲线读取对应Latency→减去CPU simulator已建模的延迟部分→反馈Memory延迟给CPU simulator。
  - **硬件架构**: Mess simulator的核心创新在于将内存模拟从"精确DRAM时序建模"转为"bandwidth-latency一致性维护"。传统模拟器试图独立预测延迟，Mess仅确保模拟的(bandwidth, latency)点始终落在实际系统曲线上。这使其：(1)极其准确（1.3%平均误差 vs Ramulator 108%）；(2)速度快（仅比fixed-latency模型慢26%）；(3)支持新内存技术无需重新开发时序模型（只需要曲线的.csv文件）。

## 79-Anda_Unlocking_Efficient_LLM_Inference_with_a_Variable-Length_Grouped_Activation_Data_Format.pdf

- baseline方法是什么？
  Baseline是weight-only quantized LLM (W4A16)在现有平台上的FP-INT GeMM执行方式，主要有三种：
  **(a) GPU FP-FP方案**：当前GPU平台（如NVIDIA Tensor Core）缺少专用FP-INT计算单元，需要将INT4权重解量化转换为FP16后，由FP16 Tensor Core执行FP16×FP16矩阵乘法。痛点：(1)反复的格式转换开销（INT4→FP16, FP32→FP16）；(2)FP计算开销高（exponent对齐、normalization）。
  **(b) GPU FP-INT方案**：增强Tensor Core加入专用FP-INT计算单元，消除INT4→FP16转换。痛点：FP-INT单元仍需exponent对齐和normalization，硬件实现复杂且计算开销仍然较大。
  **(c) FIGNA方案（HPCA 2024）**：激活以FP16存储在内存中，计算前动态转换为BFP格式（14-bit固定尾数），以INT算术执行FP-INT GeMM，结果再转回FP16。痛点：(1)FP16激活需反复从内存访问（无压缩存储节省）；(2)每次计算时FP16→BFP的动态转换引入额外开销；(3)所有模型/模块统一使用14-bit尾数，无法利用不同模块的精度敏感度差异进行针对性压缩。

  Baseline全栈执行例子（以GPU FP-FP方案, LLaMA-7B W4A16, batch_size=1）：
  - **算法pipeline**: FP16激活 + INT4权重执行FP-INT GeMM。激活保持全精度FP16不变（无量化/压缩）。在GPU上所有FP-INT操作转为FP-FP操作执行。
  - **系统框架/Serving**: HuggingFace Transformers + PyTorch，GPU上直接推理。
  - **编译框架**: 论文未明确说明。
  - **kernel调度**: GPU Tensor Core执行FP16×FP16 matmul。W4A16 kernel先将INT4 weight dequantize为FP16（乘以scale + bias），再调用cuBLAS FP16 GEMM。每步需读取完整FP16激活(2 bytes/element)和dequantized FP16 weight。
  - **硬件架构**: NVIDIA GPU (如A100/H100)，Tensor Core FP16 units。片上共享内存+寄存器文件+HBM2/3 DRAM。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Anda通过三个层面的协同设计解决baseline缺陷：
  
  **(A) 算法层 — 变长分组Anda数据格式 + 自适应精度搜索**：用BFP格式替换FP16激活，64个元素共享5-bit指数，尾数长度按模块敏感度动态分配（1-16 bit连续可调）。搜索算法仅优化4类关键张量（A_qkv/A_o/A_u/A_d），32次迭代内找到满足精度约束的最小BOPs组合。对比FIGNA的固定14-bit尾数：Anda根据模块敏感度差异（如A_qkv高敏感保留7-bit，A_d低敏感压缩到4-bit），BOPs reduction达1.46-2.69×，而FIGNA仅1.23×。
  
  **(B) 存储层 — Bit-plane数据布局**：将Anda变长尾数按bit-plane view transpose存储，同权重的bit打包为64-bit words。对比FIGNA的FP16 element-based存储：Anda激活存储量从16-bit/element压缩到平均4-8-bit/element（含符号+指数），SRAM activation buffer容量需求大幅降低。变长尾数仅改变地址深度不影响带宽利用率。
  
  **(C) 硬件层 — Bit-serial APU + Runtime BPC**：Anda PE采用bit-serial计算，尾数长度减少直接减少执行cycle数（M-bit尾数仅需M个compute cycle）；BPC在线将FP16输出压缩为Anda格式写回。对比FIGNA的bit-parallel设计：Anda在变精度场景下能效更高（skip redundant bit calculations），且直接存储Anda格式消除FP16↔BFP反复转换开销。BPC的bit-serial aligner仅需1个比较器+1个移位器（vs FIGNA的bit-parallel需多个移位器），面积更小。

  Anda全栈执行例子（LLaMA-13B, 最优精度组合 [7,5,5,4], GS=64, 1%损失, batch_size=1）：
  - **算法pipeline**: 编译时，复用weight-only PTQ校准数据（128 random sequences × 2048 tokens from WikiText-2），自适应精度搜索算法在32次迭代内找到[M_qkv=7, M_o=5, M_u=5, M_d=4]，满足<1% PPL损失。推理时，每个Transformer block的FP16激活（A_qkv/A_o/A_u/A_d）在线转换为对应尾数长度的Anda格式。
  - **系统框架**: HuggingFace模型→Anda精度指令注入（per-module mantissa length 4-tuple）→Anda accelerator执行推理。
  - **编译框架**: 论文未明确说明（离线搜索替代编译优化）。
  - **kernel调度**: 对A_qkv模块(M=7)，Anda PE从Activation Buffer按bit-plane加载7个bit-plane尾数(每plane 64b)→与INT4权重执行bit-serial dot-product（7 cycles per group）→FP accumulator跨64组累加→BPC压缩输出为Anda格式(A_d对应M=4)写回buffer。对比baseline FP16 GEMM需1个FP16 MAC cycle但每个MAC energy远高于bit-serial INT AND+ADD。Anda通过减少尾数宽度（7→5→4 bit across modules）直接proportional地减少compute cycles和memory access energy。
  - **硬件架构**: Anda accelerator (16nm, 285MHz, 0.8V, 2.17mm², 81.18mW)。16×16 APU阵列执行FP-INT GeMM，output stationary dataflow。Activation Buffer 1.125MB (1MB mant+0.125MB exp)+Weight Buffer 1MB，共2.125MB片上SRAM。HBM2外部内存(3.9 pJ/bit, 256 GB/s)。对比GPU: 2.14× speedup (0.1% loss) / 2.49× (1% loss), 3.47× area efficiency / 4.03× energy efficiency vs FP-FP baseline。

- baseline方法是什么？
  Baseline是现有单GPU-CPU系统上的LLM推理KV缓存方案，主要包括FlexGen和vLLM：
  - **FlexGen [31]**: 静态head-level KV tensor offload调度。通过offline线性规划求解固定offload比例，KV tensors沿head维度划分并静态分配在GPU和CPU之间，整个推理过程中offload策略不变。痛点：(1) 随着序列增长KV tensors线性增大，CPU-GPU频繁offload/reload导致PCIe瓶颈，执行时间放大多达5×；(2) 无稀疏attention——所有token的KV均被缓存和访问，即使大部分注意力权重极低；(3) 静态调度无法适应运行时序列长度变化的动态内存容量和访存特征。
  - **vLLM [21]**: block-level paged memory management for KV cache。将固定token数（block）的KV tensors存储在非连续paged内存中，支持CPU-GPU swapping以避免内存碎片。痛点：仅优化内存管理层面，不减少KV tensor总量；无稀疏attention优化仍处理所有token。
  Baseline全栈执行例子（以FlexGen + OPT-6.7B on 1×V100 32GB, batch_size=8, input=128, output=512为例）：
  - **算法pipeline**: Dense attention——每步对所有prior tokens执行QK^T [1×d]×[d×(s+n)]，产生(s+n)×(s+n) attention weight矩阵（仅新增一行），复杂度O(n)。无token重要性区分。
  - **系统框架层**: FlexGen的offline linear programming按head维度静态分配KV offload比例（如50% heads的KV offload到CPU）。整个推理过程固定此比例，无论序列长度如何变化。CPU-GPU I/O开销随序列增长线性增加。
  - **编译框架层**: 论文未明确说明（使用HuggingFace Transformers原生执行）。
  - **kernel调度层**: Dense QK^T matmul + softmax + AW×V matmul。所有(s+n)个token的KV均参与计算。GPU kernel无稀疏gather/reduce操作。FlexGen按head维度swap KV tensor block。
  - **硬件架构层**: 1×NVIDIA V100 32GB + CPU 128GB DRAM, PCIe 20 GB/s。当KV总量>32GB→OOM或offload→I/O瓶颈。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ALISA通过**算法-系统协同设计**解决三个核心baseline缺陷：
  **(A) Dense attention内存占用大** → SWA算法识别重要token，实现80% KV稀疏化（仅保留20%最重要token），内存占用降低5×。
  **(B) 静态offload调度效率低** → 三阶段token-level动态调度（Phase I GPU-only / Phase II GPU-CPU / Phase III Recomputation-Caching），phase switch由offline优化问题求解最优{α,β,p1,p2}。
  **(C) 全精度KV tensor I/O开销** → INT8 channel-wise KV compression（量化CPU存储的KV，dequantize用于GPU计算），几乎无损精度下大小减半。

  ALISA全栈执行例子（同一OPT-6.7B on 1×V100 32GB, 80% KV sparsity, batch_size=8）：
  - **算法pipeline→SWA**: 每步注意力计算前，对最近k步attention weights求和得到每个prior token的重要性分数S→选择top-k全局动态token + 最近k个局部静态token（共2k = 0.2×n个token）→gather稀疏KV→dense matmul仅对2k个token。对比baseline对全部n个token计算QK^T，SWA的计算量与n无关、仅与r相关（k=⌊nr/2⌋），内存占用∝k而非∝n。
  - **系统框架→三阶段调度**: (1) Phase I (step ≤ p1): KV tensors ≤ GPU capacity，全程GPU缓存无CPU开销，类似baseline但KV已稀疏化→更大序列下仍保持Phase I；(2) Phase II (p1 < step ≤ p2): KV超出GPU→CPU offload全局动态token的旧KV（局部静态token留GPU减少不可预测的全局token CPU访问），与baseline相比减少了offload量（仅稀疏KV而非全量KV）；(3) Phase III (step > p2): 序列极长时CPU reload开销 > GPU recompute开销→删除CPU中最旧KV、GPU重计算。动态phase switch确保每阶段最优执行策略，而baseline（FlexGen）始终静态offload固定比例无recompute机制→在长序列下被I/O瓶颈拖垮。
  - **编译框架层**: 论文未明确说明。
  - **kernel调度层→SWA Kernel**: Local Attention Sum (vector reduce on最近k步) → top-k argmax → token-level gather (不规则→dense转换) → dense matmul (QK_s^T/softmax/AW·V_s) → 新KV store + INT8 quantize。对比baseline的纯dense matmul，增加gather/reduce开销但大幅减少matmul计算量（矩阵维度从n降至2k）和HBM访问量（KV读取量减少5×）。Local attention sum虽引入额外kernel但数据量小（仅最近k步的AW向量）。
  - **硬件架构层**: 同baseline硬件。ALISA的token粒度offload相比baseline的head粒度offload，更能适配SWA的稀疏token选择模式——局部静态token持续保留GPU避免不可预测的全局动态token CPU I/O。Phase III的recompute利用GPU tensor core高吞吐（FP16 matmul远快于PCIe 20 GB/s），以tiny compute换取大量I/O消除。KV compression (INT8)使CPU存储和PCIe传输量再减半。

## 70-ACES- Accelerating Sparse Matrix Multiplication with Adaptive Execution Flow and Concurrency-Aware Cache Optimizations.pdf

- baseline方法是什么？
  Baseline是三种现有state-of-the-art SpMM加速器，各采用固定execution flow：
  - **SIGMA [45]**：InP执行流。通过bitmap格式提升index intersection效率，灵活PE互联。痛点：InP的B矩阵列复用差（每列被每行A refetch），index intersection在此格式下仍有大量冗余fetch。
  - **SpArch [61]**：OutP执行流。aggressive condensing压缩A减少partial matrix数量 + high-radix merger流水线化merge过程。痛点：aggressive condensing破坏input reuse导致B row被频繁重复fetch + cache thrashing，partial matrix transfer内存流量大。
  - **SPADA [32]**：ROW + Window-based Adaptive (WA)执行流。将A按window分组，multipliers在window内并行执行scalar-vector乘法。痛点：WA引入multipliers的collective dependency——所有multiplier必须等当前window task完成才能进入下一window → PE利用率受限于最慢task。Cache仅考虑locality（LRU），不考虑concurrency。
  Baseline全栈执行例子（以SIGMA InP为例：A(M×K) × B(K×N) → C(M×N)，16 PE）：
  - **算法pipeline**：SpMM = Σ_k A[i,k] × B[k,j] for each (i,j)，InP对整个row_i of A和column_j of B做内积。
  - **Serving调度**：N/A（专用加速器，非Serving框架）。
  - **编译框架**：N/A（专用加速器，非编译框架）。
  - **kernel调度**：固定InP执行流——每个PE取A的一个row和B的一个column，通过bitmap加速index intersection找到matching non-zero index pairs，计算partial dot product。B column被每个A row重复从DRAM fetch（无跨PE B复用）。所有PE并行计算不同C elements，无sync问题但cache miss阻塞整个PE（blocking cache）。
  - **硬件架构**：Flex-DPE（宽16，可配置PE互联），SRAM buffer 1.5MB，1GHz，64-bit double-precision。TSMC 28nm。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ACES通过三大创新解决baseline痛点：
  **(1) Adaptive Execution Flow**：将固定执行流改为运行时动态选择三种condensing degree。对比SIGMA的InP（B复用差），ACES在none condensing时column-by-column traverse A实现高B复用（类似OutP），但会产生sync冲突；aggressive condensing消灭sync但B复用差（类似SpArch的问题）；moderate condensing折中。Adaptive机制通过band分区（CSR offsets row length差值>10分新band）+ sampling（每个大band三次32-row trial选择最快condensing）自动选择。对比SPADA的WA（collective dependency），ACES每个MPE独立执行一个scalar-vector乘法、不依赖同window其他MPE完成即可开始下一任务 → 消除collective dependency。
  **(2) Concurrency-Aware Cache (PureFiber)**：对比所有baseline的locality-only cache（LRU），PureFiber驱逐时评估RD（reuse distance=locality）+ FD（fiber density=concurrency indicator），优先保留低FD高locality的cache lines → 最大化pure fiber（concurrent access全hit无stall）。配合NB Buffer实现non-blocking cache（miss不阻塞后续访问），对比baseline的blocking cache（一次miss阻塞整个PE chain）。
  **(3) MPE-APE一对一配对 + Schedulers**：Contrast SpArch mixer（集中式merge bottleneck）和SPADA WA collective dependency。ACES每个MPE配一个专用APE做immediate merging（MPE产出partial fiber立即可被对应APE merge），Synchronization Scheduler从SQ中选择无sync冲突的fiber分配、支持skip等待，Huffman-tree Merging Scheduler最小化final merge开销。
  论文方法全栈执行例子（ACES, A(M×K) × B(K×N) → C(M×N)，16 MPE + 16 APE）：
  - **算法pipeline**：SpMM = Σ_k A[i,k] × B[k,j]，分解为scalar-vector乘法和row-granularity partial fiber merge，Immediate merging + final merging两阶段。
  - **Serving调度**：N/A（专用加速器，非Serving框架）。
  - **编译框架**：N/A（专用加速器，非编译框架）。
  - **kernel调度**：Condensing Adapter运行时将A按row length分区为bands → 大band sampling选最优condensing degree → A Fetcher按condensed column顺序加载A non-zero elements到Global Buffer → B Fetcher根据A的original column index从DRAM加载对应B row fiber到Global Cache → MPE独立取(A_element, B_fiber)并行scalar-vector乘法，无collective dependency → SQ缓冲partial fiber → Synchronization Scheduler选择无冲突fiber分给对应APE → APE从Global Cache读取同row existing partial fiber做immediate merging（双指针walk compare-merge），不匹配则写入cache → Global Cache满时PureFiber驱逐max(RD+FD) line → Cache miss由NB Buffer非阻塞处理（登记+发DRAM请求+不阻塞后续访问+返回后notify）→ 全部MPE完成后，Merging Scheduler用Huffman tree调度final merging（优先合并最小weight fibers）→ 输出C矩阵。对比baseline：消除了InP的index intersection overhead（ACES仅取matching pairs），消除了OutP的partial matrix transfer overhead（ACES row-granularity immediate merge减少写回DRAM），消除了WA的collective dependency（ACES MPE独立工作），消除了baseline的blocking cache stall（PureFiber + NB Buffer非阻塞）。
  - **硬件架构**：ACES 16 MPE + 16 APE，1MB 16-bank set-associative Global Cache + 0.5KB NB Buffer，16×16 swizzle-switch crossbar，TSMC 28nm，3.52mm², 2.83W。对比SPADA (6.32mm²)和SpArch (13.96mm²)面积更小，但speedup分别达2.1×和8.9×。PE utilization 95.1% (avg) vs SIGMA 54.8%/SpArch 80.0%/SPADA 86.9%。

- baseline方法是什么？
  Baseline是现有分布式训练系统（Megatron-LM [50]、DeepSpeed [62]）处理MT MM模型的naïve方法：在时间维度上将MT MM模型的各sub-model解耦（decouple），每个sub-model顺序占用全部GPU cluster执行，模型间顺序串行。由于这些系统缺乏对MT MM模型workload heterogeneity和execution dependency的理解，它们将异构的多模态多任务workload作为同构模型处理，导致严重的GPU利用率波动和资源浪费。Baseline全栈执行例子（以Multitask-CLIP 4任务·16 GPU为例）：
  - **算法层**：Multitask-CLIP基于ImageBind [22]结构——6种modality的独立encoder（Vision/Audio/Text/Motion/Thermal/Depth）+ contrastive loss跨模态模块。4个task分别激活不同modality组合（如Task1: Vision+Text，Task2: Audio+Text等），各task的sub-model含独立的encoder激活路径和共享的跨模态模块。
  - **系统框架层**：DeepSpeed/Megatron-LM将4个task的sub-model按时间维度串行执行——每次iteration，Task1 sub-model占用全部16 GPU完成forward/backward → Task2 sub-model占全部16 GPU → Task3 → Task4。shared component（如Text encoder）在每个task中都被全集群并行执行，无法跨task复用。所有parallel configuration（DP/TP degree）基于单一同构模型假设设定。
  - **编译框架层**：论文未明确说明（使用PyTorch原生execution，无专门的MT MM编译优化）。
  - **kernel调度层**：DeepSpeed/Megatron-LM对每个sub-model内的所有operator采用统一的parallel strategy（如全16 GPU的DP/TP配置）。轻量operator（如Audio encoder单层，input [8,229,768]）与重量operator（LM层，input [8,512,1024]）在相同设备数上执行——轻量operator计算kernel严重underutilized，TFLOPS利用率低。Inter-task间无调度优化——task串行导致资源在task切换时短暂空闲。
  - **硬件架构层**：16×NVIDIA A800 80GB GPU，8 GPU/node内NVLink互联，node间400Gbps InfiniBand。Sub-model内TP通信走NVLink，PP跨node走IB。由于weight operator在全集群上低效执行，GPU利用率低且scalability差（16→32 GPU时speedup仅1.21×）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法Spindle通过wavefront scheduling和联合优化框架解决baseline的三个核心缺陷：(A) workload heterogeneity被忽略导致资源分配次优、(B) execution dependency被串行化解导致大量同步等待、(C) 缺乏operator级细粒度调度导致负载不均。Spindle全栈执行例子（同一Multitask-CLIP 4任务·16 GPU）：
  - **算法层**：同baseline，MT MM训练算法不变。但Spindle在graph contraction阶段将原始operator按类型+输入尺寸融合为MetaOp（如Audio Op、Text Op、LM Op、Vision Op等7种MetaOp），每种MetaOp代表独特workload类型。
  - **系统框架层→Wavefront Scheduling联合优化**：
    - **缺陷A解决（workload heterogeneity）**: Scalability Estimator（§3.2）用分段α-β建模为每个MetaOp生成独立的scaling curves T_m(n)——轻量Audio Op在n=4时已接近最优（scalability 2×而非4×），重量LM Op在n=16时仍接近线性加速。Resource Allocator（§3.3）根据scaling curves求解MPSP最优分配——如Audio Op分配4 GPU、LM Op分配8 GPU，避免同构分配造成的资源浪费。对于lightweight operator，DeepSpeed需用全16 GPU并行导致kernel underutilized甚至idle，而Spindle仅用4 GPU保持高利用率。
    - **缺陷B解决（execution dependency）**: Graph Contraction将计算图收缩为MetaGraph并解耦为MetaLevels（BFS分层，level内MetaOp无依赖）。每个MetaLevel独立求解MPSP得到分配方案。Wavefront Scheduler（§3.4）将MetaOp切分为waves——wave内多个MetaOp的不同slice并发执行在不同device group上，wave间通过transmission operators（NCCL send/recv）连接数据流。对比baseline的task级串行执行串行依赖导致的长时间等待，Spindle在wave粒度交错执行不同task的MetaOp，大幅减少依赖等待。
    - **缺陷C解决（load imbalance）**: Wavefront Scheduler的time span alignment机制——每wave以最短ASL-tuple的执行时间为wave时长，截断其他tuple，确保wave内各device group同时完成、同时进入下一wave（图5b中wave1-wave6的紧密编排）。Resource extension（§3.4 ○2）对剩余执行时间大的MetaOp动态扩展资源（如wave4中MetaOp4从1 device扩至2 devices），进一步加速straggler。Device Placement（§3.5）的memory balance + intra-island优先策略避免OOM和额外通信开销。
  - **编译框架层**：论文未明确说明（Spindle直接操作PyTorch computation graph，通过PyTorch FX Tracer自动分解model，不经过IR compiler层）。
  - **kernel调度层→MetaOp-Level Profiling + Piecewise α-β**：
    - **分段α-β建模**: 传统单模型自动并行系统（Alpa [87]、Galvatron [44]）假设同构层的统一α-β函数，在MT MM场景下因不同MetaOp调用不同compute kernel且per-device workload不同而失效。Spindle对每个MetaOp独立profiling多个(n_i, T_m(n_i))散点，分段拟合α-β曲线，更准确捕捉异构workload的scalability特征。
    - **MPSP求解+Bi-point Discretization**: Resource Allocator将per-MetaLevel资源分配形式化为MPSP（Theorem 1: 连续最优解下所有MetaOp同时开始同时结束）。Bisection search解Σ T_m^{-1}(C*/L_m) = N求得C*和n*_m。Bi-point discretization用两个离散ASL-tuple线性表示连续最优分配（满足公式10a/10b）——如MetaOp 2 n*=1.5离散为n=2, n=1, l=8.4, l=3.6（图5a），再round l为整数。与理论最优C*的偏差<7%（图11）。
    - **Wavefront Greedy算法（Alg 1）**: 每wave贪心提案→扩展→对齐→提交，每个MetaOp最多产生2个ASL-tuple（bi-point discretization），因此最多2×|V_M|个waves。与传统pipeline parallelism（固定stage划分）不同，Spindle的wave边界动态适应MetaOp workload比例。
  - **硬件架构层**：同baseline硬件（NVIDIA A800集群）。Spindle的inter-wave通信开销经由device placement优化后降至<6%（vs sequential placement的27%），intra-island placement充分利用NVLink高带宽减少跨node通信。Wave-by-wave forward/backward交错执行和group-wise parameter sync使GPU利用率在整个iteration内始终保持高位（图9a vs DeepSpeed的剧烈波动）。

## 64-POD-Attention- Unlocking Full Prefill-Decode Overlap for Faster LLM Inference.pdf

- baseline方法是什么？
  Baseline 是现有 LLM serving 系统（Sarathi-Serve [23]、vLLM [41]）中，对 hybrid batch 内的 prefill 和 decode attention **独立使用专用 kernel 串行执行** 的做法。FlashAttention [30] 和 FlashInfer [60] 分别提供 prefill-optimized kernel（compute-optimized，大 tile size，高 compute utilization）和 decode-optimized kernel（memory-bandwidth-optimized，小 tile 但仍有 ~70% compute utilization 的冗余计算）。在 hybrid batch 中，prefill attention 和 decode attention 在同一 iteration 内背靠背执行——先 prefill attention（高 compute 需求，memory BW 空闲），后 decode attention（高 memory BW 需求，compute 空闲），导致 GPU 资源利用呈周期性 "高需求→低利用" 切换。Baseline 全栈执行例子（以 Llama-3-8B on 2×A100，hybrid batch: chunk_size=1K + decode BS=64，CL=16K 为例）：
  - **算法层**：LLM 自回归推理——prefill 并行处理 prompt chunk（1K tokens）生成 KV cache + first output token，decode 对每个 request 逐 token 生成并自回归拼接。Attention 计算 QK^T×V + softmax，对 hybrid batch 拆分两次 kernel 调用。
  - **系统框架层**：Sarathi-Serve chunked-prefills + hybrid batching——将 prompt 拆为 chunk，每 iteration 调度 1 prefill chunk + N decodes 共同处理。Linear 操作已融合（model weights 一次 HBM load 服务 prefill+decode），但 attention 未融合——调用 FA prefill kernel 处理 chunk，再调用 FA decode kernel 处理 decodes。
  - **编译框架层**：论文未明确说明（使用 PyTorch + CUDA 直接调用 FA kernel，无专用编译优化）。
  - **kernel调度层→独立 prefill/decode kernel 串行执行**：
    - FA prefill kernel：QSL tile 128，高 compute utilization 可达 80%，但 memory BW utilization <5%（大部分 HBM bandwidth 空闲）。Split along K/V dim（FlashDecoding style）增加 parallelism，额外 memory reads 被并行掩盖。
    - FA decode kernel：QSL tile 64–128（即使 decode QSL=1），零填充导致 60–70% compute utilization 的冗余 tensor core 计算（实际 decode 只需 ~10%）。Memory BW utilization 高。K/V split 增加 HBM reads。
    - 串行执行导致：prefill kernel 执行时 memory BW 几乎完全 idle → decode kernel 执行时 tensor cores 几乎完全 idle → GPU 利用率在两阶段均不饱和。
  - **硬件架构层**：NVIDIA A100 80GB，108 SMs，每个 SM 含 tensor cores + L1/shared memory + execution units。Prefill/decode kernel 各自通过 CUDA CTA scheduler 分配至 SMs——但两 kernel 在不同时间执行，SM 内无不同操作并发的可能。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 POD-Attention 通过 **CTA-parallel 融合 + SM-aware CTA scheduling** 将 prefill 和 decode attention 在单个 kernel 内同一 GPU SM 上并发执行，解决 baseline 的三个核心缺陷：(A) prefill/decode 独立 kernel 导致 GPU compute 和 memory BW 交替空闲、(B) warp-parallel/intra-thread 融合面临 straggler 和 barrier 限制、(C) 缺乏保证 SM 级操作 co-location 的机制。POD-Attention 全栈执行例子（同一 hybrid batch 配置）：
  - **算法层**：同 baseline——LLM attention 算法不变。POD-Attention 的关键是不改变 attention 数值结果，仅优化执行方式。
  - **系统框架层→Sarathi+POD**：集成到 Sarathi-Serve，替换默认 attention backend。POD-Attention 在每 iteration 一次性执行 fused prefill+decode attention，替代两次独立 kernel 调用。Linear 操作继续使用 hybrid batching fusion，实现全操作融合（linear + attention）的 hybrid batch execution。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层→CTA-Parallel Fusion + SM-Aware CTA Scheduling**：
    - **缺陷A解决（compute + memory BW 并发利用）**：在单个 kernel 内同一 SM 上同时运行 prefill CTA（compute-heavy）和 decode CTA（memory-heavy），利用 GPU warp scheduler 的 cycle-level context switch——当 prefill CTA 等待 tensor core 结果时执行 decode 的 memory load，当 decode CTA 等待 HBM 时执行 prefill 的 compute。对于 compute-heavy batch（prefill 占主导），使用 2 CTA/SM（每 CTA 更多 shared memory → 更大 prefill tile → 更高 compute efficiency）；对于 balanced/memory-heavy batch，使用 4 CTA/SM（允许不同比例 CTA 配比，如 1 prefill + 3 decode）。Prefill compute utilization 保持高，decode memory BW utilization 同时保持高——达到 "both compute and memory utilized simultaneously"（Figure 1）。
    - **缺陷B解决（避免 straggler 和 barrier 限制）**：
      - Contrast warp-parallel fusion（FA_HFuse）的 straggler 问题：POD-Attention 在 CTA 粒度划分操作——每个 CTA 独立决定执行 prefill 或 decode，不同 CTA 可独立开始和完成，不会被其他 CTA 的慢 warp 阻塞。即使某个 prefill CTA 计算量大，同 SM 内的 decode CTA 不受影响。
      - Contrast intra-thread fusion 的 barrier 问题：POD-Attention 中 prefill 和 decode 的 sync barrier 各自限定在各自的 CTA 内——不影响对方 CTA。
      - Contrast FI_Batched（用 prefill kernel 同时计算 prefill+decode）的冗余 compute：POD-Attention 将 decode tile size 从 128 降至 16（CUTLASS A100 tensor op 最小值），decode 的 compute utilization 从 ~60-70% 降至 ~10%，释放 tensor cores 给 prefill。这是针对 decode attention memory-bound 特性的关键设计——decode 不需要大 tile 来最大化 compute，降 tile 不损失 memory BW（Figure 10b 验证），但大幅减少对 prefill 的 tensor core 争抢。
    - **缺陷C解决（保证 SM 级操作 co-location）**：SM-aware CTA scheduling——每个 CTA 到达 SM 后，leader thread 通过 `asm volatile("mov.u32 %0, %smid;")` 读取 SM 硬件 ID，atomically 递增 SM-level counter 获得 ticket，按 proportional policy（根据 prefill_CTAs 和 decode_CTAs 的比例）决定操作类型。硬件 CTA scheduler 不确定地将 CTA 分配至 SM，但 SM-aware scheduling 确保无论分配结果如何，每个 SM 内 prefill 和 decode CTA 按目标比例 co-locate。Contrast streams 方法（FA_Streams）无法保证 SM 级 co-location——两个 stream 的 kernel CTA 可能被分配到不同 SM，导致 colocation 失败。
    - **其他关键优化**：
      - Virtual decode CTA：将 decode CTA 拆分为 warp 粒度的 virtual CTA（warp-level barrier 替代 CTA-level barrier），各 virtual CTA shared memory 用量降至 1/4，使 prefill 和 decode 的 shared memory 需求平衡，避免 over-allocation 给 decode。
      - Limiting prefill splits：chunked-prefill 沿 K/V 维度 splits 数限制 ≤ 填满 2 full waves——减少 prefill 的额外 memory reads 对 co-located decode 的 memory BW 争抢。Table 8 证明 limited splits 使 speedup 从 0.87× 提升至 0.75×（vs FA_Serial baseline）。
  - **硬件架构层**：同 baseline（NVIDIA A100 80GB，108 SMs）。POD-Attention 利用 SM 内已有 warp scheduler 的 cycle-level context switch 能力——当 prefill warp 因 tensor core 等待而 stalled 时，warp scheduler 自动切换执行 decode warp 的 memory load，无需额外硬件修改。这是 "free hardware" 设计——利用 GPU 已有能力实现并发，不引入新硬件。

## 62-SCAR- Scheduling Multi-Model AI Workloads on Heterogeneous Multi-Chiplet Module Accelereators.pdf

- baseline方法是什么？
  Baseline是同构数据流MCM AI加速器（Simba [64]）及其单模型调度器（NN-baton [68]）。Simba MCM中所有chiplet使用相同dataflow（NVDLA-style [52] 或 Shi-diannao-style [16]），通过chiplet间协同（standalone inference或group协作处理单层）和跨层流水线实现单模型推理加速。NN-baton在chiplet粒度上为单模型工作负载做layer-to-chiplet mapping和orchestration。Baseline全栈执行例子（以多模型场景Sc4：ResNet-50 + UNet + GoogLeNet + MiDas，6×6 MCM，NVDLA同构dataflow为例）：
  - **算法层**：各模型独立推理——ResNet-50（50层Conv+FC，image classification）、UNet（encoder-decoder Conv+skip connection，segmentation）、GoogLeNet（Inception modules，classification）、MiDas（encoder-decoder，monocular depth estimation）。每模型逐层执行。
  - **系统框架层**：Simba orchestrator按单模型方式处理多模型——模型串行或简单时分复用。无跨模型调度——各模型独立占用全部chiplet进行推理，模型间无资源共享或动态重新分配。NN-baton [68] 为单模型做优化layer-to-chiplet mapping，不考虑多模型并发。
  - **编译框架层**：MAESTRO [35] 离线分析单层在固定dataflow（NVDLA）下的mapping性能。各layer的loop ordering、tiling、spatial unrolling固定为NVDLA dataflow pattern，对所有ML operator类型（Conv/FC/Pooling/Attention）使用相同映射策略。
  - **kernel调度层**：所有chiplet使用统一的NVDLA dataflow——对compute-intensive Conv层和memory-intensive FC层无差异化处理。chiplet分配为静态——模型A的所有层绑定到预定chiplet组，模型B的层绑定到另一组，期间不重新分配。无inter-chiplet pipelining——同一模型的连续层在chiplet间以finish-then-forward方式传输数据，非producer-consumer流水线。
  - **硬件架构层**：6×6 MCM，36个相同dataflow（NVDLA-style）的chiplet。2D mesh NoP。chiplet位于MCM两侧配备off-chip DRAM接口。同构dataflow导致对某些operator类型利用率低——NVDLA对memory-intensive operator产生数据移动瓶颈，Shi-diannao对compute-intensive operator产生计算瓶颈。多模型场景下operator heterogeneity（4个模型含Conv/Inception/DepthwiseConv/FC/deconv等不同operator类型）进一步放大同构dataflow的不适配。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法SCAR通过异构数据流MCM + 多层调度解决baseline的三大缺陷：(A) 同构dataflow无法适配多模型operator heterogeneity、(B) 静态chiplet分配无法动态适应workload变化、(C) 缺少跨模型资源复用机制。SCAR全栈执行例子（同一多模型场景Sc4，Het-Sides异构MCM）：
  - **算法层**：同baseline，模型推理算法不变。但SCAR在调度时将模型按layer类型和计算特征进行分段（segmentation）以匹配最优chiplet类型。
  - **系统框架层→MCM-Reconfig + PROV + SEG三引擎调度**：
    - **缺陷A解决（operator heterogeneity）**: 异构MCM集成NVDLA-style（compute-optimized）和Shi-diannao-style（memory-efficient）两种chiplet。SCAR的SEG引擎在layer segmentation后将不同segment绑定到最适合其operator类型的dataflow chiplet：Conv-heavy segment→NVDLA chiplet以获得高计算吞吐，FC/Pooling-heavy segment→Shi-diannao chiplet以获得memory访问效率。MAESTRO离线数据库为每种chiplet dataflow类型预分析每层性能，供调度时查询。
    - **缺陷B解决（静态分配）**: MCM-Reconfig实现时间窗口划分——将端到端工作负载按periodic time windows切分（nsplits=4→5 windows），每window内PROV动态重新分配chiplet给各模型。Greedy Packing Algorithm（Algorithm 1）按first-fit策略分配层到窗口，低延迟层优先执行防止starvation。Dynamic Chiplet Regrouping——chiplet组合在窗口间动态变化，对应compiler中的graph partitioning抽象。
    - **缺陷C解决（跨模型资源复用）**: PROV按uniform distribution规则 N_i = round(E(P_i)/ΣE(P_j)×|C|) 在窗口内为各模型动态分配chiplet数，确保每个模型至少获得1个node以推进执行。Inter-Chiplet Pipelining允许不同模型的layer segments在不同chiplet间以producer-consumer方式overlap执行，增强in-package data reuse，减少off-chip traffic。
  - **编译框架层→MLIR兼容集成**：SCAR的高级调度技术（dynamic chiplet regrouping → graph partitioning；inter-chiplet pipelining → inter-graph pipelining；segment assignment → buffer management + die-to-die communication）对应标准compiler IR变换。SCAR可置于现有compiler infrastructure（如MLIR [39]）之上，将调度结果转换为硬件相关的lower representations。
  - **kernel调度层→启发式搜索降维**：
    - **调度空间降维**: 原始scheduling space O(10^56)（2模型·6×6 chiplet）。SCAR通过多层分解降维：(1) Time window划分将全局调度分解为per-window子问题；(2) PROV的uniform distribution将chiplet分配独立于具体dataflow属性；(3) SEG的三个heuristics（Product-to-Summation Reduction、Inter-Layer Pipelining优先、Segment Size Balancing）将segmentation从指数复杂度降为多项式。
    - **Segmentation Heuristics详细**: Heuristic 1将跨模型联合分割搜索分解为先vertical segmentation（per-model独立搜）再horizontal segmentation（搜pipelining机会）。Heuristic 2促进layer i和i+1在不同chiplet间pipeline（生产者chiplet完成部分计算即可发送中间数据给消费者chiplet，而非等整个layer完成）。Heuristic 3平衡各node的segment大小避免straggler bottleneck。
  - **硬件架构层→异构chiplet MCM**：MCM package集成两种dataflow chiplet（Het-Sides配置：左侧全NVDLA，右侧全Shi-diannao，NoP跨侧高带宽）。NVDLA chiplet——high compute throughput via spatial unrolling，适合Conv/Attention等compute-intensive operator。Shi-diannao chiplet——aggressive data reuse via tiling + near-sensor memory hierarchy，适合FC/Pooling等memory-intensive operator。2D mesh NoP支持chiplet间任意hop通信。性能：Het-Sides异构MCM + SCAR调度 vs Simba(NVDLA)同构MCM平均EDP降低27.6%（datacenter场景）、29.6%（AR/VR场景）。Sc4场景下Het-Sides EDP为Simba(NVD)的0.3×。vs NN-baton单模型scheduler EDP降至0.3×。

## 58-ExeGPT- Constraint-Aware Resource Scheduling for LLM Inference.pdf

- baseline方法是什么？
  Baseline是FasterTransformer (FT) 及其同类LLM推理系统（DeepSpeed Inference DSI、ORCA）。Baseline全栈执行例子（以GPT-3 175B on 16×A100, 翻译任务为例）：
  - **算法层**：Transformer decoder-only模型，autoregressive decoding：每个token生成需执行self-attention（QKV projection→attention score→context vector）和feedforward network，FP16精度。输入序列经一次encoding，输出序列逐token decoding生成。
  - **系统框架层**：NVIDIA FasterTransformer推理引擎。Pipeline parallelism PP=2（16 GPUs分为2个pipeline stage，各8 GPUs），Tensor parallelism TP=8（每stage内8 GPUs tensor parallel）。Input batch经encoding后pipeline传递至decoding stages，decoding迭代固定batch size运行至所有query完成（最长的output sequence决定总decoding iteration数）。
  - **编译框架层**：论文未明确说明（FT使用预编译的优化CUDA kernel，无运行时compilation pass）。
  - **kernel调度层**：FT decoding维持固定batch size，不early-terminate completed queries——已完成query继续在batch中执行无用计算（attention on padding/已完成token），batch size随decoding推进逐渐减小（diminishing decoding batch问题），GPU资源利用率持续下降。Encoding和decoding在同一pipeline stage的同一GPU上耦合执行。
  - **硬件架构层**：16×A100 80GB，NVLink 3.0 intra-node（每node 8 GPUs），1.6Tb InfiniBand inter-node。Pipeline stage间通过InfiniBand通信，tensor-parallel组内通过NVLink all-reduce同步。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法ExeGPT通过四种机制解决baseline的两个核心缺陷：(A) diminishing decoding batch（FT/DSI）和(B) pipeline bubble（ORCA）。ExeGPT全栈执行例子（同一GPT-3 175B on 16×A100, 翻译任务）：
  - **算法层**：同baseline，Transformer decoder-only推理不变。
  - **系统框架层→RRA + WAA Scheduling**：
    - **缺陷A解决（diminishing batch）**: RRA Scheduling周期性重新运行encoding（每ND次decoding迭代喂入新input batch），保持decoding batch size充足。WAA Scheduling完全解耦encoder和decoder到不同GPU，encoding和decoding异步执行不同batch size。两者均实现early-termination并compact KV cache。
    - **缺陷B解决（pipeline bubble）**: WAA按encoding和decoding的计算量比例分配GPU（WAA-C policy: N_E = N × CE/(CE+CD)），使encoder和decoder的pipeline stage workload平衡，消除因encoding远比decoding昂贵导致的pipeline bubble。Decoder micro-batch（将BD拆为Bm个micro-batch做pipeline overlap）进一步减少bubble。
    - **Latency constraint满足**: XScheduler以branch-and-bound搜索四个控制变量（BE/BD、Bm、TP degree/applied count、FE/ND）的最优值，利用monotonicity性质高效剪枝。XScheduler对RRA和WAA分别求最优，选取throughput较高者。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层→XSimulator + XProfiler + Dynamic Adjustment**：
    - **XProfiler**: 离线profiling单层encoder/decoder在所有(batch size, sequence length, TP degree)配置下的执行时间，供XSimulator调用。
    - **XSimulator**: 利用PE(S)和PD(S)概率分布计算每个decoding iteration的expected batch size和完成query数，构建准确execution timeline估计throughput/latency。
    - **Dynamic Adjustment**: 运行时动态调整encoder batch size使实际workload（Σinput lengths）保持在阈值内；监控decoder batch size，若偏离average workload阈值则相应增减encoder batch size。
  - **硬件架构层**：同baseline硬件。WAA的额外memory开销——decoder-only模型需在encoding和decoding GPU上各存一份模型副本（GPT-3 101B: +29% model memory），但KV cache memory因解耦后各GPU的batch更小反而减少，overall内存可接受。GPU间KV cache transfer路径: GPU HBM→CPU DRAM→target GPU HBM。

## 57-HotTiles_Accelerating_SpMM_with_Heterogeneous_Accelerator_Architectures.pdf

- baseline方法是什么？
  Baseline是homogeneous SpMM加速器架构，即所有PE（Processing Elements）完全相同，对所有稀疏矩阵区域使用相同的处理方式。也包括SPADE [24]、Sextans [58]等单类型PE加速器，以及AESPA [54]的异构子加速器方案（仅按整体matrix粒度选择子加速器类型，不利用intra-matrix heterogeneity）。Baseline全栈执行例子（以SPADE homogeneous执行coPapersCiteseer矩阵的SpMM为例）：
  - **算法层**：SpMM = A_N×N × Din_N×K → Dout_N×K。每个nonzero (r_id, c_id, val) 触发：读取Din[c_id]行、读取Dout[r_id]行、SIMD multiply val×Din[c_id] + accumulate onto Dout[r_id]、写回Dout[r_id]。K=32，fp32。A中nonzeros按row-ordered traversal访问。
  - **系统框架层**：MatrixMarket格式稀疏矩阵→SPADE format conversion（COO-like，untiled）→SPADE accelerator execution。Host CPU (48-core Intel Xeon Platinum 8260M) 负责preprocessing。
  - **编译框架层**：论文未明确说明（SpMM为direct kernel execution，无compilation pass）。
  - **kernel调度层**：所有SPADE PEs（16个，OoO vector engine，1 SIMD MAC/cycle，L1 32KB）homogeneous处理整个稀疏矩阵。每个PE处理64 continuous rows的chunk。PE通过BBF bypass cache访问sparse A，通过L1访问Din rows（demand模式：每个nonzero的c_id触发一次L1 lookup，miss则从main memory读取）。Dout通过Inter-tile复用（同一row panel内的后续tiles复用首个tile带来的Dout rows）。全部矩阵区域使用相同的OoO pipeline和memory interface，不区分dense/sparse region。
  - **硬件架构层**：SPADE accelerator die，16 PEs + 3-level cache hierarchy (本文简化无L2/L3) + shared memory controllers。主存BW 205 GB/s (theoretical max)。计算瓶颈在dense tiles（更多nonzeros → 更多SIMD MACs），内存瓶颈在sparse tiles（更大稀疏度 → 更少data reuse → 更高bytes/FLOP ratio）。Homogeneous架构无法同时优化两者——cold worker（OoO latency-tolerant）处理dense tiles时compute throughput不足；hot worker（high-throughput scratchpad-based）处理sparse tiles时产生redundant memory accesses（将不需要的dense rows也stream入scratchpad）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出HotTiles：利用Intra-Matrix Heterogeneity (IMH) 的异构SpMM加速器架构，将稀疏矩阵的不同region分配给最适合的PE类型处理。核心创新：(1) IMH-aware analytical performance model——以tile粒度估计各worker type的执行时间和访存量；(2) heuristic partitioning——4种互补启发式在O(NlogN)时间内找到近似最优分区。对比baseline的全栈执行例子：
  - **算法层**：SpMM算法不变，但HotTiles在preprocessing阶段将稀疏矩阵partition为hot tiles（计算密集型→hot worker）和cold tiles（访存密集型→cold worker），并为各worker type生成对应的sparse format（COO-like for SPADE, tiled COO for Sextans; CSR-like for PIUMA）。
  - **系统框架层**：HotTiles framework运行于host CPU：(a) Profiling——small matrices homogeneous执行→搜索vis_lat参数；(b) Matrix Scan——per-tile调用performance model计算thi/tci和bhi/bci；(c) Partitioning——4 heuristics分别求解，选最优分区；(d) Format Creation——生成dual format。Framework一次性preprocessing可复用于多次SpMM inference（如GNN training→inference）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层→IMH-Aware Modeling + Partitioning**：解决baseline的homogeneous PE无法同时高效处理dense/sparse region的缺陷。**Modeling**：对每个tile分别计算hot/cold worker的5-task时间（read A/Din/Dout + MAC + write Dout），考虑4种data reuse类型（Inter-tile=0 access, Intra-tile stream=tile_width/tile_height rows, Intra-tile demand=uniq_cids/uniq_rids, None=tile_nnzs），不同sparse format的访存差异（COO=3*nnzs, CSR=2*nnzs+tile_height），以及worker的任务重叠方式（max vs sum）。引入vis_lat参数（data-driven）捕捉latency hiding。**Partitioning**：4 heuristics（MinTime Parallel/Serial, MinByte Parallel/Serial）——tiles按thi-tci或bhi-bci排序，linear scan cutoff index，选预测runtime最小者。当memory带宽压力低时MinTime优，带宽constrained时MinByte优；并行模式vs串行模式取决于merge cost与bandwidth contention的trade-off。效果：SPADE-Sextans上HotTiles vs homogeneous hot-only加速16.8×、cold-only加速2.0×、IUnaware加速2.2×、BestHomogeneous加速1.3×。Architecture exploration中HotTiles预测iso-scale架构性能趋势与实测一致，50%的matrix精确预测最优架构。
  - **硬件架构层→Heterogeneous PEs**：解决baseline单一PE类型无法匹配IMH的缺陷。Hot Worker（Sextans/STP）——高计算吞吐+scratchpad streaming，匹配dense tiles的高arithmetic intensity（data reuse through scratchpad, less pressure on memory BW）。Cold Worker（SPADE/MTP）——OoO latency-tolerant/round-robin multithreading，匹配sparse tiles的memory-bound特性（avoid redundant scratchpad fill for sparse tiles, each nonzero triggers on-demand memory access）。Merger module合并双output buffer（并行模式），面积/功耗<20% of single SPADE PE。Atomic engine (PIUMA) 消除merge cost。异构架构用较少total workers（scale 4 heterogeneous）超越2× workers的同构架构（scale 8 homogeneous）：2.9× over HotOnly8, 1.6× over ColdOnly8。

## 55-ORCHES- Orchestrated Test-Time-Compute-based LLM Reasoning on Collaborative GPU-PIM HEterogeneous System..pdf

- baseline方法是什么？
  Baseline是GPU-only系统（NVIDIA AGX Orin edge GPU [23]）执行TTC-based LLM推理，以及SOTA GPU-PIM系统（AttAcc[ASPLOS'24]、Duplex[MICRO'24]）执行标准LLM decoding。Baseline全栈执行例子（以TTC text-based reasoning，1B policy model + 8B PRM，search width=4，解MATH500一题为例）：
  - **算法层**：TTC pipeline [18]：Step 1 generation（policy model decoder生成4个候选分支，每分支decoding token-by-token）→ Step 1 verification（PRM prefill所有4个candidates进行scoring，选top-k）→ Step 2 generation（基于selected candidates继续decoding）→ ... 直至解答。Policy model decoding为memory-bound (batch size=4，arithmetic intensity低)，PRM prefilling为compute-bound (batch size ~100+ tokens)。
  - **系统框架层**：GPU-only执行。所有Linear和Attention operators在AGX Orin GPU上串行执行。Generation和Verification之间严格串行依赖（C2：verification必须完成后generation才能开始下一步）。Memory allocation为all branches预分配contiguous space，分支pruning后留下memory hole（C3：fragmentation）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：GPU执行所有operators。Linear operators：当batch size大（PRM prefilling时W~100+）→ compute-bound → GPU利用率高；当batch size小（policy decoding时W~4）→ memory-bound → GPU受限于204.8 GB/s带宽。Attention operators：shared KV query随search加深batch size渐增，unique KV query始终W=1。AttAcc baseline将全部attention固定分配给PIM、全部linear固定分配给GPU，未考虑TTC workload的dynamic parallelism变化（C1）。
  - **硬件架构层**：AGX Orin GPU：SM array + LPDDR5 memory。AttAcc PIM方案：Controller Die (aggregation/softmax) + Memory Dies (GEMV units)，但缺乏address cache和buffer应对fragmentation。Memory bandwidth受制于DRAM channel/bank multiplexing，expose仅fraction of internal bandwidth。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ORCHES提出GPU-PIM协同系统，三项technique各对应一个Challenge。ORCHES全栈执行例子（同一TTC推理任务）：
  - **算法层**：同baseline TTC pipeline，但由ORCHES的调度决定operator在GPU/PIM上的执行分配。Accuracy完全保留（无loss）。
  - **系统框架层→T2 Branch Prediction Pipelining**：解决C2（branch dependency阻碍pipeline）。Small PRM（large PRM的前10层）在GPU idle slot预执行verification，预测branch selection结果。Generation phase基于预测结果在PIM上speculative提前启动（3×–5× earlier）。若预测错误→回滚到correct selection重新generate。History alignment机制：用large PRM的历史步scores替换small PRM的历史scores，将预测准确率从~52%提升至~78%。Pipelined verification：candidate tokens部分生成后立即启动pre-verification（GPU idle且token数达threshold时），进一步overlap generation和verification。效果：消除inter-step stall，GPU和PIM同时高利用率（93.21% GPU + 61.0% PIM with T1+T2）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层→T1 Adaptive Assignment**：解决C1（variable parallelism complicating scheduling）。离线分析模型：根据batch size W、hidden dim D、compute capability CC、memory bandwidth BW（PIM有IO和internal两个BW）计算GPU和PIM的operator延时。策略：(a) W小→Linear+Attention全部分配PIM；(b) W中→shared KV query由GPU、其余PIM执行；(c) W大→Linear+shared attention由GPU、unique KV query由PIM。引入ratio α进行GPU/PIM协同处理（co-processing）：将operator output dim的α部分分配给GPU、其余PIM，求解T_GPU(α)=T_PIM(α)得最优α。在线补偿：每次step前根据shared KV的累积长度L_i动态重新计算各层α_i，按W_i从小到大逐层从GPU（α=1）reassign到PIM（α=0），直到T_PIM≥T_GPU，然后求解critical layer的α_t精确值。Data transfer overhead仅~8.3% total runtime。
  - **硬件架构层→T3 Memory Structuring**：解决C3（branch pruning引起memory fragmentation）。**(1) Address Cache**：SRAM实现的controller die cache，映射logical candidate ID→physical DRAM location，避免在DRAM cell中存pointer（2次有data dependency的DRAM access→1次SRAM+1次DRAM access，SRAM access快1-2个数量级）。**(2) Memory Reorganization**：追踪fragmentation ratio β = Total_Holes / Memory_for_Reasoning。当β→1时compact valid blocks到连续空间。每3-5步触发一次。**(3) Controller Buffer (Shared KV Buffer)**：在controller die缓冲KV segments，减少reorganization时的PIM-host data transfer，允许background reorganization。平均节省65% context memory footprint，runtime overhead仅0.12%，area overhead 12%。

## 52-Pimba- A Processing-in-Memory Acceleration for Post-Transformer Large Language Model Serving..pdf

- baseline方法是什么？
  Baseline是GPU-only LLM serving系统（NVIDIA A100），以及GPU+HBM-PIM系统。Baseline全栈执行例子（以Mamba-2 2.7B推理一个token为例）：
  - **算法层**：Mamba-2的selective state update操作——对每个head：scalar a_h decay上一token的state matrix S_{t-1}，outer product k_t v_t^T，二者相加得到S_t，GEMV S_t^T q_t得到output y_t。所有操作fp16精度。
  - **系统框架层**：PyTorch/CUDA执行，state在GPU HBM中，每token需从HBM读取整个S_{t-1} (dim_head×dim_state)、写入updated S_t。Batch size=128时，128个独立request各维护自己的state matrix，总读取量=128×dim_head×dim_state×2B(fp16)。
  - **编译框架层**：论文未明确说明（使用标准CUDA compilation）。
  - **kernel调度层**：GPU执行GEMV和element-wise操作。State update是memory-bound（arithmetic intensity仅0.25 FLOPS/byte），受限于HBM bandwidth (A100: ~2TB/s)。大规模batch下state update latency占比达74%（RetNet BS=128）。
  - **硬件架构层**：A100 GPU的HBM2E memory，数据经DRAM channel→GPU SM计算→写回。HBM-PIM baseline: time-multiplexed per-bank设计，fp16处理单元跨2 bank，但无pipeline，state update需多cycle。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法Pimba通过两个设计原则解决baseline的memory bandwidth瓶颈和面积效率问题。Pimba全栈执行例子（以Mamba-2推理一个token）：
  - **算法层**：MX8量化格式 + 随机舍入。State矩阵S每16个值共享8-bit group exponent，每对值共享1-bit microexponent，每值7-bit sign+mantissa——平均8-bit/值。相比fp16减少50%内存占用和带宽，且MX8的6-bit尾数（vs fp8的2-3bit）配合随机舍入有效避免SU-LLM连续state更新中的swamping效应（小值在累加中被淹没），perplexity与fp16几乎持平。
  - **系统框架层**：Pimba系统分prefill/generation两阶段。Prefill全在GPU（重组为compute-bound GEMM）。Generation将state update和attention offload到PIM：GPU产出operands→PIM command issue（REG_WRITE/COMP/RESULT_READ）→PIM内部计算→GPU取回partial sums→GPU执行FFN等剩余操作。PyTorch自定义操作注册，用户透明调用。与HBM-PIM的软件栈兼容，可"drop-in replacement"。
  - **编译框架层**：论文未明确说明（通过CUDA API扩展和PyTorch自定义操作实现，无独立编译框架）。
  - **kernel调度层**：PIM端SPU四阶段流水线执行state update——Stage1 fetch state sub-chunk→Stage2 decay+outer product并行（MX Multiplier）→Stage3 update（MX Adder）→Stage4 dot product + writeback。Access interleaving：2 bank共享1 SPU，iteration i从upper bank读、bottom bank写；iteration i+1互换角色——消除读写结构hazard，面积减半而吞吐不降。Command scheduling在ACT4的tFAW间隙重叠REG_WRITE、在PRECHARGES的tRP间隙重叠RESULT_READ。Chunk group数据布局最大化d_t/q_t/k_t复用。
  - **硬件架构层**：PIM accelerator集成在HBM2E DRAM中。SPU内含SPE（MX Multiplier + MX Adder + Dot Product Unit）。custom DRAM commands (ACT4/REG_WRITE/COMP/RESULT_READ/PRECHARGES)扩展标准DRAM接口。13.4%面积开销（<25%阈值）实现14.6×和6.9× state update延迟降低（vs GPU和GPU+PIM），端到端吞吐提升最高4.1× (vs GPU) 和2.1× (vs GPU+PIM)。

## 51-Crane- Inter-Layer Scheduling Framework for DNN Inference and Training Co-Support on Tiled Architecture..pdf

- baseline方法是什么？
  Baseline是现有的inter-layer schedulers，包括SET、Tangram、TileFlow、MBS和Checkmate，它们各自存在以下缺陷：SET用ratio-tree表示+模拟退火搜索，支持E+F+B但不支持R(recomputation)，且sub-batch处理顺序严格绑定到batch-level pattern导致调度灵活性受限；Tangram用coarse-grained dataflow，仅支持E+F，不支持R和B，仅适用于简单linear chain workload；TileFlow支持E+F+B(部分)的tree-based表示+遗传算法搜索，但仅限linear chain拓扑且不支持training；MBS是training专用，支持F+B但仅用sequential execution scheme；Checkmate支持R(recomputation)但仅batch-level粒度且不探索E/F。全栈执行例子（以SET为代表）：
  - **算法层**：ResNet-50 / Transformer-Large / Inception等DNN模型推理。模型表示为layer sequence的DAG，每层有固定的FLOP count和tensor size。
  - **编译框架层**：SET使用ratio-tree表示——将模型按层构建ratio tree节点，通过tree traversal决定execution pattern (sequential/pipeline)。搜索由simulated annealing执行：随机扰动tree结构 + sample候选方案 → 评估cost → 概率接受。缺陷：ratio-tree的construction inherently repeated patterns across same node，强制sub-batch处理顺序绑定batch-level pattern，无法探索如A1A2→(A3,B1)→(B2B3,C1C2)→C3等非标准顺序。
  - **kernel调度层**：SET的intra-layer scheduling使用brute-force exploration（遍历所有tiling factor组合）。Inter-layer部分：对于每个pipeline stage，分配tile资源给活跃layer——按layer workload比例定tile数。缺陷：(a) 不支持recomputation——training时activation全量存DRAM，OOM风险高；(b) fusion策略由ratio-tree隐式决定，无法精确控制哪些sub-batch存SRAM vs DRAM。
  - **硬件架构层**：NVDLA-style tiled accelerator，16-144 tiles，每tile 1024 int8 MAC+1MB SRAM，Mesh NoC，TSMC 12nm。对于ResNet-50 training BS=64，SET仅优化DRAM access（通过B和F）但无法降低DRAM capacity需求，OOM问题严重。Scheduling Inception BS=128 on 144 tiles需>2小时（AMD EPYC 7402P）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Crane提出层级化table-format表示+MILP优化，通过三个核心创新解决baseline的三重缺陷：Challenge #1 (不完整设计因子)→ScT+MeT table capturing E+F+R+B全四因子；Challenge #2 (灵活性受限)→层级化block表示+Theorem 1保证DAG完备性+pipeline state灵活分配表达任意执行模式；Challenge #3 (搜索慢)→MILP formulation使scheduling变为结构化优化问题，commercial solvers快速求解。对比baseline的全栈执行例子：
  - **算法层**：Crane支持任意DAG结构模型（ResNet skip connections, Inception branches, self-attention layers等），也支持training（forward+backward+recomputation联合建模）。Model表示为DAG→递归partition为hierarchical blocks（basic block=连续线性层，composite block=分支/合并点组成的block）。Theorem 1形式化证明了该表示对任意DAG模型的完备性。
  - **编译框架层**：Crane的核心编译流程：(1) Intra-layer exploration对每个sub-batch size候选进行brute-force tiling探索 → 得到各层tile utilization和数据movement特征。(2) Graph partition按DAG拓扑构建层级化block。(3) 对每个block构建ScT (Scheduling Table, 2N-1行×N列)和MeT (Memory Table，含SRAM和DRAM两个维度)的MILP模型——决策变量为整数ScT[i,j]和MeT[i,j]，约束为Eq.1-15（forward仅Eq.1-12，training含Eq.13-15），目标函数为EDP=Latency×Energy。关键创新：ScT+MeT的表格结构天然编码所有四个design factors——execution scheme由ScT的行累加模式决定，fusion由MeT中SRAM区间决定，recomputation由MeT_FW的DRAM checkpoints + Eq.14-15的两步backward流程决定，batch splitting由B_sub候选枚举+MILP内各sub-batch级别的track决定。(4) 层级化迭代refine——top-level优化→分解→lower-level优化→汇总cost→迭代至收敛。(5) 相比SET的ratio-tree+simulated annealing：Crane用MILP替代启发式搜索，O(m^{4n-1} log m)的search space被MILP solver的branch-and-bound + cutting plane高效剪枝，实现2.82× speedup（vs SET）和156.20× speedup（vs Tangram）。
  - **kernel调度层**：Crane的调度相比SET的关键提升：(a) **Execution scheme灵活性**——不强制sub-batch处理顺序绑定batch-level pattern。例如对3-sub-batch的3-layer pipeline，SET只能生成A1→(A2,B1)→(A3,B2,C1)→(B3,C2)→C3，而Crane可生成A1A2→(A3,B1)→(B2B3,C1C2)→C3——通过MILP内灵活设置s_i值实现。这消除了SET的rigid constraint导致的pipeline bubble，实现4.7× EDP reduction。(b) **Recomputation集成**——Crane通过MeT_FW的DRAM lower bound记录activation存储状态，MILP自动决定哪些activation保留（checkpoint），哪些丢弃后recompute。对ResNet-50 BS=64 training，Crane同时优化DRAM access和DRAM capacity（图1），使DRAM capacity需求降低2.2×，仅增加0.125× data access。(c) **Fusion精确控制**——MeT_S和ScT共同定义的SRAM存储区间精确指定了每sub-batch的fusion行为（存SRAM=fusion，存DRAM=no fusion），受Eq.11-12容量约束。
  - **硬件架构层**：与baseline相同硬件（NVDLA-style tile for inference, systolic array for training）。Crane的优势体现在相同硬件上：Inference时EDP降低1.13×–4.17× vs SET；Training时EDP降低11.01×–21.01× vs hypothetical training SET（后者因无recomputation导致OPT-6.7B OOM无法scheduling），降低3.62×–5.36× vs MBS（后者仅sequential execution scheme）。这些收益纯粹来自scheduling优化，不依赖硬件改进。

## 48-Meta_s Second Generation AI Chip- Model-Chip Co-Design and Productionization Experiences.pdf

- baseline方法是什么？
  Baseline是NVIDIA GPU（A100/H100等级别）服务器运行Meta的推荐模型（DLRM、DHEN、HSTU等）。全栈执行例子：
  - **算法层**：推荐模型推理，包含sparse network（embedding table lookup→TBE pooling）和dense network（FC layers→LayerNorm→activation→interaction layers）。模型规模50GB-2TB（90%为embedding），复杂度0.001-80 GFLOPS/request。GPU使用FP16或FP32精度。
  - **系统框架层**：PyTorch eager mode或TorchDynamo+TorchInductor→GPU CUDA runtime。GPU服务器配置Grand Teton平台，8 GPU/server，HBM提供高带宽（1.5-2 TB/s），单GPU功耗300-400W。模型通过NCCL跨GPU shard（embedding table sharding或tensor parallelism），batch从host CPU发往GPU。
  - **编译框架层**：PyTorch→TorchInductor→Triton→CUDA。Triton kernel编译为GPU SIMT指令，GPU SM内的Tensor Core做GEMM，CUDA Core做element-wise和reduction。无graph compilation时fallback到CUDA eager kernels（cudaMalloc→cudaMemcpy→kernel launch→cudaStreamSynchronize）。
  - **kernel调度层**：GPU上GEMM通过cuBLAS/cuDNN库或Triton custom kernel在Tensor Core执行。Embedding lookup通过GPU global memory gather（table在HBM中，index从host传入→thread block并行gather→shared memory reduce）。TBE kernel面临irregular memory access pattern，GPU L2 cache有限（40-80MB），HBM带宽是主要bottleneck。
  - **硬件架构层**：NVIDIA GPU使用HBM（高带宽高成本），通用SIMT架构（CUDA Core + Tensor Core），shared memory/L1 cache + L2 cache层次，PCIe连接host。GPU的通用性使其支持各类模型但功耗和成本高，HBM占总成本显著比例。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MTIA 2i通过专用硬件架构+model-chip co-design，在不追求极致通用性的前提下，用更小die面积和更低功耗达到competitive推荐模型推理性能，TCO降低44%。对比baseline的全栈执行例子：
  - **算法层**：MTIA 2i通过model-chip co-design使推荐模型匹配硬件特征：(a) 利用模型locality（activation buffer复用、weights只读）将activation pin在SRAM LLS，weights缓存在SRAM LLC，达到>95% SRAM hit rate；(b) graph fusion（sibling transpose FC fusion、MHA operator fusion、delayed broadcast）缩小activation working set以fit SRAM；(c) 拒绝SRAM-unfriendly模型变更（如triple remote embedding inputs导致activation溢出→throughput降90%），改用同效果但SRAM-friendly的替代（如加2个DHEN layer加深merge network）；(d) 动态INT8量化（RE per-row min/max + SE row-wise quantization）在FC层达到可接受的accuracy但生产部署仍有限。
  - **系统框架层**：MTIA 2i相比GPU的关键差异：(a) 24颗MTIA 2i/server vs 8 GPUs/server——更小芯片实现更细粒度资源分配，减小peak demand下的underutilization浪费，实际生产Perf/TCO比offline测试额外高5-90%；(b) PCIe Gen5 x8 per chip支持P2P通信；(c) eager mode硬件加速（<1µs job launch）使非graph化操作（merge network、real-time weight update）高效执行；(d) container management system [25] NUMA-aware调度，按accelerator粒度分配CPU cores/DRAM/NIC。
  - **编译框架层**：基于开源PyTorch 2.0+Triton，定制MTIA Graph Compiler将Triton IR映射到异构fixed-function units：GEMM→DPE，reduction→RE，SIMD→SE，layout transform→MLU，而非GPU的通用SIMT。关键优化：(a) autotuning框架自动化data placement/batch size/kernel variant/request coalescing选择，减少人工tuning工作；(b) Triton kernel编译利用multi-context和auto-increment custom instructions提高instruction issue效率。编译输出是异步数据流执行图→CP硬件管理依赖和调度。
  - **kernel调度层**：MTIA 2i的kernel调度针对memory hierarchy深度优化：(a) FC kernel通过3种stationary变体（input/output/weight）+ performance database近邻搜索自动选择最优变体；(b) 当activation fit in PE local memory时，解耦activation preload（from LLS）和weight broadcast（cross PE columns），利用hardware broadcast read消除NoC contention，weight tile prefetch到LLC隐藏DRAM延迟→latency改善45%并达到>95% DRAM带宽；(c) TBE kernel用新DMA_IN指令（支持index→auto address calculation+unaligned address）和SE 128-row accumulation减少embedding pooling指令数。与GPU的关键区别：GPU靠大HBM带宽硬抗irregular access，MTIA 2i靠SRAM caching+prefetch+instruction级优化在低带宽LPDDR上高效执行。
  - **硬件架构层**（核心差异化）：(a) **Memory hierarchy逆转**——256MB大SRAM (2.7TB/s) + LPDDR5 (204GB/s) 替代 HBM，SRAM:LPDDR带宽比13:1。设计依据：推荐模型locality强（>95% SRAM hit rate），大SRAM满足低延迟要求，LPDDR成本远低于HBM。(b) **异构PE架构**——DPE（2×32×32 MAC tile）专做GEMM、SE专做向量/SIMD、RE专做reduction、MLU专做memory layout transform，每单元针对其操作类型优化面积/功耗效率，而非GPU的统一SIMT Core+Tensor Core。(c) **硬件eager mode**——Work Queue Engine使job launch <1µs，Control Core升级到4核RISC-V并支持broadcast WQ descriptors，解决GPU eager mode时的CPU→GPU dispatch开销。(d) **Productionization优化**——overclocking（1.1→1.35GHz, +23%频率→5-20% throughput提升）、实时firmware update（2024年23次发布，GPU仅1-2次/年）、rack功率budget降低40%、LPDDR ECC tradeoff（10-15%吞吐惩罚但必需）。论文核心贡献在于这些productionization经验超出传统architectural design paper的范围。

## 37-Concerto- Automatic Communication Optimization and Scheduling for Large-Scale Deep Learning.pdf

- baseline方法是什么？
  Baseline是现有分布式训练框架通过手工通信优化实现计算-通信重叠（computation-communication overlap）。全栈执行例子：
  - **算法层**：训练GPT/Transformer等大模型，使用PTD Parallelism（pipeline+tensor+data）、ZeRO-powered data parallelism、Dynamic Axial Parallelism (DAP)、或自动并行（Alpa等生成的策略）。各并行方法引入不同通信模式（all-reduce, all-gather, reduce-scatter）。
  - **系统框架层**：不同框架各自实现通信优化：(a) Megatron-LM v2.7+重写Linear层forward/backward，使backward中all-reduce与参数梯度计算（MatMul）重叠，但forward的all-reduce完全无法overlap；(b) PyTorch DDP使用默认25MB bucket，一次backward计算完一桶梯度后异步触发all-reduce，但bucket size默认值在许多模型上非最优（GPT 2.5B需400MB，VGG19需70~200MB）；(c) DeepSpeed ZeRO使用类似bucket/wrapper机制固定调度all-gather和reduce-scatter；(d) JAX/XLA使用启发式latency hiding scheduler + Google Decomposition（仅能分解MatMul与all-reduce重叠，固定模式）。
  - **编译框架层**：(a) PyTorch eager mode：手工调用异步通信API+管理同步，难集成到PyTorch编译器栈；(b) XLA编译器：含latency hiding scheduler + multi-stream windowed einsum，但启发式算法在不同硬件环境下表现不稳定（NVLink关闭时JAX/XLA性能甚至低于Megatron-LM）。
  - **kernel调度层**：手工异步通信优化依赖CUDA Stream分离计算与通信——程序员手动将通信operator dispatch到独立stream，并通过event/callback保证正确性。受限：(1) 执行序随程序定义，无法自动重排算子最大化overlap；(2) 同步通信（如forward all-reduce）无法与前后计算overlap；(3) 通信融合需手动识别可融合算子。
  - **硬件架构层**：NVIDIA A800 GPU，NVLink 400GB/s intra-node，800Gbps inter-node。通信硬件能力充足（20%~40%训练时间为通信），瓶颈在软件调度而非硬件带宽。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Concerto提出一个编译器框架，通过RCPSP求解器自动化调度+SPMDSpec驱动的auto-decomposition，解耦并行策略与通信优化，解决baseline的三重挑战。对比baseline的全栈执行例子：
  - **算法层**：用户只需写PyTorch train_step函数+指定parallel_method（如"ptd","zero","auto"等），Concerto自动trace为fx Graph并转换ConcertoIR。支持任意模型（GPT, ViT, Evoformer, WideResNet）和任意并行方法。
  - **系统框架层**：Concerto作为PyTorch 2.0编译器栈的一部分（利用torch.fx, torch._custom_ops），提供one-line API注册并行方法（concerto.register_parallel_method）。编译阶段自动完成优化，运行时轻量——仅遍历优化后的拓扑序列dispatch到双CUDA Stream，消除手工异步通信管理的编程负担。
  - **编译框架层**（核心创新）：
    (a) **RCPSP Scheduling**：将执行建模为Resource Constrained Project Scheduling Problem（R={computation:1, communication:1}）。变量：每个任务i的执行区间[Si, Si+Ti]。约束：依赖保留（前驱完成后开始）、资源限制（每时刻≤1计算+1通信）。ILP目标：min makespan。使用Odd-even方法（分块迭代+偏移half block size）将NP-hard问题降为多项式时间，多轮迭代逐步逼近全局最优。与JAX/XLA启发式调度对比：NVLink关闭时JAX/XLA性能恶化（启发式不适应不同硬件环境），而Concerto的RCPSP求解器自动适配任意计算-通信比。
    (b) **Auto-Decomposition**：自动识别critical communication（独立计算节点提供的overlap不足以完全隐藏通信）。BFS沿各轴探索分解上下文：利用EasyDist的SPMDSpec（ShardSpec+CombineSpec）判断每个算子是否可沿给定轴分解——例如GPT Feed-Forward的LayerNorm→all-gather→MatMul1→GeLU→MatMul2，沿batch/sequence轴分解时Context包含全部5个算子；沿hidden轴时LayerNorm不可分解，仅含MatMul1。分解度N+重叠效率α=1.2+前驱/后继计算时间共同决定cost。若多critical comm上下文有交集，用ILP联合选择互不冲突的策略。对比Google Decomposition仅能分解MatMul与all-reduce的固定模式，Concerto可分解任意算子（LayerNorm, GeLU, MatMul等）。
    (c) **Communication Fusion**：自动识别同类型同参数且可互换执行序的通信算子并融合，减少kernel launch开销。ZeRO-3规模越大融合收益越显著（8 GPU: 0.517→0.505s; 32 GPU: 0.614→0.468s, FP16）。
  - **kernel调度层**：编译器生成拓扑序列后，Runtime dispatch计算→默认CUDA Stream，通信→专用CUDA Stream。End-of-communication marker确保使用通信结果的计算等待通信完成。关键优化：(1) Decoding阶段将通信提前到计算之前（因通信需少量SM launch kernel，避免被compute占满SM延迟launch）；(2) 辅助算子（getitem, view等零GPU时间）前移以腾出更多fusion空间；(3) 异步返回机制：ZeRO-2 optimizer末尾的同步all-gather在Concerto中异步返回未同步tensor，到下次计算图使用时再同步，重叠到下一forward。
  - **硬件架构层**：论文使用NVIDIA A800 GPU（同baseline硬件），无硬件修改。通过与T3（硬件-软件co-design实现fine-grained overlap，降低计算/通信相互干扰）正交互补——Concerto的RCPSP调度+auto-decomposition可配合T3的硬件机制获取更好性能。论文指出未来可扩展资源类型（当前{compute, comm}→增加{intra-node comm, inter-node comm}三种资源）以重叠不同通信。

- baseline方法是什么？
  Baseline是UPMEM-PIM商业系统上使用CPU驱动的、软件多线程的、AVX-512向量指令加速的DRAM↔PIM数据搬运方案。全栈执行例子：
  - **算法层**：PIM程序员通过UPMEM SDK API（dpu_prepare_xfer + dpu_push_xfer）显式指定每个PIM core的输入数据位置和传输大小，数据必须先分配在DRAM地址空间再显式拷贝到PIM地址空间。
  - **系统框架层**：UPMEM runtime library采用AVX-512 vector load/store指令（_mm512_stream_si512）进行大块数据搬运，多线程并发（每线程target不同PIM channel/bank）以最大化MLP。但此软件多线程方式受限于OS线程调度器的coarse-grained公平调度（CFS round-robin, ~几ms preemption quantum），导致内存流量在channel间不均衡。
  - **编译框架层**：论文未明确说明（使用标准gcc 9.4.0编译runtime library）。
  - **kernel调度层**：DRAM↔PIM数据传输分三阶段：(1) CPU从DRAM读取输入数据（AVX-512 load），(2) CPU执行data transpose预处理（因DIMM chip interleaving导致8-byte data word被拆分到8个×8 chip，须转置为8×8 byte矩阵使每个PIM core接收完整8-byte word），(3) CPU写入PIM地址空间（AVX-512 store）。整个流程占用大量CPU core资源（near 100% utilization），系统功耗达~70W。但实际内存带宽利用率极低：DRAM reads仅11.6%，PIM writes仅15.5%（8.9 GB/s vs 理论峰值57.6 GB/s）。根因：(a) 软件coarse-grained多线程调度不能像硬件fine-grained调度那样均匀分散流量到所有channel；(b) PIM-specific BIOS memory mapping（locality-centric, 禁用XOR hashing）统一应用于DRAM和PIM，使得正常DRAM访问也失去MLP优化，DRAM吞吐量仅为MLP-centric mapping的30%。
  - **硬件架构层**：UPMEM-PIM硬件：DDR4-2400 DIMM，8 chip/rank，每chip 8 PIM cores（DPUs），每core 1 GB/s峰值带宽。Host CPU: Intel Xeon Gold 5222。Memory bus integrated架构要求分离DRAM/PIM物理地址空间（通过BIOS更新），防止host processor和PIM core同时访问同一memory bank产生structural hazard。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  PIM-MMU通过硬件/软件协同设计，用三个硬件组件协同解决baseline的三重瓶颈。对比baseline的全栈执行例子：
  - **算法层**：程序员使用pim_mmu_transfer(ops) API，通过struct pim_mmu_op指定传输方向(DRAM_TO_PIM)、每bank传输大小(XFER_PER_BANK)、源DRAM地址数组(src_arr)、目标PIM core ID数组(dest_pim_id_arr)和PIM base heap pointer。单线程一次性offload所有传输信息，无需手动管理多线程。API通过runtime library → device driver → MMIO写入DCE的address buffer，之后CPU进入sleep等待DCE中断。解决baseline需要人工多线程分区管理的编程复杂性。
  - **系统框架层**：PIM-MMU runtime library + device driver通过MMIO将传输任务完全offload到DCE，CPU不再参与数据搬运。单线程提交、硬件自主完成，消除CPU core utilization和功耗问题（~70W → 大幅降低）。DCE自主完成data transpose预处理（preprocessing unit on-the-fly），消除CPU的AVX-512 power-hungry操作。
  - **编译框架层**：论文未明确说明（使用标准编译工具链）。
  - **kernel调度层**：核心创新——PIM-MS (PIM-aware Memory Scheduler)。利用DRAM↔PIM传输时不同PIM core目标地址互斥（无true data dependency）的特性，在硬件层对所有PIM core的memory requests做fine-grained reordering：(a) do-parallel所有channel同时发送请求 → channel-level parallelism最大化；(b) 优先bank group interleaving → 最小化tCCD delay；(c) bank内row buffer conflict最小化。对比baseline的OS CFS调度（~ms级quantum，channel流量不均衡），PIM-MS在cycle粒度调度，均匀分散流量。配合HetMap的dual mapping（DRAM用MLP-centric mapping保留XOR hashing和channel bits near LSB），DRAM吞吐量恢复4.9×（avg），PIM吞吐量通过PIM-MS的fine-grained调度完全解锁。
  - **硬件架构层**：PIM-MMU三个硬件组件协同：(1) **DCE**：AGU + SRAM buffers + Preprocessing unit，完全替代CPU进行地址生成和数据搬运；(2) **PIM-MS**：硬件scheduler在memory controller command queue层实现fine-grained reordering；(3) **HetMap**：根据物理地址范围动态选择MLP-centric mapping（DRAM区域，XOR hashing + channel bits near LSB）或locality-centric mapping（PIM区域，ChRaBgBkRoCo保序映射），与BIOS co-designed（BIOS boot时识别DRAM/PIM容量并告知memory controller地址分区边界）。面积开销仅0.85 mm²（0.37% CPU die），换来得益：DRAM↔PIM吞吐量4.1×提升，能效4.1×提升，端到端2.2×加速（max 4.0×）。Ablation study验证三个组件缺一不可：单加DCE（类DMA engine）反而性能下降（因失去AVX-512 wide vector并发优势），加HetMap仅恢复DRAM吞吐但PIM吞吐仍瓶颈，加PIM-MS才完全解锁。

## 31-PIM-STM- Software Transactional Memory for Processing-In-Memory Systems.pdf

- baseline方法是什么？
  Baseline是在UPMEM PIM上使用UPMEM原生同步原语（acquire/release原子指令）手写锁的并发控制。全栈执行例子：
  - **算法层**：应用使用锁（手工lock/unlock）保护共享数据结构的临界区。程序员需要手动划分数据、设计锁粒度、避免死锁、保证正确性。无事务抽象。
  - **系统框架层**：UPMEM SDK提供仅acquire/release两种原子原语，无ReadWrite Lock、无barrier、无条件变量等高级同步抽象。DPU程序使用SPMD模型，24硬件线程通过pipeline方式并行（有效深度11）。
  - **编译框架层**：UPMEM SDK基于clang编译器，无同步相关的编译优化或自动数据分区。
  - **kernel调度层**：acquire/release基于256-bit硬件原子寄存器，256个bit槽位通过硬件hash映射地址到bit index。DPU内同步通过此机制——tasklet acquire某地址→hash到bit→若bit为0则置1进入临界区→完成release清0。跨DPU同步须经CPU转发（延迟331μs vs 本地MRAM 231ns），且计算与通信不可重叠。锁粒度、竞争管理和可组合性完全由程序员手工处理。
  - **硬件架构层**：UPMEM PIM硬件。每DPU含64MB MRAM + 64KB WRAM + 24KB IRAM。无硬件CAS指令，无DPU间直连通信链路（CPU-mediated），无可重叠计算/通信。256-bit原子寄存器提供256个logical lock slot。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  PIM-STM提供7种STM实现的库，为UPMEM DPU提供事务抽象（start/read/write/commit/abort），程序员只需标记事务边界，STM库自动处理并发控制。对比baseline的全栈执行例子：
  - **算法层**：程序员用pim_stm_tx_begin()/pim_stm_read(addr)/pim_stm_write(addr,val)/pim_stm_tx_commit()替代手工锁。事务自动保证opacity（所有事务观察到的状态可序列化）。解决手工锁的死锁风险、可组合性缺失和错误倾向。
  - **系统框架层**：PIM-STM提供7种可选实现，编译时通过宏选择。支持STM元数据在WRAM（高速64KB）或MRAM（大容量64MB）灵活放置——WRAM放置减少instrumentation开销（2.86x几何平均加速），MRAM放置保留WRAM给应用数据。
  - **编译框架层**：论文未明确说明（编译层无修改，使用UPMEM标准clang工具链）。
  - **kernel调度层**：7种STM实现覆盖不同设计权衡：(a) NOrec用全局sequence lock粗粒度元数据——等待lock空闲才启动事务（内置backoff争用管理）→commit时acquire lock→写回缓冲值→release。读写时检测lock递增则触发readset value-based validation。缺点是大readset低竞争下频繁无效validation，但整体最稳健。(b) Tiny系列用细粒度ORec + 版本时钟validation——每地址有版本号，事务维护snapshot上下界，避免NOrec的频繁全局验证。在WRAM元数据下与NOrec竞争最佳。(c) VR系列用rw-lock表实现可见读——read时acquire共享锁（2bit模式+6bit计数+24bit读者身份，避免写集遍历），write时acquire排他锁。在低竞争大读取量场景下比Tiny快2x（避免readset validation，lock overhead在UPMEM上极低）。关键硬件适配：用acquire/release模拟CAS（acquire lock→检查值→release），rw-lock的owner地址编码避免多次MRAM访问，NOrec的backoff机制利用sequence lock做争用管理。
  - **硬件架构层**：PIM-STM利用UPMEM双内存层级的权衡（WRAM加速元数据 vs MRAM保留应用数据容量），利用256-bit原子寄存器的低延迟（寄存器级操作不访WRAM/MRAM），适配无CAS和无DPU直连限制（事务仅限本地DPU数据以最大化局部性，CPU-mediated通信留待应用层显式处理）。多DPU实验展示两种并行模式：KMeans（DPU协作+CPU合并centroids）和Labyrinth（DPU独立解实例），speedup达14.53x但能效增益有限（最高5x，部分场景能耗反增31.5%）。

## 32-pSyncPIM_Partially_Synchronous_Execution_of_Sparse_Matrix_Operations_for_All-Bank_PIM_Architectures.pdf

- baseline方法是什么？
  Baseline是commercial all-bank PIM（如Samsung HBM-PIM [24]、SK Hynix GDDR6-AiM [23]）只能执行密集BLAS操作，所有bank通过同一memory command同步执行相同指令序列。全栈执行例子：
  - **算法层**：仅支持dense matrix-vector操作（GEMV），无法处理COO/CSR等稀疏格式的indirect memory access和动态执行路径。稀疏矩阵应用（graph processing、linear system solvers）中的SpMV、SpTRSV、sparse vector操作无法在PIM上执行，必须由host GPU/CPU处理。
  - **系统框架层**：Host通过JEDEC标准interface控制DRAM。PIM execution使用mode switching：SB（single-bank, normal HBM）→ AB（all-bank, PIM kernel programming）→ AB-PIM（all-bank PIM, kernel execution）→ SB。所有bank在AB-PIM mode共享同一row/column/command——host无法感知各bank的不同执行状态（剩余元素数、寄存器状态等）。
  - **编译框架层**：论文未明确说明（手写PIM assembly，无自动化编译）。
  - **kernel调度层**：Dense kernel执行时所有bank处理等量workload、执行相同指令——无分歧。若强行运行稀疏kernel：(1) per-bank mode需host逐bank发送memory command，command数量增加2.74×，成为瓶颈；(2) all-bank mode无法处理稀疏矩阵的不均匀non-zero分布——不同bank访问不同memory row/column的需求与共享command冲突；(3) 稀疏格式（COO/CSR）的indirect memory access需要per-bank dynamic execution path，all-bank lock-step无法支持。
  - **硬件架构层**：Commercial HBM-PIM (Samsung)：2 banks share 1 PE（2:1 ratio），PE仅支持密集FP16/BF16 MAC操作。所有memory commands在all-bank mode按序issue。Processing unit无predicated execution能力，无CEXIT机制，无sparse vector queue——无法处理稀疏数据结构的动态元数据。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出pSyncPIM——在all-bank PIM架构上引入"部分同步执行"模型，通过predicated execution和conditional exit让每个bank的processing unit在lock-step共享command前提下各自有限度地分歧执行路径，解决commercial all-bank PIM无法处理稀疏矩阵的根本缺陷。对比baseline的全栈执行例子：
  - **算法层**：完整支持Sparse BLAS Level 1和2（Table III）——包括SpMV、SpTRSV、Gather/Scatter、稀疏向量element-wise操作等。采用COO格式（可扩展CSR/CSC），通过matrix compression（row-wise partition + all-zero column removal）和recursive block algorithm [1] for SpTRSV使稀疏计算适应PIM bank-parallel执行。ILDU分解归一化diagonal消除SpTRSV中的division操作。
  - **系统框架层**：保持JEDEC标准interface不变——host仍通过mode switching（SB→AB→AB-PIM）控制PIM。区别是：(1) 每个bank配一个独立PE（非2:1）以充分利用internal bandwidth；(2) 所有bank仍接收同一command，但per-bank execution behavior由CEXIT和predicated execution决定；(3) host通过检测所有bank完成状况确定kernel termination（无需per-bank query）；(4) matrix compression策略减少external I/O（子矩阵≤1KB匹配单memory row，避免外部traffic bottlebeck）。
  - **编译框架层**：论文未明确说明（手写PIM assembly，重排和预加载指令优化data dependency和ALU latency）。
  - **kernel调度层**：CEXIT指令是核心创新——每个PE运行infinite loop处理分配到的稀疏子矩阵，当sparse vector queue为空（用-1填充空位作为sentinel）时各自独立exit，实现"部分同步"。Predicated execution让多个PE concurrently执行同一load/store指令但根据各自queue状态（满/空）决定是否实际执行。SpMV kernel：IndMOV通过sparse queue column index实现indirect scalar read→SSpV scalar-sparse multiply→SpVDV dense-sparse accumulate→CEXIT条件退出。SpTRSV kernel：采用scalar multiplication-based算法（Algorithm 3）避免dot-product-based算法（Algorithm 1）的随机bank访问——对每列执行x[row] -= scale × value，将数据依赖转为column-level的独立并行。Recursive block decomposition将大三角矩阵分解为L0/M/L1子块递归求解。Matrix compaction减少external I/O（仅传输非零accumulated output）。对比baseline的per-bank 2.74× command瓶颈和all-bank无法分歧——pSyncPIM既保持了all-bank memory bandwidth优势又支持了per-bank execution path divergence。
  - **硬件架构层**：每个bank的PE新增：(1) 3×192B sparse vector queues（含row/col/value 3×64B子队列）存储稀疏数据元组；(2) CEXIT指令逻辑——当sparse vector queue empty触发独立退出；(3) Predicated execution硬件——指令执行取决于queue/register状态；(4) Index calculator——在VALU前union操作跳过零值（单侧非零则copy）、intersection仅计算匹配元素；(5) 32个loop counters + 5-bit ORDER字段支持嵌套循环；(6) 256-bit VALU支持INT8到FP64多精度。面积：PE 0.967mm² × 32/die = 30.94mm² PE + 38.05mm²（memory+TSV）= 68.99mm² total——介于Samsung HBM-PIM (84.4mm²) 和SpaceA (48mm², HMC) 之间，容量4GB。Power: SpMV workload max 5.0W，低于HBM2 power limit。通过保持JEDEC interface兼容，pSyncPIM比SpaceA（需要完全改变host-DRAM interface的独立PIM）更具deployability——虽然SpMV性能仅为SpaceA的0.56×，但支持SpaceA不支持的SpTRSV和多精度操作。

## 2-VQ-LLM_High-performance_Code_Generation_for_Vector_Quantization_Augmented_LLM_Inference.pdf

- baseline方法是什么？
  Baseline是将VQ codebook全部放在GPU shared memory中的naive fused dequantization-computation kernel（VQ-attn-SC版本）。全栈执行例子：
  - **算法层**：VQ算法（如CQ-2 VQ<4,8,1>）将Llama-7B的KV cache从FP16压缩为3-bit索引（1/8 size），离线完成量化。Dequantization在每次计算前执行：查codebook获取FP16 centroid→将多个sub-space结果concatenate恢复原始FP16精度。
  - **系统框架层**：无专门的VQ-aware serving framework。VQ kernel作为drop-in replacement接入LLM推理pipeline——对attention kernel，将quantized KV cache + codebooks传入替换原始FP16 attention kernel。
  - **编译框架层**：无自动代码生成。每种VQ配置×计算kernel的组合需手工编写CUDA kernel。开发者必须手动管理codebook在GPU memory hierarchy中的放置、tiling策略、dataflow设计。
  - **kernel调度层**：所有codebook entries放在shared memory中（VQ-attn-SC）。问题：(1) 大codebook占用大量shared memory → SM上concurrent thread blocks减少→SM利用率下降30%+；(2) entry数(256)远大于shared memory bank数(32)→严重bank conflict；(3) FlashDecoding沿token axis并行化→不同thread block处理不同token但访问相同codebook→重复off-chip traffic（Global→Shared），实际Global→Shared流量甚至高于FP16版本；(4) Dequantized data layout（row-wise 4 elements/thread, VQ vector_size=4）与computation所需layout（column-wise accumulation on V cache）不匹配→dequantized data需存回shared memory再由正确thread读取→引入额外Shared→Reg traffic。
  - **硬件架构层**：NVIDIA RTX 4090 GPU。L1 cache对codebook entry的temporal locality捕获失败（仅12.45% hit rate，entry size和irregular access pattern与128B cache line不匹配）。Shared memory bank conflict导致serialized access。Register和shared memory resource未充分利用（存在slack但未被利用）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出VQ-LLM，一个自动化的高性能fused VQ kernel代码生成框架，通过Codebook Cache + Codebook-Based Compute Engine解决baseline的三大挑战。
  **Challenge 1: Inefficient Codebook Access → Codebook Cache分层放置**
  Baseline将全部codebook放shared memory导致SM利用率下降和bank conflict。VQ-LLM发现codebook entries的access frequency高度不均匀（如AQLM-3有26个hot entries >µ+3σ，大量entries低于平均），因此采用分层放置：(a) Hot entries（极高frequency，如>µ+3σ）→ thread-local registers，彻底消除bank conflict；(b) Medium entries → shared memory；(c) Cold entries → global memory。实现采用reorder-based static mapping（offline按frequency排序+重编号index），runtime通过简单index比较定位entry，无tag array/lookup table开销。利用GPU resource slack（register/shared memory中不影响occupancy的余量）自适应确定nreg和nshared。这直接解决了SM利用率下降（释放shared memory后更多thread block可concurrent）和bank conflict（hot entries从shared memory移到register）。
  **Challenge 2: Uncoordinated Codebook Load and Compute → Codebook-Centric Dataflow + Hierarchical Fusion**
  Baseline沿token axis并行化导致多thread block重复加载相同codebook，以及dequantized data layout与computation layout不匹配导致shared memory round-trip。VQ-LLM：(a) **Codebook-Centric Dataflow**：沿codebook switch axis切分并行化task——如Attention按(H, C) axis切分，每个thread block仅覆盖一个codebook的范围（如4 channels），无需切换codebook，消除重复加载。split_factor通过平衡Traffic_Reduce和Traffic_Codebook的自适应公式确定。(b) **Codebook-Centric Hierarchical Fusion**：利用shfl_xor API实现register-level data exchange——将dequantization threads映射到mini-warp，通过数次shuffle将dequantized data在register中直接重排为computation所需layout，消除shared memory round-trip。自适应选择：nshuffle≤5用register fusion，否则用shared memory fusion。这直接解决了Global→Shared重复流量（codebook-centric dataflow消除重复load）和Shared→Reg额外流量（register fusion消除layout mismatch的shared memory中转）。
  **Challenge 3: VQ Algorithm/Computation Diversity → Template + Adaptive Heuristics**
  不同VQ算法（vector_size 2~8, #Entry 28~65536, Residual 1~2）和计算kernel（GeMM/GeMV/Attention）的组合空间巨大，手工优化不现实。VQ-LLM采用CUDA template-based设计：提供一组参数化CUDA模板（codebook cache + codebook-centric dataflow + hierarchical fusion），通过adaptive heuristics自动确定各参数值——nreg/nshared基于resource slack和entry size；split_factor基于output size和codebook traffic平衡；nshuffle基于vector_size和mma layout要求；thread mapping基于dequantized layout和computation layout的关联性offline确定。

  全栈执行例子（VQ-LLM CQ-4 Attention Decode on RTX 4090）：
  - **算法层**：CQ-4 VQ<2,256,1>，KV cache压缩为4-bit等价。Dequantization: index查32组codebook（每组256 entries, vector_size=2）→element-wise accumulate across residuals→concatenate sub-spaces。
  - **系统框架层**：同baseline pipeline，但使用VQ-LLM生成的fused attention kernel替换原始FlashDecoding kernel。支持通过Switch(CB) API在不同codebook间切换（适应GPTVQ等per-channel训练不同codebook的算法）。
  - **编译框架层**：VQ-LLM template自动生成kernel。(a) Offline: profile codebook entry frequency → reorder entries → 计算adaptivity参数（split_factor, nreg, nshared, nshuffle, thread_mapping）；(b) Kernel generation: Parallel_For沿(H,C) axis切分→每block Load一个codebook到cache→Access查表dequantize→Reg_Fusion via shfl_xor（CQ-4 vector_size=2, mma layout=2, nshuffle=1≤5）→temporal iteration沿非codebook switch axes做Attention compute→Reduce partial results。
  - **kernel调度层**：GPU执行时——每SM上更多thread block concurrent（shared memory footprint降低），hot entries在register中零bank conflict访问，每个thread block仅load自己的codebook segment（无重复Global→Shared traffic），dequantized data在register内通过shuffle直接对齐mma layout（无Shared→Reg round-trip），split_factor控制global reduction开销。
  - **硬件架构层**：NVIDIA RTX 4090。最终效果：相比GC baseline latency reduction 64.36%~99.1%（平均46.13%），相比AQLM open-source kernel达114× speedup。与AWQ/QoQ等element-wise quantization相比：4-bit等价下，GeMV latency为0.96×、GeMM为0.88×、Attention为1.01×——latency接近或优于element-wise方法，同时VQ可提供更高accuracy（arc-challenge上VQ-LLM比qServe高~2.5%）。End-to-end上Llama-7B生成256 tokens达~2.2× speedup vs FP16（与qServe相当）。

## 27-CoCoTree: A Computation-Capable Architecture for Collective Communication in Scalable PIM

- baseline方法是什么？
  DIMM PIM系统中的CPU-forwarding inter-PE通信机制。全栈执行例子：在算法层面，PIM workload（如BFS的node bitmap frontier reduction、GEMV的Reduce-Scatter）需要跨PE的集体通信操作（All-Reduce, Reduce-Scatter等）。在系统/Serving层面，host CPU通过UPMEM SDK提供的DMA操作显式地从各PE gather数据→CPU端执行reduction computation→再将结果scatter回目标PEs；SimplePIM [14]虽然有软件级collective API但仍依赖host CPU。在编译框架层面，PIM kernel代码（DPU binary）通过host-centric control flow编排通信：host CPU显式管理DMA传输和同步。在kernel调度层面，每个collective操作串行执行——CPU先完成一轮gather-compute-scatter后才能开始下一轮，无pipeline overlap；inter-PE带宽受限于DDR4-2400内存通道（tens of GB/s），仅占PIM内部聚合带宽的约2% [37]。在硬件架构层面，UPMEM DIMM内部bank间、chip间、rank间均无直接通信路径，所有inter-PE数据交换必须通过host CPU内存通道转发——数据路径为PE→MRAM→DDR bus→Host CPU memory→DDR bus→Target PE MRAM，造成冗余数据搬运和host CPU overhead。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出CoCoTree，一个面向DIMM PIM系统的计算能力分层树形集体通信架构。全栈执行例子：在硬件架构层面，CoCoTree引入两类硬件组件——Co-Leaf（嵌入每个PIM bank PE内，负责data packing/unpacking，通过DMA与PE交互）和Co-Node（作为tree中间节点，含可配置routing controller + multiple FUs，支持7种路由模式和byte-granular reduction）；chip内部8 Co-Leaf + 7 Co-Node形成完美二叉树，chip间通过双向SerDes链接组成rank-level tree，rank间通过Ribbon Cable组成DIMM-level tree，形成LEGO-like自相似可扩展网络。这直接解决了baseline中无direct inter-PE通信路径的缺陷，inter-PE数据不再经过host CPU。在kernel调度层面，CoCoTree采用two-phase协议：Configuration phase（一个PE发送command packet沿树上行到root再broadcast到subtree，沿途Co-Node解析INSTR字段配置FUs和routing）→ Computation phase（各PE通过CoCoTree::send()注入数据，Co-Leaf将multi-byte word拆为字节流并按操作类型重排[MSB优先/min-max，LSB优先/arithmetic] pack为32-bit packets→packets沿树上行，每层Co-Node的FUs在byte-granularity上对流经的左右子节点数据执行reduction[如sum时carry accumulator跨byte传播，overflow自动width expansion]→结果在root汇总→根据ADDR/STH向下broadcast到目标PEs→Co-Leaf unpack→PE通过polling获取结果）。这解决了baseline中CPU串行gather-compute-scatter导致的低带宽和无法scale问题——in-network computation在每个tree level将traffic减半，pipelining允许多轮通信在不同tree level重叠执行。在编译框架/系统层面，CoCoTree提供PIM-side API（CoCoTree::initConfig()/send()/getReceived()等），使PIM kernel代码内部直接表达集体通信，无需host CPU干预——例如BFS frontier reduction从host-side DMA gather+CPU compute+scatter简化为PE-side CoCoTree::send(local_bitmap)+CoCoTree::getReceived()。

## 26-SoMa: Identifying, Exploring, and Understanding the DRAM Communication Scheduling Space for DNN Accelerators

- baseline方法是什么？
  Cocco调度框架[49]代表了SOTA的layer-fusion DRAM通信调度方法。全栈执行例子：在算法层面，Cocco允许通过改变layer-fusion group（LG）的划分来减少DRAM访问（fused layers间通过on-chip buffer传递fmaps）；在系统/编译框架层面，Cocco使用heuristic策略——Computing Order和DRAM Cut由搜索决定，但Tiling Number固定为基于Core Array并行度要求（KC parallelism）的保守值，prefetching/delayed storing采用经典double-buffer策略（前一tile prefetch、后一tile store），FLC Set始终等于DRAM Cut Set；在kernel调度层面，每个compute tile的粒度较大（如ResNet-50总共约7962个tiles），backtracking halo overlap开销显著；在硬件架构层面，运行在通用DNN加速器模板（DRAM→GBUF→多Core）上，DRAM和计算利用率分别仅52.69%和62.64%（以层融合的ResNet-50 case为例），说明大量DRAM空闲带宽和计算stall未被利用。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出SoMa编译器框架，通过Tensor-centric Notation和两阶段SA探索全面定义和搜索DRAM Communication Scheduling Space。全栈执行例子：在算法/编译框架层面，SoMa的Tensor-centric Notation定义了6个属性（LFA: Computing Order / FLC / Tiling Number / DRAM Cut; DLSA: DRAM Tensor Order / Living Duration），构建了比Cocco大得多的优化空间——Cocco仅能改变2个属性，SoMa可改变全部6个属性；在编译框架层面，两阶段SA探索引擎（LFA→DLSA）配合Buffer Allocator迭代分配buffer预算，解决了Cocco的heuristic策略无法系统探索空间的问题；在kernel调度层面，(1) 第一阶段通过自适应调整Tiling Number（平均每网络751个tiles vs Cocco的7962个，coarser-grained tiles增加core内复用机会）和引入FLC（无需DRAM access即可释放weights和调整tile大小）来减少DRAM访问和fuse更多层（平均2.5个LG/网络 vs Cocco的13.0个），(2) 第二阶段通过智能调整DRAM Tensor Order和Living Duration实现精准的prefetching/delayed storing，利用DRAM空闲时段消除计算stall——例如将LG2首层的大weight提前多个tile prefetch，将FLG2末层ofmaps延迟一个tile存储并与LG2首层ifmaps交换位置，消除FLG2与LG2边界处的计算stall；在硬件架构层面，在同一加速器模板上实现了2.11×性能提升和37.3%能耗降低，第二阶段结果平均仅距理论上限3.1%，验证了充分利用buffer潜力和DRAM通信调度的有效性。

## 24-Accelerating LLM Serving for Multi-turn Dialogues with Efficient Resource Management.pdf (FlashGen)

- baseline方法是什么？
  vLLM等现代LLM serving框架在处理多轮对话时采用两种策略：(1) **Prompt recomputation**：每轮对话将全部前序轮次的prompt+generation tokens拼接为完整prompt，在prefill phase重算所有history tokens的attention KV。GPU内存仅保留当前running request的KV，不跨轮次缓存。(2) **FCFS调度**：请求按到达顺序处理，当队首请求因GPU内存不足（长prompt需要大KV空间）无法调度时，后续所有请求均被阻塞（head-of-line blocking），剩余GPU内存闲置。

  全栈执行例子（baseline: vLLM on 2×A100 80GB处理ShareGPT多轮对话session的第N轮请求）：
  - **算法层**：Transformer decoder inference（OPT 30B FP16），prefill phase + autoregressive generation。Multi-Head Attention，每层K/V shape = [total_history_tokens + prompt_tokens, num_heads, head_dim]
  - **系统框架层**：vLLM的FastAPI serving frontend接收请求→PagedAttention block manager分配KV blocks→iteration-level scheduler以FCFS从request queue取请求组batch→若队首请求所需blocks > 可用blocks，scheduler停止取新请求，GPU剩余block空间闲置
  - **编译框架层**：PyTorch v2.3 + CUDA 12.1，Flash-Attention（prefill）/ Flash-Decoding（generation）kernel，标准eager-mode execution
  - **kernel调度层**：Prefill phase——Flash-Attention kernel对[history_tokens + current_prompt_tokens]全部token序列计算Q·K^T·V。History KV在此phase从头重算，GPU global memory读取所有history token的Q/K/V。Generation phase——Flash-Decoding kernel每step对1个new token + 所有cached KV计算attention
  - **硬件架构层**：NVIDIA A100 80GB GPU，HBM带宽2.0 TB/s。Prefill phase为compute-bound（大量history token矩阵乘法），generation phase为memory-bound（读取全部KV cache）。GPU HBM中仅保留当前running request的KV blocks，历史KV不保留。Host 440GB DRAM和2×960GB NVMe SSD未用于KV缓存

- 论文方法是什么？如何对应解决Baseline的缺陷？
  FlashGen通过两项核心设计解决baseline的两个关键缺陷：

  **(1) FlashGen-Cache（多级KV Cache）→ 解决recomputation overhead**：
  Baseline重算全部history KV，浪费GPU compute。FlashGen-Cache利用三级存储（GPU → CPU host DRAM → SSD）分层缓存history KV：
  - **Proactive write-back**：推理过程中，每层decoder layer生成的new KV在写入GPU的同时异步DMA到host memory，消除reclaim时的拷贝延迟。GPU端completed request的KV保留为reclaimable cache（mark但可被running request回收）。
  - **Layer-by-layer pipelined restoration**：GPU miss时从host memory恢复KV——传输Layer L+1的KV与Layer L的attention计算并行（类似PipeSwitch pipeline switching），隐藏传输延迟。
  - **SSD staging + 动态fallback**：用户请求到达时预加载KV从SSD到host memory staging区（与当前running请求并行，非关键路径）。若SSD检索延迟>重算延迟（低负载无前序请求可并行），则动态选择recomputation而非SSD retrieval。
  - **Batch-aware KV restoration**：当batch仅含generation phase时，排除KV未完成加载的请求，避免compute unit stall。

  **(2) FlashGen-Sched（请求重排序）→ 解决head-of-line blocking**：
  Baseline FCFS调度导致长prompt队首阻塞短prompt。FlashGen-Sched贪婪地跳过被阻塞的队首请求，先调度后续内存需求小的请求（promoted requests），利用闲置GPU内存。为防止starvation：将promoted requests占用的内存计入free space计算中——当前序请求完成释放内存后，若累积free space（含可回收的promoted request空间）≥deferred request需求，则preempt promoted request并调度deferred request。Promoted request恢复时从host memory恢复KV（FlashGen-Cache保证无需重算）。

  全栈执行例子（FlashGen on 2×A100 80GB处理ShareGPT多轮对话session的第N轮请求）：
  - **算法层**：同baseline，OPT 30B FP16 Transformer decoder inference
  - **系统框架层**：vLLM + FlashGen-Cache + FlashGen-Sched。请求到达→按session ID查三级KV cache（GPU→CPU→SSD）→FlashGen-Sched reordering scheduler从request queue中跳过被阻塞的长prompt请求，优先取短prompt请求（promoted）组成batch→若history KV在host memory，启动pipeline restoration
  - **编译框架层**：同baseline PyTorch v2.3 + CUDA 12.1，但使用修改后的Flash-Attention/Flash-Decoding kernel（支持非连续KV blocks，类似FlashInfer）
  - **kernel调度层**：Prefill phase——若history KV在GPU（cache hit），直接以非连续block table传入修改后Flash-Attention kernel（gather-style block读取→拼接→tiled attention）。若history KV在host memory：后台CUDA stream DMA传输Layer L+1 KV blocks，前台Layer L执行修改后Flash-Attention（计算+传输并行）。若需SSD retrieval：SSD→host memory staging（异步，与当前running batch并行）→host→GPU pipeline传输。Generation phase——每step仅计算1 new token的attention（读取全部cached KV+history KV）。Batch中排除KV未就绪请求。
  - **硬件架构层**：GPU HBM保留running KV + reclaimable history KV cache（best-effort）。Host DRAM（224GB allocated）保active session的history KV（write-back copy）。NVMe SSD RAID-0（~1.8TB）保archived history KV。GPU HBM→host DRAM write-back bandwidth利用PCIe Gen4双向带宽（~32GB/s），host→GPU restoration同样利用PCIe带宽——每层KV传输量≈2（K+V）×num_heads×head_dim×seq_len×2 bytes。Pipelining使传输与计算overlap。GPU memory utilization提升至98%+（vLLM baseline约88%）。OPT 30B throughput从vLLM的~104 tokens/s提升至~166 tokens/s（1.63×），P99 TPOT从608ms降至103ms。

## 21-Tandem Processor: Grappling with Emerging Operators in Neural Networks

- baseline方法是什么？
  传统NPU设计将非GEMM算子交由三类方法处理：(1) off-chip CPU fallback——通过PCIe将中间数据传回host CPU执行非GEMM，再传回NPU继续GEMM；(2) dedicated on-chip hardware units——为有限几种非GEMM算子（ReLU/MaxPool/ResAdd等）设计专用硬件块，不支持的算子仍fallback到CPU；(3) on-chip RISC-V core（如Gemmini）——集成一个通用RISC-V核执行非GEMM，但单ALU处理大规模张量计算效率低；(4) on-chip general-purpose vector unit（如Google TPU VPU、NVIDIA CUDA cores）——提供向量化执行但缺乏针对非GEMM访存和循环模式的专用优化，仍需向量寄存器堆+缓存层次。
  
  全栈执行例子（baseline: off-chip CPU fallback执行BERT中的GeLU层）：
  - **算法层**：BERT inference，GeLU作为Transformer block中的element-wise激活函数，输入shape=[batch×seq_len, hidden_dim]
  - **系统框架层**：ONNX Runtime或TensorRT检测到GeLU为NPU不支持的算子→插入device-to-host memory copy→CPU执行→host-to-device copy
  - **编译框架层**：TVM/XLA将GeLU分解为CPU上的float运算序列，编译为x86 AVX指令
  - **kernel调度层**：CPU向量化执行GeLU≈0.5×x+0.5×x×tanh(sqrt(2/π)×(x+0.044715×x³))，通过PCIe Gen3×8在NPU和CPU间传输中间张量（约50-100μs per transfer for BERT hidden_dim=768 tensors）
  - **硬件架构层**：Systolic array GEMM unit空闲等待CPU完成非GEMM→PCIe写回→下一GEMM layer才能开始，GEMM和非GEMM串行执行，NPU利用率低。非GEMM占端到端runtime可达73%（EfficientNet on GPU），GEMM unit受Amdahl瓶颈限制

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出Tandem Processor——一个专门的、可编程的SIMD伴生处理器，与GEMM unit以tile粒度协同工作。四条核心设计创新对应baseline四大缺陷：
  
  **(1) 消除向量寄存器堆→解决访存开销**：将整个向量寄存器堆+缓存层次替换为软件管理的scratchpad（Interim BUF），通过Data Access Engine以tile粒度批量传输张量。去除load/store到寄存器堆的指令开销——baseline中该开销占非GEMM runtime的41%。
  
  **(2) Iterator Table间接寻址→消除地址计算**：在decode阶段通过⟨Scratchpad ID, Iterator Index⟩从Iterator Table中查找⟨Offset, Stride⟩元组，地址计算与ALU运算在pipeline中重叠执行。替代传统的每两个ALU指令需要3条额外地址计算指令——baseline中地址计算占非GEMM runtime的59%。
  
  **(3) Code Repeater硬件循环→消除分支跳转**：在fetch阶段用软件可编程表管理嵌套循环，按loop level存储迭代计数和Iterator ID映射。替代传统的条件分支实现循环——baseline中循环逻辑占非GEMM runtime的70%。
  
  **(4) Output BUF fluid ownership→实现tile粒度overlap**：GEMM unit完成tile写入Output BUF后，Tandem Processor直接取得ownership计算，无需显式数据拷贝。通过编译器插入synchronization instructions + Execution FSM handshaking实现double-buffering流水线。GEMM unit利用率提升20%，Tandem Processor利用率提升13%。

  全栈执行例子（NPU-Tandem执行BERT中GEMM→Add→GeLU subgraph）：
  - **算法层**：BERT inference，subgraph为MatMul + Add + GeLU（全部INT32量化，基于I-BERT方法）
  - **系统框架层**：GeneSys编译器从ONNX模型提取MatMul→Add→GeLU subgraph，进行layer fusion封装为一个execution block，uniform tiling分为4个tile
  - **编译框架层**：编译器生成混合指令流——GEMM instructions（配置systolic array MatMul）+ Sync instructions（SIMD_START标记Tandem Processor区域）+ Tandem Processor instructions（TILE_LD_ST加载tile到Interim BUF + ALU.ADD加法 + 5×ALU.MUL+3×ALU.ADD+ALU.SIGN+ALU.ABS+ALU.MIN实现GeLU + DATATYPE_CAST INT32→INT8 + Sync BUF release OBUF）
  - **kernel调度层**：Execution FSM→GEMM-Tandem状态。(Step 2) GEMM unit处理tile-0 MatMul→OBUF→Tandem Processor取得OBUF执行tile-0 Add+GeLU→Interim BUF。(Step 3) 同时GEMM unit处理tile-1 MatMul→OBUF（另一边Buffer bank），Tandem Processor从Interim BUF写回tile-0结果到off-chip→取得tile-1 OBUF。double-buffering持续进行。(Step 5-6) 最后tile-3 GEMM完成→Tandem Processor收尾tile-3非GEMM→Block Done
  - **硬件架构层**：32×32 systolic array（INT8乘法/INT32累加, 1GHz）+ Tandem Processor 32-lane SIMD（INT32 ALU, 128KB scratchpad, 1GHz）。每个Tandem Pipeline cycle：Fetch→Code Repeater控制循环迭代→Decode查Iterator Table获取⟨offset,stride⟩→Strided Address Calculation计算scratchpad地址→Scratchpad Read+ALU Execute+Write Back。非GEMM算子（即使是GeLU这种复合算子）在Tandem Processor上以流水线方式执行，不存在PCIe往返和CPU fallback

## 22-FEATHER_A_Reconfigurable_Accelerator_with_Data_Reordering_Support_for_Low-Cost_On-Chip_Dataflow_Switching

- baseline方法是什么？
  现有DNN加速器采用固定dataflow或有限dataflow切换，但忽视data layout对性能的关键影响。具体baseline分为三类：
  (a) **固定dataflow+固定layout**：NVDLA（fixed weights/output stationary, HWC_C32 layout）、Xilinx DPU（fixed dataflow, parallelism=(12,12,8)）、Gemmini（weights stationary, C/M parallelism=16）、Eyeriss（row-stationary, flexible Tiling/Shape but fixed layout），均固定一种layout。不同layer的最优dataflow需要不同的layout来避免bank conflict，固定layout导致大量layer处于sub-optimal dataflow下，compute utilization低（如NVDLA仅50%/39%/50% for BERT/ResNet-50/Mob-V3）。
  (b) **Flexible dataflow + 无on-chip reorder**：SIGMA（支持full TOPS flexibility）、MAERI等支持dataflow switching但无on-chip layout reorder能力。若采用fixed layout at runtime，flexible dataflow中大部分选择与layout discordant，导致bank conflicts。若采用off-chip reorder（将iActs搬回DRAM→CPU重排→搬回accelerator），引入显著latency和energy overhead（MobileNet-V3中off-chip reorder暴露24% critical latency）。
  (c) **Flexible dataflow + 有限on-chip reorder**：Medusa（line rotation，仅支持3 lines/bank concurrent access）、MTIA/TPUv4（transpose + row-reorder，无法支持multi-dimensional parallelism所需的word-granularity arbitrary reorder）。这些RAR（Reorder After Reduction）方案将reorder放在critical path上，且reorder latency无法被隐藏。

  全栈执行例子（baseline: NVDLA-like fixed dataflow + fixed HWC_C32 layout执行ResNet-50 layer 1, C=3, H=224, W=224 convolution 7×7 stride=2）：
  - **算法层**：ResNet-50 inference，layer 1 conv (C=3, M=64, H=224, W=224, R=S=7, stride=2, padding=3)
  - **系统框架层**：NVDLA fixed weights/output stationary dataflow——weights pre-loaded on-chip, iActs stream from DRAM
  - **编译框架层**：Dataflow fixed at design time——仅T(iling)可调（tile size沿C/M/H/W = 1/64/224/224或类似划分），Ordering/Parallelism/Shape全部fixed。Timeloop/LayoutLoop search只能在此受限空间内选择tile size
  - **kernel调度层**：16×16 weight-stationary systolic array——每cycle从on-chip buffer读取iActs，但通道数C=3仅能利用3/16=18.75% PE（C dimension parallelism=4, M dimension parallelism=4, 负载不均衡），steady-state utilization低
  - **硬件架构层**：HWC_C32 layout下，iActs按inter-line order H→W→C、intra-line order (32 C elements per row)组织。当C=3时，每条line仅含3个有效数据、剩余29个为padding/zero；且不同(H,W)坐标的iActs分布在大量separate lines中，单cycle需concurrently read多条lines（≥ports数），进一步引发bank conflicts降低effective bandwidth

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出FEATHER accelerator，通过两大创新实现低开销的(dataflow, layout) per-layer co-switching：
  
  **(1) NEST（Neural Engine with Spatial forwarding and Temporal reduction）→ 解决dataflow flexibility**：
  2D PE array支持完全TOPS flexibility（arbitrary parallelism over any dimensions + arbitrary shape of virtual PE grouping），通过两阶段计算（local temporal reduction + interleaved spatial reduction时分复用BIRRD）实现near-full utilization under任意dataflow。相比1D array（MAERI/SIGMA）的all-to-all NoC scalability问题，NEST通过time-multiplexing PE rows共享同一reduction network达到2D array的scalability + 1D array的flexibility。

  **(2) BIRRD + RIR（Reorder in Reduction）→ 解决layout reordering overhead**：
  BIRRD是一个基于butterfly topology的multi-stage reduction/reordering网络，支持arbitrary reduction group和arbitrary reordering（基于rearrangeably non-blocking multicasting）。关键创新RIR将layout transformation从explicit pre/post-processing转移到reduction phase内部——post-reduction oActs在BIRRD中既被reduce又被reroute到target output bank/line，写入时已经是next layer concordant layout。这消除了：(a) 显式reorder latency（被reduction隐藏）；(b) 额外intermediate storage需求（不需要reorder buffer）；(c) bank conflicts（reduce减少oActs数量，匹配memory write ports）。同时BIRRD消除了对复杂distribution NoC的需求——因为data总是以完美layout到达，仅需point-to-point connections。

  FEATHER解决baseline三类缺陷的映射：
  - vs 固定dataflow（NVDLA/DPU/Gemmini/Eyeriss）→ NEST提供full TOPS flexibility，per-layer选最优dataflow
  - vs 无on-chip reorder（SIGMA+fixed layout）→ BIRRD通过RIR实现零latency的layout reordering
  - vs 有限reorder（Medusa/MTIA/TPUv4）→ BIRRD支持arbitrary reorder（word-granularity arbitrary permutation across entire 2D buffer）

  全栈执行例子（FEATHER执行ResNet-50 layer 1→layer 47 dataflow-layout co-switching）：
  - **算法层**：ResNet-50 inference, layer 1 (C=3, H=224, W=224, 7×7 conv) → layer 47 (C=2048, H=W=7, 3×3 conv)
  - **系统框架层**：LayoutLoop offline为每层选择最优(dataflow, layout) pair——layer 1选(HWC_C32 layout, channel-parallel dataflow with C/M parallelism)，layer 47选(HWC_W8 layout, spatial-parallel dataflow with W/H parallelism)。Dataflow-layout pair的BIRRD configurations预计算并存储在Instruction Buffer中
  - **编译框架层**：LayoutLoop mapper对每层执行pruned random search over dataflow space × exhaustive search over layout space，以EDP为victory condition选择最优pair。Search过程中对每个candidate检查bank conflict（每cycle访问的lines vs ports），引入slowdown factor修正latency/energy estimation
  - **kernel调度层**：
    Layer 1执行（NEST dataflow: C-parallel=4, M-parallel=4, weights stationary）：Phase 1 (cycles 0-2)——4 PE rows × 4 PE columns同时进行local temporal reduction（每个PE内iActs × weights partial sum累加）；Phase 2 (cycles 3-5)——top row输出4 partial sums到BIRRD，BIRRD配置为4:2 reduction + reroute oActs到bank 0 line 0（next layer row-major layout）；后续rows依次时分复用BIRRD。Steady state: 100% PE utilization。
    Layer切换（layer 1→layer 47）：Layer 1 oActs已在StaB Pong中以layer 47所需的row-major layout写入。Layer 47 iActs直接以concordant layout从StaB Pong读取，无需任何reorder操作。BIRRD配置switch（instruction buffer读取新config word）。
    Layer 47执行（NEST dataflow: W-parallel=4, H-parallel=2, C=2048沿PE rows分布）：同理两阶段→BIRRD 16:4 reduction → oActs写入next layer concordant layout
  - **硬件架构层**：16×16 PE array (TSMC 28nm, 1 GHz target)。BIRRD: 16-input (AW=16), 2×log₂(16)=8 stages, 每stage 8 Eggs, 每个Egg由2-bit控制。StaB Ping/Pong: 16 banks × (num_line × 16B line_size), 每bank dual-port SRAM。Layout switch latency = BIRRD config reload from IB（negligible, ~10 cycles）。端到端效果：FPGA上2.65×/3.91× throughput vs Xilinx DPU/Gemmini，Layoutloop上1.27~2.89× speedup and 1.3~6.43× energy efficiency vs SoTAs，仅6% area overhead over Eyeriss-like fixed-dataflow accelerator

## 1-TAIDL- Tensor Accelerator ISA Definition Language with Auto-generation of Scalable Test Oracles

- baseline方法是什么？
  现有学术张量加速器的软件工具链现状：(a) **ISA定义层面**：绝大多数学术加速器（Eyeriss、MAERI、FEATHER等）缺乏良好文档化的ISA规格和指令语义，不具备类似CPU的x86手册或GPU的NVPTX虚拟ISA。这使编译器回填、优化kernel编写等软件工作无法系统化进行。(b) **正确性测试层面**：现有指令级功能仿真器（测试预言）如Gemmini Spike（C++单线程）、Intel SDE（Pin动态二进制插桩）是手写设计，针对特定加速器或ISA扩展，不可复用到其他加速器。这些仿真器用通用编译器（GCC/Clang）编译，循环嵌套和条件分支的优化能力有限，且为单线程运行，对ML工作负载中的大张量操作（如1024×1024矩阵乘法）仿真耗时超过1分钟，无法集成到日常软件开发的CI/CD测试流水线中（Figure 1所示，正确性测试必须在性能测试之前通过）。(c) **工程代价**：每个新加速器都需要从头手写仿真器，重复工程量大。

  全栈执行例子（典型学术加速器Gemmini，使用Gemmini Spike仿真器测试GEMM kernel）：
  - **算法层**：tiled matrix multiplication C = A×B + D，其中A/D shape=(I·DIM)×DIM，B shape=DIM×DIM
  - **系统框架/编译层**：Exo编译器或手写C kernel调用Gemmini ISA指令（mvin, mvout, compute_preloaded等），编译为RISC-V + Gemmini扩展二进制
  - **编译框架/测试预言层**：Gemmini Spike（RISC-V Spike扩展）逐条解释RISC-V指令和Gemmini自定义指令——每条指令在C++中通过嵌套循环模拟tile访问和数据流计算，单线程执行，无向量化、无并行化
  - **kernel调度层**：指令级串行仿真——mvin→mvin→compute_preloaded→...→mvout，每条指令内循环遍历tensor元素
  - **硬件架构层**：模拟Gemmini的systolic array（DIM×DIM MAC），spad bank访问，accumulator读写
  Baseline缺陷：仿真速度慢（1024×1024矩阵乘>60秒）、不可扩展到大张量、每个加速器需从零构建、单线程无法利用多核CPU/GPU。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出TAIDL+TAIDL-TO双层方案：

  **(A) TAIDL语言（解决ISA定义缺失和复用性）**：
  - 设计专用于张量加速器的ISA定义DSL，含数据模型（tensor buffers多维定义+control registers）和指令语义（tb_read/tb_write/hlo_op/assign/assert+IF/REPEAT控制块）
  - 用XLA-HLO算子（reshape, transpose, dot_general, convert, bitcast_convert, reduce_precision等120+算子）描述指令计算语义，无需涉及微架构实现细节
  - 支持多维base-type（如AMX tile=16×64×i8而非1024×i8）和多维addressing（如TPUv2 MXU buffer=(1,256)×128×f32），简化地址计算
  - 理论上Turing-complete（XLA-HLO superse FLooP），支持标量语义（0-D tensor）和bit-vector语义（bitcast_convert），forward compatible（自定义数据类型通过reduce_precision/clamp/custom_call）
  - 效果：TPUv1、Intel AMX、Gemmini三种架构的ISA均在TAIDL中成功实例化，证明了表达能力

  **(B) TAIDL-TO自动生成（解决测试预言慢、不可扩展、不可复用）**：
  - 关键创新——变换算法transform()：将ISA指令流→XLA-HLO计算图。具体为：(i) 控制寄存器和调用属性通过常量折叠在变换时解析（无需运行时分支）；(ii) REPEAT块展开（loop unrolling）；(iii) IF块按控制寄存器值选择分支；(iv) tensor buffer读写转为XLA-HLO slice/dynamic_update_slice算子；(v) hlo_op保持不变直接嵌入
  - XLA-HLO计算图→XLA编译器编译→生成高度并行的可执行文件（CPU多线程+GPU kernel）
  - XLA自动应用张量编译器优化：算子融合（operator fusion）、代数化简（algebraic simplification）、内存tiling和布局优化、自动向量化和并行化
  - TAIDL-TO可部署在CPU和GPU上，GPU加速使大张量仿真更快
  - 效果：Gemmini DIM=256时比Spike快1200×(CPU)/5600×(GPU)；Intel AMX oneDNN kernels比Intel SDE快约10×；端到端I-BERT模型(12层, 768 emb, 512 seq)仿真仅需0.8s(GPU)/2.4s(CPU) vs Gemmini Spike >50分钟

  全栈执行例子（同样的Gemmini GEMM kernel，使用TAIDL-TO）：
  - **算法层**：相同——tiled matrix multiplication C = A×B + D
  - **系统框架/编译层**：架构师编写TAIDL定义（约100行描述Gemmini数据模型+指令），自动生成TAIDL-TO Python库
  - **编译框架/测试预言层（核心变化）**：程序员写@kernel函数调用API→compile()触发变换算法→每条Gemmini指令展开为XLA-HLO算子序列（mvin→slice+reshape+bitcast_convert, compute_preloaded→convert+dot_general+add, mvout→bitcast_convert+reshape+dynamic_update_slice）→XLA编译器融合算子、并行化、向量化→生成.pb可执行文件
  - **kernel调度层**：不再是逐指令串行仿真。XLA生成的代码中，内存读写（33-60%）、布局变换（16-60%）和矩阵乘法（~7%）被融合优化，CPU多线程或GPU kernel并行执行
  - **硬件架构层**：TAIDL-TO不模拟微架构（不建模pipeline depth/cache hierarchy），仅确保功能正确性（bit-accurate vs RTL仿真）

- baseline方法是什么？
  Baseline是传统layer-level scheduling（单层tiling优化）和已有的subgraph-level方法的组合：(a) Layer-level scheduling：使用loop transformation（tiling, reordering）单层执行，局限于intra-layer数据复用，对memory-intensive网络效果不足。典型实现如SIMBA的output-centric tiling——将output tensor划分为tiles并逐tile计算，intermediate数据经global buffer写回DRAM供下一层使用；(b) 已有subgraph-level方法：Fused-CNN[4]的枚举式partition（对plain网络，search space=O(2^2N)无法扩展到大规模网络）；Halide[47]的greedy fusion（逐对合并最优benefit pair，易陷入局部最优，且fusion规则基于固定硬件无法co-explore DSE）；Irregular-NN[73]的DP partition（layers按depth排序后DP，仅允许depth-ordered contiguous assignment，对non-plain网络搜索空间受限）；(c) 内存设计方面：传统固定buffer容量设计（Small/Medium/Large three-tier），缺乏从workload出发的memory size探索。全栈执行例子（ResNet50在SIMBA-like accelerator上，1MB global buffer, layer-level assignment）：(i) 算法层：ResNet50的conv-bn-relu block序列；(ii) 系统框架/编译层：Timeloop/MAESTRO单层mapper确定每层output tile size→output-centric tiling→global buffer weight/activation allocation→生成单层task序列；(iii) kernel调度层：逐层执行——tile of Conv1→MAC阵列计算→写回global buffer→写回DRAM→load Conv2 weights→load Conv1 output作为Conv2 input→...，无inter-layer data reuse；(iv) 硬件架构层：4×4 PE array（含8×8 MAC），1MB global buffer，1.125MB weight buffer，DRAM 16GB/s bandwidth per core。Baseline缺陷：每层输出的intermediate data必须写回DRAM再由下一层读回，EMA随模型深度线性增长；固定buffer容量可能过大（浪费硅面积和静态功耗）或过小（频繁DRAM访问增加energy和BW需求）；无graph-level partition优化能力或有但受限。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Cocco框架，分两个层面解决baseline缺陷：
  
  **(A) Subgraph-level execution scheme（解决单层数据复用不足）**：
  - 设计consumption-centric flow替换production-centric scheme。传统production-centric会导致多余data cached在buffer中（不同分支的kernel size/stride不同导致unbalanced production），consumption-centric从output nodes反向推导tile size和update offset，用LCM对齐多consumer的input需求，使每层仅生产consumer实际需要的data量。
  - 设计MAIN/SIDE region双区域memory management。MAIN region存储PE source data并实现vertical overlap的本地复用；SIDE region保留horizontal overlap data，通过load-from-SIDE和write-to-SIDE机制实现full data reuse across sliding convolution windows。相比传统单层执行每次reload全部data，eliminate了intermediate data的DRAM往返。
  - 效果：fusing 3 layers（L=3）即可减少EMA 42.3%~74.7%，BW requirement 26.8%~67.8%。

  **(B) GA-based hardware-mapping co-exploration框架（解决partition搜索和memory DSE的自动化和协同问题）**：
  - 形式化目标：minimize combined cost = Buffer_Size + α × ΣCost_M(subgraph)，其中Cost_M可为EMA或energy。α超参数调节memory capacity和communication之间的trade-off。
  - 设计genome encoding同时编码partition scheme和memory configuration，通过customized crossover和4种mutation操作（modify-node/split-subgraph/merge-subgraph/mutation-DSE）探索完整搜索空间，GA的种群多样性天然避免local optimum。
  - 关键设计：in-situ split-subgraph调优——当genome中某subgraph超过buffer容量时，evaluation阶段自动split而非简单标记invalid，大幅提升valid sample率和搜索效率。
  - 效果：co-exploration相比固定硬件方案cost降低1.89%~50.33%，收敛样本数仅为two-step方案的3.3%~27%。

  全栈执行例子（ResNet50在Cocco优化后）：
  (i) 算法层：ResNet50 DAG提取（NN-parser），conv/bn/relu/element-wise layers作为nodes，依赖关系作为edges；
  (ii) 系统框架/编译层：Cocco GA框架——输入DAG+硬件配置+内存设计空间→经过多代crossover/mutation/selection→输出optimal partition scheme和memory config（如4-subgraph方案+704KB global buffer/864KB weight buffer）。Cocco输出的subgraph partition决定了哪些layers融合为一个subgraph，以及每个subgraph内采用consumption-centric flow执行；
  (iii) kernel调度层：每个subgraph内按consumption-centric flow执行——output node tile drives→LCM反向推导各层tile size和update offset→确定MAIN/SIDE region分配→确定upd_num和执行序列→P E Array按步骤执行。Subgraph间按topo序调度，prefetch下一subgraph的weights。只有需被后续subgraph使用的output才writeback到DRAM，其他intermediate data在on-chip buffer内完成全复用。每层无需单独的DRAM读回intermediate data；
  (iv) 硬件架构层：SIMBA-like accelerator（4×4 PE array, 704KB global buffer, 864KB weight buffer, 16GB/s DRAM bandwidth per core），buffer region manager硬件以272-Byte register file管理逻辑buffer分区（面积仅0.18%），支持MAIN/SIDE region的flexible partitioning。多核场景通过crossbar共享weights减少单核buffer压力。

  核心创新映射到baseline缺陷：(D1) layer-level limited reuse → subgraph-level consumption-centric execution + MAIN/SIDE双区域全复用；(D2) fixed buffer capacity无法适应workload → Cocco co-exploration自动为每种workload找到最优buffer size和partition；(D3) 已有subgraph partition方法（Greedy/DP）搜索受限或易陷局部最优 → GA的种群多样性和customized 4-mutation避免local optimum，支持任意拓扑网络；(D4) partition和DSE独立执行、信息不互通 → genome同时编码partition和memory config，evaluation在同一个fitness中联合优化。

## 16-RELIEF_Relieving_Memory_Pressure_In_SoCs_Via_Data_Movement-Aware_Accelerator_Scheduling

- baseline方法是什么？
  SOTA加速器调度策略（以LAX [59]和HetSched [3]为代表）是基于Laxity/Deadline的调度策略，将任务按deadline或laxity排序插入每种加速器的就绪队列，但完全忽略硬件数据转发（data forwarding）机制的存在。全栈执行例子：以2个并发DAG（Figure 2a）在拥有3种加速器类型的移动SoC上执行为例，(i) **算法层**：应用由DAG表示——节点为加速器kernel（ISP、grayscale、convolution、elem-matrix等），边为数据依赖。每个节点有runtime和deadline；(ii) **系统框架/编译层**：硬件管理器（基于GAM+ [15]的微控制器实现）提供host interface（CPU→共享内存提交DAG）、scheduler（按策略排序就绪队列）、drivers（操控加速器MMIO寄存器），**论文未明确说明编译框架层**；(iii) **kernel调度层**：SOTA策略如LAX——在每个加速器就绪队列中按laxity = deadline - runtime - current_time升序排序。当加速器空闲时，取队列头部任务执行。LAX通过将负laxity任务deprioritize提高deadline满足率。但所有SOTA策略完全忽略数据转发机会：即使consumer的producer刚刚完成（数据还在producer scratchpad中），consumer仍按laxity排序，可能被其他无关任务插队，导致producer数据被刷回主存后再从主存取回；(iv) **硬件架构层**：ARM Cortex-A7微控制器（硬件管理器）+ 7个图像处理加速器（每个含scratchpad，通过non-coherent read-only port暴露给系统互联）+ LPDDR5-6400主存 + bus/crossbar互联。硬件已支持peer-to-peer DMA数据转发（consumer可从producer scratchpad直接读数据），但调度器未利用。
  Baseline核心缺陷：(D1) 调度策略对数据转发硬件机制"无感知"（oblivious）——SOTA策略平均仅利用<65%的可能转发机会，某些场景低至8%；(D2) 虽然forwarding硬件可将数据从producer scratchpad直接传给consumer，但producer scratchpad空间有限（被后续任务覆盖时数据丢失），要求consumer在producer完成后被立即调度；(D3) deadline-driven策略如LAX/HetSched在满足deadline时可能造成不公平——某些应用被严重slowdown（starving），而另一些应用超额满足deadline；(D4) 在内存密集型RNN workload（GRU/LSTM，eem-matrix独占，75%时间花在数据搬移）中，round-robin风格调度（FCFS）和locality-oblivious调度（GEDF）forfeit所有forwarding机会，浪费高达75%的execution time在数据搬移上。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：RELIEF——基于Least Laxity First (LL)的在线调度策略，核心设计：(1) 转发节点优先级提升——producer加速器完成时，将其children标记为forwarding节点，优先插入ready_queue前端（而非按laxity排序位置），确保consumer能立即利用producer scratchpad中的数据；(2) is_feasible()可行性检查——在提升优先级前，遍历ready_queue中比候选节点优先级更高的非转发节点，检查其中首个laxity>0节点的laxity是否大于candidate runtime。若所有受影响节点的剩余laxity足够容忍额外延迟，才允许提升；否则拒绝提升，节点按原始laxity位置插入。此检查确保优先级提升不会导致deadline miss；(3) max_forwards限流——每种加速器类型最多提升的forwarding节点数不超过该类型当前空闲加速器实例数，确保forwarding节点总是下一个被调度的；(4) 运行时预测——预测compute time（固定函数加速器，按输入大小查表，误差0.03%）和memory time（三种bandwidth predictor: Last/Average/EWMA，数据搬移量通过分析DAG结构预测colocation和forward）。
  全栈执行例子：同Figure 2a的2个DAG场景，(i) **算法层**：应用DAG不变；(ii) **系统框架/编译层**：硬件管理器不变（仍为Cortex-A7微控制器bare-metal C实现），仅替换调度策略部分——RELIEF算法在node completion时被调用；(iii) **kernel调度层**（RELIEF调度过程，对应Figure 2b的理想调度）：t=0: DAG1的node(A1)和DAG2的node(A2)就绪，按laxity排序将A2排前面（deadline更紧）。A2被调度到acc1。t=3: A2完成→RELIEF(A2)被调用→A2的child node(B2)就绪，标记为forwarding→检查feasibility：ready_queue中A1的laxity>B2.runtime？是→B2提升到acc2队列前端。同时A1在acc1上执行。t=5: 2个节点完成——B2在acc2上完成，A1在acc1上完成。A1完成→RELIEF(A1)→A1的child(B1)标记为forwarding→检查feasibility通过→B1提升。B2完成→RELIEF(B2)→B2的child(C2)标记forwarding→检查通过→C2提升。t=7: B1完成→RELIEF(B1)→B1的child(C1)标记forwarding→通过检查→C1提升。t=10: C2在acc3上完成→RELIEF(C2)→child(D2)标记forwarding。t=12: C1完成,D2完成,t=17: DAG1 leaf完成,t=19: DAG2 leaf完成。RELIEF实现5个forwards + 2个colocations（理想调度结果），而其他SOTA策略最多实现5 forwards + 1 colocation(GEDF-D)或5 forwards + 0 colocations(FCFS/LAX)；(iv) **硬件架构层**：同baseline，但RELIEF通过调度策略有效利用了硬件转发机制——consumer driver在启动时检查producer_acc和producer_spm字段，若producer scratchpad数据仍有效（ongoing_reads跟踪未覆盖），则通过DMA直接从producer scratchpad读取（forward），而非从主存重新加载；若consumer与producer使用同一加速器类型且空闲，则colocate（完全消除数据搬移）。
  RELIEF对应解决baseline缺陷的设计：
  - (a) 应对D1+D2（调度器对forwarding oblivious + producer数据时效性）：通过"转发节点优先级提升"机制——producer完成后立即将children提升到队列前端，确保consumer在producer数据被覆盖前被调度执行。配合max_forwards限流（≤空闲加速器实例数），确保转发节点总是下一个被调度。
  - (b) 应对D3（QoS和公平性）：通过is_feasible()可行性检查——将优先级提升对deadline的影响量化（检查受影响节点的laxity是否大于candidate runtime），仅在不会导致deadline miss时才允许提升。RELIEF比HetSched减少worst-case deadline violation 14%，variance减少93%（high contention）至98%（continuous contention）。HetSched虽满足更多DAG deadline，但是以不公平slowdown一个application为代价（某application slowdown 22%），而RELIEF使所有application slowdown均衡(<7%)。
  - (c) 应对D4（RNN workload的高数据搬移开销）：RELIEF对GRU/LSTM（纯elem-matrix链式DAG，75%时间在数据搬移）效果尤为显著——通过colocation（所有节点使用同一加速器类型，consumer直接colocate在producer加速器上），RNN workload的主要数据搬移完全消除。结果：RELIEF相比HetSched实现up to 50%更多forwards，主存traffic降低up to 32%，energy consumption降低up to 18%，同时meet 14%更多deadline。

## 15-Data_Motion_Acceleration_Chaining_Cross-Domain_Multi_Accelerators

- baseline方法是什么？
  Multi-Axl多加速器系统基线：将不同领域的Domain-Specific Accelerator (DSA) 通过PCIe连接到同一host CPU，CPU负责加速器间的数据重构（data restructuring）和数据搬移（data movement）。全栈执行例子：以Video Surveillance（H.264解码→Object Detection）为例处理(960,540,3)视频帧，(i) **算法层**：应用kernel1=H.264解码（Xilinx Video Codec Unit hard-IP），kernel2=Object Detection（DNN Accelerator RTL [13]）；(ii) **系统框架/编译层**：host CPU运行OpenCL-style控制程序，顺序调度kernel，无专用数据重构编译器，数据重构在CPU上用Intel Math Kernel Library；**论文未明确说明编译框架层**；(iii) **kernel调度/运行时层**：Step1-DMA将H.264码流从CPU内存拷贝到Accelerator1 FPGA DRAM→Acc1执行解码→发中断→Step2-CPU DMA将(960,540,3)视频帧从Acc1拷贝到CPU系统内存（经PCIe switch upstream port）→Step3-CPU用Intel MKL执行Mul/MaxPool/Reshape/Cast（产生130-140短命线程，AVX-256 100%利用但cache thrashing严重，L2 MPKI 25-109）→Step4-CPU DMA拷贝到Acc2→Step5-CPU触发Acc2执行Object Detection；(iv) **硬件架构层**：Intel Xeon Platinum 8260L CPU，多个Xilinx UltraScale+ VU9P FPGA经PCIe Gen3 x16和Broadcom PEX88000 switch连接，switch upstream port仅x8单向。
  基线缺陷：(I1) 加速kernel后Amdahl瓶颈转移到数据重构——多加速场景下数据重构占总运行时间71.3%-97.1%；(I2) CPU参与数据搬移迫使所有数据经共享PCIe upstream port（x8单向上限带宽），多加速器并发时PCIe带宽严重争用；(I3) CPU通用核执行流式多维数组数据重构效率低下——每个数据batch 6-16MB远超L2 cache 1MB容量导致cache thrashing，需要130-140个线程并发，CPU成为瓶颈而非加速器。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：Data Motion Acceleration (DMX)——设计专用可编程Data Restructuring Accelerator (DRX)硬件，以Bump-in-the-Wire方式紧密耦合在每个加速器旁（经内部PCIe multiplexer连接），将数据重构从CPU offload到DRX，并通过point-to-point DMA在加速器-DRX间直接传输数据绕过CPU。全栈执行例子：同Video Surveillance，(i) **算法层**：应用kernel不变，数据重构kernel（Mul/MaxPool/Reshape/Cast）从CPU MKL迁移到DRX硬件执行；(ii) **系统框架/编译层**：DRX compiler将高级数据重构kernel描述编译为DRX ISA（Loop/Compute/Off-chip Memory/Synchronization四种指令），利用数据重构workload已知shape多维数组特性——编译器tile多维数组适配scratchpad大小，将kernel跨128 RE lane分区消除pack/unpack；**论文未明确说明系统框架层**；(iii) **kernel调度/运行时层**：Acc1完成解码→发中断→DRX1 driver找到Acc2的RX2队列offset，配置Acc1直接point-to-point DMA写视频帧到DRX1 RX2队列（绕过CPU）→DRX1 128 RE lane并行执行数据重构，结果写入TX2队列→发中断通知CPU→DRX1配置point-to-point DMA（经内部PCIe mux bypass DRX2）传数据到Acc2→Acc2执行Object Detection；(iv) **硬件架构层**：Bump-in-the-Wire DRX（每加速器一个，RTL@250MHz FPGA/1GHz ASIC，decoupled access-execute架构，128-lane RE vector unit，Transposition Engine，Off-chip Data Access Engine，64KB scratchpad替代cache，8GB DDR4），PCIe switch下游mux互联，CPU仅运行控制平面。
  DMX解决baseline缺陷的设计对应：(a) 应对I1：DRX是数据重构领域专用可编程硬件——128-lane RE并行利用数据重构的data-level parallelism，Instruction Repeater硬件loop消除分支开销，软件管理scratchpad直接寻址多维数组消除cache thrashing，数据重构开销从64.1%降至14.1%；(b) 应对I2：Bump-in-the-Wire放置将point-to-point DMA限制在PCIe switch下游（加速器↔DRX），不经过upstream port，每加速器拥有专属PCIe带宽（多加速器并行利用各自下游x16链路），消除带宽争用；(c) 应对I3：DRX硬件专为流式多维数组处理设计——Strided Scratchpad Address Calculator用<Base, Stride, Iteration>自动计算多维地址避免复杂地址计算，DRX compiler跨RE分区多维数组消除pack/unpack，scratchpad替代cache hierarchy完全消除cache thrashing。结果：5个跨领域benchmark上，1-15并发应用DMX vs Multi-Axl平均3.5×-8.2×延迟加速，3.0×-13.6×吞吐提升，3.8×-5.2×能耗降低。

## 14-Data_Motion_Acceleration_Chaining_Cross-Domain_Multi_Accelerators

- baseline方法是什么？
  Multi-Axl多加速器系统的基线方案：将不同领域的Domain-Specific Accelerator (DSA) 通过PCIe连接到同一host CPU，CPU负责所有加速器间的数据重构（data restructuring）和数据搬移（data movement）。全栈执行例子：以Video Surveillance（H.264解码→Object Detection）处理单个(960,540,3)视频帧为例，(i) **算法层**：应用kernel1=H.264解码（Xilinx Video Codec Unit hard-IP），应用kernel2=YOLOv3 object detection（DNN Accelerator RTL [13]）；(ii) **系统框架/编译层**：host CPU运行C++控制程序（OpenCL-style编程模型），顺序调度kernel执行，无专用编译器（数据重构在CPU上用Intel Math Kernel Library）；(iii) **kernel调度层**：Step1-通过DMA将H.264码流从CPU内存拷贝到Accelerator1的FPGA DRAM → Accelerator1执行解码 → 完成后发中断到CPU → Step2-CPU发起DMA将解码后的(960,540,3)视频帧从Accelerator1内存拷贝到CPU系统内存（经PCIe switch upstream port）→ Step3-CPU使用Intel MKL执行数据重构（Mul/MaxPool/Reshape/Cast，产生130-140个临时线程）→ Step4-CPU发起DMA将重构后数据拷贝到Accelerator2内存 → Step5-CPU触发Accelerator2执行Object Detection；(iv) **硬件架构层**：Intel Xeon Platinum 8260L CPU（多个core并发处理数据重构），多个Xilinx UltraScale+ VU9P FPGA加速器通过PCIe Gen3 x16连接（经过Broadcom PEX88000 PCIe switch），PCIe switch upstream port仅有x8单向链路。Baseline的缺陷：(I1) 加速器大幅缩短kernel执行时间后，Amdahl瓶颈转移到数据重构——多加速场景下数据重构占总运行时间的71.3%~97.1%，数据重构成为性能瓶颈；(I2) CPU参与数据搬移导致所有数据必须经过CPU的PCIe upstream link，多加速器共享同一upstream port造成PCIe带宽竞争和额外数据搬移开销；(I3) CPU通用核执行流式、多维数组的数据重构操作效率低下——cache thrashing严重（L2 MPKI达25~109，远高于CloudSuite在线服务<3），每次数据重构触发130-140个短命线程，CPU的AVX-256向量单元100%利用率说明CPU成为瓶颈而非向量化不足。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：Data Motion Acceleration (DMX)——设计专用的Data Restructuring Accelerator (DRX) 硬件模块，将其以Bump-in-the-Wire方式紧密集成在每个加速器旁（经内部PCIe multiplexer互联），将数据重构计算从CPU offload到DRX，同时通过point-to-point DMA在加速器-DRX间直接传输数据绕过CPU。全栈执行例子：以同一Video Surveillance为例，(i) **算法层**：应用kernel不变（H.264解码 + Object Detection），数据重构kernel（Mul/MaxPool/Reshape/Cast）从CPU迁移到DRX；(ii) **系统框架/编译层**：OpenCL-style host程序不变（控制平面仍在CPU），新增DRX compiler将高级数据重构kernel编译为DRX ISA——Loop指令（配置Base/Stride/Iteration遍历多维数组）、Compute指令（RE lane向量并行执行）、Off-chip Memory指令（DMA预取到scratchpad）、Synchronization指令；(iii) **kernel调度/运行时层**：Accelerator1完成H.264解码 → 发中断到CPU → DRX1 driver找到对应Accelerator2的RX2数据队列偏移，配置Accelerator1直接通过point-to-point DMA将视频帧写入DRX1的RX2队列（绕过CPU系统内存）→ DRX1从RX2队列读取数据，128个RE lane并行执行数据重构（Mul/MaxPool等），结果写入TX2队列 → DRX1发中断通知CPU数据重构完成 → DRX1配置point-to-point DMA（经内部PCIe multiplexer，绕过DRX2）将重构后数据传输到Accelerator2 → Accelerator2执行Object Detection；(iv) **硬件架构层**：Bump-in-the-Wire DRX（每个加速器一个）通过PCIe switch下游multiplexer与加速器互联，DRX RTL在FPGA@250MHz/ASIC@1GHz实现（decoupled access-execute架构，128-lane RE vector unit，Transposition Engine，Off-chip Data Access Engine with DMA，64KB scratchpad，64KB I-cache，8GB DDR4），CPU仅运行控制平面。DMX解决baseline缺陷的关键设计：(a) 应对I1（数据重构瓶颈）：DRX是可编程专用硬件，128-lane RE向量单元+硬件loop（Instruction Repeater）+ scratchpad替代cache hierarchy，利用数据重构workload的已知shape多维数组特性消除cache thrashing。数据重构时间从占端到端64.1%降低到14.1%；(b) 应对I2（PCIe带宽竞争）：Bump-in-the-Wire放置使point-to-point DMA局限在PCIe switch下游（加速器↔DRX间），不经过upstream port，每个加速器拥有专属PCIe带宽。15个并发应用时Multi-Axl数据搬移受限于共享upstream x8带宽，而DMX利用每个加速器专属的下游x16链路并行传输；(c) 应对I3（CPU效率低下）：DRX硬件特化——软件管理scratchpad直接通过Strided Scratchpad Address Calculator寻址多维数组（无需大容量cache和复杂地址计算）、Instruction Repeater消除循环分支开销、DRX ISA的Compute指令直接操作多维数组的RE分区数据无需pack/unpack指令。结果：5个benchmark平均3.5×~8.2×加速（1-15个并发应用），3.0×~13.6×吞吐提升，3.8×~5.2×能耗降低。

## 13-SmartDIMM- In-Memory Acceleration of Upper Layer Protocols

- baseline方法是什么？
  Baseline有三种ULP处理方案：(i) CPU处理（Intel Xeon + AES-NI指令加速加密），(ii) SmartNIC自主offload（NVIDIA ConnectX-6 TLS offload，TCP/IP栈仍在CPU），(iii) PCIe卡offload（Intel QuickAssist 8970）。全栈执行例子：以HTTPS web server加密4KB网页为例，(i) **算法层**：TLS AES-GCM，CPU使用AES-NI指令（单条指令操作立即数），SmartNIC使用autonomous TLS offload speculatively执行对称加密；(ii) **系统框架/编译层**：Nginx web server → OpenSSL库 → EVP_Cipher接口 → 内核网络栈（TCP/IP）→ NIC驱动；(iii) **kernel调度层**：DMA数据通过DDIO/DCA路由到LLC，但由于ULP处理的异步性和大working set，DMA数据在CPU利用前被evict到DRAM，产生ping-pong数据移动；(iv) **硬件架构层**：Intel Xeon Gold 6242 CPU（6×DIMM 16GB 3200MHz）+ NVIDIA BlueField-2 ConnectX-6 100Gbe SmartNIC。Baseline缺陷：高网络速率下，频繁的DRAM访问和SmartNIC-CPU同步（packet reorder/loss时硬件重同步+failback到CPU）抵消了硬件加速的收益；SmartNIC无法处理非size-preserving ULP（如压缩）；QuickAssist PCIe的notification overhead（中断/轮询）和PCIe传输开销使细粒度offload无效。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：SmartDIMM——在DIMM buffer device上实现bump-in-the-DDR近内存处理架构，通过CompCpy API在memcpy过程中同步完成ULP数据转换。全栈执行例子：以同一个HTTPS 4KB加密为例，(i) **算法层**：TLS AES-GCM不变，但CPU仅计算H（hash subkey）和EIV（encrypted IV）—各一条AES-NI指令操作立即值，DSA在SmartDIMM上执行pipelined GF multiplier计算GHASH tag和AES counter-mode stream XOR；(ii) **系统框架/编译层**：Nginx → 修改的OpenSSL AES-GCM engine（采样LLC miss rate决定offload/onload，配置contention threshold）→ CompCpy API（注册sbuff/dbuff为加速范围，写入key/IV/context到MMIO Config Memory，flush sbuff到DRAM，调用memcpy触发offload）→ SmartDIMM driver（字符设备映射物理内存到内核虚拟地址）；(iii) **kernel调度层**：不涉及独立kernel调度，CompCpy是同步API——CPU memory controller发出rdCAS从DRAM读sbuff数据 → SmartDIMM Arbiter在buffer device拦截数据送DSA → DSA结果暂存Scratchpad → LL C writeback dbuff时触发Self-Recycle（替换wrCAS data为Scratchpad数据并invalidate cacheline）；(iv) **硬件架构层**：Samsung AxDIMM FPGA原型（buffer device集成DDR PHY、MIG PHY、Arbiter、Translation Table/3-ary cuckoo hash、Scratchpad/SRAM、Config Memory、TLS DSA和Deflate DSA），运行在1/4 DRAM时钟频率。SmartDIMM解决baseline缺陷的关键设计：(a) 消除DRAM ping-pong——数据从DRAM读出即被DSA处理，dbuff直接写回DRAM（被offload的数据不再经过LLC），消除cache thrashing；(b) 同步offload无需CPU-DIMM同步——CompCpy是同步memcpy，DSA处理延迟<1μs（rdCAS到wrCAS的时间budget），无需轮询/中断；(c) 自适应offload——软件栈周期性探测LLC contention，仅在LLC miss率高时才offload，低contention时仍用CPU（避免额外延迟）；(d) CompCpy复用已有memcpy——不引入额外数据拷贝，不改变TCP/IP软件栈，order参数支持有序处理非size-preserving ULP（如压缩）。结果：4KB TLS offload比CPU高21.0% RPS，降低49.1% memory BW；4KB压缩offload比CPU高6.09× RPS，降低88.9% CPU利用率。

- baseline方法是什么？
  UPEA（Uniform Processing-Element Access）SDA：所有PE到内存的访存延迟统一（如2-cycle），通过全局fabric-memory network仲裁所有LS PE的访存请求。全栈执行例子：以spmspv的∩（intersection）操作为例，(i) **算法层**：spmspv的CSR格式intersection，使用stream-join实现co-iteration；(ii) **系统框架/编译层**：编译器生成DFG，PnR不考虑NUPEA域，所有load指令随机或均匀放置在LS PE上；(iii) **kernel调度层**：论文未明确说明（无独立kernel runtime，dataflow直接执行）；(iv) **硬件架构层**：Monaco 12×12 fabric，所有LS PE通过统一延迟（+N fabric cycles）的UPEA fabric-memory network访问32×banked memory。UPEA的缺陷：随着fabric scale up，统一延迟成为瓶颈——关键路径上的load（如iAnz recurrence上的LD）被迫承受与其他非关键load相同的延迟，导致loop initiation interval变长，整体吞吐下降（spmspv在2-cycle UPEA下比0-cycle理想情况慢24%）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：NUPEA（Non-Uniform Processing-Element Access）——在SDA中显式暴露不同PE到内存的非均匀延迟（NUPEA域D0/D1/D2/D3），编译器识别critical loads并将其优先放置在快速域（D0，0-cycle延迟）。全栈执行例子：以同一个spmspv为例，(i) **算法层**：不变，仍为CSR intersection via stream-join；(ii) **系统框架/编译层**：effcc编译器新增critical load analysis pass——分析DFG中loop recurrence（如iAnz的def-use chain上的LD指令），标记为最critical (a)，inner-loop的LD标记为次critical (b)。PnR时按优先级 D1.c0 ≺ D0.c2 ≺ D0.c1 ≺ D0.c0 将critical loads放入D0域，inner-loop loads放入D0/D1，其余loads放入更远域；(iii) **kernel调度层**：论文未明确说明；(iv) **硬件架构层**：Monaco fabric-memory NoC实现分层arbitration——D0域LS PE直连memory port（0 fabric-memory NoC延迟，高带宽），D1-D3域通过fanout-4 imbalanced arbitration tree逐级增加1 cycle延迟。NUPEA的关键洞察：编译器识别critical load远比识别critical data容易（不需要复杂的alias analysis），且SDA天然支持per-instruction placement。通过将稀缺的低延迟PE资源精确分配给真正关键的load指令，NUPEA在spmspv上达到UPEA0理想性能的99%，总体28%优于UPEA2，20%优于NUMA-UPEA2。

## 18-SmartMem- Layout Transformation Elimination and Adaptation for Efficient DNN Execution on Mobile

- baseline方法是什么？
  SOTA DNN执行框架（MNN [32]、NCNN [50]、TFLite [1]、TVM [10]、DNNFusion [51]）在移动GPU上的执行方式。全栈执行例子（以Swin Transformer中Conv→Reshape→Transpose→MatMul的计算片段为例）：(i) **算法层**：Swin Transformer模型结构不变，利用local attention（window-based self-attention）降低计算复杂度，但引入了大量Reshape/Transpose等layout transformation操作来分割/重组数据窗口；(ii) **系统框架/编译层**：MNN/NCNN/TFLite使用fixed-pattern fusion（仅融合预定义的特定operator组合），TVM将operator分为三类（Layout agnostic/lightly-layout sensitive/heavily-layout sensitive）进行通用规则融合，DNNFusion使用operator分类和组合规则生成更广泛的fusion plan。但是所有这些框架都无法消除layout transformation——Reshape和Transpose仍然作为独立kernel执行，在计算图中占据大量operator；(iii) **kernel调度层**：Reshape kernel复制tensor并改变维度元数据（1D buffer中几乎零开销，仅修改shape/strides；但在2.5D texture memory中需实际重排数据位置）。Transpose kernel在高内存带宽上读取输入tensor、执行维度置换、写回重新排列的输出tensor——每个Transpose需至少一次global memory round-trip。以[2,256,4]→[16,8,4,4]→[16,4,8,4]为例，Reshape+Transpose两次global memory访问完全无计算产出。随后Conv kernel和MatMul kernel各自读取可能不连续的数据，cache效率低下；(iv) **硬件架构层**：Qualcomm Adreno 740 GPU，含1D buffer memory（55 GB/s）和2.5D texture memory（511 GB/s）。Baseline framework通常将tensor放在1D buffer中，未能充分利用高速texture memory的2D spatial locality优势。
  Baseline核心缺陷：(D1) Layout transformation开销严重——Table 1显示Transformer模型43%-70%的执行时间花在implicit+explicit layout transformation上，且模型执行速度（GMACS）比传统CNN低约一个数量级；(D2) 连续layout op破坏数据局部性——Reshape/Transpose使内存访问碎片化，降低后续compute operator的cache命中率和SIMD效率；(D3) 缺乏系统的layout选择机制——在大型DAG计算图中，不同operator有不同layout偏好，现有框架无法协同选择跨operator的layout以消除transformation；(D4) 移动GPU的2.5D texture memory未被充分利用——现有框架主要使用1D buffer，texture memory带来的3.5× latency reduction机会被浪费。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：SmartMem——基于operator四象限分类的layout transformation消除与适配框架。全栈执行例子（以同一Swin Transformer片段Conv→Reshape→Transpose→MatMul为例）：(i) **算法层**：模型结构不变，但不再有独立的Reshape/Transpose operation，所有layout适配在编译期解决；(ii) **系统框架/编译层**：SmartMem在DNNFusion之上新增三步——① Operator Classification：Conv→ILD-Variable（性能依赖input layout，output layout可定制），Reshape/Transpose→ILD-Fixed，MatMul→ILD-Variable；② Layout Transformation Elimination（LTE）：查Table 5的16种producer-consumer分类决策矩阵——Conv(ILD-Var)+Reshape(ILD-Fixed)→Eliminate 2nd(Reshape)，Reshape(ILD-Fixed)+Transpose(ILD-Fixed)→Eliminate both，最终仅保留Conv和MatMul两个ILD-Variable operator。消除通过Index Comprehension实现——将[2,256,4]→[16,8,4,4]→[16,4,8,4]的索引变换链替换为fused kernel内部的简化索引计算（identity/split/merge依赖→Strength Reduction消除冗余取模/除法），无需执行独立的Reshape/Transpose kernel；③ Reduction Dimension-based Layout Selection：MatMul的reduction dimension=k→强制Conv输出tensor沿k维连续存储，使MatMul kernel沿k维做SIMD reduction（stride=1 load）。同时将k和其他consumer reduction dimension映射到2.5D memory的两维连续访问方向；(iii) **kernel调度层**：不再有Reshape/Transpose kernel——fused Conv kernel直接输出已按MatMul reduction dimension优化layout的tensor（写入时按reduction dim组织，写放代价低于子优化读取），MatMul kernel以vector load方式从2.5D texture memory沿reduction dim连续读取数据进行SIMD dot product累加。Genetic Algorithm auto-tuning决定block dimension/unrolling/tiling shape；(iv) **硬件架构层**：同baseline Adreno 740 GPU，但SmartMem的核心kernel使用2.5D texture memory——利用其多维寻址能力和自动边界检查减少索引计算开销，利用其专用cache提升2D spatial locality。同时减少manager从1D buffer到texture memory的不必要数据搬移。
  SmartMem解决baseline缺陷的设计对应：(a) 应对D1（layout transformation开销严重）：LTE通过operator分类矩阵系统性地决定哪些layout op可被消除，消除后通过Index Comprehension将layout变换替换为fused kernel内部的简化索引计算。结果：operator数量减少21%-65%，Swin上DNNFusion的135个op降至SmartMem的30.6个；(b) 应对D2（数据局部性破坏）：LTE消除碎片化的Reshape/Transpose访问后，Layout Selection按consumer reduction dimension组织producer输出——数据沿reduction dim连续存储，后续compute kernel以stride=1顺序访问，提升cache命中率（cache miss平均减少2.0×）和SIMD效率；(c) 应对D3（缺乏系统layout选择）：Reduction Dimension-based heuristic——对ILD-Variable类型operator的每条producer-consumer边，按consumer的reduction dimension决定producer应生成的输出layout。多consumer场景合并前k个reduction dimension（k=2 for 2.5D memory），超出k则维护多份数据副本（runtime冗余副本通常<3MB）。此heuristic将NP-hard的全局layout选择问题退化为局部贪心+多consumer合并，在可接受的开销下实现有效优化；(d) 应对D4（2.5D texture memory未利用）：SmartMem将reduction dimension映射到2.5D memory的两维连续访问方向——沿一个reduction dim按4元素partition（匹配0.5D vector宽度），4×沿另一个reduction dim连续存储，使consumer的SIMD load沿任意一个reduction dim均可高效访问。同时优化数据访问顺序消除不必要的数据搬移。结果：SmartMem相比DNNFusion在Transformer/Hybrid模型上1.8×-5.0×加速，在ConvNet上1.2×-3.3×加速，总体平均2.8× speedup over DNNFusion。

## 17-MAGIS- Memory Optimization via Coordinated Graph Transformation and Scheduling for DNN

- baseline方法是什么？
  DNN内存优化的SOTA方法以图调度（Graph Scheduling）为主，包括三类技术：(1) Re-materialization [DTR, Checkmate, XLA]：evict中间tensor后在backward需要时重新计算，以重计算开销换取显存；(2) Swapping [POFO, vDNN, Capuchin]：将暂不用的tensor offload到CPU内存，需要时reload回GPU，以PCIe传输开销换取显存；(3) Re-ordering [Serenity]：在不改变operator集合的前提下调整执行拓扑排序，减少同时活跃的tensor数。这些方法只能操控tensor的生命周期（何时compute/evict/recompute/offload/reload），不能改变tensor的形状。图变换（Graph Transformation）方法如TASO、PET、TenSAT主要用于延迟优化（通过A-Trans聚合小op提升硬件利用率），未被用于内存优化。全栈执行例子（以U-Net batch=32训练为例）：(i) **算法层**：U-Net模型结构不变（encoder下采样→decoder上采样→skip connection拼接），forward保存所有激活供backward使用；(ii) **系统框架层**：PyTorch baseline——按拓扑序执行operator，每个operator完成后检查输出tensor是否还有未来consumer，若没有立即释放。DTR/XLA/POFO在此之上添加各自的re-mat/swapping/re-ordering策略，论文未明确说明系统框架层的统一抽象；(iii) **编译框架层**：DTR在MegEngine eager模式下通过heuristic运行时策略动态决定evict哪些tensor、何时重计算；XLA通过HLO-level贪心算法预先规划re-materialization；POFO用DP离线求解re-mat和swapping的最优组合；(iv) **kernel调度层**：GPU上operator按调度顺序依次执行——Swapping通过cudaMemcpy在GPU↔CPU间异步传输tensor、Re-materialization重执行部分operator重新生成中间tensor、Re-ordering调整拓扑排序但不影响总延迟。若同时使用re-mat和swapping，re-mat产生的额外operator和swap的Store/Load均参与重排序；(v) **硬件架构层**：NVIDIA RTX 3090 GPU (24GB) + Intel Xeon CPU，GPU显存与CPU内存通过PCIe Gen3总线传输。Baseline的核心缺陷：(D1) 图调度（swapping/re-mat）引入显著延迟开销——PCIe数据传输带宽远低于显存带宽（~16GB/s vs ~936GB/s），重计算浪费已完成的算力——memory-performance trade-off斜率较差；(D2) 图调度只能操控tensor生命周期，不能改变tensor shape，优化空间受限于tensor原始形状。例如U-Net中skip-connection的中间tensor形状固定为(batch, channels, H, W)，即使通过re-ordering和swapping也无法减小其size；(D3) 现有图变换（TASO/PET/TenSAT）仅关注延迟优化——A-Trans聚合小op为大局以提升硬件利用率（反而增加临时内存占用），Fission Transformation虽能通过分裂op减小tensor shape降低内存，但缺少有效的表达、搜索和在内存优化中的运用方法；(D4) 图变换和图调度独立应用——图变换的memory-performance trade-off（A-Trans trade memory for perf, F-Trans trade perf for memory）与图调度的trade-off本质上相互关联，但现有工作从未协同优化两者，导致在Pareto边界上留下大量优化空白。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：MAGIS——通过协同图变换（Graph Transformation）与图调度（Graph Scheduling）进行DNN内存优化的框架。核心设计分四层：(1) Fission Transformation (F-Trans)：定义为f=(S, D, n)，将凸子图S沿D-Graph中维度D分裂为n个顺序执行的split parts，各part中间tensor可及时释放以降低峰值内存（降低至约1/n），代价是每个part的operator shape变小导致硬件利用率降低；(2) Fission Hierarchy Tree (F-Tree)：层级树抽象表达F-Trans后的图结构——每个F-Tree节点记录(S, D, n)，n=1表示候选节点（未分裂），n>1表示已启用节点（已分裂为n部分）。F-Tree避免了直接变换图的复杂度指数增长，且将所有已启用的F-Trans按层级组织（父节点子图包含子节点子图）；(3) M-Analyzer：基于dominator tree和memory hot-spot分析构造轻量F-Tree——对每个D-Graph连通分量，计算各节点的memory heat（其支配的热点tensor总size）和score（≈heat×(1-1/n)-Σinputs，估算F-Trans后的峰值内存减少量），选择不同score区间的dominator节点生成F-Tree候选节点；(4) M-Rules统一搜索空间：将re-materialization和swapping分解为4条Scheduling-based Rules（Re-mat/Swapping Rule及对应De-Rules），与F-Tree变异规则（Enabling/Lifting/Disabling/Mutating）和TASO规则（A-Trans/I-Trans）统一在M-Rules中表达——所有规则本质上都是子图替换（sub-graph substitution），将memory-performance trade-off完全移入图变换阶段，调度阶段仅需做不影响总延迟的重排序。全栈执行例子（以U-Net batch=32训练，memory constraint=60% PyTorch peak）：(i) **算法层**：U-Net模型不变；(ii) **系统框架层**：MAGIS作为独立优化框架，接受PyTorch eager模式导出的计算图，输出优化后的图和调度，代码生成后端将其转换为可执行的PyTorch Python代码；(iii) **编译框架层**：M-Analyzer识别U-Net encoder/decoder中dominator tree上的memory hot-spot → 计算各节点score → 选择dominator节点构造F-Tree（沿batch-dim分裂conv block）→ M-Optimizer贪心搜索：Enabling Rule将F-Tree叶子节点n从1改为2（每个conv block沿batch-dim分裂为2个split part，各处理batch=16，中间结果及时释放）→ Swapping Rule在skip-connection附近插入Store(tensor→CPU)/Load(CPU→tensor) → Re-mat Rule对decoder中某些无需长期保存的activation标记重计算 → TASO A-Trans融合相邻小op→增量调度（Algorithm 2）仅对变换影响子图重排序→simulator评估→push回优先队列继续搜索；(iv) **kernel调度层**：最终调度——encoder split part 1执行（batch=16的conv forward+backward）→中间结果释放→split part 2顺序执行→skip connection tensor在forward完成后Store到CPU（cudaMemcpyAsync）→decoder backward需要时Load回GPU（异步，时机经DpSchedule计算刚好隐藏PCIe延迟）→标记重计算的operator在backward中重新执行；(v) **硬件架构层**：NVIDIA RTX 3090 GPU + Intel Xeon CPU，同baseline，但通过F-Trans减小每个operator处理的batch size（峰值内存降低约50%），通过异步swapping隐藏PCIe传输延迟，通过re-mat减少需长期驻留显存的tensor数量。MAGIS解决baseline缺陷的设计对应：(a) 应对D1+D4（调度延迟开销大 + 变换与调度未协同）：MAGIS将re-mat/swapping分解为图变换，与F-Trans/TASO统一在M-Rules中搜索，将memory-performance trade-off在一个统一的优先队列搜索空间中进行协同优化。增量调度高效评估每次变换后的新图调度质量（4-30× faster than full scheduling），使协同搜索在3分钟时间预算内可行；(b) 应对D2（只能操控lifetime不能操控shape）：F-Trans直接改变tensor shape——通过沿维度D将子图分裂为n部分，每个split part的中间tensor shape缩小至1/n，执行后被及时释放，峰值内存大幅降低（U-Net从1056降至98 units），突破了图调度仅操控lifetime的限制；(c) 应对D3（F-Trans缺少有效表达和搜索）：F-Tree层级抽象避免了图分裂后复杂度指数增长（不实际变换图，记录分裂信息，执行时按需展开）；M-Analyzer基于dominator tree和memory hot-spot score筛选有潜力的子图和维度，将搜索空间从O(2^{|V|²})缩减至可控的F-Tree节点集合；F-Tree变异规则从叶子→根逐步启用（从小步smooth search）。结果：相比SOTA（POFO, DTR, XLA），MAGIS在相同延迟约束下仅需15%-85%的峰值内存，在memory-latency Pareto边界上全面占优。

## 86-Oaken- Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization.pdf

- baseline方法是什么？
  Baseline是FP16精度的标准LLM serving系统（以vLLM为代表），KV cache以FP16存储。全栈执行例子（以Llama2-7B处理batch=64, context=2K的单个生成step为例）：(i) **算法层**：标准的Multi-Head Attention计算——Q·K^T attention score，无任何KV量化；(ii) **系统框架层**：vLLM serving框架，使用PagedAttention进行page-based KV cache管理。Scheduler将多个用户请求组成batch→prefill阶段计算所有prompt token的K/V并以FP16写入KV cache pages→generation阶段每步从KV cache读取FP16 K/V计算attention，新生成的token的K/V追加写入；(iii) **编译框架层**：论文未明确说明；(iv) **kernel调度层**：NVIDIA A100 GPU上，attention kernel从HBM读取FP16 K/V（2.0 TB/s带宽），计算Q·K^T。Attention操作为bandwidth-bound——batch size增大时，共享的weight部分（QKV Gen, FFN）利用率提升，但attention的K/V读取不共享，随batch增大KV cache data movement线性增长，GPU利用率低（~20% for single request, ~22% for batched requests）；(v) **硬件架构层**：NVIDIA A100 GPU（312 TFLOPS FP16, 80GB HBM, 2.0 TB/s bandwidth）。Baseline的核心缺陷：(D1) **内存容量瓶颈**：FP16 KV cache每token每层占用2×d_head×2 bytes，大batch + 长context时KV cache迅速耗尽HBM（如Llama2-13B batch=128 context=2K KV cache = 128×2048×40×2×128×2 ≈ 5.4GB，仅KV cache已占HBM的相当比例），限制serving系统的最大batch size和context length；(D2) **内存带宽瓶颈**：attention操作需从HBM读取KV cache，batch中各请求的K/V不共享（un-sharable），随batch增大bandwidth需求线性增长——attention始终bandwidth-bound，即使weight计算（QKV Gen, FFN）随batch增大利用率提升；(D3) **现有KV量化方案精度-性能权衡差**：KIVI采用per-channel量化但使用coarse-grained grouping导致精度损失大；QServe使用fine-grained per-vector mixed-precision但online profiling引入overhead且混合精度增加硬件复杂度；Tender使用channel reordering/matrix transformation但硬件效率低。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：Oaken——算法-硬件协同设计的online-offline混合KV cache量化方案。全栈执行例子（以同一Llama2-7B batch=64, context=2K的generation step为例）：(i) **算法层**：Oaken threshold-based hybrid grouping量化算法——offline profiling（一次性~10分钟，~100 inferences on WikiText2）确定每layer的T_lo/T_hi阈值，将KV值分为outer(4%)/middle(90%)/inner(6%)三组。Online per-token量化：outer使用Group Shift INT5（12 bit/entry vs FP16 23 bit/entry），middle使用INT4（4 bit/entry），inner zero-fill（sparse, 0 bit/entry）。Fused Dense-and-Sparse Encoding融合编码为8-bit对齐格式（8 bit/entry average）。最终average bitwidth 4.4 vs FP16 16——3.6×压缩；(ii) **系统框架层**：Oaken serving系统集成page-based memory management + Quant/Dequant Engine硬件。Prefill阶段K/V经Quant Engine在线量化后写入KV cache pages（4.4 bit/entry），MMU management table记录每page的量化元数据（thresholds, scales, group config per layer）。Generation阶段每步从KV cache经Dequant Engine反量化读取FP16 K/V计算attention——量化/反量化与compute/memory传输pipeline重叠；(iii) **编译框架层**：论文未明确说明；(iv) **kernel调度层**：Oaken加速器（270 TFLOPS, TSMC 28nm）的Quant Engine（1.86%面积）和Dequant Engine（6.35%面积）以pipeline方式集成在Compute Core中。Quant Engine pipeline：Threshold Comparator→Scale Calc→Quantizer→Splitter→OR+Shifter→DMA write。Dequant Engine pipeline：DMA read→Decomposer→Dense/Sparse decode→Min&Max→Scale Calc→Dequantizer→MPU。Attention从HBM读取的数据量降至约1/3.6，从bandwidth-bound转为compute-bound或接近平衡；(v) **硬件架构层**：Oaken-HBM（270 TFLOPS, 80GB HBM, 2.0 TB/s）和Oaken-LPDDR（270 TFLOPS, 256GB LPDDR, 1.1 TB/s）两种配置。LPDDR配置以更大容量（256GB）补偿较低带宽（1.1 TB/s），在large model + large batch场景下为competitive选项。Oaken模块不修改加速器原有compute逻辑。Oaken解决baseline缺陷的设计对应：(a) 应对D1（内存容量瓶颈）：4.4-bit KV cache将有效内存容量扩展约3.6×——相同物理内存可容纳更大batch或更长context。例如80GB HBM在FP16下KV cache capacity约可容纳batch=256 context=2K的Llama2-7B，而Oaken 4.4-bit可容纳batch≈900同等context；(b) 应对D2（内存带宽瓶颈）：KV cache数据量降至1/3.6→attention的data movement需求同比例降低→attention从bandwidth-bound转向compute-bound（或更接近平衡点），batch增大时GPU利用率显著提升。Llama2-7B batch=256时Oaken-HBM throughput 1.79× over vLLM FP16；(c) 应对D3（现有量化方案精度-性能权衡差）：① Threshold-based hybrid grouping基于三个insight设计——KV分布跨layer变化（per-layer阈值）、跨数据集一致（offline profiling可复用）、存在channel-wise例外（多组magnitude分段），相比KIVI的coarse grouping精度更高（Oaken 0.87% accuracy loss vs FP16, KIVI 1.43% loss）；② Group Shift Quantization将outlier组通过shift缩小动态范围后以INT5量化（vs QServe的混合精度增加硬件复杂度），average bitwidth 5.9→4.8；③ Fused Dense-and-Sparse Encoding利用inner group的稀疏性（10% sparsity）将编码融合为硬件友好的8-bit对齐格式，4.8→4.4 bitwidth，且8-bit对齐避免非对齐内存访问的硬件开销。Overall：Oaken accuracy (avg 0.87% loss)优于QServe (avg 1.38% loss vs KIVI)，与KIVI接近（0.32% gap）但Oaken有专用硬件加速dequant。硬件overhead仅8.21%面积（Quant 1.86% + Dequant 6.35%），换来throughput 1.79×提升和3.6×有效容量扩展。

## 20-PIM-DL- Expanding the Applicability of Commodity DRAM-PIMs for Deep Learning via Algorithm-System Co-Optimization.pdf

- baseline方法是什么？
  Baseline是PIM-Enabled System（现有的基于DRAM-PIM的DNN加速方案），通过commodity DRAM-PIM（UPMEM PIM-DIMM、HBM-PIM、AiM）加速element-wise和GEMV等memory-bound算子，但GEMM（General Matrix Multiplication，占端到端延迟>85%的compute-heavy算子）仍被offload到powerful host CPU/GPU上处理。全栈执行例子（以BERT-base batch=64 seq_len=512在UPMEM PIM-DIMM上的inference为例）：(i) **算法层**：标准BERT-base模型，所有linear layers（QKV projection, FFN1, FFN2, O projection）使用GEMM计算——输入activation (N=32768, H=768) × weight matrix (H=768, F=3072 for FFN1)，FLOPs = 2×N×H×F；(ii) **系统框架层**：PIM-Enabled System——GGML tensor library (AVX/AVX2 intrinsics) 在dual-socket Intel Xeon 4210 CPUs上执行GEMM算子（QKV/FFN1/FFN2/O projection）；UPMEM PIM-DIMM仅被offload element-wise算子（ReLU, LayerNorm, Residual Add等，<15%总延迟）。PIM driver负责host→PIM数据传输和kernel launch；(iii) **编译框架层**：论文未明确说明；(iv) **kernel调度层**：Host CPUs使用MKL-like AVX INT8 GEMM（GGML INT8 with AVX2 simd）执行linear layers；PIM-DIMM PEs（RISC cores @350MHz）执行element-wise kernel——UPMEM ISA实现逐元素操作。由于UPMEM PIM-DIMM peak throughput仅43.8 GOP/s per DIMM（远低于host CPU的795.11 GOPS），compute-heavy GEMM无法在PIM上执行，PIM利用率极低；(v) **硬件架构层**：Dual-socket Xeon 4210 + 8× PIM-DIMMs（1024 PEs total）。Baseline核心缺陷：(D1) **DRAM-PIM计算能力极度受限**：UPMEM PIM-DIMM peak throughput 43.8 GOP/s per DIMM，HBM-PIM 1.2 TFLOPS per cube，AiM 1 TFLOPS per chip——均远低于DNN serving所需的>10 TOP/s computational capacity。原因：DRAM-PIM使用DRAM process制造compute unit，晶体管慢3×，逻辑密度低数倍，金属层少导致routing density低；(D2) **DRAM-PIM在DNN中的应用范围太窄**：GEMM占DNN端到端延迟>85%，但DRAM-PIM仅能加速element-wise和GEMV（<15%延迟+仅单batch GPT/LSTM场景），导致DRAM-PIM利用率低（大量计算仍在host），无法motivate data-center adoption；(D3) **现有LUT-NN算法精度不足**：baseline LUT-NN [84]尝试用LUT替换GEMM——将weight matrix转换为codebooks（聚类centroids）+ LUTs（pre-computed inner-products），推理时用centroid index查LUT累加代替矩阵乘法。但baseline LUT-NN仅能替换BERT-base的6/12层（partial replacement），全层替换时accuracy大幅下降（BERT-base GLUE avg从79.0降至35.5，90.44% CV accuracy drop），且需100%训练集calibration；(D4) **DL框架不支持DRAM-PIM backend**：Tensorflow/PyTorch/GGML等框架不支持UPMEM PIM-DIMM等commodity DRAM-PIM作为推理backend，缺乏统一的offloading机制将PIM-friendly ops分配到PIM、其余ops分配到host；(D5) **LUT-NN映射到PIM的性能挑战**：DRAM-PIM存在constrained host-PIM communication（PEs共享bus，broadcast带宽最优但cache miss导致瓶颈）、no direct inter-PE datapath（PE间需通过host forwarding交换数据）、load-balancing（PIM kernel在数千PE上执行，最慢PE决定finish time）三大架构限制，使得LUT-NN的algorithmic优势难以转化为实际加速。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：PIM-DL——基于Algorithm-System Co-Optimization的全栈框架，通过eLUT-NN算法将所有linear layers的GEMM替换为LUT-based inference，并设计PIM推理引擎和Auto-Tuner将LUT-NN高效映射到commodity DRAM-PIM上执行。全栈执行例子（以同一BERT-base batch=64 seq_len=512在UPMEM PIM-DIMM上的inference为例）：
  (i) **算法层**：eLUT-NN calibration——将FFN1的weight (768×3072)转换为VH=192个codebooks（每codebook含CT=16个1×4 centroids，通过K-means聚类calibration数据中activation sub-vectors得到）和CT=16个LUTs（每LUT shape=F×VH=3072×192，通过centroids×weight sub-vector inner-product预计算）。Calibration引入Reconstruction Loss（L = Model Loss + β Σ||Â_l W - A_l W||²）和STE gradient propagation，仅需<1% calibration data即可全层替换，BERT-base accuracy从baseline LUT-NN 35.5恢复至76.9（接近original 79.0, 仅-2.1%）。Inference中multiplication仅占LUT-NN总FLOPs的2.9%-14.3%——GEMM中2×N×H×F的一半乘法被LUT lookup取代；
  (ii) **系统框架层**：PIM-DL Inference Engine——Frontend Framework包含Host operators（CCS operator实现为GGML GEMM——N×VH × VH×CT的small GEMM, 远小于原始N×H × H×F）和PIM operators（LUT lookup+reduction kernel）。Backend Library包含Host library（MKL/OneDNN, CuBLAS/CuDNN）和PIM Runtime（UPMEM driver + PIM binary launch + result fetch）。Host→PIM offloading策略：CCS（少量GEMM）在host执行，LUT（memory-intensive lookup+accumulation）offload到PIM，其他element-wise算子根据PIM功能可选offloading；
  (iii) **编译框架层**：论文未明确说明；
  (iv) **kernel调度层**：Sub-LUT Partition将LUT workload按 (N, F)维切分为tiles分配给PE group——PE group_i共享index tile_i，跨group同位置PE_j共享LUT tile_j。每个PE独立计算(N_{s-tile}, F_{s-tile}) output tile，无inter-PE通信（CT dim不分割，无需partial sum merging）。Micro Kernel在PE on-chip buffer (64KB) 上执行：按(N_{m-tile}, F_{m-tile}, CB_{m-tile})进一步tile→加载index MTile→traverse CB_{m-tile} blocks检索LUT→accumulate→store output MTile。Auto-Tuner搜索三层参数：(a) sub-LUT tiling (N_{s-tile}, F_{s-tile})——影响host-PIM通信模式和带宽利用；(b) micro kernel tiling (N_{m-tile}, F_{m-tile}, CB_{m-tile})——影响on-chip buffer allocation和数据复用；(c) LUT load scheme（static/coarse-grain/fine-grain）——根据on-chip buffer capacity和PE threadlet数量选择最优load策略；(d) tile traversal order——影响tile reuse pattern。Auto-Tuner用analytical model估计t_sub-lut (host-PIM comm) + t_micro-kernel (transfer + reduce)，搜索空间总候选数可控（offline search, ~1s/model）；
  (v) **硬件架构层**：PIM-DL本身不修改硬件。在UPMEM PIM-DIMM (1024 PEs@350MHz, 350 GOP/s total) 上实现LUT-NN推理。RISC PE执行ISA实现的LUT kernel——per PE: load index MTile from local DRAM bank→issue read request根据index值fetch LUT entries→accumulate到output buffer→store result。Roofline analysis显示LUT operator arithmetic intensity 0.204-0.288 GOPs/Byte，完全落在memory-bound region，充分利用PIM的高带宽而非弱计算。HBM-PIM/AiM上通过PIMSimulator模拟验证可扩展性。

PIM-DL解决baseline缺陷的设计对应：
- 应对D1（DRAM-PIM计算能力极度受限）：eLUT-NN将GEMM转换为LUT lookup+accumulation，PIM侧multiplication几乎为零（仅host侧CCS有少量乘法）。GEMM的2×N×H×F中一半乘法被memory lookup替代，PIM侧operator完全变为memory-intensive——match DRAM-PIM的高带宽、弱计算特征（UPMEM PIM-DIMM 80.4 GB/s bandwidth vs 43.8 GOP/s compute），而非硬撑compute-heavy GEMM。Adder-only PIM design（Section 7讨论）——若将PIM PE的多余multiplier替换为更多adder，可进一步提升性能。
- 应对D2（DRAM-PIM在DNN中应用范围窄）：PIM-DL将all linear layers（QKV/O projection + FFN1/FFN2，占Transformer >85% 延迟）替换为LUT-based计算并offload至PIM。Figure 11-(a)显示LUT operator占总延迟51.52%~60.41%（vs PIM-Enable System中PIM仅处理<15%的element-wise ops），DRAM-PIM利用率显著提升。BERT-base ViT-base BERT-large ViT-huge均支持全层替换；
- 应对D3（现有LUT-NN精度不足）：eLUT-NN的Reconstruction Loss通过direct gradient propagation（避免layer-by-layer backprop的gradient vanishing）和computation error纳入loss function（centroids学习更准确的激活表示），结合STE替代Gumbel-Softmax加速收敛，仅需<1% calibration data即实现全层替换且accuracy接近original。BERT-base GLUE avg 79.0→76.9 (-2.1%) vs baseline LUT-NN 35.5；
- 应对D4（DL框架不支持PIM backend）：PIM-DL Engine提供统一的frontend-backend分离架构——Frontend Framework定义Host/PIM operators接口，Backend Library封装UPMEM SDK/HBM-PIM/AiM driver。Operator offloading决策通过PIM-DL framework自动路由（LUT→PIM, CCS→Host, others→by target PIM capability）；
- 应对D5（LUT-NN映射到PIM性能挑战）：Sub-LUT Partition通过index/LUT tile broadcast复用（同group共享index，跨group共享LUT）解决constrained host-PIM communication bottleneck；CT dim unpartitioned + PE group逻辑组织消除inter-PE通信需求；even tiling确保load balance。Auto-Tuner自动搜索最优(N_{s-tile}, F_{s-tile})、micro kernel参数和LUT load scheme——search仅~1s/model，average estimation error 3.44%，tuned params实测仅≤6% degradation vs best-known。最终结果：vs GEMM-based PIM inference，PIM-DL 22.57×/37.06×/27.25× speedup on PIM-DIMM/HBM-PIM/AiM；vs CPU/GPU, up to 3.54×/1.20× speedup。

## 23-Mind the Gap: Attainable Data Movement and Operational Intensity Bounds for Tensor Algorithms

- baseline方法是什么？
  Baseline是**Algorithmic-minimum access bound**——即所有input和output operand sizes之和（equivalent to compulsory misses for caches）。Hardware architects在早期DSE中用此做"speeds and feeds"分析和roofline performance model的operational intensity（OI=compute/data_movement）输入。
  
  全栈执行例子（以4k×4k×4k GEMM在NVIDIA A100 GPU上的标准DSE流程为例）：
  - **算法层**：4k×4k×4k GEMM，operand sizes: A=16M elements, W=16M elements, B=16M elements（FP16, 96MB total）。算法层仅提供algorithmic OI = 4096×4096×4096×2 FLOPs / (16M+16M+16M)×2 bytes = 137G OP / 96MB ≈ 1426 OP/B。
  - **系统框架层**：基于此algorithmic OI做roofline分析。A100 HBM bandwidth=2039 GB/s（≈2190 OP/B at FP16 compute 312 TFLOPS）。Algorithmic OI 1426 < 2190 → 判定为memory-bound。因此architect决定需高memory bandwidth而非更多MAC。但这个分析忽略buffer capacity effect。
  - **编译框架层**：论文未明确说明baseline DSE中使用的具体mapping/compiler框架。典型的compiler如CUTLASS/TVM对给定GEMM做loop tiling和schedule optimization，但baseline bounds analysis不涉及这些。
  - **kernel调度层**：Algorithmic-minimum不涉任何具体的tiling、parallelization或schedule信息。它假设所有operand仅被读取1次。实际A100 GPU上4k GEMM的DRAM实测traffic为algorithmic-minimum的6.5×，L2-to-L1 traffic为32.3×——因为实际kernel mapping需要tile reused data，受限于buffer capacity无法实现完美复用。
  - **硬件架构层**：Architect基于algorithmic-minimum info做hardware provisioning决策（例如，96MB total operand size → 认为几十MB buffer就足够实现optimal reuse）。但实际maximal effectual buffer size（实现所有reuse的最小buffer）为~32MB（smallest operand size），而完全消除Gap 1需要的buffer远大于此。Architect受algorithmic-minimum误导可能严重under-provision buffer。

  Baseline的三大缺陷（Gap问题）：
  - **Gap 0（Access Gap）**：algorithmic-minimum与实际可达data movement之间存在数量级差距（GEMM上32.3× for L2-to-L1），因为algorithmic-minimum ignoring所有buffer capacity constraints和mapping inefficiencies。
  - **Gap 1（Buffer Size Gap）**：maximal effectual buffer size（实现理论最小访问的buffer）远小于total operand size。例如，GEMM maximal effectual buffer ≈ smallest operand size = 33% total，gemv仅0.20% total——意味着即使buffer=total operand size也无法保证达到algorithmic-minimum。
  - **Mapping-dependence问题**：如果要获得比algorithmic-minimum更precise的data movement estimate，传统方法需做detailed mapping-aware DSE——对每个硬件设计点和每个mapping进行cost model评估。Simba 100-design DSE需2.6M mapping evaluations（10009s），随着design space增大exponentially增长，不适用于早期DSE。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出**Orojenesis**——一种通过Snowcat代理架构计算张量算法数据移动下界的方法论。核心设计：使用简化的单PE两级存储架构（Snowcat）进行穷举mapspace搜索，生成Pareto-optimal的ski-slope diagram（buffer size vs minimal backing store accesses）。对多Einsum链，通过Fusion Friendly Mapping Template (FFMT)约束intra-layer mappings以支持fusion分析。Orojenesis还衍生OI mesa和Performance mesa模型用于快速硬件DSE。
  
  全栈执行例子（以GPT-3-6.7b building block fusion分析在performance mesa指导下的硬件设计为例）：
  - **算法层**：GPT-3-6.7b（d=4096, h=32, f=128, c=16384, l=32768有效）。所有GEMM和BMM的Einsum表达为Timeloop workload spec。Orojenesis对完整building block做multi-Einsum chain analysis——覆盖Q_proj, K_proj, V_proj, bmm_QK, bmm_QKV, Final_proj, mm_0, mm_1共8个Einsum。
  - **系统框架层**：无具体Serving/Inference框架。Orojenesis输出的是**architecture-level design guidance**——例如fused LLM的optimal buffer area ratio=0.28（46MB buffer），2.4× higher throughput vs unfused LLM（optimal ratio=0.70, 113MB buffer）。该guidance可作为任何具体系统架构（GPU、TPU、custom accelerator）的设计输入。
  - **编译框架层**：Orojenesis基于Timeloop mapper进行Snowcat架构上的穷举mapspace搜索。对于多Einsum分析：施加FFMT约束（GEMM chain用Full/TiledK/TiledN/TiledKN四种模板）→ 穷举组合各Einsum在FFMT下的valid mappings → 穷举2^(#Einsums-1)种链分段策略 → 提取Pareto-optimal fused bounds。关键创新是**FFMT约束系统**——通过限制intermediate tensor的tiling模式确保producer-consumer data exchange无需spilling，同时通过链分段缓解长链中过于restrictive的约束。
  - **kernel调度层**：Snowcat mapspace search覆盖所有可能的tiling（M0/M1/K0/K1/N0/N1 loop bounds遍历所有可行因子组合）和loop order（所有合法permutations）。对每个mapping计算buffer size requirement（各tensor tile size之和）和backing store accesses（内层tile size × 外层迭代次数）。Pareto-optimal curve给出**mapping-independent lower bound**——任何实际mapping在任何实际架构上的data movement不可能低于此bound。验证：A100/H100 GPU上CUTLASS优化的mapping实测accesses ≥ Orojenesis bound；Simba上多buffer配置的实测points也均在bound之上。
  - **硬件架构层**：Performance mesa模型基于Orojenesis bounds + roofline model + Accelergy area estimation。对给定die area和DRAM BW，遍历buffer area ratio（0→1）计算：memory-limited perf = DRAM_BW / Orojenesis(buffer_size)，compute-limited perf = ((1-ratio)×area/area_per_MAC) × freq。Concave function的peak指示optimal buffer-to-compute ratio。论文展示了fused vs unfused LLM在此模型下的optimal design diverges significantly——不同workload需要不同硬件配置，Orojenesis能在不做per-design mapspace search的情况下给出design guidance。

  Orojenesis解决baseline三大缺陷的对应关系：
  - **解决Gap 0（Access Gap）**：Orojenesis bound考虑buffer capacity对data reuse的限制——ski-slope曲线精确给出每个buffer size下能达到的minimal backing store accesses，而非algorithmic-minimum的单一超乐观值。4k GEMM上Orojenesis bound vs algorithmic-minimum：当buffer=50MB（≈H100 LLC），Orojenesis access count远高于algo-min，与实测更接近。
  - **解决Gap 1（Buffer Size Gap）**：Orojenesis通过穷举mapspace搜索得到maximal effectual buffer size——即ski-slope曲线达到plateau处的buffer大小。论文发现GEMM maximal effectual buffer ≈ smallest operand size（formal derivation：min_operand_size + min_rank + 1），比total operand size小得多。这解释了为什么单纯增加buffer toal operand size不一定降低data movement。
  - **解决Mapping-dependence问题**：Orojenesis的bound是mapping-independent——Snowcat的unconstrained mapspace包含所有真实架构可能映射的超集。一次Orojenesis run的结果**portable到任意tensor accelerator architecture**（只要workload不变）。Runtime优势：单次Snowcat mapping evaluation 0.20ms vs Simba 3.90ms（19.5× faster），Orojenesis 90k mappings total (18s) vs Simba 100-design DSE 2.6M mappings (10009s) = **556× faster**。更重要的是Orojenesis只需run一次，而mapping-aware DSE需要per-design重新run。

## 25-Be CIM or Be Memory- A Dual-mode-aware DNN Compiler for CIM Accelerators.pdf

- baseline方法是什么？
  现有CIM编译器（PUMA, OCC, CIM-MLC）在编译优化时将所有CIM array静态视为compute-only单元，完全忽略现代CIM芯片的双模能力——每个CIM array可以通过修改输入driver信号在compute mode和memory mode之间动态切换。Baseline在映射DNN时假设on-chip compute和memory资源固定不变，对activation等中间数据的存储仅依赖original on-chip buffer和off-chip DRAM。当模型无法完全装入chip时通过tile分块执行，但每次tile切换需要off-chip数据搬移，增加延迟。

  全栈执行例子（baseline: CIM-MLC编译LLaMA2-7B单层attention到DynaPlasia 96-array CIM芯片）：
  - **算法层**：LLaMA2-7B Transformer decoder，8-bit量化权重和激活。Attention层包含Q/K/V projection (MMM)、QK^T attention + softmax + SV output、FFN (两个FC MMM)。
  - **系统框架层**：CIM-MLC接收ONNX graph→weight duplication与multi-grained pipelining→所有96个CIM array全作compute mode→weight mapping（将权重pre-load到CIM array中执行MAC）。
  - **编译框架层**：CIM-MLC基于MLIR-like multi-level IR，通过tiling、loop unrolling、operator fusion等优化映射算子到CIM array。但由于所有array固定为compute mode，activation的存储仅依赖原始的10KB×8 buffer——当activation数据量大（如长sequence的KV cache）时buffer不足，必须spill到off-chip main memory。
  - **kernel调度层**：CIM-MLC的pipeline调度将多个算子映射到固定compute array上并行执行（如Q/K/V projection weight duplication到不同array组），但operator间intermediate activation通过buffer交换。若buffer不够容纳K（用于后续QK^T），K必须写回off-chip→重新加载，增加内存搬运延迟。
  - **硬件架构层**：DynaPlasia 96个CIM array全在compute mode，on-chip scratchpad仅10KB×8，external bandwidth Dmain供应activation的读写。对低arithmetic intensity的LLaMA2（AI≈2），compute array大量MAC能力闲置等待数据。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  CMSwitch通过三大核心设计将CIM array的dual-mode能力纳入编译优化，解决baseline将array固定为compute-only导致资源错配的缺陷：

  **(1) DEHA双模硬件抽象 → 使编译器"感知"mode switch**：
  Baseline编译器不知道CIM array可以切换模式。CMSwitch在硬件抽象层引入dual-mode参数（switch method, switch latency, array数量/尺寸, compute/memory mode下的operation rate），使编译器可以将"这个array是compute还是memory"作为一个优化变量。这使得原本被baseline视为"全部compute"的96个array可以部分用作on-chip scratchpad memory，替代off-chip搬运。

  **(2) DACO两步联合优化 → 解决指数级搜索空间问题**：
  Baseline的编译优化空间已经是多项式复杂度（weight mapping + scheduling），dual-mode CIM引入每个array的memory/compute二选一后空间膨胀2^m倍（m=96 arrays → 2^96）。CMSwitch通过DP+MIP两步分治：DP在时间维度分割网络（segment间考虑mode switch overhead：数据写回→模式切换→权重重载），MIP在空间维度（每个segment内）联合优化array mode allocation + 算子pipeline调度。Gurobi求解器高效搜索最优的compute/memory array分配比例。

  **(3) 动态memory/compute资源调配 → 匹配不同模型/层的算术密度差异**：
  不同DNN模型算术密度差异巨大（ResNet50 AI≈66 vs LLaMA2 AI≈2），同一模型不同层AI也不同（CNN早期层vs后期层，Transformer的FC vs QKV）。CMSwitch通过MIP目标函数 `min max(LOi)` 和延迟模型 `LOi ∝ OPOi / min(ComOi·OPcim, (MemOi·Dcim+Dmain)·AIOi)` 自动为高AI算子分配更多compute array、低AI算子分配更多memory array。实验结果表明这种动态调配比baseline的static all-compute策略平均加速1.31×。

  全栈执行例子（CMSwitch编译LLaMA2-7B单层attention到DynaPlasia 96-array CIM芯片）：
  - **算法层**：同baseline，LLaMA2-7B 8-bit量化。但CMSwitch在编译期分析了各算子的arithmetic intensity差异——Q/K/V projection AI中等，QK^T attention AI较低，FFN FC AI较高。
  - **系统框架层**：CMSwitch通过DP将attention层分割为2个segment（QKV+QK^T为seg1，SV+FFN为seg2），segment间考虑mode switch开销。DEHA参数指导编译器了解switch只需1 cycle（通过GIA/GIAb信号修改）。
  - **编译框架层**：DACO的MIP为每个segment求解最优array mode分配——seg1中QKV projection分配40-50% compute + 50-60% memory（因需大量memory存放输入Q/K/V和中间结果K），QK^T attention分配更多compute（K已在memory中，直接原位切换为compute执行QK^T）。seg2中FFN FC1分配约60% compute（高AI），最后一层FC分配约33% compute + 67% memory（大activation需高带宽）。
  - **kernel调度层**：MIP求解的λ分配直接控制每个CIM array (x,y)的模式——例如array [0:31]设为compute mode loading W_Q，array [32:47]设为memory mode作input buffer接收Q activation，array [48:55]设为memory mode暂存K输出。Operator dependency检测到Oi→Oj时，Oi的output memory array自动转为Oj的input memory buffer（避免数据拷贝）。Key insight：K计算完后保留在memory mode array→CM.switch(TOM)将这些array切为compute→Q与K^T直接在这些array上原位MAC计算，完全消除K的off-chip搬运。
  - **硬件架构层**：DynaPlasia 96 array不再全为compute mode。在LLaMA2-7B上平均约10-25% array被设为memory mode（具体比例由sequence length和batch size决定），用作on-chip scratchpad存储activation和中间结果（包括KV cache）。这减少了off-chip main memory的读写次数，降低了memory wall瓶颈。Mode switch overhead仅占总执行时间的3%-5%，被性能提升充分抵消。相比baseline（所有array compute mode，intermediate data必须off-chip搬移），CMSwitch在长sequence下的memory需求场景加速更显著（如OPT-13B batch=1加速2.03×）。

## 28-MIMDRAM: An End-to-End Processing-Using-DRAM System for High-Throughput, Energy-Efficient and Programmer-Transparent Multiple-Instruction Multiple-Data Computing

- baseline方法是什么？
  SIMDRAM是一种state-of-the-art PUD框架，在DRAM subarray内以全行粒度（65536 bit wide）执行bit-serial SIMD操作。每个DRAM subarray作为一个固定的超宽SIMD引擎，所有16384-262144个DRAM列同时参与每一PUD操作。全栈执行例子（向量加法A+B=C，n-bit，1024元素）：
  - **算法层**：C/C++应用中循环→程序员手动识别PUD-friendly代码段→手动调用bbop_add指令，指定操作数和目标subarray
  - **系统框架层**：CPU执行到bbop指令→dispatch到SIMDRAM control unit→翻译为μProgram（n次迭代×每迭代(8n+2)个AAP/AP命令）→向DRAM发出ACT/PRE序列。数据从CPU cache通过transposition unit转换为vertical layout后写入DRAM subarray的全部列
  - **编译框架层**：论文未明确说明——SIMDRAM无编译器支持，程序员必须手动提取SIMD并行、编写bbop代码并管理数据映射
  - **kernel调度层**：每次bbop独占整个DRAM subarray（所有65536 SIMD lanes）→若实际数据并行度仅1024，仍有64512 lanes空闲→SIMD利用率=1.56%。无跨mat/data movement能力，vector reduction（如sum+=A[i]）必须回传CPU执行
  - **硬件架构层**：Ambit DRAM subarray→每次ACT激活全row所有cols→TRA同时激活三行（Data/Control/Bitwise groups）→通过charge sharing实现MAJ/NOT操作。无inter-mat或intra-mat interconnect，无法在DRAM内部做列间数据移动

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MIMDRAM通过硬件/软件协同设计，利用fine-grained DRAM（独立mat级行激活）将PUD从固定超宽SIMD升级为可变宽度+MIMD并行执行。全栈执行例子（同一向量加法A+B=C，1024元素）：
  - **算法层**：同上，C/C++循环→但**编译器自动识别并向量化**，无需手动编码。编译器选择max VF=1024（而非固定65536）→仅需1024/512=2个DRAM mats
  - **系统框架层**：CPU dispatch bbop_add(target mat_i, ML=0, VF=1024)→MIMDRAM control unit→mat scheduler查scoreboard→mats[0,1]空闲→分配μProgram engine 0→engine只激活2个mats的local wordline（mat selector仅断言mats[0,1]的isolation transistors）→在1024 SIMD lanes上执行bit-serial addition。**同时**另一独立bbop_mul(target mat_j, mats[2,3])由μProgram engine 1并在mats[2,3]上**并发执行**→这是MIMD模式的核心
  - **缺陷对应**：解决SIMDRAM的三大缺陷：
    (a) **SIMD利用率低**：MIMDRAM仅激活2个mats用于1024-wide操作（vs SIMDRAM激活整个subarray 65536列），SIMD利用率从1.56%提升到100%。实验：平均SIMD利用率提升15.6×
    (b) **缺乏reduction支持**：MIMDRAM的GB-MOV命令（inter-mat interconnect）将C[0]从mat0搬移到mat1，LC-MOV（intra-mat interconnect）在mat1内做adder tree reduction。全在DRAM内完成，无需CPU参与。Vector reduction延迟降低1.6×，能耗降低266×
    (c) **编程困难**：MIMDRAM三Pass LLVM编译器自动完成循环识别→向量化→DDG构建→mat label分配→bbop代码生成→pim_malloc数据分配。程序员无需任何修改
  - **编译框架层**：Pass 1用clang auto-vectorization识别循环→max VF=1024→生成`bbop_add<1024 x i32>`。Pass 2构建DDG→独立bbop分配不同mat→依赖bbop同mat。Pass 3生成带ML/VF的x86二进制
  - **kernel调度层**：MIMDRAM control unit的mat scheduler用online first fit算法调度bbop→mat scoreboard追踪128 mats的busy/free状态→最多8个μProgram engines并发。MIMD模式下不同mat range执行不同bbop。vector reduction的μProgram: GB-MOV跨mat搬移→LC-MOV intra-mat adder tree→最终4-element输出
  - **硬件架构层**：mat isolation transistor分段global wordline→row decoder latch存局部行地址→mat selector控制激活范围。inter-mat interconnect: global row buffer的2:1 MUX支持邻接SA组间4-bit搬移（GB-MOV tRAS+tRELOC+tWR+tRP）。intra-mat interconnect: 保持HFF enable信号，复用已有column select logic（LC-MOV 2×(tRAS+tRP)+tRELOC+tWR）。DRAM面积仅增1.11%，CPU die增0.6%

## 29-ASADI_Accelerating_Sparse_Attention_Using_Diagonal-based_In-Situ_Computing.pdf

- baseline方法是什么？
  PIM-based sparse attention加速器（Samsung FIMDRAM HBM2），使用CSR格式压缩稀疏attention矩阵，在memory bank附近的on-chip PE执行near-memory计算。

  全栈执行例子（BERT-Base单层attention处理batch of n tokens, d=64, ω=n/8）：
  - **算法层**：Sanger [22]的quantize-and-pruning动态稀疏方法生成sparse mask matrix M和sparse score matrix S。S矩阵以CSR格式存储（row_ptr + col_idx + values）。
  - **系统框架层**：无统一Serving框架。每个Transformer layer由host CPU编排：加载weight→发送到PIM→在PIM bank的on-chip PE执行计算→结果写回host→下一层。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：CSR-based SpMM (S×V)——对CSR格式的S矩阵，每个row of CSR需一次row-wise remapping以对齐S的列坐标与V的行坐标，然后做vector-vector multiplication。每次CSR iteration仅处理1行（平均2个非零元素），大量bubble（0值计算浪费）。总迭代次数=行数=n。CSR-based SDDMM (Q×K^T)——mask M的每行控制Q的第i行与K的部分行做乘法，行间共享K行时需串行执行。每次迭代仅2个有效计算。总迭代次数=行数=n。
  - **硬件架构层**：Samsung FIMDRAM——HBM2 memory stack，每个bank有独立on-chip PE（可编程计算单元在I/O circuit）。PE只能高效访问本地bank内存；访问cross-bank/cross-rank数据需通过memory controller和system bus（C/A bus），造成memory controller bottleneck和PE idle（实验显示>40% runtime PE idle）。Cross-bank transfer的延迟远高于local access。CSR格式的row-wise remapping需要大量随机访问（因稀疏attention的行局部性差），加剧cross-bank overhead。序列越长→更多cross-bank access→性能退化更严重。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出ASADI，一个DIA-based in-situ computing软件-硬件协同设计的稀疏注意力加速器。关键创新：(1) 观察到稀疏注意力有天然的对角线局部性（中心ω=n/8对角线上含>50%非零值），而非CSR假设的行局部性；(2) 设计DIA压缩格式和DIA-based SpMM/SDDMM计算范式，将对角线局部性转化为计算并行度；(3) 设计全流式in-situ ReRAM加速器，彻底消除cross-bank data transfer。

  全栈执行例子（同BERT-Base单层attention, n, d, ω=n/8）：
  - **算法层**：(a) DIA压缩——选中心ω对角线→bubble-free压缩（直接按对角线列式存储）→ω区域外非零元素移动到最近bubble→记录(Rd,Ro)坐标。解压时根据(Rd,Ro)恢复原始位置。(b) DIA-based SpMM——S矩阵按DIA格式存储，每列元素天然对齐列坐标→直接vector-vector multi，无需row-wise remapping。每次DIA iteration处理一整条对角线（平均5个非零元素），极大减少bubble。迭代次数=ω(≪n)。Longformer场景DIA比CSR减少7.5×迭代次数。(c) DIA-based SDDMM——M的每列DI控制Q shift up/down（而非M每行控制Q的某行）→各DI iteration的Q/K操作无共享冲突。迭代次数=ω(≪n)。Longformer场景节省7.5× latency。

  - **缺陷对应#1 - CSR的行局部性差**：CSR假设稀疏矩阵有行局部性（每行非零元素连续），但实验证明稀疏注意力有很强的对角线局部性（ω=n/8时>50%非零在对角线），行局部性极弱。论文提出DIA压缩和DIA-based计算范式，利用对角线局部性——如图3所示DIA格式每列存5个非零元素（vs CSR每行2个），直接提升7.5×数据复用。消融实验中CSR-ASADI（CSR+ReRAM）在短序列上性能低于baseline，验证了CSR格式本身是瓶颈。

  - **缺陷对应#2 - Cross-bank通信开销**：PIM架构的PE只能高效访问local bank，cross-bank数据需要经过memory controller→造成controller bottleneck和>40% PE idle。ASADI采用full-flow in-situ computing——所有original和intermediate matrices (Q/K/V/S/Z)均存储在ReRAM arrays中，计算直接在数据所在ReRAM cells执行（processing-using-memory），无需PE从memory fetch数据→消除了on-chip data transfer。消融实验DIA-PIM（DIA+ PIM硬件）仅1.3× speedup，因为PIM硬件无法避免cross-bank transfer，验证了in-situ computing硬件的必要性。

  - **缺陷对应#3 - 随着序列长度性能退化**：PIM baseline的cross-bank transfer随序列增长而增加（因PE数量固定），导致性能加速下降。ASADI的ReRAM rows天然随序列长度增加（更多tokens→更多ReRAM rows参与in-situ parallel→PE数量自然增长），性能反而随序列提升——Syn-8K上达63.7× speedup vs PIM baseline。

  - **系统框架层**：ASADI用Microcontroller替换host CPU的编排角色——Microcontroller发送四种控制信号(S×V/Q×K^T/Linear/Softmax)管理analog/digital module间的data transfer和稀疏mask的压缩/解压，消除了PIM baseline中host↔PIM的往返延迟。

  - **编译框架层**：论文未明确说明。

  - **kernel调度层**：In-situ S×V——DIA格式S矩阵均匀分布到d个ReRAM arrays→每个array执行vector-vector multi→transfer DIA vectors跨arrays→再次multi→decompress ((Rd,Ro) lookup)→accumulate add（Algorithm 1, O(ω)迭代）。In-situ Q×K^T——M的每个DI依次送到row selector→按DI shift Q up/down→(Rd,Ro) memory copy处理灰色元素→parallel vector-vector multi→restore Q→下一DI（Algorithm 2, O(ω)迭代）。所有ReRAM rows并行计算→时间复杂度O(n)+c+O(n)而非O(dn²)。

  - **硬件架构层**：ASADI accelerator——En-PE/De-PE结构，每个Tile含Analog Module #1 (read-only, VMM for linear layers) + Digital Module (write-enable, Q×K^T+Softmax+S×V全in-situ) + Analog Module #2 (feed-forward) + Microcontroller。Analog module使用analog in-situ (粗粒度矩阵级并行，适合bubble-free线性层)；Digital module使用digital in-situ (行级并行，适合bubble-containing的稀疏attention)。Cross-Encoder流水——batch间pipeline不同Encoder/De-PE。Intra-Encoder三阶段顺序执行（因data dependency）。总面积279.8mm²/538.9K mW，ReRAM 9.7GB容量支持MSL 8192。

## IANUS: Integrated Accelerator based on NPU-PIM Unified Memory System

- baseline方法是什么？
  Baseline是NVIDIA A100 GPU运行GPT-2 end-to-end推理，以及DFX（4 FPGA的多FPGA appliance）的专用生成阶段加速方案。全栈执行例子（A100 GPU, GPT-2 XL, (256,512)配置）：
  - **算法pipeline层**：标准GPT-2 Transformer推理。Summarization阶段：所有input tokens并行处理，FC layers为matrix-matrix乘法（compute-bound），self-attention的Q/K/V generation为matrix-matrix乘法。Generation阶段：逐token处理，FC layers为matrix-vector乘法（memory-bound），Q/K/V generation为matrix-vector乘法，需从HBM加载weight矩阵和previously generated K/V cache。
  - **系统框架层**：PyTorch 2.0 + CUDA 11.8，使用HuggingFace Transformers或Megatron-LM优化代码。标准推理pipeline：tokenize→embedding→decoder blocks（attention→FFN→layer norm→residual add）→LM head→sampling。
  - **编译框架层**：NVIDIA CUDA compiler toolchain（nvcc），cuBLAS library用于GEMM/GEVM kernel，无domain-specific compiler optimization。
  - **kernel调度层**：GPU SM上执行——matrix-matrix乘法通过Tensor Cores加速（312 TFLOPS@FP16 on A100），但matrix-vector乘法受限于HBM bandwidth（2039 GB/s），GPU SM利用率极低（generation stage memory-bound，FLOPs需求仅为summarization的1/512但耗时88.5%）。Self-attention中大量non-computing操作（key transposition, attention head splitting/merging, masking）占用41.4% decoder latency中66.1%的latency。Layer normalization和residual addition仅占总FLOPs <0.06%但占13.2% latency。DFX通过最大化bandwidth utilization（匹配peak FLOPS到memory BW）优化generation stage但summarization stage性能受限（FPGA limited FLOPS）。
  - **硬件架构层**：NVIDIA A100 SXM 80GB GPU——108 SM，1555 MHz (peak)，255 TFLOPS (FP16 Tensor Core)，2039 GB/s HBM2e bandwidth，80GB HBM2e capacity，400W TDP。DFX——4 FPGA 200MHz, 1.64 TFLOPS, external DDR4 memory。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出IANUS NPU-PIM统一内存异构系统+PIM Access Scheduling (PAS)。全栈执行例子（IANUS, GPT-2 XL, (256,512)）：
  - **算法pipeline层**：同标准GPT-2推理，无算法改动。BF16精度保持full-precision模型准确率（WikiText-2 perplexity验证：Base 30.92, M 22.60, L 19.39, XL 17.48）。
  - **系统框架层**：自定义编译器生成ordered commands，Algorithm 1在compile time自动选择FC映射目标（MU或PIM）——基于analytical model估计MU时间（含column-tiling pipeline+weight prefetching）和PIM时间（= n tokens × PIM execution time per token），选时间短者。Forward pass分两阶段：Summarization→FC映射到MU（高compute throughput处理matrix-matrix乘法）；Generation→FC映射到PIM（高effective memory BW处理matrix-vector乘法）。
  - **编译框架层**：论文compiler处理：(1) head-wise partitioning——将Q/K/V weight分布到不同PIM channel，各core独立访问对应head的weight；(2) intra-layer parallelism——column-wise weight分割到各core，减少weight data movement（weight远大于activation）；(3) 地址生成——compiler定义AM地址实现on-chip attention head splitting/merging（无需data reordering overhead）；(4) 同步点插入——4个同步点（multi-head attention后1个 + residual addition后2个 + GELU后1个）。
  - **kernel调度层**：PIM Access Scheduling (PAS)解决unified memory的核心矛盾——PIM memory既要服务NPU的normal memory access又要服务PIM computation，二者不能并发：(1) Macro PIM command封装一个操作的多个micro commands，减少调度开销；(2) PCU pipeline解码macro→micro commands；(3) PIM MC按GDDR6严格时序约束发出bank-level commands；(4) Command scheduler在macro PIM cmd issue时强制未issue的DMA commands wait，保证PIM执行不被打断；(5) Mapping-aware scheduling：Summarization阶段Q/K/V FC→MU（intra-head parallel + inter-head pipeline），key优先生成以并行执行on-chip key transposition（DMA-based，AM→WM数据搬运即partial transpose），QK^T和SV→MU或PIM根据workload特征选择；Generation阶段FC→PIM，QK^T和SV→PIM时省去K/V cache loading（weight prefetching替代），scheduling提升平均34%性能。
  - **硬件架构层**：IANUS——4 cores NPU (MU 128×64 systolic array 46 TFLOPS + VU 16 VLIW processors, 700 MHz) + GDDR6-AiM PIM (8 channels, 16 banks/channel, 1 GHz PU/bank, 1024 GB/s internal BW——9-10× external BW, 全bank并行) + unified memory system (PIM serving both NPU main memory and PIM computation, 2× memory saving)。向量操作（layer norm, softmax, GELU, residual add）→VU专用处理（two-phase layer norm, masked softmax with 1-bit bitmap mask, GELU via LUT）。Self-attention数据操作：on-chip key transposition（DMA between AM and WM + streaming buffer）+ compiler-based head splitting/merging（消除off-chip data movement）。

对比baseline缺陷：
1. **Memory-bound generation stage**：A100 GPU matrix-vector乘法受限于2039 GB/s HBM bandwidth → IANUS PIM提供1024 GB/s/chip internal BW，8 channels总计8192 GB/s internal BW，generation stage FC speedup 4.1× (FFN 5.1×)。
2. **Diverse computational requirements**：GPU和DFX各只能高效处理一种计算类型 → IANUS NPU处理compute-bound matrix-matrix（summarization），PIM处理memory-bound matrix-vector（generation），VU处理low-FLOPs vector ops。平均6.2× speedup vs A100, 3.2× vs DFX。
3. **Memory duplication in partitioned system**：传统PIM系统host和PIM分别有独立memory，shared data需duplicate → IANUS unified memory消除FC参数duplication（91%参数共享，≈2×减少memory footprint），额外的PIM capacity转化为更高的PIM throughput。Unified + scheduling vs partitioned达1.4-1.6× speedup。
4. **Non-computing self-attention overhead**：A100 GPU上self-attention的non-computing ops占用66.1% latency → IANUS通过DMA-based on-chip transposition + compiler-based head splitting/merging消除data reordering overhead。BERT模型上compute utilization 5.2×/3.3×/1.3×/1.0× higher vs GPU（B/L/1.3B/3.9B）。
5. **Energy inefficiency**：A100 GPU 400W TDP → IANUS 120W TDP（conservative estimate），3.7-4.4× energy efficiency improvement over NPU-MEM（仅NPU+标准GDDR6），cost-efficiency（performance/TDP）2.1-3.9× over A100。

## 31-UM-PIM_DRAM-based_PIM_with_Uniform_amp_Shared_Memory_Space.pdf

- baseline方法是什么？
  Baseline是现有通用DRAM-based PIM系统（以UPMEM为代表）采用Isolated Memory Space + Software Data Re-layout + Globally Turn Off Memory Interleaving三项措施来保证PIM task offloading效率。

  全栈执行例子（UPMEM PIM-Ioff系统，以BFS PIM workload为例——CPU host program offloads memory-intensive graph traversal to 64 PIM units per rank）：
  - **算法层**：BFS算法划分为CPU segment（frontier management, synchronization）和PIM segment（neighbor traversal on local partition）。每个PIM unit处理分配到其本地DRAM bank的graph partition。
  - **系统框架层**：Host CPU通过UPMEM SDK管理PIM task lifecycle。(Step 1) CPU分配isolated PIM memory space（预留连续物理地址block作为PIM space）→(Step 2) CPU在main memory中准备input data→(Step 3) **CPU-PIM Data Transfer**：CPU将input data从main memory拷贝到PIM space，同时执行software address translation（计算每个数据byte在目标PIM unit local memory中的HWAddr，涉及bank/row/column mapping）和data re-layout（按PIM unit prefer的non-interleaved layout重排数据以保证同一PIM unit可见contiguous data block）→(Step 4) CPU offload PIM task（context switching + locking memory regions，overhead >50μs）→(Step 5) PIM unit执行计算（访问local DRAM bank中的PIM memory space）→(Step 6) **PIM-CPU Data Transfer**：CPU从PIM space拷贝result data回main memory，再次执行地址翻译和数据重排→(Step 7) CPU继续执行后续computation。地址翻译+数据重排占总传输时间的~70%（以UPMEM SDK测量，length=256KB时PIM-CPU: 74.2%, CPU-PIM: 69.7%）。对于BFS这种有大量细粒度传输的workload（BFS有~1.33E4次传输、每次short transfer），per-transfer fixed overhead导致total transfer time远大于粗粒度传输workload。
  - **编译框架层**：UPMEM SDK基于clang编译器。Host program和DPU kernel分离编译。Host program负责数据管理和offload decision，DPU kernel为SPMD模型。无自动offloading compiler optimization。
  - **kernel调度层**：(Step 3+6) Software address translation：CPU遍历virtual address→通过page table获得PAddr→按PIM architecture的address mapping规则（Ch-Ra-Ro-Ba-Co，interleaving off at channel/rank level）手动计算每个data byte的HWAddr（需软件模拟memory controller的地址映射逻辑，处理bank/row/column/device各层映射）→分配临时buffer→调用memcpy重新排列数据布局。(Step 4) Offloading：CPU设置DPU program counter→发送启动信号→等待DPU完成（polling或interrupt）。(Step 7) Inter-PIM communication（如BFS的frontier exchange）：各PIM unit完成local computation→CPU gather各PIM result→CPU compute reduction/merge→CPU scatter back to PIM units。Fork-join模式，CPU为同步中心。对于全局interleaving关闭的系统（PIM-Ioff），rank/channel interleaving off导致CPU pages的内存带宽下降——CPU memory bandwidth退化为单channel单rank的带宽。
  - **硬件架构层**：UPMEM PIM DIMM。DDR4-2400，bank-level DPU（64 DPU per rank, @500MHz, 16 tasklets）。DPU local memory：64MB MRAM + 64KB WRAM + 24KB IRAM。无DPU间直连通信——cross-DPU数据交换通过host CPU memory bus转发。Device-level interleaving无法通过BIOS关闭（仅在rank/channel level可配置）→即使PIM-Ioff关闭rank/channel interleaving，device-level interleaving仍然生效→同一bank的8个device中数据仍按burst粒度交错分布→PIM unit要访问的64B cache line实际分布在8个device的各8B中，需通过软件重新排序。CPU侧性能退化：CPU pages因interleaving off导致bandwidth下降，PIM-Ioff下8核CPU性能退化25.8%（在SPEC CPU 2006上）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出UM-PIM，一个Uniform & Shared Memory Space的通用PIM系统，通过dual-track memory management + hardware address mapping + hardware data re-layout三项创新，消除isolated memory space带来的"system memory wall"。

  **缺陷对应#1 - Isolated Memory Space导致的数据拷贝overhead**：
  Baseline中CPU和PIM有各自独立memory space，PIM task offloading需要Step 3（CPU→PIM拷贝）和Step 6（PIM→CPU拷贝）两次完整的memory copy + 地址翻译 + 数据重排。UM-PIM通过uniform & shared memory space设计——CPU pages（interleaved）和PIM pages（non-interleaved）共存于同一memory address space，CPU可直接以virtual address读写PIM pages。Step 3和Step 6完全消除——CPU直接写PIM page就是PIM unit可见的contiguous数据（因为PIM pages以non-interleaved chunk方式组织），CPU直接读PIM page即可获取结果。对比PIM-Ioff，UM-PIM在PIM workload上平均减少4.93× CPU computing time，端到端加速1.96×。

  **缺陷对应#2 - Software Address Translation和Data Re-layout的高开销**：
  Baseline中地址翻译+数据重排占data transfer时间的~70%，且per-transfer有fixed overhead（multi-thread API的preparation time），导致细粒度传输workload（BFS, NW, SEL）的overhead尤其严重。UM-PIM通过DRAM-side硬件加速：(a) **RCL+ATM**（集成在RCD chip）：RCL通过9-bit CAM lookup判定page type→ATM对PIM page自动将column address bits 9-7映射为device address、其余bits左移3位→实现below-bank address mapping的硬件化。Address translation overhead从software多线程CPU cycle降为tUI（~1 cycle @2.4GHz RCD clock）。(b) **RC**（集成在DIMM buffer）：8个SB × 8 cells的buffer，通过MUX1（BL number）和MUX2（DE signal）两级选择，在DRAM侧自动聚合device-level分散的64B数据为一个完整的cache line burst，消除CPU侧的数据filtering和重排。(c) **CPU side**：仅需通过BIOS配置TAD1添加address alias，CPU访问PIM pages的address mapping自动走TAD1的non-interleaved path，对软件透明。

  **缺陷对应#3 - 全局关闭Memory Interleaving导致的CPU性能退化**：
  Baseline PIM-Ioff关闭rank/channel interleaving导致CPU memory bandwidth大幅下降（8核SPEC CPU 2006上退化25.8%）。UM-PIM通过dual-track address mapping——CPU pages保持原有interleaved mapping（TAD0, Ro-Ra-Ba-Co-Ch），PIM pages使用non-interleaved alias（TAD1, Ch-Ra-Ro-Ba-Co），bit 37 of PAddr区分。CPU访问CPU pages时仍享受full interleaved bandwidth（CPU性能退化<0.1%），仅访问PIM pages时走non-interleaved path。由于PIM pages主要用于PIM computation而不直接参与CPU compute的memory-intensive access，这几乎是零代价的折衷。

  **缺陷对应#4 - Inter-PIM Communication必须经过CPU**：
  Baseline中fork-join模式的inter-PIM通信必须CPU gather→compute→scatter（两次data transfer）。UM-PIM通过shared memory space + communication API消除此overhead：(a) Scatter/Gather/All-Gather：CPU直接在PIM pages之间做memcpy，利用RC的data locality——nested loop order优化使同一offset的burst访问相邻iteration→RC hit rate提升→DRAM read bandwidth接近CPU pages水平（仅1.6× slower）。(b) Broadcast：利用RC的broadcast mode（通过BCM指令设置）——CPU一次burst写入即复制到所有8个devices→memcpy from source PIM page to all destination PIM pages仅需#bank次读写。(c) UM-PIM的scatter/broadcast/gather分别达9.84×/12.6×/7.90× speedup vs PIM-Ioff。

  全栈执行例子（UM-PIM, BFS PIM workload，同baseline场景）：
  - **算法层**：同baseline BFS算法。
  - **系统框架层**：CPU program调用malloc_pim(len)分配PIM pages→OS通过THP mmap分配256MB chunk→mlock防止swap→读/proc/pid/pagemap获取CPN→CPU通过MMIO发送ACN指令将CPN写入RCL（同时PCL同步更新）。CPU direct write input data到PIM pages（zero-copy，使用virtual address正常访问——MMU翻译→PAddr bit37=1→TAD1 non-interleaved mapping→经UM-PIM Interface的RCL+ATM→DRAM access）。CPU offload PIM task。PIM unit执行计算时通过PCL（以PCN为索引查CPN→+PIM offset→PAddr→same HWAddr as CPU side）访问同一PIM page。CPU direct read PIM pages获取result（无PIM-CPU transfer）。CPU进行inter-PIM frontier exchange时使用Gather API→CPU读各bank的PIM page result→RC自动过滤device-level冗余→CPU memcpy scatter到目标bank的PIM pages→RC broadcast mode加速broadcast。Loop order优化：br（bank）和l（offset）在外层，dr（device）在内层→连续访问同一device的同offset→RC hit rate提升。
  - **编译框架层**：论文讨论但未明确实现compiler integration。Offloading decision algorithm受益于UM-PIM的低offloading overhead→可offload 7.8%更多program segments to PIM。
  - **kernel调度层**：Zero-copy access to PIM pages。Hardware RCL+ATM address mapping（~1 cycle tUI vs software multi-threaded CPU cycles）。RC data re-layout and filtering（DRAM-side，CPU不参与）。Inter-PIM communication API（scatter/broadcast/gather/all-gather with nested loop order optimization）。Efficient data traversal：CPU遍历PIM unit results时保证br/l在外、dr在内→RC hit rate从0提升到>80%→DRAM read bandwidth提升至接近CPU pages水平。
  - **硬件架构层**：Modified DRAM DIMM。RCD chip增加RCL+ATM+CG（0.72mm² @90nm）。DIMM buffer增加RC（0.21mm²，8 SBs）。Memory bank内PCL（0.031mm²）。CPU侧仅BIOS TAD配置（无需CPU硬件修改）。Total hardware overhead <1mm²，占DRAM chip面积可忽略。CPU pages interleaving preserved→CPU性能退化<0.1%。PIM pages non-interleaved chunk-based→每个bank贡献128KB contiguous PIM page→PIM unit可见完整contiguous data block。

## 33-CamPU_A_Multi-Camera_Processing_Unit_for_Deep_Learning-based_3D_Spatial_Computing_Systems.pdf

- baseline方法是什么？
  Baseline是基于GPU (NVIDIA RTX2080Ti) 的多相机深度学习的3D空间计算系统，采用in-order image projection unit with 4KB 2-way set-associative cache memory + full-sized中间球形图像blending。全栈执行例子：
  - **算法层**：四阶段pipeline——Stage 1: 多相机图像inverse perspective projection (iProj)映射到统一球形坐标并stitch；Stage 2: 球形图像perspective projection (Proj)生成多张切线图像（模拟虚拟相机视角，匹配DNN训练数据集rectilinear格式）；Stage 3: DNN在每张切线图像上提取语义特征（深度估计或语义分割）；Stage 4: DNN feature maps通过iProj反向映射回球形坐标并stitch。图像投影基于LUT（lookup table）存储预计算的mapping index，将公式(1)(2)的三角函数运算替换为内存查表操作。反投影(inverse warping)对每个输出像素加载4个邻域源像素做双线性插值。
  - **系统框架层**：GPU平台（RTX2080Ti, 250W TDP）执行所有stage。DNN阶段通过batch processing共享weight参数获得加速（Stage 3）。Proj (Stage 2) 可共享单个球形输入图像对多个小mapping index（0.125 MB/image）批量处理获得加速。但iProj (Stage 1和4) 因为各相机mapping index形状和值不同（不可共享），无法受益于batch processing，延迟随相机数线性增长。
  - **编译框架层**：论文未明确说明（GPU使用CUDA编程框架，无自定义编译优化）。
  - **kernel调度层**：GPU上remap操作（I(u,v)→O(θ,ϕ)）是内存密集型非线性图像变形。每个输出像素需4次remap（inverse warping的4邻域插值），导致4倍内存访问。不规则二维mapping index模式导致23% cache miss rate。LUT mapping index（iProj每图1 MB, 中间球形图每图2 MB）带来大量中间数据（24 MB/frame）。GPU通过扩展不同形状mapping index为统一full-sized矩形（含大量invalid maps）进行并行计算——88.3%数据为冗余invalid区域。Full-sized图像blending造成大量冗余内存足迹，在资源受限边缘设备上不可行。
  - **硬件架构层**：NVIDIA RTX2080Ti GPU（4352 CUDA cores, 11GB GDDR6, 250W TDP）。In-order image projection unit with 4KB 2-way set-associative cache（基准架构）。无专门图像投影硬件——图像投影和blending与DNN共享GPU SM/CUDA cores。多相机系统延迟87.3ms仅用于image projection (8 cameras)，占360° RGB-D generation总延迟69.8%（270.1ms）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出CamPU——专用多相机图像投影与融合硬件加速器，通过inter/intra-data reuse减少LUT访存、out-of-order image projection unit with cache memory隐藏延迟、overlap-aware blending unit减少冗余数据，解决GPU上多相机图像投影的性能瓶颈。对比baseline的全栈执行例子：
  - **算法层**：同baseline四阶段pipeline保持不变。CamPU核心创新在于硬件加速器替代GPU执行Stage 1/2/4的图像投影和blending，Stage 3 DNN仍由DSPU/GPU执行。不再扩展mapping index到full-sized矩形，而是使用最小无效区域的矩形mapping index。
  - **系统框架层**：CamPU-integrated DNN platform异构架构——低功耗CamPU (12.9mW, 0.54mm²) 加速图像投影和blending操作，DSPU (766.0mW, RISC-V CPU + DNN accelerator + 3D point processor) 执行DNN操作。RISC-V CPU控制CamPU（发送指令流→instruction decoder→instruction buffer→串行解码激活目标硬件单元）。多相机rig通过4 GB/s packet-based interconnection network连接CamPU。CamPU接收多相机图像→stitch成统一球形RGB图→生成多视角切线图像→DSPU batch处理DNN→CamPU stitch DNN输出→360°深度学习输出。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：
    - **缺陷对应#1——多相机iProj不可共享mapping index导致无batch加速**：Inter-data reuse利用纬度对齐多相机图像mapping index形状相同（仅中心索引θc不同，基于公式(2)），4相机共享单份mapping index→LUT footprint和带宽节省75%。Intra-data reuse利用相邻mapping index元素值相似性（如(θn,ϕn)/(θn+1,ϕn+1)/(θn+2,ϕn+2)指向同一输入像素(uk,vk)），差分编码压缩mapping index从8-bit→2-bit→额外节省75%。二者结合压缩94.4%（对256×256 18图像@4纬度）。
    - **缺陷对应#2——Inverse warping的4次remap带来冗余指令cache访问**：Out-of-order image projection unit的load OP unit动态调度——load address generator解码16条remap指令→load OP schedule unit重排序同cache line内的load→load OP fusion unit融合为单条f load指令→指令issue和cache访问减少72.7%（iProj+inverse warping）。
    - **缺陷对应#3——Cache miss延迟阻塞后续执行**：Memory load execution unit out-of-order执行f load指令——cache miss时不阻塞，推入miss OP queue并请求global memory，同时继续fetch并执行load OP queue/miss OP queue中下一条f load指令（不依赖cache hit状态）。隐藏cache miss延迟→image projection延迟减少74.9%。
    - **缺陷对应#4——多图像投影的并行性未利用**：得益于inter-data reuse，cache line直接排列4图像的对应pixel→memory load execution unit同时访问4图像pixel（3.17× vs 单图像投影）。
    - **缺陷对应#5——Full-sized无效区域导致88.3%冗余数据**：使用最小无效区域的矩形mapping index（含中心坐标(θc,ϕc)指示），中间数据减少81.9%。
    - **缺陷对应#6——非矩形投影输出与矩形内存系统不兼容**：Overlap-aware blending unit处理矩形投影输出——纬度对齐图像投影输出为相同形状矩形(θN×ϕN)，重叠区域对称相同(θov×ϕN)，blending unit并行加载对称重叠区像素对并做加权求和，offset controller添加中心索引offset。先按经度blend纬度对齐图像，再跨经度stitch。Image blending memory access减少53.1%（18图像@4纬度）。
  - **硬件架构层**：CamPU RTL设计→28nm Synopsys DC门级综合→500MHz/0.54mm²/12.9mW。4个CamPU core（各含index decoder+image projection unit+2KB output buffer+blending unit）+ 256KB global memory + coordinate converter（SIMD分段线性逼近三角函数，映射完成后deactivate）+ instruction decoder + packet-based interconnect（4 GB/s）。Pipelined架构使load OP unit/memory load execution unit/interpolation execution unit并行→隐藏各阶段延迟。4KB 2-way set-associative cache（仅支持load操作降低硬件开销），cache line=4个16-bit pixel values，hit check unit解码load操作并比较tag/valid位决定hit/miss→all-to-all write-back unit同时写入最多16个write-back寄存器。Ablation study：Baseline→+out-of-order projection (3.0×)→+overlap-aware blending (额外2.4×)→+pipeline (额外1.5×)→总计10.7× speedup vs baseline。端到端360° RGB-D generation (18 cameras)：CamPU+DSPU 94.1ms vs RTX2080Ti 270.1ms (2.9× faster)。

## 34-Stream-Based_Data_Placement_for_Near-Data_Processing_with_Extended_Memory.pdf

- baseline方法是什么？
  Baseline是将conventional NUCA cache管理方案（Jigsaw/Whirlpool/Nexus）直接适配到NDP with Extended Memory的distributed DRAM cache上。NDP stacks的DRAM作为CXL-based extended memory的cache，采用fine-grained cacheline-level管理（64B granularity），每NDP unit配备metadata cache（128 kB dual-granularity，512B block metadata + 64B fine-grained migration）。全栈执行例子：
  - **算法层**：Memory-intensive workloads（tensor compute/graph computing/Rodinia并行程序）在NDP cores上执行，数据通过标准load/store访问NDP DRAM cache。无stream抽象，程序员不提供data access pattern hints。
  - **系统框架层**：Host processor管理NDP与CXL extended memory之间的cache hierarchy。NDP memory作为transparent cache，extended memory作为OS-visible physical memory。数据最初加载到extended memory，runtime按需fetch到NDP DRAM cache。
  - **编译框架层**：论文未明确说明。无特殊编译优化。
  - **kernel调度层**：Baseline NUCA方案：(a) Jigsaw——way-based sampling + lookahead algorithm，iteratively根据miss curves的steepest slope分配cache容量，placement用center-of-mass方法（移数据到access location的加权中心）。Sizing和placement分离求解。(b) Whirlpool——分类不同data structures后partition cache，但placement仍为center-of-mass。(c) Nexus——在Jigsaw基础上加data replication，但仅支持global uniform replication degree for all read-only data。各方案均用bulk invalidation在reconfiguration时清除旧数据。关键问题：(1) Interconnect overhead——center-of-mass placement优先使用interconnect中心位置的NDP units，使边缘units数据须走更多hops。NDP interconnect latency远高于NUCA（32% vs 13% accessed cycle占比），remote DRAM access可能需要跨多个stack。若cache capacity不足，部分数据被迫放到suboptimal位置，hop count进一步增加。(2) Metadata overhead——256 MB DRAM cache的metadata（假设每64B cacheline 4B entry）需16 MB，无法全部on-chip。metadata访问自身也需要DRAM access，10% cycles消耗在remote tag access上。大workload下metadata cache hit rate降到47%。metadata存储和lookup成为critical path额外瓶颈。(3) Replication不灵活——Nexus的global replication degree无法针对每个data structure定制。某些hot read-only数据需要更多副本但被全局策略限制，某些cold数据不需要副本却占用空间。
  - **硬件架构层**：128-core 3D-stacked NDP (HBM3/HMC2) + CXL Type-3 DDR5 extended memory。每个NDP unit含128 kB metadata cache。Interconnect: intra-stack 128-bit mesh (1.5 ns/hop) + inter-stack 32 GB/s links (10 ns/hop)。CXL link 200 ns latency（default）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出NDPExt——用coarse-grained data streams替代fine-grained cachelines管理NDP distributed DRAM cache，配合software runtime周期性co-optimize sizing、placement和replication。对比baseline的全栈执行例子：
  - **算法层**：程序员用configure stream(type, base, size, elemSize [, stride, length, order]) API宣告每个data structure的stream属性。例如PageRank：vertex list (affine)、edge list (affine)、rank scores (indirect, depends on edge list)、visited array (indirect)。支持多维度affine stream的element reordering（通过order参数指定access order不同于storage order，改善spatial locality）。平均仅需4.3行代码修改per workload，>99% access被streams capture。**缺陷对应**：stream model提供粗粒度access pattern抽象，使硬件能按pattern管理而非逐cacheline追踪。
  - **系统框架层**：NDPExt runtime在host processor上周期性执行（每50M cycles epoch）。Epoch流程：samplers采集miss curves → bitvectors发送host → max-flow sampler assignment → configuration algorithm (Algorithm 1) co-optimize sizing+placement+replication → stream remap table下发NDP stacks → consistent hashing remap减少数据移动。仅read-only streams支持多replication group，read-write streams通过readOnly bit机制（首次写exception→host invalidate all copies→后续单copy直接处理）保证coherence。**缺陷对应**：co-optimization使sizing和placement同时决策（不再分离求解），避免center-of-mass方法的interconnect中心偏好和suboptimal placement。
  - **编译框架层**：论文未明确说明。当前手动annotate streams，compiler automatic stream detection留待future work。
  - **kernel调度层**：三项核心runtime创新对应baseline三个缺陷。
    - **缺陷对应#1——Interconnect overhead（32% cycles）**：NDPExt的configuration algorithm（Algorithm 1）迭代式同时决定sizing和placement。每个stream初始在每个accessing unit独立replicate（最大replication degree），随着space耗尽逐步group extending（用nearby unit空间，attenuation factor惩罚remote latency）或group merging（合并replication groups释放space）。这使每个stream演化出独立的replication scheme，hotspot从Nexus的113 ns avg interconnect latency降至38 ns（small replication groups of 1-2 units），mv workload的33% cache space用于replication。而baseline Jigsaw/Whirlpool的center-of-mass方法无法如此aggressively优化placement。
    - **缺陷对应#2——Metadata overhead（10% cycles on remote tag access）**：NDPExt用coarse-grained stream metadata替代fine-grained cacheline metadata。(a) Stream cache用global stream remap table（160 kB, 512 streams × 64 units × 40 bits）在host memory维护RShares/RRowBase/RGroups，每个NDP unit用32-entry SLB（4.6 kB SRAM, TCAM range matching）做local cache——10% metadata overhead降至near zero。SLB miss才从host refill，类似TLB。(b) Affine streams用on-chip ATA（64 kB SRAM, 1 kB block size, set-associative），限总affine cache space ≤16 MB per unit以控制SRAM tag。Indirect streams用direct-mapped DRAM tag（tag+data同存）。对比baseline 128 kB metadata cache + 47% hit rate on graph workloads → NDPExt消除了metadata access的critical path延迟。
    - **缺陷对应#3——Inflexible replication（global uniform degree）**：NDPExt通过RGroups支持per-stream replication customization。每个stream独立演进replication groups（extend/merge操作），不同stream可有不同副本数。例如backprop：layerforward（read-intensive）用91% cache space做replication，adjustweights（write-intensive）zero replication。对比Nexus只能对所有read-only data用相同replication degree。
  - **硬件架构层**：新增stream cache硬件：(1) SLB——32-entry TCAM + digital comparators做base+size range matching，3.2 ns lookup latency；(2) ATA——16k-entry SRAM，1 kB block tag array for affine streams；(3) Set-based miss curve samplers——4 per unit × 8 kB，用k=32 sample sets同时capture c=64 capacity cases（32 kB-256 MB, geometric partition）；(4) 512-bit accessed stream bitvector per unit；(5) CXL controller连接extended memory。Total per-unit SRAM ~100 kB，well aligned with prior NDP designs。Zsim-based simulation。**缺陷对应**：stream cache hardware使metadata lookup从DRAM latency降至SRAM latency（SLB/ATA），从critical path移除tag access，enable per-stream flexible replication。
  - 综合效果：NDPExt geomean 1.41× speedup over Nexus（up to 2.43× on recsys），40.3% energy saving。Interconnect energy从6.6%降至3.2%，DRAM energy降低8.3%。NDPExt surpasses NDPExt-static by 1.2× geomean（up to 1.7× on irregular workloads like pr），证明runtime reconfiguration必要。关键tradeoff：replication可能增加miss rate（mv slightly higher），但interconnect latency reduction远大于miss rate penalty，overall performance net positive。

## 36-Leviathan_A_Unified_System_for_General-Purpose_Near-Data_Computing.pdf

- baseline方法是什么？
  Baseline是**单范式NDC硬件加速器或受限可编程NDC系统**（如täkō [66] 仅支持data-triggered，Livia [47] 仅支持task offload），每个设计在cache hierarchy中加入定制或有限可编程硬件以支持单一NDC范式。全栈执行例子（以täkō运行PHI PageRank为例）：
  - **算法层**：PHI [52] 设计需要同时支持task offload（RMW operations near data）和data-triggered（cache insertion/eviction拦截以初始化/apply commutative updates）。但täkō仅支持data-triggered actions，无法offload RMW任务。täkō被迫用relaxed atomics [9, 70]（无memory fence的原子指令）在core上模拟RMO，要求程序员手动保证memory ordering，且无法消除core间data ping-pong。
  - **系统框架层**：各NDC系统独立设计，无统一编程接口。täkō要求程序员手动管理cache line alignment和padding——数据必须fit within and align to cache lines [18, 31, 47, 52, 66, 94, 95]，如24B object跨64B cache line会导致部分object在另一bank，NDC action丧失locality。täkō的onMiss/onEviction callback操作cache line而非object，程序员须自行处理line内object布局。不同范式接口互不相容——task-offload像函数调用、long-lived像spawn线程、data-triggered像注册中断handler、streaming像打开网络socket。
  - **编译框架层**：论文未明确说明（使用标准C++编译，täkō和Livia均基于同一simulator框架）。
  - **kernel调度层**：täkō仅支持data-triggered调度——cache controller在miss/eviction时触发callback，所有NDC action必须映射为onMiss/onEviction语义。无task location scheduling（如DYNAMIC probing到L1D→L2→LLC），无stream调度（pseudo-streaming通过每cache line触发新action，须每次"reinitialize"遍历栈，平均engine instructions per edge更多）。无跨范式调度——PHI的offload RMW + data-triggered无法在单一系统中协同。
  - **硬件架构层**：täkō在cache bank旁加可编程dataflow engine（类Leviathan），但缺少：invoke指令和invoke buffer（无法task offload）、rTLB（无法从物理地址反译虚拟地址，data-triggered action受限）、LLC object mapping硬件（无法将>cache line object映射到单bank）、DRAM compaction硬件（软件pad导致内存碎片——如24B pad至32B浪费25%）、stream scheduler硬件（无push/pop/head-tail管理）。每条cache line的data-triggered callback均需engine重新初始化context，开销大。Relaxed atomics方案虽有3.1× speedup但存在正确性风险且无法消除NoC traffic——täkō的NoC traffic比Leviathan多40%。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Leviathan通过**多态缓存层级架构 + actor-based reactive programming interface**统一四种NDC范式，用一套硬件和接口支持所有范式及其交互。对比baseline的全栈执行例子：
  - **算法层**：程序员用统一的actor-based接口定义每种范式——task offload用invoke actor->action()（Fig. 9），data-triggered用Morph注册constructor/destructor（Fig. 11），streaming用Stream<Edge> + genStream/push/next（Fig. 12）。四种范式共享actor model（对象+关联action），区别仅在于"when/where to execute"由runtime和硬件自动处理。**解决baseline的范式互不相容问题**：同一系统可同时运行PHI（task offload + data-triggered协同）、NSC（stream + task offload协同）等需要多范式的应用。
  - **系统框架层**：Leviathan::Allocator<T>隐藏所有microarchitectural detail——应用只需allocate/deallocate，Leviathan自动：(a) padding objects to power-of-two in cache（Fig. 8b，解决cache line crossing问题）；(b) LLC object mapping（大于cache line的object映射到同一bank，解决baseline的multi-bank fragmentation）；(c) DRAM compaction（cache中pad但DRAM中紧凑存储，解决baseline的25%+ memory fragmentation）。Data-triggered actions操作object而非cache line（Morph::getActor/getOffset），应用只需实现per-object constructor/destructor。**解决baseline要求程序员手动管理cache alignment、padding、layout的编程负担**。
  - **编译框架层**：论文未明确说明编译器修改（return→send on future的翻译由compiler处理）。
  - **kernel调度层**：Engine硬件scheduler按范式自动确定"when/where"执行：(a) **DYNAMIC task scheduling**——invoke时probe L1D→L2→LLC locate actor，1/32概率本地执行实现data migration，exclusive flag支持write hint；(b) **Data-triggered scheduling**——cache controller在miss/eviction时自动触发，对小于cache line的objects并行执行，对大于cache line的objects一次性多行操作；(c) **Stream scheduling**——long-lived producer + data-triggered constructor copy (Fig. 10) 实现解耦执行，比täkō's pseudo-streaming减少engine instructions per edge，stream可run far ahead of consumer；(d) **跨范式交互**——PHI中task offload RMW自然触发data-triggered constructor（Fig. 4）。**解决baseline单范式调度无法满足多范式应用需求**。
  - **硬件架构层**：Leviathan在baseline cache-coherent multicore上追加per-tile near-cache engine，含：(a) dataflow fabric可编程计算资源 + 32 task-context buffer（均分防死锁）；(b) rTLB使data-triggered action获得virtual address；(c) LLC bank-index函数修改（zero LSBs）使large object单bank映射；(d) DRAM translation buffer + MC FIFO cache实现无碎片的object compaction；(e) invoke buffer + flush/pop ISA指令支持全范式；(f) stream phantom地址空间 + head-tail pointer跟踪。面积开销仅~6% vs LLC data array（32.8KB/tile），Leviathan性能达ideal engine的95.2%（within 4.8%）。**解决baseline硬件无法同时支持所有范式且不增加non-NDC workload性能干扰**——Leviathan有意保持底层cache hierarchy largely unchanged，对非NDC workload影响可忽略。跨四个case study：PHI 3.7× speedup / 22% energy reduction、Decompression 2.4× / 65%、Hash-table 2.0× / 77%、HATS 1.7× / 26%，均接近ideal engine。

## 38-SpecInfer- Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification.pdf

- baseline方法是什么？
  Baseline是现有LLM serving系统的incremental decoding方法，以及sequence-based speculative inference方法。
  
  **Incremental decoding（vLLM、HuggingFace TGI、FasterTransformer等）的全栈执行**：
  - 算法层：自回归生成——每步输入prompt + 已生成tokens，LLM forward pass输出logits，取argmax（greedy）或sample（stochastic）得到单个next token。每生成一个token需要完整的LLM参数访问一次。
  - 系统框架层：vLLM使用PagedAttention管理KV-cache + iteration-level continuous batching；HuggingFace TGI和FasterTransformer使用tensor model parallelism + pipeline parallelism分布LLM参数；FlexGen使用CPU DRAM→GPU HBM的weight offloading + pipeline execution。
  - 编译框架层：论文未明确说明。
  - kernel调度层：使用cuBLAS/cuDNN/cuTLASS标准kernel库做attention和GEMM计算。FasterTransformer使用fused attention kernel（多head batch处理）。
  - 硬件架构层：NVIDIA A10 GPU，使用标准HBM→SM的数据路径。GPU memory bandwidth是主要瓶颈（HBM访问能耗比浮点运算高2-3个数量级）。
  
  **Sequence-based speculative inference的全栈执行**：
  - 算法层：单个SSM以incremental decoding方式生成一个token序列（speculative sequence），LLM以单次forward pass验证该序列所有tokens（sequence verification）。每个位置只验证一个候选token。
  - 系统框架层：与incremental decoding相同的系统框架。SSM的参数常驻GPU。
  - 其余层次同incremental decoding。
  
  **Baseline的核心缺陷**：
  1. Incremental decoding的GPU利用率低：decode阶段每token的计算量小（matrix-vector multiplication），GPU的并行计算资源未被充分利用，大部分时间花在HBM memory access上。
  2. KV-cache内存膨胀：长序列生成的KV-cache占用大量GPU memory，限制了batch size。
  3. Sequence-based speculation的token alignment成功率低：单个SSM预测的单条token序列与LLM输出的对齐概率随序列长度指数衰减。对于stochastic decoding，single sequence的token验证成功率仅为52-57%（Table 1，top-1 from SSM）。这是因为SSM与LLM之间存在固有的model capacity gap（SSM通常比LLM小100-1000×）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  
  **SpecInfer的tree-based speculative inference and verification全栈执行**：
  - 算法层：
    - **Token tree construction**代替single sequence speculation。Expansion-based方法从单个SSM的每步top-k tokens展开为tree（利用observance: LLM的token通常在SSM的top-5内，成功率从57%→97% for stochastic）；Merge-based方法用adaptive boosting无监督地collective boost-tune多个SSM（每次用前一个SSM失败的samples训练下一个），然后merge所有SSM的输出为统一token tree。
    - **Multi-step Speculative Sampling (MSS)**用于stochastic decoding时验证token tree。对每个tree node遍历各SSM预测的分支，以概率min(1, P_LLM/P_SSM)接受token；拒绝则normalize residual后尝试下一SSM。Theorem 4.2证明等价于LLM原始分布。
    - 核心洞察：**同时考虑多个候选token（tree）而非单一序列**，大幅提高每步验证成功率。
  - 系统框架层：
    - SpecInfer基于FlexFlow runtime构建，新增Speculator（expansion-based + merge-based token tree construction）、Token Tree Verifier（tree-based parallel decoding + MSS verification）、Request Manager（CPU端iteration-level调度 + token tree merge + verification）。
    - SSM使用data parallelism分布到多GPU，LLM使用tensor model parallelism + pipeline model parallelism。
    - Continuous batching (adapt from Orca)：按iteration而非request粒度调度，新请求不需等待当前batch完成。
    - 加速机制：**LLM从incremental decoder变为token tree verifier**，每次forward pass验证多个tokens而非一个，减少LLM参数访问次数和inter-GPU通信次数。
  - 编译框架层：使用FlexFlow compiler将DNN computation graph从layer→operator→task三层抽象编译为异步task graph执行。论文未明确说明对编译器本身的修改。
  - kernel调度层：
    - **Topology-aware causal mask**：将token tree拓扑编码为causal mask，使所有tree nodes的attention计算可以fused到单个CUDA kernel中。与sequence-based causal mask（按sequence position）不同，tree causal mask按token在tree中的祖先关系决定attention允许/屏蔽。
    - **Depth-first search KV-cache management**：使用DFS order遍历token tree更新shared KV-cache，消除共享prefix的冗余计算和多序列独立KV-cache的cache conflict。DFS保证计算每个token时cache中恰好是其祖先的KV。
    - **FasterTransformer-based fused attention kernel**：修改FasterTransformer attention kernel支持tree topology batching。每个thread block处理1 request × 1 head，query加载到shared memory，thread并行计算query/key dot-product分段。
  - 硬件架构层：与baseline相同，使用标准NVIDIA A10 GPU。论文未修改硬件架构。
  
  **方法如何对应解决Baseline缺陷**：
  1. 解决"GPU利用率低"：SpecInfer利用GPU中incremental decoding未使用的spare compute资源进行token tree verification。Tree verification扩展了每step的计算量（verifying更多tokens），提高了GPU利用率。同时减少总decoding步数（每步平均verify 2-4 tokens vs incremental的1 token），降低end-to-end latency 1.5-2.8×（distributed）和2.6-3.5×（offloading）。
  2. 解决"KV-cache内存膨胀"：虽然token tree增加了额外tokens的KV-cache，但论文分析指出与长序列KV-cache相比，token tree的额外memory overhead可忽略（<1% for SSM parameters + tree nodes KV）。
  3. 解决"token alignment成功率低"：Token tree通过同时提供多个候选token（top-k expansion或multi-SSM merge），将每步至少一个token匹配的成功率从sequence-based的52-57%提高到tree-based（k=5）的96-97%（Table 1, stochastic decoding）。MSS进一步比Naive Sampling的verified tokens数提升1.2-1.3×（Table 3）。

## 39-8-bit Transformer Inference and Fine-tuning for Edge Accelerators.pdf

- baseline方法是什么？
  Baseline是现有的Transformer量化和边缘推理方法，主要包括：(1) int8 quantization for Transformer inference (如LLM.int8(), SmoothQuant)，但需要per-channel/per-vector scaling factors处理outliers，部署复杂；(2) FP8 for DNN training (NVIDIA FP8, Micikevicius et al. 2022)，但仅量化GEMM的inputs，其余操作保留高精度(BF16/FP16)；(3) int8 LoRA fine-tuning，LoRA weights需要先dequantize pretrained weights到float再与LoRA矩阵merge后做高精度floating-point运算，既不能使用高效8-bit MAC，又有精度损失。全栈执行例子：
  - **算法层**：int8量化BERT/MobileBERT推理——weights per-channel量化（scale factor），activations per-tensor量化。Softmax和GeLU用FP32精确计算（反量化→FP32计算→再量化）。LoRA微调：int8 pretrained weights + BF16 LoRA matrices (A, B)，forward时dequant pretrained weights到BF16 → merge with LoRA → BF16 GEMM。
  - **系统框架层**：HuggingFace Transformers + PyTorch eager mode。量化通过PyTorch的quantization API或自定义CUDA kernel实现int8 GEMM（如FBGEMM, CUTLASS int8 kernel）。LoRA通过peft库注入adapter。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：int8 GEMM kernel在NVIDIA GPU上通过Tensor Core int8指令执行；softmax使用cuDNN/cuBLAS浮点kernel（精确exp + division）；residual add/layer norm用CUDA element-wise kernel（FP32）。反量化/再量化在kernel边界产生额外memory round-trip。
  - **硬件架构层**：NVIDIA GPU (RTX 4090) 通过FP32/FP16 Tensor Core + INT8 Tensor Core执行混合精度运算。边缘加速器方面，baseline BFloat16 accelerator面积~11.3mm²/850mW (32×32 systolic array, 40nm)。INT8 MAC面积更小但因scaling/outlier handling需要额外硬件。Softmax除法器和指数单元（BF16）面积~5098μm²/764μW（指数）+ ~6.62×10³μm²/617μW（reciprocal）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文通过三种核心设计系统性地解决baseline的缺陷：(A) 全操作8-bit量化+Operation Fusion，(B) 8-bit Native LoRA，(C) 面积功耗高效的Posit Approximate Softmax。对比baseline的全栈执行例子：
  - **算法层**：
    (A) 使用Posit8和FP8对所有Transformer操作做PTQ量化，通过**operation fusion**将激活函数、layer normalization、residual add、attention scaling全部融合进前序GEMM，在32-bit accumulator中完成所有element-wise计算后再量化到8-bit。这消除了baseline多次反量化/再量化的round-trip误差，且不需要任何scale factor。量化顺序按对准确率影响降序：attention scaling > activation > layer norm > residual add。MobileBERT需要全部融合才能<1% loss；BERTbase不需要融合即可达标。
    (B) 将LoRA adapt到Posit8/FP8：`h = quant(W0_8 + α·quant(B16)quant(A16))x`。与baseline不同，此方法使所有GEMM均在8-bit完成（无需dequant pretrained weights），LoRA merge也在8-bit完成，forward/backward使用单一数据类型。训练内存降低约3× (500MB→165MB for MobileBERTtiny)。
    (C) Posit approximate softmax：利用posit bitwise特性——sigmoid通过bitwise NOT最高位+右移2位近似；reciprocal通过bitwise XOR with negated signmask近似；指数通过e^x = 1/sigmoid(-x) - 1构造。通过threshold截断(x < -3 → 0)和shift减ε优化保持准确率<1% loss。与baseline需要精确浮点exp/div不同，此approximation消除了面积大功耗高的除法器和精确指数单元。
  - **系统框架层**：使用HuggingFace Transformers + PyTorch 2.0+。自定义quantized GEMM将E4M3/E5M2/Posit8的clipping + rounding嵌入PyTorch autograd的forward/backward hook中（training时在GPU上仍用BF16 arithmetic模拟8-bit行为）。开源代码：https://github.com/jeffreyyu0602/quantized-training (MIT License)。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：Posit8/FP8 MAC通过浮点E5M4/E5M3 MAC实现（因decoded Posit8可映射为标准浮点格式）。Operation fusion将原本独立的多个element-wise CUDA kernel合并为单个fused kernel，减少了kernel launch overhead和memory round-trip。与baseline int8 kernel需要反量化→float→再量化不同，论文的fused kernel在accumulator中完成所有element-wise操作。Approximate softmax用bitwise operations替代精确浮点exp/div，kernel从~10个浮点operation降为~3个bitwise operation。
  - **硬件架构层**：
    - Posit8 accelerator：systolic array + vector unit，40nm综合。相比BF16 baseline，总面积减少30%、功耗降低26%（32×32 array: 11.3→7.9mm², 850→630mW）。与Hybrid FP8 accelerator对比：Posit8 MAC略大（extra 1 fraction bit），但Vector Unit因approximate softmax（bitwise vs 浮点除法器+exp）面积减少33%、功耗降低35%。
    - Approximate softmax硬件收益（200MHz, 40nm）：指数单元面积762→335μm²（62%↓），功耗237→120μW（44%↓）；reciprocal单元面积6.62→1.66×10³μm²（85%↓），功耗617→161μW（75%↓）。
    - Posit8的tapered precision优势：对接近1的值（Transformer weights/activations遵循Gaussian分布）提供更多fraction bits；对大模型（LLaMA 2 7B/13B）posits优于FP8（因wider range更好代表residual layers中的outliers）。不足是encoding/decoding有额外硬件开销（~2% area），但被vector unit的节省所抵消。

## 3-AMALI- An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs.pdf

- baseline方法是什么？
  Baseline是GCoM（ISCA '22），当前SOTA GPU解析性能模型，基于增强的interval analysis进行层次化GPU kernel性能建模。

  GCoM全栈执行例子（以Llama2-7B推理中GEMM kernel fp16为例）：
  - **算法层**：LLM推理中的GEMM运算（如attention层Q/K/V projection、FFN层linear变换），输入为fp16矩阵A(M×K)和B(K×N)，使用HMMA tensor core指令加速。
  - **系统框架层**：PyTorch实现Transformer模型，调用CUDA GEMM kernel。一次LLM推理涉及数千个CUDA kernel调用，GEMM在attention层（memory-bound）和FFN层（compute-bound）中异构分布。
  - **编译框架层**：CUDA kernel编译为SASS指令。GCoM使用SASS trace进行interval analysis，但不区分HMMA指令的modifier差异（不区分16816 vs 1688，不区分F32 vs F16 accumulator type）。
  - **kernel调度层**：GCoM使用initiation_interval = warp_size / functional_unit_lanes = 32/4 = 8统一建模CUDA core和tensor core的II，这严重高估tensor core的math_pipe_throttle（图2a）。GCoM用K-means选单一representative warp代表所有warp，不考虑LLM推理中warp间指令数可达6x差异（图5a-d）。GCoM忽略constant cache miss（imc_miss）和instruction cache miss（no_instructions）两类stall，导致VELE等element-wise kernel的cycle严重低估（图2b）。
  - **硬件架构层**：GCoM未建模implicit constant memory access（如IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]中的c[...]寻址）。A100 tensor core设计为256 FMAs/clk，HMMA.16816=2048 FMA需8 cycle，HMMA.1688=1024 FMA需4 cycle，但GCoM的II方法无法区分两者。Imc cache miss和instruction cache miss在Llama2-7B推理中占CPI stack高达70%（图4），导致GCoM出现significant underestimation。

  GCoM预测结果：Llama3-8B推理平均MAPE 127.56%，prefill phase 163.68%，decode phase 82.29%。GEMM kernel MAPE 183.95%，ELE kernel MAPE 77.81%。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  AMALI通过三项创新设计直接解决GCoM的四类缺陷：

  **(1) Throughput-based Tensor Core Model (TCM) → 解决GCoM缺陷#1和#2（initiation interval不适用于tensor core + 忽略instruction modifier）**
  - GCoM: II = warp_size / FU_lanes = 32/4 = 8，对所有指令类型统一。
  - AMALI: II_TC = FMA_count / TP_dt，其中FMA_count由HMMA modifier决定（16816=2048 FMAs, 1688=1024 FMAs），TP_dt由datatype决定（A100 fp16 = 256 FMAs/clk）。
  - 效果：AMALI正确得出HMMA.16816 CPI=8, HMMA.1688 CPI=4，GCoM无法区分。GEMM kernel MAPE从183.95%降至17.84%。

  **(2) Kernel Launch Latency (KLL) → 解决GCoM缺陷#3（constant cache + instruction cache未建模）**
  - GCoM: 未建模imc_miss和no_instructions stall。
  - AMALI: 通过micro-benchmark测量不同gridsize和blocksize下的kernel launch延迟，建立KLL = s·GS + k模型，其中s = α·BS² + β·BS + γ（A100: α=0.0036, β=0.0366, γ=1.1891）。KLL直接捕获implicit和explicit constant memory access造成的imc_miss，以及instruction cache miss造成的no_instructions stall。
  - 效果：在element-wise kernel中KLL是最主要的精度提升贡献者，ELE kernel MAPE从77.81%降至27.29%。

  **(3) Warp Instruction Distribution (ID) → 解决GCoM缺陷#4（单一representative warp无法代表LLM推理的warp特征）**
  - GCoM: 一个representative warp代表整个kernel的所有warp。
  - AMALI: ID = (max_subCoreInstr - I_SC_Repr.warp) / IssueRate，建模同一kernel内不同warp间的指令数差异造成的load imbalance（在LLM推理中可达6x）。
  - 效果：精确建模warp间不平衡产生的额外idle cycles。

  AMALI全栈执行例子（以Llama3-8B推理GEMM kernel为例）：
  - **算法层**：与GCoM相同，LLM推理中的GEMM运算。
  - **系统框架层**：PyTorch/CUDA执行LLM推理，AMALI通过NVBit插桩在运行时收集SASS trace。
  - **编译框架层**：SASS Parser提取warp profile；Cache Simulator基于trace中memory access地址仿真L1/L2 cache行为计算AMAT。
  - **kernel调度层**：Interval Analyzer将warp执行划分为intervals（计算selected+wait+short_scoreboard+long_scoreboard=SI）；Interval Parser用TCM建模math_pipe_throttle，用KLL建模imc_miss+no_instructions，用ID建模warp间的load imbalance；总cycle = SI + SI_P + KLL + ID。
  - **硬件架构层**：所有架构参数（FU延迟/吞吐、内存层次延迟、KLL系数）通过micro-benchmark预先测量。AMALI支持设计空间探索：改变TP_dt参数（256→512 FMAs/clk）时，math_pipe_throttle精确减半，预测A100→H100的kernel cycle变化平均误差仅8.2%（A100）和13.2%（H100）。

  最终结果：Llama3-8B推理平均MAPE从127.56%降至23.59%。Prefill phase MAPE 15.56%（GCoM 163.68%），decode phase MAPE 34.90%（GCoM 82.29%）。H100设计空间探索最大误差不超过23%。

## 40-Fractal- Joint Multi-Level Sparse Pattern Tuning of Accuracy and Performance for DNN Pruning.pdf

- baseline方法是什么？
  Baseline是现有的单一层级结构化稀疏模式库：(1) EW（Element-Wise/unstructured）如SparTA/Sputnik，无空间约束但需>99%稀疏才加速；(2) BW（Block-Wise）如cuSPARSE-BlockELL，在粗粒度tile级剪枝全块；(3) TW（Tile-Wise）如TileWise，在细粒度tile级剪枝；(4) VW（Vector-Wise 2:4）如cuSPARSELt，利用Sparse Tensor Core硬件。这些方法都只在稠密GEMM的多级tiling中的**某一固定层级**应用稀疏模式，且kernel均为手工编写/模板生成，无法跨硬件和算子形状泛化。

  全栈执行例子（BW baseline on GPU）：
  - **算法层**：Magnitude-based importance排序 → 按Block pattern（如32×32）在块内统一剪枝/保留 → 各block维持相同稀疏率
  - **系统框架层**：PyTorch调用cuSPARSE-BlockELL的SpMM kernel
  - **编译框架层**：cuSPARSE vendor库的手写CUDA kernel，无编译自动生成
  - **kernel调度层**：BlockELL格式：value矩阵按block连续存储 + column index per block → 每个thread block加载一个non-zero block → 执行dense block×dense block的GEMM → 结果scatter到输出。仅跳过coarse-grain block级计算，block内的fine-grain零值仍参与计算（无法利用内部稀疏）
  - **硬件架构层**：GPU CUDA Core执行，无特殊硬件单元

  Baseline的缺陷：(1) 单一层级稀疏无法充分利用多级tiling的跳过机会——粗粒度模式（BW）漏掉细粒度稀疏，细粒度模式（EW/TW）损失粗粒度数据复用；(2) 固定模式无法适应不同算子的精度-性能偏好差异；(3) 手工kernel不可扩展至新型模式和硬件后端。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Fractal提出PatternIR（循环打孔抽象）将稀疏模式统一为多级tiling上的循环穿孔组合，并用自动调优系统搜索最优多级Hybrid模式。核心创新：(1) **多级稀疏tiling**：允许同时在不同tiling层级穿孔——外层穿孔跳过整个coarse block的计算和数据加载，内层穿孔跳过fine-grain元素但保留coarse block的数据复用。例如Hybrid模式`I0^4 J0_2^4 I1^4 J1^4`：外层在J0上穿孔（column-wise coarser skip），内层在I1上穿孔（row-wise finer skip），形成复合跳过策略。(2) **自动代码生成**：PatternIR → SparseTIR → TVM → MetaSchedule pipeline自动生成优化kernel，消除手写kernel的工程负担。(3) **联合精度-性能调优**：用importance score阈值约束搜索空间，ML cost model加速评估。

  全栈执行例子（Fractal on A100 Tensor Core）：
  - **算法层**：GenTilingSpace(dense OP)生成多级tiling组合 → GenPerforationSpace对所有tiling层级搜索穿孔nnz组合 → FilterByScore（magnitude threshold）→ 多级贪婪剪枝从coarse到fine逐层执行 → 输出PatternIR如`I032 K016 64 I132 K116`（2级稀疏：K0上64选16 + K1上16选16即dense）
  - **系统框架层**：Fractal Pattern Tuner协调剪枝→代码生成→调优→评估的闭环
  - **编译框架层**：PatternIR Parsing → SparseTIR Sparse Axes声明 → Fractal-ELL存储（多级index vector链） → Sparse Iteration → TVM TensorIR → Condense(B, K轴)预聚集消除随机访问 → MetaSchedule搜索ApplyHistoryBest schedule → 自动生成CUDA kernel（含blockize for TensorCore）
  - **kernel调度层**：Fractal-ELL kernel执行时，outer sparse loop index idx0选择保留的block → inner sparse loop index idx1选择block内保留的vector → dense compute执行保留元素的GEMM → Condense预聚集数据从shared memory连续读取。与BW baseline的区别：block内可进一步穿孔（inner loop跳过），减少无效计算；而block间可保留更多block（outer loop保留比例高），保持数据复用。
  - **硬件架构层**：GPU CUDA Core或Tensor Core，无特殊硬件依赖。Condense原语可利用local memory缓存过程，数据搬运融入memory pipeline无额外开销。

  对比示例：75%稀疏率的1024×1024 GEMM
  - BW baseline：32×32 block单位，block级75%稀疏=每4个block保留1个→ coarse跳过3/4的block计算，但保留block内100%元素都计算
  - Fractal Hybrid：outer I0=4（保留2/4 column block） × inner J1=4（block内保留2/4行vector）= 2级复合稀疏 → 既跳过coarse block，又在block内跳过部分vector，更精确匹配weight tensor的内在结构，accuracy损失更小同时性能更高（Fig.15 Pareto frontier）。

## 41-Splitwise_Efficient_Generative_LLM_Inference_Using_Phase_Splitting.pdf

- baseline方法是什么？
  Baseline是当前LLM推理集群的标准部署方式——所有请求的prompt computation和token generation两个阶段在同一台机器上完成，使用mixed continuous batching。全栈执行例子：
  - **算法层**：生成式LLM (BLOOM-176B或Llama2-70B) 的autoregressive推理。每个请求经历两个阶段：(i) prompt phase——所有输入prompt tokens并行通过模型forward pass生成第一个token和完整KV-cache；(ii) token generation phase——逐token串行生成后续输出，每step仅处理最新的一个token加上累积的KV-cache。
  - **Serving系统框架层**：使用vLLM的mixed continuous batching（或Orca的continuous batching）。所有机器构成一个统一的机器池，每台机器运行相同的mixed batching——在每次forward pass迭代前做出调度决策，prompt和token phase可以在同一个batch中混跑。Prompt phase优先级更高，可能preempt token phase以保证TTFT SLO。机器间独立调度，无跨机状态迁移。
  - **编译框架层**：论文未明确说明（使用标准PyTorch eager mode运行vLLM）。
  - **kernel调度层**：每台DGX-H100内8×GPU通过tensor parallelism (TP) 切分模型层。Prompt phase：KV-cache在本地GPU HBM中生成和存储。Token phase：每step从HBM读取整个KV-cache执行attention计算（compute bound vs memory bandwidth bound根据batch size变化）。batch size受GPU HBM容量（80GB）限制——KV-cache随batch size和序列长度线性增长。
  - **硬件架构层**：DGX-H100 (8× H100 80GB, 700W/GPU, NVLink 100Gbps intra-node)。Prompt phase在高batch size下compute-bound，功耗接近TDP；Token phase memory bandwidth/capacity bound，功耗仅约TDP的50%，GPU compute单元利用率低。

  Baseline的缺陷：
  1. **资源浪费**：token generation phase严重underutilize GPU compute（memory bound），却占用昂贵的H100 GPU，导致Perf/W和Perf/$低下。
  2. **batching效率低**：Mixed continuous batching下大部分时间（60-70%）仅batching ≤20个active tokens；coding workload中>20%时间仅运行单个token（batch=1）。大prompt batch影响TTFT，但小token batch无法充分利用GPU throughput。
  3. **同机混跑干扰**：Prompt phase（compute-intensive）和token phase（memory-intensive）在同一GPU上混跑时，prompt batch会延长同batch内token phase的execution time，影响TBT和E2E latency tail。
  4. **硬件换代不匹配**：H100相比A100 compute增长3.43×而memory bandwidth仅增长1.64×、capacity无增长——对token phase而言H100的额外compute无关紧要，却增加了75%的功耗和116%的成本。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Splitwise将LLM推理的两个phase拆分到分离的机器池（prompt pool / token pool / mixed pool），通过两级调度器和优化KV-cache跨机传输实现phase-specific硬件资源管理。对比baseline的全栈执行例子：
  - **算法层**：算法不变（相同模型、相同推理逻辑）。Splitwise不修改模型结构或推理算法，仅在系统层面重组执行位置。KV-cache通过InfiniBand无损传输保证accuracy不变。
  - **Serving系统框架层**（核心创新）：(i) 新增CLS：以JSQ策略为每个请求同时分配一对prompt和token机器，动态管理三个机器池（prompt/token/mixed），根据pending queue长度按需从mixed pool借调机器，队列排空后归还；(ii) 改写MLS：prompt MLS限制total batch tokens≤2048（throughput-optimal上限），token MLS尽可能增大batch（直到GPU内存满），mixed MLS优先prompt + age-based token priority防止starvation；(iii) KV-cache传输：使用MSCCL++实现per-layer异步传输（与下一层prompt计算overlap），小prompt使用serialized transfer，大prompt使用layer-wise transfer将传输延迟从64% TBT降至16.5%；(iv) 混合池：允许prompt和token机器在需求变化时动态切换角色，消除固定分区造成的fragmentation。
  - 解决baseline缺陷1（资源浪费）：Splitwise将memory-intensive token phase卸载到较低compute能力的机器（如A100 for token in Splitwise-HA，或power-capped H100 in Splitwise-HHcap），将compute-intensive prompt phase保留在H100上。Token machine GPU即使仅50% TDP功耗也能满足TBT SLO（Fig.9：token phase power cap 50%几乎无延迟影响），而prompt机器全力运行prompt phase，双方都达到高利用率。Splitwise-AA用廉价的A100集群以相同power/cost获得2.15× throughput于Baseline-A100。
  - 解决baseline缺陷2（batching效率）：Phase分离后，token machine专门处理token generation——可无干扰地将batch size推到GPU HBM容量上限（如batch=64），因为token phase batching对TBT影响极小（batch=64仅2× TBT vs batch=1）。Prompt machine专门处理prompt computation——不受token phase占用的内存和计算资源限制。Fig.17显示Splitwise-HH的prompt machine在高负载下运行更大prompt batch、token machine运行更大token batch，而baseline机池70%时间仅batching≤15 tokens。
  - 解决baseline缺陷3（同机混跑干扰）：两个phase物理隔离到不同机器，prompt computation不会拖慢token generation的TBT，反之亦然。E2E latency仅增加0.8%（per-layer KV-cache传输overhead），而second-token latency overhead从serialized transfer的64%降至Splitwise的16.5%。
  - 解决baseline缺陷4（硬件换代不匹配）：Splitwise-HA方案——H100做prompt machine（利用3.43× compute），A100做token machine（利用相同的80GB HBM和可接受的memory bandwidth 1.64×），总cost和power均低于全H100集群。Splitwise-AA方案——customer可用更廉价且供应充足的A100实现相同power budget下比Baseline-H100更高的throughput（1.4× at 20% lower cost in iso-cost design）。
  - **编译框架层**：论文未明确说明（不涉及编译框架修改）。
  - **kernel调度层**：KV-cache transfer的per-layer异步机制：各transformer layer计算完成后MSCCL++ one-sided put立即经InfiniBand推送该层KV-cache到token machine，同时GPU继续执行下一层prompt computation（计算-通信overlap）。Token machine通过semaphore等待所有层传输完成后开始token generation。vLLM内KV-cache block-by-block传输，连续block合并发送减少transfer次数。Tensor parallelism仅限intra-machine（NVLink），KV-cache transfer为inter-machine（InfiniBand）。
  - **硬件架构层**：硬件不变（DGX-A100/H100 + InfiniBand）。Splitwise-HHcap额外使用GPU power capping（token machine GPU cap至70% TDP），利用token phase对power cap不敏感的特性（Fig.9：power cap从700W→350W几乎无TBT影响）降低集群peak power和energy cost。异构方案Splitwise-HA要求H100-A100间InfiniBand互联（paper认为技术上可行但当前云环境不常见）。

  对比总结：Baseline将两个特征截然不同的phase混在同一硬件上运行，导致compute、memory、power三种资源的利用率低下。Splitwise通过phase splitting将硬件资源与phase特性精确匹配——compute-intensive phase用高compute硬件，memory-intensive phase用高memory/cost比硬件——同时通过两级调度和优化KV-cache传输将splitting的overhead降至可忽略（<1% E2E）。效果：iso-power throughput-optimized集群达到2.35× throughput提升；iso-cost throughput-optimized下1.4× throughput at 20% lower cost；iso-throughput power-optimized下25% lower power at same cost。

## 42-Cambricon-D_Full-Network_Differential_Acceleration_for_Diffusion_Models.pdf

- baseline方法是什么？
  Baseline是Diffy [15] 提出的逐算子差分计算（per-operator differential computing）加速器。Diffy在CNN推理中对每个卷积算子独立执行spatial差分计算：利用空间相邻像素的数值相似性计算delta值，用窄bitwidth PE（如INT5）处理delta以减少计算量，依赖bit-serial架构省略leading zeros。Diffy的dataflow对每层执行：加载weights→从off-chip DRAM加载上一时间步raw activation X_{t-1}→on-chip计算delta ΔX_t=X_t-X_{t-1}→PE array执行差分卷积 Conv(ΔX_t)→输出delta ΔY_t转回raw值 Y_t=Y_{t-1}+ΔY_t→写回DRAM→加载raw值执行ReLU等非线性激活。全栈执行例子：
  - **算法层**：Spatial differential computing——对图像空间维度相邻像素计算delta值，利用delta数值范围窄（INT5可表示）来用更低位宽PE加速卷积。对每个卷积算子独立应用差分，但遇到非线性激活（如ReLU/SiLU）时必须先将delta转回raw值（因F(Y+ΔY)≠F(Y)+F(ΔY)），导致差分计算被非线性层"阻断"，只能在相邻非线性层之间做逐段（per-operator）差分。
  - **系统框架层**：论文未明确说明（Diffy为专用加速器，无Serving框架）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：Diffy的dataflow在每层执行6-9步：Load W→Fetch X_{t-1}（FP16 from DRAM）→Compute ΔX_t on-chip→Write X_t to DRAM→PE Array: Conv(ΔX_t)（INT5精度）→Fetch Y_{t-1}（FP16 from DRAM）→Compute Y_t=Y_{t-1}+ΔY_t→Write Y_t to DRAM→SFU: ReLU(Y_t)（FP16）。每层需2次FP16 raw值从DRAM加载（X_{t-1}和Y_{t-1}）+2次FP16 raw值写回DRAM（X_t和Y_t）。全部激活值约1.1GB，无法全部on-chip存储。每个timestep的memory traffic因此暴增5.78×，导致总体性能下降23.4%（memory增5.78×抵消了计算减少3.3×的收益）。
  - **硬件架构层**：Diffy-like PE array采用SIMD systolic array（类似TPU），每PE支持固定窄bitwidth（INT5），省略leading zeros实现加速。无outlier处理机制——直接丢弃超出INT5范围的delta值，导致在扩散模型上精度严重退化（GUID512 precision从87%降至43%）。PE array与baseline同样规格——128×128, 1GHz, 等效A100吞吐(3×10^14 FLOPS fp16), 1.5TB/s HBM。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Cambricon-D提出三个核心设计解决Diffy baseline的缺陷。全栈执行例子：
  - **算法层**（解决缺陷：spatial→temporal differential + 全网络差分能力）：
    (a) Temporal differential：用跨时间步的delta（ΔX_t=X_t-X_{t-1}）替代Diffy的spatial delta。扩散模型中间激活含噪声，空间平滑性差，spatial delta数值分散；temporal delta数值范围远更窄（可INT3表示，entropy降低2.11×）且无精度损失。
    (b) Full-network differential via sign-mask：关键创新在于将ReLU(SiLU近似为ReLU，<0.5%精度损失)的计算从raw值域移到delta值域。利用差分ReLU近似：ΔY'_t = ΔY_t · sgn(Y_{t-1})——在Stable-Diffusion中99.59%情况下成立。只需从DRAM加载1-bit sign tensor（而非16-bit raw值），memory traffic从5.78×降至仅modest rise（1/2.92~1/5.68× vs DiffyDF）。这使得delta可以"无缝转发"穿过整个网络（全网络差分），不被非线性层阻断。
    (c) Group Normalization差分：每timestep独立计算μ_G和σ²_G，取相邻两timestep平均值用于差分GN，无精度损失。
    (d) Attention层（仅占0.9%计算时间）：直接回退到raw值计算，避免为极小开销设计复杂差分方案。
  - **系统框架层**：论文未明确说明（专用加速器，无Serving框架）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**（sign-mask dataflow，5步替代Diffy的9步）：
    Step 1: Load W（FP16 weight）到on-chip buffer
    Step 2: PE Array使用on-chip delta ΔX_t（来自上层SFU输出）执行差分卷积 Conv(W, ΔX_t)（INT3×FP16）
    Step 3: 从DRAM加载1-bit sign tensor Sgn_{t-1}（而非FP16 raw Y_{t-1}），memory traffic降至1/16
    Step 4: ΔY_t写回DRAM供NDP engine更新raw值和sign bits
    Step 5: SFU执行差分ReLU——ΔY'_t = ΔY_t AND Sgn_{t-1}（sign=1保留，sign=0 mask为零）
    关键差异：无需加载/写回FP16 raw值(X_{t-1}, Y_{t-1}, X_t, Y_t)，每层省去4次大bitwidth DRAM访问。NDP engine在DRAM侧自动更新raw值和sign bits（int2fp→decompress→FP adder read-add-write），delta传输经Compression Unit压缩（inlier list+outlier list+bitmap格式）。
  - **硬件架构层**（outlier-aware PE array解决缺陷：Diffy丢弃outlier导致精度崩溃）：
    (a) Hardware-software co-design量化：激活tensor沿inner product维度等分为group，每组最多m=60个FP16 outlier（fp×fp乘法器处理），其余INT3 inlier（int×fp乘法器处理）。超过m的outlier clip为INT MAX/MIN作inlier处理（概率<1%）。此结构化约束使得outlier处理可与inlier同步锁步执行，避免传统outlier架构的crossbar gather（OLAccel中71%面积overhead）和异步同步开销（实验中70.21%周期浪费在stall等待）。
    (b) PE multiplier group：每PE含m=60个int3×fp16乘法器+n=4个fp16×fp16乘法器。fp2int量化电路→leftmost outlier selection（收集前m个OF flag位置）→inlier走int×fp、outlier走fp×fp→adder tree汇总。
    (c) SFU：执行sign-mask AND操作（极简电路，仅占1.08%面积/1.05%功耗）。
    (d) NDP engine（DRAM侧）：int2fp模块→decompressor buffer→FP adder array→read-add-write到DRAM bank，同时更新separate sign bit tensor。避免了将整个raw tensor加载到core侧的开销。
    (e) 总面积overhead仅3.6%（vs 无差分计算经典设计），PE array面积14.66%，SRAM buffer占~84%。7nm工艺：16.24mm², 73.47W。

  效果对比：Cambricon-D vs DiffyDF（同PE但无sign-mask）——memory traffic从增加227%~1058%降至1/2.9~1/5.7倍的增长；速度从slowdown 60~77%变为speedup 1.46×~2.38× over A100。Cambricon-D vs DiffyAll（Diffy dataflow+Diffy PE）——DiffyAll speedup仅71%~96%（实质为slowdown），Cambricon-D speedup 146%~238%。

## 43-Trapezoid- A Versatile Accelerator for Dense and Sparse Matrix Multiplications.pdf

- baseline方法是什么？
  Baseline是三类各针对特定sparsity范围的矩阵乘法加速器，它们各自在目标sparsity range表现良好但无法跨range有效工作：(1) **TPU (Dense)**：2D spatial array, 128×128 MACs, IP dataflow——A stationary in PEs, B columns vertical stream, horizontal links累加partial products。D×D时quadratic compute (P² MACs/cycle) + linear communication (P I/O/cycle), 高compute intensity和data reuse，但sparse时zeros浪费time+energy，HS×HS比Trapezoid慢4134×；(2) **SIGMA (MS)**：IP-based, extends TPU with B Benes distribution network + reduction tree。Pack A's sparse rows, stream B columns, all-to-all route B to matching A nonzeros。仅exploit A-side sparsity, B fed uncompressed。MS×D表现好，但MS×MS limited（不exploit B sparsity），HS inputs快速degrade；(3) **Flexagon (HS)**：Gustavson-based, 64 MACs+1MB cache, MRN可switch IP/Gustavson/OP modes。HS×HS表现好（cache reduces B traffic），但limited peak compute throughput（仅64 MACs），D/MS inputs远慢于TPU/SIGMA。且HS accelerators使用crossbar-based interconnect，无法scale到高throughput。
  全栈执行例子（以MS×MS为例）：
  - **算法层**：矩阵乘法 C = A × B，A和B均有20% density，IP-based dataflow遍历M×N次intersections（每个row-column pair做bitvector AND找匹配k坐标）。
  - **系统框架层**：论文未明确说明（专用加速器，无Serving/编译框架）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**（TPU IP）：for m in [0,M) spatial Y → for n in [0,N) → for k in [0,K) spatial X: C[m,n] += A[m,k] * B[k,n]。每PE row处理1行A×1列B，1 fiber intersection per cycle。A和B各20% dense时，仅1/25=4%的intersections effectual，96%的multipliers idle。
  - **kernel调度层**（SIGMA IP-based）：Pack 4 A rows into PE row registers（exploit A sparsity），每cycle stream 1 B column。4 fiber intersections per cycle。但B仍uncompressed（含zeros），utilization受限于B density（20% dense→~20% utilization）。
  - **硬件架构层**：TPU用horizontal links做reduction（简单但仅支持1 output/cycle）。SIGMA加Benes network（30% area overhead）+ reduction tree支持多output。Flexagon用MRN（可switch reduction/merge mode）+ small cache（1MB）+ crossbar interconnect（灵活但不scalable）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Trapezoid通过在一颗芯片上集成4种dataflow+3种硬件扩展，解决baseline"各target一种sparsity, 在其他range低效"的根本缺陷。设计核心原则：(1) 对D/MS保留2D spatial array高throughput优势，通过MFIU+双Benes网络扩展MS range；(2) 对HS通过multi-level memory hierarchy+ Gustavson dataflows降低data movement，同时复用D/MS硬件；(3) 面积overhead modest（2.0× TPU, 1.3× SIGMA），换来全range consistent performance。
  对比baseline的全栈执行例子（以MS×MS为例）：
  - **算法层**：同样的矩阵乘法，但通过TrIP dataflow的multi-fiber intersection大幅提升utilization。
  - **系统框架层**：论文未明确说明。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**（TrIP dataflow）：for n1/tiles → for m1/tiles → for k1/tiles → for [n_l, n_h) = dynamic split of N0 → for [m_l, m_h) = PE row (spatial Y) → for m0 in [m_l, m_h) (spatial X) → for n0 in [n_l, n_h) (spatial X) → for k0 in [0, K0) (spatial X, MRN leaf)。关键差异：(a) Intersect 4 A rows × 4 B columns = 16 fiber intersections per cycle（vs TPU的1个, SIGMA的4个），即使仅25% intersections effectual (1/25概率)仍可填满128 multipliers；(b) Dynamic B column packing——预处理阶段用popcount on A&B bitmask计算每PE row的effectual computations数，取不超过128的最大B列数，自适应A/B sparsity；(c) Merge-reduction tree sliced为多个subtrees，每个subtree产出一个C元素，支持scatter-output到banked buffer。效果：MS×MS时utilization 2.1× SIGMA, 13.3× TPU。
  - **kernel调度层**（TrGT dataflow, for HS×HS）：for n1/tiles → for m2/tiles (B tile on-chip) → for m1 in M1 (spatial Y, PE row) → for m0 in M0 (spatial Y, PE subrow) → for k in K (leader-follower) → for n0 in N0: B_tmp[k,n0] = B[n1,k,n0] → merge tree归并 → reduction loop over N0→K。关键设计：PE row 分为4 subrows, 每条处理1行A，gather对应的B rows（temporal, 1 element/cycle），利用cache捕获B的irregular reuse（similar rows of A→repeated B rows access）。
  - **kernel调度层**（TrGS dataflow, for HS×MS/HS×D）：for n2/tiles → for m1 in M1 → for m0 in M0 (spatial Y) → for k in K → for n1 in N1 (cacheline) → for n0 in N0 (spatial X within cacheline): C += A[m,k] * B[k,n]。PE row（非subrow）用全部128 multipliers，从cache spatial stream B row（16 contiguous nonzeros/cycle），4× higher throughput than TrGT in HS×MS。
  - **硬件架构层**（核心创新解决baseline硬件局限）：
    (a) **MFIU**：AND gate array→prefix sum tree→shift unit。4×4=16 fiber intersections/cycle。对比SIGMA仅做B routing不做intersection（intersections在PE内部implicitly完成但精度受限），Trapezoid explicit intersection免去ineffectual multiplications同时生成routing metadata。Prefix sum tree产生的effectual computation indices直接驱动双Benes网络routing，compact design。
    (b) **双Benes分布网络**（A+B各一个，vs SIGMA仅有B）：TrIP需要同时route A和B匹配的非零值到multiplier，双网络避免sequential routing bottleneck。A Benes reuses TrGT/TrGS模式下的A broadcast功能。
    (c) **Multi-level memory hierarchy**（4MB/cluster cache + 8KB/PE row local buffer）：解决Gustavson dataflow的核心瓶颈——gather B rows across irregular k coordinates。4 bank local buffer提供4 gather reads/cycle（vs prior HS accelerators使用crossbar）。Cache 4-cluster organization用4个32×32 crossbar代替1个128×128 all-to-all network，大幅降低面积。
    (d) **MRN mode switching**：reduction mode→adder tree（Dense IP/TrIP/TrGS），merge mode→comparator tree（TrGT）。Flexagon已提出MRN但Trapezoid在更大规模array（128×128 vs 64 MACs）上实现并增强with banked local buffer for higher output bandwidth。
    (e) **面积效率**：Trapezoid 81.9 mm²（16nm），sparsity handling hardware（MFIU+双Benes+MRN）占PE row一半面积，但仅30% area increase over SIGMA换来2.1× MS×MS improvement + HS support。Compute仍占significant fraction（vs Flexagon 80%面积在SRAM）。
  效果：perf/area gmean 19.7× TPU, 4.3× SIGMA, 2.9× Flexagon。D×D仅2.0× away from TPU（因sparsity hardware面积overhead），HS×HS仅1.2× away from Flexagon（因Trapezoid的scalable cache设计）。

## 44-MECLA_Memory-Compute-Efficient_LLM_Accelerator_with_Scaling_Sub-matrix_Partition.pdf

- baseline方法是什么？
  Baseline是GPU（NVIDIA V100）上运行标准Transformer/LLM推理，以及SOTA Transformer加速器（SpAtten、FACT、Sanger）。baseline的核心缺陷：LLM的autoregressive生成（每次生成一个token需完整遍历全部模型权重）使得线性层（QKV+FFN）占>98%计算量和内存访问，而现有加速器要么只优化attention层（Sanger等），要么无法同时优化线性层的内存和计算（SpAtten仅computation sparsity，FACT仅encoder优化）。全栈执行例子（LLaMA-7B单token autoregressive推理）：
  - **算法层**：标准Transformer decoder推理（Algorithm 1）。每个生成的token作为新输入与所有历史token拼接→通过所有decoder layers。每层执行：QKV Linear（3次MatVec，输入1×4096，权重4096×4096×3）→Attention（QK^T/√dk→softmax→×V）→FFN Linear（2次MatVec，gate+up projection 4096×11008，down projection 11008×4096）。生成32个token需~14GB权重数据访问和>400 billion operations。权重矩阵以FP16/INT8存储于GPU global memory（HBM），每token计算需从HBM完整读取全部权重。
  - **系统框架层**：PyTorch + HuggingFace Transformers library。推理使用标准的`model.generate()`循环——每iteration forward一次完整模型，无memory复用优化。GPU kernel使用cuBLAS/cuDNN的GEMV实现（decode阶段batch=1，compute-bound→memory-bound因低arithmetic intensity）。
  - **编译框架层**：论文未明确说明（使用标准PyTorch eager mode或torch.compile）。
  - **kernel调度层**：NVIDIA GPU上执行GEMV kernel（矩阵-向量乘法）。由于decode阶段batch_size=1，每次MatVec的FLOP-to-byte ratio极低（O(1) per weight byte），kernel成为memory-bound。GPU的L1/L2 cache无法容纳LLM的完整权重矩阵（LLaMA-7B单层linear权重~90MB > L2 cache ~40MB），必须每次token从HBM重新读取，浪费memory bandwidth。GPU虽有FP16 Tensor Core加速，但decode阶段低利用率。
  - **硬件架构层**：NVIDIA V100 GPU：32GB HBM2（900 GB/s），125 TOPS INT8 peak，5120 CUDA cores + 640 Tensor Cores。V100的通用架构无法利用权重矩阵内部的sub-matrix相似性——每个GEMV独立计算所有MACs，不感知weight中SS/DS的可重用关系。SOTA加速器：SpAtten（40nm, 1.55mm², token pruning for attention，仅computation优化不减少weight memory），FACT（28nm, 6.03mm², sparsity on linear layer for encoder，decoder autoregressive degradation >5×），Sanger（55nm, attention-only optimization）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MECLA通过SSMP算法-硬件协同设计同时解决LLM推理的memory wall和compute wall：(1) SSMP将权重矩阵分解为source sub-matrix + 可scale生成的derived sub-matrix，用scalar替代重复weight存储，memory访问量降至16.4%（83.6% reduction）；(2) MECLA专用PE array architecture利用SSMP的PSum可scale性质，通过on-the-fly regrouping + dual-mode mapping将计算量降至27.8%（72.2% reduction）。对比baseline的全栈执行例子（LLaMA-7B单token推理）：
  - **算法层**：SSMP partition替代完整权重存储。权重矩阵W ∈ R^{4096×11008}不再直接存储，而是分解为{WSS, S}：WSS ∈ R^{(4096/(8×4))×(11008/(8×4))×8×8}，S ∈ R^{128×344×4×4}（配置(8,8,4,4)）。推理时每个region的16个DS block（4×4）由1个SS block乘以16个scalar实时生成：DS[i,j] = SS × S[i,j]。SSMP-oriented fine-tuning（Algorithm 2）通过freeze预训练权重+仅训练微小WSS/S参数+forget factor σ正则化→将预训练LLM无损转换为SSMP格式。与LoRA本质区别：LoRA为训练效率，SSMP为推理效率。相比KD/MiniLLM的知识蒸馏（将13B→固定7B），SSMP提供从5B到7B的连续压缩选项且保留PSum reuse潜力。
  - **系统框架层**：MECLA RISC-V core + 全流水数据流替代GPU kernel launch。推理循环中RISC-V core fetch instructions from external host→auxiliary unit做embedding/normalization preprocessing→data buffer broadcast input到8 PE clusters→SS buffer + DS buffer distribute weight→PE array计算SSMP MatMul→scaling accumulator执行PSum × scalar→cross-cluster reduction→output。与GPU不同，MECLA不需要每token从HBM重载完整权重——SS matrix和scaling scalar存于on-chip SRAM（1.25MB总容量，权重压缩后LLaMA-7B单层仅需~100KB），external DDR4仅用于KV cache和initial weight load。
  - **编译框架层**：论文未明确说明。RISC-V指令由根据模型维度离线生成的指令序列驱动，非动态编译。
  - **kernel调度层**（核心创新）：对比GPU的memory-bound GEMV kernel（每个weight byte仅做~1 FLOP），MECLA PE array通过三项kernel调度技术实现memory-compute co-optimization：
    (a) **On-the-fly Matrix Regrouping**：将散布在完整权重矩阵中的SS相关行/列（间隔nx/ny）重排集中到同一PE cluster，使PSume reuse不跨cluster通信。例如配置(8,8,4,4)时，同一SS对应的4×4=16个DS的output channels被regroup到一起，PE array可batch处理。
    (b) **Dual-mode Mapping**：根据nx vs ny动态选择outer-product或inner-product reuse模式。nx>ny→outer-product（4×4 PE中1行PE计算1 shared PSum，scaling multiplier乘4个scalar）；nx<ny→inner-product（先算scaling×input，再乘weight）。自适应最大化reuse效率。
    (c) **PSum Reuse**：利用SSMP的核心性质——SS和DS对相同input channels的PSum相差scalar倍。传统GEMV需为每个output channel独立计算16个8-MAC=128 MACs；MECLA仅需8 MACs（SS weight × input） + 16 scaling mults = 24 ops。示例中功耗降85.6%。配合gated multiplier（未使用multiplier断电），进一步节能。
  - **硬件架构层**（与kernel co-design）：MECLA处理器专为SSMP的PSume reuse设计：
    (a) **4×4 PE + 4×4 Scaling Multiplier阵列**：PE做matrix-vector MAC（input broadcast, weight unicast），输出的4个32-bit PSum→送入4×4 scaling multiplier（每PSum对应4个multiplier，乘以最多4个scaling factor）。16 scaling multipliers per PE cluster，8 cluster总共128 multipliers。
    (b) **Scaling Accumulator Crossbar**：8个scaling multiplier通过内部crossbar通信共享PSum，支持最多32次PSume reuse（超越单PE cluster的4次硬件上限，由SSMP配置可能要求nx=4, ny=8=32 reuse）。
    (c) **Hierarchical SRAM**：256KB data buffer（broadcast到所有clusters）+ 512KB SS buffer + 512KB scaling scalar buffer（distributed per cluster）。权重压缩后LLaMA-7B完整模型可从~14GB降至~1-2GB（具体取决于SSMP配置），使模型可完全存于on-chip+external DDR而无需HBM。
    (d) **RTL验证**：28nm CMOS，22.02 mm²，2.87W@1GHz。PE array占26% area/52.3% power（computation-dominated design vs prior accelerators' SRAM-dominated）。能效7088 GOPS/W（average），113.14× V100/12.99× SpAtten/1.62× FACT（technology normalized to 28nm）。
  - **MECLA vs GPU关键差异总结**：GPU的GEMV受限于memory bandwidth（~900 GB/s for V100），每token需重新加载完整权重；MECLA利用SSMP将weight压缩至1/6，且on-chip SS buffer消除重复access。GPU无PSum reuse能力（每个output channel独立计算）；MECLA的scaling multiplier阵列将重复计算转化为scalar multiplication（能耗极低的scaling mult替代高能耗的full MAC）。GPU通用架构无法利用权重内部相似性；MECLA通过SSMP算法与硬件PE array co-design，在ISCA'24首次实现LLM推理的memory-compute co-optimization。

## 45-VGA_Hardware_Accelerator_for_Scalable_Long_Sequence_Model_Inference.pdf

- baseline方法是什么？
  Baseline是H3模型（SSM-based全局卷积模型，替代Transformer的self-attention层）在GPU/TPU上的软件推理执行。H3 block的ROI由FFT-based convolution + state passing + pointwise operations构成，全程跑在GPU/TPU通用计算单元上。全栈执行例子（GPU A100-40GB, H3-GPT模型, 序列长度128K）：
  - **算法层**：H3 block (Fig. 3)：输入x(l×d)→3个FC层→Q/K/V。对每列(K_i, V_i)：先1D Conv on K_i→PointMult(K_i, V_i)→SSMConv(含FFT-based global convolution + state passing)→PointMult(Q_i, result)→FC输出。SSMConv的filter由SSM参数(A=diag(C^m), B∈C^{m×1}, C∈C^{1×m})通过K = {CA^0B, CA^1B, ..., CA^{L-1}B}生成。State passing (Eq. 3) 将长序列N划分为C个chunk，每个chunk L=2048内独立做FFT卷积，通过状态向量x_c在chunk间传递上下文（State Update: x_c = A^L·x_{c-1} + Mux·u_c; Output Projection: y_c = Mxy·x_{c-1} + K∗u_c + D·u_c）。
  - **系统框架层**：PyTorch + 自定义CUDA kernel。H3官方repo (https://github.com/danfu09/H3) 提供FFT convolution CUDA kernel，但论文必须自行实现state passing算法（官方repo未提供）。GPU batch size=8 (H3-GPT)/16 (H3-Speech)。
  - **编译框架层**：论文未明确说明。CUDA kernel通过nvcc编译。
  - **kernel调度层**（核心bottleneck）：ROI占H3 block执行时间的大部分（Fig. 5）：FFTConv 42% + State Passing 26% + Pointwise 32%。三类操作均memory-bound：(a) FFTConv——FFT算法每stage需全序列读写+barrier同步，即使数据fit in共享内存，DRAM bandwidth utilization仍低，compute utilization (TF-32 FLOPS)极低；(b) State Passing——Mux/Mxy矩阵维度(L×m)和(m×C)高度skewed（m=64很小），arithmetic intensity极低（batched matrix multiplication机会有限）；(c) Pointwise operations——纯memory-bound。Kernel fusion受限于SRAM容量：Mux/Mxy完整矩阵（大小正比于L×m≈131K elements FP32）占用大量shared memory，使得全融合kernel不可行。GPU DRAM traffic极大：每个chunk的FFT需要反复访问全局内存。
  - **硬件架构层**：NVIDIA A100-40GB (826mm², 400W, HBM2 1555GB/s, 108 SMs, 40MB L2)。TPUv3 core (324mm², 225W, HBM2 450GB/s)。两者均有充足SRAM（A100每SM 192KB shared memory），但对于128K长序列的完整Mux/Mxy矩阵，SRAM容量不足（需~数百KB per hidden dimension），导致必须spill到DRAM。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  VGA提出专用硬件协处理器，利用Vandermonde矩阵的数学性质（所有列/行可从首列/行通过循环乘法生成），通过on-the-fly矩阵生成消除SRAM容量瓶颈，实现ROI全融合硬件执行。对比baseline的全栈执行例子：
  - **算法层**：H3模型算法不变（VGA执行精确计算，无近似，无精度损失）。SSMConv仍使用Generalized Cooley-Tukey FFT + state passing，但计算全部offload到VGA硬件。VGA的参数生成逻辑与SSM的递归性质深度绑定——利用A的diagonal性质，Mxy的所有行 = 第一行 × A的各次幂；Mux的所有列 = 第一列 × A的各次幂（Eq. 4-5）。
  - **系统框架层**：VGA作为与GPU/TPU共享内存的协处理器（Fig. 6a）。Host accelerator执行H3 block的FC层/FFN/LayerNorm（compute-intensive, 适合GPU/TPU），ROI通过memory-mapped register offload给VGA。初始化：host分配Q/K/V/output/SSM参数/指令的内存区域→VGA DMA engine populate 32 TLB entries/PE。通信：GPU用write-through policy写入→VGA DMA读取处理→VGA写回→GPU invalidate stale cache。Workload沿hidden dimension并行：每个PE处理h/#PE个连续hidden dimension，PE间无同步。128 PE可访问8GB内存空间（2MB page），支持1M token序列。
  - **编译框架层**：VGA使用16B自定义ISA（配置指令+可执行指令）。指令预编码CCU mode/D-SRAM地址/访问pattern/DMU操作，不含数据（数据通过vector register传递）。Frontend issue logic支持loop参数化（metadata自动更新row/column地址）。无动态编译——指令离线生成。
  - **kernel调度层**（核心解决方案）：
    (a) **FFTConv全硬件流水执行**：列FFT (BF mode, 每个CCU用CMult乘twiddle factor + BF加减)→CTF Multiplication (CTFGen mode, 偶数CCU生成CTF→unidirectional传奇数CCU; CMult mode, 奇数CCU乘CTF与列FFT结果)→行FFT (BF mode)→乘filter K_f (CMult mode)→IFFT (BF mode)。关键消除baseline瓶颈：无需barrier同步（硬件pipeline自然流动），无DRAM spill（D-SRAM存chunk data），bank conflict消除（circular row rotation让行列访问均跨不同bank）。
    (b) **State Update全硬件流水执行**：CMult mode计算A^L·x_{c-1}→Update mode：每CCU赋一行Mux，N/M初始元素加载到register→广播u_c元素e到所有CCU→每cycle各CCU on-the-fly生成一个Mux元素（M×N→new N）并乘e→累加至accreg(预设A^L·x_{c-1}对应元素)→L cycle后写入SRAM。
    (c) **Output Projection全硬件流水执行**：Projection mode：每CCU赋一列Mxy，x_{c-1}对应元素加载→每cycle生成列元素乘积累加→m次accumulation后经unidirectional connection传播partial sum→写入SRAM→Residual mode加总FFTConv输出+D·u_c。
    (d) **On-the-fly Vandermonde矩阵生成**（最核心创新）：利用Vandermonde性质，Mxy的整列由第一个元素N和scaling factor M通过M×N→M循环生成；Mux整行同理；CTF矩阵的后续元素由前两个元素复数乘法生成。仅需存首行/列+factor，SRAM需求从存储完整矩阵降低410×，总SRAM容量减少5×。这使得全kernel融合在有限SRAM下成为可能。
    (e) **DMA overlap**：DMA Engine在Projection/Update mode期间（仅D-SRAM读或写一方active）并发执行，重叠数据搬运与CCU计算。EoM指令flush pipeline防止structural hazard。
    (f) **结果**：9.7× DRAM traffic reduction vs GPU；average FLOPS utilization 78%（GPU kernel远低于此）；memory bandwidth不再是瓶颈。
  - **硬件架构层**（与kernel co-design）：
    (a) **CCU (Complex-number Compute Unit)**：7种fully-pipelined mode的reconfigurable连接——CMult(4mul+2add, 复数乘法)、BF(CMult+adder butterfly)、CTFGen/Projection/Update(on-the-fly生成+MAC)、Residual(加法)、RMult(实数乘法)。FP32全精度。Register file (6 complex numbers)存twiddle factor/partial sum/Vandermonde生成状态。
    (b) **1D CCU array + systolic通信**：k=32 CCUs，正常模式各CCU独立处理input vector对应index。需多CCU协作时（CTFGen→相邻CMult; Projection partial sum→累加），通过unidirectional right→left连接传输。
    (c) **DMU (Data Manipulation Unit)**：Upper/Lower DMU用预定义control network+shifter执行bit-reversal permutation/padding/broadcast，消除I/O-heavy data formatting操作（GPU kernel中需显式warp shuffle+shared memory transpose）。
    (d) **RTL synthesis**：Chisel→TSMC 40nm, 1GHz。Per PE: 5.54mm²/1.72W。128 PE scaled to 7nm: 52.82mm²/41.1W = A100的6.4% area/10.28% power。32 PE scaled to 16nm: 42.35mm²/18.92W = TPUv3 core的13.1% area/8.4% power。SRAM+Core占>85% area/power，DMA/DMU/control开销极小。
    (e) **Area/Power效率**：76× area efficiency / 48× power efficiency vs A100 GPU（H3 128K sequence）。vs AMD AI Engine iso-area: 9.45× higher throughput（因AIE有inter-tile连接开销+非FP32单元，而VGA专用化避免这些开销）。
  - **VGA vs GPU关键差异总结**：GPU的ROI kernels全部memory-bound（FFT需频繁barrier+全局内存读写，Mux/Mxy完整矩阵过大无法存SRAM，pointwise操作浪费带宽）；VGA利用Vandermonde矩阵的on-the-fly生成消除SRAM容量瓶颈（5× reduction），使全融合执行成为可能，DRAM traffic减少9.7×，FLOPS utilization达78%（远高于GPU kernel）。GPU通用CUDA core无法利用矩阵的结构化可生成性；VGA通过7-mode reconfigurable CCU + DMU + 专用ISA，在极小的area/power budget下（6.4% area/10.28% power of A100）实现4.89× ROI speedup。GPU batch parallelism受限于SRAM容量和bandwidth；VGA沿hidden dimension并行化，PE间zero synchronization，workload perfectly partitionable。

## 46-Fast On-device LLM Inference with NPUs

- baseline方法是什么？
  - Baseline 是移动端现有多引擎方案：llama.cpp（ARM CPU/GPU INT8 推理）、TFLite（Google 移动推理引擎，CPU/GPU）、MNN（阿里移动推理引擎，CPU/GPU）、MLC-LLM（TVM-based GPU 编译器）、PowerInfer-V2（利用 NPU 加速 prefill 但未深度优化 NPU 亲和性）。
  - **Baseline 全栈执行例子**（以 llama.cpp-CPU 为例，Qwen1.5-1.8B，1024-token prompt）：
    - **模型/算法层**：使用 K-Quant（per-group W8A8 量化），将 activations 和 weights 按 group（如 group_size=128）切分成多个 group，每个 group 独立量化 scale。MatMul `X[s,:] @ W[:,h]` 被分解为 `num_groups` 个子 MatMul（如 group_size=128 → hidden_dim/128 个子 MatMul），每个子 MatMul 用整数 SIMD 计算后再浮点累加。
    - **系统/Serving 框架层**：llama.cpp 后端（C/C++）在 CPU 上顺序执行 Attention→FFN→LayerNorm 的 Transformer block。无 NPU 参与。prefill 阶段 batch_size=1，chunk 无拆分，整 prompt 一次处理。
    - **编译框架层**：无 NPU 图编译。llama.cpp 使用预编写的 NEON/ARM 汇编 kernel（如 ggml 矩阵乘），无运行时图优化。
    - **Kernel 调度层**：CPU 上逐 layer 顺序执行，INT8 MatMul 用 ARM NEON SIMD（128-bit），无跨处理器调度。Per-group MatMul 每个子 MatMul 独立调用 SIMD 指令，中间结果在 CPU 累加。
    - **硬件架构层**：ARM Cortex CPU core（Xiaomi 14 Snapdragon 8gen3），CPU 时钟 ~3.3GHz。无 NPU 使用。
    - **结果**：1024-token prefill 耗时 ~18s（Qwen1.5-1.8B，Redmi K70 Pro），prefill 占比 88–99% 总 latency，功耗高（CPU 全核运行）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法 llm.npu 通过三个层次的重构最大化 NPU 利用率：
    1. **Prompt level — Chunk-sharing graph**：将变长 prompt 切分为固定 size（chunk_length=256）的 chunks，每个 chunk 有预构建的 NPU 子图。通过识别 static ops（Linear/FFN/LayerNorm，仅依赖 chunk size）和 dynamic ops（Attention，依赖 chunk size + 序列位置），static 部分跨 chunk 共享（120/144 子图可共享），消除重复图构建/优化开销（从 11.5s 降至 256MB 内存 + 首次构建后复用）。
    2. **Tensor level — Shadow outlier execution**：将 baseline 的 per-group 量化改为 NPU 友好的 per-tensor INT8 MatMul，同时将 outlier channels（0.1–0.3%）提取出来在 CPU 上以 FP16 执行 shadow MatMul，利用加法分配律合并结果。Hot channel 缓存 + cold channel disk 加载 + 85% 层级剪枝消除内存和同步开销。
    3. **Block level — Out-of-order subgraph execution**：将 LLM 按 chunk+layer 维度分解为数十至上百个 subgraph，基于跨 chunk 依赖（Attention 等）和 intra-chunk 依赖（LayerNorm→Linear 等）构建依赖 DAG，通过贪心调度（选择对减少 NPU stall 贡献最大的 subgraph）将 bubble rate 从 37% 降至 0.7%。
  - **论文方法全栈执行例子**（llm.npu，Qwen1.5-1.8B，1024-token prompt，chunk_length=256，Redmi K70 Pro）：
    - **模型/算法层**：离线使用 wikitext profiling → 剪枝 85% 不重要 outlier layers → 仅保留 ~5–15 个 hot channels 的 CPU 权重缓存。运行时对每个 Linear layer：`y = INT8_MatMul_NPU(clip(x/s, -127, 128), W_int8) + FP16_MatMul_CPU(outlier_channels, W_hot_cpu)`。NPU 执行 per-tensor W8A8 MatMul，CPU 同步执行稀疏 outlier MatMul。
    - **系统/Serving 框架层**：MLLM + QNN 框架。1024-token prompt 被切为 4 个 chunk（每个 256 tokens），每个 chunk 内的 static subgraphs（Shared QKV Linear + Shared O Linear + Shared FFN）复用同一份预编译 NPU graph；dynamic subgraphs（Attention，不同 chunk 有不同 KV Cache 维度）独立。Chunk 间通过 KV Cache 传递 causal dependency。Decode 阶段仍用 MLLM CPU 后端。
    - **编译框架层**：QNN 框架预编译 chunk-sharing graphs。对 Linear 层进行 tensor shape 优化（如 2048×2048 权重转 32×32×2048 → NPU 1.62× 执行加速）。实现 QNN 原生不支持的 KVCache、SiLU、RMSNorm、ROPE 算子。Shared buffers 用于 CPU/NPU 中间结果同步，避免 context switch 开销。
    - **Kernel 调度层**：4 chunks × 6 subgraphs/chunk = 24 subgraphs。离线 profiling 各 subgraph 的 CPU/NPU 执行时间。运行时在线贪心调度器：当 NPU 完成一个 INT8 Linear → CPU 就绪的 LayerNorm + Attention 子图产生新可调度 NPU 子图 → 选 C 值最大的 pending subgraph 执行。结果：NPU bubble rate 0.7%（vs naive 37%），NPU 接近满利用率。
    - **硬件架构层**：Qualcomm Hexagon NPU（1024-bit INT8 SIMD，500–750MHz），与 CPU 共享物理内存。NPU 执行 INT8 MatMul（4.5–5.8× 快于 CPU INT8），CPU 处理 LayerNorm/Attention/shadow outlier（低计算量），两者通过 shared buffer 在统一内存上同步。
    - **结果**：1024-token prefill 0.5s（vs baseline CPU 18s，36× 加速），>1000 tokens/sec，能耗降低 35–59×。

## 47-SpecEE- Accelerating Large Language Model Inference with Speculative Early Exiting

- baseline方法是什么？
  - Baseline 是现有的 LLM early exiting 方法（以 AdaInfer 为代表）。Baseline 在每层 decoder layer 后部署 predictor（如 SVM），在预测时需要遍历完整词汇表（~3×10⁴ tokens in Llama2-7B）作为搜索空间：对每层的 hidden states（1×4096）与完整 lm_head（4096×32000）做 MatMul，获取所有 token 的 logits 作为 input features，再送入 SVM predictor 判断是否 exit。这导致 predictor 本身的计算开销与词汇表大小成正比，predictor 占 ~30% 总计算量和 ~20% 端到端推理延迟。
  - Baseline 全栈执行例子（AdaInfer，Llama2-7B on A100）：
    - **算法层**：每层执行 lm_head MatMul（hidden_states[1×4096] × lm_head[4096×32000] → 1×32000 logits），提取 3 个特征（top token prob, entropy, confidence），输入 SVM 分类器判断 exit
    - **系统/Serving 层**：HuggingFace Transformers inference。32 层的每一层后均部署 predictor（total 31 predictors），每层均需执行 lm_head traversal + SVM predict
    - **编译框架层**：论文未明确说明（使用标准 Pytorch + CUDA 后端）
    - **Kernel 调度层**：lm_head MatMul（4096×32000 GEMM）是对 GPU memory bandwidth 和 compute 的高开销操作，每层 executor 与 SVM predictor 串行，无 GPU 并行化考虑
    - **硬件架构层**：NVIDIA A100-80GB GPU。lm_head MatMul 是 memory-bound 操作（权重 4096×32000×2B ≈ 250MB），每层重复执行导致高内存带宽压力
  - Baseline 的核心缺陷：(1) predictor 在线搜索空间 = 完整词汇表，开销与 vocab_size 成正比；(2) 每层部署 predictor，但统计表明 ~50% 层 exit 概率极低（<3.2%），导致大量无效计算（~20% overhead）；(3) 在 speculative decoding 场景下，token tree 中每 token 被独立映射（exponential mapping complexity），无法利用 GPU 高吞吐计算。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法 SpecEE 通过三个层次的优化解决 baseline 的缺陷：
    1. **算法层（Speculation-based lightweight predictor）**：利用 speculative model（EAGLE DLM）将 predictor 搜索空间从完整词汇表（32000 tokens）缩减到 speculative tokens（4 tokens），实现 ~100× 缩减。特征从高维 raw data（~4096 dim）变为低维 engineered 特征（12 dim：4 spec_logits + 4 local_probs + 4 prob_variations）。Predictor 从 SVM 升级为 2 层 MLP（12×512+512×1），利用 GPU tensor core 并行加速。Verification algorithm 确保 global correctness。
    2. **系统层（Two-level heuristic predictor scheduling）**：通过离线频率统计 + 在线 contextual similarity（最近 5 tokens exit layer ±2 邻域 hit rate >70%）动态确定需要激活 predictor 的 layer，将 predictor 数量从 31 降至 ~10.2（~68% reduction），~1.21× 加速。
    3. **映射层（Context-aware merged mapping for speculative decoding）**：将 token tree 每条 path 合并为 hyper-token，通过 custom GPU GEMM kernel（基于 cutlass group GEMM + MegaBlocks 设计）批量计算 draft token logits 和 predictor features，将 exponential mapping complexity 降为 linear，~1.66× 加速。
  - 论文方法全栈执行例子（SpecEE on Llama2-7B, A100-80GB, speculative decoding with EAGLE）：
    - **算法层**：DLM 生成 4 speculative tokens → speculative_lm_head[4096×4]。每层 hidden states[1×4096] × speculative_lm_head → 4 spec_logits → softmax → 4 local_probs → diff with prev → 4 prob_variations → concat 12-dim feat → MLP(512 hidden, Sigmoid) → 若 pred>0.5 则 verification（hidden_states × full lm_head → check global top-1 in spec set），是则 early exit。MLP 仅 ~6.7K params，predictor overhead ~0.0009s/token（5.6% 总延迟）。
    - **系统/Serving 层**：HuggingFace + vllm PagedAttention。Heuristic scheduling：offline 统计 exit frequency → 激活高频 predictor 子集；online 维护环形队列（5 tokens）和 layer hit 数组 → 动态激活邻域 predictor。Union 后平均 ~10.2 个 predictor。EAGLE DLM 生成 token tree，merged mapping 将每条 path 视为 hyper-token 合并处理。
    - **编译框架层**：论文未明确说明（使用 Pytorch front-end + C++/CUDA backend）
    - **Kernel 调度层**：Custom CUDA kernel 基于 cutlass group GEMM 实现 merged hyper-token batch feature computation。输入 token tree hidden states batch [num_paths × 4096] → group GEMM with speculative_lm_head[4096×4] → 批量计算 draft token logits → softmax + diff 计算 features → MLP matmul on tensor cores → Sigmoid → verification。利用 GPU batch 并行，predictor 执行时间被压缩至 ~0.0009s/token。
    - **硬件架构层**：NVIDIA A100-80GB Tensor Core GPU。Predictor MLP 为 memory-bound（小 matmul），GPU 功耗从 Dense 推理的 201W 降至 182W（~10% 功耗降低，1.57× energy efficiency），因 predictor 计算量远小于 decoder layer 其他模块导致 GPU compute resource 未充分利用。论文建议未来 integrated training-inference GPU 采用 big-little core 设计以优化 power efficiency。
    - **结果**：Cloud 场景 total speedup 2.25×（vs Hugging Face on A100），PC 场景 2.43×（vs llama.cpp on RTX 4060 Laptop）。Accuracy loss <1% on MMLU/CommonsenseQA/SST2/GSM8K。可与 AWQ 量化正交集成，进一步 push Pareto frontier。

## 49-HeterRAG- Heterogeneous Processing-in-Memory Acceleration for Retrieval-augmented Generation.pdf

- baseline方法是什么？
  Baseline是现有的CPU-GPU RAG系统（Intel Xeon Gold 5117 CPU + NVIDIA Tesla V100 GPU），检索阶段在CPU侧使用HNSW图算法，生成阶段在GPU侧使用GPT-2/LLaMA2模型。全栈执行例子：
  - **算法层**：RAG流程——用户query→embedding模型转向量→CPU侧HNSW图搜索ANNS（best-first search: neighbor fetching→distance computation→queue updating迭代）→检索top-k文档→拼接query+文档输入LLM→GPU侧Transformer decoder autoregressive生成（prefilling GEMM→decoding GEMV迭代）。
  - **系统框架层**：TensorFlow/PyTorch等ML框架构建RAG pipeline。检索数据和模型权重在host DDR4和GPU HBM2之间通过PCIe传输（约16GB/s PCIe 3.0）。CPU负责检索，GPU负责生成，两者顺序执行。
  - **编译框架层**：GPU上通过CUDA runtime执行LLM推理——cuBLAS/cuDNN库处理GEMM/GEMV操作，无专门的RAG-targeted编译优化。
  - **kernel调度层**：CPU侧HNSW distance computation——随机访问稀疏图结构（scattered vector data in DDR4），内存访问pattern不规则，DDR4 bandwidth利用率低。GPU侧decode GEMV——单token输入下weight数据频繁从HBM加载，batch size小（RAG场景隐私敏感、用户数少）导致GEMV memory-bound。KV cache per request独立，on-chip cache无法容纳。
  - **硬件架构层**：Intel Xeon Gold 5117 (14nm, 2.0GHz, 256GB DDR4-2400, memory BW ~19.2GB/s per channel) + NVIDIA V100 (12nm, 32GB HBM2, 900GB/s BW, PCIe 3.0 x16 interconnect)。CPU-GPU间数据传输受限于PCIe带宽，远低于内部memory bandwidth。DDR4容量大但带宽低，HBM带宽高但容量小且成本高（>5× DDR per GB）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  HeterRAG提出异构PIM系统（AccelDIMM + AccelHBM），针对RAG两个阶段分别使用最优PIM技术解决baseline的memory bandwidth和capacity双重瓶颈。对比baseline的全栈执行例子：
  - **算法层**：RAG流程不变（HNSW检索→LLM生成），但通过三种软硬件协同优化消除算法层面的冗余：(a) locality-aware retrieval——利用文档访问power-law分布（60%请求集中在top 3%文档），在RPM内缓存频繁访问的vertex vectors（128KB LRU cache），迭代RAG时复用上一轮结果作为搜索起点，减少DRAM访问。(b) locality-aware generation——利用prefix tree组织文档KV cache，dense KV（文档自身相关）与sparse KV（序列中重要token，保留10-20%不影响质量）分离存储，新序列匹配prefix tree时仅selective compute非缓存部分。(c) fine-grained parallel pipeline——不等所有检索完成即启动生成，部分/高置信结果提前发送。
  - **系统框架层**：HeterRAG Software Stack基于现有ML框架（TensorFlow/PyTorch）扩展两个API：ANNS-ACC（ANNS加速）和LLM-ACC（LLM推理）。HeterRAG Compiler将API调用编译为host executable和device-specific binaries，通过driver下发到AccelDIMM/AccelHBM。对比baseline：CPU负责检索→GPU负责生成（PCIe传输），HeterRAG由AccelDIMM直接在DIMM内完成检索（无CPU-DIMM数据传输）、AccelHBM在HBM内完成GEMV（无HBM-GPU传输），仅top-k结果（少量KB）和tensor通过CXL（PCIe 5.0 ×8, 32GB/s）传输。
  - **编译框架层**：论文未明确说明具体的编译pass细节。HeterRAG Compiler将ANNS-ACC/LLM-ACC API调用解析为IR→编译为host可执行文件+device二进制→driver下发。
  - **kernel调度层**（核心差异化）：(a) **检索阶段**：baseline CPU距离计算受DDR4 bandwidth限制，利用率为roofline模型中的低算术强度区域（远低于bandwidth上限）。HeterRAG将distance computation offload到DIMM内的RPM——每个rank配备专用Distance Computation Unit (32 FP32 Mult+Adder)，DDR command在DIMM内部生成和执行（DDR-C/A Generator），无需跨越DIMM interface读取原始数据。通过rank-level并行（2 ranks per DIMM）和vertex vector缓存（LRU cache hit时直接计算），显著提升bandwidth利用率。(b) **生成阶段**：baseline GPU GEMV受HBM bandwidth限制（低算术强度导致计算单元等待数据）。HeterRAG将GEMV offload到HBM内的BPM——每两个bank共享一个BPM，bank-level并行（8 channels × multiple banks），充分利用HBM内部带宽。GEMM和其他操作留在TPM的systolic array和VLIW processor执行。(c) **Pipeline调度**：baseline检索和生成严格顺序执行（检索完成→生成开始），两种设备（CPU/GPU或DIMM/HBM）有一半时间idle。HeterRAG fine-grained parallel pipeline使retrieval和generation overlap，资源利用率从约20%提升到44.5%/38.6%。
  - **硬件架构层**（核心差异化）：(a) **异构PIM硬件**：AccelDIMM (DDR4-based, 大容量低成本, rank-level PIM) + AccelHBM (HBM2-based, 高带宽低功耗, bank-level PIM)，而非baseline的通用CPU+GPU。设计依据——检索需要大容量（知识库TB级），DIMM 64GB/module远大于HBM 16-24GB/stack；生成需要高带宽（GEMV memory-bound），HBM 256GB/s ≈ 13× DDR4 19.2GB/s。(b) **独立扩展**：AccelDIMM和AccelHBM数量可独立scale，据workload需求灵活配置。检索scale近乎超线性（无inter-device通信），生成scale接近线性（受device间通信限制）。(c) **芯片面积开销**：HBM BPM 6.016mm² (11.31% of die)，DIMM RPM+DPM 7.817mm² (<<10% of buffer chip)，在可接受范围内实现显著的性能提升。

## 4-BBS: Bi-Directional Bit-Level Sparsity for Deep Learning Acceleration

- baseline方法是什么？
  - Baseline是现有的bit-serial DNN加速器（Pragmatic [MICRO'17]、Bitlet [MICRO'21]、BitWave [HPCA'24]）和value-based加速器（SparTen [MICRO'19]、ANT [MICRO'22]）。这些方法的核心局限：(1) bit sparsity仅局限在zero bits——Pragmatic和Bitlet skip zero-bit operations，但zero-bit分布随机导致严重load imbalance；(2) 加载不均衡导致需要复杂的同步硬件——Pragmatic variable shifter、Bitlet 64:1 mux (占PE面积35.9%)；(3) BitWave利用sign-magnitude格式的zero bit column sparsity但需要2's complementer（1.32× Stripes面积），且仅能skip全零bit column（coarse-grained），对量化为8-bit的DNN效果有限；(4) value-based加速器在8-bit量化DNN上缺乏value sparsity（<5%，因为PTQ充分利用所有量化级别减少误差）；(5) 缺乏post-training模型压缩能力——现有方法要么需要retraining（value sparsity），要么仅做on-chip zero-bit skipping不改模型尺寸。
  - Baseline全栈执行例子（以BitWave在ResNet-50上的bit-serial推理为例）：
    - **算法层**：8-bit per-channel quantized ResNet-50 weight, sign-magnitude format。BitWave bit-flip算法将lower significant bit columns翻转为全零来生成更多zero sparse columns（但会丢失量化级别，大幅增加KL divergence vs original weight distribution）
    - **系统框架层**：无特别Serving框架修改。模型存储为sign-magnitude weight + bit-column sparsity metadata。推理时从DRAM加载weight bit-columns和activation到on-chip buffer
    - **编译框架层**：论文未明确说明。使用自定义bit-serial指令控制PE array
    - **kernel调度层**：BitWave bit-column-serial approach——PE处理同一bit significance下多个weight组成的bit column。若bit column全零则整个skip（coarse-grained sparsity）。sign-magnitude格式下较多zero bit columns（因DNN weights通常值小），但coarse-grained sparsity = skip many ineffectual bits → 每cycle只skip一个bit column vs BBS可skip per-PE内的individual bits → PE利用率低
    - **硬件架构层**：BitWave PE含2's complementer（支持sign-magnitude算术，增加1.32× area vs Stripes dense PE），variable shifter处理不同bit significance。PE array类似output-stationary dataflow处理weight channels。On-chip buffer: 256KB activation + 256KB weight。DRAM: DDR3。BitWave PE功耗0.49mW (TSMC 28nm, 800MHz)，但performance受限于仅在bit-column全零时skip

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文提出BBS (Bi-directional Bit-level Sparsity) + BitVert加速器，通过以下关键设计解决baseline缺陷：
    1. **算法层（BBS Binary Pruning）**：通过Eq. 2-3的数学等价性，在任意bit-vector中保证≥50% BBS（zero-bits > 50%→skip zeros；zero-bits ≤ 50%→invert→skip ones）。两种binary pruning策略（rounded averaging + zero-point shifting）无需retraining或calibration dataset即可将8-bit weight压缩为6-bit/4-bit。Global binary pruning基于per-channel scaling factor识别sensitive channels（保持8-bit），在其余channel上aggressive压缩（最差2个/4个bit columns被剪枝），利用hardware-aligned CH参数保证full PE utilization。
    2. **硬件层（BitVert PE + Scheduler）**：BBS保证≥50% sparsity → sub-group size=8中最多4个effectual bits → 仅需4个5:1 mux（vs Bitlet 64:1 mux）——大幅降低多路选择器面积。Priority encoder-based scheduler低开销控制skipping（仅4个5-bit priority encoders vs Bitlet/baseline复杂的crossbar）。BBS Multiplier复用（3 bits/cycle time-multiplex）+ single shifter替代variable shifter。Channel reordering + output unshuffle解决per-channel mixed precision下的memory alignment问题。
    3. **全栈对比**：
  - 论文方法全栈执行例子（BitVert, moderate pruning, ResNet-50 on cycle-accurate simulator, TSMC 28nm 800MHz）：
    - **算法层**：Global binary pruning——识别20% sensitive channels（8-bit，CH=32对齐）→其余80% channels用zero-point shifting prune 4 bit columns→Algo. 1搜索optimal zero-point（遍历[-32,31] minimize MSE）→输出4-bit effective weight + 8-bit metadata/group→total 1.66× memory compression，准确率损失<0.5%
    - **系统框架层**：无Serving框架修改。Compressed weight + metadata存储于DDR3 DRAM，activation加载到on-chip buffer。Channel reordering：sensitive channels (8-bit)分组存储→normal channels (4-bit)分组存储→memory access对齐→channel index buffer存储原始channel顺序→output unshuffle恢复正确顺序
    - **编译框架层**：论文未明确说明编译框架修改
    - **kernel调度层**（核心创新）：16×32 PE array (output-stationary)→每个PE处理16-weight group，scheduler分析每bit column：zero-bits > 50%→skip zero-bits (Eq.2: sum A where W_ib=1)；zero-bits ≤ 50%→invert bit-vector→skip one-bits (Eq.3: ΣA - sum A where W_ib=0)→4个5:1 mux per sub-group定位最多4个effectual activations→bit-serial multiplier→adder tree→subtractor→shift (per col_idx)→BBS multiplier (BBS constant × ΣA)→accumulate。关键：BBS保证每cycle最多4 effective bits/8-weight sub-group→PE内部balanced workload→消除Pragmatic/Bitlet的单bit-vector瓶颈→inter-PE stall仅0.7%（BitWave 3.1%, Bitlet 8.4%）
    - **硬件架构层**（核心创新）：BitVert PE面积739.6μm² (1.39× Stripes dense PE，但远小于Bitlet 1665.6μm² 3.13×和Pragmatic 923.1μm² 1.73×)。功耗0.45mW (1.22× Stripes)。BBS constant在2 cycle内通过time-multiplex BBS multiplier（3 bits/cycle）处理compressed bit columns (compressed weight最多含6 non-sparse columns + 2-cycle min)→无需大位宽乘法器
    - **结果**：3.03× speedup (vs Stripes), 2.44× energy saving (vs SparTen)，准确率损失<0.5%。BitVert在accuracy-EDP Pareto前沿上优于所有对比加速器

## 50-S-DMA- Sparse Diffusion Models Acceleration via Spatiality-Aware Prediction and Dimension-Adaptive Dataflow..pdf

- baseline方法是什么？
  Baseline是在NVIDIA GPU上运行dense扩散模型推理，所有激活值按dense计算（不做sparsity skipping），PE阵列数据流固定（如output-stationary的单一遍历顺序）。全栈执行例子：
  - **算法层**：DDPM扩散模型，U-Net架构denoising网络。推理过程：从随机噪声x_T开始，逐timestep (t=T→1) 调用U-Net预测噪声ε_θ(x_t, t)，然后通过DDPM采样公式更新x_{t-1} = (1/√α_t) * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ)。每个timestep内U-Net执行Conv→SiLU→GroupNorm→Attention→Upsample/Downsample→Skip Connection等操作。所有操作使用FP16或FP32 dense计算。
  - **系统框架层**：PyTorch调用CUDA runtime在GPU上执行。模型权重存储在GPU HBM中，每timestep的输入x_t从host传输到GPU或直接在GPU上生成。推理以single batch逐timestep串行执行。
  - **编译框架层**：PyTorch→CUDA kernel（cuDNN/cuBLAS）。Conv kernel使用implicit GEMM (Img2Col→cuBLAS GEMM)，Attention使用cuDNN Multi-Head Attention或手工CUDA kernel。无针对扩散模型稀疏性的编译优化。
  - **kernel调度层**：GPU上dense Conv kernel按固定数据流执行——output-stationary方式，每个thread block计算一个output tile，所有activation×weight的MAC全部执行，无论激活值是否为零。零激活值（ReLU/SiLU后的输出）同样参与乘加和访存，浪费计算和带宽。Attention kernel中SoftMax后的稀疏attention scores同样dense处理。
  - **硬件架构层**：NVIDIA GPU (如V100/RTX 3090/A100)，通用SIMT架构，Tensor Core做GEMM，CUDA Core做element-wise。HBM提供高带宽但功耗高。GPU的通用性使其不做activation sparsity的细粒度跳过，零值激活仍消耗计算和访存资源。扩散模型1000-step的串行去噪导致端到端延迟极高。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  S-DMA通过Spatiality-Aware Prediction (SAP) + Dimension-Adaptive Dataflow (DAD) 两个协同机制，在专用加速器上实现扩散模型的稀疏感知推理。对比baseline的全栈执行例子：
  - **算法层**：扩散模型算法本身不变（DDPM采样公式和U-Net架构不变），但在推理中引入sparsity-aware execution——SAP预测每个timestep/layer中哪些激活位置将为零→这些位置的计算被跳过但不改变最终生成图像的分布。论文通过实验验证FID无显著退化，说明sparsity skipping在扩散模型中的可行性。
  - **系统框架层**：论文未明确说明系统框架修改。推理流程为：初始噪声x_T加载→每timestep：SAP预测→DAD调度→Sparse PE Array执行→输出x_{t-1}→下一timestep。模型权重预加载到片外DRAM，每层前从DRAM加载weight tile到片上weight buffer。
  - **编译框架层**：论文未明确说明编译框架修改。UNet各层（Conv/Attention/Upsample/Downsample）的稀疏计算模式由DAD controller在runtime动态选择，不需要offline compilation。
  - **kernel调度层**（核心创新——DAD）：对比baseline GPU固定数据流（所有MAC dense执行），DAD的关键改进：(a) **维度自适应遍历**——不同于GPU固定的output-stationary遍历，DAD根据每层的spatial/channel稀疏率动态选择C-first或S-first数据流。C-first在外层跳过整channel→粗粒度zero-skip；S-first在外层跳过整spatial位置→也是粗粒度skip。这比GPU细粒度per-element checking更高效——GPU无法跳过整行/整channel因为warp divergence问题（同一warp内不同threads遇到不同zero pattern→部分thread idle但无法整体跳过）。(b) **SAP驱动调度**——SAP在计算前预测sparsity mask，DAD基于预测做调度决策，而非依赖runtime detection（后者需先读activation再决策→引入延迟）。SAP利用扩散模型空间相关性：相邻像素通常有相似稀疏性→轻量CNN预测准确率高。(c) **Zero-skip粒度可配置**——DAD配置router实现channel-granularity或spatial-granularity的skip，将GPU无法实现的粗粒度skip在专用硬件上高效执行。
  - **硬件架构层**（核心创新——SAP+DAD硬件）：(a) **SAP预测器**——轻量CNN/MLP硬件单元，与主PE array并行执行，在计算当前层时预测下一层的sparsity mask→无额外延迟。(b) **DAD Controller + Configurable Router**——硬件状态机统计sparsity statistics→比较→选择模式→reconfigure PE间互联路径。Router可动态切换activation broadcast方向和partial sum reduction路径以支持两种遍历。(c) **Sparse PE**——每个PE含zero-skip gating logic，当sparsity mask指示当前位置为零时，gating关闭该PE的MAC单元时钟/电源→节省动态功耗，同时该cycle用于处理下一个非零位置→提升吞吐。(d) **Memory hierarchy**——sparsity mask buffer存储SAP输出的mask，供PE array和DAD controller读取。相比GPU HBM，S-DMA的片上SRAM buffer + 粗粒度skip减少了对片外DRAM的带宽需求。
  - **全栈对比总结**：GPU baseline将扩散模型视为dense computation——所有1000 timestep中所有U-Net层所有位置的激活×权重全部计算→大量零激活值做无用功。S-DMA通过SAP提前预测哪些位置为零→DAD选择最优skip粒度（spatial/channel）→Sparse PE执行时跳过零值→节省计算和访存→提升吞吐和能效，同时保证生成质量。

## 53-Accelerating Retrieval Augmented Language Model via PIM and PNM Integration

- baseline方法是什么？
  **Baseline: GPU-based batched RALM (H100 NVL + FAISS-GPU + RETRO)**
  - **算法层**：RETRO模型（12 decoder blocks × SA/CCA/FFN）每retrieval_interval个token调用FAISS IVF-PQ检索外部database（nlist clusters, nprobe probe, top-k retrieved chunks），检索到的chunks经Encoder编码后馈入CCA层辅助token生成。Batch内32个requests同步：全部requests生成retrieval_interval个token→全部requests同步执行retrieval→retrieval完成后全部requests继续生成→循环。
  - **系统框架层**：RETRO inference runtime：Python/TensorFlow script→batch requests→for each decoding step: if step%retrieval_interval==0→FAISS-GPU search: 查询vector→cluster selection→PQ code scan→top-k selection→encode retrieved data→feed CCA→next token generation。Batch synchronization point at each retrieval→所有requests等待最慢的retrieval完成→大量GPU SM idle cycles。
  - **编译框架层**：论文未明确说明。RETRO模型通过标准ML framework（TensorFlow/PyTorch）编译到CUDA kernel。MHA使用标准attention kernel（可能为FlashAttention），FFN使用cuBLAS GEMM。IVF-PQ检索通过FAISS-GPU CUDA kernel执行。
  - **kernel调度层**（baseline bottleneck）：(a) **MHA kernel**：SA和CCA的Score操作（Q·K^T）和Context操作（softmax_out·V）本质上是GEMV（matrix-vector），每request每token的K/V矩阵唯一→batch parallelism受限→GPU SM上表现为memory-bound GEMV，peak H100 GPU memory bandwidth 3.35TB/s远不敷需求；(b) **PQ code scan kernel**：FAISS-GPU IVF-PQ的LUT lookup+dot-product，每个codeword需访问precomputed distance table→GPU shared memory容量不足以容纳完整LUT (ksub×M=256×64=16K entries per cluster)→频繁global memory access→低计算利用率；(c) **Top-k selection kernel**：每cluster需从所有probed data中选最小k个distances→GPU上warp-level sort或atomic CAS→低并行度→memory-bound。Roofline analysis：MHA ~1.0 Op/B, PQ scan ~0.1 Op/B, Top-k ~0.01 Op/B → all far below H100 990 TFLOPS peak in memory-bound region。
  - **硬件架构层**：NVIDIA H100 NVL GPU (132 SMs, 94GB HBM3 @ 3.35TB/s, L2 60MB)。HBM3通过TSV和硅中介层连接到GPU die。所有LLM计算和检索计算均在GPU SMs上执行→memory-bound kernel受限于GPU到HBM的off-chip bandwidth。Batch内sequential retrieval进一步恶化——GPU hardware parallelism被软件同步闲置。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **MNM = PIM(PIM on HBM core die) + PNM(PNM on HBM logic die) + Selective Batching + Early Generation scheduling**
  - **算法层**：RETRO模型算法不变。Early Generation调度在retrieval pending期间提前生成额外token→实际的retrieval interval可能延长→论文通过perplexity评估验证：MNM因PNM加速retrieval→early tokens数少→perplexity degradation可控（batch 32/nprobe 256下MNM perplexity远低于GPU+EarlyGen）。
  - **系统框架层**（Selective Batching + Early Generation + Request Manager）：对比baseline的同步batch-retrieval→generation流水线，MNM将retrieval和generation解耦为事件驱动：(a) 每个request的retrieval completion作为异步trigger→立即加入generation batch；(b) Early Generation: retrieval pending期间其他requests可提前生成tokens→利用MNM dual row buffer实现PIM generation和PNM retrieval的物理并发；(c) Request Manager: Runtime测量T_gen和T_ret,max→N_ret=⌊T_gen/T_ret,max⌋限制并发retrieval数→形成Retrieval Group→Retrieval Group Queue→updateQueue()按early token count重排。对比baseline：baseline的batch在每次retrieval后全局同步→所有requests idle等待最慢retrieval→MNM通过异步+early generation将大部分retrieval延迟隐藏在generation中→GPU utilization提升。
  - **编译框架层**：论文未明确说明。MNM通过自定义command set (Table 2)暴露PIM/PNM能力给GPU→GPU code中调用MNM指令（PIM_MAC_AB/PNM_RET_INIT等）替代原CUDA MHA kernel和FAISS search kernel。无编译器自动化——手工替换memory-bound kernel的dispatch target。
  - **kernel调度层**（核心创新——PIM/PNM kernel offloading）：对比baseline所有memory-bound kernel在GPU SMs上执行受限于GPU-HBM off-chip bandwidth：(a) **MHA Score/Context → PIM on core die**：K^T/V矩阵存HBM core die bank内，PIM每bank的FP16 MAC unit直接从PIM row buffer (16 FP16 elements)和global vector buffer (Q segment)读取→bank-parallel dot-product across all banks per channel→结果经global bus送logic die Softmax→返回global vector buffer→再次PIM_MAC_AB执行Context。有效bandwidth = HBM internal bank bandwidth (~4TB/s per stack) >> GPU off-chip 3.35TB/s——GEMV不再受限于off-chip带宽；(b) **PQ code scan + Top-k → PNM on logic die**：Codewords通过TSV从core die送logic die PQ code scanner→16并行scanner每nCCDL=4 cycles处理1024B codewords→MAC module做x·y_R dot-product→LUT查表+Query Register值求和→距离d送Top-k selector (Full sorter→Partial sorter→Top-k register, Odd-Even Merge Sort)。对比baseline GPU上的FAISS IVF-PQ：GPU shared memory LUT因容量不够（16K entries per cluster > 132KB shared memory）需反复访问global memory→PNM SRAM-based LUT一次加载即可服务整个cluster→bandwidth利用高效。Top-k sort在GPU上warp-divergent→PNM专用sorter pipeline每cycle处理16个距离的merge。
  - **硬件架构层**（核心创新——HBM-integrated PIM+PNM）：对比baseline H100 NVL的standard HBM3：(a) **PIM on core die**：每bank含FP16 MAC unit (650MHz, 0.11mm²) + dual row buffer (1用于PIM MAC, 1用于PNM read→并发)。Area overhead 15.0% per core die, power 134.5mW MAC + 98.2mW dual row buffer；(b) **PNM on logic die**：16× PQ code scanner (含PQ codebook SRAM 82.0% + precomputed LUT 13.7% + MAC 4.3% area, 650MHz, 1.00mm² total), Top-k sorter (0.003mm²), Softmax calculator (0.154W)。Area overhead 2.0% of HBM3 logic die。Total MNM power 5.35W = 4.6% of 116W max HBM power；(c) **MNM Controller**：GPU→MNM via high-BW interconnect (PCIe/CXL)，command decoder+MMIO register映射（HBM reserved address space），channel-level命令调度。对比baseline：standard HBM无计算能力→所有数据须经off-chip interconnect传输到GPU SMs→MNM将计算推到数据所在位置→data movement从off-chip缩小到on-die/in-die传输；(d) **Dual row buffer**是关键concurrency enabler——PIM MAC操作占用row buffer A→PNM可同时通过row buffer B读取PQ codewords→实现真正的retrieval-generation overlap（而非GPU上分时复用SM资源）。
  - **全栈对比总结**：GPU baseline将RALM的memory-bound kernel（GEMV attention + IVF-PQ search/sort）全部在GPU SMs上执行→受限于GPU-to-HBM off-chip bandwidth+sequential retrieval synchronization→大量SM idle cycles和off-chip data movement能耗。MNM将GEMV attention kernel推到HBM core die的PIM MAC bank内部（利用~4TB/s bank bandwidth）→将PQ code scan/top-k推到HBM logic die的PNM专用硬件（替代GPU上低效LUT/sort）→同时通过Selective Batching+Early Generation调度将retrieval延迟与generation重叠→最高29.2× speedup和71.5% energy savings。设计核心洞察：RALM的两大组件虽然都是memory-bound，但瓶颈特征不同——LM attention是带宽不足（needs more bandwidth→PIM），Retriever search/sort是计算利用低（needs better logic→PNM）→单一加速方案（PIM-only或PNM-only）无法同时最优加速两者→MNM通过异构集成同时解决两个瓶颈，并在调度层利用硬件并发性（dual row buffer）实现generation-retrieval overlap。

## 54-HLX- A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models..pdf

- baseline方法是什么？
  **Baseline: GPU-based Hybrid Transformer-Mamba inference (A100/H100 + FA-2/FA-3 + SSD)**
  - **算法层**：Hybrid-2.7B模型，64层（6 attention + 58 Mamba-2 layers），attention层用FlashAttention-2 (FA-2)或FlashAttention-3 (FA-3) kernel，Mamba-2层用SSD (State-Space Duality) 5-kernel分解（chunk cumsum→chunk state→state passing→BMM chunk→chunk scan）。FA-2: block-level融合（QKT→local softmax→PV→update O），沿sequence length维度并行处理Q blocks，按block-level同步执行——step 7(QKT)→8(m/l update)→9(P计算)→10(O update) 顺序依赖无法重叠，非MatMul与MatMul串行。FA-3: 在H100上引入warp-specialized 2-stage异步pipeline，但register pressure（2-stage pipeline将FA-2中间数据翻倍）和SIMT限制使非MatMul未能完全隐藏。
  - **系统框架层**：PyTorch-based Hybrid model inference runtime。论文未明确说明Serving框架。推理时按layer顺序执行：attention layer (FA-2/FA-3 kernel) → Mamba-2 layer (SSD kernels + RMSNorm + Conv1D + In/Out Proj + z-gating) → 循环。每层kernel launch有GPU kernel launch overhead（SSD尤其严重：5个独立kernel分别launch）。GPU上CPU-GPU communication和kernel launching overhead论文明确排除在latency测量外（fair comparison focused on pure computational performance）。
  - **编译框架层**：论文未明确说明。FA-2/FA-3通过hand-written CUDA kernel直接实现；SSD 5-kernel通过PyTorch + CUDA kernel实现。无自动编译器或code generation参与——全部手工优化kernel。
  - **kernel调度层**（baseline核心瓶颈——两种kernel的低compute utilization）：
    - **FA-2 kernel on A100**: Compute utilization饱和于~61% (128K seqlen)。瓶颈来自block-level同步执行——QKT和PV（MatMul）完成后才能做softmax和update O（非MatMul），非MatMul开销无法隐藏。Op/B随seqlen增长而增加但受KV cache增大影响在16K后开始下降。Roofline: compute-bound区域但利用率不高。
    - **FA-3 kernel on H100**: Compute utilization饱和于~61%。H100支持TMA异步数据搬运和warp specialization（producer/consumer warps），但：(a) 2-stage pipeline将intermediate data翻倍→register pressure（H100 256KB reg per SM）→register spilling→occupancy下降；(b) GPU SIMT model假设uniform warp execution→warp-specialized pipeline中heterogeneous warps资源需求不同→scheduling overhead和resource contention；(c) TMA仅优化coarse-grained tile transfer→fine-grained streaming/gather access仍有overhead。
    - **SSD kernel on A100/H100**: Compute utilization仅~27% (A100) / ~38% (H100)。瓶颈：(a) SSD本身有大量memory-intensive element-wise操作（exp/cumsum/softplus），Op/B极低→始终memory-bound (roofline memory-bound区域)；(b) 5个独立kernel间中间数据经DRAM传输→无数据复用——chunk cumsum输出sdt/dACS→写DRAM→chunk state读回；chunk state输出states→写DRAM→state passing读回；BMM chunk输出CBT→写DRAM→chunk scan读回；(c) 大量Einsum multi-dimensional tensor操作→GPU上非MatMul效率低。
  - **硬件架构层**：NVIDIA A100 GPU (7nm, 312 TFLOPS FP16, 1935 GB/s HBM2E, 84.3MB on-chip [reg+shared mem+L1 per SM + L2], 826mm², 300W) / H100 GPU (4nm, 756 TFLOPS FP16, 2000 GB/s HBM2E, 103.9MB on-chip, 814mm², 350W)。GPU SM体系结构：SIMT execution model→假设uniform warp execution→对heterogeneous pipeline stage支持差；TMA支持coarse-grained tile movement→fine-grained access overhead未解决；register file有限(256KB/SM)→fused kernel（如fused SSD需642KB/block）导致register spilling。TPUv3 half-chip (16nm, 61.5 TFLOPS, 450 GB/s HBM2, 16MB on-chip, 324mm², 225W)更糟：2×128×128 MXU专为dense MatMul设计→SSD的大量非MatMul操作在vector unit上执行效率极低→compute utilization仅~11%。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **HLX = PipeFlash (fine-grained pipelined FA-2) + PipeSSD (first fused+pipelined SSD) + Unified Reconfigurable Streamlined Core (URSC) hardware architecture**
  - **算法层**（PipeFlash & PipeSSD dataflows——对应解决baseline的compute utilization瓶颈）：
    - **PipeFlash vs FA-2**: FA-2 block-level同步→非MatMul无法隐藏→compute utilization ~61%。PipeFlash将粒度细化为2 rows of Q per block，形成4-stage pipeline（DPE#0:QKT 2 rows→RVPE:softmax 1 row→DPE#1:PV 2 rows→UpE:update O 1 row）→softmax和update O与MatMul完全重叠→compute utilization达97.5% (128K)。中间数据减少4.8×（score/prob 128KB→1KB）进一步降低on-chip memory压力。
    - **PipeSSD vs SSD**: SSD 5-kernel分离→DRAM中间数据大→memory-bound (compute util ~27%)。PipeSSD先做block-level fusion（合并6个operation为单一fused kernel→消除DRAM中间数据读写），再加入3-stage fine-grained pipeline：(1st) dA预处理→(2nd) YDiag→(3rd) YOff∥statesN→UpE(YFinal+update states)。通过row-level流水消除column-wise dependency (states(j-1)→states(j))和row-wise dependency (YDiag+YOff→YFinal)导致的串行瓶颈。中间数据减少11× (642KB→58.5KB)，DRAM流量减少6.8×，compute utilization达~78.4%。
  - **系统框架层**：HLX支持完整Hybrid-2.7B模型end-to-end推理。Top Controller管理computation mode切换（FA-2 mode⇌SSD mode），DMA负责DRAM↔GS数据搬运。Mode Controller根据当前layer类型配置URSC的pipeline模式。所有非FA-2/SSD操作（RMSNorm、Conv1D、In/Out Proj、SiLU、z-gating）在已有computation unit上执行（DPE demux支持conv1D、RVPE SFU支持SiLU）→无需额外硬件。对比baseline：GPU上每层kernel launch overhead（尤其是SSD 5-kernel）通过HLX的单次配置+连续pipeline执行消除。
  - **编译框架层**：论文未明确说明。HLX为hardware accelerator→无编译框架。PipeFlash和PipeSSD的dataflow mapping由硬件dataflow固定（DPE#0→RVPE→DPE#1→UpE的固定pipeline路径+Local NoC可重构数据forwarding），无软件调度层。
  - **kernel调度层**（URSC unified pipelined architecture——对应解决GPU SIMT限制和fused kernel register pressure）：
    - **对比FA-3 on H100**: FA-3因2-stage pipeline导致register翻倍→register spilling+occupancy下降→utilization仍饱和于~61%。HLX URSC通过heterogeneous pipeline stages（DPE/RVPE/UpE各有专用资源）→每个stage有独立的compute unit和local memory→无register pressure→pipeline stage数不受occupancy限制→可支持更深的4-stage (PipeFlash)和3-stage (PipeSSD) pipeline。H100 TMA仅支持coarse-grained tile→HLX的fine-grained row-level forwarding通过DPE→RVPE→DPE→UpE的direct wire connection实现→无memory access overhead。
    - **Pipeline stage balancing**（HLX核心创新）：以DPE MatMul为瓶颈，通过控制各行处理行数使所有stage cycle匹配。PipeFlash: QKT 2 rows = PV 2 rows, softmax/update O 各1 row（非MatMul比MatMul轻→被隐藏）。PipeSSD YDiag: CBT 2 rows→CBTLdt 1 row→CBTLdt×x 2 rows。PipeSSD YOff/statesN: dCOff 8 rows→YOff 8 rows, dBdtT 4 rows→statesN 4 rows→YFinal 8 rows, update states 4 rows。当dhead=dstate=blocksize时近100% utilization；维度不匹配时仍可通过调整行数保持高效。
    - **Fused SSD vs GPU fused SSD**: GPU上直接fuse SSD虽提升data reuse但中间数据(642KB)>per-SM memory (256KB reg + 164/224KB shared mem)→register spilling→性能反恶化1.74×。HLX通过PipeSSD的fine-grained pipelining将中间数据降至58.5KB→远低于HLX的on-chip容量→无spilling。
  - **硬件架构层**（URSC——对应解决GPU通用性过高、对Hybrid workload效率低的问题）：
    - **URSC结构**: 2 DPEs (MatMul专用) + RVPE (非MatMul: softmax/cumsum/exp/element-wise) + UpE (O/state更新: element-wise乘加+final O/YFinal计算)。Local NoC实现RVPU内部4种operation mode可重构连接。Global Scratchpad (GS) 经NoC连接URSC→中间数据在URSC内部forwarding无需回GS。对比GPU SM：通用ALU/SFU无专用pipeline path→异构operation间数据依赖需经shared memory中转→增加延迟和energy。
    - **面积功耗效率**: HLX60 (169mm²/201.8W @7nm) vs H100 (814mm²/350W @4nm)→89.8% area reduction, 63.8% power reduction。HLX30 (83.9mm²/108.47W) vs A100 (826mm²/300W)。关键：两个DPE占62.4% area和74.9% power→大部分硬件资源用于MatMul（与GPU SM类似），但通过专用pipeline path和RVPE/UpE消除非MatMul瓶颈→同面积下pipeline效率远高。
    - **统一架构overhead**：支持Transformer+Mamba-2双模型的HW overhead仅3.0% area + 2.9% power vs Transformer-only（增加conv1D/softplus/cumsum逻辑），4.4% area + 3.5% power vs Mamba-2-only（增加softmax/reciprocal/mux-demux逻辑）→证明URSC的reconfigurable设计实现高资源复用。
    - **DRAM流量对比**: PipeFlash较FA-2减少intermediate data 4.8×→虽然paper未直接报告DRAM traffic reduction，但因on-chip SRAM更小（HLX60 30.4MB vs H100 103.9MB→3.4×小），且pipeline中间数据在DPE/RVPE/UpE间直接forwarding→综合DRAM访问少于GPU baseline。PipeSSD DRAM traffic减少6.8× (explicit measurement)。
  - **全栈对比总结**：GPU baseline (A100/H100) 上Hybrid-2.7B模型推理：FA-2/FA-3的block-level同步执行→非MatMul延迟无法隐藏→compute utilization ~61%；SSD的5-kernel分离→memory-bound (~27%)，fuse后因register spilling反恶化。HLX通过三层面协同解决：(1) Algorithm: PipeFlash/PipeSSD fine-grained pipelined dataflow→打破op间dependency，隐藏非MatMul延迟，减少中间数据；(2) Kernel: URSC heterogeneous pipeline stages (DPE/RVPE/UpE) + pipeline balancing (row-level粒度控制)→各stage有独立计算和存储资源→无register pressure→支持更深pipeline；(3) Hardware: 专用加速器替代通用GPU→DPEs/RVPE/UpE通过direct wire forwarding而非shared memory→minimal data movement→compute utilization达97.5% (FA-2)和78.4% (SSD)→end-to-end speedup 1.56× vs A100、2.08× vs H100→面积减少89.8%、功耗减少63.8%。设计核心洞察：Hybrid模型中FA-2和SSD的瓶颈性质不同但根因相似——都是operation dependency导致无法充分利用计算资源——而fine-grained pipelining是一个统一解决方案，只需要一个能高效执行fine-grained pipeline的硬件架构（URSC）即可同时解决两个问题。

## 56-AxCore- A Quantization-Aware Approximate GEMM Unit for LLM Inference.pdf

- baseline方法是什么？
  - baseline是传统FP GEMM加速器（FPC），使用标准FP fused-multiply-add (FMA)单元实现混合精度或全精度矩阵乘法。全栈执行例子：LLM推理时，(1) 算法pipeline层：weight通过传统round-to-nearest量化映射到INT4/FP4，dequantization使用FP乘法器 w=s×wq。量化格式uniform（单一E2M1或INT4），未利用不同层的分布差异。(2) 系统框架层：论文未明确说明（AxCore自身是加速器IP，非Serving框架）。(3) 编译框架层：论文未明确说明。(4) kernel调度层：标准weight-stationary systolic array，PE内每个MAC操作=FP16乘法(宽datapath: sign XOR + exponent add + mantissa multiply) + FP32累加。每个PE需18-24bit宽乘法器和32bit累加器。量化后GEMM需先dequantize→FP乘法，或直接mpGEMM用FP乘法器乘低比特权重→仍为宽位宽乘法。(5) 硬件架构层：FPC PE面积大（FP multiplier + FP adder）。FIGNA引入INT4-FP16 mpGEMM unit用integer multiplier替代FP multiplier，但乘法器面积仍随bit-width二次增长。FIGLUT用look-up table (LUT)替代multiplier但bit-serial架构延长计算cycles→能量效率受损。所有baseline均未处理FPMA subnormal问题，在低比特FP格式下精度损失严重。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - AxCore通过co-design FPMA+量化，消除GEMM中的乘法器，用轻量整数加法器替代。全栈执行例子（W4A16 OPT-13B推理，一个token的feed-forward GEMM: activation[1×5120] × weight[5120×20480]为例）：
  - **(1) 算法pipeline层——解决Baseline单一量化格式+FP乘法造成的精度-效率tradeoff**：
    - Baseline：uniform FP4格式（如E2M1）全局使用→无法适配不同层的weight分布差异（Llama2-7B中Layer 0 weight呈sharp peaks适合power-of-two类格式，Layer 29呈uniform分布需要均匀量化）→量化误差大。FP乘法在低比特下受subnormal值影响——FP4中subnormal可represent高达0.5的值且无hidden leading 1→FPMA近似log2(1+M)≈M数学上失效→PPL严重退化。
    - AxCore：(a) Block-wise adaptive format-aware quantization：对每组128×64 (OPT) 或 64×64 (LLaMA2) weight block，离线遍历{E3M0, E2M1, E1M2}并选min ||A·W_d-A·W||²的格式。Layer 0选E3M0 (power-of-two)→Layer 29选E2M1/E1M2 (uniform)→每层最优格式令quantization error最小化。(b) FPMA-friendly quantization reformulation: wq=clamp(round(w-S+B-C))，wr=wq+S-B+C2，量化-去量化的对称加法结构使C和C2抵消→wr≈w且仅依赖加法/减法无rounding bias (vs 传统round-to-nearest的除法和乘法rounding error accumulation)。(c) Subnormal Number Conversion (SNC)：运行时对FP4 subnormal值查表映射到最近normalized编码→恢复FPMA hidden leading 1近似有效性→PPL从naive mpFPMA的11.83降至10.49（OPT-6.7B +SNC）。(d) Mean-based constant error compensation：precompute C1=avg error across所有(m_a,m_w) mantissa组合→单个常数补偿per格式对→ZERO runtime开销→PPL从10.49降至10.25（+Compensation）。
  - **(2) 系统框架层**：论文未明确说明（AxCore为GEMM加速器IP，可嵌入Serving框架如vLLM/TensorRT-LLM的GEMM kernel层）。
  - **(3) 编译框架层**：论文未明确说明。
  - **(4) kernel调度层——解决Baseline PE内乘法器面积/功耗巨大问题**：
    - Baseline FPC PE：每个PE执行 A×Wq → 15-24bit FP乘法器（sign XOR + exponent 5bit adder + mantissa 10bit multiplier + normalization）→面积大、功耗高，在64×64 array中复制64次×4 tile。FIGNA用integer multiplier替代仍是硬件乘法器→面积随bit-width二次增长（W4→W8 area跳升）。FIGLUT用LUT但bit-serial需多次cycle→能量效率差。
    - AxCore PE：Correction Advancing使PreAdd预计算 T=A-B1+C1 一次per row（仅一个15-bit adder outside PE），PE内仅需7-bit integer adder完成 R=T+Align(Wq)→eliminate所有乘法器。SNC单元仅占PE面积3.5%。Normalization Postponing将LZD/Shift/Round移出PE至共享Norm模块→在n×n array中减少logic duplication n倍。FPMA-based dequantization (AxScale)：O=Oq+S-B+C2用两次整数加法替代缩放乘法。整体效果：PE面积比FIGLUT减少34% (W4-FP32)→比FIGNA减少32-56%；Compute Density 6.7× over FPC, 1.7× over FIGNA (W4A16)；Energy：2.2× reduction over FPC, 1.5× over FIGNA。
  - **(5) 硬件架构层——解决Baseline systolic array PE逻辑冗余和precision-loss问题**：
    - Baseline array：每个PE完整执行乘法+累加+归一化，导致大量logic duplication（LZD/Shift/Round每PE一套→array总逻辑 = n² × per-PE-overhead）。多格式支持需要独立datapath→进一步增加area。
    - AxCore array：(a) Centralized PreAdd: per-row one 15-bit adder共享→减少row-wide冗余。(b) Deferred Normalization: per-column one Norm module共享→减少column-wide冗余。(c) Unified S1E3M2 internal format: 所有FP4子格式（E3M0/E2M1/E1M2）在SNC输出后统一→下游logic (adder/accumulator/Norm/AxScale) 格式无关→大幅减少multiplexing和conditional logic。(d) Weight Buffer + Unified Buffer + Vector Unit + CTRL模块化设计→支持标准LLM accelerator集成。64×64 array 4× tiling达到6.7×→12.5× compute density超过其他设计。(e) 全array面积：W4-FP16下比FIGLUT小31%，比FIGNA小37%；W8场景下平均比FIGNA小55%（因FPMA加法器面积不随bit-width二次增长，而FIGNA的INT multiplier在W8下面积翻倍）。
  - **全栈对比总结**：Baseline (FPC/FIGNA/FIGLUT) 在LLM推理GEMM中受制于乘法器面积/功耗/延迟的物理瓶颈，FPMA虽然能消除乘法器但在deep LLM的mpGEMM和低比特FP格式中受subnormal+approximation error困扰。AxCore通过三层co-design解决：(1) Algorithm: adaptive format-aware quantization + subnormal handling + mean compensation → 保持或超越baseline accuracy (PPL: OPT-6.7B 11.01 vs FP16 10.86, INT4 11.28; zero-shot avg 81.78% vs FP16 81.91% on LLaMA2-70B)；(2) Kernel: mpFPMA PE + correction advancing + Norm postponing → multiplier-free systolic array with 7-bit adder per PE；(3) Hardware: modular accelerator (weight/activation buffer + PE tile array + shared PreAdd/Norm/AxScale) synthesized at 28nm 1GHz → 12.5× compute density over FPC, 53-70% improvement over SOTA INT4 accelerators (FIGLUT/FIGNA)。核心洞察：FPMA和低比特量化的组合并非简单的"approximate computing + less bits"，而是两者在硬件上的协同——FP4的窄位宽让FPMA的整数加法器位宽更小（7-bit vs 15-bit），FPMA的additive nature又反过来消除了dequantization的乘法需求（AxScale用加法替代），形成正向反馈回路。

## 59-Proteus- A High-Throughput Inference-Serving System with Accuracy Scaling.pdf

- baseline方法是什么？
  Baseline是传统基于hardware scaling的推理serving系统（Clipper、Sommelier、INFaaS），通过添加更多设备或使用更强加速器来应对查询负载增长。在全栈执行中（以异构集群上EfficientNet image classification，Twitter trace工作负载为例）：
  - **算法层**：使用预训练的多种DNN模型变体（通过quantization [39]、pruning [6]或architecture design生成），各变体具有不同的accuracy-throughput trade-off。例如EfficientNet B0-B7，B0精度最低但吞吐最高（~85 QPS on V100），B7精度最高但吞吐最低（~20 QPS on V100）。Baseline Clipper-HA始终使用最accurate变体→accuracy始终最高但throughput低；Clipper-HT始终使用最fast变体→throughput高但accuracy最低且固定不变。
  - **系统框架层**：Clipper静态预加载资源分配——startup时确定模型变体-设备映射，运行时不变。请求路由固定：每个query type固定分配给预配置的设备。Sommelier在单设备上根据负载切换模型变体但不做cluster-level placement。INFaaS使用greedy heuristic动态选择模型变体和设备映射，但因resource allocation在query arrival的关键路径上（每个query到达触发分配决策），heuristic必须快速→易陷入局部最优（如高峰期INFaaS throughpu显著下降+SLO violations上升）。
  - **编译框架层**：论文未明确说明（使用ONNX Runtime的预编译CUDA kernel）。
  - **kernel调度层**：Clipper使用AIMD (Additive-Increase Multiplicative-Decrease) reactive batching——batch size逐步加1直到timeout出现，然后减半。Nexus使用work-conserving early-drop batching——当前batch完成后立即启动下一个batch。在uniform arrival trace下表现良好，但在Poisson/Gamma (micro-scale burst) trace下因无法预判队列状态——Clipper出现timeout后才反应→高SLO violations；Nexus因work-conserving在burst后资源不足→低throughput。
  - **硬件架构层**：20× Intel Xeon CPU + 10× GTX 1080 Ti + 10× V100 GPU异构集群，Kubernetes+Docker部署，ONNX Runtime执行。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法Proteus通过三个层面的协同一一解决baseline缺陷：(A) 静态/次优resource allocation→MILP全局最优解；(B) resource allocation阻塞推理关键路径→Control/Data path解耦；(C) reactive/work-conserving batching在非均匀请求到达下失效→Proactive Non-Work-Conserving Adaptive Batching。Proteus全栈执行例子（同一异构集群上EfficientNet classification, Twitter trace）：
  - **算法层**：同baseline，使用预训练的multiple model variants，accuracy scaling通过动态切换模型变体而非修改模型本身实现。关键差异：Proteus在高峰期选择略低精度但高吞吐的变体、低峰期切换回高精度变体——通过MILP精确计算"正确的scaling量"避免over-scale（不必要精度损失）或under-scale（SLO violations）。
  - **系统框架层→MILP + Control/Data Path Decoupling**：
    - **缺陷A解决（次优resource allocation）**：将resource management形式化为MILP——Object: max Σ_q a_q（系统effective accuracy），变量{x_d,m}（bool, device d是否host model m）、{y_d,q}（[0,1], query type q路由到device d的比例）。三项约束联合优化：(1) Model Selection——从M个变体×N台设备=O(M^N)配置空间选择最优集合；(2) Model Placement——考虑异构设备上各变体的不同throughput profile（EfficientNet B5在V100 80 QPS vs 1080 Ti 55 QPS）；(3) Query Assignment——多租户下各query type的请求按最优比例分配到不同设备。使用Gurobi精确求解（平均4.2秒），保证全局最优而非greedy heuristic的局部最优→对比INFaaS-Accuracy（heuristic）：accuracy drop从13.7%降至4.85%（2.8× reduction），对比Sommelier（单设备无placement优化）：从16%降至4.85%（3.2× reduction）。
    - **缺陷B解决（阻塞关键路径）**：Controller（Resource Manager）与Load Balancer完全分离。MILP solver在macro-scale需求变化时异步触发（默认30秒周期或Monitoring Daemon检测burst），产出optimal allocation后通过Docker container启动/终止模型变体实例→Request Router更新路由表（O(D×M×Q), <1ms lookup）。Query serving的data path走Load Balancer→Request Router→Worker，完全绕过Controller→resource allocation不增加query latency。对比INFaaS的每个query arrival触发heuristic分配→greedy必须快速→牺牲解质量。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层→Proactive Non-Work-Conserving Adaptive Batching**：
    - **缺陷C解决（reactive/work-conserving batching失效）**：Proteus batching算法基于两个关键设计：(1) Proactive——对于队列中q个请求，计算T_max_wait(q+1) = T_exp(1) - T_process(q+1)（第一个请求过期时间减去处理q+1个请求的耗时），若此时点前第q+1个请求到达则更新T_max_wait(q+2)继续等待，超时则立即以当前batch=q执行。算法始终在请求SLO violation发生前主动决策，而非AIMD的timeout后被动反应。(2) Non-Work-Conserving——在等待期间设备可能空闲，但与work-conserving（Nexus：上批完成立即执行下批）相比，非work-conserving能积累更多请求形成更大batch→提升throughput同时避免SLO violations。在Poisson trace上SLO violation ratio比Nexus batching低2-3×，比Clipper AIMD低3.8-4×；在Gamma (highly bursty) trace上优势更明显。
    - **Batching与Resource Allocation协同**：Resource Manager根据宏观QPS计算每个(model variant, device, batch size)的最大throughput capacity P_d,m,q——最大batch size由两个上限决定：min(SLO-compatible batch size, memory-fit batch size)，该值用于MILP的P_d,m,q参数。Adaptive Batching处理micro-scale的inter-arrival变化→使provisioned capacity在非均匀到达下仍能实现。
  - **硬件架构层**：同baseline集群硬件。差异在于Proteus通过accuracy scaling在固定硬件上提升throughput 60%同时减少SLO violations 10×（对比无accuracy scaling的Clipper-HA）。消融实验验证：移除Model Selection → SLO violations最大退化（无法scale accuracy）；移除Model Placement → accuracy最大退化（16% max drop，模型放错设备无法充分利用throughput）；移除Adaptive Batching → SLO violations second-highest退化。
  - **全栈对比总结**：Baseline (Clipper/Sommelier/INFaaS) 在资源受限的异构推理集群中面临三个问题：(1) Resource allocation——model selection、placement、query assignment的不同组合导致3.3× throughput差异和8% accuracy差异，静态或greedy分配在高峰期远离Pareto frontier；(2) Critical path——resource allocation在query serving路径上（INFaaS）→求解必须快速→牺牲解质量；(3) Batching——reactive/work-conserving batching在micro-scale burst下timout或低效。Proteus通过三层co-design解决：(1) System: MILP joint optimization + control/data path decoupling——resource allocation异步于serving，MILP保证全局最优；(2) Kernel: proactive non-work-conserving adaptive batching——在请求SLO violation前主动决策，非work-conserving换吞吐提升；(3) Architecture: accuracy scaling替代hardware scaling——在固定硬件上通过动态accuracy-throughput trade-off适配负载变化。核心洞察：accuracy scaling不只是一个"降精度换吞吐"的简单trick，而是一个需要联合优化model selection×model placement×query assignment三个子问题的resource management问题——三者缺一均导致sub-optimal解。

## 5-FuseMax_Leveraging_Extended_Einsums_to_Optimize_Attention_Accelerator_Design.pdf

- baseline方法是什么？
  Baseline是FLAT [ASPLOS'23]——state-of-the-art attention accelerator on spatial architecture。FLAT使用3-pass attention cascade（PyTorch/TensorFlow风格）融合attention的matmul到2D spatial array、softmax到独立1D array。Baseline全栈执行例子（以BERT-Base on TPU-style spatial array, seq len=256K为例）：
  - **算法层**：3-pass attention cascade（Cascade 4）。Pass 1: QKm,p = Qe,p × Ke,m → 计算softmax numerator QK（Einsum 33）；然后GMp = max_m(QKm,p) 计算global max（Einsum 34）。Pass 2: SNm,p = e^(QKm,p - GMp) 计算numerator（Einsum 35）；SDp = Σ_m SNm,p 计算denominator（Einsum 36）。Pass 3: Am,p = SNm,p / SDp 得到softmax output（Einsum 37）；AVf,p = Am,p × Vf,m 得到attention output（Einsum 38）。由于dependency：GMp必须等所有M fiber的QK完成后才能start SN（需re-read同一M fiber），SDp必须等所有SN完成后才能start A。因此每个M fiber需被遍历3次——live footprint为O(M)（sequence length）。FP16精度，exact attention（无approximation）。
  - **系统框架层**：FLAT mapping将QK matmul和AV matmul映射到2D spatial array（利用high compute throughput），softmax（exp/max/sum/div）映射到1D array（仅230或256 PEs）。由于无fusion跨pass boundary，QK和A tensors必须在pass间写入DRAM或global buffer。
  - **编译框架层**：论文未明确说明。FLAT使用custom dataflow search（基于Timeloop-like modeling），非通用编译框架。
  - **kernel调度层**：FLAT的3-pass cascade限制fusion——Einsums 33-34（pass 1）可以fuse（produce/consume tile at a time），Einsums 35-36（pass 2）可以fuse，Einsums 37-38（pass 3）可以fuse。但pass间无法fuse——必须等整个M fiber的pass 1完成才start pass 2。当seq len增长→M fiber超过on-chip buffer capacity→forced to spill to DRAM→memory bandwidth bound。同时，1D array的softmax compute（exp + sum + div）成为瓶颈：2D array（大量PE）必须等1D array（少量PE）完成softmax才能继续AV matmul → 2D array severe under-utilization。
  - **硬件架构层**：TPUv2/TPUv3风格spatial array——2D PE array（MAC only，固定数据流）+ 1D PE array（256 PEs for softmax compute）+ global buffer + DRAM。2D PEs数量 >> 1D PEs数量，导致compute imbalance。On-chip buffer需容纳整个M fiber of QK tensor（shape M × P），当seq len = 1M tokens时需100s of MB buffer → 超出capacity则spill → DRAM traffic = attention energy dominant factor。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出FuseMax：基于Cascade of Einsums形式化分析attention算法的pass特性，选择1-pass cascade（FlashAttention-2 variant），通过novel mapping和binding实现spatial array上的高效执行。FuseMax全栈执行例子（同一BERT-Base, seq len=256K为例）：
  - **算法层→1-Pass Cascade + Division Reduction**：解决Baseline缺陷A（3-pass导致large live footprint和forced spill）。
    - 使用Cascade 5（1-pass attention cascade）：用running max RM替代global max GM。每iteration m1处理一个M0 chunk（M0 × P tile on-chip）→ 计算local max LM → update running max RM → 用RM compute local numerator SLN/denominator SLD/numerator-times-V SLNV → 用correction factor PRM = e^{RM_old - RM_new} rescale旧running accumulator到新max → 将新local值加到corrected running值。1次traversal of M fiber → live footprint仅为O(M0 × P0) = tile size，independent of total seq len M。
    - Division reduction: 先SN × V reduce across M → 再÷SD（F×P次除法 vs baseline M×P次）。M >> F（如256K vs 64）→ division数减少~4000×。
    - 代价：1-pass cascade比3-pass增加额外compute（correction factor exp、extra multiply-add for rescaling）。但memory bandwidth-limited workload可afford此trade-off。
  - **系统框架层**：FuseMax作为accelerator architecture的mapping/binding规范，非传统系统框架。Output为Timeloop model of accelerator design。
  - **编译框架层**：论文未明确说明。使用Timeloop search最优mapping for each Einsum，non-compilation framework。
  - **kernel调度层→FuseMax Mapping + Binding**：解决Baseline缺陷B（1D array softmax bottleneck → 2D array under-utilization）。
    - **Mapping**：对M和P双重tiling（M1×M0, P2×P1×P0），M0×P0 = #2D array PEs。所有Einsum在M0/P0层maximally fuse——ComputeRNVTile包含BQK + LM + RM update + SLN + SLD + SLNV + PRM + SPD + RD update + SPNV + RNV update（Einsums 44-54）。2D array同时执行tensor product（BQK）和softmax exponentiation（SLNV），1D array执行sum/max reduce（SLD/SPD/SPNV/RNV）。这消除了baseline中2D等1D的imbalance——softmax的计算（exp部分）也在2D array上并行。
    - **Binding（Epoch-level pipelining）**：不同epoch处理不同tile coordinates（Figure 4）。Epoch i完成BQK/LM tile d → RM tile c → ... → AV tile a。Software pipeline确保：当BQK/LM的tile d drain后，2D array不idle而是开始下一tile的BQK或继续SLNV compute。1D array在等待RM drain时不idle而是处理earlier tile的SPNV/RNV。
    - **Binding（Cycle-level interleaving）**：2D array内相邻cycle交替计算BQK和SLNV（'BQK|SLNV' in Figure 5），每条neighbor-neighbor link每cycle都active。1D array同时并发SPNV和RNV。这使得PE utilization接近100%——baseline中2D array >50% idle time被消除。
    - **No Spill Guarantee**：由于1-pass cascade的live footprint仅为tile size（M0×P0），无论seq len多大（1M+），所有intermediate都在on-chip buffer内——无DRAM spill。≥95%的FuseMax energy为2D array MACC compute energy。
  - **硬件架构层→FuseMax PE + 2D-1D Tight Coupling**：解决Baseline缺陷C（2D array只能做MAC，softmax全在1D → imbalance）。
    - FuseMax PE（Figure 3c）：在灵活数据流MAC PE基础上增加exponentiation能力（通过6 sequential MACs实现，无dedicated exp unit）。2D PEs可直接计算e^(BQK - RM)，baseline中此操作在1D array串行执行。
    - 2D array底部与1D array的tight coupling：fixed-latency neighbor-neighbor communication，使spatial reduction（如从M0×P0的BQK reduce到P0的LM）低开销——drain后直接feed到1D array进行RM update。
    - Iso-area comparison：FuseMax比FLAT面积小6.4%。
  - **全栈效果对比**：
    - Attention kernel: FuseMax 6.7× speedup over FLAT，79% energy。10× over unfused baseline。
    - End-to-end transformer inference: FuseMax 5.3× speedup over FLAT，83% energy。7.6× over unfused baseline。
    - At 1M tokens: FuseMax 7.5× speedup over FLAT（attention dominates total compute at long seq len）。
    - Utilization: FuseMax ~100% both 2D and 1D arrays；FLAT drops to <20% at long seq len（memory bandwidth bound）。

## 60-SpotServe- Serving Generative Large Language Models on Preemptible Instances.pdf

- baseline方法是什么？
  Baseline是现有LLM serving系统（以FasterTransformer [5]为代表）在可抢占实例上的两种朴素部署方案：(1) Request Rerouting——固定并行配置，实例被抢占时将中断的请求reroute到其他可用pipeline重新计算；(2) Model Reparallelization——在实例变化后改变并行配置，但需要restart和reinitialize所有实例，丢失所有model parameters和KV cache，从磁盘重新加载模型并重新计算中断请求。Baseline全栈执行例子（以GPT-20B on 4×T4 GPU per instance, P=2 pipeline stages, M=8 tensor-model parallelism为例）：
  - **算法层**：GPT-20B (74.5GB参数)，autoregressive decoder-only Transformer。每个请求执行一次prefill（并行处理prompt tokens）+ incremental decoding（每iteration生成1个token，KV cache避免recomputing）。l_exe(S_out|S_in) ≈ t_exe(S_in) + S_out × t_exe(1)，t_exe(1) ≈ 14.373s/128 ≈ 0.112s per token。
  - **系统框架层**：FasterTransformer (FT)部署在spot GPU instances上。Inference Server在on-demand CPU instance上运行。FT的分布式推理pipeline使用(D=1,P=2,M=8)配置，16个GPUs（4 instances×4）。FT不支持instance preemption：当1个instance被抢占→整个pipeline链式崩溃（chain crashing）——pipeline parallelism的data dependency和tensor parallelism的all-reduce collective communication无fault tolerance。受影响GPUs halt而非物理终止。Rerouting: 丢弃整个pipeline→active requests reroute到其他pipeline重新从头decode。Reparallelization: 等待新instance后重新load model parameters (~2分钟 for GPT-20B) →重新初始化所有GPUs→restart所有interrupted requests（KV cache全部丢失→recompute所有已生成tokens）。
  - **编译框架层**：论文未明确说明（FT使用预编译CUDA kernel，无运行时compilation）。
  - **kernel调度层**：FT的tensor parallelism通过NCCL all-reduce同步各shard的中间结果。Pipeline parallelism通过p2p send/recv传递stage间activations。Preemption发生时：running decoding iteration被中断→未完成的batch中所有requests全部丢失progress→KV cache（GB级别，如LLaMA-30B 1.7GB/sequence）全部丢失。Grace period (30s)未被有效利用——现有系统在收到preemption通知后立即suspend推理引擎以保留时间做context保存，导致grace period内大量可用计算时间浪费。
  - **硬件架构层**：AWS g4dn.12xlarge instances (每instance 4×NVIDIA Tesla T4 16GB GPU, inter-instance 50Gbps网络)。GPU间通信：intra-instance通过PCIe，inter-instance通过50Gbps以太网。T4 GPU无NVLink。Grace period 30s内有效的inference时间未被利用——FT单次GPT-20B decoding需14.373s (S_out=128)，grace period仅能完成约2个full requests的重启后推理，远不够完成完整batch。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出SpotServe，通过四个关键技术解决Baseline的三个核心缺陷：(A) preemption后整个pipeline halt→chain crashing（Rerouting和Reparallelization都无法避免）；(B) 重新初始化开销巨大（model reload ~2min + KV cache recomputation）；(C) Grace period未被有效利用。SpotServe全栈执行例子（同一GPT-20B on T4 instances，但支持动态配置切换）：
  - **算法层**：同Baseline，GPT-20B autoregressive decoding不变。但SpotServe通过stateful inference recovery在token粒度（而非request粒度）commit推理进度——利用LLM的autoregressive性质：每生成1个token即完成1个iteration，可在此处安全中断。
  - **系统框架层→Dynamic Reparallelization + Context Migration + Stateful Recovery**：
    - **缺陷A解决（Chain Crashing）**: SpotServe的Parallelization Controller在收到preemption ahead-of-time notification后，不等待实例真正被抢占即主动调整并行配置。Algorithm 1 (ConfigOptimizer)在线运行（<1s overhead）：若存在C使得φ(C)≥α_t，选最小化l_req(C)且cost最低的配置；否则最大化吞吐。Config从(D=1,P=2,M=8)主动切换到(D=1,P=3,M=4)或(D=2,P=3,M=4)等，利用grace period完成迁移，避免pipeline halt。维护额外2个instance作为candidate pool加速instance substitution。
    - **缺陷B解决（Reinitialization开销）**: Device Mapper将重配置时的设备映射建模为二分图匹配问题——GPU设备为左节点，新配置的pipeline-stage-shard位置为右节点，边权重=可复用的model params+KV cache量。用KM算法O(n^3)求最大权重匹配，最大化context复用。Migration Planner用渐进式迁移：优先迁移KV cache→然后按内存优化顺序迁移layer weights→优先完成Stage 1迁移使其立即开始服务，与后续stage迁移overlap。Context Daemon进程独立于推理引擎，instance被抢占后仍存活，避免reload context到GPU。
    - **缺陷C解决（Grace Period利用）**: Interruption Arranger用JIT Arrangement：preemption前S_t = argmax{lexe(S|C_t) < T^- - Tmig}最大化grace period内的推理迭代数；acquisition前S_t = argmin{lexe(S|C_t) ≥ T^+}最小化等待时间。Stateful inference recovery：Context Daemon维护每个request的cache context（KV cache），中断后可reroute到其他pipeline从committed token state直接继续decoding，无recomputation。Fault-tolerance机制处理连续中断和grace period重叠——若cache context的all replicas丢失→回退到从磁盘/云存储重新加载model context。
  - **编译框架层**：论文未明确说明（基于FasterTransformer pre-built CUDA kernels）。
  - **kernel调度层→Context Daemon + NCCL Migration + CUDA IPC**：
    - Context Daemon管理两类GPU context：model context（模型参数，由FT推理访问）和cache context（per-request KV cache + intermediate activations）。通过CUDA IPC跨进程共享context指针（Context Daemon和FT属于不同进程）。Memory allocation替换：FT原有的tensor分配改为从Context Daemon proxy获取。
    - Context migration通过NCCL batched async send/recv primitives实现——在inter-instance 50Gbps链路上传输tensor。通信buffer GPU memory动态分配/释放。Mutex lock per tensor在迁移期间阻塞推理，迁移完成后释放。
    - Progressive migration schedule: MemOptMigPlanner以buffer memory上限U_max为约束，skip context migration会超U_max的layer→对剩余layer解min-max问题（选使max instance buffer usage最小化的layer优先迁移）→形成内存高效迁移顺序。
    - 效果：GPT-20B最小所需GPUs从16降至12（因migration planner优化了内存使用）→扩大了Parallelization Controller的配置探索空间。
  - **硬件架构层**：同Baseline硬件（AWS g4dn.12xlarge, 4×T4 GPU）。SpotServe的monetary cost优化：通过选择满足延迟约束且使用最少instances的配置来最小化cost。On-demand instances作为补充（当spot allocation失败时），但优先级低于spot（cost较高）→释放时on-demand优先释放。结果：相比纯on-demand instances节省54% monetary cost，同时P99 tail latency增加<18%。Mixing on-demand instances帮助alleviate overload（因faithful instance acquisitions）。
  - **全栈效果对比**：
    - P99 tail latency: SpotServe vs Reparallelization 1.34-2.43× reduction, vs Rerouting 2.14-9.13× reduction（LLaMA-30B最大模型上最显著）。
    - Ablation: 移除Parallelization Controller→tail latency +179% (trace BS); 再移除Migration Planner→+1.4×/+3.1× (AS/BS); 再移除Interruption Arranger→+29%; 全部移除→plain context maintenance。组合优化减少tail latency 1.61× (AS) / 3.41× (BS)。
    - MAF production trace上波动负载：SpotServe P99 latency reduction up to 2.94× vs Reparallelization, 1.73× vs Rerouting。

## 61-Duplex- A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching.pdf

- baseline方法是什么？
  Baseline是GPU-only系统（NVIDIA H100 [35]）使用continuous batching [56]执行MoE-based LLM推理。Baseline全栈执行例子（以Mixtral 47B on 4×H100 GPUs, batch size=128, Lin=2048, Lout=1024为例）：
  - **算法层**：Mixtral decoder stack：QKV Generation (FC GEMM) → Attention (GQA, deggrp=4) → Projection (FC GEMM) → MoE layer (Gate→select top-2 of 8 expert FFNs per token→FC1→Gated SiLU Activation→FC2→FC3) → 重复32 layers → LM Head。FP16 precision。
  - **系统框架层**：Continuous batching scheduler将inference分解为prefill和decoding stages并stage-level batch。模型分布[46]：tensor parallelism for non-expert weights（QKV gen/Proj按列切分到4 GPUs）+ expert parallelism for MoE（8 expert FFNs分配到4 GPUs, 2 experts per GPU）+ data parallelism across nodes。Mixed stage中prefill的2048 tokens和decoding的128 tokens batch在一起处理。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：所有GEMM/GEMV/attention kernel均在H100 GPU SMs上执行。MoE layer decoding时：每GPU处理2 experts，每个expert收到~32 tokens（128 tokens × top-2 / 8 experts均匀分布）→小batch size导致GEMV-like低Op/B操作。Attention layer decoding时：每request独立KV cache→128个requests各自的Q vector (1×D) × K matrix (seq_len×D) → low Op/B。GPU compute utilization: MoE layer <11%, attention layer <2.06%。Mixed stage中prefill使Op/B暂时升高（2048 tokens sharing same expert weights），但decoding-only stage占绝大多数（因Lout=1024 >> 1 prefill）。
  - **硬件架构层**：H100 GPU：80GB HBM3 (16GB per stack, 8-hi 2-ranks), 5 HBM stacks, 3.35 TB/s memory bandwidth。GPU SM大量计算单元但HBM带宽有限（interposer physical limit）。Op/B = compute FLOPS / memory bandwidth。低Op/B操作导致memory bandwidth bottleneck→大量DRAM access加载expert weights和KV cache→高延迟，高能耗。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Duplex——一种集成xPU（高Op/B处理器，等同H100）和Logic-PIM（低Op/B处理器，位于HBM logic die）的single-device架构，通过expert和attention co-processing最大化两种处理单元的利用率。
  论文方法全栈执行例子（以Mixtral 47B on 4 Duplex devices, batch size=128, Lin=2048, Lout=1024, Duplex+PE+ET配置）：
  - **算法层**：模型算法不变（Mixtral decoder stack）。
  - **系统框架层**：Continuous batching scheduler同上。关键区别：tensor parallelism for MoE layers（Duplex+PE+ET）替代expert parallelism——每个expert的weight列切分到所有4 devices，使每device处理全部8 experts。Memory allocation划分为4个bank bundle-aligned memory spaces，expert FFNs round-robin分配，KV cache交替分配在3 spaces，第4 space存prefill Q/K/V。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：Decoding-only stage中——QKV Gen/Proj: xPU执行（高Op/B GEMM, batch=128）。Attention decoding: Logic-PIM执行（低Op/B, 128 independent requests的GQA attention→narrow GEMM with deggrp=4→Logic-PIM GEMM modules以4x HBM bandwidth加速memory-bound操作）。MoE decoding: Gate output→expert co-processing查lookup table分配（如expert 3处理30 tokens→xPU, expert 0处理4 tokens→Logic-PIM）。Logic-PIM experts: 每Logic-PIM stack处理expert weight的列slice→GEMM→Gated Activation→partial sum→所有experts完成后xPU执行single all-reduce。xPU experts: 正常GEMM on H100 SMs。Mixed stage中——prefill attention (2048 tokens): xPU执行（高Op/B GEMM）；decoding attention: Logic-PIM执行（attention co-processing并行）。MoE混合处理：prefill tokens使Op/B升高→更多experts由xPU处理。
  - **硬件架构层**：Duplex device含xPU（等同H100规格）+ 5 Logic-PIM stacks（替换HBM3标准stacks）。Logic-PIM核心硬件：(1) DRAM die bank bundle——16 banks per pseudo channel分为upper/lower 8-bank bundles，通过switches分离xPU和Logic-PIM active paths，8 banks simultaneous read→4x effective bandwidth。(2) Dedicated TSVs placed in power TSV area（22um pitch [49]）→4x TSV count仅9% area overhead，短datapath减少energy。(3) Logic die processing units——32 GEMM modules (512 FP16 MACs @650MHz)、activation module (SiLU)、softmax module、2×1MB buffers。(4) xPU和Logic-PIM通过bank bundle parallelism同时读取数据。
  - **Baseline缺陷 → Duplex解决方案映射**：
    1. **GPU在低Op/B操作上compute utilization极低**（MoE <11%, attention <2.06%）→ Duplex用Logic-PIM（4x HBM bandwidth, 8 Op/B peak）处理低Op/B的MoE和attention decoding→median TBT降低58.3%。
    2. **Prior PIM（Bank-PIM）无法处理Op/B>1的MoE层**（计算能力不足）→ Logic-PIM在logic die上放置更多处理单元（32 GEMM modules per stack, 512 MACs each）→可处理Op/B 1-32范围的操作，EDAP在Op/B≥8时优于Bank-PIM。
    3. **Prior PIM无法处理GQA的Op/B升高**（GQA deggrp=4-8使attention Op/B升至4-8）→ Logic-PIM的compute-to-bandwidth ratio=8，匹配GQA的Op/B范围。
    4. **Heterogeneous系统在mixed stage中无法处理MoE的Op/B波动**（prefill使Op/B升高→low-compute device成为瓶颈→tail latency升高）→ Expert co-processing动态将高token数expert分配给xPU、低token数expert分配给Logic-PIM→p99 TBT和T2FT保持competitive（甚至有16-26% improvement over 2xGPU for short Lin）。
    5. **Heterogeneous系统因MoE weight duplication浪费memory capacity**→ Duplex的xPU和Logic-PIM共享同一HBM device memory→无需weight duplication→更大KV cache capacity→更高max batch size→更高throughput。
    6. **Split Prefill/Decoding节点导致prefill/decoding node utilization不平衡和weight duplication**→ Duplex通过co-processing在单一device上同时利用xPU和Logic-PIM→避免utilization imbalance和memory waste。
  - **效果**：Duplex+PE+ET vs GPU baseline: up to 2.67x throughput, 2.57x lower E2E latency, 42.03% lower energy。Area overhead仅14.71% of logic die。

## 65-MagiCache- A Virtual In-Cache Computing Engine.pdf

- baseline方法是什么？
  Baseline 是现有 array-level in-cache computing 架构——以 EVE [3] 为代表（论文实现为 SplitCache/Split-8）。EVE 将 L2 cache 的一半 SRAM 阵列（4 of 8 ways）静态预配置为 computing arrays，剩余阵列为 storage space（传统 cache）。Computing arrays 的所有行全部用作 computing lines，均匀预分配给 32 个 vector registers（每 register 固定 8 rows/array）。此方案存在三个核心缺陷：(A) 静态 coarse-grained 划分——computing/storage 空间比例在运行前固定，不同应用需求不匹配（Fig. 2: matmul 最优 62.5% vs backprop 最优 50%）；(B) 缓存空间严重低利用率——computing arrays 中所有 256 rows 分配给 32 个 vector registers，但多数应用仅使用 2-3 个 register（如 matmul 仅用 v0/v1），未用 register 的 rows 浪费且不能用作 cache；cache utilization 仅 ~56%（Table 8）；(C) 同步执行模式——所有 computing arrays 同时执行每条指令，后等待全部完成后才执行下一条，bursty memory access 的延迟累积，sync stall 长。
  Baseline 全栈执行例子（SplitCache/Split-8, matmul 向量程序，k=8 rows/register/array）：
  - **算法层**：RISC-V vector 矩阵乘法——vle32.v 加载 B 行到 v1 → vle32.v 加载 C 行到 v0 → vmacc.vx v0, a5, v1（v0 += v1 × 标量 A[i,k]）→ vse32.v 写回 C。Vector length=65536 bits=2048 elements（Split-8: 32 regs × 8 rows/reg × 512 bits/row=131072 bits total, 每次有效使用 2 registers）。
  - **系统框架层**：论文未明确说明（直接使用 RISC-V vector ISA 编程，无额外系统框架层）。
  - **编译框架层**：论文未明确说明（LLVM 17 + RISC-V GNU toolchain 编译 vector intrinsics，无专有编译优化）。
  - **kernel调度层→SplitCache 静态预配置**：L2 cache 8-way 512KB 中 4-way (256KB) 固定为 computing space。每 way 含 8 个 computing arrays → 共 32 computing arrays，每 array 256 rows。所有 rows 均匀分配为 32 个 vector register（每 array 256/32=8 rows per register）。Matmul 实际仅用 v0/v1 两 register——但 30 个未用 register 的 rows 被 computing arrays 占用无法释放 → cache capacity 减至 256KB, utilization ~56%。当 computation 需求低时（低 arithmetic intensity 应用），computing arrays 的空间浪费无缓解。Vector load 512 elements (32 cachelines/batch) → 32 MSHRs 刚好容纳但同步模式下需所有 arrays 完成当前 load 才能开始 compute → sync stall 占时间显著比例。
  - **硬件架构层→EVE style Computing Array**：Computing arrays 采用 bit-hybrid data layout（元素以矩形区域分布在 word-line 和 bit-line 上）。外围电路含 logic/add/shift layers。与 storage space 物理分离——computing arrays 的 data layout 与 cacheline 不同，数据在两个空间间移动需 transpose 操作 → 额外延迟。Storage space 仅剩 4-way 256KB 服务 processor 和 computing space 的数据请求 → 更频繁的 cache miss。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 MagiCache 通过三大设计解决 baseline 三个核心缺陷：(A) cacheline-level fused array architecture 替代 array-level 消除静态 coarse-grained 划分、(B) virtual engine + lazy initialization 实现近 100% cache utilization、(C) instruction chaining 实现异步执行降低 bursty access 延迟。
  MagiCache 全栈执行例子（同一 matmul, Chain-4 配置, k=4, Q=32, N=8, max VL=65536 bits, max occupancy=50%）：
  - **算法层**：同 baseline——RISC-V vector 矩阵乘法 ISA 代码不变，算法数值结果完全一致。
  - **系统框架层**：论文未明确说明（与 baseline 相同，直接 RISC-V vector ISA 编程）。
  - **编译框架层**：Liveness analysis（标准编译器 liveliness analysis [28]）预提取 vector register 生命周期，在适当时机插入 `vsetvli zero, zero` 指令释放 register。Release 指令 overhead <0.5%。论文未明确说明是否修改 LLVM pass 或通过外部工具预处理。
  - **kernel调度层→Virtual Engine 运行时 cacheline 级空间管理**：
    - **缺陷A解决（静态 coarse-grained 划分不适应不同应用）**：MagiCache 所有 8 个 fused arrays（每 way 1 array, 共 8 ways）均可通过 computing bit 在每行粒度动态切换 computing/storage 角色。Virtual engine 通过 Q/k 参数调节 computing/storage 比例——Chain-4 (k=4, max VL=65536, occ=50%) 给 compute-intensive 应用高并行度，Chain-1 (k=1, max VL=16384, occ=12.5%) 给 memory-intensive 应用更多 cache capacity。对比 baseline 固定 50% split，MagiCache 在运行时自适应——matmul 的 62.5% computing ratio 和 backprop 的 50% 均可通过选择不同 k 和 lazy initialization 实际使用 registers 实现。
    - **缺陷B解决（缓存空间低利用率 ~56%）**：Lazy initialization——仅在实际使用的 register（matmul 中 v0/v1）首次被指令访问时才分配 segment。VRMT entry valid bit=0 的 segment 不占用任何 cacheline。Liveness-based release——v0/v1 在 loop 末尾通过 `vsetvli zero, zero` 释放，computing lines 恢复为 cachelines。结果：cache utilization 从 Split-8 的 ~56% → Chain-4 的 ~97%（+41.2%）。L2 miss rate 降低 10%-40%（Fig. 10: add 降 36%, spmv 降 14%）。多应用 workload 下（Fig. 11），Chain-4 保持 ~90% cache space 用于缓存数据而 Split-8 仅 ~44% → 同时提高 vector 和 scalar 应用的性能。
    - **缺陷C解决（同步执行 bursty access 延迟高）**：Instruction chaining——运行时检测冲突后将无冲突指令（matmul 两条 load + mac + store，地址范围无交叉）组成 group。各 fused array 独立异步执行 group 内所有指令——Array 0 完成 segment load 后立即开始 compute，不等 Array 3 的 load miss。仅 group 边界同步。结果：synchronization stall time 减少 45.3%（Fig. 9），average memory access time 减少 2%-27%，MSHR usage 从 5.59→8.23（Table 7, 因更多 elements 被 pack + 异步 overlap 使 MSHR 利用率更高）。
    - **FFA 分配策略**：从随机位置 circular scan（每 cycle 32 cachelines, 最多 8 cycles），优先 free > 优先 available。对比 LRU/pseudo-LRU 免去额外硬件状态维护。设 minimum associativity threshold 防某 set 所有 cacheline 被分配为 computing lines。Miss rate 增加 <1%。
    - **Q/k 参数自适应**：SplitCache 固定 split ratio 对不同应用不匹配 → MagiCache 通过选择不同 k（1/2/4）使每 register 占用 1-4 rows/array, 最大 vector length 16384-65536。Compute-intensive 应用选大 k 提升 parallelism；memory-intensive 应用选小 k 保留更多 cache capacity。
  - **硬件架构层→Fused Array + Virtual Engine 硬件**：
    - **Fused Array 设计**：采用 bit-parallel data layout（与 cacheline 布局一致，无需 transpose——对比 baseline bit-hybrid 的 transpose overhead）。Peripheral circuits 5 layers（logic/add/shift/register/writeback）。256×256 sub-array 面积 overhead 17.7%，但因两 sub-array 共享 circuits → 256×256 fused array 面积 overhead 仅 8.9%。Bit-line computation 能耗比 read/write +54%，但仅占 total operations 17% → 整体功耗 +9%（免去 H-tree 能耗——传统 H-tree 占 >80% 总能耗 [4]）。Bit-line computation 延迟 1.6ns（+60% vs vanilla 1.0ns），但仍低于分别读两行的延迟。
    - **Virtual Engine 硬件**：总面积 26434 μm² @TSMC 28nm, 功耗 27.01mW @1GHz。VRMT（4.5KB SRAM）+ tag bits（2KB for computing/presence bits）→ 总存储 overhead 6.5KB。VRMT control logic ~1% 额外面积（相对 cache）。Micro-code ROM 8KB（1.6% area）。
    - **Cacheline conversion 机制**：通过 tag 上的 computing bit（1-bit）和 presence bit（1-bit/cacheline for coherence）实现。Conversion 四步：(1) evict dirty cacheline → (2) clear valid/writeable/dirty/presence bits → (3) set LRU bits invalid（replacement 不会选该行）→ (4) set computing bit。Release 反向：clear computing bit → set LRU bits=LRU → cacheline 恢复。
    - **Cache coherence**：presence bit + scalar access 时 snoop L1 + fence 指令。
  - **Baseline缺陷→MagiCache 方法映射总结**：
    1. **Array-level static split → 不同应用需求与固定配置不匹配**：MagiCache 的 cacheline-level dynamic partition + Q/k parameter tuning + lazy initialization 在运行时自适应用。
    2. **Computing arrays 中未用 register rows 浪费（30/32 wasted for matmul）**：Lazy initialization 仅分配实际使用的 register → cache utilization 从 ~56% 升至 ~97%。
    3. **Synchronous execution → sync stall 占总时间比例高**：Instruction chaining 实现 group 内异步执行 → sync stall -45.3%。
    4. **Bit-hybrid layout transpose overhead**：Bit-parallel layout（与 cacheline 一致）+ fused array 架构 → 无需 transpose。
  - **效果**：MagiCache Chain-4 vs SplitCache Split-8: 1.19x-1.61x speedup (geomean 1.39x), cache utilization 55.9%→97.1% (+41.2%), L2 miss rate 降低 10%-40%, memory access time 降低 2%-27%, 仅 6.5KB 额外存储。MagiCache 的 cacheline-level runtime space management 适用于各种 in-cache computing architecture（不同 data layout、peripheral circuits、programming framework）。

## 66-Chameleon- Adaptive Caching and Scheduling for Many-Adapter LLM Inference Environments..pdf

- baseline方法是什么？
  Baseline 是 S-LoRA [49]，一个 state-of-the-art multi-adapter LLM serving 平台。S-LoRA 的核心设计：(1) 所有 adapter weights 存储在 host memory（CPU），base model 驻留在 GPU memory；(2) 请求到达后，S-LoRA scheduler 使用 FIFO policy 组织 waiting queue；(3) 当 batch 中请求需要某个 adapter，S-LoRA 从 host 通过 PCIe 异步 prefetch adapter weights 进入 GPU memory；(4) 请求结束后 adapter 立即从 GPU memory 丢弃（deallocate）；(5) 采用 iteration-level continuous batching——每 decode iteration 动态注入/移除请求。Baseline 全栈执行例子（以 Llama-7B on 1×A40 48GB, Na=100 adapters, 9 RPS Splitwise trace 为例）：
  - **算法层（LoRA 推理）**：LoRA adapter = 低秩矩阵 B×A，推理时对 base model 每层的 weight matrix W，加 adapter contribution: y = Wx + BAx。Base model 在所有 task 间共享，adapter 则 per-task 不同。Adapter rank 范围 8–128，rank 128 adapter ~512MB，rank 8 adapter ~32MB。
  - **系统框架层（S-LoRA FIFO + 无缓存）**：请求按到达顺序进入单一 FIFO 队列。每 decode iteration，S-LoRA scheduler 从队列头部取出请求组成 batch，调用 MBGMM kernel 执行统一 batch 中的 base+adapter 推理。Adapter 需要时从 host CPU memory（377GB）通过 PCIe Gen4 ×16（~32GB/s 带宽）加载至 GPU memory，完成后释放。由于 adapter 无缓存且每次用完即丢，当 500 个 adapter 分布在请求中时，PCIe 带宽饱和（Section 3.2 Figure 4），P99 TTFT 达 8s 远超 SLO。
  - **编译框架层**：论文未明确说明（使用 PyTorch/HuggingFace Transformers + S-LoRA 的 MBGMM CUDA kernel，无编译框架修改）。
  - **kernel调度层**：S-LoRA 使用 MBGMM (Multi-size Batched Gather Matrix-Matrix Multiplication) kernel——将不同 rank adapter 的 BAx 计算统一到一个 batched gather GEMM kernel 中，减少单独 kernel launch overhead。但 kernel 执行受 PCIe adapter fetch 延迟制约——kernel 执行快但 adapter 在 PCIe 上等待时间长。
  - **硬件架构层**：1×NVIDIA A40 GPU (48GB memory)，PCIe Gen4 ×16 连接 CPU。GPU compute (37.7 TFLOPS FP16) 远超 PCIe bandwidth (32GB/s)，因此 adapter loading 成为瓶颈。GPU memory 46GB 空闲大部分时间未被利用。

  Baseline 三个核心缺陷：
  1. **忽略 workload heterogeneity**：FIFO 调度对异构请求（不同 input size、output size、adapter rank）产生 head-of-line blocking——短请求（小 input+output+小 rank adapter）被长请求（大 input+output+大 rank adapter）阻塞，P99 TTFT 恶化。
  2. **高 PCIe 带宽消耗**：adapter 每次用完即丢，500 个不同 adapter 场景下 PCIe 饱和，adapter loading 占 rank 128 TTFT 的 60%（含 load+compute overhead）。
  3. **Scheduling ineffective for tail latency**：FIFO 导致 short request 28.6% 时间在排队等待；SJF (μServe) 又导致 long request 5.15s 排队延迟和 starvation。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Chameleon 通过两大机制解决 baseline 缺陷：**Adapter Cache** 和 **Multi-Queue Adapter-Aware Scheduler**。Chameleon 全栈执行例子（同一 Llama-7B on 1×A40 48GB, Na=100 adapters, 9 RPS Splitwise trace）：
  - **算法层（同 baseline，LoRA 不变）**：推理算法与 baseline 完全相同（LoRA BAx），Chameleon 不改变 adapter 的计算方式。
  - **系统框架层（Adapter Cache + Multi-Queue Scheduler）**：
    - **缺陷1解决（workload heterogeneity → HoL blocking）**: Chameleon Scheduler 将请求按 WRS = 0.4×(InputSize/MaxInputSize) + 0.6×(OutputSize/MaxOutputSize)×(AdapterSize/MaxAdapterSize) 分为 small/medium/large 三队。每 iteration，Algorithm 1 两阶段调度——Phase 1 按各队 quota 从 small→large 选请求入 batch（保证短请求有 fast lane 不被长请求阻塞）；Phase 2 把未用完的 spare tokens 重新分配（让长请求不饥饿）。Queue 数量和 cutoffs 通过 K-Means clustering (K=1..4) 每 5min 动态调整。Queue quotas 用 M/M/1 队列理论 closed-form 公式 T_{ok}^{min} ≥ S×D×(1/SLO + λ) 计算以满足 SLO。Opportunistic Bypass 允许在头部请求内存不足时跳过执行后续请求（带 squash 保护防饥饿）。结果：所有请求类别的排队延迟降至 E2E 的 8% 以下（vs baseline FIFO 小请求 28.6%）。
    - **缺陷2解决（高 PCIe 带宽 → adapter 反复加载）**: Chameleon Cache 利用 GPU 48GB 中大部分时间空闲的内存（Section 3.2 Figure 6——~20GB 空闲）缓存 adapter weights。Cache 动态 resize——当 KV cache 增长则 shrink cache evict adapter，当请求结束则 expand cache 保留 adapter。复合淘汰 Score = 0.45×Frequency + 0.10×Recency + 0.45×Size 优先保护大 adapter（miss cost 高）和热门 adapter（访问频率高），同时保持 LRU 特性捕捉时间局部性（burst 访问）。Cache Manager 用 RC 保护 active request 的 adapter 不被淘汰。结果：75% 请求 cache hit（zero loading overhead），miss 的 25% 请求 loading 延迟从 baseline 30ms 降至 6ms。
    - **缺陷3解决（scheduling starvation）**: 两阶段调度 + 动态 quota 保证所有请求都不会饥饿。对比 SJF（长请求排队 5.15s）和 FIFO（小请求排队占 E2E 28.6%），Chameleon 将两者都降到 <8%。Dynamic reorganization (K-Means 重聚类) 保证调度适应 workload 变化。
  - **编译框架层**：论文未明确说明（与 baseline 相同，无编译框架修改）。
  - **kernel调度层**：Chameleon 复用 S-LoRA 的 MBGMM kernel 不做修改。但通过 adapter cache 将 adapter 预置在 GPU memory，消除了 kernel 执行前等待 PCIe adapter fetch 的延迟——kernel 调度路径从 "wait PCIE→load adapter→launch kernel" 变为 "adapter already cached→launch kernel directly"。Cache Manager 的 prefetch 机制进一步将 adapter loading 与 kernel 执行 overlap。
  - **硬件架构层**：同 baseline hardware (A40 GPU + PCIe Gen4)。Chameleon 不需要硬件或 CUDA kernel 修改。更多 GPU memory (24GB→48GB→80GB) 时 Chameleon 优势更大——更大 cache 空间 → 更高 hit rate → throughput improvement vs S-LoRA 从 1.4× (24GB) 升至 1.9× (80GB)。Multi-GPU TP 场景下 Chameleon 也有效——adapter 分片缓存于各 GPU，调度将多 GPU 视为单一执行引擎，TP4 High load 下 P99 TTFT 降低 95.8%。

- **整体效果**：Chameleon 在 9 RPS high load 下 P99 TTFT 降低 80.7%、P50 TTFT 降低 48.1%、throughput 提升 1.5× vs S-LoRA。Adapter cache 贡献 1.2× throughput improvement，scheduler 贡献 1.05×，两者协同达到 1.5×。能 sustain 12.9 RPS 不违反 SLO（baseline: 8.6 RPS）。跨 Llama-7B/13B/30B 模型、不同 trace (Splitwise/WildChat/LMSYS-Chat-1M)、不同 adapter 分布下性能一致。

## 67-ELORA- Efficient LoRA and KV Cache Management for Multi-LoRA LLM Serving.pdf

- baseline方法是什么？
  Baseline 是现有 Multi-LoRA serving 系统（vLLM [22,41]、S-LoRA [37]）的缓存管理方式：**静态 HBM 分区 + 独立的 LoRA/KV cache 管理**，不考虑 LoRA 与 KV cache 之间的 usage dependency。vLLM 在 HBM 中为 LoRAs 和 KV caches 分配固定大小的独立内存区域（默认 LoRA ratio=0.2），各区域使用独立的 LRU policy 管理 swap-in/out。S-LoRA 使用统一的 LoRA cache pool 但不保留历史 KV caches（query 结束后直接丢弃）。两者的共同缺陷：(A) 忽视 intra-LoRA usage dependency——LoRA 被 swap-out 后其对应 KV caches 仍在 HBM 中成为 "invalid caches"（无法被任何 query 使用），浪费 HBM 空间；(B) 静态 HBM 分区无法适应不同 LoRA 的动态 load 变化——当某时段 queries 使用更多不同 LoRAs 时 LoRA HBM 区域不够用，而当 prefix-heavy 时 KV HBM 区域不够用，导致频繁 swap-in/out；(C) LRU policy 未考虑 KV/LoRA 的 swap cost（node size）、访问频率和 LoRA 数量对 TTFT 的影响，随机性强的 swap-in/out 导致大量 cold-start。Baseline 全栈执行例子（以 Llama-7B on 1×NPU 64GB HBM，chatbot 场景，100 LoRAs，2 QPS 为例）：
  - **算法层**：LLM decoder-only transformer 推理，Multi-LoRA 通过 SGMV [6,37] 将多个不同 LoRA adapter（rank=32/64）的 query 打包为 single batch——base model weights W 一次 loaded 至 compute unit，各 query 通过各自 LoRA branch（W' = W + A_t B_t）并行推理。KV cache 计算：KV_Cache_q,t = W_{k,v} q + A_t B_t q，各 LoRA 的 KV cache 因 LoRA branch 修正而分别存储，不可跨 LoRA 共享。算法层论文方法与 baseline 相同（不改变 LoRA 计算方式）。
  - **系统框架层（vLLM baseline 缺陷 → FAST LIBRA 解决）**：
    - **缺陷A（intra-LoRA usage dependency 忽略）**：vLLM 静态 partition HBM——20% 空间给 LoRA 缓存（20GB × 0.2 = 4GB for LoRAs），80% 给 KV 缓存（含 running KV + history KV）。LoRA-1 的 2 个 KV blocks（KV1-1, KV1-2）在 KV HBM 区域被 LRU policy 保留，但 LoRA-1 被 LoRA 区域的 LRU swap-out 腾位给新 LoRA。此时 KV1-1、KV1-2 成为 invalid caches——Query-1 需要 LoRA-1 但 LoRA-1 不在 HBM，即使 prefix 命中这些 KV caches 也无法执行推理。vLLM 中高达 48.1% 的 KV caches 是 invalid 的。另外 LoRA-1 的 invalid KVs 还阻挡了 LoRA-2 的有效 KVs 被缓存——如 KV2-2 需要缓存但 HBM KV 区域已被 invalid KVs 占满。
    - **缺陷B（静态 HBM 分区无法适应动态 load）**：Translation benchmark 中 700s-1100s 时段 queries 使用 prefix-heavy KV reuse→HBM KV 区域 exhausted（利用率达 100%）但 LoRA HBM 区域仅 58.9% 利用率→KV swap-in/out 剧烈导致 TTFT 升至 5036.1ms。1100s-1800s 时段更多 LoRAs（从 41 个增至 75 个）被使用→LoRA HBM 区域 exhausted 但 KV 区域有余→LoRA cold-start 导致 TTFT 升至 8617.9ms。vLLM 无法动态调整两区域大小（redeployment 需数十秒阻塞推理），S-LoRA 虽有统一 LoRA cache 但不保留历史 KVs 故无法利用 prefix reuse。
    - **缺陷C（LRU 策略不考虑 swap cost 与频率）**：KV cache/LoRA 的 visited frequency、LRU time、swap cost（PCIe transfer cost = node_size / bandwidth）三者无明确相关性（Figure 5 散点随机分布）→LRU 单维度决定 swap-in/out 导致高 swap-cost 的高频 node 被错误 evict，或低 swap-cost 的低频 node 被错误保留→significant cold-start overhead。
  - **编译框架层**：论文未明确说明（与 baseline 相同，无编译框架修改）。
  - **kernel调度层**：SGMV kernel 不变（复用 vLLM/S-LoRA 的 CUDA kernel）。Baseline 的 kernel 调度瓶颈在于 kernel 执行前等待 LoRA adapter 从 main memory 通过 PCIe swap-in（cold-start latency = adapter_size / PCIe_bandwidth，如 100MB LoRA 经 PCIe Gen4 ×16 ≈ 32GB/s 需 ~3ms）和 KV cache swap-in。
  - **硬件架构层**：NPU × 4（256 TFLOPS FP16，64GB HBM per NPU），Arm CPU (192 cores)，256GB main memory，PCIe × 16 Gen4.0。HBM 总容量 64GB per NPU，其中 base model weights + running KV 占用约 30-40GB，余下的 ~24-34GB（取决于 model/context）为可用于 LoRA/KV caching 的空间。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 FAST LIBRA 通过 **Dependency-aware Cache Manager + Performance-driven Cache Swapper** 解决 baseline 三个核心缺陷。FAST LIBRA 全栈执行例子（同一 Llama-7B on 1×NPU 64GB HBM，chatbot 场景，100 LoRAs，2 QPS）：
  - **算法层**：同 baseline（LoRA W' = W + A_t B_t + SGMV batching 不变）。但 FAST LIBRA 通过 Cache Swapper 的 cost model formula (Eval_i = LoRA_Eval_i × Retain_Eval_i) 从算法角度量化了每个 LoRA/KV node 保留在 HBM 中对 TTFT 的预期收益。
  - **系统框架层（FAST LIBRA 对应解决）**：
    - **缺陷A解决（intra-LoRA usage dependency → Tree-based Dependency Manager）**：FAST LIBRA 用 Trie 树构建 usage dependency tree——virtual root→第二层 LoRA nodes（如 LoRA-1, LoRA-2）→各 LoRA 下的 KV cache subtrees（如 LoRA-1: {KV "How" → KV "are" → KV "you"}）。HBM 内所有 node 构成连通子图——一个 KV 只有在其所有 ancestors（含 LoRA node）都在 HBM 中才能被匹配（absence→invalid）。Swap-out 严格从叶子开始（保证上层依赖不被破坏），swap-in 严格从根开始（保证加载的 KV 立即可用）。结果：HBM 内 100% KV caches valid（eliminate vLLM 48.1% invalid KV）；FAST LIBRA-WOM（去除 dependency 维护）相比完整 FAST LIBRA TTFT 增加 1.27×、TPOT 增加 1.18×、peak throughput 降低 19.8%，证明 dependency 维护的关键性。
    - **缺陷B解决（静态分区 → Unified Caching Pool + Dynamic HBM Allocation）**：FAST LIBRA 将 HBM 和 main memory 统一划分为相同大小的 memory blocks，LoRAs 和 KVs block 大小对齐，二者可透明地共享全部 HBM 空间（不再有静态 2:8 分区）。Cache Swapper 每 100ms 评估 HBM 使用率→>95% 按 Eval_i 递增 swap-out 叶子 nodes→<70% 按 Eval_i 递减 swap-in 根 nodes→动态调整 LoRA 与 KV 的比例。结果：translation 场景 700s-1100s（KV-heavy）期间 HBM 更多空间自动分配给历史 KVs，1100s-1800s（LoRA-heavy）期间更多空间分配给 LoRAs——TTFT 对比 vLLM 静态分区降低 68.9%。HBM utilization 1.2×（vs vLLM）、2.6×（vs S-LoRA）。
    - **缺陷C解决（LRU → Unified Cost Model）**：Cost model Eval_i = LoRA_Eval_i × Retain_Eval_i = max(1, Low_lora/NowLoRA) × (cost_i × prob_i × (1 − sigmoid(t_i)))。三项联合：(1) transfer_cost_i 考量 node PCIe swap 代价→大 node evict 代价高应优先保留；(2) prob_i 考量访问频率→高频 node 保留；(3) (1 − sigmoid(t_i)) 时间衰减→最近访问的 node 权重更高。LoRA_Eval_i 额外考量 LoRA 数量——Low_lora = Σ[1 − (1 − prob_i)^BS] 预估近期 batch 需要的 LoRA 总数，NowLoRA 远低于 Low_lora 时 Eval_i 放大 LoRA node 的权重（reward >1），鼓励 prefetch 更多 LoRAs 避免 cold-start。Figure 9 实验证明 LoRA 数量不足时 TTFT 急剧上升（因 query 必须等待 LoRA swap-in 才能开始推理）。结果：FAST LIBRA-WOS（用 LRU 替代 cost model）TTFT 增加 1.24×、TPOT 增加 1.15×、peak throughput 降低 17.2%。FAST LIBRA-WOL（去除 LoRA_Eval）TTFT 增加 1.13×、TPOT 增加 1.11×、peak throughput 降低 13.1%。完整 cost model 综合三因素使 TTFT 最低。
  - **编译框架层**：论文未明确说明（与 baseline 相同，无编译框架修改）。
  - **kernel调度层**：同 baseline SGMV kernel。但 FAST LIBRA 通过 asynchronous swap-in/out（Torch Stream）将 LoRA/KV 的数据传输（PCIe Gen4 ×16）与 inference compute overlap——query A 等待 swap-in 时 query B 的 kernel 继续在 NPU 上执行（不阻塞）。此外，FAST LIBRA 通过 prefetch 策略（低 HBM 压力时主动 swap-in 高 Eval_i 的 root nodes）将 LoRA loading 提前到 kernel 执行之前，消除 kernel 启动前的 PCIe 等待。对比 baseline：vLLM/S-LoRA 在 batch 构造后逐个检查 LoRA miss→同步 PCIe load→kernel 阻塞。FAST LIBRA 将 kernel 调度路径从 "check miss→synchronous PCIE load→kernel launch" 转变为 "pre-fetched already in HBM→kernel launch directly" 或 "async PCIE load (overlapped with other queries' compute)→kernel launch"。
  - **硬件架构层**：同 baseline（NPU + PCIe Gen4 ×16）。FAST LIBRA 不依赖特定硬件架构——其 cache manager 和 swapper 逻辑对 GPU/NPU/TPU 等加速器通用（仅需 HBM + main memory 层级 + host-accelerator interconnect）。Abovethreshold 95%/70% HBM 配置适用于各类加速器。Overhead：trie 树匹配/更新平均 <0.5ms，HBM 监控 + swap decision <5ms，相对每个 query 数秒到数十秒的推理时间可忽略。

- **整体效果**：FAST LIBRA 对比 vLLM 和 S-LoRA 在 Chatbots/Translations/Personal Agents 三场景下 TTFT 分别平均降低 60.3% 和 50.1%，TPOT 降低 33.9% 和 28.6%，peak throughput 达 1.7× 和 1.6×。在 LoRA 数量 1000/2000 的大规模场景（Uniform/Distinct/Skewed 三种分布）下，FAST LIBRA 仍保持稳定低 TTFT，而 vLLM 和 S-LoRA 的 TTFT 在不同分布间波动剧烈（vLLM 静态分区的根本缺陷在极端分布下放大）。

## 68-SPARK_Scalable_and_Precision-Aware_Acceleration_of_Neural_Networks_via_Efficient_Encoding.pdf

- baseline方法是什么？
  Baseline是现有的compression-based encoding加速器方法：OLAccel（outlier-aware低精度计算，用coordinate list分别存储outlier和normal value，需额外存储空间和昂贵解码）、GOBO（coordinate list标记不同精度位置）、BiScaled DNN（相同bit-width不同scale factor，block sparse index格式）、ANT（混合INT/float/power-of-two数据类型，需finetuning）、Olive（outlier-victim pair，sacrifice adjacent normal value来存储outlier，tightly constrained on significant value location）。这些baseline的共性问题：需要复杂硬件逻辑区分不同精度值（额外index/coordinate list），memory access不对齐，解码开销大，硬件复杂度高。
  Baseline全栈执行例子（以OLAccel处理ResNet-50一层为例）：
  - **算法层**：权重离线4-bit/16-bit分层量化——多数值用INT4，outlier用INT16，以coordinate list记录outlier位置（需额外存储开销）。Activation在线量化后用coordinate list标记。
  - **系统框架层**：论文未明确说明（OLAccel为专用加速器架构，非通用Serving框架）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：PE array中专用outlier controller管理两种精度值的调度——normal values在4-bit PE计算（1 cycle），outliers在16-bit PE计算（多cycles）。Outlier controller需维护coordinate list索引，根据索引将对应位置的operand路由到不同PE。解码时需要lookup coordinate list找outlier位置，再提取对应精度值进行MAC，此过程引入额外延迟。
  - **硬件架构层**：OLAccel的core含4-bit PE + 16-bit PE（混合PE类型），outlier controller hardware（面积和功耗开销大），coordinate list buffer。数据格式不对齐——coordinate list的index和value分开存储，memory access pattern不连续。Olive的OVP encoding需要pruning adjacent parameter来腾出bit-width存储outlier，限制了可处理的outlier比例，且encoding/decoding过程复杂。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法SPARK通过**bit-level variable-length encoding**解决baseline的三个核心缺陷：(A) baseline用coordinate list/index分离outlier导致存储不对齐和额外解码开销，(B) baseline的编码方案固定了outlier位置/比例限制了灵活性，(C) baseline的decoder/encoder复杂导致面积和延迟开销大。
  SPARK全栈执行例子（同ResNet-50一层）：
  - **算法层→缺陷A/B解决（intra-value bit sparsity exploitation）**：
    SPARK不做value-level的outlier分离，而是在bit-level利用量化值高位sparsity。INT8量化后的值在[0,7]区间的MSB全是0，天然适合用4-bit表示。SPARK用1-bit identifier（c0）嵌入数据本身区分精度——c0=0表示低精度4-bit（值[0,7]），c0=1表示高精度8-bit（值[8,255]）。这样**无需额外coordinate list或index**，所有SPARK code以统一的4-bit对齐格式存储：低精度值占1个4-bit slot，高精度值占2个连续4-bit slots。内存访问完全对齐（aligned memory accesses），这是OLAccel/GOBO/BiScaled都做不到的（Table I）。Accuracy Compensation Mechanism通过b0⊕b3 XOR check决定舍入方向，最小化编码误差，95%+值无损编码（图4），**无需finetuning即可保持原精度**，这是ANT/BiScaled不具备的（它们需要finetuning或accept更大accuracy loss）。SPARK ~5.33 bit avg即可达到与ANT 6-bit相同甚至更好的准确率（Table IV），因为SPARK根据数据特性adaptive分配bit-width而非统一缩短。
  - **系统框架层**：论文未明确说明（SPARK为专用加速器架构设计，不涉及通用Serving框架修改）。SPARK的encoding/decoding机制与现有指令集兼容（IV-E），不需要引入新数据类型或修改load/store指令。
  - **编译框架层**：论文未明确说明（不涉及编译框架修改）。SPARK encoding在PyTorch端offline执行（weight encoding），activation encoding通过硬件encoder在线完成。
  - **kernel调度层→缺陷A/B解决（simplified mixed-precision execution）**：
    SPARK decoder极简——仅需MUX + OR + NOT gates实现（6.42μm²，28nm），无需OLAccel的coordinate list lookup。Decoder沿PE array边沿放置（m×n array仅需m+n个decoder），而非每个PE一个decoder。MPE默认全速INT4模式（1 cycle/op），仅当identifier c0=1时按需切换INT8模式（2-4 cycles）。Variable-speed dataflow通过stall插入维持systolic array同步——低精度值无需等待，高精度值自然消耗更多cycles。相比OLAccel需要outlier controller和coordinate list buffer管理两种PE类型，SPARK用单一MPE类型统一处理，硬件控制和数据调度大幅简化。同时，SPARK encoder（LZD-based）在activation后即时将高精度结果重新编码为SPARK格式，减少下一层的数据传输量。
  - **硬件架构层→缺陷C解决（minimal hardware overhead）**：
    SPARK decoder面积仅占core的0.251%（128个decoder共0.000822mm²），encoder仅占0.261%（64个encoder共0.000856mm²），PE array占99.49%。对比Olive需要4-bit decoder（60.29μm²）+ 8-bit decoder（80.18μm²）+ 4-bit PE（79.57μm²），SPARK的单类型轻量decoder（6.42μm²）面积小一个数量级。在iso-area约束下（Table VII：~0.327mm² core），SPARK可放置4096个4-bit PE，而OLAccel仅1152个混合PE、AdaFloat仅896个8-bit PE、Eyeriss仅168个16-bit PE。更多PE意味着更高并行度→SPARK在ResNet-50上性能领先BitFusion 80.1%、ViT上领先AdaFloat 3.3×。Energy方面，SPARK的INT4主导计算模式（~80%操作在INT4）加上窄bit-width数据传输（4-bit而不是16-bit），使DRAM和Global Buffer访问能耗大幅降低——ResNet-50总能耗比Eyeriss低74.7%。

## 69-Optimizing Dynamic-Shape Neural Networks on Accelerators via On-the-Fly Micro-Kernel Polymerization.pdf

- baseline方法是什么？
  baseline是现有tensor compiler在dynamic-shape场景下的编译优化方式，包含三类代表性方法：(1) **Vendor hand-crafted libraries**（cuBLAS/cuDNN/CANN）——为常见shape手工优化operator实现，但对任意dynamic shape性能不稳定（如A100上GEMM M=4096时262.2 TFLOPS vs M=105时仅22.3 TFLOPS，即使两者都是compute-bound）。这是因为library routine的专用实现难以适配所有shape。(2) **Static-shape tensor compilers**（TVM/TensorFlow XLA/TC）——通过auto-scheduling在编译时搜索optimal tiling结构，但必须编译时已知shape。用于dynamic场景时需为每个runtime shape在线编译（TVM约0.33 CPU hours per shape），完全不可行。(3) **Dynamic-shape tensor compilers**（DietCode/Nimble）——通过shape-generic search space在编译时预生成一组tensor program覆盖预定义的shape range，runtime时按实际shape选择。缺陷：DietCode要求编译时提供每个dynamic dimension的范围（如M∈[1,4096]），超出范围的shape产生invalid runs（out-of-bounds或resource不足）。且即使在范围内，性能仍suboptimal（因search space仅为"shape-generic"而非per-shape最优）。

  全栈执行例子（baseline DietCode处理dynamic-shape GEMM）：
  - **算法pipeline**：BERT模型定义→sequence length τ∈[1,4096]→GEMM shape (M=τ, N=3072, K=768) 在编译时以range [1,4096]描述。DietCode在编译时用shape-generic auto-scheduler生成覆盖[1,4096]范围的多个tensor programs。
  - **系统框架(Serving)**：PyTorch/TurboTransformers加载模型→runtime收到具体sequence length（如τ=8192超出预定义范围）→DietCode lookup失败→返回invalid run（error或incorrect result）。
  - **编译框架**：DietCode接收shape range描述→shape-generic auto-scheduler在tiled program template上搜索→输出一组shared tiled programs（每个覆盖一定sub-range）。以tile sizes TM.0,TM.1,…,TK.2为变量在缩减的搜索空间中搜（vs static-shape compiler的固定shape全搜索）。
  - **kernel调度**：每个tiled program编译为固定的CUDA kernel→runtime按实际shape选一个kernel launch→GPU SM执行。tiling size在编译时确定，对runtime shape suboptimal（如一个kernel cover M∈[1024,2048]，对M=1025和M=2047使用相同的tile配置）。
  - **硬件架构**：Nvidia A100 GPU→108 SMs→L1/shared memory→HBM。tile配置不当可能导致load imbalance（某些SM idle）或memory bandwidth利用不足。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MikPoly提出two-stage micro-kernel polymerization：(1) **缺陷→解决**：Baseline的shape-generic搜素空间针对预定义range，无法为每个具体shape产生最优tensor program→MikPoly的online polymerization在runtime时根据实际shape动态组合offline生成的高度优化的fixed-size micro-kernels，实现per-shape级别的优化。(2) **缺陷→解决**：Baseline（DietCode）要求预定义shape range，超出range则失败→MikPoly通过micro-kernel polymerization天然支持任意shape（无range限制），因为runtime收到任何shape后，只需选择合适的micro-kernels组合+polymerization patterns即可生成有效的tensor program（MikPoly的8192个测试case中0 invalid run，DietCode在M range [1,128]时仅254/8192 valid）。(3) **缺陷→解决**：Baseline（static-shape compiler）编译开销过高（~0.33h）→MikPoly offline仅生成少量micro-kernels（~40个），online搜索时间仅~2μs（通过cost model启发式剪枝和轻量scalar assignment instantiation）。

  全栈执行例子（MikPoly处理dynamic-shape GEMM）：
  - **算法pipeline**：BERT模型定义的GEMM operator→MikPoly不要求预知shape。编译时仅从模型计算图提取operator类型（GEMM）和target device H。
  - **系统框架(Serving)**：TurboTransformers加载BERT模型→MikPoly在首次遇到某个operator时触发online polymerization（后续同operator同shape可复用）。
  - **编译框架**：（两阶段）
    Offline阶段：从two-stage template Q的offline loops提取micro-kernel template K_hat→TVM auto-scheduler生成40个fixed-size micro-kernels（每个专为M_local=shared memory优化，如(uM,uN,uK)=(256,128,32)在单SM上最优化pipelined GEMM）→在单PE上学习performance model g_predict（t从1到5120的分段线性函数）。
    Online阶段：runtime shape (M,N,K)=(4096,1024,4096)已知→MikPoly用9种polymerization patterns（基于7-block output skeleton的分块策略）重组Q_online→遍历polymerization strategies（替换parameterized micro-kernel为offline micro-kernels）→cost model用g_predict预测每个(R_i,K~_i)的f_pipe（单PE pipelined task开销）和f_wave（并行wave数）→Σ f_wave×f_pipe得到总cost→启发式剪枝→选最优(regions,micro-kernels)。
  - **kernel调度**：选中的micro-kernels（如GEMM-AB: A处理(3072,1024,4096)、B处理(1024,1024,4096)）通过scalar assignment设置地址偏移和迭代次数→launch CUDA kernel。A=(uM=256,uN=128,uK=32, 8 warps/thread block)→96 pipelined tasks→768 warps；B=(uM=64,uN=64,uK=64, 4 warps/thread block)→256 pipelined tasks→1024 warps。组合后共3 waves，最后一wave仅含B tasks的小部分，sm_efficiency=96.06%（vs single-micro-kernel的58.90%）。
  - **硬件架构**：Nvidia A100→108 SMs→每个SM上micro-kernel执行pipelined GEMM：DMA从HBM加载tile到shared memory→Tensor Core FP16 matrix multiply-accumulate→结果累加在register→最终写回HBM。Multi-wave调度由GPU hardware scheduler自动完成。Cost model的|P_multi|=108用于预测并行度。

## 6-MCBP- A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness.pdf

- baseline方法是什么？
  Baseline是value-level INT8 LLM推理：在prefill阶段执行dense INT8 GEMM，在decoding阶段执行value-level top-k attention sparsity预测（4-bit MSB估计attention→sort选top-k→full-precision计算被选KV）。Baseline全栈执行例子（以Llama7B, Dolly S=4k prompt+48 decode为例）：
  - **算法pipeline**：INT8 post-training quantization (per-channel symmetric weight + per-tensor asymmetric activation, SmoothQuant式)，GEMM用dense INT8矩阵乘法，attention用4-bit MSB value-level top-k预测——加载每个Key的完整4-bit MSB→计算estimated attention→sort选top-k indices→仅对这些KV进行8-bit full attention计算。
  - **系统框架(Serving)**：TensorRT-LLM on A100。模型权重以INT8 value-level格式存储（值级layout：multi-bit activation连续存储）。解码阶段每迭代加载全部权重和全部KV cache条目，无需bit-level decomposition。
  - **编译框架**：论文未明确说明（使用TensorRT-LLM内置INT8 GEMM kernel，无bit-level编译优化）。
  - **kernel调度**：A100 Tensor Cores执行dense INT8 GEMM（每周期一个INT8 MAC）。Value-level top-k prediction：加载4-bit Keys→INT4 MSB矩阵乘法→sort top-k→load被选Keys的full INT8→INT8 attention计算。Value-level Huffman coding对weight压缩但需bit-reorder（将value-format weight重组为bit-slice format）才能输入bit-serial PE，产生额外on-chip overhead。
  - **硬件架构**：NVIDIA A100 GPU (624 TOPS@INT8) + HBM2e 80GB。Weights以value-level contiguous format在HBM中，每次GEMM加载所有bits（含大量零bit的LSB也在传输）。Top-k prediction中的4-bit MSB乘法也加载不需要的LSB bit。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MCBP将LLM推理优化从value-level下沉到bit-level，通过三项协同设计解决baseline的三重瓶颈：
  - **(1) BRCR解决GEMM Computation瓶颈**：Baseline的dense INT8 GEMM无法利用bit-slice向量间的冗余——MCBP发现将weight分解为bit-slice matrix后，列向量之间的repetitiveness在group-by-m=4后显著增加（pigeonhole principle，至多2^4=16种类型 vs H~4096列）。BRCR通过CAM-based merge消除重复计算：对每个Group Matrix识别重复列→合并对应activation→只计算2^m个unique MAV元素→reconstruct结果。理论计算量由kH^2降至kH×(H/m×(1-bs̃)+m×2^{m-1})，相比value sparsity最多减少12.1×。
  - **(2) BSTC解决Weight Access瓶颈**：Baseline的value-level压缩（如Huffman coding）仅利用value sparsity（~6.3%，Llama13B），且value-format与bit-serial PE不兼容需bit-reorder。MCBP利用bit sparsity平均是value sparsity的10.1×（因weight呈Gaussian分布，高bit位高度稀疏），对3rd-7th BS矩阵（SR>65%）进行two-state coding，编码与BRCR共享group粒度m=4，消除bit-reorder开销。weight memory access减少平均75.8%。
  - **(3) BGPP解决KV Cache Access瓶颈**：Baseline的value-level 4-bit MSB top-k prediction仍加载所有4-bit并执行full 4-bit计算，但MCBP发现仅前2-bit已足以判断大部分KV是否属top-k。BGPP采用progressive bit-level early termination：每轮只加载当前bit位（1 bit），计算partial QK估计→threshold过滤→仅remaining candidates进入下一轮bit加载。平均减少KV cache access 50%。
  - 全栈执行例子（Llama7B, Dolly S=4k, 48 decode, batch 8）：
    - **算法pipeline**：Offline——BSTC编码（Llama7B weights分解为8个BS matrix→3rd-7th BS按m=4 two-state coding压缩→1/2/8th BS无压缩）。Online prefill——BRCR执行bit-slice GEMM：8个BS matrix分m=4行组→CAM match 16种search keys→merge activation→reconstruct→shift-accumulate→scale+bias→INT8 output。Online decode——BGPP bit-grained progressive prediction：Round 1加载所有Keys的1st bit(MSB)→bit-serial IP unit计算partial Â₁→threshold=max(Â₁)-α₁×radius→仅indices[mask=1]的Keys进入Round 2加载2nd bit→...→早停后仅选vital KVs做full-precision attention。
    - **系统框架(Serving)**：论文未明确说明Serving框架修改（MCBP为自研accelerator，不基于开源Serving框架）。Custom 8-step pipeline controller管理data fetch/decode/compute/writeback全过程。
    - **编译框架**：论文未明确说明（offline BSTC encoding + online fixed dataflow，无动态编译）。
    - **kernel调度**：MCBP Accelerator上——Data Fetcher从HBM加载BSTC-compressed BS weight到Weight SRAM (768KB)→BSTC Decoder解压为decompressed BS matrix→CAM-based BRCR Unit执行CAM match (512B CAM, 1 cycle/search)→Index Converter翻译为activation SRAM地址→AMU merge重复activation→RU fixed datapath reconstruction→Quantizer施加scale+bias→结果写回HBM。BGPP并发运行：bit-serial IP unit逐bit计算partial QK→Progressive Filter clipping→最终vital KV indices送BRCR做attention计算。Bit shift overhead仅17%，但3× latency reduction覆盖之。
    - **硬件架构**：MCBP ASIC TSMC 28nm 1GHz 9.52mm² 2.395W。CAM hardware (面积占BRCR 25%+power 47%) 但net减少45%面积和72%功耗，因BRCR用bit-level redundancy大幅减少compute logic。BSTC轻量CODEC仅占6.2%面积和10% core power。BGPP占4.5%面积。HBM2 8ch 512-bit/cycle。Energy 22740 GOPS/W = 31.1×A100, 35×Spatten, 5.2×FACT, 3.2×SOFA。

## 71-Pre-gated_MoE_An_Algorithm-System_Co-Design_for_Fast_and_Scalable_Mixture-of-Expert_Inference.pdf

- baseline方法是什么？
  Baseline 是三种 MoE inference system 方案：
  - **GPU-only**：所有模型参数（含全部 experts）存于 GPU HBM，推理无 PCIe 传输，性能最优。痛点：内存需求巨大（Switch-Large 128 experts 需 105.6GB），单 A100 80GB OOM，需多 GPU expert parallelism。多 GPU 下因 expert sparse activation（如 top-1 of 128 仅激活 0.8% experts），大部分 GPU memory 中 expert 参数闲置，GPU compute utilization 低，TCO 高。
  - **MoE-OnDemand（HuggingFace Accelerate 风格）**：Expert 参数 offload 到 CPU DDR4，runtime 经 gate 选 expert 后按需从 CPU→GPU 迁移被激活 experts。痛点：expert selection 与 expert execution 存在串行数据依赖（必须先 gate 选出 experts 才能开始迁移，必须迁移完毕才能执行），PCIe 传输延迟直接暴露在端到端推理延迟中。
  - **MoE-Prefetch（SE-MoE 风格）**：Expert 参数 offload CPU，在当前 MoE block 执行时预取下一 block 的全部 experts 到 GPU。痛点：(a) 需要迁移全部 experts（如 128 experts），PCIe 传输量大，不可扩展；(b) GPU 需同时存当前 block 和下一 block 的全部 experts（双份），GPU memory 压力大；(c) 大量未激活 experts 被浪费性传输。
  
  Baseline 全栈执行例子（MoE-OnDemand, Switch-Base 128 experts, top-1）：
  - **算法pipeline**：传统 MoE gate function——第 N 个 block 的 gate(h_N) → softmax → top-k → 选中 expert indices，然后对选中 experts 执行 FFN。Gate 和 FFN 同一 block 内串行依赖。
  - **系统框架(Serving)**：FasterTransformer 上，每一 MoE block：gate forward (GPU, ~μs) → CPU→GPU cudaMemcpyAsync 迁移选中 expert 参数 (~4ms for 120MB, PCIe 32GB/s) → expert FFN kernel 执行 (~3ms) → 释放 expert buffer。串行三阶段不可重叠。MoE-Prefetch 改进为：当前 block FFN 执行时异步迁移下一 block 全部 128 experts (~480ms) → 传输本身成为瓶颈。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：FasterTransformer CUDA kernels——gemm for expert FFN (W1, W2)，门控 softmax + topk selection。
  - **硬件架构**：单 NVIDIA A100 80GB + AMD EPYC CPU DDR4 1.8TB + PCIe Gen4 32GB/s。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Pre-gated MoE 通过 Algorithm-System Co-Design 同时解决三大 baseline 痛点：

  **痛点 (1)：MoE 大内存 → GPU 存不下/多 GPU TCO 高**
  → **解决**：Pre-gated MoE 继承 MoE-OnDemand 的 CPU offload 策略——所有 expert 参数存 CPU DDR4，仅非 MoE dense 参数常驻 GPU。但关键区别在于 Pre-gated MoE 的 pre-gate 使 offline→online 迁移不再是 bottleneck（见痛点3的解决）。

  **痛点 (2)：Expert 动态稀疏激活 → 多 GPU 下 compute utilization 低**
  → **解决**：单 GPU 即可部署（CPU offload 缓解 Memory 压力），无需多 GPU expert parallelism，消除 inter-GPU load imbalance 和 idle GPU 问题。

  **痛点 (3)：Gate→Execute 串行依赖 → MoE-OnDemand PCIe 延迟暴露，MoE-Prefetch 传输量过大**
  → **核心创新——Pre-gate Function**：将 gate function 的角色从"为当前 block 选 experts"改为"为下一个 block 选 experts"。第 N 个 block 的 pre-gate 在第 N 个 block 开始时就确定第 (N+1) 个 block 的 experts——此时可立即启动 PCIe 异步传输第 (N+1) 个 block 的 experts，而与第 N 个 block 的 expert FFN 执行完全并发。这实现了：
  - 仅迁移 activated experts（vs MoE-Prefetch 迁移全部 128 experts），PCIe 传输量缩小 128×（top-1 case）。
  - PCIe 传输完全隐藏在 expert FFN compute 内（green compute + blue comm 重叠），延迟开销仅 19% vs GPU-only。
  - GPU 双缓冲仅需存 2×(top-k) experts 参数，GPU memory 仅 23% of GPU-only（接近 memory-optimal MoE-OnDemand 的 23%）。
  
  论文方法全栈执行例子（Pre-gated MoE, Switch-Base 128 experts, top-1）：
  - **算法pipeline**：Pre-gate 数学上为 G_N^pre: h_N → p_{N+1} ∈ R^E。Block 0 使用两个 gate：first_gate(h_0) 选 block 0 experts，pre_gate_0(h_0) 选 block 1 experts。Block 1..N-1 仅用 pre-gate（当前 block experts 已由上一 block pre-gate 决定）。Block N（最后一个）无 pre-gate。Pre-gate 为轻量 MLP，计算量 <1% MoE block FLOPs。Fine-tune 时利用 pretrained 权重，训练成本等同传统 fine-tune。
  - **系统框架(Serving)**：FasterTransformer 上修改——(a) gate 输出目标从 intra-block 改为 inter-block；(b) 双 CUDA stream 流水线：Stream A = expert FFN compute (当前 block)，Stream B = cudaMemcpyAsync CPU→GPU (下一 block activated experts)。从 Block 1 起两 stream 并发。Peak GPU memory = Non_MoE_M + Σ_{L=N}^{N+1} Act_Exp_L（见论文 Eq.1），对比 GPU-only 的 4.2× reduction。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：FasterTransformer CUDA kernels 未修改——仍使用现有 highly-optimized gemm for expert FFN。仅调度层面加入 expert prefetch 逻辑和双缓冲内存管理。
  - **硬件架构**：单 A100 80GB + EPYC DDR4 1.8TB + PCIe Gen4 32GB/s。单 GPU 部署 26.4B Switch-Large 模型（GPU-only OOM），吞吐 42 tokens/sec，1.6× MoE-OnDemand 和 52× MoE-Prefetch。Model accuracy: SQuAD F1 90.2（Pre-gated）vs 90.1（conventional MoE），无退化。

## 73-AdapTiV_Sign-Similarity_Based_Image-Adaptive_Token_Merging_for_Vision_Transformer_Acceleration.pdf

- baseline方法是什么？
  Baseline是现有GPU/CPU上运行的ViT模型结合Token Merging (ToMe[12])，以及未优化的Vanilla ViT：
  - **ToMe [12]**: 将tokens随机partition为两组，做brute-force bipartite matching找最相似token对（O(N²) cosine similarity计算），按固定merge rate(r)每层合并r×N个token。Cluster aggregation用average merge或prune merge。痛点：(1) TM overhead巨大——TM仅占0.03%总ops却占36.8% latency（图2b），因GPU上inefficient vector-wise/element-wise操作（cosine similarity, argsort, dynamic tensor cropping）；(2) Fixed merge rate——每层固定合并比例，无视per-image token similarity差异，miss掉大量合并机会（图2c：significant intra-image token similarities未被fixed MR利用）。
  - **Vanilla ViT**（无TM）: 所有N=196 tokens参与全部12层计算，Self-Attention O(N²)复杂度。痛点：图像中大量同质区域（天空、墙壁等）的相邻token高度相似→冗余计算和存储。
  Baseline全栈执行例子（以ToMe + ViT-base-patch16-224 on Nvidia Jetson Orin Nano, DDR4 2400MHz 76.8GB/s为例）：
  - **算法pipeline**: Random bipartition→brute-force cosine similarity search O(N²)→argsort→fixed MR merge（如r₁=0.1,...,r₁₂=0.8递增）。所有操作在GPU上以FP16 tensor执行。
  - **系统框架(Serving)**: 论文未明确说明（单图推理，非多请求Serving场景）。
  - **编译框架**: 论文未明确说明（使用timm/PyTorch原生GPU执行）。
  - **kernel调度**: CUDA kernels执行cosine similarity matmul（d个n-bit multipliers）+ argsort + dynamic tensor cropping（裁掉pruned tokens后的非方形tensor处理）。TM操作为vector-wise/element-wise操作（对GPU不友好→kernel launch开销大，memory带宽利用率低）。LN和TM为串行执行（无法重叠）。
  - **硬件架构**: Nvidia Jetson Orin Nano (edge GPU, Ampere架构), DDR4 2400MHz 76.8GB/s。TM的频繁kernel launch + dynamic tensor crop导致GPU硬件利用不充分。Orin Nano的CUDA core执行dense matmul高效但vector-wise/irregular操作效率低。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  AdapTiV通过**算法-硬件协同设计**解决两个核心baseline缺陷：
  **(A) TM overhead过大 (36.8% latency)** → 三大算法优化降低TM overhead：(1) LMatch将search space从O(N²)降至O(N)（仅与left/above相邻token比较），effective TMatch从9.6%升至36%；(2) Sign Similarity用d个1-bit XNOR替代d个n-bit multipliers计算token相似性（与cosine similarity correlation=0.95）；(3) Sign-Driven Scheduling将TM完全嵌入LN流程→TM latency overhead降为零。硬件的AdapTME专用模块（SSCU XNOR lines+SP+SPMU+TIM）以仅1.49%面积和1%功耗实现这些算法。
  **(B) Fixed MR miss合并机会** → Dynamic MR策略：按top-left→bottom-right顺序迭代扫描所有effective tokens，每个effective token执行LMatch+Sign similarity→相似则立即聚类合并。merge rate完全由图像内容决定（不预设），累计扩展cluster。同一模型同层不同图像merge rate可0%–96.5%，最大化利用intra-image similarity。

  AdapTiV全栈执行例子（AdapTiV ASIC, ViT-base-patch16-224, ImageNet-1K, 1GHz Samsung 28nm）：
  - **算法pipeline→Image-Adaptive TM**: 与baseline的关键区别——不随机bipartition，而是LMatch只搜索left/above neighbors（利用空间局部性先验）；不用cosine similarity而用Sign similarity（1-bit XNOR）；不用fixed MR而是全图扫描所有effective token→merge rate完全图像自适应。准确率损失<1%（ImageNet-1K），无额外training。
  - **系统框架(Serving)**: 论文未明确说明。AdapTiV-Lite可作co-processor与edge CPU/GPU配对使用。
  - **编译框架**: 论文未明确说明。定制的硬件数据通路替代了编译框架。
  - **kernel调度/硬件架构→Sign-Driven Scheduling + AdapTME**: 核心创新是将TM硬件完全嵌入LN的dataflow。VPU执行LN的x_i - μ_i操作→sign bits即时产生→AdapTME的SSCU用2×64 XNOR lines并行计算sim_left和sim_abv→PopCount+threshold compare→O_SC判定。若相似→early stop LN（跳过σ和γ×(x-μ)/σ+β的计算），当前token被merge。Baseline的ToMe在GPU上是与LN**串行**的（先完成全部LN，再执行TM），AdapTiV通过专用硬件（SP 1KB存储comparison token sign bits + SPMU bitmap管理semantic→physical address映射 + TIM per-token cluster状态跟踪 + SSCU streaming计算）将TM**化整为零**分散到每个token的LN过程中。
    具体执行流程（对比baseline）：(1) Baseline GPU: LN全部N tokens串行→TM全部N tokens O(N²)串行→QKV GEMM→Attention→FFN。TM为独立long-latency步骤不可隐藏。(2) AdapTiV ASIC: VPU-LN + AdapTME-TM并行——每个token先x_i-μ_i→sign bits→AdapTME simcheck（如果相似→early stop→prune→减少后续计算；不相似→LN完成→sign存入SP作为comparison token）→处理完所有tokens时TM也已完成且latency完全被LN掩盖→compact N' effective tokens → PE Array GEMM→Attention→FFN。TM overhead由baseline的36.8% latency降至zero（完全与LN并行）。
    硬件层面与baseline GPU的本质差异：
    - PE Array (16×64 MAC) 专为动态token数的GEMM/GEMV设计→无论TM后剩余多少token都保持高利用率（vs GPU需要crop tensor→irregular shape→低效）。
    - SSCU的XNOR lines替代GPU的FP16 multipliers→Sign similarity计算在SSCU内以64-bit粒度完成，每cycle产生一个O_SC bit→延迟d/64 cycles for d=768即12 cycles（vs GPU cosine similarity需要完整matmul→慢数百倍）。
    - SP仅1KB存储comparison token sign bits避免DRAM access（baseline GPU每次TM需要额外DRAM read 2×N tokens ≈ 2×196×768/8 ≈ 38KB→over DRAM bus）。结合Sign-Driven Scheduling将DRAM access嵌入与LN operand fetch相同的数据流。
    - SPMU bitmap (W×(W+1)-bit = 14×15 = 210-bit) 管理semantic→physical映射，单cycle查表无寻址延迟（vs GPU scatter/gather kernel）。
  - **效果**: vs edge GPU (Jetson Orin Nano): 平均18.4× speedup, 21.5× energy efficiency；vs server GPU (RTX 6000 ADA): 6.3× speedup, 11.2× energy efficiency。面积仅2.49mm² @28nm，AdapTME仅占1.49%面积和1%功耗→证明了专用硬件以极小成本实现token merging的全潜伏期隐藏是完全可行的。

## 74-SOFA_A_Compute-Memory_Optimized_Sparsity_Accelerator_via_Cross-Stage_Coordinated_Tiling.pdf

- baseline方法是什么？
  现有dynamic sparsity (DS) acceleration用于加速Transformer attention推理。典型流程为三阶段：pre-compute（低精度预测attention matrix Â）→ top-k sorting（选取每行最大的k个vital Q-K pairs生成mask）→ formal computing（基于mask以高精度计算sparse attention）。
  
  全栈执行例子（以Llama-7B, S=2048, T=128 token parallel为例）：
  - **算法层**：DS使用4-bit低精度矩阵乘法（取MSB half）预测Â，vanilla全行排序选取top-k，FlashAttention-2做tiled softmax。FA-2虽通过tiling减少off-chip memory access，但每tile需重新计算并比较global MAX，引入大量额外Exp和比较操作。当S=2048时FA-2比vanilla多9×10^6次Exp和3×10^5次比较。
  - **系统框架层**：论文未明确说明。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：GPU上使用PyTorch/Huggingface实现。GPU的vector engine难以高效执行top-k sorting的logical branching和fine-grained control。DS三阶段在GPU上串行执行——pre-compute→store Â to DRAM→load Â row-wise→top-k→store mask→load K/V→formal compute，中间结果必须经DRAM。
  - **硬件架构层**：现有DS accelerator（A3/ELSA/Sanger/DOTA/Energon/SpAtten/FACT）仅关注单一阶段优化（如降低pre-compute开销或加速sorting），忽视跨阶段协同。全行处理（whole-row-processing）要求top-k必须等待整行Â就绪，导致：(1)中间数据量激增（T=512, S=2048需5MB SRAM, 5.47mm² @ TSMC 28nm → >7× SpAtten/ELSA面积）；(2)大量off-chip DRAM access（T=128时MAT ratio达72%）；(3)三阶段严格串行 → 端到端latency累积。FlashAttention-2的tiling虽然减少内存访问但大幅增加计算开销。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  SOFA通过cross-stage coordinated tiling实现算法-硬件协同设计，将DS三阶段分解为fine-grained sub-stage tiles，实现跨阶段流水线化，同时利用前阶段信息减少后阶段计算。

  缺陷1（pre-compute计算开销大）→ **DLZS**: 将乘法替换为log域shift+add。仅将单operand通过LZE转对数域（differential），预转换Wk为LZ格式存储（4-bit）。Key prediction: 8-bit token × 4-bit LZ weight → shift-sum → 8-bit K̂。Attention prediction: 16-bit Q的LZ (5-bit) + K̂ → shift-sum → Â。比vanilla leading zero方案减少一半converter开销、一半误差、更少内存访问。DLZS engine: 128×32 systolic shift array, configurable 8/16-bit LZE。

  缺陷2（top-k row-dependency导致大量DRAM access和长latency）→ **SADS**: 利用DCE数据分布特性，将一行S长度的Â分为n=4个sub-segments独立tiled sorting，每段取top-(k/4)。用sphere search引入search radius r约束搜索范围，降低比较器开销。16-to-4 bitonic sort + adaptive clipping消除冗余比较（3rd到k-th不保留精确序）。

  缺陷3（FA-2计算开销surging，且未利用top-k信息）→ **SU-FA**: 利用top-k stage提供的sorting信息，采用descending order更新FA——从MAX index开始依次desend更新到k-th value，使li更新仅需1 Exp + 1 Add（vs ascend的1 Exp + 1 Mul + 1 Add + vanilla FA需频繁refresh MAX）。AP module以folded设计双模支持MAX ensure和tile同步，避免过量area。

  全栈执行例子（对比baseline）：
  - **算法层**：DLZS (log-domain, multiplier-free) → SADS (sub-segment distributed tiled sorting with DCE) → SU-FA (descending sorted-updating FA leveraging top-k info)。总计算复杂度降低28%。Attention+QKV计算减少56.8%-67.4%（0%-2%准确率损失）。
  - **系统框架层**：Pre-deployment Preparation (offline)：DSE via Bayesian optimization搜索每层最优Bc和top-k→fine-tuning恢复精度。User Inference (online)：加载模型后实时动态稀疏推理。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：RASS (Reuse-Aware Schedule Scheme)：greedy搜索跨query共享的KV pair优先调度→独占KV pair随后→pack到同一phase→out-of-order KV执行，减少33% memory access。FSM controller + bitmask-indexed ID Buffer实现。
  - **硬件架构层**：SOFA accelerator (TSMC 28nm, 1GHz, 5.69 mm², 0.95W core)。Cross-stage tiled pipeline dataflow替代Baseline的串行三阶段：128-token parallel处理中，DLZS prediction → SADS sorting → KV generation on-demand → SU-FA computation在一个fine-grained pipeline中执行，中间结果留在SRAM无需写回DRAM。相比Baseline的「全行处理→DRAM存储→重新加载→下阶段」，SOFA消除79% memory access。整体：9.5× speedup and 71.5× energy efficiency over A100 GPU；15.8× energy efficiency, 10.3× area efficiency, 9.3× speedup over 8 SOTA accelerators。


## 77-Make_LLM_Inference_Affordable_to_Everyone_Augmenting_GPU_Memory_with_NDP-DIMM.pdf

- baseline方法是什么？
  Baseline是现有offloading-based LLM推理系统（Huggingface Accelerate, FlexGen, Deja Vu），其典型全栈执行模式如下：
  - **算法pipeline**: 标准dense attention + FC layers，所有参数参与计算。Deja Vu引入activation sparsity预测，但使用昂贵per-layer MLP-based predictor（LLaMA-7B需2GB额外存储，增加10%-25%推理时间）。
  - **系统框架层**: FlexGen使用zig-zag offloading——将多个token打包为block，逐层计算时prefetch下一层weight以overlap PCIe传输，适合prefill阶段的大batch。但token generation阶段batch size小（本地部署场景bs=1-4），weight loading无法有效overlap。Deja Vu预测activated neuron后仍需从host memory通过PCIe加载cold neurons，导致PCIe传输占89%推理时间。
  - **编译框架层**: 论文未明确说明。
  - **kernel调度层**: Huggingface Accelerate的naïve offloading——按层整体在GPU和CPU之间搬移。所有计算在GPU执行，CPU仅作为weight的存储仓库，无计算参与。GPU kernel（GEMM/GEMV）的input activation来自GPU local memory，weight可能来自PCIe传输（>15× bandwidth gap vs GDDR6）。
  - **硬件架构层**: 标准consumer-grade GPU（如RTX 4090 24GB）+ DIMM-based DDR4 host memory via PCIe 4.0。Host DIMM无计算能力，仅提供存储扩展。GPU内存带宽936 GB/s，PCIe 4.0仅64 GB/s（~14.6× gap），导致约99%推理时间花在PCIe数据传输上。
  - Baseline全栈执行例子（以FlexGen + OPT-66B, 单RTX 4090, bs=1, token generation为例）：
    1. Host CPU scheduler决定当前层需要计算的token（zig-zag block调度）→ 触发weight prefetch from host DDR4 via PCIe 4.0 (64GB/s) to GPU GDDR6 (936GB/s) → 数据搬运瓶颈。
    2. GPU从GDDR6读取weight和activation → Tensor Cores执行GEMM/GEMV → 输出写回GDDR6。
    3. 下一层重复：prefetch下一层weight（与当前层计算overlap，但bs=1时weight仅被1个token使用，prefetch带宽无法充分摊销）。
    4. 痛点：PCIe带宽仅为GPU内存带宽的~1/15，即使利用activation sparsity（Deja Vu），cold neurons仍需从host memory搬运→ PCIe成为绝对瓶颈。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Hermes通过在host memory端引入NDP-DIMM（Near-Data Processing DIMM），将计算推到数据所在位置，从根源消除PCIe数据搬运瓶颈。
  - **算法pipeline**: 利用activation sparsity的power-law分布（20% hot neurons承担80%计算，80% cold neurons仅20%计算），将LLM权重分为hot/cold两类。设计lightweight online predictor（4-bit neuron state table + token-wise similarity + layer-wise correlation），替代Deja Vu的昂贵MLP predictor。预测准确率98%，内存<1MB（LLaMA-7B仅232KB），推理overhead <0.1%。对比Deja Vu: MLP predictor占18.1%计算时间且需2GB存储。
  - **系统框架层**: 提出GPU + NDP-DIMM异构计算策略。Hot neurons（计算密集型，~20%参数）映射到GPU GDDR6 + Tensor Cores执行GEMM。Cold neurons（存储密集型，~80%参数）存储在NDP-DIMM，由DIMM内置GEMV unit就地计算，仅传输少量计算结果（几KB）到GPU。对比baseline: 消除80% weight的PCIe搬运，将主要数据流从"PCIe weight搬运"转变为"DIMM内部访存 + local computation"。
  - **编译框架层**: 论文未明确说明。使用PIM-SYCL编程模型和统一内存编程（Unified Memory）实现GPU-NDP-DIMM协同。
  - **kernel调度层**: （1）QKV generation和MLP block被split为GPU hot GEMV ∥ DIMM cold GEMV，merge在DIMM侧（GPU完成快的优势被DIMM计算隐藏）。（2）Attention完全offload到NDP-DIMM（memory-bandwidth-intensive，DIMM高内部带宽更适合），KV cache存在DIMM内存中，节省GPU显存。（3）Window-based online scheduling：每5个token统计各DIMM负载，将高负载DIMM的最活跃cold neuron remap到低负载DIMM（通过DIMM-link 25GB/s），保证NDP-DIMM间load balance。（4）Online neuron adjustment：projection计算期间（DIMM idle），将predictor识别的新hot neuron从DIMM拷贝到GPU memory，替换GPU上state最低的old cold neuron。对比baseline: FlexGen的offline zig-zag无法适应token generation的bs=1场景；Deja Vu仍需PCIe搬运cold neuron。
  - **硬件架构层**: （1）Center buffer-based NDP-DIMM设计——GEMV unit（256 multipliers bit-serial @1GHz, reduction tree accumulator, 256KB buffer）通过center buffer访问DIMM内所有DRAM数据，同时不干扰normal memory access。（2）Activation unit（256 FP16 exp/add/mul + tree structures）支持LLM非线性操作。（3）DIMM-link（25GB/s point-to-point）实现高效inter-DIMM数据搬移（62× faster than host CPU中转）。GEMV unit面积1.23mm²/DIMM（TSMC 7nm），硬件overhead极小。对比baseline: Host DIMM完全无计算能力，所有计算依赖GPU→PCIe搬运→GPU处理→PCIe回传。
  - 全栈执行例子（Hermes, LLaMA2-70B, 单RTX 4090 + 8 NDP-DIMMs, bs=1, token generation）：
    1. **Predict (Host scheduler)**: Monitor采集前tokens neuron活动 → predictor用token-wise FSM更新neuron state table + 查layer-wise correlation table → 预测当前层activated neurons → neuron mapper判定hot (state>10) / cold → 若GPU需要新hot neuron，在projection期间从DIMM拷贝(PCIe 4.0 64GB/s, weight <几MB)覆盖GPU旧cold slot。
    2. **QKV generation (Layer l)**: GPU Tensor Cores执行hot neuron GEMM（权重在GDDR6, 936GB/s）∥ 各NDP-DIMM的GEMV unit从自身DRAM（内部带宽>>PCIe）读取cold neuron权重执行bit-serial GEMV → GPU结果(几KB)通过PCIe传到merge DIMM → 合并为完整Q/K/V。
    3. **Attention (in NDP-DIMM)**: QK^T→softmax→×V全在DIMM的activation unit + GEMV unit执行。KV cache存在DIMM内存中，利用DIMM高内部带宽。
    4. **Projection (GPU only)**: GPU Tensor Cores执行projection GEMM。此时DIMM idle → scheduler启动online adjustment（neuron swap + window-based remap check）。
    5. **MLP (Layer l)**: 同QKV generation: GPU hot GEMM ∥ DIMM cold GEMV → merge in DIMM。
    6. **Window check**: 每5个token，统计各DIMM Z_j → 配对max/min → DIMM-link remap最活跃cold neuron → load balance。
    7. **输出**: LM head → next token。关键差异：cold neurons（80%参数）从未离开DIMM，PCIe仅在传输hot neuron结果（KB级）和neuron swap时使用→ PCIe不再是瓶颈。
    8. **结果**: 端到端13.75 tokens/s（LLaMA2-70B），对比Deja Vu 75.24× speedup，对比TensorRT-LLM(5×A100) 达79.1%性能，成本$2,500 vs $50,000。

## 78-BitMoD_Bit-serial_Mixture-of-Datatype_LLM_Acceleration.pdf

- baseline方法是什么？
  Baseline是现有LLM量化加速方案，主要包括两类：
  - **软件量化方案（AWQ [30]、GPTQ [20]、OmniQuant [42]）**: 采用per-group非对称整数量化(INT4/INT3)，将LLM权重压缩到低精度整数，但推理时需先dequantize到FP16再通过GPU floating-point pipeline计算，缺乏专用混合精度计算单元导致计算效率差。per-group量化需per-group dequantization（不同于per-channel可以fuse到element-wise操作中），若缺乏专用硬件开销极大。
  - **硬件协同设计方案（ANT [26]、OliVe [25]、FIGNA [27]）**: ANT和OliVe设计custom bit-parallel PE支持低精度权重×FP16激活，但仅支持per-channel量化（缺少per-group dequantization硬件），无法利用per-group量化大幅降低quantization error的优势。且ANT/OliVe的custom data types在per-group下精度反而不如简单INT-Asym（Table I），因为其数据类型设计时未考虑per-group场景。FIGNA虽支持混合精度（FP-INT4/INT8），但不同精度需独立PE，无法在统一架构中trade-off精度和效率。精度方面，ANT/OliVe/MX在3-bit per-group下perplexity崩溃（ANT平均∆PPL=57.61）。
  - **Microscaling (MX) [40]**: 每32个低精度FP权重共享8-bit shared exponent，支持多精度但需要FP pipeline处理group-level dequantization，能耗较高。
  Baseline全栈执行例子（以AWQ INT4-Asym per-group on GPU为例）：
  - **算法pipeline**: Per-group INT4-Asym对称量化：Wq=Round(Wf/Δ)+z → 存储INT4 weight + 16-bit Δ + 8-bit zero-point per group。推理时dequantize到FP16: Wqf=(Wq-z)·Δ → FP16 matmul。
  - **系统框架层**: 无定制加速硬件，在通用GPU上运行。INT4 weight从DRAM读取→GPU ALU dequantize to FP16→写入register file→FP16 Tensor Core GEMM→每group重新dequantize。Dequantization overhead和FP16 matmul之间的切换带来pipeline bubble。
  - **编译框架层**: 论文未明确说明（PyTorch/CUDA原生执行）。
  - **kernel调度层**: CUDA kernel: INT4 weight load从shared memory→INT-to-FP16转换（dequantization）→FP16 GEMM via Tensor Cores。Per-group dequantization必须插入matmul循环内（因不同group不同Δ），打断Tensor Core的连续流水。ANT/OliVe使用bit-parallel custom PE，每个PE固定支持一种精度（如INT8-INT4），无法per-group自适应。
  - **硬件架构层**: GPU: NVIDIA A6000/A100 Tensor Core (FP16)。ANT/OliVe: custom ASIC with bit-parallel PEs (28nm/45nm), fixed precision per PE。均无专用per-group dequantization硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  BitMoD是**算法-硬件协同设计**方案，通过三个层面解决baseline缺陷：
  
  **(A) 算法层 - 缺陷：现有数据类型在per-group低精度量化下精度不足** → BitMoD设计扩展非对称浮点数据类型（FP3-ER/EA, FP4-ER/EA）：利用浮点格式的冗余零值，引入4个可编程special values ({±3,±6} for FP3, {±5,±8} for FP4)，通过细粒度数据类型自适应（每group选最优special value）实现低精度（3-bit/4-bit）per-group量化下的高精度。对比ANT/OliVe/MX在3-bit per-group下崩溃（perplexity不可用），BitMoD在3-bit下平均∆PPL仅2.94（vs INT3-Asym 24.34, ANT 57.61）。4-bit下平均∆PPL 0.48，优于所有baseline。
  
  **(B) Kernel层 - 缺陷：不同精度需要不同PE，缺乏统一高效架构** → BitMoD提出统一bit-serial PE：将INT8/INT6/FP4/FP3所有权重类型通过统一bit-serial表示（Booth编码→INT, LOD+special value→FP）映射到相同的(ws, we, wm, wbsig) term格式，在4-way bit-serial dot product pipeline上统一计算。关键创新：bit-serial computing天然trade-off精度与延迟→低精度=更少bit-serial terms=更少cycles=更高吞吐。对比baseline(ANT/OliVe/FIGNA)的bit-parallel PE：FIGNA支持多精度需独立PE设计（FP-INT4 PE面积/功耗比FP-INT8 PE更大，因dual output加倍accumulator开销），而BitMoD一个PE支持所有精度，只需改变cycles数。BitMoD PE面积比FP16 PE小24%，能在iso-area下放入更多PE。
  
  **(C) 硬件层 - 缺陷：缺乏低开销per-group dequantization硬件** → BitMoD设计bit-serial dequantization单元：将per-group INT8 scaling factor逐bit与accumulator相乘→shift-and-add，仅需8 cycles且嵌入PE pipeline的Step❹。由于group size=128, PE dot-product size=4，group dot-product需64 cycles（2×32 for FP3），8-cycle dequantization远小于计算cycles→**从不stall pipeline**。对比baseline：AWQ等软件方法在GPU上per-group dequantization必须插入matmul循环打断Tensor Core连续流水；ANT/OliVe仅支持per-channel dequantization（fuse到layer-norm），不支持per-group；MX需要额外FP pipeline处理shared exponent dequantization。
  
  BitMoD全栈执行例子（同一Llama-2-7B, FP3 weight, batch_size=1, seq_len=256, TSMC 28nm BitMoD accelerator）：
  - **算法pipeline→BitMoD Quant**: 每128个权重为group→非线性量化到FP3-EA/ER的quantization values（含选定的special value）→存储：3-bit quantized weight + 8-bit INT8 scaling factor + 2-bit special value encoding per group。量化开销：10-bit extra memory per 128 weights（0.078 bits/weight overhead）。对比baseline(AWQ INT4-Asym): 4-bit weight + 16-bit Δ + 8-bit zero-point per group（24-bit overhead），BitMoD overhead更低。
  - **系统框架层**: BitMoD custom ASIC accelerator。模型部署前一次性编程Special Value Register File (4个SV值) → Bit-serial Term Generator实时解码权重到bit-serial terms → PE array(4×4 tiles×8×8 PEs)执行systolic output-stationary GEMM。无GPU driver/CUDA runtime overhead，所有计算在专用硬件data-path上完成。
  - **编译框架层**: 论文未明确说明。量化算法(PyTorch)在GPU上offline完成→导出quantized weights+metadata→加载到accelerator weight buffer。
  - **kernel调度层→Bit-serial PE**: 每个FP3权重被LOD解码为2个bit-serial terms → 广播到PE列。FP16 activation广播到PE行。PE pipeline: ❶Exponent alignment → ❷Bit-serial multiply (1-bit wm × 11-bit am, 右移对齐) → ❸Group accumulation (adder tree + shift by bsig + normalize) → ❹Bit-serial dequantization (mACC × INT8 Δ逐bit, shift-and-add 8 cycles, 输出FP16 partial sum)。FP3=2 terms→2 cycles/MAC→2×吞吐 over FP16 MAC。INT6=3 terms→1.33×吞吐。Key insight: bit-serial dequantization的8 cycles被64 cycles group dot-product完美覆盖→零stall。对比baseline GPU: INT4 weight→dequantize to FP16 (extra ALU ops + register pressure) → FP16 Tensor Core GEMM → per-group dequantization打断pipeline。
  - **硬件架构层**: BitMoD ASIC, TSMC 28nm, 1GHz, PE Tile 8×8 PEs, 99,509 µm²/tile, 39.36 mW/tile。4×4 tile systolic array。对比baseline加速器: ANT和OliVe需更高权重精度(≥4-bit)维持per-channel量化精度→更高DRAM energy（weight读取代价更大）；BitMoD支持3-bit per-group→DRAM traffic降低至3/16=18.75%。结果: lossless BitMoD (INT6) vs FP16 baseline: 1.99×/2.41× speedup (disc/gen), 2.31× energy efficiency。Lossy BitMoD vs ANT/OliVe: 1.69×/1.48× speedup and 1.48×/1.31× energy efficiency。

## 80-CoServe- Efficient Collaboration-of-Experts (CoE) Model Inference with Limited Memory.pdf

- baseline方法是什么？
  Baseline 是 Samba-CoE [MICRO 2024]，当前唯一的 CoE 模型大规模部署系统。Samba-CoE 将频繁使用的 experts 存储在高带宽内存（HBM）中，其余 offload 到 DDR，使用 LRU（Least Recently Used）策略管理 expert 替换。其调度采用 FCFS（First-Come, First-Served）处理推理请求。核心缺陷：
  **(a) FCFS 调度忽略 request-expert dependency**：多个请求可能依赖同一个 expert，但 FCFS 将它们分散安排在队列中不同位置。处理中间不相关的请求时可能 evict 该 expert，后续又需重新加载 → 不必要的 expert switching。实测显示从 SSD 切换到 GPU 的 expert switching latency 占推理总延迟 90% 以上（NUMA/UMA），CPU→GPU 切换也占 60% 以上。
  **(b) LRU expert management 预测不准确**：LRU 仅依赖历史使用统计，无法准确预测未来 expert 使用需求。CoE 的 routing module 可以独立配置或训练，提供了 expert usage probability 的先验知识，但 LRU 未利用。如图 4 所示：Expert 2（usage probability 13%）被 LRU 优先 evict，而 Expert 3（usage probability 5%）才是更优的 eviction 候选。
  **(c) Memory allocation 的 batch-vs-expert trade-off 未解决**：增加 batch size 提升 GPU 利用率但消耗更多中间结果内存，减少可存储的 expert 数量 → 增加 expert switching 频率。不同设备（NUMA/UMA、CPU/GPU）的最优配置不同且难以人工确定。

  Baseline 全栈执行例子（以 Samba-CoE on NUMA RTX 3080Ti, circuit board inspection, 3 GPU executors 为例）：
  - **算法pipeline**: CoE 模型——routing module 根据电路板组件类型选择 classification expert (ResNet101) → 如无 defect 则调用 object detection expert (YOLOv5m/l)。每个 expert 为独立训练的模型。
  - **系统框架/Serving**: Samba-CoE, FCFS 调度 + LRU expert eviction。请求按到达顺序入队 → 按序处理 → 若所需 expert 不在 GPU memory 中则触发 expert switching（从 CPU memory 或 SSD 加载）。
  - **编译框架**: 论文未明确说明（PyTorch 原生执行）。
  - **kernel调度**: Expert switching 发生时：GPU memory 中 LRU 最久未使用的 expert 被 evict → 从 CPU DDR/SSD 加载所需 expert 参数到 GPU memory → 执行 batched inference forward pass。SSD→GPU 传输 >90% 推理时间，CPU→GPU 传输 >60% 推理时间。
  - **硬件架构**: RTX 3080Ti 12GB GPU + Xeon Silver CPU 16GB + 530MB/s SSD。Expert pool ~60GB（>300 experts, 13B 参数），无法全部加载到 GPU memory → 动态 offload/load 循环。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  CoServe 通过利用 CoE 的 **expert dependency** 特性（request-expert dependency 和 inter-expert dependency），从 serving 调度和 memory management 两个层面协同优化：

  **(A) Dependency-aware Request Scheduling（解决缺陷 a）**：
  - **Request Assigning**：预测每个 executor queue 添加新请求后的额外推理延迟（`Δlatency = K × batch_size + B + expert_switching_latency`），选择使 `max(queue_total_times)` 最小化的 queue 分配请求。
  - **Request Arranging**：在 queue 内将新请求排列到使用相同 expert 的已有请求之后 → 将使用同一 expert 的请求 group 在一起连续处理 → expert 最多被加载一次。
  - **Batch Splitting**：batch_size = min(available_memory_limit, max_batch_size) → 在高 batch 利用率和不超内存上限之间平衡。

  **(B) Dependency-aware Expert Management（解决缺陷 b）**：
  - **两阶段 eviction**：Stage 1 优先驱逐无 preliminary dependency 的 subsequent experts（按 memory footprint 降序，最小化 eviction 数量）→ 不浪费已加载的 preliminary expert；Stage 2 按 usage probability（而非 LRU history）升序驱逐 → 保留高概率 experts。
  - **Expert 使用概率**：通过 CoE routing rules 或 routing on sample dataset 预先计算，作为 expert management 和 initialization 的准确先验。

  **(C) Offline Profiler——自动化配置搜索（解决缺陷 c）**：
  - **Microbenchmarks**：对每种 expert architecture 运行 sweep batch sizes，获取 K, B 常数、max batch size、memory footprint、loading latency。
  - **滑动衰减窗口搜索**：将 experts 按 usage probability 降序排列生成 CDF 曲线 → 以衰减窗口（`decay_factor = 1 - window_size/100`）在 CDF 上滑动搜索最优 expert 加载数量 → 当实际 throughput 不再随 expert 增加而线性增长时终止（`|f(N+1) - actual| / f(N+1) > error_margin`）。
  - **自动确定配置**：离线计算出 GPU/CPU 的最优 memory allocation 和 executor 数量，无需人工调优。

  CoServe 全栈执行例子（同一硬件，3 GPU + 1 CPU executors, Task A1）：
  - **算法pipeline**: 同 Baseline（CoE routing + ResNet101/YOLOv5m/YOLOv5l）。
  - **系统框架/Serving→CoServe**: Dependency-aware Scheduler 将请求分配到总时间最短且 Δlatency 最小的 executor queue → 请求在 queue 内排列到同 expert 请求之后 → Batch Splitter 按可用内存和 max_batch_size 分 batch → 连续处理同 expert 的请求 → expert switching 次数从 513 降至 68（减少 86.7%）。
  - **编译框架**: 论文未明确说明（PyTorch 原生）。
  - **kernel调度→Expert Switching 优化**: Expert Manager 在 eviction 时：(Stage 1) 先扫描 model pool 中无 preliminary dependency 的 subsequent experts（如 YOLO expert 在 ResNet expert 之前被 evict），按 memory footprint 降序逐出最少数量满足新 expert 加载要求；(Stage 2) 若仍不足则按 usage probability 升序逐出。加载新 expert 后执行 batched inference。关键效果：expert switching latency 大幅减少。
  - **硬件架构**: 同 Baseline。Offline profiler 确定 RTX 3080Ti GPU 加载 35 experts（根据最优 CDF window），剩余 GPU memory 用于 batch inference（中间结果）。CPU executor 处理低概率 experts 的请求作为补充。Throughput: 28.7 img/s（vs Samba-CoE 3.0 img/s，9.4× 提升）。


## 84-MicroScopiQ: Accelerating Foundational Models through Outlier-Aware Microscaling Quantization

- baseline方法是什么？
  **Group A baseline (GOBO为代表)**: 离群值以全精度(FP32)存储，与低精度(4-bit)内点分离，使用稀疏编码+离群值索引存储。执行例子：`LLM weight tensor [d_row × d_col] → 3σ规则分离outliers/inliers → inliers quantize为4-bit per-channel group quantization → outliers保留FP32 + 存储outlier index和block offset → 硬件上使用两种PE (高精度Outlier PE处理FP32 outliers + 低精度Group PE处理4-bit inliers) → sparse编码导致unaligned memory access → PE complex (需不同precision PE) → 加速器面积大、EBW高(18.17 bits) → 高精度但硬件效率差`。

  **Group B baseline (OliVe为代表)**: 离群值与内点以相同精度(4-bit)但不同数据格式量化，通过剪枝离群值相邻的内点作为标识符区分inlier/outlier格式。执行例子：`LLM weight tensor → 识别outliers → 将每个outlier相邻的内点prune为0作为"identifier" → inliers用"flint"格式(4-bit)、outliers用"abfloat"格式(4-bit)统一bit-budget → aligned memory access (统一4-bit per element) → 但PE需encoding/decoding units处理exponent-integer pair → 假设outlier不邻接(modern FMs不成立: LLaMA3有>0.5% adjacent outliers per layer) → 导致unintended outlier pruning → 显著accuracy degradation (特别是ultra-low bit-width) → 对齐内存但精度差`。

  **全栈执行对比 (GOBO vs OliVe)**:
  - 算法pipeline: GOBO保留离群值高精度(FP32)/OliVe统一精度 → 前者EBW高(18.17b)/后者低(4b)
  - 系统框架/Serving: 两者均为weight-only PTQ → 论文未明确说明Serving集成
  - 编译框架: 论文未明确说明
  - kernel调度: GOBO需双PE类型调度/OliVe需PE内encoder-decoder调度
  - 硬件架构: GOBO有complex Outlier PE + sparse index处理/OliVe有complex PE内格式转换电路

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MicroScopiQ通过**剪枝互补离群值感知量化**实现Group A的高精度 + Group B的硬件效率的统一：
  
  **核心设计**：(1)离群值2×精度(MX-FP-4_{8,8})、内点低精度(MX-INT-2_{128})；(2)通过Hessian信息识别μB中最不重要的内点剪枝；(3)将离群值的额外比特位(LSB halves)重新分布到剪枝位置→每个tensor元素保持统一bit-budget (b_b)和数据格式(INT)→实现aligned memory；(4)ReCoN蝶形NoC在硬件层抽象FP离群值处理复杂度→PE保持简单homogeneous INT结构。

  **全栈执行例子 (MicroScopiQ W2A16)**:
  - **算法pipeline**: `LLM weight layer [d_row × d_col] → Hessian H^{-1} = (2XX^T + λI)^{-1} (from calibration data) → 按row block(rB=128)迭代量化 → per-row: 分为MaB(128)→3σ分离inliers/outliers → inliers: MX-INT-2_{128} group quant (shared I_sf = 2^{I_sf}, always negative power-of-two) → outliers in μB(8): 乘以2^{I_sf}预缩放 → MX-FP-4_{8,8} quant (level-1 O_sf_l1 + level-2 μX共享) → Hessian-guided pruning: 选择μB中n个least important inliers (min w_p²/[H^{-1}]_{pp}) → 剪枝=0 → 离群值拆分为{s,m1}和{s,m0}两个2-bit halves → Upper half留在原离群值位置, Lower half放到剪枝位置 → permutation list记录映射 → 更新unquantized rows的权重补偿误差(局部化在rB内→O(rB) speedup) → EBW = (24b perm_list + 16b weights + 8b MXScale)/8 ≈ 2.36 bits`。

  - **系统框架/Serving**: 论文未明确说明。GPU侧通过CUDA kernel集成到PyTorch推理pipeline。

  - **编译框架**: 论文未明确说明。

  - **kernel调度/ReCoN**: `Weight stationary systolic array → PE row i加载μB weights → MODE_2b或MODE_4b → iAct从左流入 + iAcc从上方流入 → PE内MUL(4×4×2-bit multipliers→adder/shifter组合→Res) → ADD(与iAcc累加) → 若μB无outliers: 直接传partial sum到下一PE行 → 若μB有outliers: Controller拉高Outlier_Present+OAcc_NoC/PE → PE输出→Synchronization Buffer(补偿skew)→column-wise arbiter(多行仲裁)→ReCoN多级蝶形网络 → Level-1 Switch(Swap: redirect Lower→Upper列) → Level-2 Switch(Swap: 继续redirect) → Level-3 Switch(Merge: ||运算→分离Res和iAcc→Upper>>1, Lower>>2→相加+iAct(hidden-bit)→完整FP-outlier partial sum) → pipelined输出到下一PE行或oAct buffer`。

  - **硬件架构**: `Verilog RTL → Synopsys DC + Innovus PnR → TSMC 7nm, 1GHz → 64×64 homogeneous INT PE array (Base PE 2.82μm²) → multi-precision support (0.22μm²/PE, 通过MODE signal) → 1×ReCoN unit (204.68μm², {n(log₂n+1)} switch topology, time-multiplexed) → Sync buffer (20.45μm²) → Controller (105.78μm²) → Compute area: 0.012mm² (8.63% overhead) → Compute density: 367.51 TOPS/mm² (2× vs OliVe, 14× vs GOBO) → 2MB L2 SRAM + HBM2 (256 GB/s) → Post-Processing Unit (overlapped oAct scale compute) → MicroScopiQ v2 (WxA4): 2.47× speedup vs baselines (iso-accuracy), 1.5× lower energy → ReCoN overhead: 3% area at 128×128 array → 8 ReCoN units achieve peak performance (zero access conflicts)`。

  **解决baseline缺陷的核心映射**:
  | Baseline缺陷 | MicroScopiQ解决方案 |
  |---|---|
  | Group A: 高EBW、unaligned memory、complex PE | 剪枝+离群值bit重分布→每个元素统一b_b和INT格式→aligned memory + simple PE |
  | Group B: 低精度、假设outlier locality导致accuracy drop | 离群值2×精度+MX-FP format + Hessian-guided pruning (不依赖local assumption) → 高精度 |
  | Group A/B: PE设计complex (encoder/decoder, dual PE types) | ReCoN NoC在阵列外部抽象FP处理→homogeneous INT PE阵列→低area overhead |
  | Group B: 牺牲编码值作identifier | 剪枝的是least important权重(非adjacent to outlier) → 不牺牲有效编码值 |
  | 通用: migration strength α受限(≤0.5) | MicroScopiQ对权重中更多outlier的鲁棒性→α可达0.7→迁移更多activation outlier到weights |

## 88-Amove- Accelerating LLMs through Mitigating Outliers and Salient Points via Fine-Grained Grouped Vectorized Data Type.pdf

- baseline方法是什么？
  Baseline是现有的LLM量化数据类型与架构协同设计方案，主要包括：
  1. **Scalar data type方法**（ANT [23], OliVe [22], M-ANT [26], BitMoD [7], Anda [19]）：采用per-token/per-channel或coarse-grained group-wise (g=64/128) 量化粒度，每个数据元素独立编码。ANT采用多数据类型混合精度在channel/token粒度自适应选择；OliVe通过outlier-victim pair encoding将outlier编码到邻近normal value中；M-ANT采用数学自适应数值类型支持group-wise量化。这些方法主要分为两类：要么仅支持weight-activation量化（ANT, OliVe, M-ANT专注处理activation outlier），要么仅支持weight-only量化（BitMoD, Anda专注weight的salient point保护），无法在单一框架内同时高效支持两种量化模式。
  2. **Vectorized data type方法**（MX [65]）：采用group-wise共享scale factor (FP8)，每个group内元素用FP4/INT4表示，同时支持两种模式。但因coarse-grained共享scale factor无法充分处理fine-grained outlier和salient point分布，在W4A4下accuracy有显著下降。
  3. **硬件baseline**：INT8 tensor core（NVIDIA Ampere A100架构）和INT8 systolic array，以及各量化方法的定制硬件扩展（ANT/OliVe的4/8-bit mixed-precision架构，Tender的chunk-based shared scale factor）。

  全栈执行例子（以OliVe W4A4 mixed-precision GPU tensor core推理为例）：
  - **算法pipeline**：OliVe量化——对activation执行outlier-victim pair encoding，将outlier activation值编码到邻近normal value的LSB中；weight采用INT4 group-wise quantization (g=64) + FP16 scale factor。量化后activation为4/8-bit混合精度，weight为4-bit，scale factor overhead 16/64=0.25 bits/value。
  - **系统框架**：PyTorch模型加载→OliVe quantizer将Linear层转换为W4A4/8格式→tensor core执行mixed-precision GEMM。
  - **编译框架**：CUDA kernel调用MMA指令(m16n8k16/32)，activation 8-bit + weight 4-bit → INT32 accumulation。
  - **kernel调度**：GPU warp scheduler将thread block分发到SM→operand collector从shared memory加载W/A tile→tensor core执行4×8-bit MAC阵列→outlier decoding恢复原始值。
  - **硬件架构**：NVIDIA Ampere tensor core m16n8k16，64 thread groups/SM，FP16 scale factor存于shared memory，dequantization在global stage执行后将FP16 partial sum写回register file。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：**Amove**——基于Residual Approximation Mechanism的Fine-Grained Grouped Vectorized Data Type与硬件架构协同设计。

  **解决baseline缺陷的核心设计**：

  **缺陷1**: Fine-grained group-wise quantization提高accuracy但增加scale factor memory overhead（group size 128→4时overhead增加32×）。直接quantize scale factor到4-bit导致>50% accuracy loss。
  → **解决方法（Residual Approximation Mechanism）**: 利用fine-grained quantization下scale factor分布的light-tailed特性（大多数LLM kurtosis < 3），用粗粒度group的shared base scale + 1个shared residual + per-cluster 2-bit encoding来近似所有cluster的scale factor。公式：`S_ci = S_base - R * E_ci`。这避免了存储per-cluster scale factor（例如g=128,c=16,2-bit encoding时overhead仅0.25 bits/value，相比group-wise的4 bits/value降低16×，相比MX的2 bits/value降低8×）。Residual对权重采用search-based MSE最小化（离线），对激活采用average deviation（支持在线quantization）。

  **缺陷2**: 现有data type（ANT, OliVe, M-ANT, BitMoD, Anda）只专注单一量化模式（weight-activation或weight-only），无法在同一框架内同时处理activation outliers和weight salient points。MX虽支持两种模式但accuracy显著不足。
  → **解决方法（Unified Vectorized Data Type）**: Fine-grained grouped vectorized data type通过解耦shared scale representation和localized variation（residual + encoding），灵活支持两种模式。在weight-activation模式中：fine-grained cluster-wise scale factor有效smoothing activation outliers进入scale factor并保护weight salient points；在weight-only模式中：fine-grained cluster precision降低salient points和irregular outlier的reconstruction error。通过调整group size、cluster size、encoding bit-width即可切换模式，无需改变数据格式结构。

  **缺陷3**: 现有硬件架构（ANT, OliVe, M-ANT）采用mixed-precision计算（如4-bit weight × 8-bit activation），增加硬件复杂度和限制compute unit throughput。
  → **解决方法（Uniform W4A4 co-design）**: Amove通过data type + 硬件co-design实现attention和linear层统一W4A4量化（包括attention层的QK^T和output projection），消除mixed-precision的不对称性。Tensor core和systolic array仅需添加轻量级scale factor decoder和per-group dequantization unit（area overhead <2%），原有4-bit multiplier datapath完全复用。Scale factor decoding与MAC操作并行执行（overlapped decoding），消除decode latency。Tightly coupled per-group dequantization直接将MAC结果在PE/thread group内部转换为FP16，消除传统global dequantization stage。

  **缺陷4**: 现有方法通常只量化linear层而保持attention层FP16，长序列下attention dominates runtime。
  → **解决方法（Full attention quantization）**: Amove支持self-attention层（Q/K/V projection + QK^T + output projection）的W4A4量化。对attention层使用更细粒度配置（Amove-Conservative: g=32,c=4 uniform 或 Amove-Aggressive: attention g=32,c=4, 1 bit/value overhead vs linear g=128,c=16, 0.25 bit/value），因为attention的QK^T产生大动态范围intermediate results和更多outliers。

  全栈执行例子（Amove W4A4 GPU tensor core推理，Amove-Aggressive配置）：
  - **算法pipeline**：对权重（沿channel维）和激活（沿token维）分别应用Amove量化。Linear层：g=128,c=16,2-bit encoding，S_base(FP8)+R(FP8)+8个encoding(2-bit each)→overhead=0.25 bit/value。Attention层：g=32,c=4,2-bit encoding，overhead=1 bit/value。Residual对权重使用MSE搜索（[-1,1], step=0.01），对激活使用online average deviation。量化公式：`S_ci = S_shared - R * E_ci; X_q = round(X / S_ci); X_hat = X_q * S_ci`。
  - **系统框架**：PyTorch quantized model → Amove quantizer将W/A转换为fine-grained grouped vectorized data type → custom Smma instruction调用。
  - **编译框架**：Compiler感知Amove数据格式的layout和Smma指令语义→将高层API（matrix operations）自动lowering为Smma指令+数据转换。Instruction格式：`Smma.MNK.AdtypeWdtypeSfdtypeAccdtypeOdtype`，scale factor type Sfdtype同时编码精度和格式信息。
  - **kernel调度**：Warp scheduler→operand collector从shared memory加载Amove-compact数据→5-stage pipeline: Preload(解包)→Dispatch(路由到DP unit和scale factor decoder)→Computation(INT4 MAC + decoder并行)→Dequantization(INT4 psum * decoded scale factor→FP16)→Write-back(→DRAM)。
  - **硬件架构**：Modified Ampere tensor core，新增Scale Factor Decoder（计算S_base - R * E而非LUT）+ per-thread-group dequantization unit + FP16 accumulator。Decoder与DP unit完全并行（overlapped decoding）。64 thread groups/SM，area overhead 1.62% vs INT4 baseline。




## 89-AQPIM_Breaking_the_PIM_Capacity_Wall_for_LLMs_with_in-Memory_Activation_Quantization.pdf

- baseline方法是什么？
  Baseline是典型的GPU+HBM-PIM异构LLM推理系统（以AttAcc! [56]为代表），PIM仅在memory-bound的attention kernel（GEMV of query×KV cache）上加速，全量KV cache以FP16精度存储在HBM-PIM中。全栈执行例子（Mistral-7B-Instruct-v0.2, 4K input, batch=16, H100 GPU + 4×HBM-PIM）：
  - **算法pipeline**：标准FlashAttention计算流程。Prefilling: QKV生成→分块softmax attention→projection+FFN；Decoding: qkv生成→GEMV (q×K^T, 1×d × d×N)→softmax→GEMV (a×V, 1×N × N×d)→projection+FFN。KV cache以FP16全精度存储，无压缩。
  - **系统框架**：GPU+PIM异构计算。Prefilling阶段GPU负责全部计算并offload KV到PIM；Decoding阶段GPU生成qkv后发送到PIM，PIM执行attention GEMV后将结果返回GPU做projection+FFN。GPU-PIM之间通过HBM-PHY进行数据搬运。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：AttAcc!的BankPE执行FP16 GEMV（MUL+SUM），多个bank并行计算query与各bank中K cache的部分内积，BufferPE执行softmax和最终累加。利用PIM内部高带宽（7.2× H100 HBM带宽）加速memory-bound的attention kernel。
  - **硬件架构**：HBM-PIM（HBM2/HBM3）上每bank旁放BankPE（含FP16 MAC单元、GRF、control logic），buffer die上放BufferPE（含softmax unit、accumulator等）。DRAM row buffer 1KB。**根本缺陷**：PIM设备内存容量有限（HBM-PIM的bank-level PE导致密度损失），长上下文下KV cache（可达数百GB）远超PIM容量。例如AttAcc!即使短上下文也需40个HBM-PIM设备。且**无法直接加量化硬件**：主流量化方案（FP16+INT32 MAC）在PIM BankPE上需额外增加约126%面积，破坏存储密度。Offloading（KV溢出到CPU memory）的PCIe带宽仅为HBM的1/26，会抵消PIM收益。Sparse attention的随机访问模式与PIM的数据局部性需求冲突。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出**AQPIM**，核心设计理念：**将PIM的"高内部带宽"这一underutilized优势反向赋能给clustering-based量化算法（PQ），使之前因带宽需求过大而不可行的online clustering方案变为现实，从而在不加量化硬件的情况下打破capacity wall**。

  **全栈执行例子**（Mistral-7B-Instruct-v0.2, 4K input, batch=16, H100 GPU + 4×HBM-PIM with AQPIM）：
  - **算法pipeline**：
    Prefilling：(i) GPU生成QKV，offload KV到PIM；(ii) PIM在GPU执行attention时并行进行**PQ-based在线聚类量化**：将K/V按d_head维度split为m=32个子向量→每个子向量空间执行importance-weighted k-means（K=512 centroids，4 iterations）→生成key/value codebook和indices；(iii) Channel sorting预处理（离线完成，吸收到projection weights中）。Key insight：聚类开销O(K·N)远小于attention的O(N²)，因此完全被GPU prefilling计算隐藏。
    Decoding：(i) GPU生成qkv→offload到PIM；(ii) PIM appends新token的indices（用已有codebook最近邻分配）；(iii) **PQ-based attention**：query split为m子向量→各子向量×对应key codebook子矩阵（固定K×d/m规模，**非N**）→inner product matrix (m×K)→用key indices查表取inner product值→子向量轴sum→近似qK^T→softmax→value indices查value codebook重建V×attn scores→输出。**GEMV从O(d·N)降为O(K·d/m·m) = O(K·d)**，其中K=512是常数。

  - **系统框架/Serving调度**：
    Sequence-by-sequence pipelining：GPU逐sequence生成qkv并offload→立即处理下一个sequence，在decode阶段GPU-PIM串行间隐藏GPU idle。Prefilling阶段GPU和PIM完全并行（PIM做codebook生成，GPU做attention+projection+FFN）。Memory allocation策略：codebook固定、buffer逐层复用、indices逐层分配并覆盖复用。

  - **编译框架**：论文未明确说明。

  - **kernel调度**：
    PIM BankPE执行：DC（距离计算）、ATNK（query×key codebook乘）、ATNV（intra-row indirection lookup+sum）、CC分子部分。PIM BufferPE执行：CA（cluster assignment=argmin）、SFM（softmax: EXP+DIV+MAX+SUM）、CC分母倒数。不新增任何量化专用ALU——DC/ATNK/ATNV用已有FP16 MAC，CA用已有MIN，SFM用已有EXP/DIV。

  - **硬件架构**：
    **Intra-Row Indirection**是解决PQ查表随机访问瓶颈的关键HW co-design。Page-aware windowed clustering保证任一窗口内的centroids数量≤512个（=1KB row buffer = 512×FP16），从而所有indirection查表仅需1次row activation，列解码器通过GRF中的indices信号经MUX顺序输出对应值。只需极少的硬件修改（0.0565mm² per HBM, 0.43% of BankPE area），将随机的逻辑访问变为predictable row-buffer hits。

  **核心设计→缺陷映射**：
  - **缺陷①（PIM Capacity Wall）→ Online PQ量化**：传统PIM无法容纳长上下文KV cache（数百GB超HBM-PIM容量）。AQPIM用PQ将KV cache压缩约6.53×（m=32子向量 + K=512 centroids），在不引入量化硬件的情况下将capacity wall推后多倍。关键insight：PIM的underutilized高内部带宽恰好满足online clustering的大带宽需求，使之前因带宽限制而不可行的PQ在线聚类变为现实。
  - **缺陷②（量化硬件面积过大）→ PQ-based attention computing directly on compressed data**：传统量化需FP16+INT32 MAC（+126%面积）。AQPIM通过将GEMV转为codebook查表+求和，完全跳过解量化步骤，所有计算使用已有FP16 MAC单元，无需新增量化ALU。GEMV计算量从O(N)降为O(K)（K=512常数），同时消除了解压的开销。
  - **缺陷③（随机访问破坏PIM局部性）→ Page-aware windowed clustering + Intra-row indirection**：PQ查表产生随机访问，传统方案会频繁触发DRAM row activation。AQPIM通过(algo)限制窗口内centroids≤512使其fit在single row，(HW)列解码器indirection streaming，将随机逻辑访问转为单row-buffer内的顺序流式访问。
  - **缺陷④（标准PQ精度损失）→ Importance-weighted k-means + Channel sorting**：标准PQ平等对待所有token，忽略attention score差异。AQPIM用attention score作为权重引导聚类（高注意力token量化误差更小），用cosine similarity grouping预处理提高子向量内聚性，在高压缩率（128 centroids）下维持accuracy。

## 8-SambaNova_SN40L_Scaling_the_AI_Memory_Wall_with_Dataflow_and_Composition_of_Experts.pdf

- baseline方法是什么？
  Baseline是在传统GPU架构（DGX A100/H100）上部署Composition of Experts (CoE) 模型。GPU架构的特点：
  - **算法/模型层**：CoE由多个独立的小型expert models（如Llama2-7B）和一个router组成。Router先确定使用哪个expert，再加载该expert执行推理。各expert独立编译和fine-tune。
  - **系统框架/Serving层**：GPU上部署CoE时，expert weights存储在HBM或host DRAM中。当HBM容量有限时（DGX H100: 8×80GB=640GB HBM），超过~50个7B experts后必须spill到host DRAM，触发host-to-GPU memory copy（DGX A100: 32GB/s, DGX H100: 64GB/s）。模型切换占CoE总延迟的主导地位（Figure 1）。
  - **编译框架层**：GPU使用PyTorch2 [43] 或 TensorRT [42] 进行operator fusion。传统fusion限制：(1) 仅支持1-5个operators融合；(2) 不支持含transpose/shuffle等复杂access patterns的operator chain融合；(3) 不支持pipeline parallelism across operators。因此Figure 3的Monarch FFT分解中Gemm0-Mul-Transpose-Gemm1 chain无法full fusion，operational intensity仅102.6 FLOPs/byte（memory-bound on A100），而full fusion可达410.4 FLOPs/byte。
  - **kernel调度层**：GPU kernel launch model——host CPU通过CUDA runtime逐个launch kernel → 每个kernel以grid of thread blocks执行 → inter-kernel data必须通过HBM交换（无on-chip cross-SM direct communication）→ kernel launch overhead在decode阶段（kernel执行时间极短，dominated by weight loading）变为显著瓶颈。
  - **硬件架构层**：A100/H100使用SIMT编程模型 + HBM-only memory hierarchy。GPU的rigid memory hierarchy（register→L1/shared memory→L2→HBM）无法支持跨SM的pipeline data streaming，transpose等操作需materialize到HBM再reload，造成data movement bottleneck。HBM bandwidth utilization通常<50% for LLM decode workloads。

  Baseline全栈执行例子（以DGX H100运行CoE, 150 experts, Llama2-7B decoder layer为例）：
  1. Prompt到达 → Router kernel launch (HBM中) → 确定target expert。
  2. Expert不在HBM中 → host-to-GPU DMA (64GB/s) 从host DRAM加载7B expert weights (~13GB, ~200ms)。
  3. Decoder layer执行：Q/K/V projection GEMM kernels → results to HBM → attention kernel (FlashAttention-2 fused) → results to HBM → FFN FC1 kernel → HBM → activation kernel → HBM → FFN FC2 kernel → HBM。每个kernel间data must materialize to HBM and reload。
  4. Autoregressive loop (20 iterations)：每个iteration重复上述kernel launch序列 → kernel launch overhead累积 → HBM bandwidth utilization ~50%。
  5. 若另一个prompt需不同expert → 重复步骤2-4。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出SN40L RDU（Reconfigurable Dataflow Unit）+ Samba-CoE，通过streaming dataflow和三阶存储系统解决GPU在CoE部署中的memory wall和低效问题：
  
  **缺陷①（GPU CoE模型切换延迟主导延迟）→ 三阶存储系统 (SRAM-HBM-DDR)**
  - Baseline缺陷：GPU仅有HBM，expert超50个后spill到host DRAM，host-to-GPU bandwidth仅32-64GB/s，模型切换~200ms占CoE延迟主导。
  - 论文方法：SN40L引入直接attached到accelerator的DDR tier（1.5TB, 200GB/s per socket），8-socket aggregate DDR→HBM bandwidth >1TB/s。7B expert weights (~13GB) 切换仅~10ms。LRU eviction管理HBM缓存，编译器read-only annotations跳过weight write-back。单node可serve最多850 experts at TP8 latency vs DGX在>150 experts时OOM。模型切换加速15-31×。

  **缺陷②（GPU operator fusion受限，小型模型operational intensity低）→ Streaming Dataflow with Arbitrary Access Pattern Fusion**
  - Baseline缺陷：GPU fusion limited to 1-5 operators，不支持transpose/shuffle等复杂access patterns的chain fusion。导致小型expert models (7B) 的operational intensity低（Figure 3的Monarch: unfused仅39.5 FLOPs/byte, memory-bound），GPU利用率低。
  - 论文方法：Streaming dataflow将20+ operators自动融合为单一coarse-grained pipeline kernel。PCU配置为systolic array (GEMM) 或SIMD core (element-wise) → PMU作为decoupling stage buffers → RDN vector fabric stream tensor tiles between stages。Transpose通过PMU diagonal striped format实现全带宽read/write（不需要独立operator）。Compiler自动place-and-route到PCU/PMU 2D mesh。Full fusion将operational intensity从39.5提升到410.4 FLOPs/byte。FlashFFTConv实现13× speedup, Mistral decode 13× speedup, Llama7B prefilling ~3× speedup。

  **缺陷③（GPU kernel launch overhead在decode阶段显著）→ Hardware-Orchestrated Kernel Launch**
  - Baseline缺陷：GPU decode阶段每个kernel执行时间极短（dominated by weight loading），host software kernel launch overhead成为显著瓶颈。
  - 论文方法：AGCU hardware实现autonomous kernel schedule（Program Load→Argument Load→Kernel Execute），消除host software scheduling overhead。Decode benchmarks获1.4×-8× speedup。Prefill/training kernel执行时间长，overhead amortized(~1.1×)。

  **缺陷④（GPU SIMT模型无pipeline parallelism between operators）→ Coarse-Grained Pipeline + Mixed Parallelism on RDU**
  - Baseline缺陷：GPU SIMT model将kernel作为独立grid of thread blocks执行，无法跨operators进行pipeline parallelism。多个小GEMM (如Monarch FFT的32×32×32 matmuls) 无法同时利用所有SM。
  - 论文方法：RDU的streaming dataflow支持在同一fused kernel内混合data/tensor/pipeline parallelism。多个小GEMM通过pipeline parallelism并发执行在不同PCU groups上，element-wise ops在另一些PCU (SIMD mode) 上pipeline，所有stages通过PMU双缓冲解耦。Compiler统一abstraction将multi-level parallelism映射到intra-socket (PCU mesh) 和inter-socket (P2P protocol) 资源。

  **缺陷⑤（GPU HBM带宽利用率低）→ Streaming Dataflow Saturing HBM Bandwidth**
  - Baseline缺陷：State-of-the-art GPU LLM decode仅达到<50% HBM bandwidth utilization，因kernel launch overheads和HBM data movement inefficiencies。
  - 论文方法：SN40L fuses entire decoder layer into single kernel → HBM bandwidth仅用于stream weights和KV cache → 几乎消除所有overheads → 达到>85% HBM bandwidth utilization。P2P protocol直接socket间流式传输（avoid HBM hops for collective communication）。Llama3.1 405B在16-socket SN40L上129 tokens/sec/user（BF16精度），为当时世界最快405B推理平台（GPU需INT8量化才能竞争）。

  论文方法全栈执行例子（以Samba-CoE, 150 experts, 8-socket SN40L Node, Llama2-7B decoder为例）：
  1. **Serving层**：CoE runtime在host CPU接收prompt → Router执行(in HBM, TP=8) → 确定math expert。检查HBM LRU cache → miss，但HBM有空间 → AGCU发起DDR→HBM DMA (>1TB/s aggregate) → 7B expert weights (~13GB) 在~10ms内加载到HBM。
  2. **编译框架层**：SN40L compiler已将decoder layer编译为单一fused kernel——Q/K/V projections (3×PCU systolic arrays) + attention score compute (PCU SIMD) + softmax (PCU SIMD + tail transcendental) + attention×V (PCU systolic) + FFN FC1 (PCU systolic) + activation (PCU SIMD) + FFN FC2 (PCU systolic) + residual add + layer norm (PCU SIMD)。Transpose fused to PMU read/write patterns。Memory allocation: weights in HBM, KV cache in HBM, intermediate activations in PMU stage buffers (static lifetime analysis复用地址空间)。
  3. **kernel调度层**：AGCU hardware-orchestration autonomous触发kernel schedule → streaming pipeline启动。Weights从HBM通过AGCU→RDN vector fabric→PMU stage buffers streamed进pipeline。Q/K/V投影的3个GEMM streams并行在不同PCU groups执行，attention和FFN stages通过PMU双缓冲解耦pipeline。所有数据在RDN上以credit-based flow control流式传输，不经HBM中转（intermediate results在PMU stage buffers中）。
  4. **硬件架构层**：PCU body配置为systolic array (GEMM stages) 或SIMD core (element-wise/attention stages) → PMU diagonal striped format存储tensors实现transpose全带宽 → RDN scalar fabric传输addresses, control fabric传输loop completion tokens → Sequence ID mechanism确保many-to-one data reordering → P2P protocol执行socket间AllReduce（TP=8）。Clock <2GHz, 638 BF16 TFLOPS per socket。
  5. **Autoregressive loop**：20次decode iteration → weights已在HBM中（temporal locality复用） → KV cache在HBM中更新 → 每次iteration streaming pipeline saturates >85% HBM bandwidth。
  6. **Next request**：CoE runtime切换expert → 若已在HBM LRU cache中zero overhead → 否则DDR→HBM copy (~10ms) → 执行。单SN40L Node overall speedup：3.7× vs DGX H100, 6.6× vs DGX A100 (BS=8, 20 tokens), machine footprint reduction up to 19×.

## 90-AUM- Unleasing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving.pdf

- baseline方法是什么？
  Baseline分为两类：(1) **AU-Exclusive（ALL-AU）**：将整个AU-enabled CPU专用于LLM serving，不与其他负载共享，避免管理复杂度和硬件争用。这是当前工业界实践（如AWS、Azure、Inspur部署方案）。(2) **AUV-Oblivious Sharing**：使用SMT（Simultaneous Multi-Threading）或workload-aware Resource Partitioning（如Intel RDT CAT/MBA）将AU应用与通用负载co-locate，但不感知AU Variability。

  **Baseline全栈执行例子（ALL-AU, llama2-7b BF16, GenA SPR, chatbot场景）**：
  - **算法pipeline**：标准Transformer decoder推理，prefill GEMM (8192×4096×22016, AMX TMUL @40.57 TFLOPS) + decode GEMV (16×4096×22016, AVX @3.87 TFLOPS)
  - **系统框架/Serving调度**：xFasterTransformer (xft) exclusive部署——所有48核分配给LLM serving，不共享。Prefill FCFS调度，decode continuous batching。**缺陷**：(a) 资源浪费——prefill阶段AMX密集但memory subsystem有冗余，decode阶段AVX轻量但大量核心idle；(b) 效率低下——CPU性能功耗比仅为GPU的~1/2.1（GenA），perf-per-watt显著劣于同代GPU；(c) ~50% CPU核在GPU-centric数据中心中idle，AU能力未充分利用
  - **编译框架/kernel调度**：Intel oneDNN提供AMX/AVX-accelerated GEMM/GEMV kernel，xft通过oneDNN调用。不涉及kernel级调度优化
  - **硬件架构**：每物理核心含1个AMX单元（TMUL: 1024 BF16 ops/cycle, 8×1KB TILECFG）+ AVX-512单元。AU执行受TDP限制——启用AMX的prefill核心频率降至2.5GHz（vs base 2.7GHz），但AU-disabled核心维持3.2GHz全turbo但被闲置

  **Baseline全栈执行例子（SMT-AU, llama2-7b + OLAP on GenA）**：
  - **Serving调度**：SMT将LLM serving与OLAP共享同一物理核的hyperthread → AU资源不跨HT共享 → AU性能(TPOT) slowdown >200%，OLAP throughput降40% → 因memory contention且SMT无法感知AU dynamic usage
  - **kernel调度**：SMT static scheduling不区分AMX/AVX使用强度 → 轻AU decode与重memory OLAP共享核时产生不可预测的频率干扰（decode 3.1GHz→2.54GHz）→ 无runtime资源调整机制

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出AUM（AU-aware resource Manager），核心设计：**将AU的三维变异性（Usage Pattern / Frequency Interference / Resource Bound）显式建模为离散AUV Model，并通过离线Profiler + 在线Controller两级协作实现usage-aware SLO分析→frequency-aware核心划分→bound-aware资源分配的三阶段闭环管理**。

  **AUM全栈执行例子（chatbot, llama2-7b BF16, GenA SPR, SPECjbb共享）**：
  - **Serving调度层**：①SLO Analyzer：FCFS调度prefill（SLO_H=dTTFT-t_wait）+ LAG-based动态分析decode进度（SLO_L=dTPOT+LAG_i），量化request超前/落后于SLO的程度；②Core Switcher：按加权效率E_CPU=(1.8P_H+0.2P_L+3e-5P_N)/W_CPU最大化划分核心区域，高AU Core数满足prefill TTFT SLO，余下核心给decode和共享；③Allocation Tuner：持续监控token latency，动态回收LLC way/Memory BW给共享负载
  - **kernel调度层**：①AU Selecting：根据ARI判断使用AMX(高密度GEMM)或AVX(小矩阵GEMV)，prefill→AMX, decode→AVX；②Processor Dividing：将48核分为CH(AMX,2.5GHz)、CL(AVX,3.1GHz)、CN(no AU,3.2GHz)三区域，隔离频率干扰；③Resource Allocation：CAT分配LLC way（AU应用affinity低的cache way优先给共享），MBA控制Memory BW配额（检测δ_AU超threshold时回退）
  - **硬件架构层**：利用Intel RDT硬件接口（CAT/MBA）进行细粒度资源分区 → TDP约束下自然频率分频 → 每core含AMX+AVX+通用执行单元，AU不跨HT共享

  **核心设计→缺陷映射**：
  - **缺陷①（AU-Exclusive资源浪费）→ Processor Sharing + Resource Harvesting**：AUM将闲置的None-AU核心分配给共享负载，并在AU应用性能slack时动态回收LLC/Memory BW给共享应用，将CPU perf-per-watt提升4.7-8.8%，使CPU perf-per-dollar达GPU方案的88%
  - **缺陷②（SMT无法感知AU动态使用）→ Usage-aware AU Selecting + Phase-based Division**：AUM用ARI区分prefill(高AU)/decode(低AU)使用，将核心按AU使用强度分CH/CL/CN三区域，避免decode与memory-heavy应用SMT共享造成的>200%性能退化
  - **缺陷③（频率干扰不可控）→ Frequency-aware Processor Dividing**：AUM显式建模TDP限制下的AMX频率降低规律（prefill→2.5GHz, decode→3.1GHz），将不同AU使用强度的核心物理隔离到不同区域，消除跨区域频率cascade
  - **缺陷④（静态资源分区不能适配AU资源亲和度变化）→ Bound-aware Resource Profiling + Collision-aware Allocation Tuner**：AUM离线profile不同AU使用下LLC/Memory BW的资源需求RAU，在线根据runtime SLO slack动态调整分配，优先回收AU affinity低的资源（decode LCC path <5% degradation）给共享应用