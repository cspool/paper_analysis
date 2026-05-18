## AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving

- baseline方法是什么？
  Baseline是工业界当前的**AU-exclusive**和**AUV-oblivious sharing**两种方式：
  
  (1) **AU-Exclusive (ALL-AU)**：将整个AU-enabled CPU独占分配给LLM serving，不与其他workload共享。全栈执行例子（以GenA + llama2-7b chatbot, batch=16为例）：
  - 算法层：LLM serving使用xFasterTransformer框架，prefill phase对QKV mapping执行GEMM（dim=8192×4096×22016），decode phase执行GEMV（dim=16×4096×22016）。所有核心全部使用AMX加速矩阵运算
  - 系统框架层：AU-Exclusive不共享CPU资源→所有48×2=96物理核全部运行LLM serving→无co-located workload→CPU idle核心浪费、冗余硬件资源未利用→perf-per-watt低（比GPU A100差2.1×），perf-per-dollar略优于GPU但效率不足
  - 编译框架层：论文未明确说明（使用Intel oneDNN预编译AMX算子库）
  - kernel调度层：所有核心统一使用AMX，无operator级AU选择——prefill用AMX GEMM（40.57 TFLOPS），decode也用AMX GEMV但效率低（3.87 TFLOPS，tile register配置overhead大）。因TDP限制所有AU核心频率统一降至2.5 GHz（prefill导致最大降频）。无频率区域划分
  - 硬件架构层：Intel SPR Xeon 8475B，96物理核，每核AMX单元1024 BF16 ops/cycle。AU-exclusive导致大量物理核的AU idle（decode phase的AMX cycle ratio仅1.5%），但独占策略避免management complexity
  
  (2) **AUV-Oblivious Sharing (SMT-AU / RP-AU)**：
  - **SMT-AU**：使用SMT (Simultaneous Multi-Threading) 共享AU核心——将LLM serving和通用workload混合调度到同一物理核的hyperthread上。AU不跨hyperthread共享。缺陷：AU perf degradation >200%（由于memory contention），co-running OLAP >40% slowdown（Figure 9a）。compute-intensive shared app虽对AU干扰<10%但自身40% degradation（频率拖累，Figure 9b）。无法控制可变的AU行为和干扰
  - **RP-AU**：使用Intel RDT (CAT/MBA) 做application-aware资源分区——隔离L2 cache、LLC、memory bandwidth给AU和shared应用。缺陷：单独隔离某类资源只能轻微减轻AU slowdown但无法达到最优决策（Figure 10），因为AU的critical backend bound随资源类型不同而变化，单一维度分区不足以应对三维AUV。无法handle AU frequency interference和variable usage pattern

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**AUM (AU-aware resource Manager)**，通过system-layer管理精确灵活地收获AU未利用资源给shared应用，三维度处理AUV。全栈执行例子（以GenA + llama2-7b chatbot shared with SPECjbb为例）：
  
  - 算法层：LLM serving同上（xFasterTransformer, llama2-7b BF16）。AUM不改LLM算法但通过ARI判定AU选择——prefill高ARI用AMX、decode低ARI用AVX/GEMV替代AMX，避免小矩阵AMX tile register overhead
  - 系统框架层（核心创新）：AUM作为系统层resource manager daemon运行。Background AU Profiler离线构建AUV Model（3 division × 3 sharing × 5 config × 10 rep = 450次AU执行）→Runtime Controller在线决策（每次<1ms）。Processor分区为三个频率区域：High-AU (prefill, 2.1 GHz), Low-AU (decode, 2.8 GHz), None-AU (shared SPECjbb, 3.2 GHz)。按加权效率E_CPU = (1.8×P_H + 0.2×P_L + γ×P_N)/W_CPU最大化来切换分区分频
  - 编译框架层：论文未明确说明（使用xft + oneDNN预编译算子）
  - kernel调度层：AUM通过三个runtime阶段做kernel级调度：(1) Slack-aware SLO Analyzer：通过LAG = Σ(d_TPOT - e_token)跟踪每个decode token相对deadline的位置——LAG<0表示落后，需加速→更多AU资源；LAG≥0表示领先→可收获资源给shared；(2) Efficiency-aware Core Switcher：根据SLO slack动态切换core分区分频——若decode underperforming则增加Low-AU region core数量，若prefill SLO slack充足则减少High-AU region→释放core给shared；(3) Collision-aware Allocation Tuner：检测AU shared performance collision（δ_AU > threshold=2）→优先收获对AU干扰最小的资源（如decode低AU usage时先收LLC，因decode对LLC不敏感Figure 13；高affinity的memory bandwidth根据runtime adaptively收放Figure 18），用P_a (avg perf) aggressive收获或用P_t (tail perf) conservative归还
  - 硬件架构层：Intel Xeon processors with AMX。利用Intel RDT CAT/MBA硬件接口实现cache way和memory bandwidth的硬件级隔离。利用AU的SIMD特性（frontend bound仅1% vs 通用功能单元5%）和decode phase memory-bound特性（DRAM bound 59.9%）精确收获硬件资源

  Baseline缺陷→AUM方案映射：
  | Baseline缺陷 | AUM方案 | 效果 |
  |-------------|---------|------|
  | AU-exclusive导致大量AU核心idle（decode AMX cycle ratio仅1.5%）浪费硬件效率 | AU-aware sharing：将decode/low-AU核心上的冗余LLC/BW资源精确收获给shared应用 | CPU efficiency ↑ 8.8% vs ALL-AU |
  | AUV-oblivious SMT无法handle variable AU behavior（perf degradation >200%） | Usage-aware：ARI判定AU选择（AMX vs AVX）+ Frequency-aware：分区分频避免频率interference + Bound-aware：按AU资源affinity调资源 | AU perf SLO guarantee ↑ 11% vs SMT-AU/RP-AU, efficiency ↑ 4.7% |
  | Variable AU usage导致compulsory frequency reduction（prefill→2.5 GHz）拖累共享应用 | Processor Region Division：High/Low/None三区域独立频率管理→decode不与prefill共享频率惩罚 | Shared OLAP 40% degradation → <10% (Figure 9b对比) |
  | AUV-oblivious RP单维度资源分区无法最优（单独LLC/BW隔离仅轻微减轻干扰） | 三维度AU-aware：Usage × Frequency × Resource Bound joint optimization → Runtime Controller自适应调整所有维度 | 精确资源分配（Figure 18: AUM根据runtime info灵活分配LLC和BW，vs static allocation） |
  | 无runtime SLO adaptivity导致LLM serving无法应对dynamic workload | LAG-based SLO分析：实时量化每个request ahead/behind schedule→tune AU resource accordingly | Decode TPOT SLO guarantee比AUV-oblivious高7% |
  | 无AU behavior profiling机制导致resource management盲目 | Background AUV Model：离散化continuous variation为bucket→记录P_a/P_t/W_CPU→供online look up | Profiling cost可摊销（450次执行→覆盖数千核），runtime决策<1ms |
  | GenA→GenC代际提升但efficiency提升有限（Fig 15: 仅1.55×） | AUM leverage更强大AU和memory→更多resource tuning headroom | GenC上AUM efficiency提升19%/11%/17%（vs GenA上15%/7%/10%） |
