## LRM-GPU: Alleviating Synchronization Overhead for Multi-Chiplet GPU Architecture

- 属于芯片设计的实现是什么？实验比较什么？
  提出面向multi-chiplet GPU同步优化的芯片架构设计，核心是chiplet间同步路径上的两个硬件模块：(1) Sync-variable Directory——在LLC中实现轻量级同步变量directory（4-chiplet系统共64 entries，每chiplet 16 entries），仅记录同步变量的owner chiplet（2-bit向量表示4-chiplet owner），不跟踪所有数据的sharer，总容量仅0.4KB（约为一个L1 cache容量的0.3%），避免了完整cache coherence protocol的复杂度和存储开销。(2) Synchronization Atomic Merge Unit (AMU)——嵌入在每个chiplet的interconnection network中（网络侧而非SM侧），包含merge table（支持2K entries × 16 banks，CAM存储status/opcode/address search key + SRAM存储SM list/data，dual-port支持并发读写）、instruction decoder、ALU、multicast unit。AMU面积1.84mm²（40nm）、功耗301.44mW。实验比较multi-chiplet系统同步性能：vs MCM-GPU、hLRC、HMG（使用SM-side LLC + write-through + LLC coherence directory跟踪所有数据，12K entries/chiplet）和AMU-only。还评估chiplet scale（4/6/8 chiplets）对locality和加速比的影响、inter-chiplet transmission latency（8-48 cycles）影响。能耗breakdown显示inter-chiplet和intra-chiplet network占能耗主要部分，新增AMU仅占0.13%系统能耗。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  基于GPGPU-Sim（开源，https://github.com/gpgpu-sim/gpgpu-sim_distribution）扩展为multi-chiplet GPU模拟器，集成基于BookSim 2.0（开源，https://github.com/booksim/booksim2）构建的chiplet interconnection platform。该platform支持composable network construction——每个chiplet内部NoC独立设计配置，chiplet integration阶段将pre-designed intra-chiplet networks组合为inter-chiplet interconnect。AMU的merge table使用Cadence Virtuoso定制电路实现，其他AMU组件使用Verilog RTL + Synopsys Design Compiler综合，均在TSMC 40nm工艺下评估面积和功耗。论文未提供模拟器修改和RTL的官方开源链接。

- 模拟器模拟什么的性能，修改了什么。
  模拟4-chiplet GPU系统：每个chiplet含64 SMs（总计256 SMs），每SM 128KB L1 write-through cache (128B line, 4-way)，每chiplet 2MB L1.5 write-through cache (128B line, 16-way, 仅缓存remote data)，全局8MB LLC write-back cache (128B line, 16-way)，L1.5+LLC总容量16MB。Inter-chiplet interconnect 768GB/s、32 cycles/hop，concentrated hierarchical crossbar拓扑，各chiplet两两直连。DRAM 64 channels、3TB/s。采用first-touch page allocation (4KB pages)和distributed CTA scheduling。论文修改GPGPU-Sim：(1) 实现multi-chiplet cache hierarchy和inter-chiplet network模拟（基于BookSim 2.0 composable platform）；(2) 添加sync-val directory在LLC中的查找/分配/eviction逻辑；(3) 添加每个chiplet的AMU模块（merge table CAM/SRAM双端口并发、ALU合并计算、multicast unit广播响应）；(4) 集成AccelWattch power model + inter-chiplet 0.54 pJ/bit传输能量评估能耗。AMU的merge table通过Cadence Spectre仿真（TT corner, 25°C, 1GHz, 1.1V, TSMC 40nm）评估功耗。

- 开源情况。模拟器如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：LRM-GPU的GPGPU-Sim扩展和AMU RTL均未开源。GPGPU-Sim和BookSim 2.0为开源项目。使用流程（基于论文描述）：
  1. Multi-chiplet system构建：配置chiplet数、每chiplet SM数/cache大小/cache policy → 设计每chiplet内部NoC（concentrated hierarchical crossbar）→ 通过BookSim 2.0 platform集成各chiplet network为composable inter-chiplet interconnect → 设置inter-chiplet bandwidth/latency
  2. LRM-GPU硬件配置：设置sync-val directory参数（entries/chiplet, owner bit-width）→ 设置AMU参数（channels, merge table entries/banks, SM list depth, timer值）→ 设置LLC directory eviction policy (LRU)
  3. 硬件验证流程（AMU design）：merge table → Cadence Virtuoso schematic/layout → TSMC 40nm netlist → Cadence Spectre仿真 → 测量power (185.51mW) 和 area (1.52mm²)；其他AMU组件 → Verilog RTL → Synopsys Design Compiler综合 (40nm) → power (115.93mW) 和 area (0.32mm²)；总AMU 301.44mW/1.84mm² → 对比NVIDIA V100 (300W/815mm², 12nm) → AMU占~0.1%功耗/~0.2%面积
  4. 系统级评估：运行同步workloads → 输出chiplet间synchronization performance → 通过energy breakdown分析inter-chiplet network/intra-chiplet network/cache/AMU能耗占比 → AMU仅占0.13%系统能耗
  5. Chiplet scale灵敏度：sweep chiplet数4→6→8（保持每chiplet资源不变）→ 观察accelerating ratio从1.33×降至1.21×（因为跨chiplet同步locality下降）
