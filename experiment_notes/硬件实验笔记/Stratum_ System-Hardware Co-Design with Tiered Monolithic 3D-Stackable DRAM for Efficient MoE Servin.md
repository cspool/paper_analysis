## Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving

- 属于硬件架构的实现是什么？实验比较什么？
  - 实现：Stratum 的硬件架构核心是 **NMP (Near-Memory Processor) Logic Die**，与 Mono3D DRAM die 通过 Cu-Cu hybrid bonding 垂直集成。架构分为三个层次：(1) **Chip Level**——16 个 Processing Units (PUs)，每个 PU 对应一个 Mono3D DRAM channel（16 channels total），通过 bidirectional ring-based on-chip network 互联。NMP 模式和 regular memory 模式独立运作，ring network 仅在 NMP mode 激活。(2) **Channel/PU Level**——每 PU 包含 16 个 near-bank Processing Elements (PEs) cluster、1.25 MB shared memory、256-way SIMD special function engine（支持 Softmax/SiLU/GeLU 等非线性算子，含 vector/scalar register files 和 arithmetic units）、ring router（含 local switch + aggregator 用于 in-situ data reduction）、intra-channel reducer（parallel reduction trees）。(3) **Bank/PE Level**——每 PE 含 16×16 MAC tensor core（k-tap dot-product engines + local accumulators）、64 KB psum SRAM（double-buffered）、matrix register file、programmable tiering table（16×16b registers，存储 8-tier 的末层行地址和 tRCD 值），local memory controller（含 8KB row swap buffer 用于 tier-to-tier expert 迁移而无须外部数据获取）。
  - 实验比较：(a) Logic die area breakdown（PE 主导，tiering table 仅 0.1% PE area 开销）——Figure 15a；(b) Power breakdown（DRAM 104W + Logic Die 42.67W, peak total 144.53W under 45W logic budget）——Figure 15b；(c) Mono3D DRAM latency vs WL layer（8 tiers, tRCD range 2.29-22.88ns, fast tier 1.6× faster）——Figure 14；(d) Tiering vs No-Tiering throughput（1.32-1.45× improvement across models）；(e) 512-layer tiering 效果——17.7-18.3% 性能提升；(f) 与 HBM-based NMP 对比——Duplex 2.2-3.0× throughput gap。

- 硬件平台（模拟器/工具链使用）：
  - SystemVerilog 硬件设计：Cadence Genus synthesis（7nm ASAP7 PDK），FinCACTI SRAM 建模（校准于公开 7nm SRAM spec）。
  - Mono3D DRAM device 仿真：Coventor SEMulator3D process simulator（RC 参数提取），NeuroSim V1.4（外围电路 timing/power 仿真，与 DDR5 标准校准）。
  - 3D IC 热仿真：HotSpot simulator（liquid cooling + vapor chamber heat sink 方案，convection resistance 0.01 W/K）。
  - 自研 in-house cycle-level simulator：执行 cycle/通信 cycle/能耗模拟。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - Cadence Genus Synthesis（商业）：https://www.cadence.com/en_US/home/tools/digital-design-and-signoff/synthesis/genus-synthesis-solution.html
  - ASAP7 PDK（开源）：https://github.com/The-OpenROAD-Project/asap7
  - FinCACTI（学术工具）：[73] Shafaei et al., ISVLSI 2014
  - Coventor SEMulator3D（商业/学术）：https://www.coventor.com/products/semulator3d/
  - NeuroSim V1.4（开源学术）：https://github.com/neurosim/NeuroSim
  - HotSpot（开源学术）：https://lava.cs.virginia.edu/HotSpot/

- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？至少具体到模拟器模拟性能的原理和模拟器输入到性能输出的全过程。
  - 开源情况：Stratum NMP 硬件设计未开源。使用的工具链为学术/商业标准工具。
  - 硬件设计→性能评估全流程：
    ```
    第1步：Mono3D DRAM 器件建模
      Input: Mono3D DRAM 结构规格（1024 layers, 35nm, BL/WL pitch 70nm/1μm）
      Coventor SEMulator3D: 构建 3D DRAM array 模型
        → 提取 WL/BL 寄生 RC 参数
        → 输出 per-layer access latency profile
    
    第2步：Bank 级性能仿真
      Input: Coventor RC 参数 + DDR5 standard timing
      NeuroSim V1.4: 模拟 bank 外围电路（sense amp, row decoder, etc.）
        → 输出各 tier 的 tRCD, tRP, tRAS, tRC（Table 1）
        → tRCD = [2.29, 3.92, 5.99, 8.50, 11.44, 14.82, 18.63, 22.88] ns
    
    第3步：Logic Die 处理器综合
      Input: SystemVerilog RTL (PE, PU, ring network, special function engine)
      Cadence Genus + ASAP7 7nm PDK: 综合 + 时序分析
        → 输出面积 breakdown: PE(dominant), SRAM, ring network, PHY, misc.
        → 输出关键路径延迟 → 确定 f_logic = 1 GHz
    
    第4步：功耗仿真
      Input: Post-synthesis netlist + random stimulus switching activity
      Cadence Genus power analysis: 
        → E_mac = 0.604 pJ/MAC operation
        → 输出 component-level power: PE, SRAM, ring, special function engine
    
    第5步：热仿真
      Input: DRAM power (104W max) + logic die power estimate
      HotSpot 3D IC: vapor chamber heat sink modeling
        → 输出: logic die power budget = 45W per chip
        → 验证: T_junction within safe operating range
    
    第6步：Cycle-Level 性能仿真
      Input: Workload (tensor sizes, tier assignments, routing), 
             Component timing (tRCD per tier, MAC latency, ring bandwidth)
      In-house simulator:
        → Compute cycles: GeMM tiling & scheduling on 16×16 MAC array
        → Communication cycles: ring network all-gather/reduce-scatter latency
        → Tier penalty: per-access tRCD based on tier assignment
        → Pipeline overlap modeling (GeMM2||Activation, Reduce-scatter||Next GeMM1)
        → 输出: total execution time, per-component energy breakdown
    
    第7步：System-Level Serving 仿真
      Input: Request trace (Poisson arrival, topic distribution)
      System-level simulator:
        → Scheduler: batch building + SLO enforcement
        → Memory mapper: Algorithm 1 expert placement
        → Computation mapper: prefill→xPU, decode→NMP
        → Latency/energy accumulation over entire serving period
        → 输出: decoding throughput (tokens/s), energy efficiency (tokens/J)
    
    PPA 约束（Eq. 1-3）:
      Power: P_dram + P_compute + P_misc ≤ 45W (logic die cap)
      Area:  A_PD + N_mac·A_mac + A_PHY + A_peri + A_misc ≤ 63% × 121mm²
      TSV:   A_PD derived from current delivery requirements (36mA/TSV, 25μm²/TSV)
    ```
