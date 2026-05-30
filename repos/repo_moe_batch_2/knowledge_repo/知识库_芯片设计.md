## Hybrid Bonding DRAM for 3D Near-Memory Processing

术语解释
Hybrid Bonding DRAM 是通过 Cu-Cu 直接融合键合（direct fusion bonding）技术将 DRAM die 垂直堆叠在 logic die 之上的 3D 集成存储方案。与传统的 micro-bump 或 TSV（Through-Silicon Via）互连不同，Hybrid Bonding 使用超细间距（3µm pitch，110,000 I/O/mm²）的 Cu-Cu 直接键合，同时实现超高带宽和极高能效（~0.88 pJ/b），是 3D Near-Memory Processing 加速器的核心使能技术。

术语是什么？
Hybrid Bonding（混合键合）是一种先进 3D 芯片集成技术。它将两片晶圆或芯片面对面贴合，通过 Cu pad 之间的直接金属键合（Cu-Cu diffusion bonding）和周围介电层（SiO₂、SiCN 等）之间的共价键合（oxide-oxide bonding）同时形成电气连接和机械连接。相比传统 3D 集成方式（micro-bump: ~40µm pitch, TSV: ~10µm pitch），Hybrid Bonding 的 3µm 级 pitch 使互连密度提升 1-2 个数量级，寄生电容极低（~fF 级），单 bit 传输能耗从 pJ 级降至 sub-pJ 级。HD-MoE 论文中引用的 Hybrid Bonding DRAM 方案包括 Samsung 和 SK-Hynix 的商用/概念产品，将 DRAM 存储器堆叠在逻辑计算 die 上，实现高带宽、高能效的近存计算。

从芯片设计角度拆解术语
在 3D NMP 芯片设计中，Hybrid Bonding DRAM 的物理组织和数据流：
1. **物理结构**：底部 logic die（包含 compute unit + NoC router + memory controller），顶部一个或多个 DRAM die（每个 die 包含多个 independent bank）。DRAM die 与 logic die 之间通过 Hybrid Bonding 的 Cu pad 阵列连接（每 bank 对应 1024 个 I/O pin at 3µm pitch）。
2. **Memory Controller 开销**：驱动 1024 个 Hybrid Bonding I/O pin 需要大量 memory controller 逻辑。在 40nm 工艺下，controller 占 logic die 约 40% 面积。这是在片上面积、I/O 带宽和计算密度之间的核心 trade-off。
3. **Bank-Level Parallelism**：每个 DRAM bank 可被独立访问，logic die 上的 compute unit 可同时从不同 bank 读取 expert 权重。这是 3D NMP 能提供高有效带宽的关键。
4. **与 HBM 对比**：HBM（2.5D 集成，interposer + micro-bump）功耗高（TSV + interposer 走线的寄生效应），不适合 edge 设备；Hybrid Bonding 的 3D 堆叠减少走线长度和寄生电容，适合功耗敏感的 edge inference。

术语一般如何实现？如何使用？
商用进展：Samsung 在 2020 IEDM 展示 stacked eDRAM with Hybrid Bonding（34 GB/s/1Gb, 0.88 pJ/b logic-to-memory interface）；2022 ISSCC 展示 184 QPS/W 3D Logic-to-DRAM Hybrid Bonding for recommendation。SK-Hynix 提出基于 Hybrid Bonding 的 AiM 系列产品。HD-MoE 论文在模拟 3D NMP 时假设了 Hybrid Bonding DRAM 的高带宽特性（25-75 GB/s per NoC link，等效于 bank-to-compute 带宽），但未直接模拟 Hybrid Bonding 的物理实现细节。

涉及论文标题：
- HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing
