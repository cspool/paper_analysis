## Pattern Memory Unit (PMU / 模式存储单元)

术语是什么？
Pattern Memory Unit (PMU) 是 SN40L RDU 中的片上 scratchpad 存储单元，每个 RDU 含 1040 个 PMU，合计 520 MiB 片上 SRAM。PMU 不是 cache（无硬件缓存一致性协议），而是完全由程序员/编译器管理的 scratchpad memory，组织为 SRAM banks 阵列。核心组件：(1) Scratchpad banks — 支持并发读写；(2) Scalar ALU Pipeline — 多级整数 ALU 可配置为独立并发的读/写地址生成流水线，支持 bitfield extraction、shift-and-set 等复杂地址指令，可接收 scalar RDN 输入实现 address composability（复杂地址计算跨多 PMU 分解）；(3) Address Predication and Banking — 可编程地址范围过滤（predicate bit per address）+ 可编程 bank bits 实现 tensor interleaving 和 bank conflict 消除；(4) Data Alignment Unit — 支持 transpose（diagonally striped write → normal read at full bandwidth）、cross-lane vector permute、vector-unaligned access、lookup table (LUT)、data format/layout 转换。

从硬件架构角度拆解：
PMU 在 streaming dataflow 中的三种角色：
- Stage Buffer: 存储 producer PCU 输出 → consumer PCU 输入（如 Figure 4 中的 I0, S0-S3）
- Capacity Scaling: 单个逻辑 tensor 跨多个 PMU partition（如 T00-T03, T10-T13，通过 address predication 实现）
- Bandwidth Scaling: 单个逻辑 tensor 跨多个 PMU（如 I00-I01），增加并行读写带宽
- Access Pattern Optimization: Transpose 通过 data alignment unit 的 diagonally striped write 实现 in-place，无需数据搬移

PMU ALU pipeline 的关键洞察：写和读的 access pattern 复杂度互为 trade-off（复杂写 → 简单读，反之亦然），PMU ALU pipeline 可分区为独立的读/写地址生成链来利用此特性。

术语一般如何实现？如何使用？
编译器负责：为每个 on-chip tensor 分配 PMU 数量和 partition 方案；编程 PMU ALU pipeline 生成读写地址；配置 bank bits 避免多 buffer 场景的 bank conflict；选择 data alignment unit 操作（如 transpose = diagonally striped write）。用户层面无感知，编写标准 PyTorch 模型即可。

涉及论文标题：
- SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts
