## Near-Memory Processing (NMP) Logic Die (近存计算逻辑芯片)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Near-Memory Processing (NMP) Logic Die 是 Stratum 提出的与 Mono3D DRAM die 通过 hybrid bonding 直接集成的专用计算芯片，用于在数据存储的物理近旁执行 LLM 推理计算（expert FFN, attention），避免将大量数据通过 interposer 传输到 xPU。与 PIM（Processing In Memory，在 DRAM die 内嵌入计算逻辑）不同，NMP 将计算逻辑放在独立的 logic die 上，通过 hybrid bonding 的全芯片面积高带宽接口访问 DRAM 数据。这避免了 DRAM 工艺对逻辑实现的约束（DRAM process 针对存储优化，逻辑实现 PPA 差）和散热问题。Stratum NMP Logic Die 规格：7nm 工艺，0.7V 供电，121mm² die area，16 PUs，128 TFLOPS peak（64k MACs @ 1GHz），36MB SRAM，43W power budget。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Stratum NMP Logic Die 的三级硬件架构：
1. **Chip Level**：16 Processing Units (PUs) 通过 bidirectional ring network 互联，每 PU 对应一个 Mono3D DRAM channel（16 channels total）。NMP mode 和 regular memory mode 独立运作，ring network 仅在 NMP mode 激活。
2. **PU Level**（每 PU）：
   - 16 Processing Elements (PEs) cluster，每个 PE 对应一个 DRAM bank
   - 1.25 MB shared memory（SRAM）
   - 256-way SIMD Special Function Engine（Softmax, SiLU, GeLU, 含 vector/scalar register files）
   - Ring Router（含 local switch + aggregator for in-situ data reduction）
   - Intra-channel Reducer（parallel reduction trees for partial sum aggregation）
3. **PE Level**（每 PE）：
   - 16×16 MAC Tensor Core（k-tap dot-product engines + double-buffered accumulators）
   - 64 KB psum SRAM（double-buffered）
   - Matrix Register File
   - Programmable Tiering Table（16×16b registers for 8-tier tRCD lookup）
   - Local Memory Controller（directly interfaces DRAM bank, translates row addresses to tier IDs）
   - 8KB Row Swap Buffer（for tier-to-tier expert migration）

NMP 模式执行流程：xPU streaming inputs to reserved DRAM rows → switch to NMP mode → PU accesses local DRAM channel via hybrid bonding → PE tensor cores execute GeMM/GeMV → Special Function Engine handles activation → results write back to designated DRAM space → exit NMP mode → xPU retrieves results。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Stratum NMP 使用 SystemVerilog 实现，Cadence Genus 在 ASAP7 7nm PDK 上综合。SRAM 通过 FinCACTI 建模（校准于公开 7nm SRAM spec）。面积 budget = 82mm²（逻辑 die 总面积 121mm² 减去 PHY, peripheral, TSV for power delivery）。实际 consumed area = 76.63mm²（63% utilization）。功耗 budget = 45W（由 HotSpot 3D IC thermal simulation 确定），实际 peak power = 42.67W。NMP 与 GPU 的分工：xPU 负责 prefill phase（compute-intensive）和 gating/routing（轻量操作），NMP 负责 decode phase（memory-bound expert computation + attention）。这种分工基于算术强度分析——decode 阶段的 attention 和 expert GEMM 都是 memory-bound，NMP 的 19-34 TB/s 内部带宽相比 xPU 的 HBM（~800 GB/s）有数量级优势。

涉及论文标题：
- Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving
