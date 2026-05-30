## CMOS-Under-Array (CUA / CMOS 阵列下置)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CMOS-Under-Array (CUA) 是一种 3D DRAM 结构设计范式，将 DRAM 的外围电路（sense amplifier, row decoder, DQ buffer 等）放置在 DRAM 存储阵列的正下方，而非传统 2D DRAM 的阵列外围。这样可以：(1) 减少芯片面积——外围电路不占据平面面积；(2) 缩短存储单元到外围电路的走线长度——减少 RC 延迟；(3) 支持更小的 bank 粒度——每个 bank 下方可放置独立的 peripheral。Stratum 中采用 CUA 结构：高电压 DRAM 外围电路在 32nm CMOS 工艺中实现并置于存储阵列下方，低电压逻辑电路在独立的 7nm logic die 上通过 hybrid bonding 连接。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
Stratum 的 CUA + Hybrid Bonding 分层架构：
```
Mono3D DRAM Die (1024 layers of 1T1C cells)
  ├─ WL Staircase (垂直路由)
  ├─ BL (垂直连接)
  └─ 底部: HV Peripheral Circuit (32nm CUA)
       └─ Cu-Cu Hybrid Bonding Pads (1μm pitch)
Logic Die (7nm, 121mm²)
  ├─ LV Logic (PE array, ring network, SRAM, special function engine)
  └─ TSV for Power Delivery (to interposer)
Silicon Interposer
  └─ xPU Die (H100 or RTX A6000)
```
CUA 的关键优势：DRAM 外围电路不占用 logic die 的面积 budget，使 logic die 的 82mm² 可用面积全部分配给 NMP 计算资源（PE arrays, SRAM, ring network）。HV 外围电路使用成熟的 32nm 节点（低成本、高可靠性），LV 逻辑使用先进 7nm 节点（高性能、高密度），实现了工艺节点的异构组合。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CUA 概念源于 3D NAND Flash（如 Samsung V-NAND 的 Peri Under Cell, PUC 结构），在 3D DRAM 中属于新兴方向。制造流程：先在逻辑 wafer 上制作 LV 逻辑层 → 再在上方层沉积和刻蚀构建 HV 外围电路（CUA 层）→ 最后逐层构建 1T1C DRAM 存储阵列 → 通过 hybrid bonding 连接底部逻辑 die。当前 3D DRAM 的 CUA 仍处于研究阶段（如 Samsung 3D DRAM 的 VLSI 2023 论文），尚未有商业产品。

涉及论文标题：
- Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving
