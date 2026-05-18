## HBM Pseudo Channel (PC)

术语是什么？

Pseudo Channel (PC) 是 HBM 独有的通道细分机制。每个 HBM channel 包含两个 PC，它们共享 C/A pins 但平分 data pins（如 HBM4 每个 64-bit channel 中每个 PC 占 32-bit data pins）。两个 PC 可独立运行，实现并发数据传输以最大化吞吐量。从 HBM4 开始，带宽扩展主要通过加倍 PC 数量（进而加倍 I/O）实现，而非改变 per-channel width。在 HBM 内部组织中，两个 PC 在 logic die 上有各自的 GBUS controller，各自独立接收命令和数据。

从芯片设计角度拆解术语：

PC 设计的芯片动机是在不增加 DRAM core frequency 和不增加 AGbank 的情况下提升有效带宽。每个 PC 对应一组 data pins（DQ），两 PC 共享同一组 C/A pins（row C/A + column C/A）。MC 向不同 PC 交错发命令以保持两 PC 的 data pins 同时忙碌。但 PC 数量增加会同步增加独立 C/A pin 数量——每个 PC 虽然共享 C/A pins，但不同 channel 的 PC 有独立 C/A pins。在 HBM4 中，32 channels × 2 PCs = 64 PCs，每 channel 需 18 C/A pins（10 row + 8 column），C/A-to-DQ pin ratio 随代际上升（从 HBM1 到 HBM4 近乎翻倍）。

术语一般如何实现？如何使用？

PC 在 JEDEC HBM 标准中定义，由 HBM vendor（SK hynix、Samsung）在硅片中实现。MC 需跟踪所有 PC 的 bank 状态并做 PC interleaving 调度。通常 MC 会将请求交替映射到同一 channel 的两个 PC 上以达到满带宽。RoMe 论文中，两个 PC 工作在 "legacy channel mode"（类似 HBM1/2），即命令同时发往两 PC、数据同时从两 PC 接收，从而从 MC 视角将两个 PC 视为一个 channel，消除 PC interleaving 调度需求。

涉及论文标题：
- RoMe: Row Granularity Access Memory System for Large Language Models
