## FPGA Overlay Processor for LLM（面向LLM的FPGA覆盖层处理器）

术语是什么？通过联网搜索让回答具体和精准。
FPGA Overlay Processor（覆盖层处理器）是一种在FPGA可编程逻辑上构建的专用处理器架构，针对特定计算workload（如transformer推理）进行高度优化。与传统FPGA加速器不同，Overlay Processor在FPGA上实现完整可编程处理器，包含定制PE array、on-chip buffer hierarchy、专用指令集和dataflow控制器。DFVG所属的OPU系列（MCoreOPU、ChatOPU、SkipOPU、MoE-OPU）均来自同一团队（Lei He group, SJTU/EIT），DFVG首次将Overlay Processor用于speculative decoding的draft stage。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
DFVG的Overlay Processor硬件架构（V80 FPGA, 300MHz）：
```
┌─────────────────────────────────────────┐
│        Multi Compute Core Overlay        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ Core 0   │ │ Core 1   │ │ Core N   │ │
│  │ PE×PE×.. │ │ PE×PE×.. │ │ PE×PE×.. │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ │
│       └──────────────┼──────────────┘    │
│            Parallel Adder Tree           │
│        Output Buffer + SFU (softmax,etc) │
└─────────────────────────────────────────┘
    ↑ HBM channels (64 AXI ports)
    Weight Buffer | KV Cache | DMA | PCIe Gen4
```
PE微架构特征：(1) multi-weight buffer + 额外连线支持branch concatenation——draft model多分支时可选择正确weight path；(2) DSP packing——单DSP48执行双BF16×BF16乘法，计算吞吐翻倍；(3) ping-pong data loading——KERload = PE_Num×Data_width/Bandwidth，IFMload = KERload + CAS_Latency，computation与data loading重叠，矩阵乘法loading和computation均达86.2%-97.5%效率。KV-Cache管理：on-chip temp buffer按branch临时存储KV→verified后prune无效分支→accepted tokens通过contiguous allocation最大化利用on-chip RAM→block-based批量eviction。V80资源占用：89.6% LUT, 90.9% FF, 8192 DSPs, 18MB BRAM, 67MB URAM（主要KV-cache），运行时功耗仅75W。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
OPU系列演进：OPU (TVLSI'20, CNN) → Light-OPU (FPGA'20, edge) → MCoreOPU (TRETS'25, transformer) → ChatOPU (ICCAD'25, unstructured sparsity) → SkipOPU (2026, dynamic computation allocation) → MoE-OPU (2026, expert parallelism) → DFVG (ASPLOS'26, heterogeneous speculative decoding)。DFVG使用Verilog HDL→Vivado 2024.1综合→bitstream。开源：https://github.com/ShaoqiangLu/DFVG。

涉及论文标题：
- DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU

