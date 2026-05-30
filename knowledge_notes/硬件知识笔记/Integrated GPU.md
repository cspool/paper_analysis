## Integrated GPU

术语是什么？
Integrated GPU（集成 GPU）是指与 CPU 集成在同一 SoC（System-on-Chip）上的 GPU，与 CPU 共享 DRAM 内存。与 discrete GPU（独立 GPU，有自己的专用显存）相比，集成 GPU 功耗更低（5-15W vs 150-250W），体积更小，适合嵌入式场景。TX2 的 GPU 是典型的集成 GPU。

从硬件架构角度拆解术语：
在 TX2 上，集成 GPU 与 6 核 CPU（4×A57 + 2×Denver）通过 SoC 内部总线共享 8GB LPDDR4 DRAM。这种统一内存架构避免了 discrete GPU 需要 PCIe 传输数据的开销——host 和 device 端的内存拷贝通过 Copy Engine 在共享 DRAM 内部进行。这使 CE 和 EE 的并发更加高效。但集成 GPU 的计算能力远低于 discrete GPU（TX2 仅 256 CUDA Core vs 桌面 Pascal GPU 的数千核心），因此充分利用可用的 GPU 容量尤为重要——这也是本文研究调度器行为的主要动机之一。

术语一般如何实现？如何使用？
集成 GPU 是嵌入式自主系统的 de facto 选择，因为满足 SWaP（Size, Weight, and Power）约束。NVIDIA Jetson 系列（TX1, TX2, Xavier, Orin）和 Intel 的集成显卡都是此类的代表。CUDA 编程中，集成 GPU 支持 Unified Memory（cudaMallocManaged），简化了 host/device 数据管理。在实时系统中，集成 GPU 的共享内存架构使得对 GPU 调度行为的精确理解更加关键，因为 CPU 和 GPU 任务竞争同一内存带宽。

涉及论文标题：
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

---
