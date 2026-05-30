## NVIDIA Jetson TX2

术语是什么？
NVIDIA Jetson TX2 是 NVIDIA 于 2017 年发布的嵌入式 AI 计算平台，属于 Jetson 系列。采用 SoC 设计：四核 2.0GHz 64-bit ARMv8 A57 + 双核 2.0GHz superscalar ARMv8 Denver + 集成 Pascal GPU（2 SM、256 CUDA Core @ 1.3GHz）。8GB LPDDR4 共享 DRAM，约 600 美元/板。面向"自主一切"（autonomous everything）市场，包括自动驾驶、机器人、无人机等。

从硬件架构角度拆解术语：
TX2 的架构反映了嵌入式自主系统的关键需求：计算能力与 SWaP 的平衡。其 Pascal GPU 支持 CUDA 8.0+，具有本文发现的关键调度特性——层次化 FIFO 调度、2 个 EE queue（priority-high/low）、1 个 CE、指令级 Compute Preemption（Pascal 新特性）。GPU 和 CPU 共享 DRAM 避免了 PCIe 传输延迟。2MB L2 Cache 分为两个（A57 集群和 Denver 集群各 1MB），GPU 另有 512KB L2 Cache。TX2 支持 Max-Q（7.5W）和 Max-P（15W）两种功耗模式。

术语一般如何实现？如何使用？
TX2 运行 Linux for Tegra (L4T)，支持标准 CUDA 开发环境。通过 JetPack SDK 提供 CUDA、cuDNN、TensorRT、VisionWorks 等库。开发者使用与传统桌面 GPU 相同的 CUDA 编程模型。本文的研究方法（合成 benchmark + globaltimer 时间戳测量）具有通用性，可扩展到其他 NVIDIA GPU 架构。开源代码见 https://github.com/yalue/cuda_scheduling_examiner_mirror。

涉及论文标题：
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

---
