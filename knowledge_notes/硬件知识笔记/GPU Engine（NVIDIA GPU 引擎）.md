## GPU Engine（NVIDIA GPU 引擎）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

GPU Engine 是 NVIDIA GPU 内部的独立功能单元（functional unit），负责执行特定类型的 GPU 操作。现代 NVIDIA GPU 并非单一加速器，而是由多种不同类型 engine 组成的异构系统（Fig.2）。主要 engine 类型包括：(i) **Graphics/Compute Engine**——包含所有通用计算核心（SM/CUDA Core/Tensor Core），执行 CUDA kernel 和图形渲染；(ii) **Copy Engine**——负责 GPU DRAM ↔ CPU DRAM 及其他 GPU 之间的异步数据传输，现代 GPU 通常有多个 copy engine；(iii) **Video Decode Engine (NVDEC)**——硬件视频解码；(iv) **Video Encode Engine (NVENC)**——硬件视频编码；(v) **JPEG Decode Engine (NVJPG)**——硬件 JPEG 解码；(vi) **Optical Flow Accelerator (OFA)**——光流计算。所有 engine 通过内部 crossbar 总线连接到 GPU DRAM 和 PCIe 接口。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Engine 的硬件组织具有以下关键特性（基于 Bakita & Anderson 通过 nvdebug 的 device_info PTOP 寄存器发现的规则）：

1. **Engine-Runlist 绑定（R7）**：每个 engine 仅绑定到一个 runlist——这是由 PTOP 寄存器约束的硬件限制。这意味着一个 engine 的调度不会在多个 runlist 上被重复处理。

2. **Runlist 可绑定多 Engine（R6）**：反过来，一个 runlist 可绑定多个 engine。例如几乎所有 GPU 的 Runlist 0 都同时绑定 Graphics/Compute Engine 和 Graphics Copy Engine 0/1（GRCE0/GRCE1）。在 Jetson Orin 上，Runlist 2 同时绑定 Copy Engine 3 和 Graphics/Compute 1。

3. **Engine 拓扑示例**（RTX 6000 Ada，Table IV）：16 种不同类型的 engine 分布在 16 个 runlist 上——Runlist 0 绑定 3 个 engine（Compute + 2 GRCE），其余 Runlist 1-15 各仅绑定 1 个 engine。

4. **Copy Engine 的层次结构**：Copy Engine 在逻辑层分为 LCE (Logical Copy Engine) 和 GRCE (Graphics Copy Engine)，底层由 PCE (Physical Copy Engine) 实际执行。GRCE 可通过共享 PCE 干扰独立 runlist 上的 LCE（R8）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GPU Engine 的实现因类型而异：Compute Engine 包含大量可编程的 SM (Streaming Multiprocessor)，内部分为 thread block scheduler → warp scheduler → CUDA Core/Tensor Core 的层次调度管线。Copy Engine 是固定功能的 DMA 引擎，通过 LCE→PCE 的间接映射层实现灵活的 copy 能力分配。开发者通过 CUDA API（cudaMemcpyAsync 使用 copy engine，kernel launch 使用 compute engine）隐式选择 engine。实时系统开发者需要了解 engine 的独立性和潜在干扰——在需要隔离的场景中，可以通过 nvdebug 检查 engine-runlist 拓扑来配置正确的 mutual-exclusion lock。

涉及论文标题：
- Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management
