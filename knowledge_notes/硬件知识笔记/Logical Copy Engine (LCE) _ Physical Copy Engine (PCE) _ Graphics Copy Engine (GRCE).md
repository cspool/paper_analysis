## Logical Copy Engine (LCE) / Physical Copy Engine (PCE) / Graphics Copy Engine (GRCE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

这是 NVIDIA GPU 中 copy engine 的三层抽象结构，从外到内依次为：

- **LCE (Logical Copy Engine)**：软件可见的 copy engine 抽象。CUDA 报告的 copy engine 数量即 LCE 数量。每个 LCE 在 GPU 拓扑中被分配独立的 runlist ID（如 LCE2→Runlist 5, LCE3→Runlist 6），可被独立调度。

- **GRCE (Graphics Copy Engine)**：带有 graphics 相关能力的 copy engine 变体。GRCE 被编号为 Copy Engine 0 和 Copy Engine 1，与 Graphics/Compute Engine 一起绑定在 Runlist 0 上。GRCE 的特殊之处在于它可以共享一个已与另一个 GRCE 或 LCE 关联的 PCE（R8），这打破了 copy engine 之间的调度独立性。

- **PCE (Physical Copy Engine)**：硬件层面实际执行 copy 操作的单元。PCE 是真正执行 DMA 传输的硬件——LCE 和 GRCE 只是软件/逻辑层面的抽象，它们依赖 PCE 来实际移动数据。硬件寄存器控制 LCE→PCE 和 GRCE→PCE 的映射关系。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

LCE/PCE/GRCE 的映射机制（基于 nvdebug 的 lce_for_pce 和 shared_lce_for_grce 接口，Fig.11）：

```
GPU Hardware Register Constraints:
  ① 每个 PCE 最多关联一个 LCE 或 GRCE
  ② 只有 GRCE 可以共享一个已关联另一个 GRCE 或 LCE 的 PCE

示例配置对比：

GTX 1060 3GB (Fig.11): 
  PCE0 → GRCE0 (独占)
  PCE1 → GRCE1 (独占)
  → GRCE 不共享 PCE → GRCE 和独立 LCE 之间无干扰

RTX 6000 Ada (Fig.11):
  PCE0 → 共享: GRCE0 + LCE2
  或: PCE0 → GRCE0 且 GRCE0 共享 → LCE2
  → GRCE 共享了独立 LCE 的 PCE → OpenGL texture upload (使用GRCE) 
    干扰 CUDA GPU→CPU copy (使用被共享PCE的LCE)，copy减速约2×(Fig.10)
```

干扰机制（Fig.10 实验）：在 RTX 6000 Ada 上同时运行 CUDA GPU→CPU copy 和 OpenGL texture upload → 虽然二者使用不同的 LCE 和不同的 runlist，但由于 GRCE（处理 texture upload）共享了 LCE（处理 CUDA copy）的底层 PCE → 物理 copy 资源（PCE）被竞争 → copy 时间翻倍。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

NVIDIA 专利 US 10,275,275（"Managing copy operations in complex processor topologies"）描述了 PCE/LCE/GRCE 的映射管理机制。开发者通常不需要直接管理这些映射——但实时系统开发者需要注意：CUDA 报告的 copy engine 数量（即 LCE 数量）不一定等于可用的独立 copy path 数量。在需要 copy 隔离的场景中，应通过 nvdebug 的 lce_for_pce 和 shared_lce_for_grce 接口检查实际的 PCE 共享配置，避免因 GRCE 的 PCE 共享导致不可预期的 copy 性能退化。

涉及论文标题：
- Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management
