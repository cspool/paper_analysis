## GPU Resource Constraints (Thread/Shared Memory/Register Limits)

术语是什么？
GPU 资源约束指的是每个 SM 上有限的硬件资源对同时驻留的 thread block 数量和配置的限制。在 TX2 上，关键约束为：每 SM 最多 2048 线程、每 SM 最多 64KB shared memory、每 SM 最多 65536 寄存器（256KB register file）、每 block 最多 1024 线程、每 block 最多 48KB shared memory、每线程最多 255 寄存器、每 block 最多 32768 寄存器。

从硬件架构角度拆解术语：
这些资源约束直接决定 kernel dispatch 的可行性——GPU scheduler 在将 EE queue 头部 kernel 的 block 分配到 SM 时，必须逐项检查（Rule R1-R3）。例如：一个 768 threads/block 的 kernel，每 SM 最多分配 floor(2048/768) = 2 个 block（若其他 kernel 的 block 也占用线程，则更少）。32KB shared memory/block 的 kernel 每 SM 最多 2 个 block。资源约束的交互效应可能导致非 work-conserving 行为：即使一个 SM 有足够的线程资源，若 shared memory 不足，block 也无法分配。

术语一般如何实现？如何使用？
这些约束由 GPU 硬件强制执行，CUDA 运行时在 kernel launch 时验证配置合法性。开发者可通过 cudaOccupancyMaxPotentialBlockSize 等 API 查询最优配置。在实时系统中，资源约束导致的 blocking delay 是 schedulability analysis 的关键因素——论文正是通过量化这些约束来推导完整的调度规则。

涉及论文标题：
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

---
