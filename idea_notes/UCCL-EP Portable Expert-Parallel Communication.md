## UCCL-EP Portable Expert-Parallel Communication

- baseline方法是什么？
  Baseline 是 **DeepEP**（DeepSeek, 2025），基于 NVIDIA IBGDA（InfiniBand GPUDirect Async）的 GPU-initiated token-level EP 通信系统。全栈执行例子：
  - **模型推理算法层**：MoE layer 中 gating network 为每个 token 选择 top-K experts，需要执行 dispatch（token activations 发送到 expert GPUs）和 combine（expert outputs 收集回原 GPU）。
  - **系统框架层**：DeepEP 被集成到 SGLang、vLLM、Megatron-LM 等训练/推理框架中，通过 NVSHMEM 提供 symmetric memory 和 GPU-initiated one-sided RDMA 操作。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：GPU threads 直接通过 IBGDA 接口向 NVIDIA ConnectX NIC 的 MMIO doorbell/register 写入 RDMA work requests，实现 token-level fine-grained 通信。LL mode 为 per-token immediate send，HT mode 使用多 ring buffer 实现 token deduplication + intra-node forwarding + hierarchical reduce。GPU kernel 假设 NIC 提供严格的 ordering 保证（如 write-then-atomic 语义），所有 transfer 完全 bypass CPU。
  - **硬件架构层**：NVIDIA GPU（H100/H200/B200）+ NVIDIA ConnectX-7 InfiniBand NICs（400G）。IBGDA 要求 GPU 能直接写入 NIC driver 定义的 MMIO 接口，因此仅支持 NVIDIA GPU + NVIDIA/Mellanox NIC 组合。

  Baseline（DeepEP）的核心缺陷：
  (1) **GPU-NIC 紧耦合导致可移植性差**：IBGDA 要求 GPU 直接操作 NIC MMIO doorbell/register，这意味着需要分别为每一种 (GPU vendor, NIC vendor) 组合编写集成代码。假设 m 种 GPU, n 种 NIC，需 O(m×n) 开发工作量。实际结果是 DeepEP 官方仅支持 NVIDIA GPU + NVIDIA NICs。
  (2) **GPU 对 NIC delivery semantics 的刚性假设**：GPU kernel 假设下层的 NIC 提供 strict ordering（如 write-then-atomic），但许多 NIC（如 AWS EFA SRD 协议）不保证 ordering，导致 DeepEP 无法在此类 NIC 上正确运行。
  (3) **GPU 缺少灵活的网络管理能力**：GPU threads 难以实现 congestion control、flow control、failure recovery 等高级网络策略。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **UCCL-EP**，通过将"通信发起"与"通信执行"解耦来实现可移植的高性能 EP 通信。全栈执行例子：
  - **模型推理算法层**：保持 MoE dispatch/combine 计算逻辑不变（数学等价），GPU 仍负责 token-level routing decisions 和 fine-grained communication initiation。
  - **系统框架层**：UCCL-EP 作为 DeepEP 的 API-compatible drop-in replacement，无需修改上层框架代码。移除 NVSHMEM 依赖，通过 CPU proxy 管理 symmetric memory。支持 SGLang（推理）和 AMD Primus/Megatron-LM（训练）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：核心创新——GPU-CPU 解耦通信架构：
    - **(a) Lock-free GPU-CPU FIFO channel**：GPU threads 将 token routing 决策编码为 128-bit TransferCmd（Write/Atomics/Drain/Barrier），通过 shared memory FIFO 队列传递给 CPU proxy。GPU 侧缓存 tail index，CPU 和 GPU 分别将各自常用元数据置于本地内存侧，减少 PCIe 穿越。多 FIFO channels 映射减少 GPU SM 竞争。
    - **(b) Multi-threaded CPU proxy**：每个 GPU 分配一个 CPU proxy（4 threads），通过 libibverbs（Linux 可移植 RDMA 库）发出 GPUDirect RDMA 操作。CPU proxy 负责 QP load balancing、多 NIC bandwidth aggregation、connection management。
    - **(c) Ordering emulation via immediate data**：发送端在每个 RDMA write 的 immediate data（32-bit RoCEv2 标准包头字段）中嵌入 sequence number + expert index。接收端 CPU proxy 从 CQ 提取 immediate data，若消息 out-of-order 到达，将 atomic 暂存于 control buffer，待所有 prior writes 完成后再有序 apply。实现了 partial ordering（per-channel）而非全局 ordering，避免硬件成本。
    - **(d) Software atomics on EFA**：在 EFA 等不支持硬件 RDMA atomics 的 NIC 上，CPU proxy 通过 immediate data write + host memory counter update 模拟 atomics，GPU 直接读取 host-allocated memory（cudaMallocHost）用于 control decisions。
    - **(e) AMD GPU porting**：将 CUDA PTX intrinsics → ROCm alternatives；warp (WARP_SIZE=32) → wavefront (WAVEFRONT_SIZE=64)；TMA-based copy → CU-based copy；merge coordinator wavefronts into receiver wavefronts。
  - **硬件架构层**：支持 NVIDIA GPU（H100/H200/B200/GH200）+ AWS EFA NICs（SRD unordered transport）、NVIDIA GPU + ConnectX-7 IB、AMD MI300X + ConnectX-7 IB、AMD MI300X + Broadcom Thor-2 等多种异构组合。仅需 O(m) 移植工作（GPU kernel 变化），CPU-NIC 侧通过 libibverbs 可移植层自动适配。

  **设计思路核心映射**：
  - 缺陷(1) "GPU-NIC 紧耦合 O(m×n) 移植成本" → 方案：解耦 GPU 通信发起与 CPU 通信执行，CPU 通过 libibverbs 可移植层适配任意 NIC → O(m) 移植成本（从 AMD GPU 到 Broadcom NIC 仅需修改 GPU kernel，CPU 侧无需额外适配代码）
  - 缺陷(2) "GPU 对 NIC ordering 的刚性假设" → 方案：CPU proxy 使用 RDMA immediate data 嵌入 sequence number + control buffer 延迟 apply → 在 EFA（无序传输）上正确运行，无需 NIC 硬件支持 ordering
  - 缺陷(3) "GPU 缺少网络管理灵活性" → 方案：CPU proxy 可实现 congestion control（控制 kMaxInflight 限制 in-flight 消息数）、多 QP 负载均衡、failure recovery（elastic EP）等策略
  - 最终效果：UCCL-EP 在 EFA 上 dispatch/combine 吞吐量超过最佳现有方案（PPLX）最多 2.1×；在 NVIDIA-only 平台上性能与 DeepEP 原版可比（HT mode dispatch 延迟差异 <5%）；SGLang 推理吞吐提升最多 40%（NV_EFA3），Megatron-LM 训练吞吐提升最多 45%（AMD_BRC）。
