## MSCCL (Microsoft Collective Communication Language / 微软集合通信语言)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MSCCL (Microsoft Collective Communication Language) 是微软开发的用于描述和优化 GPU 集合通信算法（collective communication）的领域特定语言（DSL）和编译器。与直接使用 NCCL 的低级 P2P API 相比，MSCCL 允许开发者以高层次 DSL 描述通信算法（如 All-to-All、All-Reduce 等），然后由编译器自动生成优化的 GPU kernel，支持 LL128 协议等高级特性。在 TUTEL 中，MSCCL 被用于编译优化 2DH All-to-All 算法，避免手写 NCCL P2P API 时的同步 barrier 开销和协议限制。

从编译框架角度拆解：

MSCCL 在 TUTEL 2DH All-to-All 中的使用流程：

```
[DSL Layer]
开发者用 MSCCL DSL 描述 2DH All-to-All 算法:
  - 4 个 phase (stride_memcpy × 2 + intra_node_a2a + inter_node_a2a)
  - 指定每个 phase 的 GPU 分组、数据布局、通信模式
  ↓

[Compiler Layer]
MSCCL Compiler:
  1. 解析 DSL → IR (Intermediate Representation)
  2. 分析数据依赖: phase 间的 barrier 可被消除或合并
  3. 协议选择: 根据 message size 自动选择
     - 小消息: LL128 协议 (低延迟)
     - 大消息: Simple 协议 (高带宽)
  4. 生成优化后的 CUDA kernel + scheduling hints
  ↓

[Runtime Layer]
Generated Kernel:
  - 消除 NCCL P2P API 中的不必要 barrier
  - 合并连续操作以减少 kernel launch overhead
  - 使用 LL128 协议: 128-byte 对齐的 RDMA write 操作
    利用 InfiniBand SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)
  ↓

[Hardware Execution]
GPU Stream 上执行:
  Phase 1 (stride memcpy) → Phase 2 (intra-node A2A with LL128)
  → Phase 3 (stride memcpy) → Phase 4 (inter-node A2A with RDMA)
```

与手写 NCCL P2P 版本的关键差异（Figure 19）：(1) MSCCL 编译版本消除了各 phase 间的显式 barrier，降低同步开销；(2) 自动选择 LL128 协议处理小消息，Simple 协议处理大消息；(3) 在 64 GPU、256 MiB 场景下，MSCCL 优化的 2DH 算法甚至优于 NCCL 的 Linear 算法。

术语一般如何实现？如何使用？

MSCCL 作为独立编译器项目（微软研究院，Cowan et al., ASPLOS 2023），接受 DSL 描述的通信算法并生成优化后的 NCCL-compatible kernel。TUTEL 将其作为 2DH All-to-All 的可选编译后端（默认使用 NCCL P2P API 版本，可通过 flag 启用 MSCCL 优化版本）。使用 LL128 协议需硬件支持 InfiniBand SHARP。

涉及论文标题：
- Tutel Adaptive Mixture-of-Experts at Scale
