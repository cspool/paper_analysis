## DualPipe Pipeline Schedule

术语是什么？

DualPipe 是 DeepSeek-V3 (Liu et al., 2024b) 提出的双向流水线并行调度算法，用于大规模模型训练中重叠计算和通信。核心特点是从 pipeline 两端同时注入 micro-batch（forward 和 backward），使前向和后向计算在时间上交错，最大化计算和通信的重叠率。

标准 1F1B（One-Forward-One-Backward）流水线并行存在 bubble（空闲等待时间），而 DualPipe 通过双向调度将 bubble 填充为反向计算或通信操作。DualPipe 与 DeepSeek-V3 的整体训练架构紧密集成，包括 expert parallelism 的 all-to-all 通信和 ZeRO-1 数据并行的梯度通信。

在 mHC 中，DualPipe 被扩展以处理 n-stream 残差引入的额外 pipeline stage 边界通信（n 倍于标准残差连接的通信量）和重计算开销。

从系统架构角度拆解：

DualPipe 在 mHC 中的扩展调度：
```
Timeline (simplified):
Stage 0 | F_A0 F_M0 | comm | B_M0 B_A0 | W | ...
Stage 1 |   | F_A1 F_M1 | comm | B_M1 B_A1 | W | ...
Stage 2 |     | F_A2 F_M2 | comm | B_M2 B_A2 | W | ...

Key extensions for mHC:
1. MLP (FFN) F_post,res kernel on dedicated high-priority compute stream
   → allows preemption by communication, reducing stalls
2. No persistent kernels in attention layers
   → prevents long-running ops from blocking scheduling
3. Recompute blocks aligned with pipeline stage boundaries
   → x_{l_0} (first layer input) already locally cached, no extra communication
4. Communication (nC elements per boundary) overlapped with compute
```

术语一般如何实现？如何使用？

DualPipe 要求 pipeline stage 数量可被 2 整除，且需要细粒度的 GPU stream 管理（通信 stream + 高优先级 compute stream）。在 mHC 中通过 CUDA stream 优先级实现 preemption——MLP 的 $\mathcal{F}_{post,res}$ kernel 放入高优先级 stream，能在通信到达时被抢占，确保通信不阻塞。Attention 层避免 persistent kernel 以防止长时间占用 SM 导致调度僵化。

涉及论文标题：
- mHC Manifold-Constrained Hyper-Connections

---
