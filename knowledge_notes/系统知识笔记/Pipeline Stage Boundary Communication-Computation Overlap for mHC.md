## Pipeline Stage Boundary Communication-Computation Overlap for mHC

术语是什么？

在 mHC 的流水线并行中，pipeline stage 边界需要传输 n-stream 残差状态（nC 元素，n 倍于标准残差连接）。为消除这一额外通信延迟，mHC 扩展 DualPipe schedule 实现了细粒度的通信-计算重叠机制。

两个关键技术：(1) MLP 层（FFN）的 $\mathcal{F}_{post,res}$ kernel 在专用高优先级 CUDA stream 上执行，能在通信到达时被抢占，避免了传统 persistent kernel 长时间阻塞 SM 的问题；(2) Attention 层避免使用 persistent kernel，防止长时间运行操作阻塞调度器。重计算过程与 pipeline 通信解耦，因为每 stage 的首层输入 $\mathbf{x}_{l_0}$ 在本地已有缓存。

从系统架构角度拆解：

```
Pipeline Stage Boundary Timeline:
... | F_M_{L-1} | {F_post,res + comm overlap} | comm (nC) | ...
                     |                              |
                     v                              v
          F_post,res on HP stream          x_{l_0} already cached
          (preemptible by comm)            (no extra recv needed)

Traditional DualPipe without mHC extension:
... | F_M_{L-1} | wait_comm | comm (C) | F_post,res | ...

mHC extension benefit:
- HP stream: F_post,res yields to communication → no stall
- Recompute decoupled: no dependency on recv'd data
```

术语一般如何实现？如何使用？

CUDA stream 优先级管理通过 `cudaStreamCreateWithPriority` 实现。高优先级 stream 上的 kernel 能在低优先级 stream 的 kernel 执行间隙被调度。mHC 限制 persistent kernel 的使用范围（如仅用于 attention 计算的特定阶段），使得 GPU 调度器有充足机会插入通信 kernel。重计算因 $\mathbf{x}_{l_0}$ 本地缓存而完全免除 pipeline 通信依赖。

涉及论文标题：
- mHC Manifold-Constrained Hyper-Connections
