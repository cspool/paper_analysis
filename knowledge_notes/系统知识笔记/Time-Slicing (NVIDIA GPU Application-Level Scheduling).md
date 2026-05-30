## Time-Slicing (NVIDIA GPU Application-Level Scheduling)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Time-Slicing 是 NVIDIA GPU 在多进程共享单 GPU 时默认的 application-level 调度机制（当 MPS 未启用时）。CUDA application-level scheduler 以固定时间片（约 2ms，本文在 Ampere RTX 3090 上证实）将 GPU 的全部计算资源（所有 SM、warp scheduler、执行核心）轮流分配给不同进程。在任意给定时刻，仅有一个进程的 kernel 在 GPU 上执行——不支持 spatial sharing（不同进程的 kernel 不会同时执行）。时间片大小和调度频率**不可配置**（除 Jetson 嵌入式平台外）。约 145μs 的 context switch 开销（save + restore）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Time-Slicing 的并发执行流程（两个 DL task 为例）：

```
Timeline (ms):  0        2        4        6        8       10
                |--------|--------|--------|--------|--------|
Process A:      [Train K0][Train K1]        [Train K2][Train K3]
Process B:                [Inf K0][Inf K1]            [Inf K2]
GPU Occupancy:   AAAAAAAAA BBBBBBBB AAAAAAAAA BBBBBBBB AAAAAAAAA
                ↑        ↑        ↑
                |~145μs switch|    |~145μs switch|

各时间片内：
  - 仅一个进程的所有 blocks 在 GPU 上
  - 空闲 SM 资源无法被另一进程使用（空间隔离）
  - Context switch 约 145μs（论文通过 global timer register 测量）
```

本文的三个关键发现：
- **(O2) Predictability vs Utilization trade-off**：Time-slicing 提供最 predictable 的 inference latency（因无 SM 内资源竞争），但 utilization 最差——training time 可比 baseline 多 100+s（如 ResNet/DenseNet）。
- **(O3) 资源共享限制**：尽管两进程不同时执行，但 register/shared memory/global memory 的**总和**需求仍不能超 GPU 硬件上限——否则第二进程 OOM 崩溃。推测是 context switch 时不传输这些资源以节省开销，导致资源"预留"在 time slice 间。
- **(O4) Memory transfer contention**：跨进程的 memory transfer（H2D/D2H）存在竞争（PCIe/DMA 共享），即使 kernel 执行被时间片隔离，memory 操作仍可能被干扰——如 ResNet-34 的 inference TT 因 memory transfer 竞争显著增加。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Time-slicing 是默认行为：多个进程向同一 GPU 发射 CUDA kernel 时自动启用（Linux），无需配置。其适合的场景：(i) latency-sensitive task 需要 predictable 执行时间且 workload 较短；(ii) GPU 资源不足以支持 spatial sharing（如 kernel 大且全部 SM 需要）。不适合：(i) 需要高 utilization 的 background task；(ii) 有频繁 context switch 的 long-running workload（accumulated switch overhead）；(iii) 需要优先级区分的场景（time slice 大小/频率不可配）。Kubernetes 中可通过 NVIDIA device plugin 的 `sharing.timeSlicing.replicas` 实现 GPU oversubscription，但不提供硬件隔离。

涉及论文标题：
- Characterizing Concurrency Mechanisms for NVIDIA GPUs under Deep Learning Workloads
