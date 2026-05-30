## HBM (High Bandwidth Memory)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

HBM（High Bandwidth Memory，高带宽内存）是一种3D堆叠DRAM技术，通过硅通孔（TSV, Through-Silicon Via）将多层DRAM die垂直堆叠在logic die上，提供比传统GDDR高得多的带宽和更低的功耗。在GPU中，HBM是片外（off-chip）主内存，存储模型参数、激活值、优化器状态和中间张量。以NVIDIA A100为例：40GB或80GB HBM2e，通过5120-bit memory bus提供1.5-2.0TB/s带宽。HBM的核心设计原理：将多个DRAM层垂直堆叠并使用宽I/O接口（1024 bits per stack × 4-6 stacks），以较小的物理面积和功耗实现极高带宽。在attention计算中的角色：HBM存储Q/K/V输入tensor和输出O tensor，以及（在标准实现中）中间$N \times N$矩阵S和P。FlashAttention的IO-awareness直接针对HBM访问优化——目标是最小化HBM↔SRAM之间的数据搬运量，因为HBM的带宽仅为SRAM的1/10。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

GPU中HBM的物理组织和数据流（以A100 40GB HBM2e为例）：
```
                   ┌──────────────────────────────┐
                   │         GPU Die (GA100)        │
                   │  ┌───┐ ┌───┐     ┌───┐ ┌───┐ │
                   │  │SM0│ │SM1│ ... │ │SMn│ │L2 │ │
                   │  └───┘ └───┘     └───┘ │Cache│ │
                   │                          │40MB│ │
                   └──────────────────────────┼────┘
                                              │
                  ┌───────────────────────────┼────┐
                  │      Silicon Interposer    │    │
                  └───────────────────────────┼────┘
                                              │
          ┌──────────┐  ┌──────────┐  ┌──────────┐ ...
          │ HBM Stack│  │ HBM Stack│  │ HBM Stack│    (5 stacks, 8GB each)
          │ 8 DRAM   │  │ 8 DRAM   │  │ 8 DRAM   │
          │ layers   │  │ layers   │  │ layers   │
          │ +Logic   │  │ +Logic   │  │ +Logic   │
          │ die      │  │ die      │  │ die      │
          └──────────┘  └──────────┘  └──────────┘
          
每stack: 1024-bit I/O × 1.6 Gbps/pin = 204.8 GB/s per stack
5 stacks × 204.8 GB/s ≈ 1.0 TB/s (base), A100 80GB版本达2.0 TB/s

HBM在attention计算中的典型数据流（标准attention, N=1024, d=64, FP16）:
  HBM →: Q(128KB), K(128KB), V(128KB) 加载
  HBM ↔: S(2MB write + 2MB read), P(2MB write + 2MB read)  共8MB中间数据
  HBM ←: O(128KB), dQ(128KB), dK(128KB), dV(128KB) 写出
  Total HBM traffic per attention layer (fwd+bwd): ~35GB per 64-batch
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

HBM的代际演进：HBM1（2015, 128GB/s per stack）→ HBM2（2016, 256GB/s）→ HBM2e（2020, 460GB/s, A100用）→ HBM3（2022, 819GB/s, H100用）→ HBM3e（2024, 1.2TB/s, B200用）。HBM在AI加速器中的使用方式：(1) 通过memory controller和L2 cache与GPU SM通信；(2) GPU kernel通过global memory load/store指令访问，程序员通过coalescing优化带宽利用率；(3) CUDA编程中HBM对程序员透明——`cudaMalloc`分配在HBM上，kernel直接通过指针访问。FlashAttention通过最小化HBM↔SRAM往返次数优化性能——HBM访问从35.3GB降至4.4GB（8× reduction），因为中间attention矩阵从不materialize在HBM中。这使得HBM带宽不再是最主要的瓶颈，kernel从memory-bound向compute-bound移动。

涉及论文标题：
- FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness
