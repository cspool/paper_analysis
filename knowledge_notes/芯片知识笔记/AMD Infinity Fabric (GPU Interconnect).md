## AMD Infinity Fabric (GPU Interconnect)

术语是什么？
AMD Infinity Fabric是AMD在CDNA架构GPU之间使用的高带宽、低延迟互联技术。在MI300X和MI325X平台上，每块GPU配备7条Infinity Fabric Link，形成8-GPU全连接(full mesh)拓扑——任意两个GPU之间可直接peer-to-peer通信，无需经过host CPU中转。每GPU还通过x16 PCIe Gen 5链路连接host CPU。Infinity Fabric提供远高于PCIe的带宽（MI300X每link双向带宽>50 GB/s），其全连接拓扑使Iris能在Triton kernel内直接执行跨GPU remote memory操作（通过指针翻译后直接tl.load/tl.store），实现tile级通信而不需host侧协调。Infinity Fabric也支持AMD的GPU间内存一致性协议，配合SC-HRF内存模型提供跨GPU的coherence保证。

从芯片设计角度拆解术语：
Infinity Fabric在AMD MI300X系统中的芯片级组织形式：
```
┌─────────────────────────────────────────────────────────────┐
│ 8-GPU Fully Connected Mesh (MI300X Node)                    │
│                                                             │
│     GPU_0 ←──7 IF Links──→ GPU_1 ←──IF──→ GPU_2            │
│       │  \                  │  \              │             │
│       │   ←──IF──→ GPU_3 ←─IF──→ GPU_4       │             │
│       │         \           /        \        │             │
│     GPU_5 ←──────IF──→ GPU_6 ←──IF──→ GPU_7  │             │
│                                                             │
│  Each GPU: 7 Infinity Fabric Links → full mesh connectivity │
│  Each GPU → Host CPU: x16 PCIe Gen 5 (~64 GB/s)             │
└─────────────────────────────────────────────────────────────┘
```

Infinity Fabric Link的物理实现：
- 每link包含多lane串行链路（类似NVLink），集成在GPU die边缘
- 链路直接连接相邻GPU的IF ports，无需外部switch芯片（区别于NVIDIA NVSwitch需要独立switch芯片）
- 全连接拓扑使任意GPU pair通过单hop可达，延迟极低（<1μs）
- 跨GPU内存访问通过IF透明执行：GPU_A发出remote load → IF PHY → 串行链路 → 目标GPU_B的IF PHY → GPU_B的HBM controller → 返回数据沿相同路径

与NVIDIA NVLink/NVSwitch的对比：
- NVLink: NVIDIA的GPU互联，H100用NVLink 4.0 (900 GB/s双向，18 links/GPU)，通过NVSwitch芯片形成全连接
- Infinity Fabric: AMD的不依赖external switch的直接mesh互联，MI300X用7 links/GPU，每个link ~50 GB/s单向
- NVSwitch提供in-network reduction (SHARP)，IF不提供——reduction需在GPU端完成

术语一般如何实现？如何使用？
在Iris中，Infinity Fabric被透明使用——开发者只需调用iris.load/store/get/put等device-side API，底层通过__translate函数计算remote pointer后直接用tl.load/tl.store执行remote memory操作，GPU硬件通过Infinity Fabric自动路由到目标GPU的HBM。开发者和Triton kernel代码无需显式管理IF链路。

涉及论文标题：
- Iris: First-Class Multi-GPU Programming Experience in Triton
- HipKittens: Fast and Furious AMD Kernels
