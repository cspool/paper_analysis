## NVLink (NVLink 3.0 / NVLink 4.0)

术语解释
NVLink 是 NVIDIA 开发的高带宽 GPU 直连互联技术，提供远高于 PCIe 的 GPU-to-GPU 通信带宽。在 MoE 推理中，NVLink 承载 all-to-all token 通信、expert 参数传输和 gradient 同步等数据密集型操作，是分布式 MoE 推理性能的关键瓶颈。

术语是什么？
NVLink 是一种点对点（point-to-point）串行互联协议，不同于 PCIe 的总线拓扑，NVLink 使用 mesh 拓扑减少节点间通信延迟和拥塞。NVLink 3.0（A100）每链路 50 GB/s 单向，每个 A100 GPU 提供 12 条 NVLink，聚合双向带宽达 600 GB/s。NVLink 4.0（H100）提升至 900 GB/s 双向。通过 NVSwitch 芯片实现全互联，所有 GPU 可直接通信无需 CPU 中转。

从芯片设计角度拆解术语：
```
NVLink A100 互联拓扑（4 GPU, 全互联 via NVSwitch）:
     GPU0 ──── GPU1
       │  ╲  ╱  │
       │   ╳   │    ← NVSwitch 提供全互联交换
       │  ╱  ╲  │
     GPU3 ──── GPU2

单向带宽: 4 × 50 GB/s = 200 GB/s per GPU
双向聚合: 600 GB/s per GPU
PCIe 4.0 x16 对比: 仅 32 GB/s 单向
NVLink 3.0 vs PCIe 4.0: ~18.75× 带宽优势
```

MoEShard 中 NVLink 的角色：MoEShard 的 token 全复制策略（每 GPU 发送全部 token 给所有其他 GPU）依赖 NVLink 高带宽吸收通信开销。以 batch=250, seq=120, h=768 (4B/element) 为例，每 GPU 需发送 ≈88 MiB，NVLink 3.0 下耗时仅 ~0.15ms，在端到端推理时间中可忽略。若没有 NVLink（如仅 PCIe），token 全复制将需 ~2.75ms，可能抵消计算加速收益。

术语一般如何实现？如何使用？
- NVIDIA GPU 硬件内置：A100 (NVLink 3.0, 600 GB/s), H100 (NVLink 4.0, 900 GB/s), B200 (NVLink 5.0, 1800 GB/s)
- NVSwitch 芯片：连接多个 GPU 的全互联交换芯片，A100 支持 8 GPU (DGX A100), H100 支持 8 GPU (DGX H100)
- 软件接口：NCCL 通信库自动利用 NVLink 拓扑优化 all-to-all / all-reduce 等集合通信
- MoE 推理中的关键影响：高 NVLink 带宽 → token 通信开销小 → 可使用更细粒度的并行策略（如 MoEShard 的 token 全复制）；低 NVLink 带宽（如 PCIe-only 系统）→ 更依赖 expert offloading / caching 减少通信

涉及论文标题：
- Accelerating MoE Model Inference with Expert Sharding
