## GDRCopy-Based Kernel Transfer

术语是什么？通过联网搜索让回答具体和精准。
GDRCopy (GPU Direct RDMA Copy) 是NVIDIA提供的一种低延迟host-device数据传输库，允许CPU直接通过PCIe BAR (Base Address Register) mapping读写GPU显存，bypasses传统CUDA API的DMA引擎和driver stack。Infera使用GDRCopy的gdr_copy_to_mapping函数将fused kernel的binary code和arguments从host端HKQ传输到device端kernel slots和DKQ，实现<100ns的小数据延迟和<5μs的典型kernel传输延迟。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GDRCopy在Infera kernel launch pipeline中的位置：
```
Host side:
  fused kernel binary + args in host memory (cudaHostAllocWriteCombined)
        ↓
  gdr_copy_to_mapping(gdr, dev_ptr, host_ptr, size)
        ↓ (direct PCIe BAR write, <5µs for typical kernel ~KB size)
  Device kernel slot (placeholder overwrite) + global memory (args)
        ↓
  Kernel pointer + arg pointer → DKQ
        ↓
  Daemon kernel → cudaLaunchDevice (fire-and-forget)
```

对比传统路径：
```
传统: cuLaunchKernel/cuModuleLoad
  → CUDA driver stack → GPU work queue → DMA engine → GPU execution
  → ~10-100µs overhead + global host-device synchronization

Infera: GDRCopy + daemon kernel CDP
  → GDRCopy BAR write → DKQ enqueue → device-side launch
  → <10µs total, no host-device sync, no HoL blocking
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GDRCopy需要：(1) GPU支持GPUDirect RDMA（需NVIDIA数据中心GPU + BAR1 mapping enabled）；(2) 安装nvidia-peermem和gdrcopy kernel module。Infera中，compiled kernel binary和arguments放置在cudaHostAllocWriteCombined标记的host memory中（优化PCIe write combining），通过GDRCopy直接写入GPU显存的预留kernel slot area和argument area。Kernel slot覆盖使用driver-level修改（绕过cuModuleLoad），因为cuModuleLoad会触发global host-device同步破坏低延迟目标。Host launcher维护fused kernel cache pool在host memory中复用。

涉及论文标题：
- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling

