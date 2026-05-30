## NUMA vs UMA in CoE Inference (CoE 推理中的 NUMA/UMA 内存架构影响)

术语解释
NUMA (Non-Uniform Memory Access) 和 UMA (Unified Memory Architecture) 是两种内存架构，对 CoE 推理中的 expert switching 延迟有显著影响。NUMA 设备（如 RTX3080Ti + Xeon CPU）的 GPU 和 CPU 有独立内存，expert 在不同 tier 间切换通过 PCIe/NVMe 传输；UMA 设备（如 Apple M2）的 CPU 和 GPU 共享统一内存，理论上应避免显式传输，但实际因 AI 框架（PyTorch）的数据重组仍会产生开销。

术语是什么？
NUMA vs UMA 对 CoE expert switching 的量化影响（CoServe 论文 Figure 1）：
- **NUMA (RTX3080Ti)**：从 SSD 切换 expert 到 GPU 占推理延迟 90%+；从 CPU 加载也占显著比例
- **UMA (Apple M2)**：从 SSD 切换 expert 到 GPU 占推理延迟 60%+（虽然 UMA 理论上无 CPU↔GPU 拷贝，但 PyTorch 等框架在统一内存上仍会做数据重组）
- 结论：无论在 NUMA 还是 UMA 设备上，减少 expert switching 频率都是提升 CoE 推理效率的关键

从硬件架构角度拆解术语：
NUMA 设备（RTX3080Ti + Xeon Silver 4214R）上 CoE 推理的 expert 加载路径：
```
SSD (MICRON MTFDDAK480TDS, 530 MB/s)
  → PCIe → CPU Memory (16GB, DDR4)
    → PCIe → GPU Memory (12GB, GDDR6X)
      → GPU SM 执行推理
层级延迟: SSD > CPU > GPU
```

UMA 设备（Apple M2 24GB）上 CoE 推理的 expert 加载路径：
```
SSD (APPLE AP0512Z, ~3000 MB/s)
  → 统一内存 (24GB LPDDR5, CPU/GPU 共享)
    → GPU 执行推理
# PyTorch 在统一内存上的数据重组仍产生相当于 CPU→GPU 拷贝的延迟
```

术语一般如何实现？如何使用？
- 在边缘设备（RTX3080Ti 12GB、Jetson Xavier NX 16GB）上部署 CoE 时，所有 expert 无法全部驻留 GPU
- Offline Profiler 需对 NUMA 和 UMA 设备分别 profile（batch size、latency、memory footprint 均不同）
- NUMA 设备上 batch size 越大内存占用越大，减少可加载的 expert 数量 → CoServe 通过搜索平衡
- UMA 设备上统一内存的带宽更高（~3GB/s SSD vs 530MB/s），但仍有 switching 开销

涉及论文标题：
- CoServe: Efficient Collaboration-of-Experts (CoE) Model Inference with Limited Memory
