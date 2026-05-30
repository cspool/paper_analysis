## SM Occupancy

术语是什么？
SM Occupancy（流多处理器占用率）是衡量 GPU 计算资源实际利用程度的指标。它定义了在一个 SM 上同时驻留的 active warp 数量与该 SM 最大可支持 warp 数量的比值。SM occupancy 不同于 GPU utilization——后者仅表示 GPU 是否有任何 kernel 在运行（binary indicator），而 SM occupancy 精确反映 SM 上有多少计算单元（thread/warp）真正处于活跃状态。

从硬件架构角度拆解术语：
HuntKTm 揭示了 SM occupancy 在 memory-intensive kernel 场景下的关键洞察：即使 GPU utilization 为 100%（GPU 始终有 kernel 在运行），SM occupancy 可能仍低于 10%。原因如下：

```
场景：memory-intensive application（如 M2，含 FasterTransformer 的 activation 和 reduction kernel）

GPU Utilization = 100%:
  - GPU 始终被 kernel 占据，没有 idle 间隙
  - DCGM 报告的 Utilization 指标显示 "GPU active"

SM Occupancy < 10%:
  - 每个 kernel 因大量 memory access 而无法维持高 warp occupancy
  - SM 上的 warp 因等待 memory access（global memory latency ~400-800 cycles）而被 stall
  - Warp scheduler 虽然尝试切换 warp 来隐藏延迟，但 kernel 本身的并行度不足以填充所有 warp slots
  - 只有少数 warp 处于 "eligible for execution" 状态

后果：
  - 即使 GPU "100% busy"，大部分 SM 计算核心闲置
  - 这就是为什么仅靠单一 kernel 或单 task 无法充分饱和 GPU
```

HuntKTm 利用这一观察，通过 hybrid scheduling 在不同层面增加并发 kernel 数量，从而提高 SM occupancy。实验显示 HuntKTm 比 SA baseline 提升 SM occupancy 2.47×-3.76×（W4/W8 workloads on 4×A100）。

术语一般如何实现？如何使用？
SM occupancy 通过 NVIDIA DCGM (Data Center GPU Manager) 或 nvidia-smi 的 `nvmlDeviceGetUtilizationRates` 获取。DCGM 提供低开销的周期性硬件指标采集，包括 `DCGM_FI_PROF_SM_OCCUPANCY`。CUDA Occupancy Calculator（`cudaOccupancyMaxActiveBlocksPerMultiprocessor`）可在编译/运行时根据 kernel 的 threads/block、registers/thread、shared memory/block 参数预测理论 occupancy。SM occupancy 是 GPU kernel 性能调优的关键指标——低 occupancy 可能表明 kernel 受限于 register 或 shared memory 资源而非计算吞吐。

涉及论文标题：
- HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs
