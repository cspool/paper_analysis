## QuCo (Queue Configurator / 队列配置器)

术语是什么？通过联网搜索让回答具体和精准。

QuCo（Queue Configurator）是QuCo论文（HPCA 2026）提出的嵌入GPU die的单次轻量级硬件单元，全自动完成ATT（Asynchronous Tile Transfer）operand queue的配置过程。核心理念是用专用硬件microcontroller替代程序员手动tuning——在kernel launch时，QuCo的RISC-V firmware读取GPU architectural parameters和kernel特征→动态计算最优tile sizes、queue slots数量和LDS partitioning→写入ATT descriptors到LDS→ATT engine直接读取。QuCo内部由三部分组成：(1) compact in-order RISC-V处理器（RV32IMF ISA, 5阶段流水线）；(2) 8 KiB ROM存储firmware + 2 KiB data buffer存局部变量和运行时数据；(3) GPU Specification Table (GST) 256-byte存储架构参数。28nm FDSOI工艺下RISC-V核心面积0.027mm²，memory subsystem约0.014mm²。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

QuCo集成在GPU Command Processor (CP) 附近，在wavefront scheduling之前介入：
```
GPU 启动 → QuCo 从8KB ROM fetch第一条指令
         → Firmware 读取 GST 到 local registers
         
Kernel Launch：
  Host code: driver.InitQuCo(CI, WG_SIZE, #CUs)
             driver.RegisterQueue(K, 4, TYPE_STREAMING)
             driver.EnqueueLaunchKernel(binary, kernArg)
             
  QuCo 执行：
    Step 1: 从 GST 读架构参数（DRAM bandwidth/latency, 
            LDS size, CU count, clock freq, MACs/cycle）
    Step 2: Algorithm 1 — 遍历 tile size [64, 8192]
             对每个tile计算 merit factor = 
               processing_time / memory_transfer_time
             processing_time = TileSize/(SIMD_Muls×ConsumerWfs)
               + scheduling roundtrip overhead
             memory_time = ATT+DRAM+L2 latencies 
               + TileSize×ElementSize/Bandwidth
               + 2×TileSize×ElementSize/CacheLineSize
             选 weighted_merit 最优 tile → 按 CI 调整
    Step 3: Algorithm 3 — streaming queues 用 Little's Law
             slots = mem_transfer_time / tile_compute_time
             → CU-aware rounding → LDS capacity check
             stationary queues 均分剩余 LDS → round
    Step 4: 写 ATT descriptors, tile params, slot pointers, 
            barrier indices 到 LDS metadata region
    Step 5: 通知 ATT units 加载 descriptors → idle
```
关键：QuCo与main compute pipeline完全解耦，不干扰wavefront scheduling、memory requests和execution flow。多kernel动态workload中QuCo可重新调用更新queue配置。NVIDIA Blackwell已引入类似的dedicated RISC-V microcontroller（AMP）处理非计算任务，QuCo顺应这一趋势。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

QuCo目前是学术proposal，未商业实现。实现要求：
1. **RISC-V microcontroller**：支持RV32IMF指令集（整数+单精度FP），in-order 5-stage pipeline，面积<0.05mm² @28nm
2. **Firmware**：8KB ROM存储（Algorithm 1-3逻辑），启动时fetch执行，可vendor-provided固件升级
3. **GST**：256-byte vendor-fused ROM，每GPU型号unique，存储精确architectural parameters。因在GPU内部，数据不暴露给外部
4. **LDS interface**：QuCo可直接读写LDS metadata region，地址空间private不与compute wavefronts冲突

涉及论文标题：
- QuCo: Efficient and Flexible Hardware-Driven Automatic Configuration of Tile Transfers in GPUs

---

