## 计算-通信干扰（Compute-Communication Interference：SM 争用、非抢占调度与内存带宽争用）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 在 GPU 上同时跑 GEMM 计算 kernel 与 NCCL 集合 kernel 时互相拖慢的现象，两个来源：(1) 计算单元争用——GPU 非抢占式调度，占满所有 SMs 的 GEMM kernel 会阻塞后续集合 kernel 启动（即使集合 kernel 优先级更高、只需 <10% SM）；(2) 内存带宽/L2 争用——并发 GEMM 使 NCCL AllReduce 的 algorithm bandwidth 降 30%（用 CUDA MPS 静态隔离 SM 后依然存在，证明是内存子系统争用）。
- 后果：MSDP（ZeRO-Infinity）训 OPT-175B（8×1-GPU、100Gbps）时 65% 集合时间无法与 GEMM 重叠，迭代时间比理想重叠多 41%，MFU 仅 15%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 调度序列例子（MSDP 反向，同一 GPU 两个流）：
```
stream_compute: launch(GEMM1); launch(GEMM2)      # GEMM2 依赖 AG2 的数据
stream_coll:    launch(AG1); launch(AG2); launch(RS1); launch(RS2)
# 非抢占：GEMM1 占满所有 SM 期间 AG2 排队等待；AG2 完成前 GEMM2 无数据可算
overlap_ideal = max(T_GEMM, T_coll)
overlap_actual = T_GEMM + T_coll_unhidden          # 65% 集合时间不可隐藏
```
- 缓解选项与局限：CUDA MPS 只能静态分区 SM，与 dynamic parallelism、CUDA Graph 等 LLM 训练重度依赖特性不兼容，且不解决内存带宽争用；DisDP 的解法是把集合整体搬到 SmartNIC——实测 PCIe DMA 与 GEMM 并发时双向几乎零干扰，因此完全解耦优于 GPU 内调度优化（仿真中 ZeRO-Inf+MPS / ZeRO-Inf+Preemp 仅达 DisDP 32.7%/32.8% 吞吐，二者均只解决 SM 争用）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 判别方法：并发跑 GEMM 与 NCCL AllReduce（用 MPS 隔离 SM 作控制变量）测 algorithm bandwidth 降幅；或并发跑 PCIe 设备 DMA 与 GEMM 测双向干扰。使用要点：追求计算-通信重叠时应优先 copy engine/DMA 类传输（不占 SM）或 SmartNIC 卸载，而非 GPU 内核级重叠或 MPS 分区。信息缺口：论文未给出内存带宽争用中 L2 与 DRAM 分量的拆解。

MTIA 300 补充视角（ISCA'26，硬件级消除计算-通信干扰）：MTIA 300 以专用 ME + NMC 把 collective 完全卸载到网格边缘的独立引擎，从硬件上消除 GPU 式"SM 做归约与 GEMM 争用"的干扰。重叠微基准：1000 次 TF32 4K×4K×4K GEMM 与代表消息大小的 collective 并发（16 加速器），MTIA 300 计算与通信双 ~100% 效率（等于单独运行性能），H100 因 SM 争用明显退化（见 C3/计算-通信干扰通用条目）。40 卡 DLRM 训练实测通信超 H100 3.9×，体现"计算-通信重叠 + 无干扰"的端到端收益（Perf/TCO 1.42×）。

涉及论文标题：
- DisDP: Disaggregating Compute, Network, and Storage for Model-Sharded Data-Parallel Training
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
