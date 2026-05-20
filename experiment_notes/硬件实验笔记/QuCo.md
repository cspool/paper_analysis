## QuCo: Efficient and Flexible Hardware-Driven Automatic Configuration of Tile Transfers in GPUs

- 属于硬件架构的实现是什么？实验比较什么？
  提出QuCo（Queue Configurator），一个嵌入GPU die的单次轻量级硬件单元，自动完成ATT（Asynchronous Tile Transfer）的全配置过程。QuCo内部包含：(1) compact in-order RISC-V处理器（RV32IMF指令集，5级流水线），运行轻量级固件；(2) 8 KiB ROM存储固件（microcode）；(3) 2 KiB local data buffer存储局部变量和运行时数据；(4) GPU Specification Table (GST)，256-byte只读block，由vendor在制造时写入，存储关键架构参数（memory latencies、clock frequency、LDS size、CU数量、arithmetic throughput等）。QuCo集成在GPU Command Processor附近，在kernel launch时RISC-V core执行固件从GST读取架构参数+结合kernel特征（compute intensity、vector sizes、queue类型和数量）→动态计算最优tile sizes、queue slots和LDS partitioning→将ATT descriptors/tile参数/slot pointers写入LDS→完成后进入idle等待下次reconfiguration。QuCo可在多kernel动态workload中重新调用来重新计算queue layout和更新descriptors。硬件实现：28nm FDSOI工艺，700MHz，RISC-V core面积0.027mm² [42]，memory subsystem（8+2 KiB + 256-byte GST）约0.014mm²（CACTI评估），总物理面积约0.041mm²。访问延迟0.37ns，per-read 0.0032 nJ、per-write 0.0061 nJ dynamic energy。reconfiguration耗时~6,300-8,300 cycles（IPC=1时）。实验比较：(1) 6种执行case：NoATT/Non-Tuned、NoATT/Fine-Tuned、ATT/Non-Tuned、ATT/Informed-Tuned、ATT/Fine-Tuned、QuCo；(2) 3种GPU架构（MI-100 high-end、R9 Nano mid-end、Radeon 530 low-power）的portability测试；(3) DVFS频率变化下QuCo-HW vs QuCo-SW对比；(4) 消融实验：移除CU-aware slot scaling、Little's Law、CI-based scaling；(5) 6种benchmark（AlbertV2/T5-Small/Whisper Tiny/Norm-Project/Attention-Score/Residual-MLP）上的端到端对比。QuCo性能在expert hand-tuned ATT配置的1.04%以内，DNN端到端geomean平均1.15× over ATT/Fine-Tuned。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  MGPUSim（cycle-accurate GPU simulator，开源 https://github.com/umd-memsys/mgpusim），校准为AMD R9 Nano（GCN3 ISA），作为mid-range GPU baseline。portability测试额外覆盖MI-100（high-end）和Radeon 530（low-power）。MGPUSim开源。

- 模拟器模拟什么的性能，修改了什么。
  模拟器模拟ATT-enabled GPU的cycle-accurate性能：(1) global memory与LDS之间的asynchronous tile transfers（背景数据移动）；(2) operand queue management（producer wavefront Push/Wait_For_Push + consumer wavefront Peek/Pop + asynchronous transaction barriers）；(3) LDS coordination at functional and cycle levels。论文修改：扩展MGPUSim支持ATTs——添加background data movement模型、operand queue management逻辑、LDS allocation和synchronization barrier建模。ATT设计是architecture-neutral的，任意GPU带asynchronous global-to-shared memory transfer均可受益于QuCo。GPU配置参数（CU数量、频率、memory latency、LDS大小、带宽等）通过GST概念建模，具体值见Table II（R9 Nano: 64 CUs 1.0GHz 8x512MB DRAM 190/300/450 L1/L2/DRAM latency; MI100: 120 CUs 1.5GHz 32x1GB DRAM; Radeon 530: 6 CUs 1.0GHz 8x256MB DRAM）。

- 开源情况。模拟器如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：QuCo firmware源码和MGPUSim ATT扩展未明确提供开源链接（HPCA 2026）。MGPUSim本身开源（https://github.com/umd-memsys/mgpusim）。硬件架构使用流程：
  1. 配置GPU模型：编辑MGPUSim配置文件设置CU数量、频率、memory hierarchy参数（L1/L2/DRAM capacity/latency/bandwidth）→GST内容对应这些参数
  2. 配置QuCo单元：QuCo的RISC-V firmware（8KB ROM）、data buffer（2KB）和GST（256B）在GPU startup阶段加载→firmware读取GST架构参数到local registers
  3. 编写host code：driver.InitQuCo(CI, WG_SIZE, #CUs)初始化→driver.RegisterQueue(K_size, data_type_size, TYPE_STREAMING/STATIONARY)注册operand queues→driver.EnqueueLaunchKernel(binary, kernArg) launch kernel
  4. kernel launch时QuCo执行：Algorithm 1计算optimal tile size（遍历64-8192 elements，evaluate merit factor = processing time / memory transfer time × cost function → weighted merit → 按CI调整）→Algorithm 3计算optimal slots（streaming queues用Little's Law，stationary queues用remaining LDS capacity均分 → CU-aware rounding → LDS capacity check with CI-based fallback）→写入ATT descriptors/slot pointers/barrier indices到LDS
  5. 模拟执行：MGPUSim运行kernel → 输出DRAM activity traces、execution cycles、speedup vs ideal ATT（unlimited LDS）
  6. 面积/能耗验证：Synopsys Design Compiler (28nm FDSOI)综合RISC-V core → CACTI评估memory structures → 总QuCo面积~0.041mm²
