## Observability-aided GPU Memory Oversubscription

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - ObservUVM：基于 NVIDIA UVM（Unified Virtual Memory）的软件框架，通过复用现有硬件 access counters，为 CPU 侧 UVM 驱动提供对 HBM-resident 页访问的（采样）可观测性（observability），从而做出更明智的 HBM 换出（eviction）与预取（prefetching）决策，全程无需硬件修改。核心机制：从 HBM-resident 2MB 区域中抽取一个 64KB 页迁移到 DRAM 并 pin 住、映射到 GPU 页表；GPU 对该采样页的 PCIe 访问触发 access counter 通知，驱动据此推断其所属 2MB 区域是否被 GPU 活跃访问。机制（observability/eviction/prefetch 执行）留在驱动内（修改 NVIDIA 开源 UVM 驱动 v525），策略放 userspace（C++11 引擎 + eBPF 通信层）。
  - 实验比较：默认 UVM（LRM 换出 + TBP 预取）vs ObservUVM 的三种 eviction 策略（近似 LRU/LFU/Cyclic Protection）+ Tournament meta-policy（运行时选策略）+ 两种预取策略（FDP 反馈驱动、RGP 区域粒度预取），以及 ACBM（access counter 迁移，计数器原始用途）与 EarlyAdaptor（EA，先前工作）。指标：执行时间（归一化到 UVM）、GPU 页错误数、eviction 次数。14 个应用在 30%-70% 内存超订下平均加速 34%（几何均值，最高 64% SPM）；TM++ 相比 ACBM/EA 平均快约 20%。

- 后端平台是什么，配置是什么。
  - NVIDIA GeForce RTX 3090 GPU（24 GB GDDR6/HBM），AMD Ryzen 9 7950X CPU，PCIe 4.0 互连，64 GB DRAM（最低 32 GB）。软件：Linux 6.2 内核、libbpf、make/gcc/clang、CUDA 11.8（可选）、root 权限（运行 eBPF 所需）。驱动编译一次，复用 baseline 与 ObservUVM 两套驱动。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件：ObservUVM 框架（修改的 NVIDIA open-gpu-kernel-modules v525 UVM 驱动 + userspace 引擎 + eBPF 通信层）+ 14 个 UVM 应用（MM tiled matmul 10.1GB、GMM cuBLAS matmul 12GB、SRK/SR2 cuBLAS symmetric rank 9.6/8.3GB、HEL Hellinger 7.9GB、2DC 2D convolution 18GB、GMV cuBLAS matrix-vector 23.6GB、LU cuSOLVER decomposition 12GB、BLK Black-Scholes 10GB、SPM cuSPARSE sparse matmul 8GB、AN AlexNet batched inference 10.3GB、SN SqueezeNet batched inference 19.8GB、BTR B+tree query 5GB、BFS Rodinia 2.1GB）。
  - 修改了什么：driver 目录编译出 nvidia.ko、base-driver.ko、super-driver.ko（baseline/ACBM/EA/ObservUVM 驱动，compile_drivers.sh）；driver_change_base/ac/ea/super.sh 切换 DPATH 指向驱动目录；userspace 侧 compile_userspace.sh + gen_configs.sh；workloads/bfs/inputGen 生成 BFS 输入；run_key.sh 跑关键配置（复现 fig 9-13），fig9-14.sh 生成 csv，粘贴进 Graphs.xlsx 出图。超订模拟：用 cudaMalloc 预留部分 HBM 使其对应用不可用，制造 x% 超订（footprint 比可用 HBM 大 x%）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/csl-iisc/ObservUVM ；artifact DOI https://doi.org/10.5281/zenodo.19428841（aeisca26.tar.gz，约 20GB，准备约 2 小时，实验约 24 小时）。含预编译 workload 二进制与源码。
  - 使用例子：tar xvf aeisca26.tar.gz → cd driver && bash compile_drivers.sh → 编辑 driver_change_*.sh 设 DPATH → cd userspace && bash compile_userspace.sh && bash gen_configs.sh → 生成 BFS 输入 → bash run_key.sh → bash fig9-14.sh 生成 csv → 导入 Graphs.xlsx。
  - 评估原理与全过程：GPU 访问 HBM 中不存在的 64KB 页 → 页错误写入 GPU-CPU 共享缓冲 → 驱动读错误并迁移 DRAM→HBM（PCIe），映射 GPU 页表并 replay 指令（单错误 10-50us）→ ObservUVM 驱动经 eBPF tracepoint 把 page fault/access counter/prefetch/eviction 事件上行到 userspace 引擎 → 引擎事件循环调用已注册策略的 onPageFault/onAccessCounter/onEviction/onPrefetch 回调 → 策略通过 setEvictionRegion/setPrefetchThreshold/setPrefetchRegion/setObservabilityCandidate/setFeedbackCandidate 返回决策 → 下行驱动强制执行。可观测性链路：策略把链表头部（濒临换出）最多 100 个 2MB 区域设为 observable → 驱动将每区域一个 64KB 采样页迁到 DRAM 并 pin、映射 GPU 页表 → GPU 活跃访问采样页（PCIe，阈值=1）→ access counter 通知 → 驱动告知策略（如 LRU 中 move_to_tail 保护）。输出：执行时间（wall-clock，归一化 UVM）、GPU 页错误数、eviction 数。自定义策略：新建目录，eviction 继承 EvictionPolicy 基类、intra-2MB 预取继承 ShallowPrefetch、inter-2MB 预取继承 DeepPrefetch，实现全部虚函数。
