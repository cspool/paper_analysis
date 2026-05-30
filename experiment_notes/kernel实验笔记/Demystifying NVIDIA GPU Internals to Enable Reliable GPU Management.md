## Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management

> **近似层次匹配说明**：本文的核心实验通过自研的nvdebug内核模块和gpu-microbench微基准测试套件，在kernel/OS级别研究NVIDIA GPU的硬件调度行为。实验直接测量compute kernel和copy操作在不同runlist、channel配置下的timeslicing、并行度、互斥和干扰模式，属于kernel调度层面的运行时计算行为研究。

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  论文实现了两个核心工具套件用于kernel调度行为研究：
  (i) **nvdebug**：Linux内核模块，通过MMIO直接访问GPU寄存器（绕过GPU驱动），暴露GPU调度状态的检查和修改接口（/proc/gpuX/）。关键能力：查看runlist内容、禁用/启用channel、查看device_info（engine-runlist拓扑）、查看PCE-LCE映射。
  (ii) **gpu-microbench**：微基准测试库，包含exec_logger（持续执行compute kernel并以微秒精度记录每次执行时间戳）和copy_monitor（类似但仅使用copy engine，不触发compute工作），用于精准测量各引擎的调度行为。

  实验比较的核心是GPU在不同硬件配置下的kernel调度行为模式：
  - **Channel级别（R1-R2）**：验证所有GPU操作必须经过channel（禁用channel后kernel/copy/device-mapped memory allocation无法完成）；channel数限制intra-task并行度——x86_64默认8 compute channel，9个stream时第9个stream产生false dependency；增加CUDA_DEVICE_MAX_CONNECTIONS可消除
  - **Runlist级别（R3-R5）**：验证channel必须在runlist中；单runlist上每约2ms compute timeslice互斥、约1ms copy timeslice互斥；多runlist支持compute/copy独立调度无干扰；单runlist（Jetson TX2）上compute和copy共享导致copy被compute干扰（中断间隔1024µs=compute timeslice而非copy的1049µs）
  - **Engine映射级别（R6-R8）**：发现所有GPU的Runlist 0同时绑定Compute和GRCE；每个engine仅绑一个runlist（PTOP寄存器约束）；GRCE可通过共享底层PCE干扰独立runlist上的LCE——RTX 6000 Ada上OpenGL texture upload使CUDA GPU→CPU copy减速约2×

  还使用已有工具cuda_scheduling_examiner（Otterness et al. [17]）进行部分辅助实验。

- 后端平台是什么，配置是什么。
  9款NVIDIA GPU（覆盖5代架构，2016-2022）：
  GTX 1060 3GB (Pascal, CC 6.1)、GTX 1080 Ti (Pascal, CC 6.1)、Jetson TX2 (Pascal embedded, CC 6.2)、Titan V (Volta, CC 7.0)、Jetson Xavier (Volta embedded, CC 7.2)、RTX 2080 Ti (Turing, CC 7.5)、A100 40GB (Ampere, CC 8.0)、Jetson Orin (Ampere embedded, CC 8.7)、RTX 6000 Ada (Ada Lovelace, CC 8.9)。OS：x86_64和aarch64 Linux。所有实验前禁用后台GPU任务（Jetson TX2额外通过nvdebug清空runlist残留条目）。

- 评估性能的软件/脚本是什么。修改了什么。
  - **自研**：nvdebug（内核模块，~1500行C代码，MMIO寄存器访问 + GPU页表解析）、gpu-microbench（exec_logger记录compute kernel每次迭代的开始/结束时间戳和SM编号，微秒精度；copy_monitor仅使用copy engine记录copy进度）、多个实验编排脚本（同时运行多个微基准测试实例，交叉验证调度行为）
  - **已有工具**：cuda_scheduling_examiner（Otterness et al. [17]，部分实验使用）
  - **修改/新增**：exec_logger和copy_monitor为全新开发（需多年专家级调优和bug修复）；nvdebug为全新开发（需解决GPU页表访问/解析/遍历、多代GPU寄存器地址兼容性等问题）；实验脚本组合使用以上工具进行cross-validation

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - nvdebug: http://rtsrv.cs.unc.edu/cgit/cgit.cgi/nvdebug.git/
  - gpu-microbench及实验脚本: https://www.cs.unc.edu/~jbakita/rtas24-ae/

  评估原理与流程（以验证R2——channel数限制并行度实验，Fig.5，GTX 1060 3GB为例）：

  1. **环境**：加载nvdebug内核模块 → cat /proc/gpu0/device_info确认GPU拓扑 → 确认默认compute channel=8（CUDA 12.2 on x86_64）→ 禁用后台GPU任务

  2. **实验**：创建9个CUDA stream（Stream 1-9），每个stream内顺序launch 4个kernel（K1, K2短kernel, K3长kernel, K4短kernel）。长kernel K3含大量thread blocks以延长dispatch时间。

  3. **数据采集（exec_logger）**：每个kernel的每个thread block在开始/结束时通过CUDA event记录时间戳 → 记录(kernel_name, thread_block_index, start_time, end_time, SM_id) → 微秒精度时间线

  4. **分析（Fig.5 top）**：Stream 1-8的头kernel几乎立即开始执行 → Stream 9的头kernel等到t≈0.7s（Stream 1的K3所有blocks dispatch完毕释放channel）→ 验证false dependency：Stream 9需等待任意channel释放

  5. **对照（Fig.5 bottom）**：CUDA_DEVICE_MAX_CONNECTIONS增加channel至≥9 → 所有9个stream头kernel同时开始执行，无false dependency → 确认R2

  6. **跨GPU验证**：9款GPU上重复实验，确认x86_64默认=8，Jetson默认=2-4

  验证R4（单runlist互斥timeslicing）流程（Fig.6, 7）：
  - exec_logger双实例 → 时间线显示严格互斥（Fig.6 inset），约2ms timeslice切换
  - copy_monitor双实例 → copy进度交替推进（Fig.7），约1ms timeslice切换

  验证R5（多/单runlist调度独立性）流程（Fig.8, 9）：
  - GTX 1080 Ti（多runlist）：exec_logger + 2×copy_monitor + copy-and-compute task → compute和copy时间线非同步、不相关（Fig.8）→ 多runlist独立调度
  - Jetson TX2（单runlist）：exec_logger + copy_monitor → copy出现1024µs周期性中断（=compute timeslice，非copy的1049µs）（Fig.9）→ 单runlist导致跨引擎干扰

  验证R8（GRCE→LCE PCE共享干扰）流程（Fig.10, 11）：
  - GTX 1080 Ti vs RTX 6000 Ada → 各运行CUDA GPU→CPU copy + 并发OpenGL texture upload → RTX 6000 Ada copy被减速约2×（因GRCE映射到同一LCE→共享PCE），GTX 1080 Ti几乎不减速
  - cat /proc/gpu0/lce_for_pce0和shared_lce_for_grce0确认映射关系
