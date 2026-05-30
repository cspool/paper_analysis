## Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management

> **近似层次匹配说明**：本文并非实现新的RTL IP模块或修改仿真器，而是通过对NVIDIA GPU硬件调度架构进行寄存器级逆向工程，实验性推导出8条GPU调度规则（R1-R8）。这些规则跨越channel、runlist、engine和PCE/LCE映射等硬件调度原语，直接揭示此前未公开的GPU硬件架构行为。因此归入硬件架构层。

- 属于硬件架构的实现是什么？实验比较什么？
  论文通过自研的 **nvdebug** Linux内核模块（通过MMIO直接读取GPU寄存器，绕过用户态和内核态驱动）和 **gpu-microbench** 微基准测试套件（含exec_logger和copy_monitor），对NVIDIA GPU硬件调度架构进行反向工程，实验性推导出8条调度规则：
  - **R1**: 所有GPU引擎操作（kernel launch、copy、device-mapped memory allocation）必须经过channel
  - **R2**: Task的channel数量限制intra-task并行度（默认x86_64上compute channel=8，嵌入式Jetson上仅2-4），超channel数目的stream间产生false dependency
  - **R3**: Channel必须属于某个runlist才能被调度
  - **R4**: 每个runlist最多有一个task per关联engine处于active状态（互斥timeslicing）
  - **R5**: GPU的runlist数量限制独立inter-task并行度——多个runlist支持引擎独立调度，单runlist导致不同引擎间非独立干扰
  - **R6**: 一个runlist可绑定多个engine（如Runlist 0同时绑定Compute/Graphics和Copy Engine 0/1）
  - **R7**: 每个engine只绑定一个runlist（硬件PTOP寄存器约束）
  - **R8**: Copy engine因LCE→PCE的硬件间接映射层（Physical Copy Engine共享）可能违反R7的表观独立性——GRCE可通过共享PCE干扰独立runlist上的LCE copy

  实验比较：展示每条规则的实验证据（如Fig.5: 8 vs 9 channel下的stream并行度对比；Fig.6: 单runlist上compute task的互斥timeslicing；Fig.7: copy task在单runlist上的互斥行为；Fig.8: 多runlist下copy+compute独立调度；Fig.9: 单runlist上copy被compute timeslicing干扰；Fig.10: GRCE→LCE mapping导致的copy干扰）。Evaluation章节将8条规则应用于三个已有GPU管理框架的反例分析，证明规则是safe GPU management的必要条件。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  论文不使用传统模拟器，而是直接对真实NVIDIA GPU硬件进行寄存器级检查和实验。核心工具：
  - **nvdebug**：Linux内核模块，通过MMIO操作直接访问GPU寄存器（绕过GPU驱动），暴露GPU调度状态的透明检查和修改接口。开源地址：http://rtsrv.cs.unc.edu/cgit/cgit.cgi/nvdebug.git/
  - 辅助参考的开源GPU文档：NVIDIA Open GPU Documentation (https://github.com/NVIDIA/open-gpu-doc)、NVIDIA Open GPU Kernel Modules (https://github.com/NVIDIA/open-gpu-kernel-modules)、Nouveau开源驱动 (https://nouveau.freedesktop.org/)、NVIDIA nvgpu (git://nv-tegra.nvidia.com/linux-nvgpu.git)

- 模拟器模拟什么的性能，修改了什么。
  nvdebug不是模拟器，是真实的硬件内省工具。它通过/proc/gpuX虚拟文件系统暴露以下GPU硬件调度状态的直接检查与修改接口：
  - `device_info`：打印GPU引擎信息，包括引擎名称、类型、关联runlist ID——直接读取PTOP（Device Topology）寄存器
  - `runlistY`：打印runlist Y的内容，包括TSG条目（scale、timeout、length）和channel信息（enabled、busy、status、PBDMA faulted、ENG faulted等）——需解析GPU页表访问GPU物理内存中的runlist条目
  - `disable_channel` / `enable_channel`：写入channel ID来禁用/启用对应channel——直接操作GPU寄存器
  - `lce_for_pceY`：读取PCE Y被映射到哪个LCE——读取PCE-LCE映射寄存器
  - `shared_lce_for_grceY`：读取GRCE Y映射到哪个LCE（若存在）——揭示GRCE可共享PCE的硬件配置
  - `pce_map`：读取可用PCE的bit mask

  性能评估原理：nvdebug不是性能模拟器，而是硬件状态检查器。性能数据由gpu-microbench中的exec_logger（记录compute engine活动时间线，微秒级精度）和copy_monitor（记录copy engine活动时间线）通过CUDA event timing和GPU global timer采集。这些工具组合使用，可在不修改GPU驱动的情况下，在microarchitectural级别监控GPU引擎调度行为。

- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？至少具体到模拟器模拟性能的原理和模拟器输入到性能输出的全过程。
  nvdebug和gpu-microbench均已开源：
  - nvdebug: http://rtsrv.cs.unc.edu/cgit/cgit.cgi/nvdebug.git/
  - gpu-microbench（含exec_logger, copy_monitor等）: https://www.cs.unc.edu/~jbakita/rtas24-ae/

  nvdebug使用方式（以检查GPU引擎拓扑和runlist为例）：
  ```
  # 加载内核模块后
  cat /proc/gpu0/device_info
  # 输出：每个engine的名称及其关联runlist ID
  # 例如GTX 1060 3GB:
  #   Graphics/Compute 0        -> Runlist 0
  #   Copy Engine 0 (GRCE0)     -> Runlist 0
  #   Copy Engine 1 (GRCE1)     -> Runlist 0
  #   Video Decoder 0 (NVDEC0)  -> Runlist 1
  #   Copy Engine 2 (LCE2)      -> Runlist 5
  #   Copy Engine 3 (LCE3)      -> Runlist 6

  cat /proc/gpu0/runlist0
  # 输出：Runlist 0中的TSG和channel条目
  # 显示每个channel的enabled/next/busy/status/PBDMA faulted/ENG faulted等状态

  cat /proc/gpu0/lce_for_pce0
  # 输出：PCE 0当前映射到的LCE编号
  # 用于检查PCE-LCE映射关系（R8规则验证）

  cat /proc/gpu0/shared_lce_for_grce0
  # 输出：GRCE 0是否共享了某个LCE（R8实验核心）
  ```

  调度规则发现的工作流程：
  1. **初始化**：加载nvdebug内核模块→通过MMIO读取GPU PTOP寄存器获取engine-runlist拓扑→通过GPU页表解析读取runlist内容
  2. **实验设计**：基于从device_info获得的拓扑信息，设计定向微基准测试（如：已知GTX 1060有多个runlist，实验测试compute和copy是否可独立调度）
  3. **执行与监控**：使用exec_logger（持续执行compute kernel并记录每次执行的时间戳，微秒精度）和copy_monitor（类似，但仅使用copy engine）同时运行，通过nvdebug实时查看runlist状态
  4. **规则推导**：从时间线数据推断调度行为模式（如：copies在单runlist上约1ms timeslicing；compute在独立runlist上不受copy干扰；单runlist上compute中断copy的间隔=compute timeslice而非copy timeslice）
  5. **跨GPU验证**：在9款GPU（Pascal~Ada Lovelace，2016-2022）上重复实验，确认规则适用性

  nvdebug的硬件访问原理：绕过GPU驱动，直接通过memory-mapped I/O (MMIO)访问GPU寄存器。对于runlist等位于GPU物理内存中的数据结构，nvdebug通过解析GPU页表（GMMU page tables）进行虚拟地址→物理地址转换后直接读取。这使得nvdebug可以在任何驱动（包括无驱动）下工作。
