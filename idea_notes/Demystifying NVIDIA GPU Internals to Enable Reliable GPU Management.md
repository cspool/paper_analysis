## Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management

- baseline方法是什么？
  已有三种主要的实时GPU管理/分析方法，均因对NVIDIA GPU硬件调度行为的理解不完整而存在结构性缺陷：

  **(1) Management-Free Analysis（Yang et al. [4]）**：通过限制编程模型（每job仅一个kernel、单CUDA stream）来简化分析，给出不需要GPU管理中间件的响应时间bound。基于Amert et al. [5]的简化调度规则：(i) kernel launch时入EE FIFO队列；(ii) head-of-queue kernel被fully dispatched后dequeue；(iii) block在资源满足时eligible for assignment。全栈执行例子（10个OpenVX DAG任务，每任务含1个kernel in 1 stream, NVIDIA Jetson GPU）：
  - **算法层**：OpenVX vision graph → 每节点编译为单个CUDA kernel → 无跨kernel依赖的任务模型。论文未明确说明具体算法层修改。
  - **系统框架层**：无GPU管理中间件——task直接通过CUDA API launch kernel到各自的stream，依赖GPU原生调度。OpenVX runtime负责任务间依赖管理。
  - **编译框架层**：论文未明确说明。OpenVX graph编译器→CUDA kernel，无编译器级优化。
  - **kernel调度层**：Kernel通过CUDA stream提交到GPU → GPU HW scheduler按channel→runlist→engine dispatch chain处理 → Yang et al.假设所有kernel按FIFO顺序dequeued from EE queue → 实际上当stream数>channel数时（R2），后续kernel的channel assignment非FIFO（Corollary 2），导致later-released kernel可cut-ahead先执行。
  - **硬件架构层**：NVIDIA Jetson TX2/Xavier等嵌入式GPU。默认每个context仅2-4个compute channel（R2）——无法同时支持超此数目的独立stream。Yang et al.的分析未限制co-running job数量。

  Baseline缺陷：当co-running job数（即stream数）> GPU compute channel数时，(i) 额外stream产生false dependency（R2/Corollary 1），kernel需等任意channel释放而非FIFO调度；(ii) channel assignment非FIFO（Corollary 2），later-released kernel可能先于earlier-released kernel执行 → 响应时间bound proof（Theorem 3）假设的FIFO dequeue顺序被打破 → 未计入的额外delay导致deadline miss。

  **(2) Preemptive EDF via Runlist Management（Capodieci et al. [3]）**：通过只将一个应用的channel放在runlist上实现任务调度，通过重置runlist实现抢占。基于TX2特有属性：仅一个runlist，并错误地将此概括为所有NVIDIA GPU的通用规则。全栈执行例子（2 GPU-using tasks on Jetson Xavier, Task1=compute+copy, Task2=compute-only, EDF调度）：
  - **算法层**：EDF（Earliest Deadline First）调度算法 → 根据absolute deadline确定优先级。Task1: period=3s, deadline=2s, cost=2s；Task2: period=3s, deadline=3s, cost=1s。
  - **系统框架层**：Capodieci et al.的runlist管理机制——CPU端维护task队列，运行时按EDF选择最高优先级task将其channel插入runlist → 抢占通过重置runlist实现。假设仅一个runlist存在（因此抢占compute runlist后task不再执行任何GPU工作）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：Runlist管理假设抢占compute runlist → task停止所有GPU活动 → 但Jetson Xavier有第二个runlist（copy-only）（R5）→ Task1的copy操作在第二个runlist上不受抢占影响继续执行 → copy操作干扰Task2的compute（已在Jetson Xavier上被Olmedo et al. [7]证明严重延迟compute）。
  - **硬件架构层**：NVIDIA Jetson Xavier（Volta embedded）——含有至少2个独立runlist（R5, R7），而非Capodieci et al.假设的1个。

  Baseline缺陷：(i) 将TX2的单runlist配置错误泛化为所有GPU的通用规则；(ii) 未考虑其他非compute engine（copy engine等）需要独立runlist访问（R7）；(iii) 抢占仅作用于一个runlist，其他runlist上task的未完成操作继续执行并干扰被抢占到的task → 响应时间bound失效、deadline miss。

  **(3) Granular GPU Locking（Elliott et al./GPUSync [2, 14]）**：将每个compute engine和copy engine视为独立可锁资源，应用mutual-exclusion lock，使已有资源管理分析可直接应用于GPU-using task。全栈执行例子（RTX 6000 Ada, 2 GPU-using tasks: Task1=CPU→GPU copy+graphics, Task2=GPU→CPU copy）：
  - **算法层**：实时资源管理analysis（如suspension-based locking protocol）→ treat GPU engines as lockable resources。
  - **系统框架层**：GPUSync框架——每个engine对应一个lock → task需acquire对应engine的lock才能使用 → release后下一task可acquire → 假设不同方向copy engine（CPU→GPU vs GPU→CPU）为独立资源→创建Lock1(CPU→GPU copy), Lock2(GPU→CPU copy), Lock3(compute)。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：Lock1和Lock2被视为非竞争资源 → Task1和Task2可同时分别持有Lock1和Lock2并行执行 → 但实际上，RTX 6000 Ada的GRCE0和GRCE1均映射到同一LCE→同一PCE（R8）→ 两task的copy操作竞争同一个物理copy engine→ copy时间翻倍。
  - **硬件架构层**：NVIDIA RTX 6000 Ada——CUDA报告2个copy engine但底层LCE→PCE映射（Fig.11）显示GRCE共享PCE配置 → 实际只有1个独立的物理copy path可用。

  Baseline缺陷：(i) 假设多个copy engine提供独立copy path（R8推翻——GRCE通过共享PCE破坏独立性）；(ii) 假设copy engine和compute engine总是独立调度（R6推翻——可共享runlist）；(iii) 未考虑硬件LCE→PCE间接映射层（需通过nvdebug的lce_for_pce和shared_lce_for_grce接口检查实际配置）→ execution time bound被打破→ unreliably-met deadlines。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法是通过**深度硬件逆向工程**实验性推导出**8条NVIDIA GPU调度规则**（R1-R8），覆盖从GPU library到on-engine-dispatch的全链路调度行为。不提出新的GPU管理方法，而是提供所有safe GPU管理方法必须遵守的硬件调度ground truth。

  核心贡献对应关系：

  **(a) 实验方法学：组合硬件内省+微基准测试** → 解决baseline对GPU硬件行为的不完整/错误假设
  - **nvdebug内核模块**：绕过GPU驱动，通过MMIO直接读写GPU寄存器，解析GPU页表访问runlist物理内存。暴露PTOP寄存器（engine-runlist拓扑）、PCE-LCE映射寄存器、channel状态等此前仅存在于NVIDIA内部文档的硬件信息。
  - **gpu-microbench（exec_logger + copy_monitor）**：精密计时（微秒级）的定向微基准测试，精确测量特定engine的调度时间线。通过组合测试（如同时运行compute-only和copy-only任务）交叉验证硬件行为。
  - **跨9 GPU验证**：在Pascal到Ada Lovelace的5代架构、9款GPU（含嵌入式Tegra和离散GPU）上重复所有实验，确保规则的普适性。

  **(b) 规则体系：3层调度规则→纠正baseline假设**
  - **Channel层规则（R1-R2）**→ 纠正Yang et al. [4]的假设：
    - R1: 所有GPU操作必须经过channel → 提出disable_channel验证方法
    - R2: Channel数限制并行度（x86_64默认8, Jetson默认2-4）→ Fig.5实验证明了超过channel数时false dependency和non-FIFO channel assignment（Corollary 1, 2）→ 在Evaluation §VI-A中给出违反R2的counter-example：10 jobs → 8个先执行 → kernels 9和10竞争释放的channel → 非FIFO → Yang et al.的response-time bound proof被打破

  - **Runlist层规则（R3-R5）**→ 纠正Capodieci et al. [3]的假设：
    - R3: Channel必须在runlist中 → 与Capodieci et al.一致
    - R4: 每runlist最多一个task per engine active → 既支持又挑战Capodieci et al.——单engine task互斥（正确），但多engine task可在单runlist上co-run（他们未考虑）
    - R5: Runlist数量限制独立inter-task并行度 → 多runlist支持独立调度（Fig.8），单runlist导致跨引擎干扰（Fig.9, Jetson TX2）→ 在Evaluation §VI-B中给出counter-example：Jetson Xavier有2个runlist → 抢占compute runlist不能阻止copy在另一个runlist上继续 → 被抢占task的copy干扰EDF高优先级task的compute → deadline miss

  - **Engine映射层规则（R6-R8）**→ 纠正Elliott et al./GPUSync [2, 14]的假设：
    - R6: 一个runlist可绑定多engine → copy和compute可能共享runlist（Table II-IV都显示Runlist 0同时包含Compute和GRCE）→ 非独立的copy和compute调度
    - R7: 每个engine只绑定一个runlist → PTOP寄存器约束 → 确保引擎不会在多个runlist上被重复调度
    - R8: LCE→PCE硬件间接映射层（GRCE可共享PCE）→ Fig.10实验（RTX 6000 Ada上OpenGL texture upload使CUDA GPU→CPU copy减速2×）+ Fig.11 LCE/PCE/GRCE映射图 → 在Evaluation §VI-C中给出counter-example：GPUSync为2个copy engine创建2个lock → 但实际GRCE共享PCE → 一个物理PCE被竞争 → copy时间翻倍

  **(c) 工具开源** → 使其他研究者可以验证和扩展规则
  - nvdebug开源（http://rtsrv.cs.unc.edu/cgit/cgit.cgi/nvdebug.git/）：支持Kepler(2011)至Ada(2022)全系列GPU，同时支持x86_64和aarch64，零依赖，无需配置
  - gpu-microbench开源（https://www.cs.unc.edu/~jbakita/rtas24-ae/）：经多年专家级调优和bug修复的微基准测试套件

  全栈执行例子（基于论文8条规则的correct GPU management, Jetson Xavier, 2 tasks, EDF）：
  - **算法层**：同baseline EDF调度算法 → 但调度决策基于正确硬件模型
  - **系统框架层**：改进的runlist管理——不仅管理compute runlist（对应Compute/Graphics engine），也必须管理copy runlist（对应Copy Engine）。抢占时需要同时重置所有相关runlist，确保被抢占task的所有engine操作停止
  - **编译框架层**：论文未明确说明
  - **kernel调度层**：基于R5→在Jetson Xavier上，compute和copy的独立runlist意味着抢占compute runlist后copy继续 → 正确的管理必须同时preempt copy runlist → 消除copy对compute的干扰
  - **硬件架构层**：通过nvdebug的device_info获取正确的engine-runlist拓扑→根据R7（每engine单runlist）和R8（检查GRCE→PCE映射）确定引擎独立性→创建正确数量的lock（而非盲目按CUDA报告的copy engine数创建lock）
