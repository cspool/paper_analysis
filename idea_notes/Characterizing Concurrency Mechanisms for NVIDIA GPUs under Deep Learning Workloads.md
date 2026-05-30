## Characterizing Concurrency Mechanisms for NVIDIA GPUs under Deep Learning Workloads

- baseline方法是什么？
  Baseline是三种NVIDIA GPU现有的application-level并发机制，各自存在结构性缺陷：

  **(1) Priority Streams**：两task在同一进程的不同CUDA stream上运行，stream可设三级优先级（-2/1/0），thread block scheduler优先从高优先级stream取blocks。但**不抢占已执行blocks**。全栈执行例子：
  - **算法层**：PyTorch/TensorFlow模型编译为CUDA kernel序列（convolution、SGEMM、batch norm等），inference请求按MLPerf server mode（Poisson arrival）或single-stream mode（连续）到达。
  - **系统框架层**：CUDA runtime创建高优先级stream（inference）和低优先级stream（training），同一进程内异步提交kernel。无Serving框架，直接CUDA API。
  - **编译框架层**：论文未明确说明。使用PyTorch/TensorFlow默认compilation pipeline（torch.compile或TF XLA，论文中未描述具体编译过程）。
  - **kernel调度层**：Thread block scheduler采用leftover policy（最近到达kernel的所有blocks优先调度完）和most-room placement（选剩余资源最多的SM放置blocks）。Warp scheduler可能使用greedy-then-oldest或loose round-robin，且官方文档未说明priority streams如何与warp scheduling交互——可能导致warp scheduler实际"去优先级化"高优先级blocks。
  - **硬件架构层**：NVIDIA GeForce RTX 3090（Ampere），82 SMs，每SM 4 warp scheduler单元（每两周期发射一条warp指令），fixed resources per SM。
  - **Baseline缺陷**：Compounded delay——inference kernel完成后、下一inference kernel到达前的窗口期，training kernel抢占GPU所有SM。下一inference kernel到达后必须等待已执行training blocks完成，造成2-4× turnarround time增加且variance大。

  **(2) Time-Slicing**：两task独立进程，CUDA application-level scheduler以约2ms固定时间片轮转分配整个GPU。**不支持spatial sharing**。全栈执行例子：
  - **kernel调度层**：GPU交替专属于单进程。时间片切换约145μs开销。Blocks可被coarse-grained抢占（整个GPU清空），但无partial preemption。Register和shared memory不传输（推测为避免高开销），导致两进程资源需求总和不能超硬件上限。
  - **Baseline缺陷**：无法spatial sharing——资源在时间片内空闲时无法被另一进程使用。Utilization最差（training time可比baseline多100+秒）。Memory transfer contention跨进程干扰（如ResNet-34 case）。时间片长度/频率不可配置。

  **(3) MPS**：MPS server调度不同CUDA context的kernels，允许SM-level spatial sharing（blocks colocation）。但**无优先级**且采用FCFS+leftover policy。全栈执行例子：
  - **kernel调度层**：MPS server接受多client kernel dispatch请求，FCFS顺序处理。Leftover policy导致后到达kernel需等待当前kernel所有blocks调度完。Per-client thread limit可设（实验中100%）。
  - **Baseline缺陷**：无优先级概念——inference和training task被同等对待。Load-balancing behavior使得training task受益多于inference task（inference degradation显著）。Compounded delay在100% thread limit时也影响MPS。大kernel（grid size超SM容量）长时间独占GPU时，inference被starved。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法是**Fine-Grained Block-Level Preemption**（细粒度thread block抢占），即thread block scheduler能在任意时刻中断任意subset of thread blocks的执行，并在之后恢复。论文未实现此机制（需要NVIDIA硬件修改），但通过10个Observations系统论证了其必要性和可行性。解决Baseline缺陷的机制：

  **解决Priority Streams的compound delay**：Inference kernel到达时立即抢占部分training blocks腾出空间，消除等待。全栈执行例子对比如下：
  - **系统框架层**：仍使用CUDA streams + priority，但thread block scheduler支持fine-grained preemption。论文未明确说明用户态API设计。
  - **kernel调度层（核心变化）**：Thread block scheduler在inference kernel到达时：(i) 选择被抢占的training blocks（基于contention-aware placement policy）；(ii) 保存被抢占blocks的context（register file、shared memory、warp state）到global memory；(iii) 将inference blocks调度到腾出的SM空间；(iv) inference kernel完成后恢复training blocks。抢占开销估算：per-SM约448KB context（128KB L1/shared + 256KB register file + 64KB constant），按11.4 GB/s per-SM带宽约37μs；或基于time-slicing实测（145μs/2切换=73μs per save）。开销可被隐藏（见下）。
  - **硬件架构层**：需在SM内增加context save/restore hardware state machine。论文建议复用现有time-slicing context-switching硬件。

  **解决MPS的FCFS无优先级问题**：Fine-grained preemption + MPS可实现"minimum resource guarantee" + "priority over-alloc"——为inference设定最小资源预留，training用剩余资源。Inference kernel到达时若资源不足，抢占training blocks达到预留阈值。

  **抢占开销隐藏策略（论文O8-O9核心贡献）**：
  (a) **利用memory transfer latency**：Preemption state save可在host-to-device memory transfer期间并行执行（DMA引擎独立于SM计算）。
  (b) **利用kernel序列特性**：当已知大kernel紧跟小kernel时，在小kernel执行期间预抢占training blocks为后继大kernel腾空间。举例（ResNet-152 Figure 8 Region B）：第一个kernel仅32 blocks × 64 threads（只占32 SM），执行时间137μs，后续512-block kernel仅需2μs。137μs足够完成抢占操作（~73μs）。
  (c) **保持空间开放**：小kernel完成后不立即用training blocks填充，直接留给下一kernel（如Region A：136-block kernel 400μs → 112-block kernel 6μs，没必要在6μs kernel执行前后做抢占）。

  **Utilization度量优化（O10）**：提出用best-effort training task execution time作为utilization proxy（优于简单的thread occupancy或SM occupancy metrics），因为同是100% thread usage的两个kernel，实际register/shared memory利用率可差异很大（如49152 vs 61440 registers per SM）。
