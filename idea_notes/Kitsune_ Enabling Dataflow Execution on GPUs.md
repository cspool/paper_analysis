## Kitsune: Enabling Dataflow Execution on GPUs

- baseline方法是什么？
  Baseline是GPU上两种现有的DL执行范式：(1) Bulk-Synchronous Programming (BSP)——每个DL算子映射为单个CUDA kernel，kernel独占GPU运行直到所有CTA完成后才launch下一个kernel，通过global barrier串行化执行；(2) Vertical Fusion（垂直融合）——将多个DL算子融合为单个"mega kernel"，在单个CTA内temporal multiplexing各算子的部分执行，通过shared memory/register file传递tile级中间数据。商业代表：TensorRT，学术代表：Welder、AStitch、Chimera。

  Baseline三大缺陷（对应Kitsune的三个untapped opportunities）：
  1. **资源闲置**：垂直融合的temporal multiplexing导致任一时间点只有TensorCore或SIMT core之一活跃，另一资源空闲。实测数据显示inference中20-25%、training中37-67%的runtime中SM和DRAM利用率均低于33%峰值。
  2. **大intermediate spilling到DRAM**：当intermediate的hidden dimension较大时（如MLP hidden dim ≥ 768 on A100），即使垂直融合的tile也超过shared memory capacity（192 KB/SM），不得不spill到off-chip DRAM。A100 round-trip DRAM latency ≈ 409ns (572 cycles @ 1.4GHz)。若通过多CTA/SM来增加并行度会进一步分割shared memory，加剧容量问题。
  3. **无法利用reduction/hidden维度并行**：Back-propagation中gradient reduction over batch dimension（split-K GEMM类似）产生少量CTA执行reduction，绝大多数SM空闲。垂直融合不支持back-propagation。

  全栈执行例子（以BSP执行MLP forward pass：Linear(768→3072) → ReLU → Linear(3072→768)为例）：
  - 算法层：三个DL算子顺序执行，每个算子的kernel独占所有SM资源。
  - 系统框架层：PyTorch eager mode依次dispatch kernel_Linear1 → kernel_barrier → kernel_ReLU → kernel_barrier → kernel_Linear2。每次barrier后中间结果由DRAM写入/读出。
  - 编译框架层：无融合优化，或TensorRT将ReLU作为Linear1的epilogue融合（垂直融合），但两个Linear因intermediate过大（3072×batch×4B ≈ 大tile）spill到DRAM。
  - kernel调度层：单个kernel独占GPU时，TensorCore执行GEMM期间SIMT core空闲，反之亦然。108 SM的A100上仅一种类型资源被利用。
  - 硬件架构层：intermediate数据在DRAM↔L2↔SM的shared memory间反复搬运，消耗大量DRAM bandwidth。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Kitsune——通过两个互补的SW/HW原语在GPU上实现synchronous dataflow执行：
  **(1) 软件Ring Buffer Queue（§4.1）**：纯软件实现的inter-CTA通信队列，通过L2 cache + global atomics在多CTA间传递tile级数据。Queue为双buffer设计，使用sequence number实现producer/consumer同步，metadata由atomic操作保护。Queue数据保持在L2 cache中（通过CUDA API pinning），避免了DRAM round-trip。
  **(2) Modified GPU Grid Scheduler（§4.2）**：将GPU的单round-robin arbiter扩展为双arbiter（SIMT和Tensor各一个），通过cudaPipeline API标注每个kernel的primary resource type，使grid scheduler能将不同类型的CTA配对到同一SM，实现fine-grained co-execution。
  基于这两个原语，Kitsune编译器将DL计算图自动lowering到spatial pipeline——不同算子映射到不同CTA，通过on-chip queues传递tile级中间数据，不同类型CTA在SM上co-locate，实现空间上的并发执行（而非时间上的multiplexing）。

  对应解决Baseline三大缺陷：
  1. **解决资源闲置**：Modified grid scheduler使SIMT-heavy CTA（如ReLU）和TensorCore-heavy CTA（如Linear GEMM）配对到同一SM，TensorCore执行矩阵乘的同时SIMT core执行elementwise。Kitsune将runtime中"both low utilization"从26%降至15%（inference），44%降至18%（training）。
  2. **解决大intermediate spilling**：Dataflow下，大hidden dimension被split到多个CTA并行处理，每个CTA仅需容纳一个tile（64-256KB），无需在shared memory中存储整个intermediate。Tiles通过on-chip queue直接传递，无需经DRAM。DRAM traffic减少41-98%（inference）、16-42%（training）。
  3. **解决reduction并行不足**：Dataflow下reduction操作通过queue构建reduction tree（多对一通信），将并行度从单CTA扩展到多CTA/SM。Back-propagation中的gradient reduction成为并行reduction pipeline stage。

  全栈执行对比baseline（以Kitsune执行同一MLP forward pass：Linear(768→3072) → ReLU → Linear(3072→768)为例）：
  - 算法层：同一MLP计算，Kitsune将其spatial fusion为一个sf-node（包含3个stage的spatial pipeline）。
  - 系统框架层：PyTorch Dynamo → Kitsune compiler backend → 生成cudaPipeline（3 kernels + 2 queues）。cudaPipeline启动后，3个kernel的CTAs co-resident在GPU上，不再需要中间barrier。
  - 编译框架层：Kitsune compiler执行：Subgraph Selection（模式匹配识别Linear→ReLU→Linear chain）→ Pipeline Design（在Linear1/ReLU间插入queue0，ReLU/Linear2间插入queue1）→ Load Balance（ILP求解：Linear1分配64 Tensor CTAs，ReLU分配44 SIMT CTAs，Linear2分配44 Tensor CTAs，利用SIMT/Tensor的重叠将152 CTAs压缩到108 SM预算内）。
  - kernel调度层：CUDA kernel改写——Linear1 kernel原来`store C[id]`到global memory → 改为`wr_acquire(queue0, tile_id)` → 写入queue → `wr_release(queue0)`。ReLU kernel原来从global memory读取 → 改为从queue0 `rd_acquire/release`获取tile。同一SM上，Linear1 CTA使用TensorCore执行GEMM的同时，ReLU CTA使用SIMT core执行elementwise。
  - 硬件架构层：Intermediate tile数据流：Linear1_CTA → L2-resident queue0 → ReLU_CTA → queue1 → Linear2_CTA。全程无DRAM访问。Modified grid scheduler的双arbiter机制确保每个SM同时有Tensor和SIMT CTA。当2× L2 bandwidth和2× SM时，Kitsune额外获得47%（inference）和27%（training）加速比，而baseline仅18-26%。

  设计思路核心：Kitsune的本质是将DL计算图中operator间的**时间串行**（temporal BSP/vertical fusion multiplexing）转换为**空间并行**（spatial dataflow pipeline）——仅需两个最小化原语（软件queue + 修改grid scheduler），无需clean-slate架构。这证明在现有GPU架构上的"modest adjustments"即可实现高效的dataflow执行。
