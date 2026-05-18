## TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling

- baseline方法是什么？
  Baseline包括三套：(1) SGLang conservative scheduling（默认FCFS/prefill优先continuous batching调度器）；(2) SGLang with chunked prefill（在SGLang上启用chunked prefill的改进版本）；(3) Andes（QoE-aware scheduling with Token Pacer，论文在SGLang中用recompute-based preemption方式实现）。

  SGLang FCFS全栈执行例子（以Llama3-8B + SGLang on H200为例）：
  - 算法层：Llama3-8B标准Transformer decoder，无算法修改。prefill阶段将prompt tokens批量处理，decode阶段自回归逐token生成。
  - 系统框架/Serving层：请求按FCFS到达顺序进入SGLang scheduler → prefill阶段获得GPU优先调度（prefill-prioritized）→ decode阶段continuous batching合并多个请求的decode step → KV cache提前分配（pre-allocation）在GPU显存中 → 请求完成后释放KV cache。在burst负载下形成head-of-line blocking：早到请求持续占用GPU生成token（约30 tokens/s），后到请求在队列中等待→TTFT激增。SGLang H200 micro-benchmark显示burst时TTFT可超过用户可接受阈值。
  - 编译框架层：论文未明确说明（SGLang默认编译路径，PyTorch + CUDA）。
  - kernel调度层：SGLang默认CUDA kernel，无定制KV cache I/O调度。普通write-back策略：仅在显存压力下被动evict KV cache到CPU memory。
  - 硬件架构层：NVIDIA H200/RTX 4090/A6000 GPU。GPU显存直接持有KV cache，CPU memory作为被动溢出空间。

  Baseline的根本缺陷：
  (1) 调度与用户消费速率脱节：FCFS/prefill优先调度不感知"用户只按固定速率消费token"这一text streaming特性。早到请求即使已生成大量超出用户阅读/听取速率的token，仍继续占用GPU→资源错配。过快生成的token堆在客户端buffer中无实际价值→raw throughput虚高，effective throughput低。
  (2) Head-of-line blocking在burst下恶化：突发请求到达时，已在运行的请求持续占用GPU和KV cache→新请求排队→TTFT膨胀。SGLang在burst load上升时TTFT可超过用户可接受阈值。
  (3) 抢占无KV cache协同：Andes引入QoE-aware scheduling和Token Pacer改善感知延迟，但其抢占机制带来频繁context switch，缺少与KV cache管理的协同。直接抢占会把KV cache搬移变成显存和I/O瓶颈，吞吐和资源利用受损。
  (4) Compute-bound到memory/I/O-bound的相变：随着更多请求交替在GPU上运行，KV cache总量超过GPU显存容量→系统从compute-bound转向memory/I/O-bound→PCIe带宽成为瓶颈。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  TokenFlow提出buffer-aware preemptive scheduling + hierarchical KV cache management，把调度目标从最大化raw token throughput改为最大化effective throughput和响应性。核心由三段协同设计构成：

  (1) Buffer-Aware Two-Step Scheduling：第一步determine working set——根据GPU显存、请求KV footprint、等待队列长度、I/O队列长度和buffer安全条件决定可过量承诺的请求集合。第二步buffer balancing——在working set内按buffer size、weighted token generation quantity、required output rate和内存约束做greedy selection，再用局部交换优化优先级。请求buffer越低、用户消费速率越高，越容易获得运行优先级；buffer较高的请求可被暂时抢占，继续由客户端buffer平滑输出。

  (2) Streaming QoS Metric：用effective throughput替代raw throughput作为核心优化目标。effective throughput根据text streaming体验对token加权：buffer小于总输出长度10%的token全计（用户急需消费），10%-20%之间线性衰减，超过20%的token不计入有效吞吐（用户尚未消费到的"过早token"无实际价值）。

  (3) Hierarchical KV Cache Manager：GPU显存作为CPU memory上大容量KV cache的高速cache。Write-through策略（每次decode iteration同步新KV chunk到host）替代普通write-back策略（仅在抢占时写回）→抢占时大部分KV cache已在host同步，剩余chunk通过load-evict overlap与load stream重叠传输。三并行CUDA stream pipeline（compute/load/evict）+ 动态chunk sizing + batched transfer + CUDA events协调非阻塞执行。

  对应解决Baseline缺陷的具体设计：
  - 对缺陷(1)：TokenFlow把"请求是否继续占用GPU"变成随buffer实时变化的在线决策。Scheduler根据每个请求的实时buffer token数、用户token消费速率和I/O状态决定admission/preemption/resumption。buffer较低或尚未获得首token的请求优先获得GPU→让GPU时间投入到"用户真正要消费的token"。
  - 对缺陷(2)：预占式调度让GPU在早到请求积累足够buffer后安全转向紧急请求（新到达或即将耗尽buffer的请求）。利用客户端buffer覆盖切换延迟→burst场景下P99 TTFT最多降低80.2%，mean TTFT最多降低48.4%。
  - 对缺陷(3)：Scheduler和KV Cache Manager双向协作——scheduler在决策时考虑I/O overhead和recompute/load代价，memory manager在后台提前write-through可能被抢占请求的KV cache。真正发生preemption时不必完整同步所有cache→抢占恢复延迟大幅降低。
  - 对缺陷(4)：Write-through策略将KV cache持续同步到CPU memory，GPU显存作为高速cache。当working set超过GPU显存时，evict/load操作通过CUDA streams与compute overlap，避免同步I/O stall。消融实验：去掉offload完成时间从66.00s恶化到127.28s（恶化93%），说明分层内存管理是收益核心。

  论文方法全栈执行例子（以TokenFlow + Llama3-8B + H200为例）：
  - 算法层：Llama3-8B标准Transformer decoder，无算法修改。控制面新增Streaming QoS metric（effective throughput加权公式）和buffer safety condition（buffer >= 切换延迟 × 消费速率）。
  - 系统框架/Serving层：TokenFlow基于SGLang ~3000行Python代码。五组件协同——Request Tracker（实时追踪buffer/消费速率/资源占用）→ Buffer-aware Scheduler（两步调度：working set determination + buffer balancing，周期性reschedule）→ Request Offload Manager（请求状态转移管理）→ LLM Executor（SGLang continuous batching engine）→ Hierarchical KV Cache Manager（write-through + 三CUDA stream并行pipeline）。请求生命周期包含多次可能的pause/resume循环：早到请求积累buffer → 安全抢占 → GPU转向紧急请求 → 原请求buffer耗尽前恢复 → 循环。
  - 编译框架层：论文未明确说明（SGLang默认PyTorch + CUDA编译路径）。
  - kernel调度层：Hierarchical KV Cache Manager使用PyTorch CUDA stream API管理三并行stream（compute/load/evict）。Write-through在每次decode iteration后将新KV chunk写入write buffer并异步同步到host。Load-evict overlap通过CUDA events协调：preempted请求已同步chunk直接释放，未同步chunk与load操作重叠。动态chunk sizing根据compute duration预估选择传输大小，最大化compute-I/O overlap。
  - 硬件架构层：NVIDIA H200/RTX 4090/A6000 GPU + host CPU memory。GPU显存为高速cache层，CPU memory为大容量KV cache存储层。PCIe作为KV cache搬移通道，write-through策略持续占用PCIe带宽但换取抢占时更低的上下文切换延迟。Huawei Ascend 910B也被报告支持。
