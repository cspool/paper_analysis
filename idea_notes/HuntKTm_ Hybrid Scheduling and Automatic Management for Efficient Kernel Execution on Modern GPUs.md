## HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

- baseline方法是什么？
  Baseline是三种互不协作的GPU并发方案：(1) 仅kernel-level并发（CKE within single application）——如Taskflow要求用户显式指定kernel依赖并手动分配到stream，GrSched引入Python DSL的runtime动态调度但开销大且性能差；(2) 仅task-level并发（space-sharing across applications）——如CASE在multi-GPU系统上编译期静态分析资源需求并runtime动态调度task到GPU，但不做kernel内部并发优化，也不做memory management；(3) 手动编程的async并发——开发者手动使用CUDA stream/event API编写并发代码，需要大量专业知识和编码劳动。
  
  全栈执行例子（以CASE为例，运行W4 workload）：
  - 算法层：multi-kernel GPU程序（如M2包含多个activation和reduction kernel）以串行方式执行，每个kernel独占所有SM资源，但SM occupancy仅<10%（memory-intensive kernel的典型特征）。
  - 系统框架层：CASE runtime分析task资源需求，将task分配到有足够内存的GPU。但task内部kernel串行执行，无法利用kernel间并发。
  - 编译框架层：CASE在LLVM编译期分析资源需求，但不做DA分析也不修改kernel执行流。无memory management。
  - kernel调度层：每task内的kernel在同一default stream上串行执行，GPU硬件利用率低。仅靠不同task的kernel通过MPS space-sharing在同一GPU上并发，但内存容量限制了可并发task数。
  - 硬件架构层：4×A100 GPU，每GPU 40GB HBM。W4 workload中大量M1/M2/B&S应用（内存>8GB），仅能2个task同时运行。SM有大量idle周期。

  Baseline缺陷：(1) 仅kernel-level或仅task-level并发无法充分饱和GPU资源——即使GPU utilization=100%，SM occupancy仍可能<10%（memory-intensive kernel场景）；(2) 内存容量是task并发的瓶颈，不解决内存占用就无法增加并发task数；(3) 手动编程负担大——Taskflow需显式声明依赖（平均27 LoC），GrSched需重构到Python DSL（平均127 LoC），Async需手动管理stream/event（平均40 LoC）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出HuntKTm——一个结合kernel-level和task-level混合调度与自动内存管理的LLVM编译框架，包含三个协同组件：
  (1) **Stream Scheduler**：编译期自动分析kernel间RAW/WAR/WAW数据依赖，构建DFG后通过PP-Set启发式算法分配kernel到多stream实现kernel-level并发，并通过依赖传递性和串行隐式同步剪除冗余barrier。解决"需手动编写CKE代码"问题——仅需每kernel一行writable参数标注（平均15 LoC vs Async的40 LoC）。
  (2) **Task Scheduler**：编译期resource analyzer +运行时lazy engine收集task资源需求（threads/registers/shared memory/memory），task dispatcher基于三维度SM可用量评估选择最优GPU。解决"多task自动放置和负载均衡"问题——将task从编译期的资源分析到运行时的精确调度无缝衔接。
  (3) **Memory Manager**：在stream graph上执行memory object liveness分析，通过Algorithm 2推迟allocation到live range起点、提前free，将memory object lifetime缩短至live range，使非重叠lifetime的memory object可复用同一内存区域。解决"内存容量是并发瓶颈"问题——平均减少22.3% peak memory（M2从17.6GB→11.2GB，减少36.4%）。

  全栈执行对比baseline（以W4 workload运行M2为例）：
  - 算法层：同一M2应用（包含FasterTransformer的activation和reduction kernel），kernel间天然依赖可通过DFG自动发现后并行化。
  - 系统框架层：stream scheduler自动构建M2 DFG（宽度=6），分配kernel到6个stream并发执行，同步剪除后仅保留最少barrier。task scheduler同时将多个task（含其他应用的multi-stream版本）动态分发到4×A100 GPU上。
  - 编译框架层：LLVM pass自动转换M2源码：DFG constructor→kernel distributor→synchronization generator→memory manager（liveness分析+延迟allocation）→resource analyzer（nvcc获取register/shared memory）→function wrapper（注入lazy engine拦截逻辑）。
  - kernel调度层：M2的kernel从串行执行变为6-stream并发，同时与其他task（B&S、IMG等）的kernel通过MPS在同一GPU上space-sharing。lazy engine在task调度确定GPU后，顺序执行deferred CUDA操作（cudaMallocAsync→cudaMemcpyAsync→kernel launch），memory pool减少频繁alloc/free开销。
  - 硬件架构层：4×A100 GPU，HuntKTm在W4 workload下FP32 utilization提升3.54x、memory bandwidth utilization提升2.83x、SM occupancy提升2.47x（vs SA）。Memory management使system memory从232.3GB降至173.9GB，更多task可同时运行。最终HuntKTm比CASE throughput提升33.2%。
