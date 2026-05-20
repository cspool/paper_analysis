## High Throughput and Low Latency LLM Serving via Adaptive KV Caching

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  论文在eLLM系统的layer-level implementation中引入kernel调度优化，核心包括：(1) layer-wise kernel fusion：将下一层未缓存token的KV recomputation kernel K1与当前层当前token的decode attention kernel K2融合为单一fused kernel，K1是compute-intensive GEMM（对旧token重算K/V projection），K2是memory-intensive attention decode（对当前token做MHA/GQA），融合后减少kernel launch overhead并提升SM utilization；(2) dual CUDA stream异步执行：通过torch.cuda.stream启动两个CUDA stream，Stream A负责host↔GPU cached KV数据传输，Stream B负责K1+K2 fused computation，实现communication-computation overlapping；(3) 动态线程分配：预编译多组CUDA shared libraries，对recomputation和decoding kernels生成32到1024、步长32的线程配置，运行时根据layer-level K1/K2估计计算量比例选择合适.so，线程数调整为32的倍数匹配NVIDIA warp granularity；(4) layer-granular KV block管理：将vLLM的粗粒度KV block按F个连续layer划分为更小单元（默认F=4），维护map table用于runtime精确定位每个token在每个layer的KV cache状态，减少内存碎片。实验比较：(a) ablation study：禁用Kernel Fusion后TPOT和throughput退化，禁用Comm-Com Overlapping后退化更明显（因PCIe 4.0 x16带宽成为瓶颈）；(b) 融合vs非融合kernel的SM utilization对比，fused kernel显著减少GPU idle bubble；(c) different F values对性能影响，F=4在映射开销和碎片间取得最佳平衡。

- 后端平台是什么，配置是什么。
  NVIDIA A100-80GB GPU，PCIe 4.0 x16连接（无NVLink），CUDA 12.4，NVIDIA Driver 550.107.02。Llama2-13B单卡MHA，Llama2-70B四卡tensor parallel GQA。

- 评估性能的软件/脚本是什么。修改了什么。
  评估软件：基于vLLM框架的内置profiling机制和torch.cuda.event计时。修改：(1) 预编译CUDA shared libraries：对recompute K1和decode K2 kernel预生成32-1024 thread stepped by 32的多组.so文件；(2) torch.cuda.stream异步CUDA stream管理重构：替代vLLM原有同步执行路径；(3) fused kernel launch：将K1(GEMM)+K2(attention)合并为single kernel launch，内部按计算量比分配thread blocks；(4) layer-granular KV map table runtime lookup：O(1) hash-based查找替代原有粗粒度block遍历。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源状态：eLLM整体未找到官方GitHub仓库，kernel-level CUDA代码属于eLLM系统内约1,700行CUDA代码的一部分。kernel调度使用例子（以Llama2-13B单卡decode为例）：
  1. 编译阶段：对Llama2-13B的MHA attention，论文预compile K1(GEMM for old token K/V projection)和K2(MHA decode attention)的fused kernel variant，线程数从32到1024 step 32（共32组.so），覆盖不同layer维度（hidden_size=5120, num_heads=40）和不同uncached token count的workload组合。
  2. 调度阶段：request-level optimizer确定b和r后，layer-level读取每个layer需要重算的uncached token数（r × 历史token数），估算K1 FLOPs = 2 × hidden_size × head_dim × num_heads × num_uncached_tokens，K2 FLOPs = O(hidden_size × num_cached_tokens)，计算FLOP ratio = K1_FLOPs/K2_FLOPs。
  3. 线程分配：按FLOP ratio分配thread blocks——若K1占总FLOP 70%则分配约70% threads给K1子任务、30%给K2子任务，总threads取32的倍数。从预编译.so库中加载最接近目标thread数的variant。
  4. 运行时：Stream A异步执行cudaMemcpyAsync将host memory中swapped KV传输到GPU（对当前layer i+1），Stream B同时执行fused kernel：K1子kernel为layer i+1的uncached old token执行attention projection生成KV（临时存于workspace buffer），K2子kernel用layer i已准备好的完整历史KV（cached+recomputed）对current token执行decode attention。两个kernel共用一次launch，内部thread blocks按计算量比例分区，完成MHA输出。K1产生的临时KV在K2使用后立即释放。
  5. 同步：每layer结束时cudaStreamSynchronize对齐Stream A和Stream B。论文消融显示kernel fusion使TPOT降低而额外显存开销约1 layer的KV workspace。

