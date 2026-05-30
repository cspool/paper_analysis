## Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是MPK in-kernel parallel runtime——完全嵌入在单个mega-kernel内部的并行运行时系统，包含以下核心组件：(1) SM分区：将GPU的SM物理划分为workers（每个SM一个worker，维护独立task queue）和schedulers（warp粒度，每个SM 4个scheduler warp，维护event queue）；(2) Event-driven execution model：tGraph从start event开始，scheduler dequeue event后dispatch依赖该event的所有tasks到workers，worker执行完成后notify triggering event，event接收足够trigger次数后激活并入队scheduler event queue，循环推进直至所有tasks完成；(3) Hybrid task launch：JIT（Just-In-Time）模式——scheduler在event激活后才assign task，适应workload imbalance但需要worker↔scheduler两次同步；AOT（Ahead-Of-Time）模式——预分配tasks到workers，worker仅需等event激活即可执行，仅需一次同步，消除dispatch开销但缺乏动态负载均衡；(4) Cross-task software pipelining：将每个task分解为pre-loading phase（TMA异步加载数据到shared memory）和compute phase（Tensor Cores/CUDA Cores计算），在当前task compute阶段未结束时即启动下一task的pre-loading，条件是当前task已发出所有data-transfer指令且有足够shared memory page；(5) Paged shared-memory abstraction：将shared memory分为固定大小pages（32KB），task需acquire/release pages，支持跨task的shared memory复用和数据prefetching；(6) Task description prefetching：每个task描述符352 bytes存储在device memory，worker prefetch到shared memory隐藏访问延迟。

  实验比较了MPK vs SGLang/vLLM/PyTorch在单GPU和多GPU（tensor parallelism）下的吞吐量。消融实验：(a) cross-task pipelining对Qwen3-8B final linear layer的影响（B200，1.2-1.3x加速）；(b) compute-communication overlap对Qwen3-1.7B在4×H100 TP下的per-iteration延迟影响（1.1x加速）。

- 后端平台是什么，配置是什么。
  NVIDIA A100（108 SMs, 104 workers + 4 schedulers [16 warps]）、H100（132 SMs, 128 workers + 4 schedulers）、B200（148 SMs, 144 workers + 4 schedulers）。Shared memory page size: 32KB → A100: 5 pages/SM, H100/B200: 7 pages/SM。多GPU：NVIDIA H100 DGX（8×H100, NVLink），tensor model parallelism。精度：bfloat16。Task描述符：352 bytes/task。Task队列：GPU device memory circular buffer，使用atomicAdd操作。Worker-scheduler同步：device memory semaphores。Inter-GPU通信：NVSHMEM nvshmem_signal_wait_until。

- 评估性能的软件/脚本是什么。修改了什么。
  评估在offline batched-inference设置下进行（消除request-arrival变异性），所有请求prompt_len=64、decode 1024 tokens，batch size 1-16。使用PyTorch + torch.compile(backend=MPK)生成mega-kernel并直接测量吞吐量。对比系统SGLang和vLLM使用各自默认配置（FlashInfer/FlashAttention + cuBLAS/cuTLASS + CUDA Graphs）。多GPU实验使用Megatron-LM tensor model parallelism，AllReduce由NVSHMEM实现。

  MPK修改/新增：
  (1) In-kernel page allocation和request scheduling——将传统CPU端的continuous batching逻辑全部移入mega-kernel内的单个task执行：在start event处理时，scheduler (a) 移除上一iteration完成的请求，(b) 接纳新到达请求，(c) 更新per-request KV-cache metadata。消除CPU-GPU同步延迟。
  (2) 支持dynamic batch sizes——编译器为2的幂次batch sizes（up to max_batch）分别生成专用tGraph，运行时按当前batch size选择。
  (3) Moe hybrid workload balancer——编译期静态分配expert-specific tasks，运行期利用topk-softmax产生的global metadata（activated experts数、per-expert token数）动态调整workload分配。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源链接：https://github.com/mirage-project/mirage

  评估原理（单GPU offline batched inference throughput测量）：
  1. 模型加载：通过HuggingFace Transformers加载模型架构，bfloat16精度
  2. Mega-kernel编译：torch.compile(backend=MPK) → 生成针对(batch_size, GPU_type)的专用tGraph和mega-kernel
  3. Page attention启用，continuous batching由mega-kernel内scheduler task处理
  4. 预热后测量decode 1024 tokens的总时间，计算throughput (tokens/s)
  5. Batch size = {1, 2, 4, 8, 16}，每个配置测量3次取平均

  全过程（以Qwen3-8B在H100上batch_size=1 decode为例）：
  ```
  Host: torch.compile(backend=MPK)(model) → 编译完成
  Host: mega_kernel() → 单次CUDA kernel launch

  GPU Mega-Kernel 内部执行 (持久运行至所有token decode完成):
  ┌─────────────────────────────────────────────────────────────┐
  │ SM Partitioning: 128 Workers + 4 Scheduler-SMs (16 warps)  │
  │                                                             │
  │ Scheduler (start event e0):                                │
  │   ① Remove completed requests                              │
  │   ② Admit new requests (batch_size=1)                      │
  │   ③ Update KV-cache metadata                               │
  │   ④ Dispatch Q_proj tasks → Workers                        │
  │                                                             │
  │ Worker SM_i (execute Q_proj task):                         │
  │   Pre-load phase: TMA load Q_weight tile → SMEM page       │
  │   Compute phase: Tensor Core MMA (input × Q_weight)        │
  │   └─ 同时: Pre-load K_proj weight tile → another SMEM page │
  │   → 完成, notify event e_Q                                 │
  │                                                             │
  │ Worker SM_j (execute K_proj task): 类似                    │
  │ Worker SM_k (execute V_proj task): 类似                    │
  │                                                             │
  │ Event e_QKV 激活: 所有 Q/K/V tasks 完成                    │
  │ Scheduler → dispatch Attention tasks                       │
  │                                                             │
  │ Worker SM (execute Attention task):                        │
  │   FlashAttention-style kernel on single SM                 │
  │   [JIT mode: 执行时间data-dependent (sequence length)]      │
  │   → 完成, notify event e_Attn                              │
  │                                                             │
  │ Event e_Attn 激活: 所有注意力tasks完成                      │
  │ Scheduler: JIT dispatch O_proj + RMSNorm tasks             │
  │   (workers更快完成attention的获得更多下游tasks → 负载均衡)    │
  │                                                             │
  │ [AOT mode] MLP tasks 已预分配到 workers:                    │
  │   Worker SM: check AOT queue → event已激活? → gate_proj    │
  │   Pre-load gate_weight → compute GEMM →                   │
  │   同时 pre-load up_weight →                                │
  │   SiLU activation → down_proj GEMM →                       │
  │   同时 pre-load 下一层 Q_weight... (cross-task pipelining)  │
  │                                                             │
  │ ... 循环处理所有Transformer layers ...                      │
  │ 直至 generate stop token → mega-kernel return              │
  └─────────────────────────────────────────────────────────────┘

  输出: per-token latency 12.5ms (vs vLLM/SGLang 14.5ms, 理论下限~10ms)
  Throughput: 80 tokens/s (batch=1, decode)
  ```
  对比kernel-per-operator系统（SGLang/vLLM）：每个operator为独立kernel launch → kernel barriers阻止跨算子pipelining和细粒度overlap → CPU端page allocation/scheduling产生额外CPU-GPU同步 → 数百次kernel launch/iteration的overhead在latency-critical场景不可忽略。
