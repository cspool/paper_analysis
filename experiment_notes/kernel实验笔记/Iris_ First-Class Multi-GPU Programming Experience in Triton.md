## Iris: First-Class Multi-GPU Programming Experience in Triton

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是Iris——一个纯Python+Triton实现的多GPU通信库，提供tile级device-side API(load/store/get/put/copy/atomic_add/atomic_cas等)，使开发者能在单个Triton kernel内无缝交织计算和通信。核心kernel调度实现包括：(1) 指针翻译机制——通过__translate函数将本地symmetric heap指针转换为远程GPU地址(偏移计算+heap base加法+类型cast)，实现跨GPU内存访问；(2) 融合kernel模式分类——Unfused Bulk-Synchronous（先后两个kernel，中间global barrier）、Unfused Producer-Consumer（两个kernel在不同CU异步stream上执行，通过CU分区和atomic锁同步）、Fused Sequential（单kernel内GEMM tile产出后立即iris.store到远程GPU，顺序依赖）、Fused Workgroup Specialization（单persistent kernel内按pid划分workgroup：前256个workgroup做GEMM计算并atomic_cas发信号，后48个workgroup等待信号执行iris.put通信）；(3) tile级同步——使用gpu-scoped atomic_cas(acquire/release)替代kernel级barrier，实现fine-grained overlap；(4) cache感知调度——cache_modifier(".wt"等)控制写策略，chiplet_swizzle映射workgroup到XCD分组优化LLC locality，GROUP_SIZE_M做L2 spatial swizzle。

  实验比较：(a) Microbenchmarks——load/store/atomic point-to-point操作带宽利用率(heatmap)，all-load/all-store多GPU同时操作带宽利用率；(b) GEMM+All-Scatter workload——Iris Unfused Bulk-Synchronous、Unfused Producer-Consumer、Fused Sequential、Fused Workgroup Specialization四种overlap模式 vs PyTorch torch.matmul + RCCL AllGather baseline，6种problem shape(M=8192固定，N×K变化)，2/4/8 GPU配置。

- 后端平台是什么，配置是什么。
  8×AMD Instinct MI300X GPU，全连接Infinity Fabric拓扑(7条Infinity Fabric Link/GPU)，NPS1/SPX memory和compute partition模式，ROCm 6.3.1。MI300X每GPU 304 Compute Units。

- 评估性能的软件/脚本是什么。修改了什么。
  自研Iris库(Python+Triton)，使用PyTorch Distributed初始化rank、HIP IPC(hipIpcGetMemHandle/hipIpcOpenMemHandle)建立symmetric heap。修改：(1) 新增device-side API——load/store(值语义，register↔remote memory)、get/put/copy(指针语义，buffer↔buffer)、atomic_*系列操作(算术/位运算/交换/比较)，所有操作遵循acquire/release memory ordering + block/gpu/sys scope；(2) 实现多种融合GEMM+All-Scatter kernel变体(对应Listings 3-5)；(3) GEMM loop复用统一gemm_loop模板。Baseline使用torch.matmul(PyTorch GEMM) + RCCL AllGather。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/ROCm/iris

  评估原理：
  1. Microbenchmarks：各GPU同时执行load/store/atomic操作测量remote memory access带宽，归一化到理论可达带宽，绘制heatmap。All-load/store benchmark：所有GPU同时跨所有链路执行load/store，不同buffer size测量带宽利用率。
  2. 应用级评估：对每种problem shape和GPU数，分别用Iris的四种overlap pattern和PyTorch+RCCL baseline执行GEMM+All-Scatter，测量wall-clock时间计算speedup。
  3. Deep-dive分析：分解GEMM(深色区域)和Communication(浅色区域)时间占比，展示overlap效果。

  Kernel输入到性能输出全过程（以Fused Workgroup Specialization GEMM+All-Scatter，8 GPU，M=8192，N=3584，K=14336为例）：

  ```
  Host: iris.init()初始化——分配symmetric heap，IPC handle exchange
  Host: 输入矩阵A[M,K]分片在各GPU本地，B[K,N/8]各GPU持有N维1/8
  Host: launch wg_specialized_gemm_all_scatter[(304,)] (304 = MI300X CU数)

  GPU Mega-Kernel内部:
  ┌──────────────────────────────────────────────────────────────┐
  │ Workgroup分配: 256 GEMM workers (pid 0-255), 48 COMM workers (pid 256-303) │
  │                                                              │
  │ GEMM Worker (pid 0-255, persistent for-loop):               │
  │   for tile_id in range(pid, total_tiles, 256):               │
  │     ① gemm_loop(A, B, C):                                    │
  │       for k in range(0, K, BLOCK_SIZE_K):                    │
  │         a = tl.load(A_tile)   // global→register             │
  │         b = tl.load(B_tile)                                   │
  │         acc += tl.dot(a, b)   // Tensor Core MMA              │
  │       → 产出C_tile [BLOCK_M, BLOCK_N] in registers           │
  │     ② tl.store(C_local + offset, c, cache_modifier=".wt")    │
  │       → 写本地GPU memory（write-through for coherence）      │
  │     ③ tl.atomic_cas(locks + tile_id, 0, 1, sem="release",    │
  │                      scope="gpu")                            │
  │       → 发信号: tile_id已就绪                                 │
  │                                                              │
  │ COMM Worker (pid 256-303, persistent for-loop):              │
  │   for tile_id in range(pid-256, total_tiles, 48):            │
  │     ① spin-lock:                                            │
  │       while atomic_cas(locks+tile_id, 1, 0, sem="acquire")==0│
  │         pass  // 等待GEMM worker完成                          │
  │     ② for remote_rank in range(8):                           │
  │         if remote_rank != cur_rank:                          │
  │           iris.put(C_local+offset, C_local+offset,           │
  │                    cur_rank, remote_rank, heap_bases)         │
  │           → translate: offset_in_heap = ptr - local_heap_base│
  │           → remote_ptr = remote_heap_base + offset_in_heap   │
  │           → tl.store(remote_ptr, data) // 跨GPU写             │
  └──────────────────────────────────────────────────────────────┘

  输出性能：
    - Unfused Bulk-Synchronous (Iris baseline): 与PyTorch+RCCL性能相当，验证无抽象开销
    - Unfused Producer-Consumer: up to 2.5× speedup (8192×3584×14336, 8 GPU)
      → 小N(被8分割后)+大K使通信完全隐藏在GEMM后面
    - Fused Sequential: up to 1.79× speedup (8192×4608×36864, 4 GPU), 1.5× (8 GPU)
    - Fused Workgroup Specialization: 通信近乎100%隐藏于GEMM
    - 跨所有配置平均1.21× speedup vs PyTorch+RCCL
    - Microbenchmarks: near-optimal bandwidth utilization

  关键kernel调度优势：
  - tile级同步替代kernel barrier：消除bulk-synchronous模式的"bubble"
  - GEMM tile产出后立即scatter：无intermediate global memory write→read往返
  - Workgroup specialization使GEMM和通信使用不同CU子集并发执行
  - 值语义(iris.store)直接从register scatter到remote GPU，无需本地buffer中转
