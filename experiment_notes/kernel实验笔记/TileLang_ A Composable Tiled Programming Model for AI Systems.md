## TileLang: A Composable Tiled Programming Model for AI Systems

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是TileLang的调度空间（Scheduling Space）系统，由四种核心调度机制组成，均通过解耦dataflow与scheduling实现：(1) **Thread Binding**——通过Layout Inference Pass按三层优先级（GEMM > Element-wise > Copy）自动推断buffer layout和thread binding。LayoutMap记录所有buffer的layout信息，从高到低优先级逐层推断，直至无更多buffer可推断。Fragment Layout支持repeat/repeat_on_thread/replicate四种组合操作构造复杂block级layout。对于GEMM后的element-wise操作（如bias add），自动推断bias buffer的replication策略以匹配GEMM output的thread分布。(2) **Memory Layout Composition**——基于IterVar的composable Layout抽象（f: K^n → K^m），支持swizzle layout（避免shared memory bank conflict）、padding layout（优化access pattern）、Fragment Layout（f: K^n → K²，输出thread index和register index）。Layout通过forward_index表达式和arithmetic analyzer推断buffer shape和访问边界。(3) **Pipeline**——T.Pipelined自动推导pipeline schedule：分析loop body各语句的buffer使用，确定Copy和GEMM的依赖关系，生成interleaved schedule（Copy→GEMM与其他copy重叠）。Ampere: 自动插入cp.async/cp.async.commit_group/cp.async.wait_group。Hopper: 自动TMA + mbarrier + warp specialization（通过Live Variable Analysis确定同步点，生产者/消费者根据threadIdx分入不同执行路径）。AMD CDNA: 利用s_waitcnt lgkmcnt和buffer_load_dword lds指令。(4) **Intrinsic Tensorization**——两种硬件指令利用方式：Tile Library-based (CUTLASS cute / AMD CK; 默认方案，自动选择最优指令) 和 Direct PTX/C++ source injection (T.ptx + T.import_source + T.call_extern)。

  实验比较：FlashAttention (H100, vs FA3 1.36×/Triton 1.41×/PyTorch 1.70×)，Linear Attention (H100, vs Triton 平均1.77×和2.10×)，MLA (H100 1075.9× vs Torch, 98% of FlashMLA; MI300X 129.2× vs Torch, 95% of AITER)，GEMM (4 GPU type × vendor libs/Triton)，Dequantized Matmul (A100, INT2 7.65× vs cuBLAS, INT4 1.04× vs Marlin, NF4 1.62× vs BitsandBytes)。

- 后端平台是什么，配置是什么。
  NVIDIA H100 (80 GB, CUDA 12.4)，NVIDIA A100 (80 GB, CUDA 12.4)，AMD Instinct MI300X (192 GB, ROCm 6.1.0)，RTX 4090。所有平台Ubuntu 20.04。

- 评估性能的软件/脚本是什么。修改了什么。
  TileLang kernel通过@tilelang.jit decorator编译，tilelang.compile(program, target="cuda"/"hip")生成可执行kernel。FlashAttention benchmark使用Table 3的5种shape配置（batch=1, nheads=32, seq_len=512/1024/4096, head_dim=128, causal/non-causal）；Linear Attention使用Mamba-2的chunk-scan和chunk-state函数，Table 4的12种shape；MLA在H100和MI300X上对比；GEMM使用Table 2的16种矩阵shape (M 1-8192, N 9216-57344, K 9216-57344)；Dequantized Matmul基于BitBLAS-TileLang后端，覆盖W_INT2A_INT8 / W_INT4A_FP16 / W_NF4A_FP16 format。Baselines: FlashAttention-3(手写CUDA), Triton(开源框架), cuBLAS(NVIDIA), rocBLAS(AMD), PyTorch(手写FA2 kernel), BitsandBytes(NF4 kernel), Marlin(INT4 kernel), FlashInfer, FlashMLA(手写), AITER(手写AMD kernel)。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/tile-ai/tilelang

  评估原理：TileLang将kernel执行建模为dataflow-centric tile operators + scheduling annotations的组合。关键调度优化如何转化为性能增益：

  1) **Layout Swizzling去除Bank Conflict**：T.gemm默认对A_shared和B_shared应用MakeSwizzleLayout。swizzle通过异或位操作重排shared memory地址，使得warp内不同thread的shared memory访问映射到不同bank。无swizzle时bank conflict导致shared memory bandwidth下降，GEMM性能损失可达20-30%。Layout swizzling确保所有测试设备上zero bank conflict。

  2) **Pipeline Overlap**：T.Pipelined(K // block_K, num_stages=2)自动推导Copy-GEMM overlap。对每个k-iteration i，编译器分析(i+1)轮Copy与(i)轮GEMM无依赖，生成interleaved schedule。在Ampere，cp.async用于异步global→shared copy，与Tensor Core GEMM计算overlap。在Hopper，TMA hardware unit接管copy，warp specialization将线程分为producer(TMA copy)和consumer(wgmma.mma_async)，通过mbarrier同步。与Triton的num_stages参数不同，TileLang允许用户通过自定义pipeline order实现更复杂的overlap pattern。

  3) **Thread Binding + Vectorization**：T.copy在Layout Inference Pass后自动parallelize和vectorize（图8）。以(8,32)的2D copy为例：Pass推断loop axes → 自动分配thread binding（如threadIdx.x映射到i轴, vectorize 4 elements along j轴） → 应用SwizzleLayout。生成代码使用128-bit vectorized load/store (uint4/float4)，最大化memory bandwidth利用率。

  4) **Warp Specialization (Hopper独占)**：TileLang自动分析buffer usage确定各语句的producer/consumer角色 → 按threadIdx分离执行路径 → Live Variable Analysis确定同步点 → 插入mbarrier。这使得在FlashAttention实现中达到与FlashAttention-3手写kernel相当的pipeline复杂度。

  Kernel输入到性能输出全过程（以H100 FlashAttention为例）：

  输入: Q[batch, heads, dim] f16, KV[batch, seq_kv, kv_heads, dim] f16
  1. T.Kernel(batch, heads // min(B_H, kv_group), threads=256) → grid_size, block_size
  2. T.alloc_shared + T.alloc_fragment: 分配Q_shared, KV_shared, S_shared共享内存; acc_s, acc_o, scores_max等register files
  3. T.copy(Q → Q_shared): Layout Inference → thread binding + vectorized load (128-bit)
  4. T.Pipelined loop over KV tiles:
     a) Producer threads: TMA async copy KV[bx, k*BN:(k+1)*BN, ...] → KV_shared (global→shared via TMA hardware)
     b) mbarrier arrive (producer signals data ready)
     c) Consumer threads: mbarrier wait → wgmma.mma_async(Q_shared, KV_shared, acc_s) (async Tensor Core matmul)
     d) T.reduce_max(acc_s, scores_max, dim=1) → online softmax rescaling
     e) T.gemm(S_shared, KV_shared, acc_o) → output accumulation
     f) Producer continues TMA copy for next KV tile (overlapped with consumer compute)
  5. T.copy(acc_o → Output): register → shared → global store with thread binding + vectorization
  6. 评估: tilelang.compile返回的kernel函数，通过CUDA events测量wall-clock latency。与FlashAttention-3比 speedup = latency_FA3 / latency_TileLang
