## μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs

- baseline方法是什么？
  Baseline是NVIDIA GPU的默认硬件调度器行为：stacked co-location。GPU的dispatch unit在kernel launch后将同一kernel的所有block按顺序分配到SM cores，当kernel线程数超过GPU总thread capacity时，block占满所有SM。由于同kernel所有block的hardware resource需求相同，产生"1 more, 5 less"资源利用模式——1种主要hardware resource高利用率（avg 30.19%），其余5种极低（avg 5.07%）。NVIDIA-SMI报告81.16% utilization但Nsight Compute仅报告9.28% low-level hardware utilization。

  全栈执行例子（以PyTorch + NVIDIA A40 GPU + 默认FCFS kernel调度为例）：
  - 算法层：推理模型（如BERT/ResNet50）由多个CUDA kernel组成（matrix multiplication、layer normalization、convolution等），每个kernel的blocksize由PyTorch/TVM/TensorRT在dedicated GPU场景下静态preset（枚举+选最优，如roll kernel preset 512）
  - 系统框架/Serving层：PyTorch sequential launch kernel到CUDA stream → GPU Command Processor串行接收 → dispatch unit按block粒度调度到SM。跨kernel的并发通过multi-stream/persistent kernel尝试有限。INFless/Orion等Serving系统在inter-SM层面做spatial/temporal sharing，但intra-SM仍为stacked co-location
  - 编译框架层：论文未明确说明（PyTorch 2.2.0 + CUDA 11.8默认编译路径，blocksize preset在模型导出时固化）
  - kernel调度层：硬件dispatch unit left-over scheduling——SM满足线程数要求即可入驻block。同一kernel blocks因需求同构→全部stack进同一批SM→同构资源需求→single-dominant resource pattern。例如：vectorized kernel (layer norm) block在SM内LDST利用率58.02%，其余5种HW avg < 15%。Roll kernel block INT32利用率33.25%。两者stacked co-locate时各自独占SM→无法互补
  - 硬件架构层：NVIDIA A40 GPU（84 SMs, 每SM 1536 threads, 32 FP64 cores/64 FP32/64 INT32/4 Tensor cores/16 SFU/32 LDST units）。硬件scheduler闭源，不暴露intra-SM resource allocation接口

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出μShare：通过非侵入式修改kernel launch参数（blocksize）间接操纵闭源GPU硬件scheduler，实现intra-SM scattered kernel co-location。

  **缺陷1：Baseline的stacked co-location导致intra-SM硬件资源"1 more, 5 less"——同kernel blocks因需求同构独占SM，剩余5种hardware units闲置**
  → μShare提出half-plus blocksize shaping：将kernel blocksize设为略超半SM thread容量（A40: 768+32=800），使同一SM无法容纳两个该kernel block（800×2=1600 > 1536）→同kernel blocks被迫散布到不同SM→每SM剩余threads（1536-800=736）可容纳另一kernel的小block（512-640 threads）→实现scattered co-location。实验：当dominant resource不同的kernel配对时half-plus提升throughput 19.94%；若dominant resource相同则下降10.37%

  **缺陷2：Static preset blocksize在co-location场景下non-optimal——PyTorch/TVM/TensorRT在dedicated GPU下枚举选最优blocksize（如roll kernel=512），但SM内可用资源动态变化时该blocksize不再optimal**
  → μShare发现roll kernel与vectorized kernel（blocksize=256）co-locate时最优blocksize从512 shift到1024（1.98× throughput improvement）。μShare的block shaper根据实时kernel launch slack动态调整blocksize：slack positive→half+32（最小warp粒度）、slack negative→逐步+32加速执行

  **缺陷3：48.37%的kernel blocksize不可修改（cuBLAS/cuDNN闭源wrapper + tiling kernel如Conv2d修改后产生CUDA internal error）→仅partial co-location**
  → μShare提出time-shifted launching：对unmodifiable kernel保持default blocksize，检查6种low-level hardware resource combined utilization + shared memory/register availability→满足互补条件则立即launch→不满足delay β=10μs后重检→更新slack重新排序kernel set→若延迟导致其进入top-x slack最小的kernel→升级为half-plus shaping。实验：unmodifiable kernels从100%→48.37%时throughput从47.59→58.81单调提升；worst-case (100% unmodifiable) 回退到resource-coupled co-location（等效INFless）

  **缺陷4：Existing intra-SM sharing方法需要侵入性修改——kernel fusion (Tacker/T3/Rammer/COMBO)需重写合并kernel代码、persistent kernel (ISPA/Plasticine/Elastic kernel)需空non-terminating kernel驻留、hardware modification (CCWS/Prema/PriorityRR)需重新设计GPU scheduler→在public cloud不feasible**
  → μShare完全非侵入：仅在Linux userspace通过LD_PRELOAD劫持kernel launch函数→无需修改kernel代码、无需修改GPU硬件scheduler、无需额外CPU-GPU通信。Kernel interception overhead仅60.35ns/kernel，CPU overhead 6.85% of single core。PyTorch仅需设置C10_LAUNCH_BOUNDS与CUDA limit一致

  **缺陷5：A800/A100/H200等2048 threads/SM的GPU不支持half-plus（即使max blocksize=1024，两个1024 block仍可在1SM内stacked co-locate）**
  → μShare改用1/3-plus shaping：blocksize=2048/3+α≈704→同kernel两个1/3-plus block (704×2=1408) 可入1SM→剩余threads (2048-1408=640) < 704→不能放第三个large block→只能放small block (512-640) from complementary kernel→实现scattered co-location。尽管从half(1/2)提升到2/3上限可能导致轻微thread分配不均衡，但在A800上仍获得16.45%-52.29% throughput improvement

  **缺陷6：Baseline使用NVIDIA-SMI的"active time ratio"夸大了GPU utilization（81.16% vs Nsight Compute 9.28%）和SLO管理粗糙（仅inter-request层面调整batch size）**
  → μShare引入dual-level SLO guarantee：(1) 请求侧：exponential decay-based SLO slack feedback control → 基于monitored latency保守增加（linear, bj+1 = bj + k×s→j）、激进减少（exponential, bj+1 = max{bj - e^(λ×s→j), 1}）batch size；(2) kernel侧：基于kernel launch slack调整blocksize加速kernel执行。v7配置(k=0.05, λ=-0.2) SLO violation rate 0.84%低于INFless/Orion的2.05%/1.12%，同时throughput仍提升19.28%-44.83%

  全栈执行例子（以PyTorch + μShare + NVIDIA A40 + roll kernel与vectorized kernel scattered co-location为例）：
  - 算法层：与原模型相同（无kernel代码修改）。模型kernel分类：roll kernel（INT32 dominant, blocksize=512 default, modifiable 51.63%）+ vectorized kernel（LDST dominant, blocksize modified to half-plus 800）+ cuBLAS gemm（Tensor core dominant, blocksize unmodifiable 48.37%）等
  - 系统框架/Serving层：Kernel Interceptor通过LD_PRELOAD+dlsym拦截cudaLaunchKernel → 读取blocksize参数 → 计算kernel launch slack（s_k = tLaunch - tIntercept）。Batch Manager根据实时SLO slack (SLO=200ms) 采用conservative-increase/aggressive-decrease调整batch size。离线Profiler对每个kernel记录9-tuple resource profile
  - 编译框架层：论文未明确说明（PyTorch 2.2.0 + CUDA 11.8默认编译，model权重无修改）
  - kernel调度层：Block Shaper将vectorized kernel blocksize设为800 (half+32) → 1SM内800 threads → 2个vectorized block需1600 threads > 1536上限 → 只放1个 → 剩余736 threads放1个roll block (512) → LDST (vectorized) + INT32 (roll) 互补执行。同时cuBLAS gemm (Tensor core) 根据time-shifted launching在vectorized kernel接近完成时启动 → Tensor core + LDST/INT32 互补。结果：intra-SM 6种hardware avg utilization从10.90% (INFless) 提升到15.10% (μShare)
  - 硬件架构层：NVIDIA A40 GPU（84 SMs, 1536 threads/SM），硬件scheduler left-over策略保持不变。μShare仅通过控制stage 1 (launching) 的blocksize参数间接影响stage 2 (scheduling) 和stage 3 (execution) 的behavior。修改在CPU userspace完成（dlopen/dlsym/mmap），GPU硬件执行路径不变
