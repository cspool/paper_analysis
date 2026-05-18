## QuCo: Efficient and Flexible Hardware-Driven Automatic Configuration of Tile Transfers in GPUs

- baseline方法是什么？
  Baseline是当前GPU ATT编程的最佳实践：程序员手动进行wavefront specialization（将workgroup内wavefront分为dedicated producer wavefront执行ATT memory transfer + consumer wavefronts执行computation），手动选择tile sizes、queue slots、LDS partitioning和synchronization barriers。NVIDIA提供cuda::pipeline API包装TMA的producer-consumer wavefronts为reusable queues，CUTLASS3+CuTe和ThunderKittens提供高级ATT抽象（TMA pipelines和asynchronous I/O），但获得hardware-specific peak ATT performance仍需程序员deep understanding of underlying GPU microarchitecture和手动tuning——不同kernel间最优ATT配置差异可达1.2×，不同GPU间可达1.4×。

  全栈执行例子（以手动ATT/Fine-Tuned执行Matrix-Matrix kernel on MGPUSim R9 Nano为例）：
  - 算法层：矩阵乘法kernel [512,2048]×[2048,128]，CI>4（compute-bound），程序员需理解kernel compute-to-memory ratio和data reuse pattern决定wavefront specialization策略
  - 系统框架/Serving层：论文未明确说明（无serving framework，直接launch kernel）
  - 编译框架层：论文未明确说明（kernel binary通过AMD ROCm编译，但无QuCo时编译框架不参与ATT配置）
  - kernel调度层：程序员手动编写producer wavefront（通过ATT descriptors指定global memory addresses、tile dimensions、memory strides、LDS destination）→调用Push/Wait_For_Push→consumer wavefronts通过Peek/Pop配合asynchronous transaction barriers（arrive+wait两阶段）消费LDS中tile→程序员需遍历exhaustive DSE（对Matrix-Matrix为2.6×10^14种组合）找最优配置，每次尝试需一次kernel launch profiling
  - 硬件架构层：GPU Compute Unit内含ATT engine（bypass L1 cache, 直接issue read memory requests到global memory, 自行生成address/transfer count, 直接写LDS）→Sync Unit管理asynchronous barriers→ATT descriptors指针从LDS读取。缺陷：参数选择kernel-specific且architecture-specific（R9 Nano的最优配置在MI-100上差1.4×），手动tuning laborious且impractical for large design spaces（Whisper Tiny 2.1×10^17 kernel launches）

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出QuCo（Queue Configurator），一个嵌入GPU die的轻量级硬件单元，在kernel launch时通过RISC-V firmware自动计算ATT队列最优配置，无需程序员手动tuning。方法对应baseline缺陷如下：

  **缺陷1：手动tile size选择需要遍历exponentially large design space（64-8192 elements × 1-8 slots × 多queue组合，从25到2.6×10^14种组合）→programmer effort impractical**
  → QuCo Algorithm 1+2自动计算：遍历tile size candidate→对每个tile计算merit factor（processing time / memory transfer time）× cost function→processing time含scheduling roundtrip overhead（wavefront dispatch limitations建模）→memory time含ATT/DRAM/L2 latency + bandwidth inverse penalty + cache-line inverse penalty→选weighted merit最优tile→按CI调整（低CI放大tile提升memory throughput，高CI缩小tile平衡overlap）。无需profiling，单次pass完成。

  **缺陷2：手动slot选择缺乏系统化方法，过多slots导致memory contention，过少导致underutilization和idle cycles**
  → QuCo Algorithm 3：streaming queues用hardware-aware Little's Law（slots = memory transfer time / tile compute time）推导ideal queue depth → CU-aware rounding（更多CU→减少slots降低memory system pressure） → LDS capacity check with CI-based fallback（低CI多slots提升throughput，高CI少slots）。stationary queues均分remaining LDS capacity后同样round。通过GST获取精确hardware参数（bandwidth/latency/LDS size/CU count）确保architecture-specific optimal。

  **缺陷3：post-compilation binary缺乏跨GPU portability——同一ATT配置在不同GPU上性能差异可达1.4×**
  → QuCo的GST（256-byte vendor-written table）存储每个GPU型号的architectural parameters（memory latencies/clock/LDS size/CU count/arithmetic throughput）。同一compiled binary在不同GPU上执行时，QuCo firmware读取该GPU的GST→自适应计算tile/slots→preserve same binary across GPU family。evaluation证明QuCo on MI-100/R9 Nano/Radeon 530均near-optimal。

  **缺陷4：DVFS和多租户环境下静态tuned配置失效——frequency变化或partitioned resources打破assumptions**
  → QuCo-HW在每个kernel launch时重新读取当前operating frequency和resource availability→动态调整queue配置适应runtime变化。DVFS实验：QuCo-HW在3种频率变化scenario下vs QuCo-SW（假定固定频率）获up to 17% improvement。Multi-tenant场景下QuCo根据实际available resources动态调整。

  **缺陷5：per-layer/model-level手动tuning不scale——DNN模型如Whisper Tiny有827层，每层特性不同，Semi-Tuned复用早期层配置导致后续层性能下降**
  → QuCo per-kernel invocation自动reconfigure queue参数（tile size从256到1024、slots从2到4不等，因层CI和dimensions而异）。Whisper Tiny layer-wise ablation证明QuCo在所有unique层上均outperform ATT/Semi-Tuned和ATT/Fine-Tuned。

  全栈执行例子（以QuCo执行Matrix-Matrix kernel on MGPUSim R9 Nano为例）：
  - 算法层：与baseline相同kernel binary（无代码改动），host code仅需：driver.InitQuCo(HIGH, 512, 64) → driver.RegisterQueue(K, 4, TYPE_STREAMING)注册8个streaming queues + 1个stationary queue
  - 编译框架层：论文未明确说明（kernel编译路径不变，QuCo在post-compilation runtime介入）
  - kernel调度层：kernel launch时QuCo RISC-V firmware从GST读取R9 Nano参数（64 CUs, 1.0GHz, LDS size, DRAM bandwidth, latencies 190/300/450）→计算CI>4（compute-bound）→Algorithm 1选tile 512（对高CI缩小tile平衡overlap）→Algorithm 3计算streaming queue slots = 2（Little's Law + CU-aware rounding从4降为2降低memory contention）→stationary queue slots = 2（remaining LDS均分后round）→写入ATT descriptors到LDS→ATT unit加载descriptors开始异步tile load→consumer wavefronts消费LDS中tile不感知配置过程。结果：QuCo performance在ATT/Fine-Tuned（exhaustive DSE）的1.04%以内，但仅需单次pass无任何tuning努力。
  - 硬件架构层：QuCo hardware（RISC-V core 0.027mm² + memory 0.014mm² @28nm）与main compute pipeline解耦，不干扰wavefront scheduling和memory requests→配置完成后idle，多kernel workload中reconfiguration overhead ~6300-8300 cycles远小于kernel execution time
