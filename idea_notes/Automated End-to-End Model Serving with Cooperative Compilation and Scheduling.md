## Automated End-to-End Model Serving with Cooperative Compilation and Scheduling

- baseline方法是什么？
  Baseline是operator-level compilation和coarse-grained scheduling的DNN serving系统。编译侧：Ansor/MetaSchedule依赖长时间搜索，Roller减少搜索空间但仍需性能模型评估，cuDNN依赖手工优化库——它们都以完整operator kernel为编译和调度基本单位，无法为推理期动态调度提供足够细粒度和多版本选择。服务侧：CUDA Stream/MPS允许并发但不能保证GPU内部空间共享；Triton是通用serving平台但调度受kernel launch/queue机制约束；Paella改善kernel执行顺序控制但仍以较粗的kernel/operator为调度对象。

  Baseline全栈执行例子（以TVM + CUDA Stream serving ResNet-50 on A100为例）：
  - 算法层：ResNet-50 DNN inference，operators按ONNX computation graph顺序执行。每个operator对应一个完整kernel，kernel整体执行或整体不执行。
  - 系统框架/Serving层：CUDA Stream并发——多个请求的不同operator kernel被放入不同CUstream，GPU scheduler按stream优先级和时间顺序发射kernel到硬件队列。kernel间存在tail effect（前序kernel结束阶段pipeline bubble导致GPU idle 1-3μs）和cold start（后续kernel preamble包括thread block dispatching/资源分配/prologue bubble）。MPS进一步允许不同进程context共享GPU，但空间共享由GPU inner scheduler控制，不可精确管理。
  - 编译框架层：TVM compiler将ONNX model转为computation graph→schedule primitives生成tensor program→codegen生成CUDA kernel。Kernel编译针对单个operator，ILP/TLP/arithmetic intensity trade-off固定在编译期确定，无法根据推理期GPU并发状态调整。
  - kernel调度层：operator kernel在CUDA stream中排队，GPU thread block scheduler (GigaThread Engine) 管理SM分配——block-fused和warp-fused colocation均需编译期预先融合（如HFuse），无法运行时动态决定。Kernel launch通过CUDA runtime API，host-device同步开销大。
  - 硬件层：NVIDIA A100 GPU (Ampere, 108 SM, 40GB HBM2e)。SM内warp scheduler以warp粒度选择ready warp执行。当单个operator kernel的instruction pattern单调时（如仅FP密集或仅memory密集），部分GPU pipeline units空闲，GPU utilization低。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Infera，通过cooperative compilation and scheduling将编译和调度从"动态耦合"改为"静态提供调度空间、运行时动态选择"。

  Infera全栈执行例子（以mixed-model BERT+ViT inference on A100为例）：
  - 算法层：与baseline相同，DNN inference，但operator被切分为tile/micro operators，下游operator的某些tile可在上游operator完整结束前开始执行（sub-operator级data asynchrony）。
  - 系统框架/Serving层：Infera inference server pipeline——JDU将job按GPU可用显存和estimated remaining time分发到目标GPU→TSU按priority（ddl_rq用EDF、rt_rq用FIFO、gcfs_rq用GCFS按nice value分配instruction budget）生成VTB→TEU三阶段执行（SelectKernels→FuseKernels→LaunchKernel）。Priority preemption：高优先级任务到达→TSU暂停调度保存状态→TEU保存HKQ/DKQ kernel context→in-flight kernel通过flag快速终止。
  - 编译框架层：Infera compiler基于TVM 0.16.0——ONNX→TVM Relay→tile-based TensorIR→为每个micro operator生成多个ILP/TLP/intensity配置的kernel candidates（register 64/96/128, shared memory 48/80/112/144 KiB, pipeline stage 2/3/4）→cut-and-patch instruction scheduling在SASS级别优化ILP→warp specialization（4 mainloop warps+4 data copy warps）→打包为CUDA binary static library。Zero-tuning：完全基于静态分析，无需GPU profiling或性能模型评估。
  - kernel调度层：TEU SelectKernels在data dependency DAG中选zero in-degree且maximize asynchronous wavefront的data blocks→online regression model估计IPC→选#inst/IPC最小且TLP≥4的kernel→FuseKernels在CUDA binary level做warp-level horizontal fusion（prologue恢复special registers+shared memory offset+barrier重组）→LaunchKernel通过HKQ→GDRCopy→DKQ→daemon kernel CDP fire-and-forget launch（<10μs, 避免HoL）→SM内不同task的warps空间共享GPU pipeline units。
  - 硬件层：NVIDIA A100 GPU。daemon kernel独占一个SM用于device-side kernel launch和DKQ管理。GDRCopy bypass DMA engines实现<100ns小数据/<5μs典型kernel传输延迟。Driver-level placeholder kernel slot覆盖避免cuModuleLoad的global host-device同步。Preemption latency：Infera-P约10μs（保存HKQ/DKQ），Infera-R约5μs（仅保存不暂停），比REEF-N快约2.5×。

  **缺陷1：Operator-based compilation无法提供sub-operator级parallelism → operator内部分tile必须等整个operator完成**
  → Infera方案：tile-based partition将大operator切分为micro operators/tiles，每个tile独立编译和调度。下游operator的某些tile在上游operator完成对该tile的输入后即可开始执行（data dependency DAG中zero in-degree节点），扩大data asynchrony。Multi-version kernel generation为每个tile提供不同ILP/TLP/intensity trade-off：register limit 64/96/128控制ILP vs TLP，shared memory limit 48/80/112/144 KiB控制tile size和intensity。编译完全并行化且zero-tuning，比Ansor/MetaSchedule编译时间低2-3个数量级。

  **缺陷2：Monotonous instruction pattern导致GPU pipeline unit利用率低（FP unit忙时memory unit空闲，反之亦然）**
  → Infera方案：multi-version kernels在编译期覆盖ILP、TLP和arithmetic intensity的多种组合，runtime SelectKernels根据当前GPU occupancy、kernel hazard和data/structure hazard估计选择最适合的kernel版本。Warp-level binary fusion将不同task的不同类型kernel（如BERT MatMul+ViT Conv）的warps在SM内空间共存，使不同instruction pattern的warps同时利用GPU的不同pipeline units，降低scoreboard和throttle stall cycles。

  **缺陷3：Baseline serving (Stream/MPS/Triton/Paella) 调度粒度粗，无法精确控制GPU内部kernel空间共享和公平性**
  → Infera方案：JDU基于GPU estimated remaining time做load-balancing dispatch；TSU 64-priority runqueue支持deadline (EDF)/real-time (FIFO)/normal (GCFS)三类调度策略，aging mechanism防止starvation；VTB + instruction budget让normal任务按nice value公平共享调度周期；TEU三阶段pipeline (SelectKernels→FuseKernels→LaunchKernel)在每个调度周期动态组织micro-kernels。在multi-model mixed serving中（uniform requests/uniform models）至少快1.6×，bursty lognormal scenarios最高快3.5×。

  **缺陷4：CUDA runtime kernel launch有head-of-line blocking和host-device同步开销 → tail effect/cold start造成GPU idle**
  → Infera方案：daemon kernel常驻一个SM，通过CUDA Dynamic Parallelism fire-and-forget launch直接在device side发射kernel，消除stream tracking overhead和HoL，launch latency <10μs。GDRCopy host-device kernel code transfer bypass DMA engines。Driver-level placeholder kernel slot覆盖避免cuModuleLoad的global同步。Kernel completion通过cudaGetLastError异步检查。

  **关键trade-off**：Infera以高系统复杂度（SASS级代码改写、driver-level placeholder slot覆盖、GDRCopy、CDP daemon kernel、device-side queue、自研调度器约17k LoC）换取更细粒度的GPU控制。daemon kernel独占一个SM带来<1/#SM的设备侧开销。host侧kernel fusion、queue管理和fused kernel缓存带来CPU和host memory开销（最大吞吐前约13% CPU和600 MiB host memory）。实现强依赖NVIDIA GPU执行模型、SASS格式和底层driver行为，跨GPU代际和跨厂商可移植性需额外验证。tile粒度也有trade-off：过细增加调度/fusion/launch元数据开销，过粗退化回operator-based scheduling。
