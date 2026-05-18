## RPU - A Reasoning Processing Unit

- baseline方法是什么？
  Baseline是NVIDIA H100/H200 GPU系统上的低batch LLM decode推理。H100系统特征：(1) 内存层使用HBM3e（如单个stack 1280GB/s、48GB、BW/Cap≈27），带宽和容量强绑定，大量容量在低batch decode中未使用；(2) 系统层为monolithic die设计，compute-to-bandwidth ratio约200 OPs/Byte，仅30-40% TDP分给memory interface，decode时大量compute/cache资源闲置（power trace: decode avg 239.9W vs prefill 634.2W, BW utilization仅32.2%）；(3) 微架构层为shared memory NUMA + 统一memory access (UMA) + randomized address mapping，kernel launch和synchronization overhead在低batch小kernel下显著（小矩阵VMM kernel执行时间tens of microseconds, overhead同量级），无法持续饱和HBM带宽。

  全栈执行例子（以Llama3-70B FP8, 4×H100, BS=32, 16K prefill / 2K decode为例）：
  - 算法层：标准transformer decode（FP8 weights, BF16 activations），weight matrices column-sharded across 4 GPUs via tensor parallelism。
  - 系统框架/Serving层：vLLM + NVIDIA Dynamo。Decode phase下batch中每个query串行生成token→attention sequential computation→KV cache随seq len增长持续膨胀→memory bandwidth bound。
  - 编译框架层：PyTorch 2.2 compiled dense-linear kernels。
  - kernel调度层：GPU CUDA kernel launch model→host-driven offload→每个VMM kernel launch开销、tensor-parallel通信延迟（NVLink/NVSwitch）、cross-SM synchronization。小batch下kernel执行时间短（tens of microseconds），launch+sync overhead占显著比例，导致仅32.2% BW utilization。
  - 硬件架构层：H100 SXM (132 SMs, 80GB HBM3, 3.35TB/s peak BW)，HBM energy per bit ~3.44pJ/bit [43]，monolithic die ~814mm²，UMA memory controller + L2 cache (50MB) → 长距离片上数据移动增加energy。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出RPU（Reasoning Processing Unit），通过三层协同设计系统性地解决H100 baseline在低batch LLM decode中的memory wall瓶颈。

  **缺陷1：HBM高容量过度配置 → 为带宽买容量造成成本/能耗浪费（memory overprovisioning paradox）**
  → **RPU方案（HBM-CO）**：保留HBM的shoreline bandwidth架构但削减主要贡献容量的结构（ranks/banks/subarrays）→BW/Cap从27提升到341（候选Pareto-optimal），energy per bit从~3.44降至1.45pJ/bit（2.4× improvement），带宽/美元提高5×，总模块成本降低35×（去掉192×未使用容量）。Chiplet-based modular memory architecture允许在系统level灵活scale bandwidth and capacity across many smaller stacks。

  **缺陷2：GPU monolithic die的power/area provisioning偏向compute而非bandwidth → memory-bound decode闲置大量compute/cache资源**
  → **RPU方案（Chiplet Compute Fabric + Power/Area Reprovisioning）**：将compute die从monolithic拆为多个chiplet→相同compute die area下暴露近10× memory IO shoreline（~600mm vs H100 ~60mm）→每chiplet tightly coupled with HBM-CO stacks→70-80% TDP分配给memory interfaces→compute-to-bandwidth ratio调至32 OPs/Byte（H100 ~200 OPs/Byte）→Roofline向左下移以匹配低batch decode的low arithmetic intensity。ISO TDP下RPU提供2×+ bandwidth。

  **缺陷3：GPU的host-driven offload + shared memory NUMA + barrier synchronization → 低batch小kernel无法持续饱和HBM带宽（32.2% utilization）**
  → **RPU方案（Decoupled Microarchitecture + NUMA at All Scales）**：每core私有HBM-CO channel + local SRAM buffer，全NUMA无shared memory→消除coherence overhead。Memory/Compute/Network三pipeline硬件解耦→Pipeline Arbiter用buffer entry粒度valid counter实现data-driven同步（非global barrier）→memory pipeline在compute/network stall时继续预取weights/KV cache到on-chip buffer→compute后续"catch-up"消耗已预取数据。BS=1时RPU完全饱和memory bandwidth（roofline performance）。BS=32时decoupled pipeline吸收compute-bound weight layers和memory-bound KV cache layers间的phase imbalance→overall latency improvement up to 1.6×。

  **缺陷4：GPU通用编程模型和runtime scheduling overhead → 确定性执行不足，难以在低batch下持续利用峰值带宽**
  → **RPU方案（Custom ISA + Deterministic Compilation）**：RPU ISA将优化dataflow硬化为CISC-style指令，每条指令执行固定streaming schedule→compiler static order所有DMA和compute指令→消除runtime scheduling overhead。Autonomous execution消除GPU的host-driven offload模型→每core独立执行long-running instruction loop→只在layer边界trigger host interrupt。Pipeline Arbiter flags嵌入每条指令→software-defined但hardware-enforced同步→deadlock-free保证。

  论文方法全栈执行例子（以Llama3-8B, 64-CU RPU, MXFP4 weights, FP8 KV cache, BF16 activations, BS=1, Seq Len=16K为例）：
  - 算法层：不修改模型或推理算法，使用MXFP4 block quantization（Stream Decoder on-the-fly dequantize to BF16）压缩off-chip weight storage。
  - 系统框架/Serving层：Prefill和decode分离（Dynamo/Splitwise execution model）→prefill由GPU处理→KV cache转入RPU HBM-CO memory→RPU执行decode。Host仅在layer transition接收interrupt，不参与per-token kernel offload。
  - 编译框架层：Python compiler trace PyTorch model→lowering torch.nn.Linear到三阶段micro-kernel（Loading/Looping/Launching）→pre-shard weights by C=64 column-wise→pre-quantize to MXFP4→generate synchronized memory/compute/network instruction streams with Pipeline Arbiter flags→static order所有操作。
  - kernel调度层（核心创新）：Decoupled three-pipeline execution。wQKV: network DMA broadcast activation（network latency limited）→memory DMA prefetch weights→compute waits→memory keeps prefilling buffer（~80KB ahead）。QK^T: network gather Q/K/V shards→distributed max collective + exp-sum reduction→compute stalls briefly→memory pipeline prefetches KV$/weights。wUp/wGate: compute runs at full utilization→memory prefetches deep ahead（~6MB/CU at BS=32）。Pipeline Arbiter在buffer entry粒度用valid counter协调，memory始终前进不因compute/network stall而停顿。
  - 硬件架构层（核心创新）：RPU chiplet system。每Core: 4×8×8 TMAC (BF16+FP32), HP-VOPs (FP32 vector ops), Stream Decoder (4-8bit on-the-fly dequant), Memory/Compute/Network DMA + Pipeline Arbiter, 1MB SRAM buffer, 32GB/s HBM-CO channel。每CU: 16 cores, 16MB on-chip memory, 512GB/s BW。每Package: 4 CUs, 2TB/s BW, 64MB on-chip memory。Ring network topology: in-package UCIe-S short-reach (<10mm), off-package PCB-routed interconnect。Llama3-405B at 428 CUs: 1.0ms/token, sustaining >200 TB/s tensor-parallel memory bandwidth。
