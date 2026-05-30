## SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现：基于SN40L streaming dataflow的自动算子融合与kernel调度。编译器自动将PyTorch级别算子图编译为空间融合dataflow kernel，将20+ operators融合到单个kernel launch（参见Figure 11，llama70B-4k-inf-prefill的fused/unfused kernel call比率约11×）。两种kernel调度模式对比：(1) Software Orchestrated (SO) — host CPU通过AGCUs发出Program Load→Argument Load→Kernel Execute命令序列调度kernel；(2) Hardware Orchestrated (HO) — AGCUs内置硬件kernel调度器，将静态kernel schedule offload到硬件，消除host往返延迟。
  实验比较：在8-socket SN40L Node上，对Table III所列benchmark（Llama2-7B/70B、sparseGPT-13B、Bloom-176B、Mistral-7B、Falcon-40B、LLaVA1.5-7B、FlashFFTConv）对比三种配置：(a) Unfused — 每个PyTorch operator作为独立kernel执行，中间结果materialize到HBM/DDR；(b) Fused+SO — 编译器自动融合+host软件调度；(c) Fused+HO — 编译器自动融合+AGCUs硬件调度。测量speedup（Figure 10）和kernel call数量比（Figure 11）。

- 后端平台是什么，配置是什么。
  SN40L RDU（8 socket Node for大多数benchmark，单socket for FlashFFTConv）：638 BF16 TFLOPS/socket，1040 PCU + 1040 PMU，SRAM 520 MiB，HBM 64 GiB/1.8 TB/s，DDR 1.5 TiB/200 GB/s。额外16 socket用于Llama 3.1推理benchmark。

- 评估性能的软件/脚本是什么。修改了什么。
  SambaNova自研编译器（非开源）— 接收PyTorch/Python级别模型描述，自动将计算图编译为PCU/PMU/AGCU/RDN上的空间融合dataflow kernel。编译器核心修改/功能包括：(1) 静态符号生命周期分析实现garbage collection — 将非重叠生命周期的逻辑符号映射到相同设备虚拟地址；(2) 符号temporal locality分析 + 带宽估计 — 决定哪些符号溢出到DDR（优先溢出总传输带宽最小的符号）；(3) 静态带宽建模 — 建模RDN/TLN上的并发数据流带宽需求，指导PCU/PMU资源分配；(4) Place-and-Route层 — 配置RDN routing table、flow ID、multicast路径。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未开源。Kernel调度评估原理和执行流程：以FlashFFTConv benchmark为例，原始PyTorch实现包含多个GEMM、element-wise Mul、Transpose等算子的Monarch FFT decomposition。Unfused配置下，每个算子编译为独立kernel — 数据流：HBM→AGCU→RDN→PCU执行GEMM→RDN→AGCU→materialize到HBM→AGCU→RDN→PCU执行Mul→RDN→AGCU→HBM...每步都产生HBM读写。Fused配置下，编译器将整段Monarch FFT编译为单个kernel：HBM通过AGCUs流式加载→PCU(systolic GEMM)→PMU(stage buffer I0)→PCU(SIMD Mul)→PMU(transpose via data alignment unit, T0*→T1*)→PCU(systolic GEMM)→AGCUs写回HBM。操作强度从39.5 Ops/Byte提升至410.4 Ops/Byte，FlashFFTConv实现13× speedup。性能测量：在硬件上运行warmup + timed iterations，使用SN40L switch和PMU内置performance counter监控RDN拥塞和bank冲突。
