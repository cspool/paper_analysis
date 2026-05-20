## VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

- 属于硬件架构的实现是什么？实验比较什么？
  提出VAR-Turbo专用accelerator，由以下模块组成：(1) Unified Attention Core：PE Cluster/Node/Cell三级层次，BA模式（Big Attention）采用Row dataflow按attention head分配PE Cluster，SA模式（Small Attention）采用OP dataflow，通过Snooper配置PE Cell接收的packet ID，由Fat Tree分发到lanes实现dataflow动态切换。Row+OP MAC通过Divide-and-Conquer、Fluid Zone Detection和shared FP accumulator降低FP累加功耗面积。(2) Radix Sort Core：执行CountBin→PrefixSum→SelectBin→Filter四阶段大K TopK dataflow，加入Locality-aware Scheduling利用confidence map空间偏斜优先处理高置信区域。(3) MLP Core（执行FFN计算）、Non-Linear Core（softmax、element-wise等辅助操作）、SIMD Core + global memory + weight memory + DMA + AXI interconnect + 控制器。RTL以SystemVerilog实现，Synopsys Design Compiler在TSMC 28nm+HPC 1P8M CMOS、TT 25C下综合，DCG+ICC2完成placement/routing并得layout/netlist，经DRC/LVS后以VCS+PrimeTime做post-layout simulation获片上功耗。片外DRAM使用2×64bit HBM2 channel @2GHz, 32GB/s。V100对比使用scale-up建模：14 TFLOPS、16 HBM2 channels即512GB/s。实验比较：vs Xeon 8168 CPU、V100 GPU、ViTCoD、AdapTiV，VAR-Turbo平均speedup分别为5047.4×、210.3×、6.1×、3.8×；平均energy-efficiency improvement分别为24818.2×、423.5×、6.0×、7.8×。面积7.09 mm²，功耗1.98 W。Attention Core与Memory为主要面积/功耗来源；Radix Sort Core仅占4.9%面积和6.3%功耗。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  论文自研cycle-accurate simulator（未开源，无公开链接）。RTL仿真使用Synopsys VCS；综合使用Synopsys Design Compiler + DCG + ICC2；功耗分析使用Synopsys PrimeTime PX。模拟器与RTL延迟匹配率：ViTCoD 0.96、AdapTiV 0.94、VAR-Turbo 0.90。baseline ViTCoD/AdapTiV RTL在相同工艺条件下重新评估以公平对比。

- 模拟器模拟什么的性能，修改了什么。
  Cycle-accurate simulator模拟VAR-Turbo accelerator执行VAR图像生成的完整pipeline：Attention Core的Row和OP两种dataflow切换及PE Cluster/Node/Cell cycle-level延迟；Radix Sort Core的四阶段TopK流水线延迟（含Locality-aware Scheduling）；MLP Core/Non-Linear Core/SIMD Core执行延迟；global memory/weight memory访问延迟；DMA和AXI interconnect数据传输。模拟器输入VAR模型配置+图像分辨率，输出execution cycles和energy consumption。论文为公平比较，将ViTCoD/AdapTiV限定在同一ViT backbone上通过simulator评估。

- 开源情况。模拟器如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：VAR-Turbo accelerator RTL和cycle-accurate simulator均未开源。硬件设计流程（基于论文描述）：
  1. RTL设计：SystemVerilog实现Unified Attention Core（PE Cluster/Node/Cell+Row/OP dataflow+Snooper+Fat Tree）、Radix Sort Core（TP+CountBin+PrefixSum+SelectBin+Filter+Locality-aware Scheduling）、MLP Core、Non-Linear Core、SIMD Core等→Synopsys VCS功能仿真+测试用例校验
  2. 综合+布局布线：Synopsys Design Compiler (TSMC 28nm)综合→DCG+ICC2 placement/routing→DRC/LVS→生成layout、netlist
  3. 后仿真：VCS+PrimeTime PX post-layout simulation获取片上功耗（1.98W）和per-module功耗
  4. 性能模拟：cycle-accurate simulator加载VAR model config+图像分辨率→模拟VAR-Turbo pipeline（PD/TA/DB算法→Unified Attention Core/Radix Sort Core/MLP Core执行）→输出speedup和energy efficiency
  5. Area/Power breakdown: Attention Core和Memory占主要面积/功耗；Radix Sort Core仅4.9%面积、6.3%功耗，但解决大K TopK延迟瓶颈（传统排序方案TopK仅3.5%操作数却占20.9%延迟）
