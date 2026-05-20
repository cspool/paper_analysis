## Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates

- 属于硬件架构的实现是什么？实验比较什么？
  实现为HNLPU（Hardwired-Neuron LPU）完整硬件架构，将gpt-oss 120B（FP4）模型权重物理固化到硅片中。芯片由五大模块组成：HN Array（硬连线权重计算，基于accumulate-multiply-accumulate的Hardwired-Neuron架构，权重参数嵌入金属互联而非硅器件）、VEX Unit（向量执行单元：FlashAttention计算、RMSNorm、SwiGLU、softmax、residual add、multinomial sampling，每cycle处理32 cached KV-heads无停顿）、Attention Buffer（320MB片上KV Cache，20,000 banks，每bank 16KB，1W1R，32-bit访问宽度，80 TB/s带宽）、Control Unit（on-chip调度和inter-layer pipeline管理）、Interconnect Engine（CXL 3.0 inter-chip通信）。系统由16芯片经4×4行列全连接fabric组成，每芯片面积827.08 mm²、功耗308.39W，总die面积13,232 mm²。实验比较：(1) Embedding methodology对比：单层矩阵向量乘法(1×1024 input × 1024×128 FP4 weight)在5nm下对比MAC Array (MA，64KB SRAM+1024 MACs)、Cell-Embedding (CE)和Metal-Embedding (ME)的面积/执行周期/能耗；(2) 系统级对比：完整HNLPU系统 vs NVIDIA H100 (TensorRT-LLM部署) vs Cerebras WSE-3 (云端实测)，均运行gpt-oss 120B model with 2K token length。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  硬件实现使用完整ASIC设计流程而非模拟器：RTL以Verilog实现，Synopsys Design Compiler综合，Synopsys IC Compiler布局布线，PrimeTime PX分析功耗（workload-derived SAIF文件），Memory Compiler生成片上SRAM，均在5nm工艺节点。多芯片系统建模使用CNSim（开源多芯片系统分析框架）。论文还构建了cycle-level单芯片性能模拟器。

- 模拟器模拟什么的性能，修改了什么。
  CNSim模拟16芯片间CXL 3.0协议的通信延迟和功耗，包括PHY延迟、协议开销和物理布线延迟。Cycle-level simulator模拟单芯片性能。论文未说明对CNSim的具体修改。

- 开源情况。论文未明确说明HNLPU RTL或相关工具的开源情况。CNSim为开源框架（论文引用[25]）。
