## RPU - A Reasoning Processing Unit

- 属于硬件架构的实现是什么？实验比较什么？
  提出RPU（Reasoning Processing Unit），一种面向低延迟LLM decode的chiplet-based硬件架构，由三层硬件抽象组成：(1) Package Architecture：每package集成4个Compute Unit (CU)，每个CU含一个compute chiplet和两个HBM-CO chiplet（共2TB/s memory bandwidth per package）。Compute chiplets通过UCIe-S互联（in-package 0.5 pJ/bit, off-package 0.75-1.2 pJ/bit）；(2) Compute Unit (CU)：含16个Reasoning Core、16 MB on-chip memory、512 GB/s memory bandwidth。每core拥有独立HBM-CO DRAM channel（32 GB/s）和local SRAM buffer，全NUMA无shared memory；(3) Reasoning Core：每core含4个8×8 TMAC（Tile Multiplier, BF16×BF16→FP32 accumulation, weight-streaming output-stationary dataflow）、HP-VOPs（高精度向量操作单元，处理SiLU/GeLU/normalization/RoPE）、Stream Decoder（on-the-fly BFP/MxFP/NxFP 4-8 bit dequantization）、Memory/Compute/Network三套DMA engine和Pipeline Arbiter（SRAM buffer entry粒度valid counter同步）。System-level通过Ring Station和PCB board-level ring扩展多package。实验比较：(1) 强扩展：Llama3-8B/70B/405B和Llama4-Maverick/Scout在ISO TDP下对比H100，BS=1/8K seq len下RPU latency up to 45.3× lower、throughput up to 18.6× higher（Llama3-405B）；(2) Batch scaling：BS=1-32/128，Llama4-Maverick在128 CUs下达80%+ BW utilization至BS=128；(3) Energy per inference：HBM-CO vs HBM3e，RPU达到6.5× lower EPI（Llama3-405B, 428 CUs）和412× EDP improvement；(4) 与Groq/Cerebras/SambaNova/H200在speculative decoding下对比throughput。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  自研event-driven simulator + RTL模型（SystemC/Catapult HLS）。RTL target TSMC N16并project到N2。VMM/DMA microkernel通过VCS、Design Compiler、PowerPro提取能耗和面积。SRAM/interconnect用分析模型建模。HBM-CO能耗通过自研分析模型（基于wire-length scaling from HBM core-die floorplans [35][47][54]）。论文未提供simulator/RTL开源链接或仓库。

- 模拟器模拟什么的性能，修改了什么。
  Event-driven simulator用symbolic transaction（address/size/type）替代真实tensor data，参数由RTL calibrate。模拟全模型DSE：覆盖model（Llama3/Llama4系列）、batch size（1-128）、sequence length（8K/16K/32K/64K/128K）、memory device（HBM-CO variants）、deployment scale（36-500+ CUs）。暴露瞬态行为：buffer occupancy、pipeline stall、synchronization delay、memory/compute/network utilization和power traces per CU。RTL model验证functional correctness和dataflow behavior（如2k×2k VMM on 4-core RPU ~6.5min）。模拟器还用于debug：data dependency violation表现为execution stall，可visual trace。

- 开源情况。模拟器如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：论文未提供官方开源仓库或Artifact Evaluation链接。模拟器使用流程（基于论文描述）：
  1. RTL calibration：SystemC + Catapult HLS实现proof-of-concept RPU→VCS/Design Compiler/PowerPro synthesize VMM/DMA microkernel→提取calibrated throughput/bandwidth/latency参数
  2. Compiler生成instruction stream：Python compiler trace PyTorch model graph→lower to RPU ISA (CISC-style instructions with operand addresses/tensor dims/data types/Pipeline Arbiter flags)→static order DMAs→pre-shard and quantize weights→generate synchronized memory/compute/network instruction streams
  3. Event-driven simulation：simulator加载compiled instruction stream→model data transfers as symbolic transactions→calibrated parameters applied per event→输出per-kernel timeline (memory/compute/network utilization, buffer occupancy, power per CU)
  4. DSE：sweep CU count (36-500+)、HBM-CO config (BW/Cap ratio)、batch size和sequence length→输出latency (ms/token)、throughput (tokens/s)、energy per inference、system cost
  5. 硬件平台依赖：RTL simulation需要Synopsys工具链（VCS, Design Compiler, PowerPro）和TSMC工艺库（N16→N2 projection）；event-driven simulator为Python-based，可独立运行
