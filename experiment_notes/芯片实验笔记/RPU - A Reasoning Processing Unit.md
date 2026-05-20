## RPU - A Reasoning Processing Unit

- 属于芯片设计的实现是什么？实验比较什么？
  提出三层芯片设计优化：(1) HBM-CO（Capacity-Optimized High-Bandwidth Memory）：基于HBM core-die floorplan（如HBM3 [35][47][54]），选择性缩减主要贡献容量而非带宽的DRAM结构——将banks per group从4减到1、ranks从4减到1、channels per layer从4减到1、subarray等比例缩减，保留HBM的shoreline bandwidth（102.5 GB/s/mm）和TSV/command/peripheral逻辑不变。候选Pareto-optimal HBM-CO配置：768MB容量、256GB/s带宽、BW/Cap=341、1.45pJ/bit。相对HBM3e：能耗per bit低~2.4×、总模块成本低35×（去掉了192×容量）、带宽/美元高5×，代价是每GB成本高1.81×。(2) RPU Chiplet Compute Fabric：将传统GPU monolithic die替换为chiplet-based设计——每compute chiplet面积约600mm shoreline（vs H100 ~60mm）暴露近10× memory IO shoreline，支持更高perimeter/area比例支撑bandwidth扩展。Package-level使用EMIB [40]或CoWoS-L [25]集成，compute chiplets通过UCIe-S互联，短距in-package 0.5 pJ/bit、off-package 0.75-1.2 pJ/bit。(3) Power/Area Reprovisioning：将70-80% TDP分配给memory interfaces（vs H100的30-40%），compute-to-bandwidth ratio调整为32 OPs/Byte（H100约200 OPs/Byte），去掉underutilized compute和cache资源以降低die cost和功耗。Roofline model在BS=1时工作于pure memory BW bound，BS=32时straddle roofline。实验比较：(1) HBM-CO vs HBM3e的energy per inference和system cost（Fig.9,12）；(2) 不同CU规模下HBM-CO Pareto frontier selection与BW/Cap优化（Fig.11）；(3) RPU vs H100 ISO TDP强扩展下的latency/throughput/energy per inference；(4) RPU system cost breakdown（silicon/memory/substrate/PCB）vs 8×H100 DGX。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  HBM-CO：自研分析模型（analytical modeling approach based on [45]），从HBM core-die floorplan的wire-length scaling trends估算energy per bit和cost per module。RPU compute和package：RTL model使用SystemC + Catapult HLS，target TSMC N16并project到N2。SRAM/interconnect用分析模型。Event-driven simulator整合所有模型。论文未提供HBM-CO模型或RTL的开源链接。

- 模拟器模拟什么的性能，修改了什么。
  HBM-CO分析模型：输入HBM core-die floorplan参数（ranks/banks per group/channels per layer/subarray scaling、TSV/command/peripheral region面积）→计算energy per bit（DRAM array activation + internal wire/TSV movement + IO）和cost per module（硅面积×yield + 封装）→生成BW/Cap tradeoff curves。RPU RTL模型验证VMM/DMA dataflow functional correctness。Event-driven simulator集成HBM-CO模型进行系统级DSE：sweep CU count、HBM-CO configuration（BW/Cap from Pareto frontier）、workload parameters→输出energy per inference、system cost和memory energy breakdown。

- 开源情况。模拟器如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：论文未提供官方开源仓库。HBM-CO设计方法论基于公开的HBM core-die floorplan [35][47][54]和DRAM energy模型 [43][45]。使用流程（论文描述）：
  1. HBM-CO配置选择：根据target deployment（model size, CU count, batch size, sequence length）计算所需per-core memory capacity→从Pareto frontier选择满足capacity的最小BW/Cap配置→权衡energy和cost
  2. Example：64-CU RPU运行Llama3-405B, BS=1, 8K seq len→optimal HBM-CO = 192MB/core (2 ranks, 1 bank/group, 1x subarray)→BW/Cap=171→相对HBM3e降低1.7× energy per inference、5.2× per-device cost、4.3× total system cost
  3. RPU chiplet scale-up：从小scale（如36 CUs, lower BW/Cap）逐步增加CUs→每CU存储更少模型fraction→可使用更高BW/Cap memory→energy per inference改善→在268 CUs时选中design space中最高BW/Cap（HBM-CO upper bound）→energy最优
  4. System cost建模：silicon cost（die area × TSMC N2 wafer price × yield）、memory cost（HBM-CO module × unit cost）、substrate + PCB cost（package和board routing）→对比8×H100 DGX系统cost ratio→HBM-CO系统在scale时memory-to-compute cost ratio与DGX相当，总成本降低up to 12.4×

