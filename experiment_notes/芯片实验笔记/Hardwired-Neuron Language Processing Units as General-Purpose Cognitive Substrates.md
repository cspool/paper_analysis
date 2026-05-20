## Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates

- 属于芯片设计的实现是什么？实验比较什么？
  实现为Metal-Embedding (ME)方法论和Sea-of-Neurons芯片架构。ME将LLM权重参数从硅器件的2D网格嵌入金属导线的3D拓扑结构中（M8–M11金属层），带来两个收益：(1) 密度提升15×（面积比Cell-Embedding降低93.4%，0.95× vs SRAM面积 vs CE的14.3×），将gpt-oss 120B从200+芯片减少到16芯片；(2) 70层光罩中60层（包括所有EUV光罩）跨芯片同质化可共享。Sea-of-Neurons是metal-programmable structured ASIC，预制造HN Array（M0–M7层），再通过10层DUV光罩（M8–M11）自定义金属嵌入，使光罩成本从$480M降至$65M（初始tapeout）、参数更新respins仅需$37M。实验比较：(1) ME vs CE vs MA的面积/性能/能耗；(2) 系统级吞吐/能效vs H100和WSE-3；(3) TCO经济分析和碳排放分析（3年生命周期，低/高部署量）。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  未使用传统芯片设计模拟器。使用Synopsys全套EDA工具（Design Compiler综合、IC Compiler布局布线）在5nm工艺实现post-layout sign-off。多芯片建模使用CNSim框架。经济分析基于光罩成本模型（EUV=6× DUV权重因子）和公开晶圆/HBM/封装价格数据。

- 模拟器模拟什么的性能，修改了什么。
  布局布线完成sign-off检查：M0–M7和M8–M11均有充足布线密度余量；时序在1.0 GHz闭合（SSG/0.675V/125°C worst-case）；寄生提取确认信号完整性（avg R=164Ω，C=7.8 fF）；热分析确认功耗密度（avg 0.3W/mm²，peak 1.4W/mm²）在2.5D封装冷却极限内；DRC/LVS clean；Murphy模型验证可制造性（缺陷率 0.11/cm²，yield ~43%）。

- 开源情况。论文未明确说明开源情况。

