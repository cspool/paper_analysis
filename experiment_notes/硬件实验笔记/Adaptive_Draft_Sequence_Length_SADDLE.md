## Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems

- 属于硬件架构的实现是什么？实验比较什么？
  提出SADDLE完整PIM-enabled heterogeneous硬件架构，包含三大部分：(1) SADDLE Manager：Draft Generator (Controller + Eager Pool)、Shared Pool、Scheduler。Controller集成softmax unit、multipliers和comparators实现低延迟H_t更新和阈值τ比较；Shared Pool (1KB, CAM-based)跨micro-batch聚合draft tokens；Eager Pool (1KB, 按micro-batch划分，每pool最多512 tokens)暂存乐观执行的draft tokens；另有1KB SRAM存logits和累积接受概率。(2) SADDLE PIM Device：centralized processor (A100 GPU)、router、多个PIM chips。PIM芯片基于HBM-PIM商业架构，buffer die堆叠于8 DRAM dies之上通过TSV垂直互联。每DRAM die 8个独立pCH，每pCH 4 bank groups (各4 banks)。每bank附1 PE (16 FP16 multipliers + 16 FP16 adders + registers, 256-bit operands/cycle从local row buffer和pCH global buffer获取)，pCH内所有PE跨bank并行最大化内部带宽。pCH含global buffer和accumulator做局部partial result聚合。(3) SFU (Specialized Functional Unit)：集成在每HBM stack buffer die上，支持softmax、layer normalization、activation、residual addition等非矩阵运算。实验比较：throughput/energy efficiency/latency breakdown/GPU+PIM utilization/communication cost/area overhead，对比GPU-AD/GPU-SD/PIM-AD/PIM-SD。SADDLE相比GPU-AD/GPU-SD/PIM-AD/PIM-SD平均能效提升6.81×/5.96×/2.32×/1.45×。GPU utilization提升1.13×(vs PIM-AD)和1.37×(vs PIM-SD)，PIM utilization提升1.84×(vs PIM-AD)和1.18×(vs PIM-SD)。面积开销：每DRAM die ~16.24mm²、每buffer die ~1.62mm²（28nm综合缩放到DRAM process），占121mm² HBM3 die的~13.4%。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  构建cycle-accurate simulator，修改Ramulator2 (DRAM simulator, 开源 https://github.com/CMU-SAFARI/ramulator2) 和ATTACC (PIM accelerator simulator, ASPLOS'24论文)。PE面积/能耗评估用Synopsys Design Compiler (28nm, 1GHz) + 缩放到1z-nm DRAM process。HBM energy modeling参考prior work [34] activation/read energy值。Ramulator2为开源，ATTACC代码状态论文未说明。

- 模拟器模拟什么的性能，修改了什么。
  模拟器输入系统配置和模型规格，输出execution time和energy consumption for GPU systems和SADDLE。模拟器模拟：(1) GPU vs PIM operator mapping下的执行延迟和能耗；(2) HBM-PIM内部bank-level并行度和带宽利用；(3) cross-pCH和cross-stack通信延迟；(4) pipeline parallelism跨S组PIM devices的micro-batch调度。论文修改：在Ramulator2+ATTACC上增加SADDLE-specific modules——Controller (H_t更新+阈值比较)、Shared Pool/Eager Pool (CAM-based token存储和migration)、Scheduler (arithmetic intensity估算和动态remapping决策逻辑)、SFU (softmax/layer norm运算延迟建模)。pipeline parallelism建模各micro-batch在各pipeline stage的processing time和inter-stage communication。

- 开源情况。模拟器如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：SADDLE simulator未开源。Ramulator2开源 (https://github.com/CMU-SAFARI/ramulator2)。模拟器使用流程：
  1. 配置输入：系统配置（8 PIM devices × 5 HBM3 stacks, PIM PE规格, SFU配置, Manager Pool大小, threshold τ）+ 模型规格（OPT-66B/175B, Llama3.1-70B的d_model, layers, heads）+ workload参数（batch size 16-128, max seq length 1024/512, Dolly dataset requests）
  2. 模拟执行：Ramulator2模拟DRAM access timing→ATTACC模拟PIM PE computation timing→SADDLE模块模拟Controller draft length决策 (H_t=∏p_i < τ停止drafting)、Shared Pool token accumulation和verification trigger、Eager Pool optimistic token暂存和migration、Scheduler arithmetic intensity估算和operator remapping
  3. 输出：每system的end-to-end execution time和energy consumption→计算normalized throughput (tokens/s)和energy efficiency (tokens/J)
  4. 面积验证：Synopsys Design Compiler (28nm)综合PE和SFU RTL→面积缩放到DRAM 1z-nm process→与HBM3 die 121mm²对比确认overhead (13.4% DRAM die, ~1% buffer die)
