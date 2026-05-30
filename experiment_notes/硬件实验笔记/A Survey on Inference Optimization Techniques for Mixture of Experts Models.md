## A Survey on Inference Optimization Techniques for Mixture of Experts Models

- 属于硬件架构的实现是什么？实验比较什么？
  本论文是综述，不提供原始实验。它在硬件级别（Section 5）综述了以下硬件架构方案：
  - **MoNDE**：近数据处理（NDP）方案，通过CXL-based NDP控制器+专用NDP核实现内存内计算。采用混合计算策略——GPU处理高频"hot"expert、NDP单元处理"cold"expert，以Activation Movement替代传统Parameter Movement范式。
  - **Duplex**：结合xPU和Logic PIM的设备，为每层执行选择合适目的地。集成两种共享内存的处理单元（计算密集型xPU + 内存密集型PIM），通过高带宽TSV实现DRAM die和Logic die间通信，并行执行expert和attention阶段。
  - **Space-mate**：面向移动设备SLAM任务的加速器设计，包含Out-of-Order SMoE Router（减少数据事务降低延迟）、Single Skip和Dual Skip异构核心架构（利用同expert内相似零模式实现粗粒度稀疏性，提高吞吐量和能效）。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - **MoNDE**：论文未明确说明使用的模拟器。其架构基于CXL（Compute Express Link）协议，可能在CXL模拟器或自研仿真环境上评估。使用LPDDR SDRAM作为内存。
  - **Duplex**：论文未明确说明模拟器。涉及PIM（Processing-in-Memory）微架构设计和TSV（Through-Silicon Via）建模。
  - **Space-mate**：使用ISSCC 2024发表的芯片测量数据（303.5mW实时稀疏MoE NeRF-SLAM处理器），可能是实际芯片流片结果而非纯模拟器仿真。
  - **M3ViT/Edge-MoE**：可能使用Xilinx Vivado/Vitis HLS等FPGA开发工具进行综合和仿真。

- 模拟器模拟什么的性能，修改了什么。
  - **MoNDE**：模拟CXL NDP架构下的expert计算性能。修改：将传统的Parameter Movement（移动expert参数到GPU）改为Activation Movement（移动激活值到NDP），GPU和NDP并行处理hot/cold expert。
  - **Duplex**：模拟xPU+PIM混合架构的性能。修改：设计了替代PIM微架构（logic PIM），优化低Op/B操作；通过TSV实现高带宽通信。
  - **Space-mate**：评估移动端SLAM的加速器性能。修改：设计了OoO SMoE Router（乱序路由器）和SS/DS异构核心。

- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？至少具体到模拟器模拟性能的原理和模拟器输入到性能输出的全过程。
  **开源情况**：论文未统一说明各硬件方法的开源情况。以下基于论文描述推断：

  **MoNDE NDP架构执行原理与流程**：
  1. 输入：模型定义（MoE各层expert配置）、workload trace（token序列和expert激活模式）
  2. Hot/Cold分类：基于expert访问频率统计，将expert分为hot（GPU常驻）和cold（NDP处理）
  3. 内存映射：Hot expert权重分配在GPU HBM，Cold expert权重分配在NDP附带的LPDDR SDRAM
  4. 运行时执行：
     - Router计算完成后，对于hot expert → GPU本地计算
     - 对于cold expert → 通过CXL将activation值发送到NDP核，NDP核在LPDDR SDRAM内执行expert计算
     - 结果通过CXL返回GPU
  5. 性能输出：延迟（latency）、带宽利用率、能效比
  6. 模拟原理：基于CXL协议带宽模型 + LPDDR SDRAM带宽模型 + NDP核计算能力模型，计算hot/cold混合场景下的端到端延迟

  **Duplex PIM架构执行原理与流程**：
  1. 输入：MoE模型配置、batch请求
  2. 层执行目的地选择：对每层评估在xPU vs Logic PIM上的Op/B效率
  3. 低Op/B操作（如expert FFN）→ Logic PIM执行（高带宽TSV通信）
  4. 高Op/B操作（如Attention）→ xPU执行
  5. Expert和Attention阶段并行执行最大化效率
  6. 性能输出：吞吐量、能效、内存带宽利用率
