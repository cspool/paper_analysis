## Uni-STC: Unified Sparse Tensor Core

- 属于硬件架构的实现是什么？实验比较什么？
  提出Uni-STC统一稀疏张量核硬件架构，替代GPU SM中原有dense tensor core，包含三个核心功能单元：(1) TMS（Task Merge & Splitting unit）：从Meta Buffer读取top-level bitmap，将16x16x16 T1任务拆分为4x4x4 T3 task，写入Tile queue，内部含M-merge/K-merge逻辑和task ordering决策；(2) DPG（Dot-product Generation unit）×8：并行读取bottom-level bitmap，对A/B tile bitmap做overlay生成sparse dot-product pattern的4-bit code，结合C tile结构信息生成8-bit T4 task code，写入Dot-product queue；(3) SDPU（Segmented Dot-Product Unit）：弹出合并后的T4任务，执行segmented dot-product（多个短向量点积），累加到1KB accumulator buffer，merge-forward结构在写C前预合并最多4个partial products。架构特点：两层Benes/MUX network + dynamic DPG activation（TMS根据queue head的intermediate product prefix sum决定需开启DPG数量，多余DPG及关联网络power-gate）。面积用Yosys+FreePDK45+CACTI 7建模，critical path满足1.5 GHz，在A100类826 mm² die上额外面积约0.0425 mm²（2.12%）。实验对比GAMMA、SIGMA、Trapezoid、NV-DTC、DS-STC、RM-STC的面积/能耗/性能。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  基于Accel-Sim扩展STC simulator。Accel-Sim为开源GPU模拟器（https://github.com/accel-sim/accel-sim）。论文在其上加入：asynchronous memory access support、TMS/DPG/SDPU cycle-level pipeline model、UWMMA指令序列模拟、BBC format数据路径、Benes/MUX network延迟模型、dynamic DPG activation power-gating模型。面积评估用Yosys开源综合工具（https://github.com/YosysHQ/yosys）+ FreePDK45工艺库。能耗和面积建模部分参考CACTI 7（https://github.com/HewlettPackard/cacti）和Sparseloop方法。

- 模拟器模拟什么的性能，修改了什么。
  模拟GPU SM内Uni-STC的cycle-level性能：(1) TMS tile queue occupancy和task generation吞吐；(2) DPG dot-product queue occupancy和task concatenation吞吐；(3) SDPU execution latency和accumulator buffer利用率；(4) 两层network的数据移动能耗；(5) dynamic DPG activation的power-gating效果（DPG数量从4到8的EED trade-off）。修改：在Accel-Sim warp scheduler和instruction decoder中增加UWMMA opcode解析和dispatch逻辑；增加asynchronous memory access模型以支持stc.load与stc.task/stc.numeric的异步重叠；实现T1→T3→T4的task decomposition tracing以统计MAC utilisation周期分布。

- 开源情况。模拟器如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：论文Uni-STC simulator通过Google Drive Docker artifact提供，未找到GitHub仓库。Accel-Sim（https://github.com/accel-sim/accel-sim）、Yosys（https://github.com/YosysHQ/yosys）、CACTI 7（https://github.com/HewlettPackard/cacti）均为开源。模拟器使用流程：
  1. 启动Docker容器，加载预编译的Accel-Sim+Uni-STC扩展
  2. 配置GPU参数（SM数量、Uni-STC单元数、MAC array宽度、DPG数量等）
  3. 输入BBC格式矩阵和kernel类型（SpMV/SpMSpV/SpMM/SpGEMM），运行cycle-level模拟
  4. 输出：performance（cycle count/speedup）、energy（nJ/operation）、MAC utilisation分布、network traffic、EED（GFLOPS/W/mm²）
  5. 面积验证：Yosys综合Uni-STC RTL（FreePDK45），CACTI 7建模SRAM buffer（Meta Buffer、A Buffer、accumulator buffer），计算critical path验证目标频率
