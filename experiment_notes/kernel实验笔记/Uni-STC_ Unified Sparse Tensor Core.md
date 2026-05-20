## Uni-STC: Unified Sparse Tensor Core

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出UWMMA（Unified Warp-level Matrix Multiply-Accumulate）指令序列和软硬件协同数据流，将稀疏计算任务从"gather data"转为"gather tasks"模式。软件侧用BBC格式表达sparse matrix，硬件侧通过load/task generation/numeric execution三类指令驱动Uni-STC tensor core。指令生命周期：stc.load同步收集metadata和values；stc.task异步触发TMS/DPG生成task queue（T1→T3→T4 task decomposition）；stc.numeric检查READY/BUSY状态驱动SDPU执行，完成结果写回register file。TMS的task ordering在outer-product与row-major顺序间动态选择以提升A/B tile复用并降低write conflict。实验比较SpMV、SpMSpV、SpMM、SpGEMM四类kernel，Uni-STC相对DS-STC和RM-STC的几何平均speedup为3.35x和2.21x，energy reduction为1.97x和1.27x，energy efficiency gain为7.05x和2.96x。

- 后端平台是什么，配置是什么。
  后端平台为GPU SM内集成的Uni-STC coprocessor。GPU架构参考NVIDIA Ampere/A100类设计：432个Uni-STC单元，MAC array配置64 MAC@FP64或128 MAC@FP32。SM需要扩展现有instruction decoder解析UWMMA opcode，扩展warp scheduler分发指令。数据路径通过register file和operand collector：SM90+利用高带宽operand collector，Ampere需拓宽register-file port（每线程每周期最多16个FP64 source + 4个FP64 destination operands）。模拟器基于Accel-Sim扩展STC simulator，加入asynchronous memory access support。

- 评估性能的软件/脚本是什么。修改了什么。
  基于Accel-Sim扩展STC simulator，加入asynchronous memory access支持。论文用同一T1 task调用粒度比较GAMMA、SIGMA、Trapezoid、NV-DTC、DS-STC、RM-STC和Uni-STC。公平比较：按理论计算吞吐对齐MAC array（64 MAC@FP64或128 MAC@FP32），采用SIGMA的PE设计缩放不同accelerator。能耗根据register activity和Sparseloop方法外推。Artifact包含Python/Bash/C++ scripts，qrun自动化复现流程。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：论文未提供GitHub代码仓库，但Artifact Appendix标注Publicly available并提供Google Drive Docker artifact。使用流程：
  1. 导入Docker镜像，配置SuiteSparse/DLMC数据集路径
  2. 运行qrun fast verification（约5小时）：执行SpMV/SpMSpV/SpMM/SpGEMM的kernel-level性能模拟，输出cycle count和energy
  3. 运行qrun complete verification（约75小时）：全量2893矩阵+DLMC+AMG应用级模拟
  4. 结果包含各STC architecture（DS-STC、RM-STC、NV-DTC、GAMMA、SIGMA、Trapezoid、Uni-STC）的performance/energy/area对比数据

