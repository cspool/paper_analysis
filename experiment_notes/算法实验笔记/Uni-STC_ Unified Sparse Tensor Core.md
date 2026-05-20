## Uni-STC: Unified Sparse Tensor Core

- 属于算法pipeline的实现是什么？实验比较什么？
  提出BBC（Bitmap-Bitmap-CSR）稀疏存储格式和TMS/DPG/SDPU三阶段task decomposition pipeline，统一支持SpMM、SpMV、SpMSpV、SpGEMM四类稀疏算子。BBC用CSR组织4x4 sparse tile，用两级bitmap描述tile内非零位置，使得四类kernel共享同一数据结构而无需在线格式转换。T1(16x16x16)任务经TMS拆分为T3(4x4x4)任务，再经DPG拆分为T4(1x1x4) segmented dot-product任务，最后由SDPU合并执行并预合并partial products。实验比较SpMV、SpMSpV、SpMM、SpGEMM四类kernel的performance、energy和energy efficiency density，对比DS-STC、RM-STC、NV-DTC、GAMMA、SIGMA、Trapezoid，并在DLMC DNN inference（ResNet-50/Transformer，sparsity 70%/98%）和AMG solver上做应用级评估。

- 硬件平台是什么，配置是什么。
  模拟器实验配置：GPU SM内集成Uni-STC tensor core，MAC array配置64 MAC@FP64或128 MAC@FP32。面积用Yosys + FreePDK45 + CACTI 7建模，critical path评估目标频率1.5 GHz。应用级：A100类GPU集成432个Uni-STC单元，额外面积约0.0425 mm²。BBC离线构建在64-core AMD EPYC 7702 CPU和NVIDIA A100 GPU上评估。

- 模型是什么。数据集和bench分别是什么。
  数据集：(1) SuiteSparse 2893个矩阵用于SpMV/SpMSpV/SpMM，2126个方阵用于SpGEMM（C=A²）；(2) DLMC 302个权重矩阵（ResNet-50、Transformer），sparsity 70%和98%；(3) AMG solver的FP64稀疏矩阵。SpMSpV输入向量随机生成50% sparsity，SpMM的B矩阵列数固定64。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：论文未提供GitHub代码仓库。Artifact Appendix标注Publicly available: Yes，提供Google Drive Docker artifact（https://drive.google.com/file/d/1o_pdtPdox7aEdRE2e4GtbEPiMFGpPHCu），含Python/Bash/C++ simulators，Ubuntu 22.04 + GCC/OpenMP/OpenCV，fast verification约5小时，complete verification约75小时需500-600GB存储。算法pipeline：
  1. 预处理：将sparse matrix从CSR/CSC转换为BBC格式（offline，64-core CPU <1000ms或A100 <100ms），BBC=top-level CSR组织4x4 sparse tile + tile内两级bitmap描述非零位置
  2. Load阶段：warp发出stc.load指令，收集A/B矩阵的tile metadata（top-level bitmap, index）和数值到Uni-STC的Matrix A/B Buffer
  3. Task Generation阶段：stc.task指令触发TMS读取top-level bitmap，将16x16x16 T1任务沿M/N/K三维拆为4x4x4 T3 task，写入Tile queue
  4. Task Concatenation阶段：8个DPG并行读取bottom-level bitmap，对T3任务内的A/B tile bitmap做overlay，形成表示sparse dot-product pattern的4-bit code，生成8-bit T4 task code，写入Dot-product queue
  5. Execution阶段：stc.numeric指令驱动SDPU弹出T4任务，执行segmented dot-product（1x1x4），累加到1KB accumulator buffer，最多合并4个partial products后写回C，结果通过register file返回SM
