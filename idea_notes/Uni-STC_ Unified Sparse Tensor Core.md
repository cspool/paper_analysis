## Uni-STC: Unified Sparse Tensor Core

- baseline方法是什么？
  Baseline包括DS-STC（面向SpGEMM的outer-product路径）、RM-STC（面向SpGEMM的row-row路径）和NV-DTC（NVIDIA dense tensor core）。DS-STC以A的半列和B的半行为基本单位组织outer-product任务，RM-STC以scalar-vector组合组织row-row任务。两者都采用"gather data"模式：将sparse data gather成固定形状的T2/T3子任务后送入MAC array，遇到长行、长列、对角集中或双边稀疏矩阵时，固定形状任务导致大量低MAC utilisation周期（DS-STC和RM-STC分别有61.68%和62.78%的周期低于50% utilisation），且大中间乘积网络持续搬运带来能耗。

  全栈执行例子（以RM-STC执行SpMV为例）：
  - 算法层：输入CSR格式稀疏矩阵A和dense vector x，RM-STC用row-row scalar-vector组合方式沿N维拼接任务到固定MAC array
  - 编译框架层：论文未明确说明（无专用编译框架，硬件直接解析CSR-derived任务）
  - kernel调度层：warp读取CSR row一行的非零值→硬件decoder将非零位置映射到固定shape T2/T3任务→送入MAC array执行partial product→结果通过shfl_gather聚合
  - 硬件架构层：RM-STC硬件含hardware decoder解析CSR格式生成固定shape任务、固定width network搬运中间乘积、accumulator做partial sum。缺陷：遇到长行时一行被拆成多个独立任务而无法跨K维拼接；遇到窄行时MAC利用率低；decoder面积开销大

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Uni-STC的核心设计哲学从"gather data"转为"gather tasks"。方法包括四层协同设计：(1) BBC格式用CSR组织4x4 tile+两级bitmap描述tile内非零，避免CSR hardware decoder和跨kernel格式转换；(2) TMS/DPG将固定形状的T1→T3→T4 task做三级灵活分解，T3在M/N/K三维对称便于统一多kernel支持，T4的1x1x4 dot-product片段可以跨多低负载任务灵活拼接；(3) SDPU的segmented dot-product+merge-forward在写C前预合并最多4个partial products，减少中间乘积网络流量；(4) TMS的task ordering在outer-product与row-major间动态选择提升A/B tile复用，dynamic DPG activation按需power-gate多余DPG。

  全栈执行例子（以Uni-STC执行SpGEMM为例）：
  - 算法层：输入BBC格式矩阵A和B（一次offline转换，<100ms），warp通过stc.load收集tile metadata/values到Uni-STC的Matrix A/B Buffer
  - 编译框架层：论文未明确说明（UWMMA通过汇编级stc.load/stc.task/stc.numeric指令序列编程，无高级编译器）
  - kernel调度层：stc.task触发TMS→从Meta Buffer读top-level bitmap→按K维动态拼接T3 task（避免RM-STC沿K维无法拼接的缺陷）→8个DPG并行将T3转为T4 sparse dot-product task→以Z-shaped顺序入Dot-product queue→stc.numeric检查READY驱动SDPU执行→相邻低负载T4拼接成segmented dot-product
  - 硬件架构层：两层精简network仅搬运控制信息（T4 code）而非全量中间乘积→SDPU内局部执行短向量点积→merge-forward预合并partial products后写C→dynamic power-gating按需关闭闲置DPG和network。量化效果：Uni-STC相对DS-STC和RM-STC几何平均speedup 3.35x和2.21x，energy reduction 1.97x和1.27x，SuiteSparse全量矩阵上的MAC低利用率周期比例显著降低
