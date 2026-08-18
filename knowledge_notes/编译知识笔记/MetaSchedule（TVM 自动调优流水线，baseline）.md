## MetaSchedule（TVM 自动调优流水线，baseline）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MetaSchedule 是 Apache TVM 的自动调优/自动调度框架（Ansor 之后、基于 TensorIR）：用可学习的进化式搜索（进化搜索 + 随机森林/XGBoost cost model + 数据库回放）在调度规则空间里为算子自动搜索高性能 kernel，支持 CUDA、TensorCore、CPU 等后端，提供 task scheduling、tuning database、数据库 replay 等基础设施。QiMeng-Tensify（ISCA'26）把它作为最重要的编译框架 baseline（TVM commit 567eeed3，MetaSchedule 自动调优流水线，FP32 用默认 CUDA 后端、FP16 用 Tensor Core 后端）：它依赖手工设计的规则应用策略（表 I：对"有数据复用且可融合"子图静态选 Tiling+Fusion），只适合单算子或简单子图，对 GatedMLP 之类图级计算只能切成两个子图分别优化、错过跨算子融合。

从编译框架角度拆解术语，比如术语所在编译框架的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
作为 baseline 的运转流程（GatedMLP）：输入计算图 → MetaSchedule 按硬编码策略把图划分成预定义子图（fuse matmul+SiLU+mul 为一个、另一个 GEMM 单独）→ 对每个子图在规则空间（>1e10）进化搜索（进化算子变异/交叉 tile size、unroll 等）→ cost model 筛选 + 真机测量回填 → 输出两个分离 CUDA kernel。局限：图划分与融合策略是专家写死的（Reasoning Compiler/AMOS 也继承 TVM 的图划分与融合策略）；最终 kernel 无法表达 SiLU/MUL 与第二个 GEMM 的融合与 partial reduction（QiMeng-Tensify 案例中 TVM MetaSchedule 最优程序慢 2.80×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：TVM 中 meta_schedule 包，TuneTIR 等 API 对算子/图自动调优，结果存 tuning database 供 relay 图编译回放。使用方式：作为论文对比实验的 exploration-based baseline；其进化搜索被 Reasoning Compiler 用 LLM-guided MCTS 替换（Reasoning Compiler 是"把 MetaSchedule 的进化搜索换成 LLM 引导 MCTS"，但仍受 TVM 图划分约束）。编译时间对比（A100，小时）：TVM(MetaSchedule) GatedMLP 3.08/SelfAtten 3.66/LoRA 2.98/QKNorm 1.53/nTrans 2.57，均慢于 QiMeng-Tensify（1.37/1.83/1.92/1.17/1.69）。

涉及论文标题：
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
