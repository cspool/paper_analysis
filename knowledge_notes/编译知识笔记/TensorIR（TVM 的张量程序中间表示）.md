## TensorIR（TVM 的张量程序中间表示）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TensorIR 是 Apache TVM（Ansor 一脉）的张量程序（tensor program）中间表示：把"算子算法"与"调度（schedule）"分离——IR 描述算子的计算语义（循环结构、tensor 读写、buffer），调度原语（tiling/fusion/vectorize/parallel 等）在不改变语义的前提下重写循环结构，从而在一个 IR 上穷举不同低层实现。它统一了 TVM 早期 relay/topi 与逐算子手写 CUDA 之间的鸿沟，是 MetaSchedule、Meta-Scheduler 等自动调优框架的承载 IR。QiMeng-Tensify（ISCA'26）直接以 TensorIR 作为程序定义与初始 IR：初始 IR 捕获算子（如 GatedMLP 的 GEMM/SiLU/mul）的计算语义，之后的全部图重写（7 条调度规则）、LLM 引导的 MCTS 搜索、细粒度参数规格与最终 CUDA 代码生成都围绕 TensorIR 进行——LLM 的 prompt 里也直接嵌入 TensorIR prim_func 文本。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在编译框架中的运转流程（以 QiMeng-Tensify 优化 GatedMLP 为例）：输入算子（O = SiLU(X·W1) ⊗ (X·W2)）→ 前端写成 TensorIR（含三个 GEMM 的 buffer 声明与循环）→ MDP 状态化（状态 = 计算图 G + 调度节点 n）→ 每一步图重写动作把 TensorIR 的循环结构改写（AutoInline 把 SiLU 的 exp/add/div/mul 折叠进单一 block；MultiLevelTiling 在共享 (i0,j0,k0) 循环下重排 GEMM1/GEMM2；ComputeAtLocation 把 SiLU/MUL 的 compute 位置挪进 GEMM 的 reduction 循环）→ 搜索结束后由 TVM 后端（codegen）把最终 TensorIR 降到 CUDA kernel。也就是说 TensorIR 是"图重写动作的作用对象"与"LLM 观察程序结构的文本载体"（prompt 中 Given the following TensorIR: {prim_func}）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：TVM 中以 Python 前端（tvm.script / tir prim_func 语法）描述算法，调度原语（tile/fuse/reorder/compute_at/vectorize/unroll/bind 等）以 imperative 方式应用；IR 是带 buffer/stmt/expr 节点的 AST，可序列化为文本（LLM 可直接阅读）。使用方式：在 QiMeng-Tensify 中 (1) 程序定义用 TensorIR（捕获语义）、(2) 7 条图重写规则在该 IR 上做结构变换、(3) LLM 读取 TensorIR 文本输出 1×7 规则概率、(4) 细粒度参数规格对 IR 填 tile size/unroll 参数、(5) 最终 CUDA 生成。基线 TVM/MetaSchedule 也以 TensorIR 为 IR，其硬编码规则策略（如表 I 的 Tiling+Fusion 固定策略）正是 QiMeng-Tensify 要替换的对象。

涉及论文标题：
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
