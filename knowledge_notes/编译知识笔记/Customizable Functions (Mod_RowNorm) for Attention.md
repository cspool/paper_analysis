## Customizable Functions (Mod/RowNorm) for Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Customizable Functions（可定制函数）是 MetaAttention 编程接口的核心机制，允许用户在固定的 attention 模板（relevance scoring + aggregation）中注入自定义的数值变换，以表达不同的 attention 变体。分为两类：(1) **Modification Function (Mod)**——支持细粒度元素级变换，包括 scaling（如 Q_mod: scale Q by 1/√d_k in standard softmax attention）、masking（如 scores_Mod: multiply with causal/sparse/bool mask）、output transformation（如 output_Mod: apply final scaling/activation）。Mod 仅含 elementwise 操作，无 reduction；(2) **Row-wise Normalization Function (RowNorm)**——支持含 row-reduce 的全局行调整，如 softmax（reduceMax + exp + reduceSum + div）、sigmoid（exp + reduceSum）、L2 norm、RetNet 的 reduceAbsSum-based normalization。RowNorm 可定义为 online 版本（RowNorm online interface），使 normalization 在分 tile 处理 KV sequence 时逐步更新状态变量，无需物化完整 score matrix。

从编译框架角度拆解：MetaAttention 编译框架处理 customizable functions 的完整流程：
1. **Tracing**：用户 Python 函数被 trace 为 directed acyclic graph (DAG) of tensor operations。每个 node 分类为 elementwise（add/mul/tanh/exp 等）或 row-reduce（reduceSum/reduceMax/reduceAbsSum 等）。
2. **Lowering**：elementwise nodes → SIMT-style register-level fused execution（利用 CUDA core 的 warp-level parallelism）；row-reduce nodes → intra-warp parallel reduction（warp shuffle + sync，最小化 synchronization 开销）。DAG 节点携带 metadata（tensor shape、dependency、grad field for autograd）。
3. **Code Inlining**：Map 后的 hardware-specific code snippets 被 inline 到 attention kernel 模板的固定位置——scores_Mod 在 QK^T 之后、RowNorm 之前；scores_RowNorm 替代 kernel 模板中的 normalization 逻辑；output_Mod 在 aggregation 之后。Inlining 保证 customizable logic 零额外 kernel launch overhead，且受益于与核心 attention 相同的 memory-efficient pipelining 和 hardware-native 优化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

用户使用 Python 定义 customizable functions（22-90 LoC per attention variant），无需编写 CUDA/Triton 代码。示例——定义 RetNet attention：
```python
# 定义 scores_Mod: 应用 mask
def scores_Mod(scores):
    return scores * mask

# 定义 scores_RowNorm: reduceAbsSum-based normalization
def scores_RowNorm(scores):
    t = scores.reduceAbsSum()
    t = max(t, 1)
    return scores / t
```
或使用 RowNorm online 接口获得更高性能（避免 HBM 中间写入）。编译框架自动 tracing → lowering → inlining。限制：customizable functions 目前仅支持 elementwise 和 row-reduce 操作（覆盖绝大多数 attention 变体的数值变换需求），不支持引入新的 matmul 或全局依赖操作。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework across Hardware Backends
