## Memory Update Rules in Linear Sequence Models (统一视角下的记忆更新规则)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Memory Update Rules 是从递归形式统一描述各种线性序列模型的记忆更新公式。所有线性模型都可以从"memory state 更新"的角度统一表达为 M_t = f(M_{t-1}, k_t, v_t) 的形式。不同方法差异在于：(1) 是否有数据依赖的门控参数；(2) 门控是标量还是向量；(3) 更新是基于外积（k_t^T v_t）还是基于梯度。

MoM 论文 Table 1 提供了完整的统一视角：

| Method | Memory Update Rule |
|--------|-------------------|
| Linear Attn | M_t = M_{t-1} + k_t^T v_t |
| RetNet | M_t = γ M_{t-1} + k_t^T v_t |
| GLA | M_t = (a_t^T 1) M_{t-1} + k_t^T v_t |
| DeltaNet | M_t = (I - k_t^T k_t) M_{t-1} + b_t k_t^T v_t |
| G-DeltaNet | M_t = a_t (I - k_t^T k_t) M_{t-1} + b_t k_t^T v_t |
| TTT | M_t = M_{t-1} + b_t ∇l(M_{t-1}; k_t, v_t) |
| Titans | M_t = a_t M_{t-1} + b_t ∇_M l(M_{t-1}; k_t, v_t) |
| Mamba2 | M_t = a_t M_{t-1} + b_t k_t^T v_t |
| HGRN2 | M_t = (a_t^T 1) M_{t-1} + (1 - a_t)^T v_t |
| RWKV6 | M_t = a_t M_{t-1} + k_t^T v_t |
| RWKV7 | M_t = (a_t^T 1) M_{t-1} + b_t ∇l(M_{t-1}; k_t, v_t) |

其中 a_t, b_t ∈ (0,1) 通常是数据依赖的标量门控参数，γ 是数据无关常量。

关键演进趋势：(1) 早期方法数据无关（Linear Attn, RetNet 的 γ）；(2) 中期引入标量数据依赖门控（Mamba2, RWKV6）；(3) 近期引入向量门控（GLA, HGRN2）或 Delta Rule 自适应更新（G-DeltaNet）；(4) 最前沿方法引入 test-time regression 的梯度更新（TTT, Titans, RWKV7）。

从算法pipeline角度拆解术语。

所有 update rule 都可以融入 MoM 框架：将单一 memory M_t-1 替换为 M 个独立 memory M_t^m，每个 memory 独立执行 update rule，最终通过 router 权重混合。

```
通用 MoM 更新流程:
  for each activated memory m:
    k_t^m = x_t @ W_k^m           # memory-specific key
    v_t^m = x_t @ W_v^m           # memory-specific value
    # 任选 update rule:
    M_t^m = a_t^m · M_{t-1}^m + b_t^m · k_t^{m,T} v_t^m  # 示例: Mamba2 风格
```

术语一般如何实现？如何使用？

Memory update rules 通过 Triton chunk-wise parallel scan kernel 实现：将序列切分为 chunks，chunk 内并行矩阵运算（intra-chunk），chunk 间以 recurrent 方式传递 memory state（inter-chunk）。MoM 在此基础上增加了 token reordering（按 routing 结果分组）和 varlen 支持。代码开源：https://github.com/OpenSparseLLMs/Linear-MoE。

涉及论文标题：
- MoM: Linear Sequence Modeling with Mixture-of-Memories

---
