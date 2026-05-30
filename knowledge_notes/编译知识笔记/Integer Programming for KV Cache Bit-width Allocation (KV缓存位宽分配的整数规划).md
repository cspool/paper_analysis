## Integer Programming for KV Cache Bit-width Allocation (KV缓存位宽分配的整数规划)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Integer Programming (IP) for KV Cache Bit-width Allocation 是 PM-KVQ 中用于求解 transformer block 间非均匀 KV Cache 量化位宽分配的数学优化方法。给定 N 个 transformer block，每个 block i 在候选位宽 b ∈ B 下的 KV Cache 量化敏感度 s_{i,b}，以及总显存预算 M，求解 binary decision variable x_{i,b} ∈ {0,1} 使总敏感度加权和最小。数学形式：min Σ_i Σ_b x_{i,b}·s_{i,b} subject to Σ_b x_{i,b} = 1 (∀i) 和 Σ_i Σ_b x_{i,b}·Mem(b) ≤ M。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

IP 在 PM-KVQ 中充当离线"编译器"：输入为逐 block 敏感度 profile + 硬件显存约束，输出为 per-block 量化配置方案。CVXPY 构建 Boolean 变量 x[N][B]、约束 (one_hot: Σ_b x[i][b]=1 + memory: Σ_i Σ_b x[i][b]*Mem(b) ≤ M)、目标 (min Σ_i Σ_b x[i][b]*s[i][b])，调用 open-source solver (ECOS/GLPK_MI) 求解。对于 N=28-60 blocks、|B|=2 候选位宽的典型问题，求解时间 < 5 秒。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

PM-KVQ 使用 CVXPY + open-source solver。实现分三步：(1) 校准阶段 forward+backward 计算 s_{i,b}；(2) CVXPY solve IP；(3) 推理时每个 block 按分配的 Fbit 独立执行渐进量化。代码开源：https://github.com/thu-nics/PM-KVQ。

涉及论文标题：
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs
