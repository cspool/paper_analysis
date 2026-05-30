## MIQCP for Serverless MoE Deployment Optimization（Serverless MoE 部署优化的MIQCP建模）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mixed-Integer Quadratically Constrained Programming (MIQCP) 是一类数学优化问题，目标函数或约束中包含二次项，且部分变量为整数。该论文将 serverless MoE 模型的最优部署形式化为一个 MIQCP 问题（公式 12a-12k），联合决策：(1) 通信方法选择 a_e ∈ {1,2,3}；(2) 每个 expert function 的内存配置 x_{e,i,j} ∈ {0,1}；(3) 每个 expert 的副本数 y_{e,i,g} ∈ {0,1}；(4) pipeline degree β ∈ Z。目标是最小化所有 MoE 层的 billed cost Σ c_e。

从编译框架角度拆解术语：
MIQCP 求解 = 编译框架中的最优配置选择 pass：

1. **问题分解**：由于 a_e（通信方法）是离散变量且与其他变量耦合，将原 MIQCP 分解为三个子问题（分别固定 a_e=1, a_e=2, a_e=3），每个子问题仍是 MIQCP 但规模减小。
2. **线性化**：max 函数（如 MoE-E2E latency 中的 max over experts）通过辅助变量 φ 线性化：φ ≥ h, ∀h ∈ H。
3. **求解**：用 Gurobi solver 求解三个子 MIQCP（时限 60s/子问题），得到三组候选解。
4. **ODS 后处理**：ODS 算法从三组解中逐层选择最低 cost 的通信方法，若不满足 E2E latency 约束则迭代替换最高延迟层的方法。算法复杂度 O(|E|)。

关键约束包括：
- (12c)：内存约束 `P_{e,i} + M^{itrm}_{e,i} + r_{e,i}(D^{in}+D^o) ≤ Σ x_{e,i,j} · M_j`
- (12d)：E2E 延迟约束 `T^{head} + T^{tail} + Σ (t^{lat}_e + T^{NE}_e) ≤ T^{limit}`
- (12f)：payload 约束 `(a_e-3)(r_{e,i}·D^{in} - D^p) ≤ 0`（当 a_e=3 时要求数据量 ≤ payload）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Python + gurobipy API 构建 MIQCP 模型 → `model.optimize()` 求解。
- 论文对 MIQCP 直接求解（不分解）设定 180s 时限，ODS 方法三个子问题各 60s。ODS 在高吞吐量目标下优于直接 MIQCP（后者 180s 内无法收敛）。
- 理论保证：ODS 得到的 billed cost 上界为 OPT 的 M_{|M|}·G·(U_1+max{1/B^s,1/B^f}+T^{dl})/(U_{|M|}+min{1/B^s,1/B^f}) 倍。
- 适用场景：资源分配问题中包含离散选择（通信方法）、连续配置（memory size）、整数变量（replica count、pipeline degree）混合时，MIQCP + decomposition 是有效的建模方法。

涉及论文标题：
- Optimizing Distributed Deployment of Mixture-of-Experts Model Inference in Serverless Computing
