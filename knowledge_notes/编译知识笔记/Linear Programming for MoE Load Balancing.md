## Linear Programming for MoE Load Balancing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Linear Programming for MoE Load Balancing 是 FineMoE 将 MoE 训练中的 GPU 负载均衡问题建模为线性规划（LP）的优化方法。LPP 1 定义：变量 `x_e^g`（expert e 在 GPU g 上的 replica load），约束 `Σ_{g∈EDP^e} x_e^g = load_e`（每个 expert 的总 load 分配完），`x_e^g ≥ 0`（非负），目标 `min max_{g∈G} Σ_{e:g∈EDP^e} x_e^g`（最小化最大 GPU load）。LPP 1 可转化为标准 LP 形式（引入辅助变量 z ≥ Σ_e x_e^g, ∀g，目标 min z）。变量数 O(|E|d)，约束数 O(|E|+|G|)，规模适中（~数百变量），使用 HiGHs solver 在 CPU 单线程上 ≤1 ms 求解。与 Integer Programming for bit-width allocation（MC-MoE）不同，此 LP 是连续优化（x_e^g 为实数），用于运行时 per-micro-batch 决策而非离线一次性分配。

从编译框架角度拆解术语：
LP 作为 MoE 训练中每 micro-batch 的"在线编译器优化 pass"：
```
输入: {load_e = Σ_g input_e^g}（各 expert 的 total token count）
      EDP groups {G_EDP^e ⊆ G}（由 expert placement 决定，跨 micro-batch 不变）
输出: {x_e^g}（各 expert replica 的目标 load）

LPP 1 (standard form):
min z
s.t. z - Σ_{e: g∈EDP^e} x_e^g ≥ 0, ∀g ∈ G    // max load constraint
     Σ_{g∈EDP^e} x_e^g = load_e, ∀e ∈ E       // expert total load
     x_e^g ≥ 0, ∀e∈E, g∈EDP^e
```
注意约束矩阵由 expert placement（G_EDP^e）决定，跨 micro-batch 不变；仅 RHS（load_e）随 micro-batch 变化。这使得 warm-start 非常有效——复用前次 HiGHs 的 basis/simplex state。

术语一般如何实现？如何使用？
- 依赖 HiGHs（开源 LP solver，支持 simplex + interior point），C++ 调用。
- Warm-start 关键：保留上次求解的 basis factorization，新 RHS 下快速 re-optimize。
- Communication-Aware 扩展（LPP 4）：目标改为 `min comp + α·comm`，加入 send_g/recv_g/local_g 变量，仍为 LP。
- 局限：(a) LP 假设 load 可任意细粒度切分（token 为整数但 LP 连续解足够近似）；(b) 极端偏斜下 LP 可能不可行（需配合 AR 调整 placement）。

涉及论文标题：
- FineMoE: Fine-grained Load Balancing for Mixture-of-Experts with Token Scheduling

---
