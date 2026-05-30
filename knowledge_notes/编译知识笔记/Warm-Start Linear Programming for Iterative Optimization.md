## Warm-Start Linear Programming for Iterative Optimization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Warm-Start LP 是一种在连续求解结构相似但参数变化的 LP 问题时复用前次求解状态（basis, simplex tableau, or interior point central path）以加速后续求解的技术。在 FineMoE 的 token scheduling 中，每个 micro-batch 需要求解同一结构的 LPP 1（约束矩阵由 expert placement 决定，跨 micro-batch 不变），仅 RHS `load_e` 随 micro-batch 的 token 分布变化。通过保留前次 HiGHs solver 的 basis factorization（simplex）或 interior point warm-start point，新 micro-batch 的求解时间显著减少。论文报告 LPP solving 总时间 ~100 μs 到 <1 ms，warm-start 贡献了大部分效率增益。

从编译框架角度拆解术语：
Warm-start 的工作机制：
```
// Micro-batch 0（冷启动）
solver = HiGHs()
solver.build(constraint_matrix, bounds)  // O(|E|d) 变量
solution_0 = solver.solve()             // full simplex iterations
warm_state = solver.save_basis()        // 保存 optimal basis

// Micro-batch 1..N（warm-start）
for t in 1..N:
    solver.update_rhs({load_e}_t)       // 仅更新 RHS
    solver.set_basis(warm_state)        // 注入前次 basis
    solution_t = solver.resolve()       // 极少数 simplex iterations
    warm_state = solver.save_basis()
```
关键：Simplex method 在 warm-start 下通常仅需 O(1) iterations（vs 冷启动 O(√n) 或更多），因为 optimal basis 在小 RHS 变化下通常保持不变或仅需 1-2 次 pivot。

术语一般如何实现？如何使用？
- HiGHs 原生支持 warm-start via basis file 或 API（`setBasis` / `getBasis`）。
- 适用条件：约束矩阵不变（由 expert placement 固定），仅 bounds/RHS 变化。
- 局限：(a) Adaptive Replacement 改变 expert placement 时约束矩阵变化，warm-start 失效（需冷启动）；(b) Interior point warm-start 实现更复杂（需保存 central path 信息）。

涉及论文标题：
- FineMoE: Fine-grained Load Balancing for Mixture-of-Experts with Token Scheduling

---
