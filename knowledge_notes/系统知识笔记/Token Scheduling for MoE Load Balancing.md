## Token Scheduling for MoE Load Balancing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Scheduling for MoE Load Balancing 是 FineMoE 提出的一种系统性 MoE 负载均衡方法，通过在每个 micro-batch 中动态调度 token 到 expert replica 的映射来实现细粒度 GPU 负载均衡。核心思想：利用 Expert Data Parallelism（同一 expert 的多个 replica 分布在多个 GPU 上），将 token 调度建模为线性规划问题（LPP），目标 min(max GPU load)，约束为每个 expert 总 load 分配到其 replicas。相比 expert scheduling（FlexMoE/SmartMoE 以 expert replica 为调度单元），token scheduling 的调度粒度从 expert 细化为单个 token，实现近乎连续的调度空间。调度在 per-micro-batch 级别完成，持续适应动态 load 波动。与 algorithmic solutions（load-balancing loss、token dropping）不同，token scheduling 不修改模型 routing 逻辑，不损害模型精度。

从系统架构角度拆解术语：
以 FineMoE (GPT 32×1.3B, DP=8, EP=4, d=2) 单 micro-batch 为例：
1. Gate + Load Collection: 各 GPU 执行 gate network（top-2 routing）→ all-gather 收集全局 `input_e^g`（~32×8 integers, 数 μs）。
2. LP Solving（CPU，与 GPU token permutation 重叠）: HiGHs 求解 min max_g Σ_e x_e^g s.t. Σ_g x_e^g = load_e → `{x_e^g}`。
3. Locality-Aware Routing（Algorithm 1）: 优先将 GPU g token 发给 local replica（无通信），剩余发给 remote replica。
4. All-to-All Dispatch: 8 GPU FineEP group 内 all-to-all → expert FFN → all-to-all combine。
5. Adaptive Replacement（周期性）: 后台监控 load → 预测 → Equation 3 评估 → 必要时生成新 placement。

术语一般如何实现？如何使用？
- 依赖 HiGHs LP solver（单 CPU thread ~100 μs），warm-start 复用上次求解状态。
- 开销：~100 μs（最小）到 <1 ms（64 GPU + 256 experts）。
- 效果：s<1 skewness 下完美均衡；端到端加速最多 47.6% vs Megatron-LM。
- 适用：DP_degree > EP_degree（需多 EP groups 提供 N>1 replicas per expert）。

涉及论文标题：
- FineMoE: Fine-grained Load Balancing for Mixture-of-Experts with Token Scheduling

---
