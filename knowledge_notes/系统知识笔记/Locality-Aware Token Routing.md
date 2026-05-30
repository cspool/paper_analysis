## Locality-Aware Token Routing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Locality-Aware Token Routing 是 FineEP Algorithm 1 的通信优化策略：LP 求解得到各 replica 目标负载后，路由 token 时优先选择同 GPU 上的 local replica（无网络传输），仅 local replica 满载后才路由到 remote replica。两阶段：(A) 对每个 expert e，遍历 EDP group 中各 GPU g，最多将 min(input_e^g, x_e^g) tokens 分配给 local replica；(B) 剩余 token 分配给 remote replica。

从系统架构角度拆解术语：
以 expert e 在 GPU {0,2} 有 replica, input_e^0=10, x_e^0=8, x_e^2=7 为例：
- Phase A: GPU 0 local 分配 min(10,8)=8 tokens（无通信）；GPU 2 local 分配 min(5,7)=5 tokens。
- Phase B: GPU 0 剩余 2 tokens → 路由到 GPU 2（需通信）；GPU 2 replica 缺 2 tokens 由 GPU 0 补足。

术语一般如何实现？如何使用？
- Algorithm 1 Lines 4-9（Phase A）+ Lines 10-16（Phase B）。
- 效果：使用 NCCL 时 FineEP dispatch 时间低于 EP（因 locality 减少通信量）。
- 扩展：Communication-Aware Scheduling（Appendix A.1）将通信纳入 LP 目标 `min comp + α·comm`。

涉及论文标题：
- FineMoE: Fine-grained Load Balancing for Mixture-of-Experts with Token Scheduling

---
