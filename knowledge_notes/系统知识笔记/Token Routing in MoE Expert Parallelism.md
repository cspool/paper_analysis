## Token Routing in MoE Expert Parallelism

术语解释
Token Routing 是 Expert Parallelism 中的第三阶段（在 expert replication 和 placement 之后），负责将每个 GPU 上的 token 按 top-k gating 结果路由到对应 expert 的具体 replica 上。它决定每个 expert replica 是否被激活来处理 token。

术语是什么？
在 EP MoE 推理中，token routing 解决的问题是：对于每个 token，经过 router 选出 top-k experts 后，若某个 expert 有多个 replicas（分布在多个 GPU 上），应将 token 发送到哪个 replica？传统方法（EPLB default）将每个 expert 的 token 均匀分配到所有 replicas 以平衡各 GPU 的 token 数。METRO 提出 counter-intuitive 的方法：在 memory-bound 的 decode 阶段，应将每个 expert 的所有 token 集中路由到单个 replica 以最小化 activated experts 数量，从而减少 HBM → Tensor Core 的 weight 加载内存流量。

从系统架构角度拆解术语：
Token routing 在 EP inference pipeline 中的位置：

```
=== MoE Layer Forward Pass with Expert Replication ===

1. Gate/Router: x → W_g → TopK → selected experts {e1, e2}
2. Token Routing (本术语): 
   对于选中的每个 expert e，token 应发到 e 的哪个 replica？
   
   EPLB routing (token-balancing):
     e1 有 3 replicas (GPU 0,3,5)，9 tokens 选中 e1
     → 每个 replica 收到 3 tokens（均匀分配）
     → 3 个 replicas 全部激活，各自计算 3 个 token
     
   METRO routing (expert-minimizing):
     e1 有 3 replicas (GPU 0,3,5)，9 tokens 选中 e1
     → 查 L[0..2] = 当前各 GPU activated expert 计数
     → 选 L 最小的 GPU（如 GPU 5, L[5]=0）
     → 所有 9 tokens 路由到 GPU 5 的 e1 replica
     → 仅 1 个 replica 被激活
     
3. All-to-all/All-gather Dispatch: 实际数据传输
4. Expert FFN: 在激活的 replica 上计算
5. All-to-all Combine: 结果返回原 GPU
```

术语一般如何实现？如何使用？
- Token routing 通常与 expert placement 协调——placement 决定 replicas 的 GPU 位置，routing 决定运行时 token 流向
- 实现方式：(a) **Token-balancing**（EPLB, Tutel, FasterMoE 等）：均匀分配 tokens；适合 compute-bound prefill 阶段；(b) **Expert-minimizing**（METRO）：最小化 activated experts；适合 memory-bound decode 阶段
- METRO 的路由决策在单 SM 的 CUDA kernel 中完成，使用 test-and-set lock 和 SM shared memory 计数器
- 路由质量衡量：max activated experts per GPU per decode batch

涉及论文标题：
- Efficient MoE Serving in the Memory-Bound Regime Balance Activated Experts, Not Tokens

---
