## GShard

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GShard (Lepikhin et al., 2020, ICLR 2021) 是 Google 提出的大规模 MoE 分布式训练系统，首次将 MoE 扩展到 600B+ 参数。核心设计：(a) 将标准 Transformer 每隔一层的 FFN 替换为 MoE 层（被后续 Switch Transformer、Expert Choice 等工作沿用）；(b) Top-2 gating：每个 token 选择 top-2 scoring experts，第二个 expert 通过概率性路由（noise-based）改善负载均衡；(c) Expert Parallelism：将不同 experts 分布到不同 TPU devices，MoE 层使用 All-to-All collective 进行 token dispatch/combine，非 MoE 层使用标准 DP + AllReduce；(d) Expert Capacity = (tokens_per_device / num_experts) × capacity_factor，超容量 token 通过 residual connection 旁路 MoE 层（token dropping）；(e) Auxiliary load balancing loss = (1/E) × Σ_e (c_e/S) × m_e 鼓励均匀分配。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
GShard 的分布式 MoE 训练执行流程（16 devices, EP=4, EP_DP=4）：
```
EP groups (All-to-All):           EP-DP groups (AllReduce):
  [g0,g1,g2,g3]   expert shard      [g0,g4,g8,g12]  same experts
  [g4,g5,g6,g7]   expert shard      [g1,g5,g9,g13]  same experts

Forward pass per micro-batch:
1. Non-MoE layers: 标准 DP forward (AllReduce not needed per layer)
2. MoE layer:
   a. Router: 计算每个 token 的 top-2 expert assignment
   b. All-to-All dispatch: 将 token 发送到 assigned expert 所在 GPU
   c. Expert FFN: 每个 GPU 在其 expert shard 上批量计算
   d. All-to-All combine: 结果返回原始 GPU
   e. Gate-weighted sum: 合并 top-2 expert outputs
3. Backward: MoE gradients → AllReduce within EP-DP groups
              Non-MoE gradients → standard AllReduce
```
关键系统权衡：All-to-All 通信量 ∝ tokens × hidden_dim，与 expert 总数成对数关系；small batch 时通信成为瓶颈（Expert Choice 论文对比显示 EC-CF2 每步 latency 比 GShard top-2 快 20%，因消除负载不均导致的 straggler）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 基于 XLA 编译器 + GSPMD 自动并行化。使用 `mesh_split` API 标注 tensor sharding，XLA 自动插入 All-to-All/AllReduce collectives。
- 后续影响：Switch Transformer 简化为 Top-1 gating；GLaM 扩展到 1.2T 参数；DeepSpeed-MoE 提供 PyTorch 实现；DeepSeek-V3 使用 fine-grained expert allocation + shared experts。
- Expert Choice Routing 直接以 GShard top-2 为 baseline 对比：相同计算量下 EC-CF2 训练收敛快 2×+，下游 GLUE/SuperGLUE 平均 accuracy 提升 2%+。

涉及论文标题：
- Mixture-of-Experts with Expert Choice Routing
