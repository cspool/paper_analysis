## Expert-wise Model Parallelism

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert-wise Model Parallelism 是 Lory 论文（Section 6）提出的一种用于扩展 Fully Differentiable MoE 模型至 100B+ 参数的并行策略。与传统稀疏 MoE 使用的 Expert Parallelism（每个设备持有不同专家的完整参数，通过 all-to-all 通信 dispatch/combine tokens）不同，Expert-wise Model Parallelism 按 hidden dimension 对所有专家进行分片：每个设备持有所有专家在 hidden dim 上的一个分片，即设备 d 持有 expert 0 的 hidden dim shard_d、expert 1 的 hidden dim shard_d、...、expert E-1 的 hidden dim shard_d。

核心优势：在软路由（soft routing）MoE 中，每个 token 需要所有专家的参数参与合并（计算加权平均 θ̄ = Σ e_i · θ_i）。若使用传统的 Expert Parallelism（每个设备持有部分专家完整参数），合并操作需要 all-gather 所有专家参数，通信量随专家数 E 线性增长。Expert-wise Model Parallelism 下，每个设备本地计算其 hidden dim 分片上的合并参数，通信仅需传递合并后的 hidden activations（不随 E 增长）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Expert-wise Model Parallelism 的训练并行策略（图 7）：

```
=== Expert-wise Model Parallelism (Partition along hidden dim) ===
设备分配（以 4 devices, 8 experts, hidden dim d_ffn=4096 为例）:
  Device 0: expert[0:8].W_gate[:, 0:1024], W_up[:, 0:1024], W_down[:, 0:1024]
  Device 1: expert[0:8].W_gate[:, 1024:2048], W_up[:, 1024:2048], W_down[:, 1024:2048]
  Device 2: expert[0:8].W_gate[:, 2048:3072], W_up[:, 2048:3072], W_down[:, 2048:3072]
  Device 3: expert[0:8].W_gate[:, 3072:4096], W_up[:, 3072:4096], W_down[:, 3072:4096]

前向传播（一个 MoE layer）:
1. Router 计算 (all devices, no communication):
   e = softmax(R(h_bar))  # (E-dim) routing weights
2. 各 device 本地合并其分片上的专家参数:
   Device i: merged_W_gate_i = Σ_j e_j * expert_j.W_gate[:, i*1024:(i+1)*1024]
   Device i: merged_W_up_i   = Σ_j e_j * expert_j.W_up[:, i*1024:(i+1)*1024]
   Device i: merged_W_down_i = Σ_j e_j * expert_j.W_down[:, i*1024:(i+1)*1024]
3. 分发输入 token 到各 device (non-MoE 部分用 data parallel):
   各 device 已有完整 token hidden state h (data parallel copy)
4. 各 device 计算其分片的 FFN 输出:
   Device i: gate_i = SiLU(h @ merged_W_gate_i.T) (shape: T, 1024)
   Device i: up_i   = h @ merged_W_up_i.T
   Device i: out_i  = (gate_i ⊙ up_i) @ merged_W_down_i.T
5. All-gather 各 device 的 output shard (沿 hidden dim 拼接):
   output = concat(out_0, out_1, out_2, out_3)  # (T, d)
```

混合并行策略（图 7 right）：
- **非 MoE 组件**（Attention、LayerNorm）：Data Parallelism——每个 DP group 中的设备处理不同 batch data，无需通信（或仅 all-reduce gradients）
- **MoE 层**：Expert-wise Model Parallelism——按 hidden dim 分片，每 group 内设备 all-gather hidden activations（不随 E 增长）
- 这种分层策略避免了传统 Expert Parallelism 中 all-to-all 通信随 E 增长的瓶颈

与 Expert Parallelism 的对比：
- **Expert Parallelism** (传统稀疏 MoE)：all-to-all dispatch/combine tokens，通信量 ∝ K·T·d（K 为激活专家数）。适用于只有 k 个专家被激活的场景（稀疏）
- **Expert-wise Model Parallelism** (Lory)：all-gather hidden activations，通信量 ∝ T·d_ffn（hidden dim 分片大小）。不依赖专家数 E——因为本地已有所有专家的对应分片，合并操作零额外通信

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
- **适用场景**：当模型参数超过 100B，数据并行 + ZeRO 的参数通信成为瓶颈时（每个 device 需维护所有专家完整副本或通过 ZeRO 同步）
- **通信效率**：hidden activation 的通信量不随专家数 E 增长（仅与 hidden dim 和 token 数相关），而 Expert Parallelism 的 token dispatch 通信量随 E 增加（更多 target expert groups）
- **与 ZeRO 的关系**：Lory 实验中使用的 ZeRO 数据并行可视为 Expert-wise Model Parallelism 的前置步骤——当专家数少时（E≤32），ZeRO 已足够；E 和模型规模更大时，才需切换至 Expert-wise Model Parallelism
- **未被论文实验验证**：Lory 论文仅在 Section 6 讨论了该策略的设计思路，未实现和实验验证。论文将此列为未来工作。
- **相关方法**：MoEShard (2025) 实现了专家矩阵的 tensor sharding 用于推理（row- 和 column-wise 分解），与 Lory 的 hidden-dim sharding 思路类似但面向推理。Learn to Shard (2025) 使用 RL 自动搜索最优 sharding 维度组合。

涉及论文标题：
- Lory: Fully Differentiable Mixture-of-Experts for Autoregressive Language Model Pre-training
