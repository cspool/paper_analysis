## Module Decomposition for MoE Inference Parallel Strategy

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Module Decomposition for MoE Inference Parallel Strategy 是 HAP 提出的将 MoE 模型按计算特征分解为 Attention 模块和 Expert 模块两个独立计算单元进行并行策略选择的方法。核心洞察：Attention 模块和 Expert 模块在推理时具有截然不同的计算特征（FLOPs 量、参数规模、通信模式敏感性），因此需要不同的并行策略。Attention 模块参数量小但包含 KV cache 内存需求，适合 DP/TP/DP+TP；Expert 模块占总参数约 90%，适合 EP/TP/EP+TP（排除 DP 以节省显存）。每个模块配备专用的推理延迟仿真模型（计算仿真基于 FLOPs，通信仿真基于数据量和带宽），支持对任意策略组合的端到端延迟进行精确估计（计算误差 <10%，通信误差 <5%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

MoE Transformer 层的 Module Decomposition 结构：

```
# MoE Transformer Layer: Module Decomposition View
# 每层包含两个独立模块，可分别选择并行策略

# ┌─ Attention Module ─────────────────────────────────────┐
# │  可选策略: DP, TP, DP+TP                                │
# │  参数量: ~d_model² × 4 (Q/K/V/O) + KV cache            │
# │  计算特征: O(b × s × d_model²)，prefill 计算量大        │
# │  约束: prefill 和 decode 必须使用相同策略 (KV cache)     │
# │                                                        │
# │  Input: h [b, s, d_model]                              │
# │  Q/K/V = h @ W_qkv                                     │
# │  Attention(Q, K, V) = softmax(QK^T/√d_k) × V           │
# │  Output = Attn_out @ W_o                                │
# └────────────────────────────────────────────────────────┘
#                          ↓ h_attn
# ┌─ Expert Module ────────────────────────────────────────┐
# │  可选策略: EP, TP, EP+TP (排除 DP)                      │
# │  参数量: ~(d_model × d_intermediate × 3) × num_experts  │
# │  计算特征: O(b × s × top_k × d_model × d_intermediate)   │
# │  约束: prefill 和 decode 可使用不同策略                  │
# │                                                        │
# │  gate_logits = h_attn @ W_gate  [b×s, num_experts]     │
# │  topk_idx, topk_w = topk(softmax(gate_logits), k)      │
# │  for expert in topk_idx:                               │
# │      expert_out += topk_w × SwiGLU(h_attn, expert)     │
# │  Output = h_attn + expert_out                           │
# └────────────────────────────────────────────────────────┘
#                          ↓
#                     h_next = LayerNorm(h)

# Module Decomposition 的优势:
# 1. Expert Module 在 prefill/decode 可独立切换策略
# 2. Attention Module 不受 Expert 策略影响 (KV cache 独立)
# 3. 仿真模型可按模块粒度分别校准
```

在推理全流程中，每层延迟 = T_attn + T_experts + T_comm。T_attn 取决于 Attention 模块的并行策略（DP=无通信全独立计算, TP=局部计算+AllReduce 聚合），T_experts 取决于 Expert 模块的并行策略（EP=All-to-All dispatch/combine+本地计算, TP=局部计算+AllReduce 聚合），T_comm 是两个模块通信的总和（可能重叠）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Module Decomposition 在 HAP 中实现为 DeepSpeed-FastGen 的扩展。每个模块的可行并行策略由硬件配置（GPU 数、显存、带宽）和模型配置（hidden dim、expert 数、层数）决定。策略空间的构建规则：(1) Attention: DP degree A_d 必须整除 batch size；(2) TP degree A_t 必须整除 hidden dim 和 KV head 数；(3) 总设备数 N = A_t × A_d = E_d × E_t × E_e；(4) EP degree E_e 必须整除 expert 数，TP degree E_t 必须整除 expert 中间维度。内存约束检查包含 KV cache（与 A_d 相关）、Attention 权重（DP 时复制 d 倍）、Expert 权重（各策略下 per-device 相同）、activation（EP 时按 2× TP 保守估计）。Module Decomposition 的设计使得搜索空间从单一策略（TP or EP）扩展为组合空间（Attn DP+Exp EP、Attn TP+Exp TP、Attn DP+Exp TP 等），ILP 在更大的空间中寻找到真正的最优解。

涉及论文标题：
- HAP: Hybrid Adaptive Parallelism for Efficient Mixture-of-Experts Inference
