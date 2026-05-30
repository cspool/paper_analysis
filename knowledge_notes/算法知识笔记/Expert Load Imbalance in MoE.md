## Expert Load Imbalance in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Load Imbalance 是 MoE 模型的固有特征：由于 gate network 是动态可训练的，不同 expert 被路由到的 token 数量存在显著差异。Lazarus 论文（Figure 2）展示了 GPT-L (16 experts) 训练过程中，up to 87% tokens 被路由到最热门的 2 个 experts，而最冷门的 experts 几乎不被激活。这种不平衡在不同层之间、以及同一层的不同训练迭代之间动态变化（gate network weights 随时间更新）。

在传统 EP 中，所有 experts 被等分到不同 GPU，load imbalance 直接导致 GPU 间计算不均衡——持有 popular experts 的 GPU 处理远超其他 GPU 的 token 数，其他 GPU idle waiting。这不仅降低了训练吞吐（straggler effect），也使得故障恢复更困难（如果 cold expert 的唯一 replica 所在 GPU 故障）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

MoE 层中 expert load 的计算过程：

```
# MoE layer forward pass (per layer, per iteration)

# Step 1: Gate network computes routing probabilities
gate_logits = Linear_gate(hidden_states)  # [B, S, E]
gate_probs = softmax(gate_logits)          # [B, S, E]

# Step 2: Top-k selection + auxiliary load balancing loss
# Load balancing loss (Switch Transformer style):
# f_e = (1/T) * Σ_{tokens} 1{top-k includes expert e}     # fraction of tokens
# P_e = (1/T) * Σ_{tokens} softmax(gate_logits)[e]         # avg gate prob
# L_balance = E * Σ_e f_e * P_e                             # scalar loss
# Total loss = LM_loss + α * L_balance

# Step 3: Expert load t_e for iteration
t_e = Σ_{tokens} 1{e in top-k(token)}   # actual token count per expert

# Lazarus observation: t_e varies significantly
# e.g., 16 experts: t_1 ≈ t_2 ≈ 43% each, t_15 ≈ t_16 ≈ 0.5% each
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

缓解 Expert Load Imbalance 的常见策略：(a) **Load Balancing Loss** (Switch Transformer, GShard)：在 training loss 中加入辅助项惩罚不均衡路由；(b) **Expert Capacity**：设置每个 expert 的最大 token 容量，超出则 drop tokens；(c) **Dynamic Parallelism Switching** (Tutel, SmartMoE)：根据 expert load 动态切换 parallelism 策略；(d) **Expert Replication** (FasterMoE, FlexMoE, Lazarus)：为 popular experts 分配多个 replicas 增加计算容量。Lazarus 使用策略 (d) 在弹性训练环境下——根据运行时收集的 routing history 的 t_e，用 Eq. 1 计算自适应 replica 分配，使 r_e ∝ t_e。

Lazarus 的消融实验（single MoE layer with 8 experts）显示：当 load ratio 从 1:1 (balanced) 变为 4:1 (imbalanced)，DS baseline 吞吐急剧下降（因 straggler GPU），而 Lazarus 通过 adaptive expert allocation 保持恒定吞吐。

涉及论文标题：
- Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement
- Least-Loaded Expert Parallelism: Load Balancing An Imbalanced Mixture-of-Experts

**LLEP 对 Expert Load Imbalance 的洞察**：

LLEP 从系统而非算法角度重新框定了 expert load imbalance。论文实证分析 gpt-oss-20b (32 experts, 8-way EP) 在数学数据集上的路由模式：(a) 特定 expert (E11) 持续接收最多 token（up to 20% load vs 3% balanced）；(b) 某些 GPU 整体过载（GPU 0 有 30-35% vs 12.5% balanced）；(c) 不均衡程度 per-batch 动态变化。

LLEP 的核心论点：mild imbalance 是训练良好的 MoE 的自然属性（专家专业化），而非需要算法层面修正的缺陷。强制均衡路由（如 auxiliary load balancing loss）会破坏已学习的专家专业化模式。因此 LLEP 采取系统级负载均衡——在 dispatch 阶段动态将超载 GPU 的 excess load 溢出到欠载 GPU，保持 exact MoE computation。α 容量因子、m 最小 GEMM token 数、λ 自适应阈值构成超参调优空间，允许用户根据硬件配置权衡均衡度与通信开销。
