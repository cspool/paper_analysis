## Branch-Train-MiX (BTX)

术语解释
由 Sukhbaatar et al. (FAIR at Meta, 2024) 提出的一种三阶段 LLM 继续预训练方法，将 embarrassingly parallel 的领域专家训练与 Mixture-of-Experts 架构结合，使多个领域专用 LLM 合并为一个统一的 MoE LLM。

术语是什么？
BTX 包含三个阶段：(1) **Branch**：从预训练 Seed LLM（如 Llama-2 7B）复制 N 份，每份作为领域专家初始化；(2) **Train**：各副本在对应领域数据（如 Math、Code、Wikipedia）上独立继续预训练，完全无同步通信（embarrassingly parallel），训练吞吐线性 scaling；(3) **MiX**：将所有 expert 的 FFN 子层组合为 MoE 层（每个 expert FFN 成为 MoE 的一个 expert），self-attention 和 embedding 等参数直接平均，随机初始化 Router W_l，在全部数据混合上进行 MoE finetune 学习 token 级路由。

```
# === BTX 三阶段算法 ===

# 阶段一: Branch
seed_model = Llama-2-7B
experts = [copy(seed_model) for _ in range(N)]  # N=3 for Math/Code/Wiki

# 阶段二: Train (embarrassingly parallel, 无同步)
for expert_i, data_i in zip(experts, domain_datasets):
    # 各 expert 在不同 GPU 组上完全独立训练
    for batch in data_i:
        loss = CrossEntropy(expert_i(batch), labels)
        loss.backward()
        optimizer_i.step()  # 无 all-reduce

# 阶段三: MiX
# 3a. 组合 FFN 为 MoE 层
for layer_l in range(L):
    # Attention: 对所有 expert 平均
    W_attn[l] = mean([expert_i.attn[l] for i in range(N)])
    # FFN: 构建 MoE
    moe_ff[l] = MoELayer(
        experts = [expert_i.ffn[l] for i in range(N)],  # 4 experts
        router  = Linear(4096, N, init=random)           # 唯一新增参数
    )
    # Router: Top-2 + Load Balancing
    logits = x @ router.weight        # [seq, N]
    top2_vals, top2_idx = TopK(logits, k=2)
    weights = SoftMax(top2_vals)
    output = sum(weights[i] * experts[i](x) for i in top2_idx)

# 3b. MoE Finetune (学习路由)
L_total = CrossEntropy(output, labels) + α * N * sum(u_i * p_i)
# 其中 u_i = mean(g_i(W_l x)), p_i = mean(SoftMax_i(W_l x)), α=0.01
moe_model.train(all_data_mixture, tokens=80B)
```

BTX 泛化两个特例：(1) BTM = 100% expert training + 0% MoE finetune；(2) Sparse Upcycling = 0% expert training + 100% MoE finetune。BTX 在两者之间分配 compute（expert training 512B tokens + MoE finetune 80B tokens），取得最优 accuracy-efficiency tradeoff。

BTX 路由分析关键发现：无 load balancing 时 Code expert 成为 "dead expert"（不被激活），load balancing 使其 "back to life" 并在 code/math domain 成为主导 expert；freeze FFN experts 在 MoE finetune 时对性能几乎无影响（34.7 vs 34.7），说明 domain knowledge 已在 expert training 阶段获得，MoE finetune 主要训练 router 和调优平均的 attention 权重。

术语一般如何实现？如何使用？
- Seed 模型：Llama-2 7B（32 layers, hidden 4096, FFN 11008, 32 heads）
- Expert 训练：Math 48k steps/201B tokens, Code 50k steps/210B tokens, Wiki 42B tokens
- MoE Finetune：80B tokens（Math 30.16%, Code 40.31%, Wiki 10.30%, Llama-2 data 19.23%）
- 最终 BTX 有 4 expert（Math + Code + Wikipedia + original Llama-2 7B generalist）
- 路由选择：Top-2 routing with load balancing α=0.01 为默认；Sample Top-1 更高效（可训练 160B tokens）
- 激活参数：Top-2 11.1B, Sample Top-1 6.7B（vs seed 7B）
- 论文未公开开源代码

涉及论文标题：
- Branch-Train-MiX Mixing Expert LLMs into a Mixture-of-Experts LLM
- BTS Harmonizing Specialized Experts into a Generalist LLM

---
