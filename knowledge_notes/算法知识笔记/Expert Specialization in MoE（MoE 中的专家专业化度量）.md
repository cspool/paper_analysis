## Expert Specialization in MoE（MoE 中的专家专业化度量）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Specialization（专家专业化）是指 MoE 模型中不同 expert FFN 对特定数据域或任务形成差异化处理能力的现象。一个"专业化"的 expert 意味着它在处理其专长域的数据时被 router 频繁选择。衡量专业化程度的标准方法是计算路由概率矩阵：对每个域采样的 token，统计其在各 Transformer block 中被路由到各 expert 的平均概率。DeepSeek-MoE (Dai et al., 2024) 将"ultimate expert specialization"作为核心目标。Nexus 系统性地量化了 upcycled MoE 中的 expert specialization：通过计算每个域 token 跨所有 Transformer block 的平均路由频率矩阵（Figure 5）来验证。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Expert Specialization 度量（Nexus Figure 5 方法）
for domain_d in [ArXiv, Books, C4, SE, Wiki, Code]:
    samples = sample_from_domain(domain_d, n=512)
    routing_count = zeros(n_experts)
    for each token in samples:
        for each MoE layer l:
            expert_chosen = router.forward(token)  # top-1 index
            routing_count[expert_chosen] += 1
    routing_freq[domain_d] = routing_count / sum(routing_count)
# 理想专业化: routing_freq[domain_d][expert_d] → 1.0
# Nexus 结果: ArXiv→ArXiv: 63.0%, Books→Books: 64.7%, Wiki→Wiki: 69.8%,
#             C4→C4: 40.9% (C4 覆盖广), Code→Code: 69.1% (新增后)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **度量工具**：通过 hook router 的 top-k 选择（记录每个 token 的 selected expert index），在评估集上统计路由频率矩阵。
- **影响因素**：(a) 域数据的覆盖范围（C4 因覆盖广导致路由分散到其他 expert）；(b) 训练数据采样策略（均匀采样 vs 比例采样——Nexus 发现均匀采样将 C4 路由精度从 27.6% 提升至 71.1%）；(c) load balancing loss factor（过高会强制分散路由，降低专业化显示度）。
- **与 vanilla MoE 的对比**：standard MoE training 中专家通常不展示明确域专业化（Jiang et al., 2024; Zoph et al., 2022），因为 router 仅基于 token hidden state 选择 expert，无域语义信息作为路由依据。Nexus 通过域嵌入 router 的归纳偏置实现并保持了专业化——这是区分"语义专业化"（由数据驱动）和"统计均衡"（由 load balancing 驱动）的关键洞见。

涉及论文标题：
- Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts
- Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models

**局部路由一致性视角下的专家专业化** (来自 "Not All Models Suit Expert Offloading", ICLR 2026)：论文将 Expert Specialization 区分为两种类型：(1) Domain Specialization——expert 对不同领域数据的激活频率差异（用 Coefficient of Variation across domains 量化）；(2) Vocabulary Specialization——expert 对特定 token ID 的激活频率差异（分为 input/predicted output/ground-truth 三种）。关键发现：Domain-specialized experts 对局部路由一致性的贡献显著大于 vocabulary-specialized ones；高 SRP 且 global load balance 良好的模型（如 Qwen3, GRIN-MoE, OLMoE）同时具有强 domain specialization。机制：domain-specialized expert 在匹配其专长领域的上下文中持续激活（高局部一致性），而在不相关领域则保持 inactive（实现全局负载均衡）。Paper Figure 7 展示了 SRP 与 domain specialization 的正相关，而与 input vocabulary specialization 的负相关或无显著相关。

---
