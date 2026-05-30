## Expert-Balanced Self-Sampling (EBSS / 专家均衡自采样)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert-Balanced Self-Sampling (EBSS) 是 MoEQuant 论文提出的专家均衡校准集生成方法。传统 PTQ 使用固定校准集（如 WikiText2），由于 gating 路由机制，不同 expert 收到的 token 数量极不均匀（长尾分布），导致欠载 expert 校准不足。EBSS 利用 LLM 自身的自采样能力生成校准数据，同时优化两项指标：（1）perplexity（低 PPL 保证与预训练分布一致）；（2）expert balance（σ，即各层 expert 使用频率的标准差）。

EBSS 将目标形式化为联合优化：D* = argmin_D {PPL(M, D) · exp(σ(M, D)/τ)}，其中 τ 控制专家均衡的重要性权重（论文取 τ=1.2）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# EBSS: Expert-Balanced Self-Sampling
# Input: MoE model M, beam width w, seq length n, temperature τ

# Initialize w empty sequences
beams = [""] * w
R = [0.0] * w  # cumulative log-probability per beam

for step in range(n):
    candidates = []
    for b in range(w):
        # Get next-token probability distribution from M
        probs = M.forward(beams[b])  # P(v|S^b) for all v in V
        expt_dist = M.get_expert_distribution()  # expert usage stats
        
        for v in top_k_by_prob(probs, top_k_prune):
            cum_log_prob = R[b] + log(probs[v])
            ppl = exp(-1/(step+2) * cum_log_prob)
            sigma = std(expert_usage_frequencies)
            score = -ppl_log + sigma / τ  # Eq. 13
            
            candidates.append((beams[b] + v, cum_log_prob, v, score))
    
    # Select top-w candidates by score
    candidates.sort(key=lambda x: x[3], reverse=True)
    beams = [c[0] for c in candidates[:w]]
    R = [c[1] for c in candidates[:w]]

D_star = beams  # w sequences, each n tokens
```

关键设计：（1）**Deferred Expert Imbalance Calculation**：候选 token v 不参与当前步的 expert balance 评估，而是使用当前序列 S 的已知 expert 分布（因为遍历词汇表计算每个 token 的 expert 分布开销过大），这实际上做的是 beam 级剪枝而非 token 级剪枝；（2）**Probability-Guided Path Pruning**：只对 vocab 中概率最高的部分 token 展开搜索，忽略低概率分支；（3）**复杂度**：从暴力搜索的 O(m^n) 降至 O(wn)，w=4 即可取得最优效果。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

EBSS 在 MoEQuant 中用于生成替代 WikiText2/C4 的校准集。生成的校准数据有两个特性：（1）低 PPL——与模型预训练分布高度一致（甚至低于 WikiText2 和 C4 的 PPL）；（2）专家均衡——各 expert 分配到的 token 数量接近均匀分布。EBSS 生成的校准集可直接替代 GPTQ/AWQ 的原始校准数据输入，实现插件式集成。论文实验设定 w=4 branches，τ=1.2，sequence length=512（与 WikiText2 校准集相同），在 DeepSeek-MoE-16B 上 EBSS 单独使用带来约 1.3% 的平均分提升。

涉及论文标题：
- MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance
