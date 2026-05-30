## DAPO (Dynamic Sampling Policy Optimization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DAPO 是 ByteDance (Yu et al., 2025) 提出的开源 LLM RL 框架。相比 GRPO 的关键改进：(1) 更高 clipping upper-bound (ε_high=0.28) 避免 entropy collapse；(2) token-level policy gradient（每 token 独立计算梯度，非序列平均）；(3) 移除 KL penalty 消除 exploration 上限；(4) decoupled optimization 简化训练 pipeline。QeRL 在 BigMath 上用 DAPO 训练 7B/14B/32B。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# DAPO (vs GRPO 差异)
# 1. Token-level: loss 对每 token 而非序列平均
for each (q, o_i, A_i):
    for each token t in o_i:
        ratio_t = π_θ(o_{i,t}|q)/π_θold(o_{i,t}|q)
        L_t = -min(ratio_t*A_i, clip(ratio_t,0.8,1.28)*A_i)
    L = mean(L_t)                            # 无 KL 项

# 2. 更高 clip upper (ε_high=0.28 vs GRPO α=0.2)
# 3. on-policy: μ=1 (每次 rollout 后仅 1 次更新)
```
QeRL 配置：G=16, on-policy, max response=8192, clip(0.2,0.28)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/volcengine/verl。DAPO 的 token-level loss 对长序列训练更鲁棒，无 KL 约束使 exploration 更激进——适合 reward 信号明确的数学推理，但对 reward hacking 更敏感。

涉及论文标题：
- QeRL Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs

---
