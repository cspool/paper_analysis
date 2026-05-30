## Distribution-Only Prediction (仅分布预测)

术语是什么？
Distribution-Only Prediction 是 MoE-GPS 提出的一种轻量级 expert 预测策略。与预测每个 token 具体路由到哪个 expert（Token-to-Expert Prediction）不同，Distribution-Only Prediction 仅预测 coarse-grained 的 token 分布比例（如 Expert 1 将收到 75% 的 tokens），不指定哪些具体 token 去哪个 expert。它使用 Multinomial Distribution + MLE (Maximum Likelihood Estimation) 建模每层 MoE 的 expert 激活概率：$\hat{p}_i^l = n_i^l / N$，其中 $n_i^l$ 为训练集中第 l 层 expert i 被激活的总次数，N 为总 token 数。该预测是 offline 完成的，运行时 zero overhead。

从算法pipeline角度拆解术语：
```
# Offline 训练阶段
for layer l in 1..L:
    for batch in training_data:
        expert_assignments = MoE_Router(hidden_states, layer=l)
        for expert e in 1..E:
            n_e[l] += count(expert_assignments == e)
    p_hat[e][l] = n_e[l] / total_tokens[l]  # MLE

# 推理阶段（每层）
predicted_distribution = p_hat[:, l]           # 各 expert 预期 token 比例
target_tokens_per_gpu = total_tokens / G      # 均衡目标
P, d = ExpertDuplication(f, predicted_distribution, M, C_max)
# Token 仍通过 All-to-All Scatter 随机分发（通信未优化）
tokens = AllToAllScatter(tokens, d)
output = FFN_Experts(tokens, P)  # compute 已均衡化
```
关键特性：(1) zero prediction overhead（offline MLE 估计，运行时仅查表）；(2) 仅均衡 FFN compute，不减少 All-to-All 通信开销；(3) 高 skewness 时 estimation error 增大（因冷门 expert 训练样本不足），但整体性能仍优于无 prediction baseline；(4) 在 skewness≤1.4 时比 Token-to-Expert 最佳配置快 23%。

术语一般如何实现？如何使用？
MLE 假设 expert selection 是 i.i.d. Multinomial draws（因为 expert activation 主要受 local token features 影响）。Error rate 定义为 $|\hat{p} - p| / (1/E)$。实验在 Mixtral 8×7B 上 MMLU（skewness=1.39, error=1.80%）、Alpaca Eval（skewness=1.40, error=0.98%）、SST2（skewness=1.99, error=16.00%）上验证。适用场景：low skewness 或 high-bandwidth interconnect (NVLink) 的推理，此时通信不是瓶颈。

涉及论文标题：
- MoE-GPS: Guidelines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing

---
