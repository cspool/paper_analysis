## Token-to-Expert Prediction (Token级专家预测)

术语是什么？
Token-to-Expert Prediction 是传统 MoE expert 预测策略：直接预测每个 token 将被路由到哪个 expert，即 exact token-to-expert mapping。预测结果用于跳过 All-to-All 的 Scatter 阶段（Direct Routing），同时均衡 FFN compute 和通信。代价是 predictor 本身的 inference overhead。MoE-GPS 将 expert selection 建模为多分类问题，探索三类 predictor：(a) Probability Model——始终选全局频率最高的 expert；(b) Conditional Probability Model——按 token index 或 position index 条件化选择；(c) Neural Networks——FFN（2 层 MLP: 4096→128→64→8 logits）和 LSTM with Sparse Attention（2-layer LSTM, hidden 64 + sparse attention + residual connection）。

从算法pipeline角度拆解术语：
```
# 训练 Predictor（以 FFN 为例）
# 输入: token_embeddings ∈ R^{seq_len × 4096}（Mixtral）
# 输出: expert_logits ∈ R^{seq_len × 8}（每层独立 head）
class FFNPredictor:
    def forward(x):
        h = ReLU(Linear(x, 4096→128))
        h = ReLU(Linear(h, 128→64))
        return Linear(h, 64→8)  # 每层一个 head

# 推理阶段：Predictor 插入每层 Attention 之前
for layer l in 1..L:
    predicted_experts[l] = Predictor[l](hidden_states)  # overhead
    P, d = ExpertDuplication(predicted_experts[l], ...)
    # Direct Route: 跳过 Scatter，token 直接发送到目标 GPU
    tokens = DirectRoute(tokens, predicted_experts[l])
    output = FFN_Experts(tokens, P)
```
Accuracy-overhead trade-off 呈 U 形：更高 accuracy → 更好的 load balancing → 更好系统性能，但也 → 更大 predictor overhead → 降低 net 收益。最优配置通常在 intermediate accuracy 处。高 skewness 时 prediction 更容易（accuracy 更高、overhead 更低），sweet spot 向高 accuracy 方向移动。

术语一般如何实现？如何使用？
Prediction error 建模三种场景：Optimistic（errors 不影响 load balancing）、Typical（errors 均匀分布在各 GPU，最负载 GPU 处理 (1+ε)×avg_tokens）、Pessimistic（errors 集中在单 GPU，最坏 N×(1+ε)×avg_tokens）。默认使用 Typical。适用场景：high skewness（prediction 更容易）+ low-bandwidth interconnect（PCIe）时，节省的 communication 超过 predictor overhead。

涉及论文标题：
- MoE-GPS: Guidelines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing

---
