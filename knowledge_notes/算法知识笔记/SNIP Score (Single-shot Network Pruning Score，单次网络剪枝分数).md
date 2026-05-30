## SNIP Score (Single-shot Network Pruning Score，单次网络剪枝分数)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SNIP (Single-shot Network Pruning) Score 是 Lee et al. (ICLR 2019) 提出的一种连接敏感性（connection sensitivity）度量，用于在训练前的初始化网络中一次性识别重要连接。其核心公式：对权重 W_ij 和输入 x，定义重要性分数 I(W_ij, x) = |W_ij · ∇_{W_ij} L(x)|，即权重值与对应损失梯度的乘积的绝对值。直觉：(1) |W_ij| 大 → 该连接当前贡献大；(2) |∇_{W_ij} L| 大 → 该连接的微小变化对损失影响大；(3) 两者乘积同时考虑当前贡献和敏感性，比单独用权重大小或梯度更全面。SNIP 在训练**前**一次性计算，不依赖已训练模型，因此是 "single-shot" 的。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 Q-resafe 中的应用（安全关键权重识别）：
```python
# Q-resafe 将 SNIP score 从初始化前剪枝改编为训练中周期性地识别安全关键权重
# 对量化 LLM 的每层权重矩阵 Q

def compute_safe_score(model_Q, D_calib):
    scores = {}  # layer_name -> importance matrix
    for x in D_calib:
        # 前向传播：计算条件负对数似然
        log_probs = model_Q.log_prob(y | x)
        loss = -log_probs  # L(x) = -log p(y|x)

        # 反向传播：计算梯度
        loss.backward()

        # 对每层量化权重 Q，计算逐元素 SNIP score
        for name, Q in model_Q.named_parameters():
            I = abs(Q * Q.grad)  # |W_ij · ∇_{Q_ij} L(x)|
            scores[name] += I    # 累积跨校准样本

    # 对所有校准样本取平均
    for name in scores:
        scores[name] /= len(D_calib)
    return scores

# 选择 top-τ 百分比的权重
def select_safety_critical(scores, tau):
    M_Q = {}
    for name, score in scores.items():
        threshold = percentile(score.flatten(), (1 - tau) * 100)
        M_Q[name] = (score >= threshold).float()  # 1 = 安全关键
    return M_Q
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
原始 SNIP 实现于初始化阶段：随机采样一个 batch → 前向计算 loss → 反向计算梯度 → 计算 |w ⊙ g| → 全局排序 → 保留 top-k% 连接 → 设置 mask → 训练。在 Q-resafe 中改编为：(1) 在训练过程中周期性（每 K 步）重新计算 SNIP score（因为 Q^t 随训练变化）；(2) 损失函数不是原始分类交叉熵，而是条件语言模型负对数似然；(3) 不是用于剪枝（去除不重要连接），而是反向使用——保留 top-τ% 高 score 权重作为安全关键权重进行更新。这一改编的关键洞察：安全能力集中在少数权重中（类似 LLM 能力集中性假设），可通过 SNIP 敏感性度量定位这些权重。

涉及论文标题：
- Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models
