## Expert Selection Prediction with Multi-Feature Bayesian Posterior（基于多特征贝叶斯后验的专家选择预测）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
该术语描述一种在 MoE 推理前预测每个 expert 将处理多少 token 的方法，用于 serverless 平台上预先配置 expert function 的内存大小。传统的 expert 预测方法（如 Lina）仅使用 token ID 作为特征，但论文发现具有相同 token ID 的 token 可能被路由到不同 expert（Fig. 3 所示），因此 token ID 不足以唯一确定 token-to-expert 映射。

该方法设计了三个 token 特征：(1) **token ID (f_1)**：tokenizer 分配的 token 标识，推理前已知；(2) **position ID (f_2)**：token 在输入序列中的位置，假设均匀分布；(3) **attention ID (f_3)**：在每层 self-attention 中与该 token 有最高累积 attention score 的 token 的 token ID，反映 token 间的依赖关系。由于 f_3 在推理前未知，用 f_1 的概率近似 f_3 的概率。

后验概率计算通过 Bayes 定理将三个特征融入：
$$P(N_{e,i}|f_1') = \int_{f_2} \int_{f_3} P^*(N_{e,i}|f_1',f_2,f_3) \cdot \frac{P^*(f_1',f_2,f_3)P'(f_3)}{P^*(f_1',f_2)} \cdot \frac{P^*(f_1',f_2)P'(f_2)}{P^*(f_1')} df_3 df_2$$
其中 P*(·) 是从 profiled data 的 key-value table 计算的概率，P'(·) 是真实请求分布的概率。取 argmax 得预测专家：`î_e = argmax_i P(N_{e,i}|f_1')`。

从算法pipeline角度拆解术语：
专家选择预测在 MoE 推理 pipeline 中的位置和计算流程：

```
// 离线Profiling阶段
for each sample in profiling_dataset (≥100 samples):
    for each MoE layer e:
        for each token t:
            extract features: f1=token_id(t), f2=position(t), f3=attention_id(t)
            observe routing: expert_i = gate_network(t)
            key = (f1, f2, f3, e, i)
            Ω[key] += 1  // 记录token-to-expert映射频次

// 在线预测阶段
for each new token with known f1':
    for each MoE layer e:
        for each expert i:
            // 计算联合概率P*(f1',f2,f3) from key-value table
            // f2: uniform distribution
            // f3: approximated by P(f1)
            compute P(N_{e,i}|f1') via double integral (Eq.1)
        predicted_expert[e] = argmax_i P(N_{e,i}|f1')
    return predicted_expert
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：P*(·) 直接从 key-value table 的频率统计获得；P'(f_2) 取均匀分布 1/seq_len；P'(f_3) 用 P*(f_1) 近似。
- 与 Lina 对比：Lina 只用 token ID 单特征 + MAP 估计；本方法加入 position ID 和 attention ID，在多个模型/数据集上预测差异显著优于 Lina。top-2 routing 时预测准确度进一步提升（另一 expert 可修正预测错误）。
- Profiling 开销：~28.89s（100 个 batch）；预测时间：~20.31s（10 个 batch）。
- 适用场景：serverless 平台等需要提前（部署前）知道 expert 负载的场景；也可用于 CPU/GPU cluster 上的资源预分配。

涉及论文标题：
- Optimizing Distributed Deployment of Mixture-of-Experts Model Inference in Serverless Computing
