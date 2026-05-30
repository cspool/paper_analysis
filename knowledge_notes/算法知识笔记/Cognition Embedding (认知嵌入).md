## Cognition Embedding (认知嵌入)

术语解释
Cognition Embedding 是 BrainMoE 中各 brain expert 输出的特征表示 Z ∈ R^{C_hid}，代表在特定认知状态下脑活动的 latent representation。不同 expert 产出的 cognition embeddings 在同一 latent space 中表示不同认知视角下的脑活动模式。

术语是什么？
Cognition Embedding 是 brain expert f_i(·) 的前向输出：Z_i = f_i(X)，X 为输入 fMRI 数据（FC 或 BOLD），Z_i ∈ R^{C_hid} 为压缩后的特征向量。关键性质：
1. 认知状态特异性：Rest expert 产出的 Z_rest 编码 resting-state 下的脑网络模式；Emotion expert 产出的 Z_emotion 编码情绪加工时的脑激活模式
2. 架构无关性：与 expert 内部架构（BrainMass/brainJEPA/classifier）和输入类型（FC/BOLD）无关，只要求输出维度统一
3. 正交性：不同认知状态的 embeddings 相关性低（绝对 Pearson correlation < 0.5），表明 expert 间互补而非冗余
4. 下游任务适配性：某些认知状态的 expert 对特定疾病有天然的诊断优势（如 Language expert 对 Alzheimer's、Working Memory expert 对 Parkinson's）

从算法pipeline角度拆解术语。
```
# Cognition Embedding 的生成和使用
# 预训练阶段
for cognitive_state in [resting, emotion, WM, language, ...]:
    data_cog = fMRI_data[cognitive_state == state_label]
    expert_i = train(data_cog, objective=recon/classif)
    # expert_i 输出: Z_i = f_i(X), Z_i ∈ R^{2048}

# 微调阶段
X_downstream = fMRI_subject  # 来自任意下游数据集
Z_all = []                    # 收集所有expert的cognition embeddings
for i in range(N_experts):   # N=12
    Z_i = expert_i(X_downstream)  # frozen expert forward
    Z_all.append(Z_i)             # [12, 2048]

# Router 混合
P = Softmax(Linear(concat(Z_all)))   # [12] expert weights
Z_weighted = Z_all * P               # [12, 2048] weighted cognitions
# 送入 Cognition Adapter 继续处理
```

术语一般如何实现？如何使用？
- 相当于 MoE 中 expert 的"中间产品"而非最终输出——不直接用于分类，而是输入 Router 获得权重后进一步由 Adapter 加工
- 不同于 LLM MoE 中 expert 输出直接加权求和产生 token hidden state——BrainMoE 的 cognition embedding 是 sample-level 表示，需要 cross-attention 与 FC 信息融合后才产生分类结果
- 维度 C_hid=2048，对应 BrainMass 的 bottleneck 维度

涉及论文标题：
- BrainMoE Cognition Joint Embedding via Mixture-of-Expert Towards Robust Brain Foundation Model

---
