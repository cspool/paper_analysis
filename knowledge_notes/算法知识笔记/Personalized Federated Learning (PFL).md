## Personalized Federated Learning (PFL)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Personalized Federated Learning (PFL) 是联邦学习的一个子领域，旨在为每个参与客户端训练定制化的个性化模型，而非为所有客户端训练单一的全局模型。传统 FL（如 FedAvg）假设一个全局模型可以服务所有客户端，但在实际场景中，不同客户端的数据分布（non-IID）、任务类型（跨任务）和资源能力（异构硬件）差异巨大，单一模型难以同时满足所有客户端的需求。

PFL 的核心思路是允许不同客户端拥有不同的模型参数或结构。实现策略包括：(1) 正则化方法——在本地训练 loss 中加入 proximal term（如 FedProx）或修正梯度方向（如 SCAFFOLD）；(2) 模型拆分——将模型分为共享层和个性化层（如 FedPer）；(3) 知识蒸馏——用全局模型蒸馏个性化小模型；(4) 模型剪枝——为不同客户端剪裁不同子网络；(5) MoE 方法——利用 expert 并行结构为不同客户端选择不同的 expert 子集（如 FedMoE）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 FedMoE 中，PFL 通过 MoE 架构的 expert 级个性化实现——不是为每个客户端训练独立模型，而是从共享的全局 MoE expert 池中为每个客户端 sub-sample 最优 expert 子集：

```
# PFL via MoE Expert Sub-Sampling (FedMoE)
# 全局模型: 32 experts/layer
# 客户端 k 的个性化子模型: 平均 65 experts 从 32 experts 中选出

# 个性化子模型构建
for client k in selected_clients:
    for layer_i in 1..L:
        # 从 client-expert map 中提取该客户端该层的 expert 子集
        kept = expert_map[k][layer_i]
        W_k.experts[layer_i] = {W_global.experts[layer_i][j] for j in kept}
        W_k.router[layer_i] = W_global.router[layer_i][kept]  # 仅保留相关维度
    W_k.dense = W_global.dense  # dense 层全员共享

    # 本地个性化训练 (仅训练子模型参数)
    W_k* = TRAIN(W_k, D_k)  # D_k 为客户端 k 的私有数据

# 知识聚合: Modular Aggregation 将个性化知识吸收回全局 expert 池
for expert_j in all_experts:
    clients_using = {k: expert_j in W_k.experts}
    if |clients_using| == 0:  W_global[j] unchanged
    elif |clients_using| == 1:  W_global[j] = W_k[j]  # 直接更新
    else:  W_global[j] = weighted_avg(W_k[j] for k in clients_using)
```

MoE-PFL 的独特优势：个性化体现在 expert 选择层面而非整个模型空间——不同客户端共享同一个 expert 池但训练不同子集，既能充分个性化（不同 expert 处理不同数据/任务），又能在重叠部分共享知识（使用相同 expert 的客户端通过聚合相互学习）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

PFL 的实现框架包括：
- **Flower (flwr)**：提供 FedProx、FedAvgM 等策略的内置实现
- **FedML**：支持 PFL 算法库
- **FATE**：企业级 FL 框架，支持多种 PFL 策略
- **自定义实现**：FedMoE 基于 PyTorch + HuggingFace Transformers 自建 FL 模拟框架

FedMoE 的方法适合：(1) 底层使用 MoE 架构模型（如 Switch Transformers），(2) 客户端数据/任务异构性强，(3) 客户端资源有限需要个性化轻量模型，(4) 需要兼顾协作学习和个性化。

涉及论文标题：
- FedMoE Personalized Federated Learning via Heterogeneous Mixture of Experts
