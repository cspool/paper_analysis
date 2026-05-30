## Modular Aggregation (in Federated Learning with MoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Modular Aggregation 是 FedMoE 提出的联邦学习模型聚合策略，专门针对 MoE 架构设计，用于替代传统 FL 中的 FedAvg。其核心思想是按"模块粒度"（即每个 expert）独立决定聚合方式，而非对所有参数执行统一的加权平均。

三种聚合模式：(1) Unactivated experts——未被任何客户端使用的 expert，保持不变；(2) Single-client experts——仅被单个客户端使用的 expert，直接替换为该客户端的更新；(3) Multi-client experts——被多个客户端共享的 expert，使用 FedAvg 加权聚合。Router 参数按对应 expert 维度同步更新。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Modular Aggregation (FedMoE)
def ModularAggregation(w_global, client_models, data_sizes):
    # Dense 层: 标准 FedAvg
    for param in dense_params:
        w_global[param] = weighted_avg(client_models[k][param], data_sizes[k])

    # Sparse (expert) 层: 按模块粒度
    for layer_i in 1..L:
        for expert_j in 1..E_i:
            clients_using = [k for k in client_models if expert_j in w_k]
            if |clients_using| == 0:
                continue  # 未激活 → 不变
            elif |clients_using| == 1:
                w_global[layer_i][expert_j] = w_{clients_using[0]}[layer_i][expert_j]
            else:
                w_global[layer_i][expert_j] = weighted_avg(
                    w_k[layer_i][expert_j] for k in clients_using, weights=data_sizes[k])
```

相比 FedAvg，Modular Aggregation 防止不相关客户端相互干扰（负迁移），保留个性化 expert 的专用性，在共享 expert 上实现协作学习，天然支持异构子模型。

涉及论文标题：
- FedMoE Personalized Federated Learning via Heterogeneous Mixture of Experts
