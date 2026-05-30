## FedProx (Federated Proximal Optimization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FedProx 是 Li et al. (2020) 提出的 FL 优化算法，在客户端本地 loss 中添加 proximal term 限制模型偏离全局模型：

$$\min_w h_k(w; w^t) = \mathcal{L}_k(w) + \frac{\mu}{2}\|w - w^t\|^2$$

其中 $\mu$ 控制正则化强度。还引入 $\gamma$-inexactness 允许不同客户端执行不同数量的本地更新。

在 FedMoE 中作为 baseline，全局模型 8 experts/layer（受限于内存 18-24GB），内存 24.71GB，通信 2.30GB。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# FedProx 客户端本地训练
w_k = w_global
for epoch in 1..E:
    for batch in D_k:
        L_CE = cross_entropy(model(w_k, x), y)
        proximal_term = (mu/2) * ||w_k - w_global||^2
        loss = L_CE + proximal_term
        w_k = w_k - lr * gradient(loss)
# 上传 → 服务器 FedAvg 聚合
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Flower 内置 `FedProx` 策略（`proximal_mu` 参数），FedML 和 FATE 也支持。μ 通常 0.001–0.1。FedMoE 实验中 FedProx 在跨任务场景不如 FedMoE——proximal term 只能约束参数距离但无法从根本上解决不同任务需要不同参数的问题。

涉及论文标题：
- FedMoE Personalized Federated Learning via Heterogeneous Mixture of Experts
