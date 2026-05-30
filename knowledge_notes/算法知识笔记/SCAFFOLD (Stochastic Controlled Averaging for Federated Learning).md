## SCAFFOLD (Stochastic Controlled Averaging for Federated Learning)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SCAFFOLD（Karimireddy et al., ICML 2020）使用控制变量修正本地梯度方向以解决 client-drift。维护全局控制变量 c（全局梯度估计）和本地控制变量 c_k（客户端梯度估计），本地更新规则：$w_k \leftarrow w_k - \eta \cdot (\nabla \mathcal{L}_k(w_k) - c_k + c)$。

在 FedMoE 中作为 baseline，因控制变量传输额外开销，通信量最大（4.61GB）。Enforced-Hetero-T 下 TC accuracy 仅 36.17（FedMoE 94.85），验证了 task-level heterogeneous 场景下仅修正梯度方向不足以弥合不同任务的根本差异。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# SCAFFOLD 每 round
c_k_new = c_k - c + (w_global - w_k) / (T * lr)  # Option II 更新控制变量
# 上传 (w_k, Δc_k)  →  额外通信开销约为模型大小 2 倍
w_global = weighted_avg(w_k)
c = c + (|S|/N) * avg(c_k_new - c_k)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

SCAFFOLD 理论收敛速度最优但实际挑战：(1) 控制变量通信量翻倍，(2) 控制变量存储需与模型相同内存，(3) task-level heterogeneous 场景表现差——不同任务的最优梯度方向本身不一致，仅靠修正无法弥合。

涉及论文标题：
- FedMoE Personalized Federated Learning via Heterogeneous Mixture of Experts
