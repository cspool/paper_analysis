## Non-IID Data Heterogeneity in Federated Learning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Non-IID 数据是 FL 的核心挑战，指不同客户端数据不满足独立同分布。类型包括 Label skew（标签分布偏斜）、Feature skew（特征分布偏斜）、Quantity skew（数据量偏斜）、Task-level heterogeneity（任务级异构——FedMoE 重点场景）。

Non-IID 导致 client-drift：各客户端本地训练的梯度方向不一致，聚合后的模型在不同目标间摇摆。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FedMoE 设置 4 种 FL 模拟：(1) Standard-Hetero-T——30 客户端各异构任务，(2) Standard-Hetero-TD——额外 label-skewed non-IID，(3) Enforced-Hetero-T——强制选不同任务客户端制造冲突，(4) Enforced-Hetero-TD——Enforced + label-skewed。FedMoE 通过 expert 级个性化使不相关任务的梯度互不干扰：client 1（分类）的 expert 3 不受 client 2（阅读理解）的梯度影响。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

处理 non-IID 的主流方法：FedProx（proximal term）、SCAFFOLD（control variates）、FedMoE（expert 级个性化解耦参数空间）。FedMoE 在 Enforced 设置下优势更显著，验证了 MoE 个性化对强异构场景的有效性。

涉及论文标题：
- FedMoE Personalized Federated Learning via Heterogeneous Mixture of Experts
