## Multi-Domain Multi-Task (MDMT) Recommendation

术语是什么？
Multi-Domain Multi-Task (MDMT) Recommendation 是指在推荐系统中同时处理多个业务域（如不同 tab、不同终端、不同场景）和多个优化目标（如点击率、点赞率、收藏率、长播率）的问题。与单一的多域推荐（MDR）或多任务推荐（MTR）不同，MDMT 引入了一个更高维度的交叉关系：域-任务交互（domain-task interplay）。即同一域信息传递策略在不同任务上效果不同，同一任务平衡策略在不同域上效果也不同。这是一个尚未被充分研究的实际推荐场景，相比 MDR 或 MTR 更具挑战性。

从算法pipeline角度拆解术语：
MDMT 问题的形式化定义：
- 设用户集 U，物品集 I，D 个域，T 个任务
- 每个样本 (x_d, y_{d,1}, ..., y_{d,T}) 属于某个域 d
- 目标：学习 T × D 个预测函数 f^{d,t}(x_d) → ŷ_{d,t}
- Loss: L = Σ_{d=1}^{D} Σ_{t=1}^{T} BCE(ŷ_{d,t}, y_{d,t})

挑战在于：不同域和任务的最优信息共享和融合策略各不相同。M3oE 通过 α_d/α_t（控制模块间贡献）和 β_d/β_t（控制专家间贡献）两级融合权重实现自适应。PEPNet（快手 KDD 2023）则通过 EPNet 对齐域间 embedding 语义 + PPNet 个性化 tower 参数来解决此问题。

术语一般如何实现？如何使用？
MDMT 推荐通常有两种实现路径：(1) 联合训练（如 M3oE）：一个模型同时处理所有域和任务，通过解耦模块和自适应融合实现信息共享与隔离的平衡；(2) 分离+转移（如 M2M）：用 meta-learning 或迁移学习方法将知识从一个域/任务转移到另一个。实际部署中，工业界（如快手、字节跳动）的推荐系统通常面临 3-5 个域和 3-8 个任务的同时优化需求。

涉及论文标题：
- M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework

---
