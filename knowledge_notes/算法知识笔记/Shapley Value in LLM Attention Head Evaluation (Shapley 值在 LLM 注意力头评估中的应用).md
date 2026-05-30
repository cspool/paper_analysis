## Shapley Value in LLM Attention Head Evaluation (Shapley 值在 LLM 注意力头评估中的应用)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Shapley Value (Shapley, 1953) 是合作博弈论中的经典公平分配解。在 LLM 语境下，CoKV 将每个 attention head 视为博弈中的"玩家"，定义效用函数 U(S) 为 coalition S 中 head 未被 mask、其余 head 被 mask 时模型在验证集上的准确率。Shapley Value SV_i 衡量 head h_i 在所有可能 coalition 中的期望 marginal contribution：

$$SV_i = \frac{1}{n} \sum_{S \subset N \setminus \{h_i\}} \frac{U(S \cup \{h_i\}) - U(S)}{\binom{n-1}{|S|}}$$

直接计算 Shapley Value 需要评估指数级数量的 coalition（#P-hard），在 LLM 中完全不可行——Llama-3-8B-Instruct 有 256 个 KV groups（via GQA），枚举所有 coalition 需要 2^256 次模型推理。

CoKV 利用 complementary contribution 形式重写 Shapley Value：

$$SV_i = \frac{1}{n} \sum_{S \subset N \setminus \{h_i\}} \frac{U(S) - U(N \setminus S)}{\binom{n-1}{|S|}}$$

这使一次采样可同时更新 coalition S 中所有 head 的估计值，而非仅更新一个 head（如传统 marginal contribution 形式）。CoKV 进一步提出 Sliced Shapley Value (SSV)，基于"不同 coalition size 下分布高度相关"的实证观察，仅计算少数 coalition size 的贡献。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Shapley Value 在 LLM 推理中的语义**：

- U(S) = accuracy when heads in S are active (retain all KV), heads in N\S are masked (retain only local window KV)
- SV_i > 0: head h_i 在合作中对模型性能有正贡献，其 KV 应被保留更多
- SV_i ≈ 0: head h_i 对合作贡献微小，可分配较少 cache
- SV_i < 0: head h_i 对合作有负贡献（removing it improves performance），可不分配额外 cache

CoKV 将效用函数 U 定义为模型在特定 task 验证集上的准确率，这意味着 Shapley Value 是 task-specific 的——同一 head 在不同 task 上的重要性不同。这与 HeadKV 等 baseline 的 task-agnostic 评估形成鲜明对比。

术语一般如何实现？如何使用？

Shapley Value 在 LLM 中的直接计算不可行，需通过采样近似。CoKV 的 SSV 是当前该场景下的 SOTA 近似方法。CoKV 推荐在 8 卡 GPU 上并行计算，250 samples/coalition size 时精度满足要求（MAE<1/n）。SSV 预计算后存储为 per-task 的分数表，推理时查表加载即可，不增加推理延迟。代码开源：https://github.com/nawei1010/CoKV。

涉及论文标题：
- CoKV: Optimizing KV Cache Allocation via Cooperative Game

---
