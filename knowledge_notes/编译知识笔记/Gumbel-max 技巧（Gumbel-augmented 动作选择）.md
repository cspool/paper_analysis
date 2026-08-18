## Gumbel-max 技巧（Gumbel-augmented 动作选择）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gumbel-max trick 是把概率分布转化为 argmax 采样的经典技巧：给每个选项的 logit 加上 Gumbel 噪声（Gumbel(0,1) 分布），argmax 结果即按原分布的一次采样。QiMeng-Tensify（ISCA'26）把它用于 MCTS Selection 阶段的动作选择：a* = argmax_a[g(s,a) + π(s,a) + σ·Q(s,a)]，其中 g(s,a) 为 Gumbel 噪声项、π(s,a) 为 LLM prior logit、Q(s,a) 为经验动作值、σ 平衡先验与经验。Gumbel 噪声替代传统 UCB 公式的探索项：在"LLM 先验可能不准"时保持有原则的探索多样性（guard against LLM inaccuracies），又不像纯随机探索那样丢弃语义先验——即"用 Gumbel 扰动把 LLM 语义先验和启发式探索融合"。

从编译框架角度拆解术语，比如术语所在编译框架的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 MCTS Selection 中的使用（论文 Eq.2）：
```
# 对节点 s 的每个候选动作 a，算 Gumbel-augmented 分数并取 max
for a in Actions(s):
    g = -log(-log(U(0,1)))          # 采样 Gumbel(0,1) 噪声
    score[a] = g + LLM_prior(s,a) + sigma * Q(s,a)
a* = argmax_a score[a]              # 选择下钻动作
```
例子（GatedMLP）：GEMM 后紧跟 elementwise 的节点上，LLM 给 ComputeAtLocation 高 prior（π 大），Gumbel 噪声让搜索偶尔也尝试其他规则（如先 AutoInline 再融合），Q 值把历史最优路径的经验拉进来——三者的 σ 加权决定最终下钻方向。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Gumbel 噪声由 -log(-log(U)) 变换标准均匀随机数得到；在每节点选择时对全部候选动作一次性加噪声再 argmax（等价 Gumbel-max 采样，效率高）。使用方式：作为 Selection 的探索机制，替代 UCB1（论文引用 Tolpin & Shimony 的 simple regret MCTS）；与 LLM prior 叠加使"语义引导 + 原则性探索"同时成立，是 LLM-guided MCTS 区别于"纯 LLM 生成（顺序敏感、无反馈，Fig.1(d) 失败）"与"裸 MCTS（无语义）"的关键。消融（Fig.8）：LLM prior 配置比随机 prior 的裸 MCTS 明显更优且收敛更快。

涉及论文标题：
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
