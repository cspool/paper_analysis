## Speculative Sampling（投机采样 / Rejection Sampling 接受机制）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
采样解码下保证投机解码输出分布与目标模型逐 token 数学等价的概率接受机制（Chen et al. arXiv:2302.01318 的投机采样；Leviathan et al. arXiv:2211.17192 的拒绝采样证明）。对第 i 个草稿 token：抽 r_i ~ U(0,1)，若 r_i ≤ p_i(x)/q_i(x) 接受（p=target 概率、q=draft 概率），否则拒绝并在该位置从 max(0, p−q) 归一化残差分布重采样一个 token 后终止本轮；全部接受时从 target 分布补采 bonus token。该构造使整个系统的输出分布与直接对 target 采样不可区分——这是投机解码"无损"的数学基础；贪婪解码可视为其退化特例。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
n ← min({ i−1 | 1 ≤ i ≤ γ, r_i > p_i(x)/q_i(x) } ∪ {γ})      # 论文式 (1)
# r_i ~ U(0,1)；p_i(x)、q_i(x) 为 target/draft 在第 i 个候选处的 logits 概率
# 若 n < γ: 在位置 n+1 从 max(0, p_{n+1} − q_{n+1}) 归一化分布重采样一个 token
# 若 n = γ: 用 p_{γ+1} 采样 bonus token
```
本文用法：Cassandra 在采样模式下用该机制接受草稿 token，保证与 BF16 目标模型输出分布一致（Cassandra-1 完全无损、精度表 III 与 BF16 逐项相同）；实测接受率 Cassandra-1(γ=5) 0.74–0.88、Cassandra-2(γ=3) 0.74–0.91（按模型/benchmark）。接受率高低由 draft 与 target 的分布接近度决定——这正是 Cassandra 用细粒度剪枝+截断而非粗粒度层跳过构造草稿的原因。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
几乎所有投机解码系统默认实现（vLLM/SGLang/TensorRT-LLM/llama.cpp 均内置）；实现要点：draft 与 target 共享同一次验证前向的 logits，对 p_i/q_i 向量化比较，拒绝位置只重采样一次。局限：低接受率时每轮只能推进 1 个 token（比贪婪更糟），故实际系统常混合 greedy 验证。

HybridSpec 补充视角（ISCA'26）：论文以"speculation efficiency = 接受 token 数 / draft budget"（非链式接受率，因树形不满足链式 Markov 假设）扫 draft 精度对延迟的敏感性（Fig.21）：效率上升延迟下降、超过某水平后边际趋缓——draft 精度达到一定程度后其对延迟的影响有限；chain 式方法按 (1-α^B)/((1-α)B) 折算（α=接受率）落点在这些参考线左侧，tree 式在同预算下效率更高、落在右侧区域。接受/拒绝由 target 验证前向的 logits 与 draft 概率比较决定。

从算法pipeline角度拆解：验证前向同时得到 target 对各候选位置的概率 p，与 draft 概率 q 比较（贪婪：取首个不一致前；采样：r≤p/q 接受否则重采样），树形下沿树取最长接受前缀、分支被拒绝部分的 KV 清除。

实现与使用：rejection sampling 是 SD 无损性的基础（vLLM/SGLang 内置）；HybridSpec 用它做 draft 精度敏感性分析（speculation efficiency 扫描）与 KV 回滚边界确定。

涉及论文标题：
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding
