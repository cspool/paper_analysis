## Adaptive Draft Sequence Length（自适应起草序列长度）

术语是什么？通过联网搜索让回答具体和精准。
Adaptive Draft Sequence Length是SADDLE提出的一种运行时动态调整speculative decoding中每个请求draft token数量的机制。与传统的固定draft length（如d=8，所有请求统一）不同，SADDLE为每个请求独立动态决定draft长度：Controller在DLM每生成一个draft token时读取该token的采样概率p_t，维护累计接受概率H_t = ∏_{i=1}^{t} p_i，当H_t低于预设阈值τ时停止该请求继续drafting。阈值τ通过离线验证集校准——对每个请求运行完整prediction-verification pipeline，记录每draft step的H_j和验证结果，估算条件成功率曲线，选择20%区间内平均draft length最高且≥90%验证成功率的τ。运行时τ可动态调节：轻负载时降低τ允许更长drafts提升并行度。该方法不需要额外模型训练或分类器，仅基于DLM自身输出概率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
Adaptive Draft Length的pipeline（per request per speculative iteration）：
```
Input: prefix X, DLM M_d, threshold τ
Output: draft tokens {x_1, ..., x_k} where k is adaptive

// 初始化
H_0 = 1.0
draft_tokens = []
context = X

// 逐token Draft Generation with early stopping
for t = 1, 2, ...:
    // DLM forward → next token distribution
    P_t = M_d(context)  // probabilities over vocabulary
    
    // Sample token and get its probability
    x_t ~ P_t
    p_t = P_t[x_t]  // probability of sampled token
    
    // Update cumulative acceptance probability
    H_t = H_{t-1} * p_t
    
    // Adaptive stopping check
    if H_t < τ:
        break  // stop drafting for this request
    
    draft_tokens.append(x_t)
    context = context + [x_t]

// Return variable-length draft sequence
return draft_tokens  // length k ≤ max_draft_length
```
关键特征：(1) 每步仅需一次乘法（H_t更新）和一次比较（H_t < τ），开销极低（SADDLE中仅占0.83% end-to-end latency）；(2) 简单请求（高p_t token，如常见continuation）H_t下降慢→自动获得更长drafts；复杂请求（低p_t token，如creative writing）H_t下降快→更早停止，避免生成可能被TLM拒绝的token；(3) 决策依赖DLM自身概率，无需TLM反馈，可完全在prediction stage内完成。

术语一般如何实现？如何使用？
SADDLE在Manager的Controller中实现：Controller集成softmax unit（计算P_t）、multipliers（更新H_t）和comparators（比较H_t与τ），以专用硬件实现低延迟决策。离线校准：用验证集（如Dolly的subset）对每个model pair（如OPT-66B+OPT-1.3B）扫描τ值→选满足90%+验证成功率的最大draft length对应的τ。SADDLE的实验显示仅用自适应draft length（SADDLE-d）不加pipeline优化时吞吐反比PIM-SD低1.22×，说明自适应drafting需要配合Shared Pool和异步pipeline才能发挥收益。该技术与Disco（classifier-based stop decision）、OPT-Tree（greedy draft tree construction）等自适应draft方法形成对比——SADDLE的方法更轻量（无额外模型/树搜索），但threshold tuning依赖workload分布。

涉及论文标题：
- Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems

