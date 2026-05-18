## Hotness-aware Prompt Scheduling（热度感知的Prompt调度）

术语是什么？通过联网搜索让回答具体和精准。
Hotness-aware Prompt Scheduling是Bat系统中用于动态决策每个GR请求使用User-as-prefix还是Item-as-prefix attention的调度策略。其核心问题是在有限cache budget下最小化总计算token数（或最大化cache hit token数）。调度策略基于两个关键观察：(1) user token数呈long-tail分布——活跃用户有大量历史行为token（up to 8K），不活跃用户token很少；(2) 用户连续行为具有时间相似性——通过sliding-window频率估计可近似预测近期访问频率。调度器采用greedy policy：当user token数≥item token数且user的estimated frequency > cache中最冷user page的频率时选User-as-prefix，否则fallback到Item-as-prefix。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Hotness-aware Prompt Scheduling的决策流程：

```
// Sliding-window频率估计 (periodically updated by cache meta service)
// f_u: user u在最近W秒/分钟内的访问频率估计
给定: request r with user u, candidate items I, 
     user token count τ_u(r), item token count τ_i(r)

// Greedy决策规则
if τ_u(r) >= τ_i(r) AND f_u(r) > min_{pages p in C_u} f_p:
    // User-as-prefix: user token更多且user足够"热"
    // 替换cache中最冷user page
    prefix = USER
    evict coldest pages in C_u
    admit new user cache pages for u
else:
    // Item-as-prefix: item token更多或user不够"热"
    // 利用item cache避免compulsory/capacity miss
    prefix = ITEM
    // 不额外消耗user cache空间

// 直觉:
// - τ_u ≥ τ_i: User-as-prefix节省更多computation (条件1)
// - f_u > min_cache_freq: 值得占用cache空间 (条件2)
// - 否则: Item-as-prefix避免低频user的compulsory miss

// 频率一致性验证 (Figure 4)
// Consecutive window similarity = 1 - |f(t) - f(t-δ)| / (f(t) + f(t-δ))
// 多数用户5min/60min窗口间consistency score > 0.8
```

相比cache-agnostic baseline（naively选token数更多的做prefix），hotness-aware策略在user cache空间受限时（如25GB）throughput和cache hit rate显著更高。原因是cache-agnostic会为大量低频长profile用户选择User-as-prefix，导致frequent compulsory miss（user cache未命中）和capacity miss（高频user cache被低频user evict），而hotness-aware将这些请求路由到Item-as-prefix利用item cache。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在Bat中实现在hotness-aware prompt scheduler（centralized component）。调度器周期性（如每batch）查询cache meta service获取相关user/item的cache状态和estimated frequency。频率估计使用asynchronous sliding-window更新（cache meta service在每次cache access时decay更新），避免阻塞在线推理路径。Window大小W可根据业务特征配置（论文评估了5min和60min窗口，实验显示两者都有效）。该策略对推荐场景的独特价值：推荐user访问频率遵循long-tail + temporal locality，与通用LLM chatbot场景的请求模式显著不同，因此需要推荐专用的调度策略。

涉及论文标题：
- BAT: Efficient Generative Recommender Serving with Bipartite Attention

---
