## Hot-replicated Cold-sharded (HRCS) Item Cache Placement（热复制冷分片的Item缓存放置策略）

术语是什么？通过联网搜索让回答具体和精准。
HRCS是Bat系统中面向item KV cache的分布式放置策略。其核心思想是利用item访问频率的高度skewed分布（约90%访问集中在top 10%热门item），将热门item的KV cache在所有KV cache worker间完整复制（hot-replicated），而长尾cold item均匀分片（cold-sharded），从而在最小化跨节点通信开销的同时保持cache效率。HRCS通过三步算法确定replication ratio r：(1) offline profiling网络带宽B和prefill computation time t → 计算max allowed communication ratio R_max；(2) 按item access frequency CDF扫描确定满足1-R_max覆盖的top item比例r；(3) top-r% item replicate到所有worker，剩余均匀分片。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
HRCS在Bat中的完整工作流程（Algorithm 1 in paper）：

```
Input: 
  B: 实测网络带宽 (tokens/sec, offline profiling)
  F: item访问频率分布 (past period, e.g., 过去一天)
  τ_u, τ_i: 平均user/item token数
  α: 可容忍通信时间占computation的比例阈值
  c: 每请求candidate item数
  N: KV cache worker数量

Step 1: 计算最大允许通信比 R_max
  t = PrefillTime(τ_u, c×τ_i)  // polynomial regression, offline拟合
  T_max = α · t                // 最大允许通信时间
  S_item = τ_i                 // 平均item size (tokens)
  R_max = T_max · B · (N-1) / (c · S_item · N)  // 最大允许通信ratio

Step 2: 计算replication ratio r
  Sort F by descending frequency
  CDF = 0
  for i = 1 to |F|:
      CDF += F[i]
      if CDF >= 1 - R_max:
          r = i / |F|
          break

Step 3: 放置KV cache
  // Hot-replicated: top r% items (覆盖1-R_max的访问频率)
  for item in top_r_percent:
      replicate KV cache on ALL workers
  
  // Cold-sharded: 剩余(1-r)% items
  for item in remaining_items:
      partition uniformly across workers
      store on designated worker only
  
  // 动态更新: online替换
  for burst hotspot items (新热item):
      background compute KV cache → replace in replicated area
```

核心tradeoff：完全复制（Bat-Replicate）最大化cache hit rate但消耗最多内存；完全分片（Bat-Hash）最大化内存效率但引入大量网络通信；HRCS通过R_max在两者间取得最优平衡。实验中，100Gbps网络下Bat throughput比Bat-Replicate高16%；10Gbps下Bat-Hash throughput仅为Bat-Replicate的78%（受网络开销拖累）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
HRCS实现在Bat的cache meta service中。Replication ratio r通过offline profiling一次性确定（每小时或每天根据item frequency变化周期性更新）。Item KV cache在离线阶段预计算（因为item的token描述很少变化），新item上线时通过background compute+update策略加入。HRCS的关键设计决策：(1) 以token为单位的通信带宽建模（而非bytes），直接对齐KV cache传输的单位；(2) prefill time通过polynomial regression预测，适配不同模型和输入长度；(3) α为可调hyperparameter，控制通信开销容忍度。适用于item数量在百万级、访问分布高度skewed（典型推荐场景）的环境。

涉及论文标题：
- BAT: Efficient Generative Recommender Serving with Bipartite Attention

---
