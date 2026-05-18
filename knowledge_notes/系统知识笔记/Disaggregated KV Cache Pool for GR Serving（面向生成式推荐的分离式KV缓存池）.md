## Disaggregated KV Cache Pool for GR Serving（面向生成式推荐的分离式KV缓存池）

术语是什么？通过联网搜索让回答具体和精准。
Disaggregated KV Cache Pool for GR Serving是Bat系统提出的面向生成式推荐(GR)的compute-storage分离式KV cache管理架构。与现有LLM serving系统（如Mooncake、vLLM）被动管理KV cache（即user query带固定prefix顺序，系统仅做存储和查找）不同，Bat的KV cache pool主动管理user-prefix和item-prefix两个独立的cache组件。系统由三类worker组成：(1) KV Cache Worker管理paged memory pool（CPU/GPU memory），以user/item粒度存储prefix KV cache（物理组织为PagedAttention兼容的fixed-size pages），内置transfer engine支持GPUDirect RDMA；(2) Cache Meta Service逻辑集中式追踪每个user/item KV entry的索引和热度（不含物理数据），处理cache read/write/check请求；(3) Inference Worker（基于vLLM+FlashInfer）专注forward computation。该架构使item cache和user cache独立管理、独立扩展。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Disaggregated KV Cache Pool在Bat中的运行流程：

```
// Offline初始化
1. 预计算所有item的KV cache (offline, item token很少变化)
2. 执行HRCS item cache placement → 分布式加载item cache到各KV cache worker
3. 分配剩余pooled memory给user cache

// Online Serving (per-request)
1. Hotness-aware Prompt Scheduler接收请求 (user_id + candidate_item_ids)
2. Scheduler查询Cache Meta Service:
   - user cache status: hit/miss, hotness (estimated frequency)
   - item cache status: hit/miss (for candidate items in this request)
3. Scheduler决策prefix类型:
   - User-as-prefix: 若user cache hit率高且user token多
   - Item-as-prefix: 若item cache hit率高或user为低频用户
4. Scheduler分发input sequences到inference workers
5. Cache Meta Service并发起physical KV cache transfer:
   - 从KV cache workers → Inference workers (via GPUDirect RDMA)
   - 仅传输所需prefix的KV pages
6. Inference worker执行forward computation:
   - 仅计算non-prefix tokens (suffix)
   - Prefix KV由transfer engine提前送达
7. 若user cache miss (recompute prefix):
   - Background: write new user KV cache back to KV cache workers
   - Cache meta service更新索引和hotness
```

分离式设计的核心优势：(1) Compute和storage独立扩展——inference workers和KV cache workers可分别scale out；(2) Item cache完全独立于user cache管理——item cache离线预计算后分布式部署，user cache在线动态更新；(3) 与vLLM PagedAttention兼容——KV entries按fixed-size pages组织，物理存储和传输对齐已有paging机制。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在Bat中，KV cache worker使用CPU memory作为主要存储层（论文主要评估CPU memory方案），通过GPUDirect RDMA在worker间传输KV cache pages。Cache meta service为逻辑集中式进程，可通过分布式一致性协议扩展到多副本。每个inference worker持有完整model weight replica，以data-parallel方式服务请求。生产环境16节点×H20 GPU可scale到100M item corpus。该架构适用于推荐系统特有的workload特征：item数量可控（百万级）、user数量极大（亿级）、item token相对稳定、user token高度个性化且访问频率呈long-tail分布。

涉及论文标题：
- BAT: Efficient Generative Recommender Serving with Bipartite Attention

---
