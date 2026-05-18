## Expert Cache（专家缓存）

术语是什么？
Expert Cache是GPU显存中用于存放预取或最近使用的expert权重的高速缓存区域。FineMoE的Expert Cache包含两个核心策略：(1) Prefetch priority = p/(l-l_now)：概率p越高、距离当前layer (l-l_now)越近的expert优先预取；(2) Eviction priority = 1/(p×freq)：低概率、低访问频率的expert优先驱逐。与LRU（时间局部性）和LFU（频率）不同，FineMoE直接量化"未来哪些expert更可能且更快被用到"。

从系统架构角度拆解术语：
```
Expert Cache状态机：
- Prefetch: CPU memory → GPU cache（异步，按p/(l-l_now)排序）
- Hit: gate选出expert已在GPU cache → 直接计算
- Miss: gate选出expert不在GPU cache → 暂停普通prefetch → on-demand CPU→GPU loading
- Evict: GPU cache超限 → 按1/(p×freq)驱逐最低priority expert → CPU memory
- Offload: 预测未来长时间不用的expert主动迁回CPU
```

术语一般如何实现？
FineMoE的Expert Cache在MoE-Infinity codebase上用C++/CUDA Runtime APIs实现。GPU显存allocate固定大小cache区域（实验中从4GB到12GB），CPU memory存放完整expert权重池。多GPU配置下expert parallelism通过hash map和round-robin将expert分布到不同GPU，每个GPU各自管理expert cache。与baseline的LRU cache对比：LRU假设最近使用的最可能再用，但MoE layer-wise顺序执行中，已用过的expert在同一iteration不再被使用，LRU与访问模式不匹配。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading
