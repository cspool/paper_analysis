## Group Batching（分组批处理）

术语是什么？通过联网搜索让回答具体和精准。
Group Batching是MFS论文针对multi-tier model提出的一种批处理调度技术。与传统的continuous batching假设batch内所有请求共享同一完整模型结构不同，Group Batching利用multi-tier模型中不同tier的共享公共前缀层特性，将需要相同前缀层的不同tier请求组成一个group batch，在公共层一起执行；公共层执行完毕后，请求按各自指定的tier分流继续执行剩余层。Group Batching的核心优势是突破了传统selective batching无法跨不同规模模型高效batch的限制，使得在model family serving场景下GPU compute和显存可以跨"模型大小"共享。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Group Batching在MFS serving系统中的运转流程：

```
# Tier-level scheduler在每个scheduling cycle:
for each tier in [tier_1, tier_2, tier_3]:
    # 为当前tier选择待执行请求（plug-and-play策略）
    selected = scheduling_policy(tier_queue[tier])
    working_set[tier] = selected

# 从最低tier开始group batching:
for tier in [tier_1, tier_2, tier_3]:
    # 所有需要当前tier层的请求组成group
    group = []
    for t in [tier, tier+1, ..., tier_max]:  # 当前tier及更高tier的请求都需要这些层
        group.extend(working_set[t])
    
    if group is not empty:
        # 公共前缀层：所有group成员一起执行
        execute_shared_layers(tier_boundary[tier-1]:tier_boundary[tier], group)
        # Attention fusion: 拼接group内多个请求的QKV提升GPU并行度
        # KV cache: 标记公共层KV为multi-tier shareable
    
    # 分离：tier级请求采样输出并返回
    for req in working_set[tier]:
        sample_and_return(req, tier_head[tier])
    
    # 剩余更高tier请求继续下一轮
```

具体例子：假设同时到达R1(tier-1)、R2(tier-2)、R3(tier-3)三个请求。
1. tier-level scheduler检测到三者都需要tier-1公共前缀层（第0-17层）→将它们组成group batch→在一个batch中执行第0-17层的prefill/decode→attention fusion拼接三个请求的QKV矩阵。
2. tier-1执行完毕→R1采样tier-1输出头→返回结果→从working set移除。
3. R2和R3继续→二者都需要tier-2层（第18-31层）→组成group batch执行第18-31层→tier-2完毕→R2采样返回。
4. R3单独执行tier-3层（第32-39层）→采样返回。

相比Orca selective batching（7B请求和13B请求必须分别batch，各自占用GPU资源和waiting queue），Group Batching在model family serving场景下实现了跨模型规模的batch合并，显著提高GPU利用率。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MFS在Orca scheduler基础上实现了Group Batching，替代了原有的selective batching逻辑。核心修改包括：(1) tier-level scheduler维护多tier请求队列；(2) 每个scheduling cycle从低tier向高tier扫描，动态组建group batch；(3) 与multi-tier KV cache manager协同，确保公共前缀层的KV cache被正确标记为shareable。论文未开源MFS代码。Group Batching的性能收益：相比Orca的独立模型部署，MFS在model family batching场景中JCT最多提升31.2%，per-token response latency最多降低56.1%。Group Batching的技术前提是multi-tier模型结构（所有tier共享公共前缀层），因此无法直接应用于独立部署的多个异构模型。

涉及论文标题：
- MFS: An Efficient Model Family Serving System for LLMs
