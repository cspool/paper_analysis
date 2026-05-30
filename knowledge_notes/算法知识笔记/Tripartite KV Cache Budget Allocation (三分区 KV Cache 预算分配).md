## Tripartite KV Cache Budget Allocation (三分区 KV Cache 预算分配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Tripartite KV Cache Budget Allocation 是 ZSMerge 提出的 KV Cache 内存管理策略——将总缓存预算 B 划分为三个功能互补的子预算：$B = B_p + B_c + B_r$（Eq. 3）。

1. **Proximity Component** ($B_p$)：保留最近 B_p 个 token 的 KV 对（sliding window），捕获局部上下文模式和短程依赖。这是所有 KV cache 管理方法的标准组件（StreamingLLM、H2O 均保留最近 token）。
2. **Context Component** ($B_c$)：按贡献分数 s^{(T)} 排序保留 top-B_c 个历史 token——s^{(T)}_t = λ·s^{(T-1)}_t + a^{(T)}_t，λ=0.98 为指数衰减因子（Eq. 5）。从全局历史中选出最具信息量的 KV 对。
3. **Residual Component** ($B_r$)：动态维护 B_r 个残差合并 slot，将被驱逐 token 通过 key 相似度匹配 + 增量均值聚合（Eq. 6-7）压缩编码——这是 ZSMerge 区分于纯驱逐方法的核心创新。

最终压缩 cache 为三部分拼接（Eq. 4）：$\mathbf{K}_B = [\mathbf{K}_p \| \mathbf{K}_c \| \mathbf{K}_r], \quad \mathbf{V}_B = [\mathbf{V}_p \| \mathbf{V}_c \| \mathbf{V}_r]$。

预算分配分两步：
1. Proximity ratio $B_p/B$ 控制局部/全局比（推荐 0.5）
2. Residual ratio $B_r/(B-B_p)$ 控制剩余预算中残差占比（推荐 0.02）
3. 剩余为 Context budget $B_c$

当 $B_r=0$ 时退化为纯驱逐策略（H2O-like）：仅保留 proximity + context，其余永久删除。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# 预算分配（固定配置）
B = total_cache_budget    # 如 512, 1024, 18K
B_p = 0.5 * B             # proximity: 一半用于最近 token
B_r = 0.02 * (B - B_p)    # residual: 剩余预算的 2%
B_c = B - B_p - B_r       # context: 剩余全部用于高分 token

# 每个解码步构建压缩 cache
def build_compressed_cache(K, V, s, T):
    # Proximity: 最近 B_p 个 token
    K_p, V_p = K[-B_p:], V[-B_p:]

    # Context: top-B_c 按贡献分数 s
    candidate_tokens = T - B_p  # 排除 proximity 的 token
    idx_c = topk(s[:candidate_tokens], B_c)
    K_c, V_c = K[idx_c], V[idx_c]

    # Residual: 剩余 token (被驱逐) 合并入 B_r 个 slot
    evicted_mask = all tokens NOT in proximity NOR context
    for (k_t, v_t) in evicted_tokens:
        merge_evicted_token(k_t, v_t)  # → update K_r, V_r

    # 拼接
    K_B = concat([K_p, K_c, K_r])  # [B, d]
    V_B = concat([V_p, V_c, V_r])
    return K_B, V_B
```

消融实验推荐配置：$B_p/B=0.5$, $B_r/(B-B_p)=0.02$, $\alpha=1.0$。极端 budget（B_p/B < 0.3 或 > 0.7）显著损害性能。B_r > 0 持续优于 B_r=0（纯驱逐），验证残差合并的有效性。

术语一般如何实现？如何使用？

在实际部署中，三个 budget 参数为静态配置——推理前设定，生成过程中不改变。ZSMerge 基于 Transformers 库实现，通过 `change_mode` 方法支持运行中切换配置。不同任务/模型可独立调参，但由于 ZSMerge 的零样本/无参数特性，默认配置在多数场景下可直接使用。

涉及论文标题：
- ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs
