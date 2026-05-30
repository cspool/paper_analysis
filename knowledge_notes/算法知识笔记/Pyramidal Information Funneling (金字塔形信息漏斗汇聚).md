## Pyramidal Information Funneling (金字塔形信息漏斗汇聚)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Pyramidal Information Funneling 是 PyramidKV（Cai et al., 2024）通过系统性分析 LLM 中跨层注意力模式发现的一种信息聚合现象。该现象描述了 LLM 处理长上下文输入时，注意力机制的信息流呈现"金字塔形漏斗汇聚"模式：

- **底层（Lower Layers, e.g., layer 0-5）**：注意力近似均匀分布（broad-spectrum mode），模型从全局所有可用内容中聚合信息，不优先关注特定输入片段。注意力分数覆盖几乎全部 token。
- **中层（Middle Layers, e.g., layer 6-18）**：注意力逐步收窄到局部区域（localized attention），每个文档/段落内部 token 之间的注意力显著增强（可视化为沿对角线的红色三角形状），信息在各个上下文内部被精细化聚合。
- **顶层（Upper Layers, e.g., layer 24-30）**：出现 "Massive Attention" 现象——绝大多数注意力集中在极少量关键 token 上（concentrated attention bars），这些 token 承载了聚合后的核心信息，用于最终答案生成。

这种从"全局广播→局部聚拢→关键 token 集中"的递进模式在 LLaMA、Mistral、Mixtral 等多个模型家族中均被验证（Appendix D），表明其跨模型架构的普适性。该发现超越了过去孤立记录的 "Massive Activation"（Sun et al., 2024，仅关注个别层的大激活值）和 "Attention Sink"（Xiao et al., 2023，仅关注首 token 的注意力异常），提供了跨层信息流动态的全景视角。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

Pyramidal Information Funneling 的核心机制通过注意力分数的跨层演变来量化：

**注意力分布分析流程**：
```
Input: Long-context sequence of n tokens, LLM with L layers, H heads per layer
Output: Per-layer attention pattern (average over heads)

for l in 0..L-1:
    // 获取第 l 层所有 head 的注意力矩阵
    A_l = []  // [H, n, n]
    for h in 0..H-1:
        Q_lh = W_Q[l,h] @ x_l    // query projection
        K_lh = W_K[l,h] @ x_l    // key projection
        A_lh = softmax(Q_lh @ K_lh.T / sqrt(d_k))  // [n, n], causal masked
        A_l.append(A_lh)
    
    // 跨 head 平均得到层注意力模式
    A_avg_l = mean(A_l, dim=0)   // [n, n]
    
    // 分析注意力分散度
    // 底层 (l=0):   A_avg 近似均匀分布，entropy 高
    // 中层 (l=12):  A_avg 沿对角线集中，块状结构明显
    // 顶层 (l=30):  A_avg 在少数列上有极高值（massive attention columns）

// 观察结论：
// entropy(A_avg_0) ≈ log(n)         → 信息分散（广播模式）
// entropy(A_avg_12) ≈ log(n/4)      → 信息局部化（聚类模式）
// entropy(A_avg_30) << log(n)       → 信息集中在极少数 token（massive attention）
```

**金字塔形信息流的量化指标**：

对于第 l 层，定义注意力集中度：
```
// 每列（key token）的平均注意力
col_attn_l[j] = mean(A_avg_l[:, j])   // token j 收到的平均注意力

// 注意力集中度 = top-k columns 占有的注意力比例
concentration_l(k) = sum(top_k(col_attn_l, k)) / sum(col_attn_l)

// 金字塔性质：
// concentration_0(10) ≈ 10/n       (底层——均匀)
// concentration_15(10) ≈ 0.3       (中层——部分集中)
// concentration_30(10) ≈ 0.8       (顶层——高度集中，massive attention)
```

术语一般如何实现？如何使用？

Pyramidal Information Funneling 作为观察到的现象（非算法），其价值在于指导算法设计：

1. **KV Cache 压缩设计**（PyramidKV 的核心应用）：
   - 底层注意力分散 → 需要更多 KV cache budget 覆盖全局信息
   - 顶层注意力集中 → 仅需少量 KV cache budget 保留关键 token
   - 实现为算术序列递减的 budget 分配：k^l = k^0 - (k^0 - k^{m-1})/(m-1) × l

2. **Token Selection 策略**：
   - 由于顶层 massive attention 集中在特定位置（不限于首 token），token 选择应基于实际注意力分数（attention score-based selection），而非仅依赖位置启发式（如仅保留首尾 token）

3. **验证方法**：
   - 在多文档 QA 任务上可视化每层平均注意力矩阵（Figure 2）
   - 通过 LongBench 17 个数据集验证基于该现象的 KV cache 压缩策略有效性
   - 通过 Needle-in-a-Haystack 验证长上下文信息检索能力保持

4. **跨模型泛化**（Appendix D）：
   - LLaMA 系列（dense）：所有层显示清晰的金字塔形信息汇聚
   - Mistral-7B（dense）：同样显示该模式，注意力收窄略有提前
   - Mixtral-8x7B（MoE）：尽管有专家路由，仍显示一致的注意力收窄趋势

涉及论文标题：
- PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference
