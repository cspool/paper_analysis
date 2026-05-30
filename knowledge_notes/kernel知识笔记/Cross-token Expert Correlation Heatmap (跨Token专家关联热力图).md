## Cross-token Expert Correlation Heatmap (跨Token专家关联热力图)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cross-token Expert Correlation Heatmap 是记录 MoE 模型中相邻 token 之间 expert 选择条件概率的二维矩阵。其数学定义：对同一层 l，heatmap 元素 $H_l[i][j] = P(e_j^{t+1} | e_i^t)$，即在 token t 中选择了 expert i 的条件下，token t+1 中选择 expert j 的概率。矩阵维度为 n×n（n=该层 expert 数量）。Heatmap 通过离线 profiling 构建：对大量请求的每层每 token 记录 expert selection → 统计所有相邻 token pairs → 计算条件概率。论文对 4 个模型的 >24,000 requests 构建了完整的 cross-token 和 cross-layer heatmap（>150 GB JSON traces）。

从kernel调度角度拆解术语：
Heatmap 在 kernel 调度中的核心作用是作为 **Data-Driven Predictor** 的 lookup table。在 MoE kernel launch 时：

```
Predictor 算法（基于 cross-token heatmap）:
Input: 当前 token 的 expert selection E_curr = {e1, e2, ..., ek}
       cross-token heatmap H (预计算并缓存在 Global CP)
Output: 预测的下一 token 热门 experts 列表 E_pred

1. 对 E_curr 中的每个 expert e_id:
   从 H 中定位第 e_id 行 → row = H[e_id][:]
   取 row 中 top-n 个最大概率对应的 expert IDs

2. E_pred = 所有 e_id 的 top-n 结果的并集

3. 为每个 die 生成 cp_en bits:
   for each die d:
       该 die 当前计算涉及哪些 experts → E_die
       E_pred 中与 E_die 相交且不在本地的 experts → 标记为应缓存
```

论文使用此 predictor 指导 hardware-managed HBM 的本地缓存决策。Cross-token heatmap 区别于 cross-layer heatmap：token-level correlation 的 reuse distance 较长（遍历所有层后才生成下一 token），适合映射到大容量存储（DRAM）；layer-level correlation 的 reuse distance 短（相邻层连续执行），适合映射到快速小容量存储（LLC）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为 Global CP DRAM 中的 50 MB full heatmap + 0.5 MB on-chip SRAM cache（一次缓存一层）。对 100 layers × 512 experts 的支持远超当前 SOTA（Kimi K2: 61 layers, 384 experts）。
- Heatmap 构建需要 <2000 GPU hours 的离线 profiling（论文使用 8×H100 DGX + 8×H200 AWS instances 收集 traces）。
- 在 kernel 执行过程中：(1) Global CP 在 kernel launch 时查 heatmap 生成 prediction → (2) 将 cp_en bits (prediction table) 配置到各 die 的 PDU → (3) 在后续 remote data access 时 PDU 自动决定是否缓存 → (4) 已缓存数据通过 ATU 地址翻译从本地读取。
- 开源 traces 和 heatmap：https://huggingface.co/datasets/core12345/MoE_expert_selection_trace

涉及论文标题：
- Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting

---
