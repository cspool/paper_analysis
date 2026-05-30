## Expert Activation Matrix (EAM / 专家激活矩阵)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Activation Matrix (EAM) 是 MoE-Infinity 提出的用于追踪 MoE 模型中 expert 激活情况的核心数据结构。对于一个有 $L$ 层 MoE 层、每层 $E$ 个 expert 的模型，EAM 是一个 $L \times E$ 矩阵，其中 $M[i][j] \in \mathbb{Z}$ 表示被路由到第 $i$ 层第 $j$ 个 expert 的 token 数量。EAM 分为两个粒度：(1) Iteration-level EAM (iEAM)：每次 forward pass 中 per-layer 更新的即时 trace，prefill 阶段 $n$ 等于 sequence length，decode 阶段 $n=1$；(2) Request-level EAM (rEAM)：从请求开始累积所有 iteration 的 iEAM，记录整个请求生命周期中每个 expert 的累计使用次数。rEAM 在 prefill 和 decode 阶段分别追踪。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
EAM 在 MoE-Infinity offloading 系统中的完整生命周期（以 DeepSeek-V2-Lite 64×2.4B 处理一个 prompt 为例）：

```
# EAM 构建与使用流程
# 每个 MoE 层 l 执行后更新 iEAM，请求结束后 EAM 写入 EAMC

初始化:
  iEAM = zeros(L, E)     # iteration-level，每次 forward 清零
  rEAM = zeros(L, E)     # request-level，整个请求累积

Prefill Phase (prompt 128 tokens):
  for token t in 1..128:
    for layer l in 1..L:
      Router(t) → activated experts [e1, e2, ..., ek]
      for e in [e1, ..., ek]:
        iEAM[l][e] += 1         # 当前 iteration 计数
        rEAM[l][e] += 1         # 累积到 request-level
    # Prefill 完成：所有 128 tokens 流经各层
    # iEAM 每行之和 = 128（每层处理全部 128 tokens）
    # rEAM 每行之和 = 128

Decode Phase (逐 token 迭代):
  for iteration in 1..output_len:
    iEAM = zeros(L, E)          # 每 iteration 重置
    for layer l in 1..L:
      Router(token) → activated experts [e_i1, ..., e_ik]
      for e in [e_i1, ..., e_ik]:
        iEAM[l][e] += 1
        rEAM[l][e] += 1
    # 当前 iEAM 用于实时 PredictEAM(iEAM, EAMC) → pEAM
    # pEAM 指导下一层的 prefetch 和 cache eviction

请求结束:
  rEAM 归一化后写入 EAMC（用于未来请求的激活模式匹配）
  EAMC 容量满时：新 rEAM 替换与之余弦距离最小的已有 rEAM
```

EAM 的关键约束：$\sum_j M[i][j] = n \ \forall i$（每层处理的 token 总数相等），$M[i][j] \in \{0,\ldots,n\}$。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **实现**：在 MoE-Infinity 中用 PyTorch tensor 存储 $L \times E$ 矩阵，CPU 侧维护和更新（匹配 cost 21μs/query @1K EAMs，226μs @10K EAMs，小于推理延迟的 1%）。iEAM 在每次 forward 中 per-layer 更新；rEAM 在 GPU 执行 MoE forward 后从 iEAM 累积。EAM 之间的匹配使用展平向量的余弦距离（cosine distance），因为：(i) 序列长度不同导致绝对计数不可比，余弦距离归一化后关注相对频率；(ii) 稀疏向量在高频 expert 上匹配收益更大。
- **使用**：EAM 的核心用途是作为 PredictEAM 的输入——当前 iEAM 与 EAMC 中历史 rEAM 做余弦匹配 → 找到最相似的 rEAM → 聚合归一化 → pEAM → 指导 cache eviction priority 和 prefetch target。
- **局限**：FineMoE 指出 request-level rEAM 的粗粒度聚合会冲淡 iteration-level 的清晰模式（entropy 分析显示 coarse-grained patterns 比 fine-grained 更难预测），因此 FineMoE 改用 per-iteration per-layer 的 Expert Map Store 替代 rEAM。

涉及论文标题：
- MoE-Infinity Activation-Aware Expert Offloading for Efficient MoE Serving

---
