## Per-Token Quantization (逐Token量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Per-Token Quantization 是将张量沿 token（序列）维度分组量化的策略。对于 KV Cache $X \in \mathbb{R}^{l \times d}$，per-token 量化意味着沿 l 维度分组，每若干 token 共享一组量化参数。这种量化方式与自回归生成的流式特性天然兼容——新生成 token 的量化张量可直接沿 token 维度 append 到已有量化缓存。

KIVI 论文的核心发现：value cache 必须使用 per-token 量化。value cache 无 channel-wise outlier 模式，但由于 attention output $t_O = AX_V = \sum_j A_{ij}[X_V]_{j*}$ 是 value cache 行向量的加权求和（权重为稀疏 attention score），per-token 量化将误差限制在每个 token 内部，使重要 token 不受其他 token 量化影响。per-channel 量化 value cache 会导致 attention output 相对误差比 per-token 高约 15×。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
KIVI 中 per-token value cache 量化：

```
# Prefill Phase: 初始量化
X_V_g = X_V[:l_prompt - R]     # grouped part
X_V_r = X_V[l_prompt - R:]     # residual (FP16)
Q(X_V_g) = GroupQuant(X_V_g, dim=token, numGroup=d // G)
# dim=token: 沿token维度分组, 每G=32个channel共享scale

# Decoding Phase: 流式追加
X_V_r = Concat([X_V_r, t_V], dim=token)   # 新token进入residual
if len(X_V_r) > R:
    outdated = X_V_r[:-R]                  # 超出窗口的旧token
    Q(outdated) = GroupQuant(outdated, dim=token, G=32)  # per-token quant
    Q(X_V_g) = Concat([Q(X_V_g), Q(outdated)], dim=token)
    X_V_r = X_V_r[-R:]                     # 保留最近R个token
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FlexGen 最先使用 4bit per-token group-wise 量化 key 和 value cache，但未区分 key/value 的不同分布。KIVI 在 FlexGen 基础上提出非对称策略：key per-channel、value per-token。标准化实现步骤：(1) 沿 token dim 分 group；(2) 每 group 计算 min/max → scale/zero-point；(3) round-to-nearest 量化。per-token 量化与流式推理天然兼容，是 KV Cache 量化中最常见的量化维度选择。

涉及论文标题：
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

---
