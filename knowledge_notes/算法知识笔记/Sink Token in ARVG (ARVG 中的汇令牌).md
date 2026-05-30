## Sink Token in ARVG (ARVG 中的汇令牌)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sink Token（汇令牌）在 ARVG 中的概念借鉴自 LLM 的 KV Cache 研究——在 LLM 中，Attention 的首 token 对模型性能高度敏感，被称为 "attention sink"。ARVG 中的 sink token 特指以条件信息（类别标签 + 位置编码）作为初始 token 的现象，且这种 sink token 的特殊性体现在：(1) 首 token 含关键类别信息，对条件生成起决定性作用；(2) 首 token 对所有后续 token 可见（因果注意力中首 token 参与所有后续 token 的 attention 计算），被训练为高度敏感的 token；(3) 首 token 的激活分布与其他 token 显著不同。PTQ4ARVG 进一步发现，与 LLM 仅在 Attention 中存在 sink token 不同，ARVG 中 sink token 出现在 MHSA 和 FFN 的所有线性层中。成因：(a) ARVG 固有地使用类别条件作为初始 token；(b) 初始 token 被模型训练为承载关键信息的枢纽；(c) 其分布与其他 token 的分布差异显著。

从算法pipeline角度拆解术语，给出具体例子。
Sink token 在 ARVG 量化中的影响（以 RAR-B 为例）：

```
# ARVG 的 token 序列结构 (T 个 token)
Token[0]: cond_embedding  # ← Sink Token (类别条件 + 位置编码)
Token[1]: image_token_1   # 正常图像 token
Token[2]: image_token_2
...
Token[T-1]: image_token_{T-1}

# 因果关系: Token[0] 对所有 Token[1..T-1] 可见
# 量化影响:
#   Token[0] 的激活分布: outlier 严重, range 大 → 需单独 high-precision 量化
#   Token[1..T-1] 的激活分布: 相对均匀 → 可共用一组量化参数
```

STWQ 的处理方式：对线性层输入，sink token (t=0) 和 normal tokens (t≥1) 分别使用独立的静态量化参数（δ_sink, z_sink）和（δ_normal, z_normal），从而在不引入在线开销的前提下处理 sink token 的分布特殊性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Sink token 在 LLM 中被广泛研究（如 StreamingLLM 发现保留 attention sink 对长文本生成至关重要），但在 ARVG 中 PTQ4ARVG 是首次对其进行系统分析并用于量化优化。关键洞察：ARVG 的 sink token 不仅存在于 Attention，还扩散到 FFN 的所有线性层，这使得其影响范围比 LLM 更大。在实现中，识别 sink token 的方法是观察每层激活沿 token 维度的分布：首 token 的激活幅值和方差显著高于其他 token。PTQ4ARVG 的 STWQ 方法通过分离 sink token 和 normal token 的量化参数，在 W6A6 VAR-d16 上将 FID 从 18.54（无 token-wise）降至 10.41（SQ+STWQ）。

涉及论文标题：
- PTQ4ARVG Post-Training Quantization for AutoRegressive Visual Generation Models
