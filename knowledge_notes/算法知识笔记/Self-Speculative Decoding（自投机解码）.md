## Self-Speculative Decoding（自投机解码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
训练无关的投机解码变体：草稿模型直接由目标模型自身派生，无需额外训练、不引入独立草稿权重。派生手段三类：(1) 层跳过——Draft&Verify（贝叶斯优化选跳层）、Swift；(2) KV cache 简化——MagicDec（KV 稀疏检索）、QuantSpec（KV 低精度量化）；(3) 参数位级子集/压缩——Cassandra。Cassandra 的构造：权重经 Wanda 非结构化剪枝、KV 经 per-token 幅度剪枝、再叠加 4-bit 尾数截断与指数压缩（unary/MX），把每个张量拆成 speculation data + verification data；草稿模型 = 严格比特子集（零额外显存，显存甚至低于 BF16 原模型），验证 = 全量数据重建原始模型。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
spec_W, ver_W = split(W, mask)            # mask = Wanda importance top-k
spec_kv, ver_kv = split(KV, mask)         # per-token 幅度 top-k
draft_logits = LM(spec_W, spec_kv, x)     # zero-padding 重建、标准 FP GEMM
drafts = [sample(draft_logits) ...]
p = LM(spec_W∪ver_W, spec_kv∪ver_kv, x + drafts)   # 全量并行验证（拒绝采样）
```
为何细粒度优于粗粒度：低 batch、中等序列长度下 decode 瓶颈从 attention 转移到 FFN 权重加载——层跳过方法不压缩 FFN 权重（Draft&Verify 对 32 层模型只跳 9 个 FFN 层、草稿仍须加载 70.7% 参数），KV-only 方法（MagicDec）低 batch 下有时慢于 baseline；Cassandra 对权重与 KV 都做位级压缩，直击 FFN 带宽瓶颈。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
代表实现：Draft&Verify、MagicDec、Lookahead Decoding、Swift、QuantSpec、Cassandra。适用场景：边缘低 batch（单/少数并发用户，batching 不可行）、资源受限（无训练算力、显存紧张）。局限：接受率与跨任务泛化一般弱于训练型草稿（EAGLE-3 在 AIME2025/GPQA 更优但依赖训练数据分布、长序列任务收益骤降）；MagicDec 依赖 KV 剪枝在低 batch 失效。

涉及论文标题：
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding
