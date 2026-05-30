## Self-Speculative Decoding / SSD（自投机解码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Self-Speculative Decoding (SSD) 是一种无需独立 draft model 的投机解码变体。核心思想：利用同一个模型的**浅层子网络作为 draft model**，**完整模型作为 target model**，通过"draft-then-verify"的迭代方式加速自回归解码。与传统投机解码（需要另一个独立的小模型做 draft）不同，SSD 的 draft 和 target 模型共享部分层的权重和 KV-cache。最简单的方式是 LayerSkip 的 early-exit：让 LLM 的中间层直接预测下一个 token 作为 draft，然后用完整模型验证。TwigVLM 的 SSD 设计：在 base VLM 第 K 层后附加 T 层 twig block，形成浅层子网络 Ms（前 K 层 + twig）作为 draft model，完整模型 Mb（全部 L 层）作为 target model。Draft 每次自回归生成最多 δ=5 个候选 token（配合 early-exit 机制：当预测概率 < θ=0.6 时提前停止 draft），target 通过一次并行前向验证所有候选 token。接受匹配的 draft tokens 后，target 追加一个 bonus token。由于 draft 和 target 共享前 K 层的计算和 KV-cache，SSD 的开销远低于独立双模型方案。SSD 是 lossless 的：最终输出与 target model 原生自回归解码完全一致。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Self-Speculative Decoding 单次迭代
# Ms: 浅层 draft model (前K层 + twig)
# Mb: 深层 target model (完整L层)
# θ: 置信度阈值, δ: max draft length

# === Draft Phase (draft model 自回归生成候选tokens) ===
draft_tokens = []
for step in range(δ):
    logits_s = Ms.forward(current_token)  # 浅层前向
    prob = softmax(logits_s)
    next_tok = argmax(prob)
    draft_tokens.append(next_tok)
    if max(prob) < θ:           # early-exit
        break
    current_token = next_tok

# === Verify Phase (target model 并行验证) ===
logits_b = Mb.forward(draft_tokens)  # 一次并行前向
# 逐个对比 draft token 与 target 预测
accepted = []
for i, (draft_tok, logits) in enumerate(zip(draft_tokens, logits_b[:-1])):
    target_tok = argmax(logits)
    if draft_tok == target_tok:
        accepted.append(draft_tok)
    else:
        break
# 追加 bonus token
accepted.append(argmax(logits_b[len(accepted)]))
```

TwigVLM 的 SSD 关键设计：
- draft model Ms = {T_1..T_K} ∪ {G_1..G_T}（前K层+twig）
- target model Mb = {T_1..T_L}（完整VLM）
- 共享前K层 KV-cache，draft 只需计算 twig 的 forward
- δ=5，θ=0.6（early-exit 阈值）

TwigVLM++ 的 Tree-based SSD 扩展：
- Draft model 构建 token tree（expansion width E=10, selection width K=10, depth D=4）
- Target model 用 tree attention（topology-aware causal mask）并行验证所有候选路径
- 从根节点遍历，接受匹配的子节点，直到某层无匹配
- 追加一个 bonus token
- 每次验证接受更多 tokens，RelSpd 从 154% 提升到 ~197%（长 response 场景）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：LayerSkip 已集成到 HuggingFace Transformers 的 `generate()` 中（`assistant_early_exit` 参数）。TwigVLM 的 SSD 在 https://github.com/MILVLG/twigvlm 开箱可用。SSD 的关键优势：(1) 无需额外存储独立 draft model（节省 GPU 内存）；(2) 共享 KV-cache 减少冗余计算；(3) Lossless：输出与 target model 完全一致；(4) 特别适合长 response 场景（decode 阶段为主要瓶颈）。局限性：draft 的 token acceptance rate（TokAR）对 speedup 至关重要，受 draft model 质量影响。当 response 较短时加速有限（prefilling 时间占比高）。

涉及论文标题：
- Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models
