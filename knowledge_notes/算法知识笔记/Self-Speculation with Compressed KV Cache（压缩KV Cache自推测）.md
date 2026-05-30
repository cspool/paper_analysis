## Self-Speculation with Compressed KV Cache（压缩KV Cache自推测）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Self-Speculation with Compressed KV Cache 是 MagicDec 提出的核心 SD drafting 策略：使用 target model 自身（而非独立的小 draft model）作为 draft model，配合稀疏化的 KV cache 进行 token 推测。与传统 SD 使用独立小模型作为 draft 不同，self-speculation 不额外加载任何模型权重（draft 与 target 共享），仅维护一份压缩 KV cache（budget K << full sequence length S）。

核心优势：(1) **低 draft cost**：draft 无额外参数加载，仅 KV loading cost = B × K × model_dim（vs target 的 B × S × model_dim），当 S > S_inflection 时 T_D/T_T → 0；(2) **高接受率**：KV 压缩比模型压缩更容易达到 >90% 的 token 接受率——因为 target model 看到的是自己"实际会关注的" KV 子集，而非一个能力更弱的小模型的预测；(3) **lossless**：验证阶段仍使用完整 KV cache，输出与 target model AR 解码完全一致。

在 MagicDec 中，self-speculation 在小 batch + 短序列时性能不如小 draft model（因为参数加载占主导，小模型的参数更少），但在大 batch + 长序列时超越小模型（KV bottleneck 主导，self-speculation 的高接受率优势体现，Figure 7c）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Self-Speculation 与传统 SD 的对比

# 传统 SD: 小 draft model + 完整 KV
draft_params = Theta_draft   # 独立加载（如 LLaMA-3.2-1B ~2.5 GB）
K_draft = K_full             # 完整 KV（~25 GB for B=128, S=32K）
T_D = Load(Theta_draft) + Attend(K_draft)  # 两项都大

# Self-Speculation: target 自身 + 压缩 KV
draft_params = Theta_target  # 复用 target 已在 GPU 上的权重
K_draft = Select(K_full, budget=K)  # 压缩 KV（~1.6 GB for B=128, K=2049）
T_D = Attend(K_draft)                # 仅 KV attention，无额外参数开销

# MagicDec 的 self-speculation decode 循环
# 每步 draft phase 使用压缩 KV:
q = W_q @ embed(last_token)
s = q @ K_draft^T / sqrt(d_head)     # K_draft size: [B, K, n_heads, d_head], K << S
a = Softmax(s)
o = a @ V_draft
# ... FFN + LM Head → next draft token

# 新 token 的 KV 追加到 K_draft, V_draft（而非重新压缩）
# 保证 K_draft 始终保持 K 核心 + 最近新增 token
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Self-speculation 在 MagicDec 中通过 GPT-Fast 后端实现。预填充阶段通过 SnapKV/StreamingLLM 选择压缩 KV 索引 → 构建 K_draft, V_draft。解码阶段 draft step 复用 target model weights + 压缩 KV attention，verify step 使用完整 KV + FlashInfer attention。KV 预算 K 由 MagicDec 框架根据模型/硬件/任务通过公式 (4) 优化选择。开源：https://github.com/Infini-AI-Lab/MagicDec。

涉及论文标题：
- MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding
