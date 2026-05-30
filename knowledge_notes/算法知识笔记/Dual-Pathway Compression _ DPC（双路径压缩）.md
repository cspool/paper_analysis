## Dual-Pathway Compression / DPC（双路径压缩）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dual-Pathway Compression (DPC) 是 FlexMem 提出的视觉 KV cache 压缩方法，核心洞察：MLLM 的 prefill 阶段（需要历史上下文聚合）和 decoding 阶段（需要显著性保留）对"哪些 visual token 重要"有不同标准。DPC 将压缩解耦为两条路径：(a) **Context Compression**——使用 context aggregation score s_j^l = Σ_{k∈C} a_{jk}^l + Σ_{h∈Vi} a_{hj}^l 选择最能"聚合历史信息并传播给后续"的 token，产生 Context Memory Ci（用于迭代信息传递）；(b) **Local Compression**——使用 local saliency score ŝ_j^l = Σ_{k∈Vi} a_{kj}^l 选择 clip 内最具"显著性"的 token，产生 Local Memory Mi（写入 Memory Bank 供最终召回）。DPC 与现有 KV cache 压缩方法（如 AdaRETAKE、Video-XL）的核心区别在于区分 prefill 和 decoding 的不同目标，而非对所有 token 使用统一的重要性度量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DPC 在每层 l 的具体计算（共享同一 forward pass 的 attention matrix）：
```
# === Dual-Pathway Compression (per layer l) ===
A = Attention([Q_{Vi}, Q_{Tq}], [K_C, K_{Vi}, K_{Tq}])  # cross-clip attention

# Pathway 1: Context Memory (服务于prefill的信息传递)
for token j in Vi:
  s_j = sum(A[j, :C]) + sum(A_self[h, j] for h in Vi where h>j)
  # 第1项: 从历史context聚合的信息
  # 第2项: 对后续token的因果传播
c_i^l = {K[j], V[j] for j in topK(s, alpha_c * |Vi|)}

# Pathway 2: Local Memory (服务于decoding的显著性保留)
for token j in Vi:
  ŝ_j = sum(A_self[:, j])  # clip内部影响力
m_i^l = {K[j], V[j] for j in topK(ŝ, alpha_s * |Vi|)}

Ci = [c_i^1..c_i^L];  Mi = [m_i^1..m_i^L]
```
消融实验验证：(1) Context Compression Only 在长视频上丢失局部显著信息；(2) Local Compression Only 在需要跨 clip 理解的场景下不足；(3) Dual-Pathway 在所有时长上均最优，性能增益随视频长度增加而扩大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DPC 在 MLLM 的 forward pass 中自然计算——attention matrix 是 prefill 的副产物，topK 选择是纯排序操作，计算开销可忽略。compression ratio α_c 和 α_s 控制两种记忆的压缩程度，需要在"信息保留"和"内存节省"之间权衡。在 LLaVA-Video 7B 上，每 clip 8 帧经 DPC 压缩后，总解码 token 数仅 13k（vs AdaRETAKE 的 40k），同时实现了 8× baseline 的帧覆盖。

涉及论文标题：
- Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism
