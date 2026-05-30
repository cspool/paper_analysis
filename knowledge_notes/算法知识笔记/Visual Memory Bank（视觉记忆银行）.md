## Visual Memory Bank（视觉记忆银行）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Visual Memory Bank (M_bank) 是 FlexMem 中持久存储所有已处理 video clip 的压缩 local memory 的数据结构。每处理完一个 clip，其 local memory Mi 被追加到 M_bank。特点：(a) 固定每 clip 内存——由压缩比 α_s 决定，不随 clip 数增长；(b) 线性总内存——M_bank 大小 = N × |Mi|，随 clip 数 N 线性增长（vs 原始 KV cache 的 O(N²)；在 512 帧/64 clips 时内存约 134MB，远小于原始 KV cache）；(c) 结构化存储——按 clip 索引组织，支持按 relevance score 随机召回任意连续 clip 组；(d) 可选长期记忆召回——在迭代编码中可从 M_bank 预先调取长期记忆 `<Ml>` 作为当前处理的附加上下文。M_bank 使 FlexMem 同时具备 RAG 的精确定位（从记忆库检索）和压缩方法的全面理解（所有 clip 参与信息流）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# M_bank: 存储与召回
M_bank = []
for clip_idx in 1..N:
  M_i = local_compress(KV_i, alpha_s)  # DPC的local pathway
  M_bank.append(M_i)

# 召回 (encoding阶段可选，decoding阶段必须)
# encoding中: 召回长期记忆辅助当前处理
long_term = recall(M_bank, Tq, top_k=1)  # optional

# decoding前: 召回na个最相关连续clip
g = [relevance(M_i, Tq) for M_i in M_bank]
recalled = select_consecutive(M_bank, g, na)
Y = MLLM.decode(recalled, Tq)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 GPU 显存受限时，M_bank 可存储在 CPU memory 中，仅在召回时将选中片段加载到 GPU。这一设计类似于 RAG 的 offline index 但更轻量——不需要额外的 embedding model，所有 visual representations 来自 MLLM 自身的 KV cache。M_bank 的检索粒度是 clip 级（连续 na 个 clip），保证了召回的时序连续性。

StreamingEval 中的 Visual Memory Bank 变体：StreamingEval 采用固定容量 FIFO 内存银行作为评估离线 VideoLLM 的统一适配器。每帧视觉编码 $z_i = g_{\theta}(v_i)$ 经投影层对齐 LLM embedding 空间后写入 memory bank，超出字节预算 M 时按 FIFO 淘汰最旧内容。字节预算公式：$\operatorname{Mem}_i(B) = B \cdot d_i \cdot s_{\text{emb}} + B \cdot 2L_i \cdot h_i^{\text{kv}} \cdot s_{\text{kv}}$，其中 $d_i$ 为投影后 visual token embedding 维度，$L_i$ 为 LLM 层数，$h_i^{\text{kv}}$ 为 per-layer KV channel width。内存银行写入规则：$M_{\tau_i^+} = \mathcal{U}(M_{\tau_i^-}, z_i; B, \pi)$，B 为容量约束，π 为淘汰策略（离线模型 FIFO，在线模型用原生策略）。与 FlexMem 的 recall-based M_bank 不同，StreamingEval 的 FIFO memory bank 是线性顺序存储的无压缩 buffer，用于公平比较而非内容检索。

涉及论文标题：
- Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism
- StreamingEval__A_Unified_Evaluation_Framework_for_Streaming_Video_Understanding
