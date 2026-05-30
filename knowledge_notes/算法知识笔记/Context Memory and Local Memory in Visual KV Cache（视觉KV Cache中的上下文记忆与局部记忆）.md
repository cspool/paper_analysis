## Context Memory and Local Memory in Visual KV Cache（视觉KV Cache中的上下文记忆与局部记忆）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Context Memory (C) 和 Local Memory (M) 是 FlexMem Dual-Pathway Compression 产生的两种压缩视觉 KV cache。**(a) Context Memory Ci**：度量 token 的"信息桥梁"能力——s_j^l = Σ_{k∈C} a_{jk}^l + Σ_{h∈Vi} a_{hj}^l，选取最能聚合历史信息并传播给后续的 token。Ci 在迭代编码中传递给后续 clip，实现跨 clip 的时序连续性。**(b) Local Memory Mi**：度量 token 的"显著性"——ŝ_j^l = Σ_{k∈Vi} a_{kj}^l，选取 clip 内被广泛关注（即"显著"）的 token。Mi 存入 M_bank 供最终召回。两者的关系：(i) 互补——C 保证时间线的信息连续性，M 保证每个时刻的视觉显著性；(ii) 消融实验（Table 5 Block 2）验证 C+M 的组合显著优于单独使用任一；(iii) 生命周期不同——C 在迭代中流式传递（生产→消费→丢弃），M 持久存储（生产→存储→召回→解码）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Context vs Local Memory 对比计算
A_cross = softmax(Q_{Vi} @ K_C^T / sqrt(d))     # [|Vi|, |C|]
A_self  = softmax(Q_{Vi} @ K_{Vi}^T / sqrt(d))  # [|Vi|, |Vi|]

# Context: 信息桥梁 (第j个token的context aggregation能力)
s_j = row_sum(A_cross[j,:]) + col_sum_upper(A_self[:,j])
#     聚合历史 ↑               传播给未来 ↑

# Local: 视觉显著性 (第j个token在clip内的受关注度)
ŝ_j = col_sum(A_self[:,j])
#     所有token对j的attention总和 ↑
```
Context 和 Local 共享相同的 attention matrix，计算无额外开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
两种记忆的关键设计选择：Context Compression 保留能"讲好故事"的 token（连接前后文的桥梁），Local Compression 保留能"提供证据"的 token（回答问题时需要的视觉线索）。这一设计可推广到其他需要渐进式信息压缩和理解的长序列任务。压缩比 α_c 和 α_s 可在信息保留和内存效率之间权衡。

涉及论文标题：
- Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism
