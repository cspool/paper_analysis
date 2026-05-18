## Token Pruning (Prompt-aware Token Pruning)

术语是什么？通过联网搜索让回答具体和精准。
Token Pruning是一种通过移除不重要tokens来减少VLM/LLM推理计算量的技术。传统token pruning方法（如Prumerge、FrameFusion）依赖静态heuristics如token magnitude、saliency或visual token间相似度，忽略文本prompt对token重要性的影响。Focus提出的prompt-aware token pruning利用cross-modal attention scores动态识别与当前prompt语义相关的visual tokens，保留semantically important tokens同时prune不相关tokens。Prune掉的tokens在后续P×V计算和下游层中不再加载，实现cumulative computation和memory access节省。保留比例逐层递减（40%→10%），深层layer prune更激进。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Prompt-aware token pruning流程：
```
# 在attention层中，SEC执行token pruning
attn_softmax = SoftMax(Q @ K^T / sqrt(d))  # (M+T)×(M+T)
cross_modal_I = attn_softmax[M:M+T, 0:M]  # text-to-image block
for j in 0..M-1:  # for each image token
    importance[j] = max(cross_modal_I[:, j])  # max over text tokens
    for head in 1..n_heads:
        importance[j] = max(importance[j], max(cross_modal_I_head[head][:, j]))

# streaming top-k selection
top_k_indices = streaming_bubble_sort(importance, k=retain_ratio*M)
retained_tokens = image_tokens[top_k_indices]

# pruned tokens excluded from downstream:
# P(i)×V only loads retained_tokens (not full M tokens)
# FC layers only process retained_tokens
```
Pruning ratio per layer: [3: 40%, 6: 30%, 9: 20%, 18: 15%, 26: 10%] of original M image tokens retained。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
算法实现：在PyTorch中hijack VLM的attention层，从SoftMax输出中提取text-to-image attention block，计算importance vector并做top-k mask，将prune后的token indices传到下游层。硬件实现：Focus的SEC模块将importance analyzer（parallel max units + 25KB buffer）、streaming bubble sorter和offset encoder集成到systolic-array accelerator的attention pipeline中。与静态pruning（如FrameFusion固定70% sparsity）相比，prompt-aware pruning使Focus在更高sparsity下（82.82%）保持更好accuracy（62.74 vs original 64.15, -1.41）。开源实现见https://github.com/dubcyfor3/Focus。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

