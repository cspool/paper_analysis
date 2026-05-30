## Vision Token Bidirectional Attention (视觉Token双向注意力)

术语解释
在VLM中让不同来源（不同视频帧、不同图像、不同crop）的vision tokens在LLM的self-attention层中互相attend的注意力策略，打破标准causal attention中vision token仅被前置token attend的单向限制。Molmo2发现启用cross-frame/cross-image双向注意力能显著提升视频理解性能（Table 8b: +0.4 QA avg, +1.0 Cap F1）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
标准VLM的LLM decoder使用causal attention mask——每个token只能attend到它之前的token。这意味着frame_2的vision tokens无法attend到frame_1的vision tokens（它们都在text prompt之前）。Vision Token Bidirectional Attention修改attention mask：所有vision tokens区域（[video_start]到[video_end]之间）设为全1（bidirectional），而text tokens保持标准causal。使不同帧/位置的vision tokens能直接交互信息（跨帧object re-identification、motion understanding），而非仅依赖text token作为中间桥梁。Gemma 3等模型也采用类似设计。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Custom Attention Mask for Bidirectional Vision Tokens
# Sequence: [BOS][video_start][frame1_tokens][t0.0][frame2_tokens][video_end][text]
#           |← vision token region (bidirectional) →| |← text (causal) →|

mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
# Vision region: bidirectional (all-to-all)
mask[vis_start:vis_end, vis_start:vis_end] = 0.0
# Text keeps causal, can attend to all preceding vision
# Use PyTorch SDPA (NOT FlashAttention — doesn't support arbitrary custom masks)
attn_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=False)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
必须使用SDPA而非FlashAttention（因不兼容custom mask）。与torch.compile兼容（static shape required）。与message tree encoding + packing的custom mask协同叠加。适用场景：任何多帧/多图VLM的LLM decoder训练和推理。计算成本：vision region ~10K tokens时增加额外~10K^2/2 pairwise attention（vs causal下三角），但Molmo2通过packing和pooling已在其他维度压缩。

涉及论文标题：
- Molmo2__Open_Weights_and_Data_for_Vision-Language_Models_with_Video_Understanding_and_Grounding
