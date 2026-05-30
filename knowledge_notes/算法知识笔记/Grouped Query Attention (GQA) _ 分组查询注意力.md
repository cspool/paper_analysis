## Grouped Query Attention (GQA) / 分组查询注意力

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GQA (Grouped Query Attention) 是介于 Multi-Head Attention (MHA) 和 Multi-Query Attention (MQA) 之间的注意力机制 (Ainslie et al., 2023)。MHA 为每个 head 分配独立 Q/K/V (H heads → H 组 K/V)，KV cache 大；MQA 所有 heads 共享一组 K/V，节省内存但可能损失质量。GQA 折衷：H 个 query heads 分 G 组，每组共享一组 K/V (H/G heads/group)。KV cache 从 MHA 的 2×H×L×d_head 降至 GQA 的 2×G×L×d_head。LongLLaVA 在 hybrid 架构的 Transformer attention 层中使用 GQA。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# GQA: H=32 query heads, G=8 KV heads, r=H/G=4
def GQA(x):
    Q = reshape(W_Q@x, [B,L,H,d])     # H heads
    K = reshape(W_K@x, [B,L,G,d])     # G heads only
    V = reshape(W_V@x, [B,L,G,d])
    # Expand K,V: G → H via repeat
    K = repeat(K, "... G d → ... (G r) d", r=H//G)  # [B,L,H,d]
    V = repeat(V, "... G d → ... (G r) d", r=H//G)
    scores = Q @ K.T / sqrt(d); attn = softmax(scores)
    return W_O @ reshape(attn @ V, [B,L,D])
```

Annotations: H=32, G=8, r=4 (common); KV cache: MHA 2*H*L*d, GQA 2*G*L*d (4× smaller); FlashAttention supports GQA natively via num_kv_heads parameter.

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GQA 通过 up-training 从 MHA checkpoint 转换 (Ainslie et al., 2023)：mean pool K/V projections 后少量步骤微调。Llama 2/3、Mistral、Qwen 等主流 LLM 均采用 GQA。FlashAttention 原生支持。LongLLaVA 中 GQA 仅用于 attention layers (12.5% layers)，配合 Mamba layers 整体 KV cache memory 大幅低于纯 Transformer。

涉及论文标题：
- LongLLaVA__Scaling_Multi-modal_LLMs_to_1000_Images_Efficiently_via_Hybrid_Architecture
- VideoNSA__Native_Sparse_Attention_Scales_Video_Understanding

VideoNSA 基于 Qwen2.5-VL-7B 构建，其 LLM decoder（Qwen2.5-7B）使用 GQA 配置 28 query heads 共享 4 KV heads（group ratio = 7:1）。在 VideoNSA 的 hybrid attention 设计中，GQA 被用于 text token 的 standard attention path（保留指令跟随能力），而 vision tokens 则使用 NSA（三支路稀疏注意力）。这种 hybrid design 的关键优势：GQA 的 KV cache 复用特性天然降低了 text-side 的 KV cache 内存；NSA 的稀疏 attention 则将 vision-side 的计算从 O(L²) 降至 O(L×K_attn)。两者互补，使 VideoNSA 在 128K context 下仅需使用 3.6% 的 attention edges 且保持文本理解精度。
