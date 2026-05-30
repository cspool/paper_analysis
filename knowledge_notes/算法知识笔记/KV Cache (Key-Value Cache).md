## KV Cache (Key-Value Cache)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

KV Cache（Key-Value Cache）是Transformer自回归推理中用于消除冗余计算的核心技术。在自回归生成（autoregressive generation）中，模型逐token生成输出。第t步生成时，需要计算当前token对之前所有t-1个历史token的attention。若每步都重新计算所有历史token的Key和Value投影，则第t步需O(t·d²)新计算。KV Cache的核心思想：每步生成token后，将其Key向量k_t = x_t·W^K和Value向量v_t = x_t·W^V存储（缓存）在GPU内存中。下一步attention计算时，query q_{t+1}只需与已缓存的K矩阵（形状[t, d]）做attention，无需重新投影历史token。这减少计算量从O(t²·d²)到O(t·d²)。代价是存储L层×2（K和V）×序列长度×head_dim×num_heads×precision bytes的缓存，在长序列和大模型中成为内存瓶颈。对于MLLM，visual tokens（来自ViT patch embeddings，通常数百到数千个）显著增加了KV cache的序列长度压力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# KV Cache在自回归生成中的工作流程：
# L层transformer, 每层有独立的KV cache

# === Prompt Encoding Phase (Prefill) ===
X = concat(prompt_embeddings)  # [L_p, d]
for layer l in 1..L:
    K_0^l = X · W_K^l          # [L_p, d] × [d, d] → [L_p, d]  (full projection)
    V_0^l = X · W_V^l          # [L_p, d] × [d, d] → [L_p, d]
    缓存 K_0^l, V_0^l           # 存储到GPU memory

# === Generation Phase (Decode) ===
for step t = 1, 2, ...:
    x_t = embedding(token_{t-1}) # 上一个生成的token
    for layer l in 1..L:
        k_t^l = x_t · W_K^l      # [1, d]  (仅计算新token的K/V)
        v_t^l = x_t · W_V^l
        K_t^l = [K_{t-1}^l; k_t^l]  # 追加到已有cache: [L_p+t, d]
        V_t^l = [V_{t-1}^l; v_t^l]
        o_t^l = Softmax(q_t^l · (K_t^l)^T / √d) · V_t^l  # attention
    生成 next_token = argmax(output_projection(o_t^L))

# 内存占用示例（Qwen2.5-VL-7B: visual tokens=1024, text tokens=256, 28 layers, 28 heads, head_dim=128, BF16）:
# 单层K cache: (1024+256) × 28 × 128 × 2 bytes = 9.2 MB
# 单层V cache: 同上 = 9.2 MB
# 总KV Cache: 28 layers × 2 × 9.2 MB ≈ 515 MB
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

KV Cache在PyTorch中的实现：每层维护两个tensor缓冲区，prefill时批量写入所有prompt token的K/V，decode时每次追加一个token（torch.cat操作）。HuggingFace Transformers的`DynamicCache`类管理动态增长的KV cache。推理框架如vLLM使用PagedAttention将KV cache按block（page）管理以减少内存碎片。KV cache压缩技术分为三类：(1) eviction——丢弃低重要性token（H2O, SnapKV）；(2) quantization——降低KV精度（KIVI, GEAR, MiKV）；(3) merging——将低重要性token合并到保留token（KVMerge, CaM, FlowMM）。

涉及论文标题：
- FlowMM Cross-Modal Information Flow Guided KV Cache Merging for Efficient Multimodal Context Inference
