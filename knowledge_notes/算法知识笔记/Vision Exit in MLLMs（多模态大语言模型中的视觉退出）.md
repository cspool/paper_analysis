## Vision Exit in MLLMs（多模态大语言模型中的视觉退出）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Vision Exit（视觉退出）是 VisiPruner 提出的 MLLM 深层优化策略：在 cross-modal fusion 完成后的某一层（称为 vision exit layer ℓ_exit），将所有保留的视觉 token 从后续层中移除，使深层仅做纯文本语言 refining。其核心发现是：MLLM 的深层（LLaVA-v1.5 7B 中约 layer 26+）已不再依赖视觉 token——跨模态信息已在中间层充分集成到文本表示中，继续保留视觉 token 不仅无益，反而引入噪声。实验证据：(1) 从 layer 26 起 discard 所有视觉 token，GQA 几乎不变（61.95→61.91）；(2) 但如果 skip layer 26 的视觉处理却继续在后续层处理视觉信息，性能反而下降（61.95→61.40）——说明 layer 26 的视觉处理本身在引入噪声；(3) mask 深层 cross-attention 对性能无影响（Fig. 3）；(4) 移除深层 KV cache 中的视觉信息性能略升（Tab. 2, MM-Vet 31.2→31.1 at layers 26-32）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Vision Exit Detection in VisiPruner
# After filtering layer, continuously track retained tokens' influence

filtering_layer_found = False
no_impact_counter = 0
vision_exit_layer = None

for l in range(S_mid + 1, L):
    if filtering_layer_found:
        # Track influence of currently retained visual tokens
        influence_score = compute_influence(H_v_retained, H_t)
        
        if influence_score < eps:  # negligible impact
            no_impact_counter += 1
        else:
            no_impact_counter = 0
        
        if no_impact_counter >= 2:
            vision_exit_layer = l  # exit here
            H_v = []                # remove ALL visual tokens
            break
    
    H = TransformerLayer(concat(H_v, H_t))
    # ... rest of forward pass (text-only from here on)

# Post-exit: pure language layers
for l in range(vision_exit_layer, L):
    H_t = TransformerLayer_text_only(H_t)
```

Annotations: ℓ_exit 通过"连续两层无 influence"的准则确定，避免单层波动导致过早退出。在 LLaVA-v1.5 7B 上 ℓ_exit 平均为 layer 23.9。退出后仅剩 text self-attention + FFN，compute 大幅减少。视觉退出与浅层 attention merging 互补：浅层减少 N_v×N_text attention，深层消除剩余的视觉 token 处理。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Vision exit 在 VisiPruner 推理代码中实现：在每层 forward 后检查保留视觉 token 的 cosine similarity / L2 distance 是否仍超过阈值，若连续两层都不超过则标记退出。该机制即插即用，不修改模型权重。适用于所有 decoder-only MLLM 架构（LLaVA、InternVL、Qwen2-VL 等），因为深层 focus on linguistic refinement 的现象是 MLLM 深层 encoder 行为的通用性质。可将 vision exit 机制嵌入 MLLM 训练 pipeline——如 VisiPruner Section 6 建议的"(c) Enable early exiting in deep visual layers once modality fusion is established"——通过在训练时加入 visual token dropout 或 exit loss 使模型学会自动决定退出时机。与"Confidence-Based Early Exit"（推理链提前终止）不同，Vision Exit 专指视觉 token 的退出，属于跨模态计算优化。

涉及论文标题：
- VisiPruner__Decoding_Discontinuous_Cross-Modal_Dynamics_for_Efficient_Multimodal_LLMs
