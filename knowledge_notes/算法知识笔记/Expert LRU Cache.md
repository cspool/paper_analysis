## Expert LRU Cache

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert LRU Cache 是一种针对 MoE 推理的 GPU 显存管理策略。在 MoE offloading 场景中，expert 参数大部分存储在 host RAM 中，每次推理仅加载当前 token 所需的 top-k expert 到 GPU。LRU Cache 利用相邻 token 间 expert 使用的局部性（expert locality），在 GPU 显存中为每个 MoE 层维护 k 个最近使用过的 expert 作为缓存。处理新 token 时，若所需 expert 已在 cache 中（cache hit），则无需 host-to-device 传输直接使用；若不在（cache miss），则从 host RAM 加载，并淘汰 cache 中最久未使用的 expert（若 cache 已满）。对于 Mixtral-8x7B，k=2（12GB GPU）或 k=4（16GB GPU）。

该策略的核心洞察来自对 MoE 模型 expert 激活模式的观测（图 1）：某些 expert 在 2-4 个连续 token 上反复使用，另一些则以"间隔"模式被复用。LRU 是最简单的缓存替换策略——不考虑 expert 激活频率、不同 MoE 层 cache 大小的变化或 expert 激活的序列模式——但即使如此简单的策略也能显著加速 MoE 推理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Per-layer Expert LRU Cache for Mixtral-8x7B
# 每层维护 C_l: OrderedDict, max size=k (LRU order via move_to_end)

for token t in generate():
    for layer l in 0..31:
        h = attention_block[l](h)             # attention 常驻 GPU
        gate_scores = W_gate[l] @ h           # gate 常驻 GPU
        top2_idx = topk(gate_scores, k=2)    # [e_a, e_b]
        
        output = zeros_like(h)
        for e_id in top2_idx:
            if e_id in C_l:
                # Cache hit: expert 已在 GPU 显存
                expert_w = GPU_expert_buf[l][e_id]
                C_l.move_to_end(e_id)          # 标记为 most recently used
            else:
                # Cache miss: 从 host RAM 加载
                if len(C_l) >= k:
                    evict_id, _ = C_l.popitem(last=False)  # 淘汰 LRU
                    # 若 host RAM 不足, 将 evicted expert 写回 host
                    copy GPU_expert_buf[l][evict_id] → host_pinned[l][evict_id]
                # 加载新 expert
                copy host_pinned[l][e_id] → GPU_expert_buf[l][e_id]
                C_l[e_id] = True               # 加入 cache
            
            gate_w = gate_scores[e_id] / sum(gate_scores[top2_idx])
            output += gate_w * expert_ffn(GPU_expert_buf[l][e_id], h)
        
        h = output
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现细节（论文 Section 3.3）：
- Expert 参数在 pinned memory 中以连续 buffer 存储，单次 `cudaMemcpyAsync` 完成 host-to-device 传输
- GPU 侧预分配 b=4 个临时 device buffer 用于异步 expert 交换，所有 MoE 层共享以减小内存足迹
- 当 host RAM 也无法容纳完整模型时（如 Google Colab），expert 在 host RAM 和 GPU 之间按 LRU 策略换入换出，换出时写回 host
- 实现代码开源在 https://github.com/dvmazur/mixtral-offloading
- **FloE 中的扩展**：FloE 同样受益于 expert locality——相邻 token 倾向于激活相同或相近的 expert。FloE 的 inter-expert predictor (MLP) 学习捕获这种时序关联性，但预测失败时仍需 fallback 到 LRU cache 机制。FloE 的 VRAM 消融实验（Figure 8）表明，随 VRAM 增加，可缓存更多 MoE 层的 expert，减少 misprediction reload overhead。

涉及论文标题：
- Fast Inference of Mixture-of-Experts Language Models with Offloading
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU
