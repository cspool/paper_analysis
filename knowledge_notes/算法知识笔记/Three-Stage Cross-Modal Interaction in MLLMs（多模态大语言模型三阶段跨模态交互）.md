## Three-Stage Cross-Modal Interaction in MLLMs（多模态大语言模型三阶段跨模态交互）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VisiPruner (EMNLP 2025) 通过系统消融实验揭示的 MLLM 内部跨模态信息处理的三阶段规律。该框架将 MLLM 的 32 层 transformer 按功能分为三个阶段：(1) **Shallow Layers（浅层，layer 1-8）— Task Recognition**：视觉和文本 token 在浅层独立演化，无有意义的跨模态融合。文本 token 的 hidden state 经 vocabulary projection 显示浅层编码的是 task type（如"number"对应 counting、"type"对应 classification）而非视觉内容。视觉 token 在浅层仅作为 attention sink 稳定 softmax 分布，其 self-attention 也无实质作用——mask 后性能几乎不变。(2) **Middle Layers（中层，layer 9-23）— Sparse Cross-Modal Grounding**：跨模态融合在此阶段突然发生（abrupt onset），由少数关键视觉 token 驱动（~10/576）。这些关键 token 对应 instruction-relevant 的图像区域，且在不同中层间保持稳定（不需要每层重新识别）。然而 attention-based 方法无法准确选出这些 token——因为 attention sink 现象污染了 attention score 与信息贡献的对应关系。(3) **Deep Layers（深层，layer 24-32）— Linguistic Alignment**：视觉信息已集成到文本表示中，模型进入纯语言 refinement 阶段。深层将已识别的视觉答案（如"Lux"）组织为符合自然语言语法和流畅性的完整回答（如"The scene text is 'Luxmi Jewellers'"）。继续在深层处理视觉 token 会产生噪声，反而降低性能。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
三阶段在 MLLM 推理中的行为差异（以 LLaVA-v1.5 7B 为例）：
```
# Three-Stage Cross-Modal Processing in MLLM Inference
# N_v=576 visual tokens, N_t~74 text tokens, L=32 layers

# Stage 1: Shallow (layers 1-8) — Task Recognition
for l in 1..8:
    # Cross-attention exists but contributes NO visual content
    H_t = TextSelfAttn(H_t) + CrossAttn(H_t, H_v)  # cross-attn ≈ noise
    H_v = VisualSelfAttn(H_v) + FFN(H_v)            # self-attn ≈ redundant
    
    # Probing: project last text token's hidden state to vocab
    D_last = softmax(W_u @ H_t[-1])
    # Layer 7-10 outputs: "number", "type", "count" — task semantics, NOT visual

# Stage 2: Middle (layers 9-23) — Sparse Grounding  
for l in 9..23:
    # Cross-attention now fuses real visual info
    # BUT: only ~10/576 vision tokens matter
    H_t = TextSelfAttn(H_t) + CrossAttn(H_t, H_v)  # real fusion happens
    H_v = VisualSelfAttn(H_v) + FFN(H_v)
    
    # Top attended tokens consistently focus on instruction-relevant regions
    # e.g., "What kind of apple?" → tokens 107,108,222 (apple region)

# Stage 3: Deep (layers 24-32) — Linguistic Alignment
for l in 24..32:
    # Visual tokens no longer needed — can be safely discarded
    H_t = TextSelfAttn(H_t) + CrossAttn(H_t, H_v)  # cross-attn ≈ noise again
    # Probing: vocab projection shows "Lux" → "The" → "All" → grammatical refinement
```

Annotations: 阶段边界通过"mask cross-attention from shallow-to-deep / deep-to-shallow"的双向实验确定（Fig. 3）。Mask layer 1-7 跨模态 attention 性能仅轻微下降，但 mask layer 9-15 显著下降——标记中层开始。Mask layer 27-32 性能再次几乎不变——标记深层开始。该三阶段框架在 LLaVA-v1.5 7B/13B、InternVL2.5 8B、Qwen2-VL 7B、MobileVLM-v2 3B 上均验证有效，但各模型的阶段边界因架构不同而有所偏移（如 MobileVLM 3B 浅层更宽）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
三阶段框架的使用方式：(1) 分析工具：通过 semantic projection（W_u @ h_last, Eq. 2）和 value-output matrix projection（W_u @ V_last @ O, Eq. 3/9）观察每层的功能语义；(2) 剪枝指导：每阶段采用不同的压缩策略——浅层 attention merging + attention skipping，中层 influence-based token selection，深层 vision exit；(3) 训练指导：VisiPruner Section 6 基于三阶段提出三条 MLLM 训练建议：(a) 截断浅层视觉层数并消除 cross/self-attention；(b) 训练模型在中层进行稀疏注意力直接识别关键 token；(c) 在深层引入视觉退出机制。框架推翻了此前"浅层是跨模态融合主要场所"的共识（Wu et al., 2024; Zhang et al., 2025a），指出该共识因过度依赖 attention scores 作为信息流代理而产生误导——attention scores 在浅层的分布主要由 attention sink 决定而非指令相关性。

涉及论文标题：
- VisiPruner__Decoding_Discontinuous_Cross-Modal_Dynamics_for_Efficient_Multimodal_LLMs
