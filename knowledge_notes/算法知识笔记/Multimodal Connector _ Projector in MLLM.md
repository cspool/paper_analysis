## Multimodal Connector / Projector in MLLM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
多模态连接器（Multimodal Connector / Projector）是MLLM中位于视觉编码器和LLM之间的模块，负责将视觉特征映射到LLM的embedding空间，并可能对视觉特征进行进一步处理（如序列压缩、空间关系建模）。最简单的连接器是3层MLP（如LLaVA-1.5），更复杂的包括Q-Former（BLIP-2，使用learnable queries通过cross-attention压缩视觉token）、C-Abstractor（MobileVLM，通过depth-wise conv减少visual token数量）、MSC（ML-Mamba，使用Mamba-2 scan进行2D空间建模）。连接器设计影响视觉token数量、信息损失程度和计算效率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 不同连接器设计对比:

// 1. MLP (LLaVA-1.5): 最简方案
V_out = Linear2(GELU(Linear1(V_img)))  // 2-3层MLP

// 2. Q-Former (BLIP-2): 固定数量可学习queries
Q = nn.Parameter(torch.randn(K, D))  // K << N_v
V_out = CrossAttention(Q, V_img, V_img)  // K个输出token

// 3. MSC-MLP Advanced (ML-Mamba): Mamba-2 scan + SwiGLU + MLP
V_scan = MVSS(V_img)            // 2D spatial context via Mamba-2 scan
V_swiglu = SwiGLU(V_scan)       // gated feature extraction
V_out = MLP_3layer(V_swiglu)    // dimension alignment

// 4. C-Abstractor (MobileVLM): depth-wise conv压缩
V_out = Conv2D(reshape(V_img))  // 降低token数量 729→144
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
连接器的选择由三个因素决定：(1) 视觉token数量——token越多质量越好但推理越慢；(2) 压缩率——高压缩率（如C-Abstractor的5x）可以加速但可能损失信息；(3) 空间建模能力——如MSC的Mamba-2 scan可以建模patch间2D关系。ML-Mamba消融（Table 6）证实：MLP (VQAv2 73.42) → MSC-MLP Basic (+1.67) → MSC-MLP Advanced (+1.84)，处理729个visual tokens仍保持171 tokens/s的生成速度（vs MobileVLM v2用144 tokens仅50 tokens/s）。连接器训练通常分两阶段：先在caption数据上对齐（train connector only），再在instruction数据上联合微调（train connector + LLM）。

涉及论文标题：
- ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2

---

---
