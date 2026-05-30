## Multimodal Large Language Model (MLLM) / Visual Language Model (VLM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multimodal Large Language Model (MLLM) / Visual Language Model (VLM) 是将视觉理解能力与语言理解和生成能力结合的模型架构。典型架构包含三个核心组件：(1) 视觉编码器——将图片转换为特征向量（如CLIP、SigLIP、DINOv2）；(2) 多模态连接器/Projector——将视觉特征映射到LLM的输入空间（如MLP、Q-Former、MSC）；(3) 大语言模型（LLM）——处理拼接后的视觉+文本token并生成回答。代表性模型包括LLaVA、GPT-4V、Qwen-VL、BLIP-2等。MLLM的训练通常分两阶段：预对齐（visual-text alignment on image-caption pairs）和指令微调（instruction tuning on multi-turn dialog data）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 标准MLLM推理pipeline:
Input: Image X_v, Text Question Q

// Step 1: 视觉编码
V = VisionEncoder(X_v)  // ∈ R^{N_v×D_v}, N_v=patch数量

// Step 2: 特征投影
V_proj = Connector(V)  // ∈ R^{N_v×D_llm}, 映射到LLM embedding空间

// Step 3: token拼接
T = Tokenizer(Q)  // ∈ R^{L_text}, token IDs
Input_emb = concat([V_proj; Embedding(T)], dim=0)

// Step 4: LLM自回归生成
for each token position:
    Answer = LLM(Input_emb)  // 多轮自回归生成

// Transformer-based MLLM: O((V+T)²) attention计算, KV-Cache = O(V+T)
// SSM-based MLLM (ML-Mamba): O(V+T) scan计算, Hidden State = O(1) fixed size
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
主流实现框架：LLaVA (https://github.com/haotian-liu/LLaVA), LLaMA-Adapter, BLIP-2。ML-Mamba是将Mamba-2 SSM引入MLLM的早期工作之一，证明SSM-based backbone在保持线性复杂度的同时能匹敌Transformer-based MLLM（在POPE上88.3 vs LLaVA-1.5-7B的85.9）。通用benchmark包括VQAv2、GQA、TextVQA、POPE、VizWiz、MMBench等。当前趋势包括：(1) SSM/Linear Attention替代Transformer backbone（如ML-Mamba、VL-Mamba、Cobra）；(2) 更强视觉编码器（DINOv2+SigLIP/CLIP双编码器）；(3) 更高效的多模态连接器（Q-Former、MSC、C-Abstractor）；(4) 端到端训练（减少预对齐阶段）。

涉及论文标题：
- ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2

---
