## Multi-modal LLM (MLLM / 多模态大语言模型)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Multi-modal LLM (MLLM) 是一种将预训练 LLM 与视觉编码器结合，使其能理解图像/视频等多模态输入的模型架构。典型 pipeline：Visual Encoder (如 CLIP-ViT) → Projection/Adapter (如 MLP) → LLM (如 LLaMA/Qwen/Vicuna) → Text Decoder。视觉数据经 Encoder 转换为 visual token embeddings，经 Adapter 投影到 LLM 的 embedding 空间，与 text tokens 拼接后送入 LLM 做多模态推理。

AIM 论文使用的基座模型：LLaVA-OneVision-7B（video, Qwen2-7B backbone）、LLaVA-1.5-7B（image, Vicuna-v1.5-7B backbone）。video LLM 从视频均匀采样 32~192 帧，每帧经 ViT 编码为数百 tokens，总计可达数千 visual tokens。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**MLLM 标准推理 Pipeline（以 LLaVA-OV-7B 为例）**：

```
// 1. 视觉编码阶段
frames = sample_frames(video, num_frames=32)  // 均匀采样 32 帧
for frame in frames:
    v_tokens_per_frame = ViT(frame)  // [H×W/patch², D_vis]
// 总计：32 × (336/14)² = 32 × 576 ≈ 18432 tokens → Adaptive Pooling → ~2304 tokens

// 2. 投影阶段
v = MLP_Adapter(v_tokens)  // D_vis → D_llm (embedding 对齐)

// 3. 多模态推理阶段
text_tokens = Tokenizer(prompt)
x = Concat([v; text_tokens])  // [N_v + N_t, D_llm]
for l in 1..L:
    x = TransformerLayer_l(x)  // Self-Attn + FFN + Residual

// 4. 文本生成
output = lm_head(x[-1])  // 取最后一个 token 的 logits
```

**AIM 在此 pipeline 中的插桩位置**：
- Token Merging：在 Step 2 和 Step 3 之间（Adapter 输出→Token Merging→LLM 输入）
- Token Pruning：在 Step 3 的每个 TransformerLayer 的 Self-Attention 之后

术语一般如何实现？如何使用？

主流 MLLM 实现包括 LLaVA 系列（https://github.com/haotian-liu/LLaVA）、Qwen-VL（https://github.com/QwenLM/Qwen-VL）等。AIM 以即插即用方式集成到 LLaVA 推理流程中，无需修改模型权重。安装依赖：PyTorch 2.3.1, CUDA 12.1, 修改版 transformers/lmms-eval/qwen-vl-utils。

Dynamic-LLaVA 在 LLaVA-1.5 标准 pipeline 第 l=2 层 decoder 后插入两个轻量 predictor，分别对 vision 和 language token 做 keep/discard 决策，决策共享至所有后续层。训练时冻结 Vision Encoder 和 Projector，仅更新 LLM 和 Predictor 参数（LLM lr=5e-6, Predictor lr=2e-4）。额外的 MLLM 效率问题：prefill 仅执行一次，image token 减少的收益在 decoding 阶段随 output token 数量增长而逐渐湮没，因此需同时稀疏化 vision 和 language 上下文。

涉及论文标题：
- AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning
- Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification

---
