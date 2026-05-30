## Discrete Diffusion Multimodal Large Language Model (DMLLM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Discrete Diffusion Multimodal Large Language Model (DMLLM) 是将视觉编码器与离散扩散语言模型（DLM）结合的多模态大模型。与标准MLLM（如LLaVA）使用自回归生成不同，DMLLM使用扩散过程生成文本回答。架构组成：(1) Vision Encoder（如Qwen2.5-VL ViT，冻结）编码图像为visual tokens；(2) Projector（2层MLP）将visual tokens映射到LLM embedding空间；(3) DLM Backbone（如Dream）处理拼接后的visual+text tokens，通过bidirectional attention和迭代去噪生成回答。Dimple是首个公开的DMLLM，证明DMLLM在相似训练预算下可达到与自回归MLLM相当的性能（13个benchmark平均62.4% vs LLaVA-NEXT 58.5%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

DMLLM推理pipeline（Dimple为例）：

```
Input: 图像 I, 文本问题 Q, 预定义生成长度 L_answer

1. Vision Encoding:
   visual_tokens = VisionEncoder(I)  # 冻结ViT, N_v个token
   visual_emb = Projector(visual_tokens)  # 2层MLP → LLM dim

2. Input Construction:
   prompt = [BOS] + visual_emb + text_emb(Q) + [EOS]
   answer_init = [[MASK]] * L_answer  # 全部初始化为[MASK]
   x_T = concat(prompt, answer_init)  # 总长度 L = L_prompt + L_answer

3. Structure Prior（可选）:
   预置特定位置token（如"Thus, the answer is \box{"），标记为"已确定"

4. Iterative Diffusion Decoding:
   For step t (直到所有[MASK]被填充):
     a. z_t = f_θ(x_t)  # bidirectional forward（可能使用Prefilling）
     b. p_t = softmax(z_t)  # pre-revision probabilities
     c. For each masked i: c^(i) = max(p_t^(i))  # confidence
     d. 选择c^(i) >= γ 的位置一次性批量更新；若无则fallback随机选择K个

5. Output: 去除[Padding] tokens，提取有效文本
```

Annotations: L_answer由response_length参数预设（Dimple使用4/8/16/64取决于benchmark）；N_v取决于图像分辨率和patch size；[Padding]是Dream tokenizer中的特殊token，用于填充answer长度不足的部分。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DMLLM训练使用Autoregressive-then-Diffusion策略：(1) Phase I AR Alignment & Tuning: causal attention + next-token prediction进行视觉-语言对齐和instruction tuning；(2) Phase II Diffusion Tuning: 恢复bidirectional attention + masked LM loss，仅mask answer部分，复用相同instruction数据。此策略解决纯扩散训练的两个低效——监督覆盖率低、每样本仅一个timestep监督。推理时Confident Decoding可将迭代数压缩到response_length/3左右。已开源：https://github.com/yu-rp/Dimple。

涉及论文标题：
- Dimple Discrete Diffusion Multimodal Large Language Model with Parallel Decoding
