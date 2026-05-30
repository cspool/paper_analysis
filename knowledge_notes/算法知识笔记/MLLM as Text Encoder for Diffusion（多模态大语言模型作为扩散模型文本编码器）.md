## MLLM as Text Encoder for Diffusion（多模态大语言模型作为扩散模型文本编码器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
使用多模态大语言模型（如 Qwen2-VL、InternVL）替代传统 CLIP/T5 作为扩散模型的文本编码器。传统 CLIP 限制输入 77 tokens，T5 对细粒度场景理解不足。MLLM 在视觉-语言任务上预训练，具有更好文本理解能力，且统一文本-视觉 token 空间与视频生成任务天然匹配。EasyAnimate 从 Qwen2-VL-7B 倒数第二层提取 hidden features，经 RMSNorm + FC 对齐后输入 DiT。VBench 验证 Total Score 从 80.42% (T5+CLIP) 提升到 81.57% (Qwen2-VL)。由于 MLLM 特征的 L2 norm 远大于视频噪声 latent，需要 RMSNorm 归一化避免训练不稳定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
class MLLMTextEncoder(nn.Module):
    def __init__(self):
        self.mllm = load_qwen2vl("Qwen2-VL-7B")
        self.mllm.requires_grad_(False)  # 冻结
        self.rms_norm = RMSNorm(mllm_hidden_dim)
        self.fc_align = nn.Linear(mllm_hidden_dim, dit_hidden_dim)

    def forward(self, text_prompt):
        hidden_states = self.mllm(text_prompt, output_hidden_states=True)
        text_features = hidden_states[-2]  # 倒数第二层
        text_features = self.rms_norm(text_features)  # 归一化
        return self.fc_align(text_features)  # 对齐到 DiT dim
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MLLM 编码器的优势：(1) 支持多语言输入；(2) 支持远长于 77 tokens 的文本；(3) VBench 验证有效。代价是推理时额外显存和延迟（7B 参数）。MovieGen 等也在探索更强的文本编码器。

涉及论文标题：
- EasyAnimate__A_High-Performance_Long_Video_Generation_Method_based_on_Transformer_Architecture
