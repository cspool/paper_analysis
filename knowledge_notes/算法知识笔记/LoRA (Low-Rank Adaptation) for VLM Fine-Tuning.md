## LoRA (Low-Rank Adaptation) for VLM Fine-Tuning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
LoRA (Low-Rank Adaptation) 是 Hu et al. (ICLR 2022) 提出的参数高效微调（PEFT）方法。冻结预训练模型权重 W₀ ∈ R^{d×k}，注入可训练低秩分解 B·A（B∈R^{d×r}, A∈R^{r×k}, r≪min(d,k)），前向计算 h = W₀x + BAx。推理时 BA 融入 W₀ 无额外延迟。相比 full fine-tuning，LoRA 将可训练参数减少 10000×，GPU memory 减少 3×。

从算法pipeline角度拆解术语：
Mordal 中使用 LoRA 对 LLM 进行 fine-tuning（当 `freeze_llm=False` 时）：
```
# VLM alignment with LoRA:
for each training step:
    img_emb = VE(images)               # frozen
    aligned_emb = Projector(img_emb)   # trainable (from scratch)
    text_emb = TokenEmbed(text)
    all_emb = concat([aligned_emb, text_emb])
    for layer in LLM:
        h = Attention(QKV_proj(x))     # QKV: W₀x + BAx (LoRA)
        h = FFN(gate_up_down(x))       # FFN: W₀x + BAx (LoRA)
    loss = CrossEntropy(output, labels)
# 仅更新 Projector 参数 + 所有 LoRA 的 B,A 矩阵
```
Mordal 在 LLM 的 Q/K/V/O projection 和 FFN 层注入 LoRA（通过 PEFT 库配置），仅训练 LoRA 参数 + 从头训练 Feature Projector。

术语一般如何实现？如何使用？
HuggingFace PEFT 库（https://github.com/huggingface/peft）实现。典型配置：`LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"])`。Mordal 通过 `vlm_kwargs={'freeze_llm': False}` 启用 LoRA fine-tuning。对于 VLM alignment，LoRA 在 7B LLM 上仅增加 ~10M 可训练参数（vs 7B full），大幅降低每候选计算成本。局限：某些任务 full fine-tuning 仍优于 LoRA；LoRA rank 和 target modules 需经验性选择。

涉及论文标题：
- Mordal: Automated Pretrained Model Selection for Vision Language Models
