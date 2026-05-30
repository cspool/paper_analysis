## Quantization Preprocessing via Restorative LoRA（基于恢复性LoRA的量化预处理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
基于恢复性 LoRA 的量化预处理是 PTQ1.61 提出的新范式：量化前用轻量级 LoRA 微调，目标是将预训练模型中散乱的显著权重分布改写为行集中模式，使模型更适配逐通道 PTQ。动机：逐通道 PTQ 按行分配量化参数，但预训练模型显著权重呈散乱分布，行内方差大，同参数难以同时覆盖高/低幅值元素。LoRA 低秩补偿 ΔW=BA 被吸收进权重，将显著权重推向集中行模式。与 QLoRA/QA-LoRA 等后量化 PEFT 区别：(1) 目的——优化量化而非下游任务；(2) 无需存储额外 FP 矩阵；(3) 用预训练数据（RedPajama）而非下游微调数据。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 输入: M_FP (FP16), RedPajama, rank=64, steps=20K
M_q = extremely_low_bit_quantize(M_FP)
for layer in M_q:
    layer.lora_A = Linear(d_in, 64); layer.lora_B = Linear(64, d_out)
for step in range(20000):
    x = next_batch(RedPajama)
    loss = MSE(M_q(x), M_FP(x))  # teacher-student recovery
    update(lora_A, lora_B)
W_preprocessed = W + lora_B @ lora_A  # 融合 LoRA
W_final_q = PTQ1.61(W_preprocessed)   # 在预处理模型上 PTQ
```
LLaMA-13B WikiText2 PPL：直接量化 14.22 vs 预处理后量化 9.67（提升 4.55 PPL）。通用性：GPTQ/OmniQuant 等 baseline 使用预处理同样提升（Figure 5）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
资源：LLaMA-7B <1.2h（单 A100）。局限性：增加预处理时间（总 PTQ 2h vs OmniQuant 1.1h）；FP16 模型本身性能轻微退化。适合极低位 PTQ 性能瓶颈场景，作为可选增强模块。开源：https://github.com/zjq0455/PTQ1.61。

涉及论文标题：
- PTQ1.61 Push the Real Limit of Extremely Low-Bit Post-Training Quantization
- LoRA: Low-Rank Adaptation of Large Language Models

---
