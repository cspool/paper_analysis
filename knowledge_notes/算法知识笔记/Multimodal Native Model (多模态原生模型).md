## Multimodal Native Model (多模态原生模型)

术语解释
Multimodal Native Model 是 ARIA 提出的可量化定义：指一个单一模型在多模态输入（文本、代码、图像、视频）上具有强大的理解能力，且其性能匹配或超过同规模的单模态专用模型。核心特征是用户无需区分不同模态的输入，模型无缝处理和整合多模态信息。

术语是什么？
ARIA 给出了 multimodal native 的量化标准：一个 multimodal native model 在所有输入模态上的性能应匹配或超过类似容量的 modality-specialized models。这不同于简单的"多模态模型"（可能在不同模态上性能不均衡），也不同于通过 upcycling 从 dense model 转成 multimodal MoE 的方法。

关键设计原则：
1. **从零开始的多模态预训练（not upcycling）**：不依赖 dense checkpoint 初始化，language 和 multimodal 数据混合从头训练
2. **Modality-Generic Architecture**：不设计 modality-specific expert，所有 expert 对所有模态通用，expert specialization 在训练中自然涌现
3. **统一的 next-token prediction objective**：visual tokens 和 text tokens 使用相同的自回归 loss

从算法pipeline角度拆解术语：
ARIA 的 multimodal native pipeline：

```
Stage 1 - Language Foundation:
  MoE decoder only, 6.4T language tokens, 8K context
  → 建立通用知识和语言理解

Stage 2 - Multimodal Integration:
  Visual encoder + MoE decoder 联合训练
  1T language + 400B multimodal tokens
  → 多模态理解能力，维护语言能力

Stage 3 - Long Context Extension:
  33B tokens (69% long sequences)
  RoPE theta: 100K → 5M, context: 8K → 64K
  → 长视频/多页文档理解

Stage 4 - Instruction Following:
  20B high-quality QA data, LR annealing
  → 指令遵循和对齐
```

术语一般如何实现？如何使用？
- 开源实现：Aria (github.com/rhymes-ai/Aria), Apache 2.0
- 模型变体：Aria-Base-8K, Aria-Base-64K, Aria-Chat
- 推理：HuggingFace Transformers 或 vLLM，单 A100 80GB 即可 bf16 全模型推理
- 微调：支持 LoRA（单 GPU）和 Full parameter（8×A100 + DeepSpeed ZeRO）
- 对比 proprietary multimodal native models (GPT-4o, Gemini-1.5)，关键区别是训练配方的透明度

涉及论文标题：
- Aria An Open Multimodal Native Mixture-of-Experts Model

---
