## PEFT (Parameter-Efficient Fine-Tuning)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Parameter-Efficient Fine-Tuning (PEFT) 是一类方法的统称，旨在仅更新预训练模型极小部分参数（通常 <1%）来适配下游任务，冻结绝大部分预训练权重。相比全参数微调（fine-tuning），PEFT 的核心优势：(1) 显著降低训练存储——仅需存储和更新少量 adapter 参数（如 LoRA 的 W_a/W_b）；(2) 减轻灾难性遗忘——冻结原权重使模型在通用能力上的退化更少；(3) 低数据场景表现更优——参数空间受限起到正则化效果；(4) 多任务部署灵活——同一 base model + 不同 adapter 可服务多种任务。主要技术类别包括：Adapter 类（在 Transformer 层间插入小型 bottleneck 模块，如 Houlsby 2019）、LoRA 类（低秩分解旁路）、BitFit（仅训 bias）、Prefix/Prompt Tuning（在输入前添加可训练虚拟 token）、IA3（缩放激活值）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# PEFT 通用流程（以 HuggingFace PEFT 库为例）
from peft import get_peft_model, LoraConfig

# 1. 配置 PEFT 方法
config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"])

# 2. 注入 adapter
model = get_peft_model(base_model, config)
# → 冻结 base_model 所有参数，仅 adapter 参数可训练
print(model.print_trainable_parameters())  # trainable: 1.0M / total: 125M (0.8%)

# 3. 标准训练（仅 adapter 参数更新）
trainer = Trainer(model=model, ...)
trainer.train()

# 4. 保存/加载（仅 adapter 权重，~几MB vs 全模型几GB）
model.save_pretrained("./lora_adapter")
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
主流实现：HuggingFace PEFT 库（GitHub: huggingface/peft），支持 LoRA/QLoRA/DoRA/AdaLoRA/IA3/PromptTuning/PrefixTuning/BitFit 等。安装：`pip install peft`。在 SSMLoRA 论文中，PEFT 作为实验对比框架——SSMLoRA 属于 LoRA 变体，通过引入 SSM 状态转移和稀疏插入进一步降低参数。SSMLoRA 在 GLUE 上仅 1.0M 参数（vs LoRA 1.3M）实现可比/更优性能。

涉及论文标题：
- SSMLoRA__Enhancing_Low-Rank_Adaptation_with_State_Space_Model

---
