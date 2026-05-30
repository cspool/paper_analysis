## Modality Separation（模态分离：Deep vs Shallow）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Modality Separation是多模态模型中将不同模态（文本、图像）的计算路径分割到独立参数集合中的架构设计策略。LMFusion系统化探索了三种分离程度：(1) No Separation（无分离/Dense）——所有模态共享单一QKV/O/FFN参数，仅在U-Net有模态差异；(2) Shallow Separation（浅层/仅FFN分离）——FFN为模态特异性（FFN_text ≠ FFN_img），QKV和O共享，类似Mixture of Modality Experts (MoMa)；(3) Deep Separation（深层/FFN+Attention分离，LMFusion最终设计）——QKV、O和FFN均为模态特异性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
消融实验关键结论（训练250K步，0.03T text + 0.03T image data）：
- No Separation + r=1：语言能力严重退化（HellaSwag -15% initially, -7% persistent gap），即使降低文本lr（r=0.1）也只能缩小gap到-2%而image性能受损——存在trade-off
- Shallow Separation + r=0（仅FFN分离、文本冻结）：明显优于No Separation，但image generation性能受限——FFN仅处理attention后的特征变换，attention pattern的模态特异性更重要
- Deep Separation + r=0（FFN+Attention分离、文本冻结）：所有benchmarks最佳，且image性能甚至超越r=1的dense模型

核心洞察：attention层的模态分离比FFN层更重要——不同模态的attention pattern有本质差异（文本需causal/语义关联，图像需bidirectional/空间关联），共享attention参数时两种模式相互干扰（gradient conflict）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
设计可推广到将预训练text-only LLM适配到任何多模态任务的场景。关键实现要点：图像模块参数从预训练LLM权重初始化（获language knowledge transfer）；文本模块冻结；学习率解耦实现对不同模态的差异化训练速度控制。LLaVAFusion验证了相同范式可直接应用于已有VLM。

涉及论文标题：
- LMFusion: Adapting Pretrained Language Models for Multimodal Generation
