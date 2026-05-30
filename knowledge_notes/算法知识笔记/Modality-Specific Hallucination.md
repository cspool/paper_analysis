## Modality-Specific Hallucination

术语是什么？
Modality-Specific Hallucination（模态特定幻觉）是 OmniVinci 提出的概念，指单模态感知模型由于缺乏其他模态的互补信息而产生的系统性理解错误。具体表现为：纯视觉模型只能看到画面但听不到语音，可能将"深海探索视频"误解为仅关于"人类科技"（因为画面上有潜艇和设备，但语音讨论的是海洋生物）；纯音频模型只能听到语音但看不到画面，可能仅凭讨论内容将同样的视频误解为关于"地球内部"。

这一概念的学术价值在于：它从理论上论证了 omni-modal 联合理解的必要性——单模态感知不仅是不完整的，而且是**系统性错误的**，因为它缺乏跨模态纠错的机制。这与传统的 multi-modal fusion 有本质区别：传统 fusion 追求"更多信息→更好理解"，而 modality-specific hallucination 揭示了"信息不完整→**错误**理解"的定性差异。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
Modality-Specific Hallucination 是数据层面的概念，在 Data Engine 的 Step 2（跨模态纠错）中被显式解决。该概念指导了数据合成策略的设计：Data Engine 中的跨模态纠错 LLM 的核心任务就是识别和修正 modality-specific hallucination。在训练策略中，Explicit Omni-Modal Learning 通过提供"正确"的 omni-modal 标注，直接训练模型抵抗 modality-specific hallucination。

术语一般如何实现？如何使用？
该概念用于：(1) 指导 omni-modal 数据合成 pipeline 设计——必须包含跨模态纠错步骤；(2) 评估 omni-modal 模型——设计测试用例检验模型是否会在单模态信息不足时产生幻觉；(3) 论证 omni-modal 模型的必要性——纯 vision-language 或 audio-language 模型存在系统性的幻觉风险。OmniVinci 在 Omnibench (Image-Audio QA) 上 45.74（+OmniAlignNet）的表现验证了解决 modality-specific hallucination 的有效性。

涉及论文标题：
- OmniVinci Enhancing Architecture and Data for Omni-Modal Understanding LLM
