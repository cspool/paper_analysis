## Omni-Modal Data Engine

术语是什么？
Omni-Modal Data Engine 是 OmniVinci 提出的全模态数据合成流水线，用于从视频中自动生成高质量的 omni-modal（视觉+音频）对话数据。流水线分三步：(1) **独立模态 Captioning**：使用预训练视觉 captioning 模型（如 InternVL3）和音频 captioning 模型（如 Qwen2.5-Omni 的音频模块）分别对视频的视觉轨和音频轨生成独立标注；(2) **跨模态纠错与总结**：使用 LLM（如 Qwen3）接收视觉和音频两个独立 caption，基于双方信息进行纠错和综合，生成准确的 omni-modal joint caption；(3) **QA 合成**：使用 reasoning LLM（如 DeepSeek-R1）从 omni-modal caption 中合成带推理链的 QA 对。

核心动机是解决 **Modality-Specific Hallucination**（模态特定幻觉）：纯视觉 captioning 模型看不到音频信息，可能将深海探索视频误判为"人类科技"；纯音频 captioning 模型看不到视觉信息，可能仅凭语音内容误判为"地球内部"。跨模态纠错 LLM 综合两者信息后可生成正确的综合描述。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
Data Engine 是离线数据合成流水线，在模型训练前执行：
```
输入: 带音频轨的视频集合

for each 2-min video segment:
    # Step 1: 独立 captioning（可并行）
    vis_caption = VisualCaptioningModel(video_frames)
    aud_caption = AudioCaptioningModel(audio_track)

    # Step 2: 跨模态纠错
    prompt = f"Visual caption: {vis_caption}\nAudio caption: {aud_caption}\n\
              Please correct and summarize into a joint caption."
    joint_caption = LLM_Corrector(prompt)

    # Step 3: QA 合成
    qa_pairs = ReasoningLLM_Synthesize(joint_caption)
    # 生成 MCQ 或开放式 QA，含 reasoning trace

输出: omni-modal QA dataset (3.6M conversations)
```
最终生成的 omni-modal 数据占训练数据总量的 15%（omni QA 12% + omni captioning 3%），配合 modality-specific 数据（image 36%, sound 21%, speech 17%, video 11%）共 24M 样本。

术语一般如何实现？如何使用？
Data Engine 是离线 pipeline，各组件可独立替换：(1) Captioning 模型可根据场景替换为更强的模型（如 GPT-4o 替代 InternVL3）；(2) 跨模态纠错 LLM 可使用任何 instruction-tuned LLM；(3) QA 合成可用 reasoning LLM（DeepSeek-R1、Qwen3 等）生成带 CoT 的复杂问题。关键在于跨模态纠错步骤——直接拼接两个 caption 给 LLM 不够，需明确 prompt 指示 LLM 识别并解决两个 modal caption 的矛盾。Data Engine 生成的数据用于 Explicit Omni-Modal Learning，与 Implicit Learning（利用视频自带 audio track 的隐式监督）互补。

涉及论文标题：
- OmniVinci Enhancing Architecture and Data for Omni-Modal Understanding LLM
