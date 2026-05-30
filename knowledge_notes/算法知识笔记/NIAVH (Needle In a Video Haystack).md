## NIAVH (Needle In a Video Haystack)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NIAVH (Needle In A Video Haystack) 是 VideoLLaMB 提出的长视频帧检索 benchmark，用于评估视频模型在超长视频中定位特定信息的能力。设计灵感来自 LLM 的 "Needle In A Haystack" (NIAH) 测试（将事实信息"针"插入长文档"干草堆"中测试检索能力），VideoLLaMB 将其扩展到视频多模态域。NIAVH 的 "haystack" 使用 Ego4D 数据集的 egocentric 视频拼接，长度从 1 到 320 秒；"needle" 支持三种模态：(1) 文本 needle——直接插入描述文本；(2) 图像 needle——使用 DALL-E 根据描述生成对应图像；(3) 视频 needle——使用 Sora 生成 1 秒短视频片段。Needle 插入到 haystack 的不同 depth（位置深度）和 length 组合中，评估模型回答"the young man seated on a cloud in the sky is doing what?" 的能力，LLM 对回答打分 1-10。NIAVH 区别于已有的 MM-NIAH（聚焦图像+文档混合 haystack）和 V-NIAH（纯合成视觉 benchmark），聚焦于流式视频 haystack 和多模态 needle 的组合。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
NIAVH benchmark 的构建和评估流程：

```
# === NIAVH 基准构建 ===
# Haystack: 从 Ego4D 拼接 egocentric 视频
haystack = concat(random_ego4d_clips(total_duration=320s))
# 帧率: 1 fps → 320 frames

# Needle 生成 (基于同一描述 "young man sitting on cloud reading book"):
needle_text = "A young man is sitting on a piece of cloud in the sky, reading a book."
needle_image = DALL-E(needle_text)          # 图像模态 needle
needle_video = Sora(needle_text)            # 视频模态 needle, 1秒

# Insertion: 在指定深度插入 needle
depth = 12  # needle 位置在 haystack 的 12/40 ≈ 30% 处
haystack_with_needle[depth] = needle_video  # 替换1帧

# === 评估 ===
question = "What is the young man seated on a cloud in the sky doing?"
answer = model(haystack_with_needle, question)
score = LLM_judge(answer, ground_truth="reading a book")  # 1-10

# === 测试矩阵 ===
# X轴: 视频长度 (1-320s)
# Y轴: needle 深度 (1-40 intervals)
# 每个 (length, depth) 组合评估一次 → 热力图 (Figure 3)
```

VideoLLaMB 在 NIAVH 上的表现（Table 6, Figure 3d）:
- 320s video, depth=12: VideoLLaMB score=5.73, 推理时间=4.21s
- 对比: MA-LMM 3.39, PLLaVA 1.82, LLaVA-NeXT-Video-DPO 1.72
- 显示 VideoLLaMB 的 retrieval 机制有效保留了早期信息

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
NIAVH 的实现依托 Ego4D、DALL-E 3、Sora 和 LLM judge。VideoLLaMB 开源代码中应包含 NIAVH 的构建和评估脚本。NIAVH 与同类 benchmark 的关系：(1) VNBench/VideoNIAH (Zhao et al. 2024, ICLR 2025)：独立的视频 NIAH 框架，支持 retrieval/ordering/counting 任务；(2) MM-NIAH (Wang et al. 2024)：多模态文档 haystack (1k-72k tokens)，支持文本+图像 needle；(3) MMNeedle (Wang et al. 2025, NAACL)：图像拼接 haystack，支持子图像检索；(4) V-NIAH (Zhang et al. 2024, LongVA)：纯合成视觉 NIAH benchmark。VideoLLaMB NIAVH 的独特优势在于支持视频 needle 模态（使用 Sora 生成）和流式视频 haystack，更贴近真实长视频理解场景。局限：(1) 当前仅单一 needle 类型和问题，缺乏多 needle、推理、counting 等多样化任务；(2) 仅用 Ego4D 作为 haystack 来源，可能引入领域偏见；(3) 评估依赖 LLM judge 的可靠性。

涉及论文标题：
- VideoLLaMB__Long-context_Video_Understanding_with_Recurrent_Memory_Bridges
