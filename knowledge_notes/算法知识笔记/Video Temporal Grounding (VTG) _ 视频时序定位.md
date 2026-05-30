## Video Temporal Grounding (VTG) / 视频时序定位

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Video Temporal Grounding (VTG) 是视频理解领域的核心任务，目标是根据自然语言查询在视频中精确定位对应事件的起止时间戳。含两个子任务：(1) Moment Retrieval (MR) — 识别与查询对应的单个时间片段，输出 start/end 时间戳，评估指标 R1@t (t∈{0.3,0.5,0.7}) 和 mIoU；(2) Highlight Detection (HD) — 输出视频中所有与查询相关的显著时刻及其 saliency scores，评估指标 mAP 和 Hit@1。标准 benchmark: Charades-STA（日常活动）、ActivityNet-Captions（网络视频事件）、QVHighlights（MR+HD 联合评估）。GroundVTS 在两个变体上验证：GroundVTS-Q (Qwen2.5VL-7B) 和 GroundVTS-I (InternVL3.5-8B)，Charades-STA mIoU 达 50.1 (+18.4 over baseline)，QVHighlights HD mAP 达 52.5 (+20.6)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MR 任务在 Vid-LLM 中的指令格式（GroundVTS Grounding-FT 数据集）：
```
# 输入
"<video>At what point in the video did the following events occur: a person takes a book off a shelf. Output the start and end timestamps."
# 输出
"from 6.0s to 12.0s"
```
HD 任务输出格式："The highlights are: important from 96.0s to 98.0s; less important from 100.0s to 102.0s"。时间戳信息不放入 text prompt，模型仅依赖 visual token 的位置编码 (PE) 推断时间信息——因此 VTS 保留原始位置编码对 VTG 精度至关重要（消融实验中去除 PE 后 mIoU 从 50.1 降至 9.5）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VTG 在 Vid-LLM 中通过 instruction tuning 实现：在预训练 Vid-LLM 基础上，用 VTG 数据集 (Charades-STA, ActivityNet-Captions, QVHighlights) 进行 SFT。GroundVTS 使用自建 Grounding-FT (70K, ShareGPT format, 含多样化 prompt templates)。评估使用 llm-eval 或自定义脚本计算 IoU-based metrics。代码开源: https://github.com/Florence365/GroundVTS。

涉及论文标题：
- GroundVTS__Visual_Token_Sampling_in_Multimodal_Large_Language_Models_for_Video_Temporal_Grounding
- ReVisionLLM__Recursive_Vision-Language_Model_for_Temporal_Grounding_in_Hour-Long_Videos
- Temporal Preference Optimization of Large Multimodal Models

ReVisionLLM 将 VTG 扩展到小时级长视频，引入递归层次化处理：(1) 顶层 hierarchy 用 sparse temporal features 扫描全视频粗定位感兴趣区段，(2) 中间层聚焦预测区域进一步细化，(3) 底层用 dense temporal features 精确定位秒级起止时间。使用 LLM 输出熵的倒数作为置信度排序（替代 CLIP 相似度），ECE 从 0.62 降至 0.46。在 MAD (1200h movies) 和 VidChapters-7M (817K videos, up to 12h) 上建立 SOTA。代码: https://github.com/Tanveer81/ReVisionLLM。
