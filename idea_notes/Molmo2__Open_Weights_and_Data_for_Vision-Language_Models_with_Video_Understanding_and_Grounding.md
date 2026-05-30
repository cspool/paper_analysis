## Molmo2__Open_Weights_and_Data_for_Vision-Language_Models_with_Video_Understanding_and_Grounding

- baseline方法是什么？
  Baseline分为三类：(1) **闭源API模型**（GPT-5, Gemini 2.5/3 Pro, Claude Sonnet 4.5）—— 强大的视频理解能力但不公开训练数据/权重/recipe，训练中可能使用自产VLM互蒸馏。(2) **开源权重模型**（Qwen3-VL, InternVL3.5, Eagle2.5, GLM-4.1V）—— 公开权重但不公开训练数据和recipe，数据集可能由GPT-4等闭源VLM生成，存在循环依赖。(3) **全开源模型**（PLM, LLaVA-Video, VideoChat-Flash）—— 公开权重+数据，但数据严重依赖闭源VLM蒸馏（GPT-4V生成captions/QA pairs），受闭源bias污染。
  这些baseline的共性问题：(a) **视频grounding能力严重缺失**——即使是闭源API模型，视频pointing F1仅2.2-20.0，视频tracking HOTA仅~30，无法进行pixel级时空定位；(b) **训练数据依赖闭源VLM蒸馏**——形成封闭循环，全开源社区无法自主迭代优化；(c) **视频caption dataset偏短/偏粗**——现有开源数据描述长度~75-547 words/video，细节不足，无法支撑细粒度视频理解；(d) **长视频理解弱**——开源模型在10min+视频上性能急剧下降（open models long QA avg仅56.2-60.4 vs closed 66.4-80.4）；(e) **缺少多图像grounding**——图像pointing仅限单图，不支持跨多图的pointing QA。

  Baseline（以Qwen3-VL-8B为例）全栈执行例子：
  - 算法层：视频→帧均匀采样→SigLIP/ViT逐帧编码→每帧生成vision tokens→LLM处理→生成文本答案。视频pointing几乎完全不可用（Molmo2-VP F1仅1.5）。视频tracking HOTA平均~16.5（Molmo2-Track across all categories）。video counting在高count区间(25-60)准确率0.0%。
  - 系统框架层：PyTorch + HuggingFace。无专用Serving框架修改。
  - 编译框架层：论文未明确说明。
  - kernel调度层：FlashAttention标准实现。
  - 硬件架构层：Nvidia H100/A100 GPU集群。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Molmo2通过以下设计系统性解决baseline缺陷：

  (1) **9个全新自建数据集（无需闭源VLM）** → 解决fully-open数据生态缺失。Molmo2-Cap用人工语音描述+转录+LLM润色+Molmo frame-level caption补充的pipeline，产出avg 924 words/video的dense captions（vs 基线75-547 words）。Molmo2-VideoPoint/Molmo2-VideoTrack用人工点标注+时戳记录，产出650k pointing queries和15k tracking queries。Molmo2-CapQA用自训练(基于Molmo2-Cap)的video captioner + LLM合成QA，避免闭源VLM依赖。Molmo2-AskModelAnything用人工提问+LLM辅助answer+人工refine。全流程仅使用text-only LLM辅助（非VLM），打破闭源VLM蒸馏循环。

  (2) **视频Grounding：spatio-temporal pointing + tracking + counting** → 解决视频grounding缺失。将2D image pointing (PixMo的<x,y>点格式)扩展到3D spatio-temporal域：引入timestamp + obj_id + (x,y)的压缩HTML-like格式。Pointing→counting：先point objects再count IDs（"point then count"策略，vs 直接预测count更优 Table 9a）。Tracking→point tracks with IDs：每帧标注object点+关联ID，支持HOTA等tracking指标（MOTA测量association accuracy）。相比API模型的bbox centerpoint策略（论文实验表明API模型无法生成准确point tracks），Molmo2的native point generation在video pointing F1达38.4（vs Gemini 3 Pro 20.0），video counting close accuracy 35.5（vs GPT-5 35.8），video tracking HOTA 57.5 across all categories（vs Gemini 3 Pro 29.1）。

  (3) **训练技术创新** → 解决训练效率和数据不平衡。(a) Packing + Message Trees：动态规划solver pool=48，平均3.8 examples/packed sequence，~15x训练效率；message tree允许同一video/image有多个annotations并存，用custom attention mask防止跨分支attention。(b) Token weighting：video caption weight=0.1, pointing weight=0.2, 其他 √(4/n)策略，防止少数长输出样本主导loss。(c) Bidirectional attention on vision tokens：vision tokens可互相attend（cross-frame/image），提升性能（Table 8b: 无bidir导致QA avg -0.4, Cap F1 -1.0）。(d) Pointing预训练：在pre-training阶段引入pointing数据，使SFT阶段不再需要学习basic pointing format，整体pointing更稳定。(e) SlowFast encoding：推理时用query-based frame selection + 3×3 slow/9×9 fast pooling，在~43% fewer tokens下匹配224 frame性能（Table 20）。

  对比baseline的全栈执行例子（Molmo2-8B, 128 frame video + tracking query）：
  - 算法层：视频→torchcodec 2fps抽128帧→SigLIP 2 ViT逐帧编码（384px, 27 layers）→取layer 3和layer 9 hidden states concat→Connector MH pooling 3×3 window→每帧81 visual tokens→128×81≈10,368 vision tokens。Vision tokens双向attention（可跨帧attend）→LLM 36层处理。Point output format：
    `<tracks coords="0.0 1 635 522;0.5 1 606 490;1.0 1 515 164">person in red</tracks>`
    `timestamp obj_id x y;...` 多个frames按timestamp排序，相同obj_id关联为track。64s超长视频→SFT 128帧处理；若384帧则需long-context SFT + context parallelism（8 GPUs Ulysses attention）。
  - 系统框架层：PyTorch + FSDP2 + SDPA（非FlashAttention, 因custom attention mask不兼容）+ torch.compile（静态shape）+ AMP bfloat16。HuggingFace model + vLLM serving。Multi-node 128 H100 SFT training。
  - 编译框架层：torch.compile用于LLM和ViT的静态编译优化吞吐量。论文未额外修改编译框架。
  - kernel调度层：PyTorch SDPA（Scaled Dot Product Attention），因需要custom attention mask（packing + message trees + bidir vision）无法使用FlashAttention。Context parallelism用Ulysses attention all-gather实现。
  - 硬件架构层：Nvidia H100 128节点SFT训练（8.1k GPU hours for 8B）。推理384 frames + greedy decoding on single H100。

  核心差异映射：
  | Baseline缺陷 | Molmo2解决方案 |
  |---|---|
  | 无视频grounding能力 | Spatio-temporal pointing format + native point generation（pointing F1 38.4 vs API 20.0） |
  | 训练数据依赖闭源VLM蒸馏 | 9个自建数据集，仅用text-only LLM辅助（+Molmo自用frame-captioner） |
  | 视频caption粗短 | 人工语音+transcribe+润色+frame visual detail补充，924 words/video |
  | 长视频理解弱 | Long-context SFT (384 frames, 36864 tokens) + SlowFast query-based inference |
  | 缺少多图grounding | Molmo2-MultiImagePoint (470k examples) + canonical label cross-image一致性算法 |
  | 训练效率低 | Packing (15x) + message trees + token weighting + bidir vision attention |
  | 开源模型使用闭源数据 | 全栈fully-open (权重+数据+代码)，可被community完全reproduce和extend |
