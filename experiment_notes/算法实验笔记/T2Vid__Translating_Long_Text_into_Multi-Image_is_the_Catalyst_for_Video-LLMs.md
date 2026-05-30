## T2Vid__Translating_Long_Text_into_Multi-Image_is_the_Catalyst_for_Video-LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出 Sparrow 数据增强方法，将纯文本 instruction 数据转化为类视频的多图像序列（text-to-image synthesis），然后与真实视频数据混合训练视频-LLM。核心流程：(1) 从现有文本 instruction 数据集中取 (long-context, instruction, answer) 三元组；(2) 按词数（~115词/段）用 NLTK 将 long-context 分割为多段；(3) 每段用 Pillow ImageFont 渲染为 448×448 白底黑字图像（20pt Arial, 20px margin）；(4) 将合成图像序列与真实视频帧混合，统一格式后 fine-tune image-LLM。该方法旨在解决视频训练数据 instruction diversity 不足导致的数据效率低下问题。
  - 实验比较：
    - 数据缩放实验：不同数据量（0/10K/30K/60K/100K）下，纯视频数据 vs Sparrow 混合数据（视频:合成=2:1）的性能对比
    - 消融实验：30K 规模下，纯 ShareGemini、纯 Video-ChatGPT、各半混合、Sparrow（Video+合成）、Video+纯文本 五种数据组合的性能对比
    - 主流方法对比：在 Video-MME 上与 proprietary models (GPT-4V, GPT-4o, Gemini 1.5 Pro) 和 open-source video-LLMs (VideoChat2, VideoLLaMA 2, VITA, Kangaroo 等) 对比
    - 长视频理解：LongVideoBench、MLVU-M、Video-MME-Long 上评估长视频理解能力
    - 帧数泛化：推理时从 24 帧扩展到 48/128/256 帧的性能变化
    - 细粒度任务分解：Video-MME 上按 Perception (OCR/计数/物体识别等)、Cognition (时序推理/空间推理等)、Information Synopsis 三类任务性能分解

- 硬件平台是什么，配置是什么。
  - 论文未明确说明训练使用的具体 GPU 型号。给出 GPU hours 作为效率度量：全量 200K 视频数据训练需 276.8 GPU hours，Sparrow 30K 混合数据仅需 33.6 GPU hours，效率提升 8.2×。推测使用多卡训练（batch 含多视频帧）。环境依赖：conda Python 3.9 + PyTorch + Flash-Attention 2。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mini-InternVL-Chat-4B-V1.5（简称 InternVL，3.8B，最多 13 子图 patch，每子图 256 visual tokens）和 MiniCPM-Llama3-8B-V2.5（简称 MiniCPM-8B，最多 10 patch，每 patch 96 visual tokens）。均为预训练 image-LLM，通过 fine-tuning 扩展到视频理解。InternVL 训练时冻结 vision encoder，其余参数全量训练（lr=5e-6）。MiniCPM-8B 全量训练。训练时关闭动态分辨率 patchifying 选项。
  - 数据集：
    - ShareGemini-Webvid-core100k：100K 视频-描述对，源自 WebVid（开源短视频 <30s），caption 由 Gemini-1.5-Pro API 标注，经聚类去重
    - Video-ChatGPT：100K 视频-instruction 对，源自 ActivityNet，含视频摘要/内容问答/创造性生成三类 instruction，半自动标注（人工精炼 + GPT-3.5 辅助）
    - 文本数据源：LongAlpaca（5K）和 LongQLora（5K），各取 5K 样本合成
  - Benchmarks：
    - Video-MME：综合视频-LLM 评估，含短视频 (<2min)、中等 (4-15min)、长视频 (30-60min)，手动收集标注
    - MVBench：20 个视频任务，覆盖感知和认知（场景转换、情节推理等）
    - TempCompass：时序理解评估（动作、速度、属性变化），MCQ 格式
    - LongVideoBench：长上下文交错视频-语言理解
    - MLVU-M：多任务长视频理解

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源仓库：https://github.com/VITA-MLLM/Sparrow，arXiv: 2411.19951
  - 合成数据集：https://huggingface.co/datasets/xjtupanda/Sparrow-Synthetic
  - 环境搭建：`conda create -n sparrow python=3.9` → `pip install -r requirements.txt` → `pip install -U flash-attn --no-build-isolation`

  **算法 Pipeline 详解**：

  **Step 0 — 模型架构（标准 MLLM）**：
  ```
  输入视频 V 下采样到 T 帧: F = {f_i}_{i=1}^T
  逐帧特征提取: E = {E_i}_{i=1}^T = ViT(F),  E_i ∈ R^{(H×W)×C}
  投影到 LLM 空间: Ê = Proj(E),  Proj 为 MLP
  与文本 token 拼接: [w_V; w_T] → LLM → 自回归生成答案
  ```

  **Step 1 — Sparrow 数据合成**：
  ```
  输入: (long_context, instruction, answer) 三元组

  # 1. 按词数分割 long_context
  segments = nltk.word_tokenize(long_context)
  chunks = [segments[i:i+115] for i in range(0, len(segments), 115)]

  # 2. 每个 chunk 渲染为图像
  for chunk in chunks:
      text = " ".join(chunk)
      img = Image.new('RGB', (448, 448), 'white')
      draw = ImageDraw.Draw(img)
      font = ImageFont.truetype('arial.ttf', 20)
      # 逐行绘制，左右 margin 20px
      y = 20
      for line in wrap_text(text, width=408):  # 408 = 448 - 2*20
          draw.text((20, y), line, fill='black', font=font)
          y += font_height
      synthetic_frames.append(img)

  # 3. 输出: (synthetic_frames, instruction, answer) → 类 video 格式
  ```

  **Step 2 — 混合训练**：
  ```
  # 每个 batch 以 2:1 比例采样
  real_sample  = (video_frames, instruction, answer)  # 来自 ShareGemini/Video-ChatGPT
  syn_sample   = (synthetic_frames, instruction, answer)  # 来自 Sparrow

  # 视频帧处理（InternVL, max 64 frames, FPS=1）
  if len(real_frames) > max_frames:
      real_frames = uniform_downsample(real_frames, max_frames)

  # Vision encoder 编码（InternVL: 冻住, MiniCPM: 训练）
  real_visual_tokens = ViT(real_frames)  # shape: [T, H*W, C]
  syn_visual_tokens  = ViT(syn_frames)
  # Projector 投影
  real_embeds = Proj(real_visual_tokens)
  syn_embeds  = Proj(syn_visual_tokens)

  # 与文本拼接送入 LLM
  input_ids = concat([vision_embeds, text_embeds])
  logits = LLM(input_ids)
  loss = -log P(answer | video_frames, instruction)
  ```

  **关键张量维度**：
  - InternVL-4B: 每子图 → 256 visual tokens × 13 tiles max; 训练时关闭 patchify，video max 64 frames; 每帧 token 数 = 256（单 tile 模式）
  - MiniCPM-8B: 每子图 → 96 visual tokens × 10 tiles max; video max 24 frames; 每帧 token 数 = 96（单 tile 模式）
  - 合成图像: vision encoder 同样处理，448×448 白底黑字图像 → 经 ViT patch embedding → visual tokens
  - LLM backbone: InternVL 用 InternLM2, MiniCPM 用 LLaMA3-8B

  **训练配置**：
  - 学习率：5e-6，全量端到端训练（InternVL 冻 vision encoder）
  - 数据混合：视频数据 (ShareGemini:Video-ChatGPT = 1:1) + 合成数据 (LongAlpaca:LongQLora = 1:1)，视频:合成 = 2:1
  - 长视频推理：24 帧训练，48/128/256 帧推理可扩展（但超 LLM context 则性能崩溃）

  **效率对比**：
  | 配置 | 样本数 | GPU hours | Video-MME Overall |
  |------|--------|-----------|-------------------|
  | Full video data | 200K | 276.8 | 56.3 |
  | Sparrow 30K | 30K (20K video + 10K syn) | 33.6 | 56.7 |
  | 效率提升 | 15% 样本 | 8.2× faster | +0.4 pts |
