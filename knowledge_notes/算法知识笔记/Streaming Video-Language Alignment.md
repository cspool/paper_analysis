## Streaming Video-Language Alignment

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Streaming Video-Language Alignment（流式视频-语言对齐）是 LiveStar 论文提出的训练范式，旨在替代传统 EOS-based 在线 Video-LLM 的训练方式。传统方法（VideoLLM-online 等）训练模型在非响应帧输出 EOS token，破坏了预训练的视觉-语言对齐（vision-language alignment，即每个视觉输入应对齐有意义的语言输出）。流式视频-语言对齐的核心创新是将训练目标重构为：对每个语义片段 C_k = {t_m, ..., t_n}，所有帧共享相同的语义字幕，训练目标为 `max P([Cap^k] | [Ctx^{<t_i}], [Frm^{t_i}])`（而非 `max P(EOS | ...)`）。这通过 SCAM（Streaming Causal Attention Masks）实现，保证每个视觉帧始终与有意义的语言内容对齐，与 pretraining 的 image-text pair 对齐范式一致。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
两阶段训练 pipeline：
```
Phase I: Temporal Alignment Pretraining (63K samples)
  ActivityNet Captions (9K) + Shot2Story (33K) + Ego4D (20K) + MVBench (1K)
  → 构建交错帧-字幕序列
  → SCAM attention mask
  → 标准 autoregressive cross-entropy loss (仅assistant tokens)

Phase II: Multi-Task Online Adaptation (20K OmniStar samples)
  5 tasks: RNG / OTG / FDQ / COQ / MIQ
  → Task-specific adapters
  → Simultaneous multi-objective alignment
```

全微调配置：Vision Encoder (InternViT) 冻结，MLP Projector + LLM (InternLM2.5-7B) 可训练。AdamW (lr=4×10⁻⁵, β1=0.9, β2=0.999, weight decay=0.05)，cosine LR schedule with warmup ratio=0.03，effective batch size=32（per-device bs=1 × gradient accumulation 4 × 8 GPUs）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖于：(1) 细粒度语义片段标注 — 需要将视频划分为带时间戳的字幕段落（OmniStar 使用 semi-automated pipeline 进行 temporal dense annotation），每个段落的帧属于同一语义clip；(2) 释义池 — 为每个语义clip准备 M 个 paraphrased captions（M=1 默认，M=3 时 SemCor +1.57% 但 TimDiff +3.14%）；(3) SCAM mask 生成 — 需要在 DataLoader 中实时构建，随 batch 一起送入模型。训练数据格式为 chat-style interleaved format。

涉及论文标题：
- LiveStar__Live_Streaming_Assistant_for_Real-World_Online_Video_Understanding
