## LLaVA-Mini__Efficient_Image_and_Video_Large_Multimodal_Models_with_One_Vision_Token

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：LLaVA-Mini —— 引入 query-based compression 和 modality pre-fusion 两个模块，将输入 LLM backbone 的 vision token 从 576 压缩到 1 个（压缩率 0.17%），同时保持与 LLaVA-v1.5 可比的多模态理解性能。
  核心设计：(1) Query-based Compression —— 引入 C×C 个可学习压缩 query Q^v，通过 cross-attention 与全部 N^2 个 vision token 交互，使用 2D sinusoidal positional encoding 保留空间信息，产生 C^2 个压缩后 vision token Ĥ^v。C 默认设为 1（标准分辨率）或 8（高分辨率 HD 模式）。(2) Modality Pre-fusion —— 在 LLM backbone 之前，用 N_fusion=4 个与 LLM 同构的 Transformer decoder 块，将全部 vision token 和 text token 拼接后输入，提取文本位置的输出作为 fusion token Ĥ^q，提前将视觉信息融入文本表示。
  训练两阶段：(1) Stage 1 Vision-Language Pretraining：仅训练 projection layer，冻结 vision encoder 和 LLM，使用 558K caption 数据。(2) Stage 2 Instruction Tuning：引入 compression + pre-fusion 模块，除 vision encoder 外全可训练，使用 665K instruction 数据。增强变体额外加入 100K Video-ChatGPT 视频数据及开源数据共 3M samples。

  实验比较：
  (a) 图像理解 —— 11 benchmarks (VQA-v2, GQA, VisWiz, SciQA-IMG, TextVQA, POPE, MME, MMBench, SEED-Bench, LLaVA-Bench-in-the-Wild, MM-Vet)，对比 LLaVA-v1.5 以及 BLIP-2, InstructBLIP, IDEFICS, Qwen-VL, SPHINX, mPLUG-Owl2 等，还有 token 压缩方法 MQT-LLaVA, PruMerge/PruMerge++, LLaMA-VID, VoCo-LLaMA, TokenPacker。
  (b) 视频理解 —— 5 video QA benchmarks (MSVD-QA, MSRVTT-QA, ActivityNet-QA) + video-based generative performance benchmark + MVBench 20 子任务，对比 Video-ChatGPT, Video-LLaVA, Video-LLaMA, LLaMA-VID 等。
  (c) 长视频 —— MLVU 和 EgoSchema，对比 MovieChat, MA-LMM, TimeChat 等。
  (d) 效率分析 —— FLOPs (calflops) + latency (A100, no engineering acceleration) + VRAM usage，对比 LLaVA-v1.5。
  (e) 消融实验 —— Modality pre-fusion 层数 (0/1/2/3/4)、vision token 数量 (1/4/16/64/144/576)、query-based compression vs average pooling、compression 与 pre-fusion 在 LLM 内外执行对比、纯 pre-fusion 无 compression 效果。
  (f) 跨硬件效率 —— RTX 3090 (24G), A100 (40G), A800 (80G) 延迟测试。
  (g) 各组件 FLOPs 分解 —— Vision Encoder / Projection / Compression / Pre-fusion / LLM。

- 硬件平台是什么，配置是什么。
  训练：8 NVIDIA A800 GPU。Batch size 256，Stage 1 1 epoch，Stage 2 2 epochs。Optimizer AdamW，learning rate 1e-3 (Stage 1 projection) / 1e-4 (Stage 2 LLM)，cosine decay schedule，warmup ratio 0.03。
  推理延迟测试：NVIDIA A100 (40G)，无工程加速技术。跨硬件延迟测试额外含 RTX 3090 (24G) 和 A800 (80G)。
  VRAM 测试：RTX 3090 (24G) 处理 3 小时视频。
