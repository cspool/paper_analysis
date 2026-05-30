## GroundVTS__Visual_Token_Sampling_in_Multimodal_Large_Language_Models_for_Video_Temporal_Grounding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：GroundVTS 是一个面向 Video Temporal Grounding (VTG) 的 Vid-LLM 框架，核心创新是 **Visual Token Sampling (VTS)** 模块——在 visual encoder 和 multimodal projector 之后、LLM 输入之前，对 visual token 进行 query-guided 细粒度采样。VTS 包含两个步骤：(1) **Query-Guided Token Scoring** —— 将 visual embeddings V 和 text query embeddings Q 通过可学习投影矩阵 W_v, W_q 映射到低维子空间 D_r，计算温度缩放的点积相似度 w = softmax(V'q'^T / τ)，得到每个 visual token 的 query 相关性权重；(2) **Differentiable Top-K Selection** —— 基于 Gumbel-Softmax 松弛 + Straight-Through Estimator (STE) 实现可微分的 top-K 选择，z_i = softmax((log w_i + g_i)/τ_g)，forward 使用 hard top-K mask，backward 通过连续松弛 z_i 传播梯度。最终 selected tokens 通过 MLP + 重归一化权重获得。采样比例 ρ ∈ (0,1] 控制保留的 visual token 数量 K = ⌈ρ·N_v⌉。保留原始位置编码（仅 mask 掉未选中 token 的位置编码）以维持时间一致性。训练采用三阶段渐进式优化：(Stage 1) VTS Warm-up —— 冻结 LLM，仅训练 VTS 模块；(Stage 2) Joint LoRA Adaptation —— LoRA (rank=8, α=16) 微调 LLM + VTS + Projector，使用 LLaVA-Video-178K；(Stage 3) Grounding Fine-tuning —— 继续 LoRA 微调，使用自建 Grounding-FT 70K 样本。两个模型变体：GroundVTS-Q（基于 Qwen2.5VL-7B, 2 FPS, ρ=0.5, D_r=512）和 GroundVTS-I（基于 InternVL3.5-8B, 16 frames/video, ρ=0.5, D_r=128）。

  实验比较：(a) Moment Retrieval —— Charades-STA 和 ActivityNet-Captions 上对比 LLaVA-OV、TimeChat、VTimeLLM、Momentor、HawkEye、ChatVTG、NumPro、LLaVA-ST 等 SOTA 方法，以及 Qwen2.5VL-7B 和 InternVL3.5-8B 的微调 baseline；(b) QVHighlights —— MR + HD 对比 SeViLA、UniVTG、VTG-LLM、TimeChat、NumPro 等；(c) Out-of-Distribution —— NExT-GQA（零样本 grounded VQA）、DiDeMo（OOD moment retrieval）、LongVideoBench（长视频理解迁移）；(d) Visual Token Density —— ρ 从 0.1 到 1.0 的稳健性分析；(e) 消融实验 —— 训练阶段组合、采样策略（Token-Level vs Frame-Level vs Uniform vs Random）、位置编码有无；(f) 额外消融 —— 参数自由投影 vs 可学习投影、数据集组成；(g) MVBench 通用 VQA 能力保持验证。

- 硬件平台是什么，配置是什么。
  训练使用 GPU，batch_size=2 per GPU, gradient_accumulation=4，优化器 AdamW (β1=0.9, β2=0.999)。具体 GPU 型号论文未明确说明。推理评估使用标准 PyTorch + HuggingFace Transformers。

- 模型是什么。数据集和bench分别是什么。
  模型：GroundVTS-Q（基于 Qwen2.5VL-7B, total 8.32B params, trainable 153.0M）和 GroundVTS-I（基于 InternVL3.5-8B, total 8.56B params, trainable 145.2M）。VTS 模块参数 ~29-35M，Projector ~34-45M，LoRA (rank=8, α=16, dropout=0.05) ~77-79M。
  训练数据集：(1) LLaVA-Video-178K —— 大规模视频多模态数据集（Stage 1 和 Stage 2）；(2) Grounding-FT —— 自建 VTG 指令微调数据集，聚合 Charades-STA、QVHighlights、ActivityNet-Captions 训练集，70K 标注视频-查询对，统一为 ShareGPT instruction-response 格式，含 MR 和 HD 两种任务。
  Benchmarks：(1) Charades-STA (R1@0.3/0.5/0.7 + mIoU)；(2) ActivityNet-Captions (R1@0.3/0.5/0.7 + mIoU)；(3) QVHighlights (MR: R1@0.5/0.7; HD: mAP + Hit@1)；(4) NExT-GQA (mIoU, mIoP, IoU@0.5, IoP@0.5, Acc@GQA)；(5) DiDeMo (R1@0.3/0.5, mIoU)；(6) LongVideoBench (Acc by duration)；(7) MVBench (20 子任务通用 VQA)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/Florence365/GroundVTS

  算法 pipeline 伪代码（VTS 核心 + 三阶段训练）：

  ```
  # ==========================================
  # GroundVTS 推理流程 (VTS 核心)
  # ==========================================
  # 输入: 视频 V = {F_t}_{t=1}^T, 文本查询 text_query
  # 超参数: ρ (sampling ratio), τ (temperature), τ_g (Gumbel temp), D_r (hidden dim)

  # 步骤 1: Vision Encoding + Projection
  H_v = VisionEncoder({F_t})  # T frames → N_v visual tokens ∈ R^{N_v × D_v}
  V = Projector(H_v)          # MLP → ∈ R^{N_v × D}, D 对齐 LLM embedding 维度

  # 步骤 2: Text Tokenization
  Q = TextTokenizer(text_query)  # ∈ R^{N_t × D}

  # 步骤 3: VTS - Query-Guided Token Scoring (Eq.2-3)
  V' = W_v @ V              # W_v ∈ R^{D × D_r}, V' ∈ R^{N_v × D_r}
  q' = W_q @ mean(Q, dim=0) # W_q ∈ R^{D × D_r}, q' ∈ R^{D_r}
  w = softmax(V' @ q' / τ)  # ∈ R^{N_v}, query-token relevance weights

  # 步骤 4: VTS - Differentiable Top-K Selection (Eq.4-6)
  K = ceil(ρ * N_v)
  g_i ~ Gumbel(0, 1)
  z_i = exp((log w_i + g_i) / τ_g) / Σ_j exp((log w_j + g_j) / τ_g)  # soft
  I_K = TopK_indices(w, K)
  z_i^hard = 1 if i ∈ I_K else 0                                    # hard (forward)
  \tilde{z}_i = z_i^hard + z_i - stopgrad(z_i)                       # STE

  # 步骤 5: Weighted Token Re-encoding (Eq.7)
  \hat{w}_i = exp(w_i/τ') · \tilde{z}_i / Σ_j exp(w_j/τ') · \tilde{z}_j
  \tilde{v}_i = \hat{w}_i · MLP(v_i)

  # 步骤 6: Position Encoding + LLM Input
  PE_selected = PE_original[I_K]  # 保留 dense sampling 原始位置编码
  input_seq = concat([\tilde{V} + PE_selected, Q])
  answer = LLM.generate(input_seq)

  # ==========================================
  # 三阶段训练流程
  # ==========================================
  # Stage 1: VTS Warm-up (lr=1e-5, 1 epoch, LLaVA-Video-178K)
  freeze(LLM, VisionEncoder, Projector)
  trainable = [W_v, W_q, MLP_vts]  # 仅 VTS

  # Stage 2: Joint LoRA Adaptation (lr=2e-4, 2 epochs, LLaVA-Video-178K)
  unfreeze(Projector)  # VTS + Projector + LoRA(LLM)
  # LoRA: rank=8, α=16, dropout=0.05

  # Stage 3: Grounding Fine-tuning (lr=1e-4, 3 epochs, Grounding-FT 70K)
  # 同 Stage 2 冻结配置，使用 MR + HD instruction-style QA pairs
  ```

  关键张量维度：
  - Visual tokens N_v: QwenVL @ 2 FPS → 动态; InternVL @ 16 frames → 固定
  - Token 投影: V' ∈ R^{N_v × D_r}, q' ∈ R^{D_r}; D_r=512(GroundVTS-Q), D_r=128(GroundVTS-I)
  - VTS 采样: K = ⌈ρ·N_v⌉, ρ=0.5 即保留 50% visual tokens
  - 非均匀 token 分布: 高 query 相关性区域 dense sampling, 低相关性区域 sparse/zero
  - Trainable params: VTS ~29-35M + Projector ~34-45M + LoRA ~77-79M = 145-153M total
  - 训练配置: batch_size=2/GPU, grad_acc=4, AdamW (β1=0.9, β2=0.999)
