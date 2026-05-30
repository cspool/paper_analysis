## Machine_Mental_Imagery__Empower_Multimodal_Reasoning_with_Latent_Visual_Tokens

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：Mirage —— 在两阶段微调范式下，让 VLM 在文本 token 之间插入紧凑的 latent visual tokens（压缩后的隐空间视觉特征向量），替代显式图像生成，实现多模态推理链。核心设计：
  (1) 数据合成：对每个任务用 task-specific tool 生成 helper image I（如 VSP 用 OpenAI Gym 渲染 annotated map，SAT 用 CogVideoX-5B 生成场景帧），再用大推理 VLM (Qwen2.5-VL 32B) 基于 x + I + y 生成 interleaved reasoning chain o = o_pre ⊕ I ⊕ o_post，构成训练集 D = {(x, I, o, y)}。
  (2) Stage 1 — Joint Supervision for Latent Grounding：将 helper image I 输入 VLM 得到 patch-level features {e_1,...,e_n}，通过 average pooling 压缩为 k 个 salient vectors {ê_1,...,ê_k}。训练目标 L_1 = L_visual + γ·L_text，其中 L_visual = ℓ_cos(ê_j, g_θ(o_pre, ê_{1:j-1})) 用 cosine similarity 强制隐层状态逼近压缩后的 visual embedding，L_text 为左右两段文本的标准 cross-entropy。γ=0.1 控制 visual alignment loss 权重。
  (3) Stage 2 — Text-Only Supervision with Latent Relaxation：移除 L_visual，仅保留文本 CE loss。模型自回归生成自己的 latent tokens {e_i} = f_θ(x, o_pre, e_{<i})，梯度通过 o_post 的文本 CE loss 反向传播到这 k 个连续 latent embeddings，使其在 visual subspace 内自适应优化，不再强制匹配固定的 compressed image embeddings。
  (4) Stage 3 — Reinforcement Learning (GRPO)：使用 VERL 框架 + GRPO，优化文本 token 概率同时允许梯度流经 latent tokens。奖励 = σ_f·r_format + σ_c·r_correct（σ_f=0.1, σ_c=0.9），KL 正则系数 λ_kl=0.01。rollout num=5, mini batch=8。

  实验比较：
  (a) VSP（Spatial Reasoning + Spatial Planning, Level 3-6）—— 对比 Zero-Shot, Direct SFT, CoT SFT, GRPO, CoT SFT+GRPO, Anole, MVoT。Mirage (Direct) 在 Spatial Reasoning 86%、Spatial Planning 76%；+GRPO 后提升至 89% 和 60%。
  (b) COMT（Math Geometry, 200 test）、BLINK-Jigsaw（150 test）、SAT（Synthetic GoalAim/ObjM + Real, 500 test）—— 对比 Zero-Shot, Direct SFT, CoT SFT, GRPO, SFT+GRPO。Mirage 在 COMT 77%, Jigsaw 88%, SAT Avg 98% (Synthetic) / 72% (Real)。
  (c) 模型规模泛化 —— Qwen2.5-VL 3B 上重复 COMT/Jigsaw/SAT 实验，Mirage 相比 text-only baseline 在 Jigsaw 上 +5%, SAT Real 上 +10%。
  (d) 消融实验 —— (i) 训练阶段：w/o Stage 1：VSP Spatial Planning 降至 52% Avg；w/o Stage 2 (仅 Stage 1)：降至 21% Avg。(ii) latent token size k ∈ {2,4,6,8}：k=4 和 k=6 最优（87-88%），k=8 骤降至 75%。(iii) loss coefficient γ ∈ {0.1,0.5,1}：γ=0.1 最优 (87%)，增大 γ 相当于弱化 visual alignment。(iv) helper image as prior 数据质量验证：配合 helper image 直接输入 fine-tuned 模型可接近 100% 准确率。
  (e) 潜变量行为分析 —— t-SNE 可视化 100 samples 的 latent tokens 与 text/image embeddings：latent tokens 聚集在 visual cluster 外侧，与 text distribution 明显分离，验证 Stage 2 在 visual submanifold 内的灵活偏移。

- 硬件平台是什么，配置是什么。
  训练：单个 NVIDIA H100 GPU。Stage 1 约 3.5 hours，Stage 2 约 7.2 hours（VSP Spatial Reasoning 为例）。text-only CoT SFT 约 5.5 hours 作为参考。
  推理：论文未明确说明推理平台，但基于 Qwen2.5-VL 7B 规模，可在单 H100/A100 GPU 推理。

- 模型是什么。数据集和bench分别是什么。
  模型：Base VLM — Qwen2.5-VL-7B-Instruct（默认），小模型验证使用 Qwen2.5-VL-3B-Instruct。Vision encoder 部分冻结不训练。Loss weight γ=0.1，latent token size k=4 (default)。
  数据生成模型：外部推理 VLM — Qwen2.5-VL-32B 生成 textual thoughts。Helper image 生成 — OpenAI Gym (VSP map rendering), CogVideoX-5B (SAT video generation, sampling 9 frames + VLM selects most informative frame)。
  数据集配置 (Tab 8)：
  - VSP Spatial Reasoning: #SFT 3000, #RL 2000, #Test 400
  - VSP Spatial Planning: #SFT 3000, #RL 2000, #Test 400
  - BLINK Jigsaw: #SFT 1000, #RL 2000, #Test 150
  - SAT (GoalAim + ObjM): #SFT 1000, #RL 2000, #Test 500
  - COMT Math Geometry: #SFT 820, #RL -, #Test 200
  Benchmarks: VSP (spatial planning + spatial reasoning sub-tasks, binary→three-way extended), BLINK-Jigsaw (visual extrapolation), SAT (static + dynamic spatial relations, GoalAim/ObjM subtasks, Synthetic + Real split), COMT (Mathematical Geometry subset)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/UMass-Embodied-AGI/Mirage
  代码已开源，论文给出了 Method 章节的完整数学公式推导。

  算法 pipeline 执行例子（以 VSP Spatial Reasoning 为例，k=4 latent tokens, Qwen2.5-VL 7B）：
  ```
  # === Stage 1: Joint Supervision ===
  Input: x = (map_image, text_query), helper image I
  # 1. Extract & compress helper image features
  patch_features = VLM.vision_encoder(I)  # {e_1, ..., e_n}, n patches
  latent_target = avg_pool(patch_features, k=4)  # {ê_1, ê_2, ê_3, ê_4}

  # 2. Forward pass with interleaved training
  # o = o_pre ⊕ <latent_slot> ⊕ o_post
  h_pre = VLM.forward(x, o_pre)                    # text tokens before latent
  h_1 = VLM.forward(x, o_pre, <bos_latent>)         # hidden state → latent token 1
  h_2 = VLM.forward(x, o_pre, h_1, <bos_latent>)    # hidden state → latent token 2
  h_3 = VLM.forward(x, ..., h_2, <bos_latent>)      # hidden state → latent token 3
  h_4 = VLM.forward(x, ..., h_3, <bos_latent>)      # hidden state → latent token 4
  logits_post = VLM.forward(x, o_pre, h_{1:4}, o_post)

  # 3. Loss computation
  L_visual = avg([cos_sim(ê_j, h_j) for j in 1..4])  # Eq.1, cosine similarity
  L_text   = CE(o_pre) + CE(o_post)                   # Eq.2, cross-entropy
  L_1      = L_visual + 0.1 * L_text                  # γ = 0.1

  # === Stage 2: Text-Only Supervision ===
  # Latent tokens are self-generated (no external ê_j)
  e_1 = VLM.hidden_state(x, o_pre)                    # Eq.3
  e_2 = VLM.hidden_state(x, o_pre, e_1)               # autoregressive latent gen
  e_3 = VLM.hidden_state(x, o_pre, e_1, e_2)
  e_4 = VLM.hidden_state(x, o_pre, e_1, e_2, e_3)
  logits_post = VLM.lm_head(VLM.forward(x, o_pre, e_{1:4}))
  L_2 = CE(o_pre) + CE(o_post | e_{1:4})             # Eq.4
  # Gradients flow back to e_{1:4} through o_post CE loss

  # === Stage 3: GRPO RL (VERL framework) ===
  for each query in RL_train:
      samples = [VLM.generate(x) for _ in 1..5]  # rollout_n=5
      for each sample:
          r = 0.9 * (1 if answer_correct else 0) + 0.1 * (format_correct)
      # Group Relative Policy Optimization on text tokens
      # Latent tokens receive gradient but are excluded from KL penalty
  ```

  Adam optimizer, β1=0.9, β2=0.95, weight_decay=0.01, lr=1e-5, batch_size=8, grad_accum=2, warmup=10 steps, epochs=10. SFT LR 1e-5, RL LR 1e-6. Trainable: all except vision encoder.

- 模型是什么。数据集和bench分别是什么。
  模型：Vision Encoder — CLIP ViT-L/336px (Radford et al., 2021)。LLM Backbone — Vicuna-v1.5-7B (默认)，增强变体使用 LLaMA-3.1-8B-Instruct。Vision token 压缩参数 C=1 (标准分辨率 336×336) 或 C=8 (高分辨率 672×672, LLaVA-Mini-HD)，pre-fusion 层数 N_fusion=4。
  训练数据：Stage 1 — 558K caption data (LLaVA pretraining data)。Stage 2 — 665K instruction data (LLaVA instruction data)。增强变体额外使用 100K Video-ChatGPT instruction data + 部分开源数据，共 3M training samples。
  Benchmarks 图像：VQA-v2, GQA, VisWiz, ScienceQA-IMG, TextVQA, POPE, MME, MMBench, SEED-Bench, LLaVA-Bench-in-the-Wild, MM-Vet (共 11 个)。
  Benchmarks 视频：MSVD-QA, MSRVTT-QA, ActivityNet-QA, Video-based Generative Performance (Correctness/Detail/Contextual/Temporal/Consistency), MVBench (20 子任务), MLVU (7 子任务), EgoSchema (共 7 个)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/ictnlp/LLaVA-Mini。模型权重：https://huggingface.co/ICTNLP/llava-mini-llama-3.1-8b

  算法 pipeline 伪代码：

  ```
  # ===== 输入 =====
  # X_v: 输入图像 (H×W×3), X_q: 语言指令文本 (l_q 个 token)

  # ===== Vision Encoding =====
  # CLIP ViT-L/336px: 图像 → 576 个 vision token (N^2=24×24)
  H_v = CLIP_ViT(X_v)              # [576, d_h]
  H_v = Projection(H_v)            # [576, d_h], 映射到 LLM embedding 空间
  H_q = LLM_Embedding(X_q)         # [l_q, d_h], 文本 token 嵌入

  # ===== Query-based Compression =====
  # C 个压缩 query (默认 C=1), 使用 2D sinusoidal position encoding
  Q_v = learnable_queries           # [C^2, d_h], C=1→[1, d_h]
  pos = 2D_Sinusoidal_PE()
  A = Softmax((Q_v + pos(Q_v)) @ (H_v + pos(H_v)).T)  # [C^2, 576]
  H_v_compressed = A @ H_v         # [C^2, d_h], 压缩后 vision token

  # ===== Modality Pre-fusion =====
  # N_fusion=4 层与 LLM 同构的 Transformer decoder blocks
  # 将全部 vision token 与 text token 拼接后输入
  concat = Concat(H_v, H_q)        # [576 + l_q, d_h]
  output = PreFusion(causal_mask=concat)  # [576 + l_q, d_h]
  H_q_fused = output[-l_q:]        # [l_q, d_h], 仅取文本位置的输出

  # ===== LLM Backbone =====
  # 只输入 1 个压缩 vision token + 融合后的 text token
  llm_input = Concat(H_v_compressed, H_q_fused)  # [1 + l_q, d_h]
  response = LLM(llm_input)        # 自回归生成回复

  # ===== 高分辨率图像 (HD) =====
  # 图像分割为 4 个子图 (2×2), 每子图独立编码
  H_v_sub = [ViT(sub) for sub in split(X_v, 2×2)]  # 4 × [576, d_h]
  H_v_full = ViT(X_v)              # 原图 [576, d_h]
  # 5 组 vision tokens 全部送入 pre-fusion
  # compression 压缩 4×576 子图 token 为 C^2 (C=8, 即 64) 个
  # LLM 输入: 64 vision tokens + l_q text tokens

  # ===== 视频处理 =====
  # M 帧视频, 每帧独立处理, C=1
  # Per frame: 1 compressed vision token + l_q fused text tokens
  # 视频总输入: M×1 vision tokens + l_q fused text tokens (pooled from M frames)
  # 对比 LLaVA-v1.5: M×576 vision tokens + l_q text tokens
  ```

  张量计算流程（标准分辨率 336px, C=1）：
  - Vision Encoder: 336×336×3 → ViT → [576, 1024] → Projection (Linear) → [576, 4096] (Vicuna-7B hidden dim)
  - Compression: Q_v [1,4096] + PE, H_v [576,4096] + PE → Cross-Attention → A [1,576] → Ĥ_v [1,4096]
  - Pre-fusion: Concat([576,4096], [l_q,4096]) → 4×Transformer → output[-l_q:, :] → Ĥ_q [l_q, 4096]
  - LLM: Concat([1,4096], [l_q,4096]) → Vicuna-7B 32 layers → autoregressive response
  - FLOPs: Vision 0.35T + Projection 0.02T + Compression 0.001T + Pre-fusion 0.13T + LLM 1.46T = **1.96T** (vs LLaVA-v1.5 8.55T, 减少 77%)
