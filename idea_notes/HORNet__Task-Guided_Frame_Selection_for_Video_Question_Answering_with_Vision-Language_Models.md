## HORNet__Task-Guided_Frame_Selection_for_Video_Question_Answering_with_Vision-Language_Models

- baseline方法是什么？
  Baseline 方法是现有的 frame-sampling 策略用于 VideoQA with VLMs，主要包括两类：(1) **Uniform Sampling** —— 以固定间隔从视频中均匀采样 T 帧（如 T=32），每帧经 Vision Encoder 编码为 visual tokens 后送入 VLM（Qwen3-VL）进行 QA 推理。这是最广泛使用的 baseline，完全不考虑帧内容与问题的相关性；(2) **Learned Selection 方法** —— SeViLA（chain Localizer+Answerer fine-tuned from BLIP-2, pseudo-label self-refinement）、Frame-Voyager（enumerate frame combinations + supervised selector via prediction loss ranking）、ReFoCUS（autoregressive frame selector via VLM confidence margin rewards）、ViaRL（co-evolve selector+answerer via iterated amplification RL）。这些方法或需微调 VLM（SeViLA, Frame-Voyager, ViaRL），或需修改 VLM 架构（ReFoCUS 部分微调），参数效率低且不通用。

  Baseline（Uniform Sampling + Qwen3-VL-2B, MSVD-QA）全栈执行例子：
  - 算法层：输入视频（~10s, ~300 frames @ 30fps） + 问题 → 均匀采样 T=32 帧（fps=2, 约覆盖 1/12 原始帧） → Qwen3-VL Vision Encoder 逐帧编码为 visual tokens → Projector 映射 → 拼接 text tokens → LLM 28 层 prefill + decode → 生成答案。32 帧中大量冗余帧（静态背景、重复动作）与关键帧（动作瞬间、物体交互）被等同处理，关键帧可能落在采样间隔之间被跳过 → F1-Lev = 0.3483, Qwen Proc. 0.28s, Avg. Frames=11.65（实际均匀采样后送入 Qwen 的帧数可能因内部处理而异）。
  - 系统框架层：基于 HuggingFace Transformers 推理，论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 FlashAttention，论文未明确说明
  - 硬件架构层：单张 NVIDIA A100 40GB GPU

  Baseline 的缺陷：
  1. **Uniform sampling 无法优化下游回答质量**：固定间隔采样完全忽略帧内容与问题的相关性——对于"the boy on the green disc goes down"这样的时序问题，关键交互瞬间可能仅占 32 帧中的 2-3 帧，且不一定落在均匀采样点。论文 Table 5 显示 NExT-QA 上 uniform 仅 64.24% vs HORNet 71.50%（+7.3 points），证明对于需要时序/因果推理的长视频，盲采样严重不足。
  2. **冗余帧注入噪声、增加计算**：MSVD（~10s 视频）中均匀采样的 11.65 帧大多冗余——HORNet 仅用 4 帧就达到 +1.7% F1 提升，证明大多数帧是噪声而非信号。MSRVTT（~15.5s）47.52 帧 → 4 帧，NExT-QA（~43.7s）1157.88 帧 → 8 帧，冗余比例极高。
  3. **现有 learned selection 方法参数效率低**：SeViLA fine-tunes BLIP-2（~1B+ params），Frame-Voyager 需 combinatorial enumeration，ReFoCUS 需修改 VLM——这些方法在 small-data 场景（数据稀缺、标注昂贵）下不可行，也无法在不同 VLM 间 transfer。
  4. **GRPO 此前仅用于优化 VLM 输出，未探索优化 VLM 输入**：Video-R1、R1-VL、DeepVideo-R1 等均用 GRPO 优化 VLM 生成分布（outputs），但没有人用 GRPO 优化"VLM 看到什么"（inputs）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：HORNet 首次将 GRPO 从优化 VLM 输出重定向到优化 VLM 输入，通过三项核心设计解决 baseline 缺陷：

  (1) **Select Any Frames (SAF) 问题形式化** → 将帧选择形式化为 RL 问题：学习参数化策略 π_θ(V,q) → 二进制选择 mask b ∈ {0,1}^T，最大化 E[R(M(b⊙V, q), a)]。SAF 无时序顺序或连续性约束——策略可自由选择时间上稀疏的关键事件、短关键片段或密集运动段，完全取决于什么能最大化任务驱动的 reward。

  (2) **GRPO-trained lightweight policy (<1M params)** → 解决参数效率。TimeSFormer-Tiny encoder 提取 per-frame spatiotemporal features → MLP (768→512→256→1, GELU, sigmoid) 输出 per-frame keep probability → 每步生成 K=8 个候选 mask（top-k sweep + stochastic Bernoulli） → frozen Qwen3-VL 回答 → 计算 reward (0.1·F1 + 0.9·EditSim) → GRPO 更新（group-normalized advantage, critic-free）。仅 MLP + encoder 可训练，VLM 始终冻结，参数 <1M。

  (3) **GRPO redirected from outputs to inputs** → 概念创新。传统 GRPO（DeepSeek-R1, Video-R1 等）优化 π_θ(a|V,q) 即生成分布；HORNet 优化 π_θ(b|V,q) 即选择分布。这一概念转换使方法更参数高效（不需修改 VLM）、更通用（policy 可 transfer 到不同 VLM answerer）。

  对比 baseline 的全栈执行例子（HORNet + Qwen3-VL-2B, MSVD-QA, 同一视频 + 问题）：
  - 算法层：输入同一视频 + 问题 → 均匀采样 T=32 帧 → TimeSFormer-Tiny 提取每帧 spatiotemporal 特征 F ∈ R^{32×768}（spatial self-attention per frame → temporal self-attention across frames → spatial avg pool） → MLP Policy 输出 p = [p_1,...,p_32] ∈ (0,1)^32 → 排序选 top-4 概率帧 → 仅 4 帧送入 frozen Qwen3-VL-2B → 生成答案。32→4 帧压缩 87.5%，且 F1-Lev 从 0.3483 提升到 0.3543 (+1.7%)。
  - 系统框架层：基于 HuggingFace Transformers，Qwen3-VL-2B 全程冻结，仅 MLP + TimeSFormer-Tiny 训练
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 FlashAttention（TimeSFormer 的 spatial-temporal factorized attention 仍与 FlashAttention 兼容）
  - 硬件架构层：单张 NVIDIA A100 40GB GPU，训练两阶段：Stage 1 (MSVD+MSRVTT, F1-Lev reward) → Stage 2 (NExT-QA, MCQ accuracy reward)

  解决对应关系：
  | Baseline 缺陷 | HORNet 解决方案 |
  |---|---|
  | Uniform sampling 无法优化回答质量 | SAF + GRPO: task-grounded reward signal（VLM QA accuracy）直接驱动帧选择策略优化。NExT-QA 上 HORNet 71.50% vs Uniform 64.24% (+7.3), 证明 GRPO 学到了时序关键帧的选择 |
  | 冗余帧注入噪声 | MLP policy 学会丢弃噪声帧：MSVD 11.65→4 帧 +1.7% F1, MSRVTT 47.52→4 帧 -5.6% F1（可控 trade-off）。短视频上甚至提升质量（丢弃噪声聚焦关键内容） |
  | 现有 learned selection 参数效率低 | <1M trainable params, VLM frozen: 比 SeViLA（fine-tune BLIP-2）、Frame-Voyager（combinatorial）、ViaRL（co-evolve）参数效率高 3 个数量级。适合 small-data/资源受限场景 |
  | GRPO 仅用于优化 VLM 输出 | 首次将 GRPO 从 output optimization 重定向到 input optimization。这一"概念转换"被论文 Table 4 验证：GRPO OOD generalization (MSRVTT 0.3029) 优于 PPO (0.2948) 和 SFT (0.2882)，证明 group-relative advantage 学到的选择策略更可迁移 |
  | 策略难以跨 VLM transfer | 同一 HORNet policy 换 Qwen2.5-VL-3B 后 F1-Lev 从 0.3543 提升到 0.3846 (+8.5% relative)，无需任何 retraining。Policy 与 answerer 解耦 |

  训练策略的关键设计：
  - **两阶段训练**：Stage 1 用短视频+one-word answer 学习"有用帧识别"（F1-Lev reward），Stage 2 用长视频+MCQ 学习"因果/时序推理所需帧选择"（accuracy reward）
  - **Candidate generation 策略**：K=8 = 7 top-k sweep (deterministic exploitation) + 1 Bernoulli (stochastic exploration)，平衡探索与利用
  - **Reward 设计**：0.1·F1_token + 0.9·EditSim (lemmatized)，比 exact match 对 minor lexical variations 更鲁棒
  - **GRPO 优势**：Table 4 消融证明 GRPO 的 OOD generalization 优于 PPO 和 SFT——group-relative advantage estimation 学到更可迁移的选择策略，而 PPO/SFT 更容易 overfit 到训练分布
