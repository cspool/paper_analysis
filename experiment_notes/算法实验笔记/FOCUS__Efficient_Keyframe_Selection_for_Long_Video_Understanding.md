## FOCUS__Efficient_Keyframe_Selection_for_Long_Video_Understanding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 FOCUS（Frame-Optimistic Confidence Upper-bound Selection），一个训练无关（training-free）、模型无关（model-agnostic）的 keyframe selection 模块，将 query-aware keyframe selection 建模为 multi-armed bandit 中的 Combinatorial Pure-Exploration (CPE) 问题。核心流程：将视频划分为固定长度 clip 作为 bandit arms→ Stage I 并行采样探索所有 arm → Stage II 对 optimistic UCB 最高的 α*m 个 arm 做细化采样 → 基于经验均值选择 top-m arm → 在选中 arm 内通过 nearest-neighbor 插值采样 keyframes。实验比较：(1) 与 uniform sampling 的 QA 准确率对比（GPT-4o、Qwen2-VL-7B、LLaVA-OV-7B、LLaVA-Video-7B）；(2) 与 SOTA keyframe selection（Top-K、AKS、Q-Frame）对比；(3) 效率对比：GPU hours 和 frames seen 占比；(4) α 超参数消融；(5) 消融实验：two-stage vs single-stage、Bernstein confidence radius vs empirical mean、clip length、vision-language encoder 选择；(6) 额外 benchmark：MLVU、VSI-Bench。

- 硬件平台是什么，配置是什么。
  单卡 NVIDIA H100 (80GB) GPU。所有 keyframe selection 方法的 GPU hours 均在该 GPU 上测量。MLLM 推理所用 GPU 论文未明确说明具体型号。

- 模型是什么。数据集和bench分别是什么。
  模型：FOCUS 作为前置 keyframe selection 模块，接在四个 MLLM 之前：GPT-4o (0513)、Qwen2-VL-7B、LLaVA-OV-7B、LLaVA-Video-7B。Frame scoring 使用 BLIP ITM (Li et al., 2022) 计算 cosine similarity 作为 frame-query relevance。也可替换为 CLIP (Radford et al., 2021) 或 SigLIP (Zhai et al., 2023)。
  数据集/Benchmark：LongVideoBench (Wu et al., 2024)、Video-MME (Fu et al., 2025)、MLVU (Zhou et al., 2025)、VSI-Bench (Yang et al., 2025)。评价框架使用 LMMs-Eval (Zhang et al., 2024a)，禁用字幕，zero-shot 评估。Video-MME 按 Short(<2min)/Medium(4-15min)/Long(30-60min) 分类；LongVideoBench 按 Short(<3min)/Medium(3-20min)/Long(>20min) 分类。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：代码在 https://github.com/NUS-HPC-AI-Lab/FOCUS。
  算法 Pipeline（对应 Algorithm 2——两阶段 Optimistic Confidence Upper-bound Arm Selection）：

  ```
  输入: 视频 V = (x_1, ..., x_T), query q, clip 长度 l, arm 数 M, 目标 arm 数 m,
        探索因子 α, 初始化 pulls q, 细化 pulls z, 目标 frame 数 k

  Stage I: Coarse Exploration
  1. 将 V 划分为 M 个固定长度 l 的 clip，每个 clip 作为一个 bandit arm A_a
  2. 对每个 arm a ∈ {1..M}，并行均匀采样 q 帧，通过 BLIP ITM 计算 reward:
     r_t = cosine_similarity(BLIP.encode_image(x_t), BLIP.encode_text(q))
  3. 更新每个 arm a 的经验均值 μ̂_a 和经验方差 σ̂_a²
  4. 计算 Bernstein confidence radius:
     β_a(n) = sqrt(2 * σ̂_a² * ln(n) / max(1, N_a(n))) + 3 * ln(n) / max(1, N_a(n))
     其中 n = M*q 为总 pulls 数，N_a(n) = q
  5. 计算 optimistic mean: μ̃_a(n) = μ̂_a(n) + β_a(n)

  Stage II: Fine-grained Exploitation
  6. A_coarse = TopM(μ̃, α*m)  // 取 optimistic mean 最高的 α*m 个 arm
  7. 对 a ∈ A_coarse，并行采样 z 帧，更新 μ̂_a(n), σ̂_a², N_a(n)
  8. A_fine = TopM(μ̂, m)  // 取经验均值最高的 m 个 arm（无偏估计）

  Frame Selection within Selected Arms
  9. 每个 arm a ∈ A_fine 分配 k_a = round(k/m) 个 frame slot（均匀分配，调整至总和=k）
  10. 在 arm 内通过 nearest-neighbor 插值所有 frame 的 reward r̂_{a,t}
  11. 构建 per-arm 采样分布 p_a ∝ r̂_{a,t}，不放回采样 k_a 帧
  12. 合并为最终 keyframe 集 K = ∪_{a∈A_fine} K_a，|K| = k

  输出: K
  ```

  关键张量计算：BLIP 视觉编码器对每帧输出 visual embedding e_t ∈ R^d，文本编码器对 query 输出 text embedding e_q ∈ R^d，reward r_t = (e_t · e_q) / (||e_t|| · ||e_q||) ∈ [0,1]。所有 BLIP forward 批处理执行（并行 arm-pull），避免串行 GPU 利用率浪费。

  超参数默认值：clip length l = 16 秒，α = 0.25，q = 论文未明确说明，z = 论文未明确说明。效率：处理 LongVideoBench 上仅需 1.6% 帧的 BLIP forward（vs AKS w/ pre-filtering 的 3.7%），5.5 GPU hours（vs AKS 的 9.3 GPU hours）。
