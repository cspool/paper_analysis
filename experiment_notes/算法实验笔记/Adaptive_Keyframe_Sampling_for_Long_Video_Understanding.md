## Adaptive_Keyframe_Sampling_for_Long_Video_Understanding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：Adaptive Keyframe Sampling (AKS) 是一个 plug-and-play 的 keyframe 选择模块，放在 MLLM visual encoder 之前，目标是在固定数量的视频 token 约束下最大化 keyframe 的信息量。核心设计：(1) Relevance 计算 —— 使用 VL 模型（默认 BLIP ITM）计算每个候选帧 $\mathbf{F}_t$ 与 prompt $\mathbf{Q}$ 的匹配分数 $s(\mathbf{Q}, \mathbf{F}_t)$；(2) Coverage 估计 —— 基于 Ripley's K-function 的递归分 bin 机制，将时间轴 [0, T) 递归二分为 bin，在每个 bin 内统计 keyframe 数量，通过不均分布惩罚项 $|m_1 - m_2|$ 量化 coverage；(3) ADA（Adaptive Sampling）算法 —— 综合 relevance 和 coverage 的分层优化：在每层递归中，若 $s_{\text{top}} - s_{\text{all}}$ 超过阈值 $s_{\text{thr}}$，则倾向于保留高相关性帧（TOP 模式）；否则将当前 bin 二分为子 bin 并均匀分配 keyframe 数（BIN 模式）。ADA 是 TOP（纯相关性最大化）和 BIN（纯 coverage 保证）的自适应折中。
  实验比较：(a) 将 AKS 应用于三种 baseline MLLMs（Qwen2VL-7B、LLaVA-OV-7B、LLaVA-Video-7B），对比 uniform sampling 基线在 LongVideoBench val 和 VideoMME 上的 QA accuracy；(b) 与 SOTA MLLMs（GPT-4V, GPT-4o, Gemini-1.5-Flash/Pro, VideoLLaVA, MiniCPM-V 2.6, PLLaVA, VILA 等）对比；(c) 诊断实验 —— 不同 sampling 策略（UNI/TOP/BIN/ADA）在 LongVideoBench val 和 VideoMME 上的 accuracy 对比；(d) 消融实验 —— sampling frequency（1/0.5/0.25/0.125/0.1 fps）、VL scorer 选择（BLIP/Sevila/CLIP）、ADA 超参数 L 和 $s_{\text{thr}}$ 的影响；(e) 泛化实验 —— AKS 扩展到 video referring 和 video captioning 任务。

- 硬件平台是什么，配置是什么。
  论文未明确说明 GPU 型号。论文提到以 1 fps 采样候选帧，使用 BLIP ITM 计算 prompt-frame relevance 分数（预计算并存储），MLLM 推理基于 Qwen2VL/LLaVA-OV/LLaVA-Video 的标准推理流程。使用 LMMs-Eval 框架进行评估。

- 模型是什么。数据集和bench分别是什么。
  模型：作为 baseline MLLM 的有 Qwen2-VL-7B（使用 Qwen2-7B LLM）、LLaVA-OneVision-7B（使用 Qwen2-7B LLM）、LLaVA-Video-7B（使用 SigLIP 视觉编码器 + Qwen2-7B LLM，支持最多 64 帧输入）。VL scorer 默认使用 BLIP ITM，可选 Sevila 和 CLIP。
  数据集与 Benchmark：LongVideoBench（3763 个视频，最长 1 小时，6678 个多项选择题，17 个类别）的 val 子集；VideoMME（900 个视频，256 小时，2700 个多项选择题，30 个子领域，含 Short/Medium/Long 三个时长子集）。均不使用视频字幕辅助回答。
  评价指标：Accuracy（多项选择题正确率百分比）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/ncTimTang/AKS
  
  算法 pipeline 伪代码：
  ```
  # ==== AKS: Adaptive Keyframe Sampling ====
  # 输入: 视频 V ∈ R^{T×W×H×C}（T 帧）, prompt Q, 目标 keyframe 数 M
  # VL scorer: BLIP ITM 或其他 VL 模型

  # Step 1: 候选帧采样与预计算 relevance score
  candidates = sample_frames(V, fps=1)  # 1 fps → 约 T 个候选帧
  matching_scores = []  # 预计算并存储
  for each candidate frame F_t in candidates:
      # 将 prompt Q（text）和帧 F_t（image）送入 VL 模型
      q_emb = VL_text_encoder(Q)       # text embedding
      f_emb = VL_image_encoder(F_t)    # image embedding
      s_t = ITM(q_emb, f_emb)          # image-text matching score, scalar
      matching_scores.append(s_t)
  # matching_scores: list of shape (T,), 与 question 对应

  # Step 2: ADA 递归分层优化 keyframe 选择
  def ADA(matching_scores, level, max_level, s_thr, M):
      # 计算当前 bin 内所有帧的平均分和 Top-M 帧的平均分
      s_all = mean(matching_scores)
      s_top = mean(topk(matching_scores, M))
      margin = s_top - s_all

      if margin >= s_thr or level >= max_level:
          # TOP 模式: 帧间区分度足够 → 直接选 Top-M 高分帧
          return argtopk(matching_scores, M)
      else:
          # BIN 模式: 拆分 bin → 均分配
          mid = len(matching_scores) // 2
          left_scores = matching_scores[:mid]
          right_scores = matching_scores[mid:]
          M_left = M // 2
          M_right = M - M_left
          left_indices = ADA(left_scores, level+1, max_level, s_thr, M_left)
          right_indices = ADA(right_scores, level+1, max_level, s_thr, M_right)
          # 将右半部分的索引偏移回全局索引
          right_indices_global = [idx + mid for idx in right_indices]
          return left_indices + right_indices_global

  # Step 3: 选择 keyframe 并送入 MLLM
  selected_indices = ADA(matching_scores, level=0, max_level=L, s_thr, M)
  # selected_indices: 长度为 M 的整数列表，指示选中帧的全局索引
  keyframes = [candidates[i] for i in selected_indices]

  # Step 4: MLLM 推理
  visual_tokens = [VisualEncoder(frame) for frame in keyframes]
  # visual_tokens: M 组 token，经 Projector 对齐后与 text prompt 拼接
  answer = MLLM(visual_tokens, Q)
  ```

  关键设计细节：
  - Coverage 度量：递归 level-0: 2 个 bin（[0,T/2), [T/2,T)），bin width T/2；level-1: 4 个 bin，bin width T/4；... level-L: 2^L 个 bin。在每层，penalty term = |m_1 - m_2| + |m_3 - m_4| + ...，其中 m_i 是第 i 个 bin 内 keyframe 数量。最大递归深度 L ≤ ⌈log₂ M⌉。
  - ADA 的两个超参数：L（最大递归深度）和 s_thr（区分度阈值）。L 控制 coverage 粒度，s_thr 控制何时放弃 coverage 约束转向 TOP 模式。LongVideoBench 偏好较小的 L 和 s_thr（问题多聚焦于单个时刻），VideoMME 偏好较大的值（问题需要多个时刻的信息）。
  - VL scorer：默认 BLIP ITM（基于 object-level 预训练，对 object 相关问题更敏感），可选 CLIP（基于 generic image-text 预训练，对全局感知问题更好）。预计算所有候选帧的 relevance 分数存储在 matching_scores 列表中。
  - 候选帧采样频率：默认 1 fps；可降低至 0.1 fps 仍保持高于 uniform baseline 的 accuracy（LongVideoBench: 60.1 @ 64 frames, 0.1 fps vs 58.9 uniform）。

  关键张量维度：
  - 视频帧数 T: 可变化，主流实验 32 或 64 个 keyframe 输入 MLLM
  - 候选帧数（1 fps）：约为视频时长（秒），如 600s 视频 ≈ 600 候选帧
  - VL 模型 embedding 维度: BLIP text/image embedding 维度取决于具体 BLIP 变体
  - L (max recursion level): 典型值 3-5（Table 5），覆盖 2³ 到 2⁵ 个 bin
  - s_thr: 典型值 0.2-1.0（Table 5），控制 TOP vs BIN 的切换倾向

  复杂度分析：
  - 预计算 relevance 分数：O(T × (text_enc + img_enc + ITM))，使用 BLIP ITM（轻量模型），远低于 MLLM forward cost
  - ADA 递归：O(T × L)，L 最大为 ⌈log₂ M⌉ ≤ 6（M=64 时），可忽略
  - 总 overhead：主要来自 VL scorer 预计算，相比 MLLM 推理开销小
  - 与 MLLM 无关：AKS 仅改变输入帧，MLLM 本身（Qwen2VL/LLaVA-OV/LLaVA-Video）不做任何修改
