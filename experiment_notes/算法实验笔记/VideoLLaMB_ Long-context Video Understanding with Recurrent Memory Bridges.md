## VideoLLaMB: Long-context Video Understanding with Recurrent Memory Bridges

- 属于算法pipeline的实现是什么？实验比较什么？
  VideoLLaMB 提出一种用于长视频理解的递归记忆桥接框架，核心实现包括：(1) SceneTiling 算法——基于 ViT [CLS] token 的帧间余弦相似度计算 depth score，按 μ+α·σ 阈值将视频语义分割为 K 个语义段；(2) Recurrent Memory Bridge Layers——单层 Transformer，在每个语义段前 prepend 固定数量（32）的 memory tokens，通过 self-attention 递归更新 memory tokens 并输出视觉表示；(3) Memory Cache with Retrieval——将历史 memory tokens 存储在 memory cache 中，以当前 memory token 为 query、cache 为 key/value 进行 cross-attention 检索更新，缓解 BPTT 梯度消失问题。实验比较长视频理解（EgoSchema/NExTQA/VideoMME）、综合视频理解（MVBench）、规划任务（EgoPlan）、流式视频字幕生成以及自建 NIAVH benchmark 上的帧检索能力。

- 硬件平台是什么，配置是什么。
  训练使用 4× NVIDIA A800 GPU；推理评估使用单张 NVIDIA A100 (80GB) / A800 GPU。论文声称单张 A100 即可处理 320 帧视频，仅训练于 16 帧。

- 模型是什么。数据集和bench分别是什么。
  模型：LLM 使用 Vicuna-7B-v1.5，视觉编码器使用 ViT-L/14（基于 CLIP），遵循 Video-LLaVA 架构。训练初始化自 LLaVA-1.5 配置。
  数据集：训练数据与 PLLaVA 相同（Video-LLaVA 视频数据 + LLaVA-1.5 微调图像数据），额外使用 VideoChat2 数据集进行扩展训练。
  Benchmark：EgoSchema（零样本子集评估，180s 平均长度）、NExTQA（验证集，45s 平均长度，含 temporal/causal/description 三类问题）、VideoMME（11s~1h 视频，short/medium/long 子集）、MVBench（20 类多选问答）、EgoPlan（3355 题，零样本具身规划）、自建 NIAVH（Needle In A Video Haystack，基于 Ego4D + Sora/DALL-E，320s 上下文）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/bigai-nlco/VideoLLaMB，基于 LLaVA 代码库构建（Python 99.3%），README 包含安装步骤、CLI 使用、streaming、Gradio demo、训练/评估脚本和模型 zoo。无 license 文件。

  算法 pipeline 伪代码：
  ```
  # === VideoLLaMB 推理 Pipeline ===
  # 输入: video V = {v_1, v_2, ..., v_n} (n frames), query Q
  # 输出: answer A

  # Step 1: Vision Encoding
  for each frame v_i in V:
      f_i = ViT-L/14(v_i)  # 每帧提取特征, f_i ∈ R^{H×W×D}
  F = {f_1, f_2, ..., f_n}

  # Step 2: SceneTiling 语义分割
  for i = 1 to n-1:
      c_i = CosineSim(ViT.cls_token(v_i), ViT.cls_token(v_{i+1}))  # 相邻帧CLS余弦相似度
  for i = 1 to n-1:
      cl_i = max(c_{1..i-1}), cr_i = max(c_{i+1..n-1})
      d_i = (cl_i + cr_i - 2*c_i) / 2  # depth score
  μ = mean({d_1..d_{n-1}}), σ = std({d_1..d_{n-1}})
  threshold = μ + α * σ
  seg_boundaries = {i | d_i > threshold}  # 选取K-1个分割点
  将视频划分为K个语义段 {s_1, s_2, ..., s_K}, s_i = {f_{start_i}..f_{end_i}}

  # Step 3: Recurrent Memory Bridge (逐段处理)
  m_0 = random_init_memory_tokens(N_mem=32, D=1024)  # 初始化memory tokens
  MemoryCache = []  # memory cache用于跨段检索

  for i = 1 to K:
      # 3a: prepend memory tokens to segment features
      input_i = Concat(m_{i-1}, s_i)  # [32+M_i, 1024], M_i为段内帧数
      
      # 3b: Self-attention in Bridge Layer (单层Transformer, 8 heads, hidden=1024)
      [m_i', o_i] = BridgeLayer_SelfAttn(input_i)  
      # m_i': 更新后的临时memory tokens [32, 1024]
      # o_i: 段内视觉表示 [M_i, 1024]
      
      # 3c: Memory Retrieval (cross-attention with cache)
      if MemoryCache is not empty:
          M_cache = Concat(m_0, m_1, ..., m_{i-1})  # 拼接历史memory
          # Cross-attention: query = m_i', key = M_cache, value = M_cache
          m_i = Softmax(W_Q*m_i' * (W_K*M_cache)^T / sqrt(d_k)) * W_V*M_cache
      else:
          m_i = m_i'
      
      MemoryCache.append(m_i)

  # Step 4: LLM输入投影
  visual_summary = Projector(o_1, o_2, ..., o_K)  # 所有段视觉输出投影
  final_input = Concat(m_K, visual_summary)  # 当前memory + 视觉表示送入LLM

  # Step 5: LLM生成
  A = Vicuna-7B(Concat(final_input, Tokenize(Q)))

  # 计算复杂度:
  # Bridge层: O((C+M)^2) per segment, C=段帧数, M=32 memory tokens
  # Memory Retrieval: O(M*K) per step, K=段数
  # 总时间: O(K^2), 空间: O(K)
  # LLM: O(M^2) token输入 (vs 原始 O(n^2))
  ```

  张量计算层面：ViT-L/14 对每帧 224×224 图像提取 patch features（patch_size=14, 256 patches + 1 CLS token, D=1024）。SceneTiling 仅计算 CLS token 间余弦相似度，额外开销极小（O(n) 次余弦计算）。Memory Bridge 的 self-attention 输入为 [32+M, 1024]，M 为段内帧数（如 16 帧分 4 段则 M≈4），计算量极小。Memory Retrieval 的 cross-attention 为 [32] query × [32*K, 1024] key/value，线性增长。最终 LLM 输入约 32+N_proj 个 token（vs 无压缩的 n_patches * n_frames = 256*16=4096 tokens），GPU 显存呈线性缩放（Figure 4），支持 320 帧处理。

  Streaming caption 模式下：SceneTiling 仅用左侧相似度 d_i = (cl_i - c_i)/2 实时检测场景边界，无需预知全视频，在边界处自动生成事件字幕。
