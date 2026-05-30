## CurveStream__Boosting_Streaming_Video_Understanding_in_MLLMs_via_Curvature-Aware_Hierarchical_Visual_Memory_Management

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：CurveStream 是一个 training-free 的 curvature-aware 层级化视觉记忆管理框架，通过特征流形几何曲率评估语义转换强度，动态管理 MLLM 的视觉记忆队列。核心包含两个模块：(1) Curvature-Aware Scorer (CAS) —— 使用冻结的 DINOv2-small 视觉编码器提取每帧的全局特征表示 F_t ∈ R^D 并 L2 归一化，计算一阶 Motion Variation M_t = 1 - cos(F_t, F_{t-1}) 和二阶 Geometric Curvature C_t = 1 - cos(d1, d2)，其中 d1 = F_{t-1} - F_{t-2}，d2 = F_t - F_{t-1}，最终 Curvature Score CS_t = M_t + λC_t；(2) Hierarchical Visual Memory Management (HVMM) —— 使用 EMA 在线更新曲率分数的运行均值 μ_t 和方差 σ_t²，构建 K-Sigma 动态双阈值 g1 = μ_t + k1σ_t, g2 = μ_t + k2σ_t (k1 < k2)，将每帧根据 CS_t 自适应分为 Clear Memory（CS_t ≥ g2，保留原始高分辨率）、Blurred Memory（g1 ≤ CS_t < g2，降采样到 224×224）或 Discard（CS_t < g1，丢弃），当 |M_t| > N_max=20 时执行 FIFO 驱逐最旧 token。
  实验比较：(a) 与 Open-source Online MLLMs（Flash-VStream-7B, VideoLLM-online-8B, Dispider-7B, TimeChat-Online-7B, StreamForest-7B）和 Training-free Offline-to-Online Methods（ReKV, HERMES, FreshMem）在 StreamingBench（10 个实时视觉理解子任务）和 OVOBench（6 个实时视觉感知子任务）上的 accuracy 对比；(b) 跨 Base MLLM 的泛化实验（LLaVA-OneVision-7B, Qwen2-VL-7B, Qwen2.5-VL-7B, Qwen3-VL-8B）；(c) 离线 benchmark 实验 —— MVBench（20 子任务）, EgoSchema, VideoMME, FAVOR-Bench；(d) 跨参数规模 scalability 实验 —— Qwen3-VL 系列 4B/8B/32B；(e) 消融实验 —— curvature metric 有效性（vs Uniform Sampling, Cosine Similarity, Optical Flow, Pyramid Optical Flow）、各组件独立/联合贡献（CAS only, HVMM only, CurveStream full）、curvature score weight λ 的 robustness、K-Sigma 双阈值 (k1, k2) 的 sensitivity、Clear Memory 保留比例对 accuracy 和 token 成本的影响。

- 硬件平台是什么，配置是什么。
  所有 benchmark 评估在单张推理 GPU 上独立执行，以充分验证严格受限内存条件下的框架鲁棒性。论文未明确说明 GPU 型号。特征提取前端使用 DINOv2-small 模型获取时序特征的局部几何表示。所有方法统一建立 memory bank 容量上限 N_max=20 frame tokens，严格模拟流视频处理的物理 GPU 内存约束。

- 模型是什么。数据集和bench分别是什么。
  模型（Base MLLMs）：LLaVA-OneVision-7B, Qwen2-VL-7B, Qwen2.5-VL-7B, Qwen3-VL（4B/8B/32B）。
  视觉编码器（CAS 前端）：DINOv2-small（冻结）。
  在线 benchmark：StreamingBench（10 个实时视觉理解子任务：OP/CR/CS/ATP/EU/TR/PR/SU/ACP/CT），OVOBench（6 个实时视觉感知子任务：OCR/ACR/ATR/STU/FPD/OJR）。
  离线 benchmark：MVBench（20 个细粒度子任务，短视频），EgoSchema（自我中心长视频），VideoMME（含 Short/Medium/Long 子集，最长数小时），FAVOR-Bench（微动动力学感知）。
  对比 baseline：Base MLLMs（均匀采样 1fps 或 64 frames），Streaming 方法（Flash-VStream, FreshMem, HERMES, ReKV, StreamForest, Dispider, TimeChat-Online），封闭源 MLLMs（GPT-4o, Gemini 1.5 Pro, Claude 3.5 Sonnet）。
  评价指标：Accuracy (%)，各子任务细分准确率。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/streamingvideos/CurveStream（论文声明的代码仓库）

  算法 pipeline 伪代码（基于 Algorithm 1）：
  ```
  # === 初始化 ===
  M_0 = []                           # 空视觉记忆队列
  μ_0 = 0, σ_0 = 0                   # 曲率分数瞬态分布参数
  N_max = 20                         # 最大记忆容量
  λ = 0.2                            # 几何曲率惩罚权重
  k1 = 0.0, k2 = 1.0                 # K-Sigma 阈值乘数
  visual_encoder = DINOv2-small      # 冻结的视觉编码器

  # === 在线处理每个输入帧 I_t ===
  for each frame I_t from infinite video stream:
      # Step 1: 提取特征
      F_t = visual_encoder(I_t)      # shape: (D,)，DINOv2-small 输出
      F_t = F_t / ||F_t||_2          # L2 normalization

      if t > 3:
          # Step 2: CAS — 计算曲率分数
          # 一阶 Motion Variation: M_t = 1 - cos(F_t, F_{t-1})
          M_t = 1 - dot(F_t, F_{t-1}) / (||F_t|| * ||F_{t-1}||)

          # 二阶 Geometric Curvature: 特征位移向量的角度偏差
          d1 = F_{t-1} - F_{t-2}     # shape: (D,)
          d2 = F_t - F_{t-1}         # shape: (D,)
          C_t = 1 - dot(d1, d2) / (||d1|| * ||d2||)

          # 最终曲率分数
          CS_t = M_t + λ * C_t       # scalar

          # Step 3: HVMM — 在线更新分布参数 (EMA)
          μ_t = γ * μ_{t-1} + (1 - γ) * CS_t          # γ ∈ (0,1) 历史窗口控制
          σ_t² = γ * σ_{t-1}² + (1 - γ) * (CS_t - μ_t)²

          # Step 4: 计算 K-Sigma 动态双阈值
          g1 = μ_t + k1 * σ_t         # 模糊记忆下界
          g2 = μ_t + k2 * σ_t         # 清晰记忆下界 (k1 < k2)

          # Step 5: 层级状态路由
          if CS_t >= g2 or t == t_q:   # t_q: 查询时刻
              s_t = Clear Memory
              r_t = High               # 保留原始高分辨率（base model native resolution）
          elif g1 <= CS_t < g2:
              s_t = Blurred Memory
              r_t = Low                # 降采样到 224×224
          else:  # CS_t < g1
              s_t = Discard            # 丢弃低信息冗余帧
              # 不存入 memory bank

          # Step 6: 更新记忆队列
          M_t = Update(M_{t-1}, I_t, s_t, r_t)

          # Step 7: FIFO 驱逐（确保常值内存占用）
          if |M_t| > N_max:
              evict oldest tokens from M_t (FIFO)
      else:
          # t ≤ 3: 前 3 帧积累期，直接以 Clear Memory 存入
          M_t = Update(M_{t-1}, I_t, s_t=Clear, r_t=High)

  # === 查询时刻 t_q ===
  # M_tq 中的视觉 tokens 与自然语言查询 Q 拼接，送入 MLLM 生成答案 A
  answer = MLLM(concat([visual_tokens_from_M_tq, text_tokens_of_Q]))
  ```

  关键张量维度：
  - 输入帧 I_t: H×W×3，分辨率取决于 base MLLM 的动态高分辨率策略
  - Feature dim D: DINOv2-small 输出维度（~384 或 ~768 取决于变体）
  - 记忆队列 M_t: 最大容量 N_max = 20 frames
  - Clear Memory 帧保留原始分辨率（base MLLM native）
  - Blurred Memory 帧统一降采样到 224×224
  - λ = 0.2, k1 = 0.0, k2 = 1.0, γ 论文未明确给出（EMA momentum）
  - 曲率分数 CS_t: scalar ∈ [0, 2]（M_t ∈ [0,1], C_t ∈ [0,1], λ=0.2）

  几何理论解释（Appendix C）：
  - C_t 严格等价于单位切向量变化平方的一半：C_t = 1/2 ||T2 - T1||²
  - 恒速运动（如平滑相机平移）: T1 ≈ T2 → C_t ≈ 0 → 曲率惩罚自然抑制物理运动噪声
  - 语义突变（如镜头切换/新实体进入）: T2 投射到近乎正交子空间 → C_t 急剧增大 → 明确的曲率尖峰
