## EVA__Efficient_Reinforcement_Learning_for_End-to-End_Video_Agent

- baseline方法是什么？
  Baseline 方法是 perception-first 的视频理解范式，分为两类：(1) Passive MLLM 方法（Qwen2.5-VL、LongVA、LongVila、Video-R1）—— 将整个视频或均匀采样帧作为静态 context 输入 MLLM，不进行选择性注意或自适应推理。模型被动消费所有帧后一次性生成答案。(2) Adaptive Agent 方法（FrameThinker、VideoAgent、VideoMTR）—— 引入外部帧选择工具，但仍遵循"先给均匀采样帧 + query，再 tool call"的 perception-first 流程，且工具控制维度单一（仅可调时间范围，不能调帧数和分辨率），工作流基于固定参数和刚性规则，探索能力受限。

  Baseline（Qwen2.5-VL-7B, 32 frames uniform sampling, LSDBench）全栈执行例子：
  - 算法层：输入一段长视频（>6600s）+ 问题 "figure out action sequences" → 以固定间隔均匀采样 32 帧 → Vision Encoder 逐帧编码为 650 tokens/frame → 32×650=20.8K visual tokens → Projector 映射 → 与 text prompt tokens 拼接 → LLM 28 层 prefill + decode → 生成答案。对于超长视频，32 帧均匀采样严重欠采样，关键帧被跳过，模型只能猜测。
  - 系统框架层：基于 HuggingFace Transformers 或 vLLM 推理
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 FlashAttention，论文未明确说明
  - 硬件架构层：推理用 vLLM on GPU（720p 视频），论文未明确说明

  Baseline 的缺陷：
  1. **被动识别导致冗余视觉处理**：MLLM 被当作被动识别器，预处理整个视频或均匀采样帧，无法选择性关注信息量高的时刻。对于超长视频（>6600s），32 帧均匀采样导致关键帧完全丢失，模型只能猜测。即使在 agent 方法中，模型也是在看到均匀采样帧后才开始推理——先被灌入大量可能无关的视觉 token，影响后续规划。
  2. **固定刚性工具控制维度**：FrameThinker 等 agent 方法仅允许调整时间范围，无法同时控制帧数（nframes）和空间分辨率（resize）。当需要放大关键区域细节时只能增加时间精度但无法提高空间分辨率，导致信息丢失。
  3. **缺乏自主探索策略**：现有方法依赖手工设计的固定工作流（如"先均匀采样 32 帧→再 tool call 特定时间段"），无法根据 query 动态调整策略。模型不能决定"先低分辨率全局浏览，再高分辨率聚焦关键段"这样的自主策略。
  4. **训练与推理行为脱节**：Agent 方法通常在推理时使用 tool call，但在训练时缺乏探索性 tool-use 的奖励信号，导致 trained behavior（SFT 格式跟随）与 expected behavior（高效自适应探索）之间存在 gap。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：EVA 通过以下设计实现 planning-before-perception 的自主视频 Agent：
  (1) **Planning-before-perception 范式**：初始仅提供 query，模型先基于文本推理生成 plan，再调用 tool 获取视觉信息。通过迭代 summary–plan–action–reflection 循环逐步完善感知和推理。
  (2) **灵活四参数帧选择工具**：start_time、end_time、nframes（帧数）、resize（空间下采样比），同时控制时间和空间粒度，支持"低分辨率全局浏览 → 高分辨率聚焦"等灵活策略。
  (3) **三阶段训练 pipeline**：SFT Cold-Start（学习 tool-call 格式和基本帧选择策略）→ KTO Correction（从典型失败中学习，纠正猜测、欠采样等模式）→ Data-Enhanced GRPO（在线 RL 优化，通过收集失败案例为 teacher MLLM 生成新 QA 来增强训练多样性）。

  对比 baseline 的全栈执行例子（EVA + Qwen2.5-VL-7B, 同一超长视频 + 同一问题）：
  - 算法层：
    1. Round 1（仅 query，无视觉输入）：s_0 = {q, [], []} → MLLM Planning: "需要先获取视频全貌，先低分辨率快速浏览" → Action: frame_select(start=0, end=6600, nframes=10, resize=0.1) → 提取 10 帧低分辨率全局帧 → Summary: 描述每帧内容 → Reflection: "全局帧中看到多个场景...需要聚焦 [200,250] 时间段的高分辨率信息"
    2. Round 2：MLLM Planning: "在 [200,250] 时间段观察到了目标动作的迹象，需要高分辨率确认" → Action: frame_select(start=200, end=250, nframes=100, resize=0.4) → 提取 100 帧高分辨率聚焦帧 → Summary + Reflection: "已获得足够证据" → 生成正确答案
    3. 总 visual tokens: 10×650×0.1 + 100×650×0.4 ≈ 650 + 26000 tokens（但仅展示采样帧中的部分帧给 MLLM，实际使用远少于计算值）
    ↑ LSDBench accuracy: 51.0%（vs baseline 49.2%），仅用 ~76.9 frames/10.3K tokens（vs baseline 32 frames/21K tokens）
  - 系统框架层：vLLM serving（temperature=0），720p 原始视频
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：32 × NVIDIA H100（训练），推理用 vLLM

  解决对应关系：
  | Baseline 缺陷 | EVA 解决方案 |
  |---|---|
  | 被动识别导致冗余视觉处理 | Planning-before-perception：先基于 query 推理规划，再有针对性地获取视觉信息。round 1 用低分辨率全局浏览，round 2+ 聚焦关键段高分辨率。LSDBench: 10.3K vs 21K tokens 且 +1.8% accuracy |
  | 固定刚性工具控制维度 | 四参数工具 (start/end/nframes/resize)：同时控制空间和时间粒度。Case study 展示从 resize=0.1 全局到 resize=0.4 聚焦的自适应切换 |
  | 缺乏自主探索策略 | 三阶段 RL 训练：SFT 学习格式 → KTO 纠正失败模式 → GRPO 在线优化 exploration-exploitation 平衡。消融实验证明 GRPO 阶段 agent 从"格式跟随者"进化为"策略探索者"（更多轮次但每轮更精准 token 分配） |
  | 训练与推理脱节 | KTO 从真实失败轨迹学习（63% correct + 37% rejected），GRPO 用 ROUGE/CSV reward 实现端到端优化，使训练目标与推理表现对齐。SFT→KTO 减少帧数和轮数但提升 3-4%，KTO→GRPO 增加轮数但最高 accuracy |

  训练阶段渐进演化（Ablation 核心发现）：
  - SFT only: 大量帧数 + 多轮交互 → 最低 accuracy（学会了 tool-call 格式但不会高效探索）
  - +KTO: 显著减少帧数和轮数 → 大幅提升 accuracy（学会了避免错误策略：猜测、欠采样、过采样）
  - +GRPO: 帧数再减少但轮数增加 → 最高 accuracy（学会了 multi-round deliberate reasoning + 精准 token 分配）
  - 这揭示了一个**策略演化路径**：格式跟随者 → 纠错后谨慎探索者 → 策略性主动探索者

- baseline方法是什么？
  Baseline 分为两个层面：(1) **无压缩 baseline** —— LLaVA-OneVision VLLM 完整处理所有输入帧的 visual tokens（32 frames × 196 tokens/frame = 6272 visual tokens for 7B），prefilling 阶段计算所有 token 的 QKV 并填充 KV cache，decoding 阶段每步对完整 KV cache 计算注意力，生成每个新 token 在 7B 模型上耗时约 42ms；(2) **One-shot token pruning 方法** —— FastV [3] 在 prefilling 阶段基于第 5 层 attention score 一次性评估 visual token 重要性，保留 top-35% tokens 到 KV cache，之后 decoding 不再改变剪枝结果；LLaVA-PruMerge [39] 基于 CLIP 视觉编码器的 attention score 一次性选择关键 visual token（保留约 55%），同样在整个 decoding 阶段固定不变。两种方法都是 "先评估，一次性剪枝，不再调整" 的单阶段静态策略。

  Baseline（Full tokens + LLaVA-OV-7B, 32 frames, MVBench）全栈执行例子：
  - 算法层：输入一段 32 帧视频 + question → CLIP vision encoder 逐帧编码为 196 tokens (N_v=196) → 视觉嵌入 Z_v ∈ R^{6272×D} → projector 映射到文本空间 H_v' → concat[H_v', H_q] → LLM (28 transformer layers, d=3584, m=18944) prefilling 计算 6272 + N_q 个 token 的 QKV → 全部写入 KV cache → decoding 每步对 6272 visual tokens 做完整 attention → 自回归生成答案。FLOPs 约 41.4T。
  - 系统框架层：LMMs-Eval 评估框架，PyTorch，LLaVA-NeXT 推理代码
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明（兼容 Flash Attention，无额外 kernel 修改）
  - 硬件架构层：NVIDIA A6000 (48GB) / RTX 4090 (24GB) / A100 (80GB)

  Baseline 的缺陷（基于论文 Figure 2 的核心发现）：
  1. **时间注意力漂移（Temporal Attention Shift）**：论文通过可视化 LLaVA-OV-7B decoding 阶段各迭代步的 attention score 分布（Figure 2）发现：不同预测 token 关注不同的视觉 token，且某些视觉 token（如 frame #1）随 decoding 进行重要性下降，而另一些（如 frame #16）重要性上升。这意味着 one-shot pruning 在 prefilling/首步 evaluation 后固定剪枝集，在后续 decoding 中必然错误剪除后期变得重要的 token 或保留错误 token，导致信息丢失。
  2. **时空冗余未充分利用**：视频存在大量 temporal redundancy（相邻帧相似内容）和 spatial redundancy（单帧内冗余 token），但 one-shot 方法仅基于单次 attention score 剪枝，未利用帧间时序相关性进行合并压缩，无法在保持性能的同时最大化压缩率。
  3. **永久性信息丢失**：FastV 和 PruMerge 一旦剪枝就永久丢弃 token 的 KV cache，如果后续 decoding 需要重新关注这些 token，模型无法恢复。这是 "one-shot" 的根本局限。
  4. **Prefilling 阶段注意力不可靠**：FastV 在 prefilling 阶段基于 prompt token 对 visual token 的注意力来评估重要性，但 prefilling 阶段的注意力分布与 decoding 阶段各步的实际需求不一致，导致 "错误地剪除 LLM 在 decoding 中需要的 token"（原文：FastV incorrectly prunes important tokens overlooked by the LLM during the prefilling phase）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：DyCoke 首次引入 **training-free 的动态 token 压缩** 策略，通过两阶段设计系统性地解决上述缺陷：(1) Token Temporal Merging (TTM) —— 利用帧间 cosine similarity 合并相似 token，解决时序冗余；(2) KV Cache Dynamic Pruning —— 在 decoding 阶段动态评估并调整 KV cache 中的 visual token 集合，配合 DP cache（Dynamic Pruning cache）实现 token 的可召回机制。

  对比 baseline 的全栈执行例子（DyCoke + LLaVA-OV-7B, K=0.7, L=3, P=0.7, 32 frames, MVBench）：
  - 算法层：
    1. 视觉编码器产出 6272 visual tokens → TTM 阶段：滑动窗口 (window=4) 采样 → O/E 组余弦相似度计算 → E 组高相似 token 剪枝 + O 组内首帧全保留/其余剪枝 → visual tokens 压缩 70%（K=0.7, 保留约 1882 tokens）→ concat 文本 token 送入 LLM prefilling → 计算 QKV 并填充 KV cache。
    2. Decoding 第 t=1 步：在第 L=3 层计算预测 token 对 visual tokens 的 cross-attention A^(3) → Softmax(QK^T/√D) → 取 top P=70% 高 attention token 保留在 KV cache → 剩余 30% 移入 DP cache（~565 visual tokens 在 KV cache, ~1317 在 DP cache）。
    3. Decoding 后续步：监控相邻迭代 attention 分布的 cosine similarity。当 similarity 下降（模型关注点变化），重新在第 3 层计算 cross-attention → 从 DP cache 将注意力回升的 token 动态加回 KV cache → 同步将 KV cache 中注意力下降的 token 移回 DP cache → KV cache 和 DP cache 动态双向流动。每帧平均约 15 tokens 参与注意力矩阵计算。
    4. FLOPs 从 41.4T 降至 17.9T（约 43%），实测 latency 1.49s/example (1.54× speedup), GPU memory 从 34GB 降至 24GB。
  - 系统框架层：LMMs-Eval 评估框架，PyTorch，LLaVA-NeXT（兼容 Flash Attention，仅在第 L 层额外计算 cross-attention）
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明（兼容 Flash Attention，无自定义 kernel）
  - 硬件架构层：NVIDIA A6000 (48GB) / RTX 4090 (24GB) / A100 (80GB)

  方法 vs Baseline 缺陷的对应映射：
  1. **动态剪枝解决 "时间注意力漂移"**：通过在 decoding 每一步动态重新评估 token 重要性（而非 prefilling 一次评估），模型可以随时调整保留的 token 集合。论文消融实验证明：去除 DP（动态剪枝→one-shot 剪枝）后 VideoDC 性能显著下降（Tab. 5, green），验证了动态机制的有效性。
  2. **TTM 解决 "时空冗余未充分利用"**：TTM 专门利用视频帧间的 temporal coherence（帧间余弦相似度）在 prefilling 阶段合并冗余 token，而非仅靠 attention score 剪枝。随机替换 TTM 的相似度选择为随机选择后性能剧烈下降（Tab. 5, random pruning），验证了基于时序相关性合并的正确性。
  3. **DP Cache 解决 "永久性信息丢失"**：DP cache 存储被剪枝的 token，后续 iteration 可重新召回。双向流动机制（KV cache ↔ DP cache）保证了 token 信息不会永久丢失。这直接对比 baseline 的 "一次性丢弃" 策略。
  4. **TTM 预处理 + Dynamic 评估解决 "Prefilling attention 不可靠"**：TTM 不依赖 attention score（而是语义相似度），避免了 prefilling 阶段注意力不可靠的问题；后续 Dynamic Pruning 在 decoding 真实 attention 分布上评估，比 prefilling 更准确。
  5. **Cost-Effectiveness（同计算预算性能提升）**：压缩后相同 FLOPs 预算下可处理更多帧 → VideoMME 上 32 frames (DyCoke, 17.91T FLOPs) 性能 58.3% vs Full 16 frames (18.99T FLOPs) 56.2%（Tab. 6），验证了压缩的实际效益。

- baseline方法是什么？
  Baseline 是标准的 uniform frame sampling（UNI）策略：从长视频中以固定间隔均匀采样 N 帧，每帧用 56 visual tokens 表示，拼接后送入 LMM（Qwen2.5-VL-7B/32B 或 Qwen3-VL-8B）进行视频问答推理。这种方法是完全的 query-agnostic（查询无关）策略，对所有类型的查询一视同仁。

  Baseline（uniform sampling + Qwen2.5-VL-7B, N=32 frames, MLVU）全栈执行例子：
  - 算法层：输入一段 12 分钟的叙事视频 + 问题 "What color is the man's bike at 3:15?" → 以固定间隔 = (12×60×fps)/32 均匀采样 32 帧 → 每帧经 vision encoder 编码为 56 tokens → 32×56=1792 visual tokens + Q text tokens → LLM 自回归生成答案。过程中，与"3:15 时刻骑车人"无关的帧（如开头风景、结尾字幕）同样占据 token 配额，关键帧可能恰好落在采样间隔之间被跳过 → 准确率随帧数增加反而下降（Figure 2 和 Figure 3 证明：localized queries 在 N=32 时比 N=8 性能更低）。
  - 系统框架层：LMMs-Eval 评估框架 + vLLM 推理加速
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：8 × NVIDIA A100 GPU

  Baseline 的缺陷：
  1. **噪声累积（Noise Accumulation）**：均匀采样给 localized query 注入了大量无关帧作为噪声。Figure 3 和 Figure 5 证明：随着帧数 N 增大，localized query 性能显著下降（Qwen2.5-VL-7B 在 MLVU 上 N=8→32, LQ accuracy 明显下降），而 global query 性能保持稳定。根本原因是 LQ 只需少数特定时间段的帧即可回答，多余帧不仅无益反而干扰模型注意力。
  2. **信息遗漏（Information Missing）**：固定间隔采样可能恰好跳过关键事件所在的帧。长视频中信息并非均匀分布，固定采样无法保证覆盖到查询相关的短暂时段。
  3. **效率低下（Efficiency Gap）**：对所有查询（包括 global query）都使用 query-aware selection 是一种浪费。现有方法（AKS, Q-Frame, BOLT）对每个查询都执行耗时的帧搜索，但论文证明对于 global query，uniform sampling 已足够（Figure 5 右侧两图），高级选择方法在 GQ 上收益递减甚至无益。
  4. **不可扩展（Scalability Limitation）**：AKS 和 Q-Frame 在 frame 数增大（>128）时性能退化，甚至会低于 uniform sampling（Table 1 中红框标记的退化结果），因为它们的候选帧采样池（如 Q-Frame 限于 128 帧）或搜索策略在长视频中不能很好地扩展。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：DIG（DIvide, then Ground）是一个 training-free 的帧选择框架，通过"先分类再定向"的策略解决上述缺陷：(1) Query Identification —— 用 LLM（Qwen3-Next-80B-A3B）以 CoT 方式自动将查询分为 global 或 localized；(2) 对 global query 直接用高效 uniform sampling；(3) 对 localized query 启动专门 pipeline：CAFS（基于 DINOv2 语义内容的自适应代表性帧选择）→ LMM Reward Assignment（用 LMM 自身而非 CLIPScore 评估帧与查询的相关性，含二维评分：直接有用性 + 相邻帧补充信息潜力）→ Video Refinement（迭代式 reward 阈值化 + 窗口合并，构建仅含查询相关段的 refined video）→ uniform sampling。

  对比 baseline 的全栈执行例子（DIG + Qwen2.5-VL-7B, N=32 frames, 同一视频 + 问题 "What color is the man's bike at 3:15?"）：
  - 算法层：
    1. Query Identification: Qwen3-Next-80B-A3B 分析问题 → "man's bike", "at 3:15" 是具体 referents → isGlobal=false (localized)
    2. CAFS: video 2 fps 采样 → DINOv2 逐帧提取 768-d feature → 计算余弦距离序列 → 检测 prominence>0.1 的峰值 → 选相邻峰值间中间帧为 r-frames（如 47.9 个 r-frames 代表 10 分钟视频）
    3. Reward Assignment: 对每个 r-frame, Qwen2.5-VL-32B 执行 {"description": "...", "reward": 85} — 帧中有自行车和人物（直接有用性 70）+ 暗示相邻帧有其他角度拍摄（补充信息 15）= 总分 85
    4. Video Refinement: 迭代 thresholding（R̄=45 → 更新 rewards → 下一轮 R̄=28 → 稳定 → positive r-frames = 5 个）→ wlen=2 窗口合并 [K_{j-2}, K_{j+3}] → refined video 仅含"3:15 附近骑自行车"相关段
    5. Uniform sample 32 frames from refined video → LMM 推理 → 答案 "Red"
    MLVU accuracy: 70.69%（DIG）vs 61.91%（UNI 32 frames）= +8.78%
  - 系统框架层：LMMs-Eval 评估框架 + vLLM（query identification 和 reward assignment 加速）
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：8 × NVIDIA A100 GPU

  方法如何对应解决 Baseline 缺陷：
  1. **噪声累积** → Query Identification 区分 GQ/LQ + CAFS 只对 LQ 启动 pipeline，且在 refined video 中排除了无关段。Figure 5 左侧三图证明：pipeline 在 LQ 上持续大幅超越 uniform sampling。
  2. **信息遗漏** → CAFS 用内容语义变化（DINOv2 距离峰值）自适应选取 r-frames 而非固定间隔，保证覆盖所有语义场景。Figure 6 证明 CAFS 在长视频（>10min）上的 GIC/LoC 均优于 UNI 和 FPS。
  3. **效率低下** → Query Identification 使全局查询走 uniform sampling 快速路径，节省约 13-20% 的总选择时间（Table 11: VideoMME 节省 19.9%，MLVU 节省 13.3%），同时保持性能（Figure 5 右侧两图：GQ 上 uniform ≈ pipeline）。
  4. **不可扩展** → CAFS 自适应选择的 r-frames 数量随视频内容密度而非长度线性增长（Figure 10 证明），且 LMM-based reward 比 CLIPScore 更可靠（Table 2 证明 LMM reward 在所有 frame 数上一致优于 CLIPScore），使 DIG 在 256 甚至 768 frames 仍保持增益（Table 5: 768 frames 时 MLVU +4.7%, LVB +3.7% vs UNI）。

- baseline方法是什么？
  Baseline 方法是 naive training-free extension of LLaVA-NeXT 到视频领域（Zhang et al., 2024），采用两种静态压缩策略：(1) Uniform Frame Sampling —— 从视频中均匀采样固定数量帧，不考虑帧内容的信息密度差异；(2) Spatial Average Pooling —— 对每帧的 visual token 做空间平均池化压缩，对所有空间位置一视同仁。这种方式将图像预训练的 VLM 直接用于视频输入，不做任何额外的训练或自适应处理。

  Baseline（LLaVA-NeXT 7B, naive training-free video extension, 10 frames, EgoSchema）全栈执行例子：
  - 算法层：输入一段长视频 + 问题 "What did the person do after entering the kitchen?" → 均匀采样 10 帧（固定间隔）→ CLIP 视觉编码器逐帧编码 → 每帧产生 ~576 个 visual tokens → Spatial Average Pooling 压缩 → 拼接为 10 × compressed_tokens 的序列 → 与问题 text tokens 拼接 → Vicuna-7B LLM 前向推理 → 生成答案。整个过程对关键动作帧（如"拿杯子"）和冗余静态帧（如"站立等待"）完全等同处理，关键视觉细节在平均池化中被模糊。
  - 系统框架层：基于 HuggingFace Transformers 的自定义推理代码，论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：单卡 NVIDIA RTX A6000 GPU

  Baseline 的缺陷：
  1. **感知瓶颈（Perception Bottleneck）**：静态压缩策略（uniform frame sampling + spatial average pooling）对所有内容等同处理，丢弃了在时间和空间维度上不均匀分布的关键信息。在 EgoSchema 5-frame 实验（Figure 2a）中，uniform sampling + spatial pooling 比无压缩 baseline 精度显著下降，证明静态压缩损失了关键视觉线索。
  2. **Token 过载（Token Overload）**：即使经过压缩，视频输入的 visual token 数量仍远超静态图像，超过了图像预训练 VLM 的处理容量。在 EgoSchema 10-frame 实验中（Figure 2b），baseline 在增加 token 数时性能先升后饱和（plateau），证明模型无法有效利用超出其容量的额外 token。
  3. **Temporal Perception Blindness**：均匀采样无法感知视频内容的时间变化密度——可能在信息密集段采样不足（丢失关键动作），在冗余段采样过多（浪费 token 配额）。
  4. **Spatial Redundancy Unaddressed**：平均池化等静态空间压缩无法区分高信息量 token（如物体边界、人脸）和低信息量 token（如纯色背景），导致空间维度信息损失。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：D-CoDe 通过两个 training-free 组件解决上述问题：
  (1) **Dynamic Compression（内容感知动态压缩）**：时间维度——在均匀采样基础上，从剩余帧中迭代选择与已选帧语义最不相似的 supplementary frame（基于 CLIP global feature 余弦相似度），确保保留信息丰富的关键片段；空间维度——对每帧 token 按 ℓ2 norm 计算 salience，保留高激活 token（top-β），再通过余弦相似度（τ）贪婪聚类合并冗余 token（anchor + cluster 取平均），在减少冗余的同时保留语义信息。
  (2) **Question Decomposition（问题分解）**：用 GPT-3.5 将复杂问题分解为聚焦于视频不同方面的子问题（如角色位置、动作、交互、场景转换），每个子问题独立用压缩后 visual tokens 推理得到子答案，然后将子答案拼接作为辅助信息与原始问题一起送入 LLM 生成最终答案，引导模型关注视频的不同方面，实现对大量 visual tokens 的全面理解。

  对比 baseline 的全栈执行例子（D-CoDe + LLaVA-NeXT 7B, 同一视频 + 同一问题）：
  - 算法层：输入同一段长视频 + 问题 "What did the person do after entering the kitchen?" → Dynamic Temporal Selection: 均匀采样 12 帧（α=0.85, N=15）→ 从剩余帧中选 3 帧与已选帧语义最不相似的帧（基于 CLIP global feature cosine similarity）→ 共 15 帧 → 每帧: ℓ2 norm salience 计算 → 保留 top 62.5% 高激活 token → 余弦相似度 >= 0.9 的 token 合并为代表 token（mean pooling）→ 拼接压缩后 visual tokens F_final → Question Decomposition: GPT-3.5 生成子问题 ["Where is the person at the start?", "What actions does the person perform?", "What objects does the person interact with?"] → 每个子问题独立用 F_final 推理 → 子答案拼接 + 原始问题 + F_final → LLM 生成最终答案。EgoSchema accuracy 从 44.8%（baseline）提升至 58.0%（+13.2%）。
  - 系统框架层：基于 HuggingFace Transformers + OpenAI API (GPT-3.5-turbo-0125)，论文未说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：单卡 NVIDIA RTX A6000 GPU

  解决对应关系：
  | Baseline 缺陷 | D-CoDe 解决方案 |
  |---|---|
  | 感知瓶颈：静态压缩丢弃关键信息 | Dynamic Compression: 时间维度 supplementary frame selection 补充语义多样帧，空间维度 salience-based pruning + similarity-based merging 保留关键视觉细节（EgoSchema +5.8% over baseline w/ uniform+pooling） |
  | Token 过载：模型无法有效利用超量 token | Question Decomposition: 将复杂问题分解为聚焦子问题，引导模型关注视频的不同方面，使模型能有效利用更多 visual token（Figure 2b: decomposition 的 accuracy 随 token 数持续增长，无 plateau） |
  | 时间感知盲区：均匀采样忽略内容密度差异 | Supplementary frame selection 基于 CLIP semantic dissimilarity（Eq.3-4）：从信息丰富的候选中补充帧，确保不遗漏关键动作。NExT-QA +3.7% (65.4→68.3) |
  | 空间冗余未处理：平均池化无法区分 token 重要性 | ℓ2-norm salience pruning (Eq.5-6) + greedy cosine-similarity merging (Eq.7-9)：低激活 token 被剪枝，相似 token 被合并，保留语义信息同时减少冗余。Ablation: +dynamic spatial compression → +5.8% on EgoSchema |

  效率分析（Table 16, EgoSchema）：
  - Baseline: 44.8% accuracy, 3.927 s/sample
  - + Dynamic Compression: 51.8% (+7.0%), 6.115 s/sample (+55.7% latency)
  - + Question Decomposition: 58.0% (+6.2%), 37.395 s/sample (+511% latency)
  - 轻量变体：替换为更小 CLIP（35% params）→ 58.2%, 35.466 s/sample；限制子问题数 = 5 → 56.0%, 26.273 s/sample；限制子问题数 = 7 → 57.8%, 33.704 s/sample
