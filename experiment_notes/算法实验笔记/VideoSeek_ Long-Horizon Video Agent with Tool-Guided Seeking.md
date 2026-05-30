## VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：VideoSeek 是一个 model-agnostic 的长时域视频 agent，遵循 ReAct 风格的 think-act-observe 循环。核心创新是用多粒度工具（overview/skim/focus）按视频逻辑流主动 seek 答案关键帧，而非密集解析全视频。
  - 实验比较：在四个 benchmark 上对比 standalone LMMs（GPT-4o, Gemini 1.5 Pro, Qwen2.5-VL-72B, Gemini 2.0 Flash, GPT-5）和 video agents（VideoAgent, VideoTree, DrVideo, VCA, MR. Video, DVD）。主要指标：accuracy（%）和 #Frames（处理的帧数）。消融研究分析 thinking model 替换（GPT-5 vs o4-mini vs GPT-4.1）和工具配置影响（逐一移除 overview/skim/focus）。附录有 α 参数（帧预算缩放因子）的 sensitivity 分析以及 intermediate reasoning 效果分析。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明推理所用 GPU 或硬件配置。论文指出 runtime 受网络延迟、GPU 类型、API 调度等因素影响，因此不将 runtime 作为主要效率指标。推理通过 API 调用 GPT-5/o4-mini/GPT-4.1 完成，视觉内容也由 GPT-5 解释。论文未说明本地部署配置。

- 模型是什么。数据集和bench分别是什么。
  - 模型：默认 thinking LLM 为 GPT-5（API），ablations 中使用 o4-mini 和 GPT-4.1。视觉解释也由 GPT-5 完成。对比的 LMMs：GPT-4o, GPT-5, Gemini 1.5 Pro, Gemini 2.0 Flash, Gemini 2.0 Flash Thinking, Gemini 2.5 Pro, Qwen2.5-VL-32B/72B, Video-R1, VideoChat-R1, SEED-Bench-R1。对比的 agents：VideoAgent, VideoTree, DrVideo, VCA, MR. Video, DVD。
  - Benchmarks：
    - LVBench：1,549 MC 题，103 个 hour-long 视频，评估 long-term memory 和 extended comprehension
    - Video-MME (long subset)：900 题，300 视频，平均 2,466 秒
    - LongVideoBench (long split)：564 题，188 视频，时长 900–3,600 秒
    - Video-Holmes：1,837 题，270 个 suspense 短片，7 个推理维度（SR, IMC, TCI, TA, MHR, PAR, CTI）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：github.com/jylins/videoseek（CVPR 2026）。核心 agent 依赖闭源 LLM（GPT-5 API），但开源的 toolkit 设计和 prompt 策略可直接复用。
  - 算法 pipeline（参照论文 Algorithm 1）：

    输入：用户 query Q，视频 X，系统指令 I，thinking model θ_think，工具集 T（overview, skim, focus, answer），最大轮次 N

    ```
    1. 初始化 trajectory τ ← ⟨I, Q⟩
    2. T ← T ∪ {answer}
    3. for t = 1 to N:
    4.   (z_t, a_t) ← θ_think(τ)          // 基于已有 trajectory 推理 + 工具规划
    5.   if a_t 仅含单个 answer:
    6.     Y ← parse_answer(a_t); break
    7.   o_t ← call_tools(a_t, X, T)      // 执行工具，获取观察
    8.   τ ← τ ∪ ⟨z_t, a_t, o_t⟩         // 附加到 trajectory
    9. if Y 为空:
    10.  Y ← θ_think(τ ∪ I_answer)        // 直接回答指令
    11. return Y
    ```

    工具说明：
    - overview: 从全视频均匀采样 16α 帧，生成每帧简要描述（~50 words），构建粗略 storyline
    - skim: 在选定区间 [t1, t2] 上采样 4α 帧，以 ~25 words/帧 描述并高亮与 query 相关的时间戳（~50 words）
    - focus: 在短片段 [t3, t4] 上以 1 FPS 密集采样，直接回答 query 或返回 "No relevant content found"
    - 工具设计约束：每轮仅调用一个工具；α 为帧预算缩放因子（LVBench α=4，其余 α=2）

    张量计算层面：VideoSeek 本身不涉及底层张量计算，它是 prompt-based agent 框架。每轮 think-act-observe 的输入为完整 trajectory τ 的文本表示，由 GPT-5 API 处理。视觉帧图像的视觉 token 化由 GPT-5 的 vision encoder 内部完成（论文未给出 encoder 细节）。每轮工具返回的 observation（帧描述文本 + 时间戳）追加到 τ 中。trajectory 的文本 token 数量随轮次线性增长，论文报告 LVBench 无字幕平均 49K tokens，有字幕 57K tokens。
