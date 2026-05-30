## VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking

- baseline方法是什么？
  - 现有 video agents（如 VideoAgent [ECCV 2024], VideoTree [CVPR 2025], DrVideo [CVPR 2025], MR. Video, DVD [NeurIPS 2025]）主要依赖两种范式：(a) 密集预处理：以 0.2–2 FPS 将全视频转为详细文本描述或结构化记忆数据库，然后基于数据库检索或推理回答；(b) 贪婪搜索：逐段扫描视频并调用 CLIP 等检索相关帧。这些方法在长视频上计算开销随视频长度线性增长，且论文发现 LVBench 中 >80% 的问题仅需 <5% 的帧即可回答——证明 exhaustive parsing 极低效。
  - 全栈执行例子：
    - 算法pipeline：DVD agent 在推理前以 2 FPS 构建 multi-granular 视频数据库（8,074 帧 for LVBench），再在数据库上执行 tool-augmented search。p(τ | X, Q) 被替换为预先构建的静态数据库查询，trajectory 不随推理动态演化。
    - 系统框架：DrVideo 先用 0.2 FPS（493 帧 for Video-MME）将长视频转为长文档，再用 LLM 索引和检索。推理和感知分离：感知固定在前处理阶段，推理阶段只能从预建文档中检索。
    - 编译框架/kernel调度/硬件架构：论文未明确说明（这些层次与 video agent 的算法设计无关）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - VideoSeek 不依赖预构建视频数据库，而是通过 think-act-observe 循环利用视频逻辑流（temporal order, causality）主动 seek 答案关键帧。三个多粒度工具（overview→skim→focus）按需调用，trajectory 随时间动态累积观测，使得每步的工具选择基于"前序观测揭示了什么 + 还需要什么"。
  - 全栈执行例子（沿一个 query 的数据流穿过整个 agent pipeline）：
    - 算法pipeline：给定 query Q 和视频 X，初始化 τ=[I, Q]。Turn 1：θ_think(GPT-5) 读 τ，判断零先验，调用 overview 获得 16α 帧的全局 storyline（如 "1480s 两人在 mall 入口交谈，旁有 B1 标志"）。Turn 2：基于 overview 的推测，θ_think 规划调用 skim(1465-1510s)，获得粗粒度扫描（4α 帧）确认 "1465-1497s: 两人在 B1 标识旁说话；1503s: 镜头转向红色高楼"。Turn 3：θ_think 决定调用 focus(1499-1507s, 1 FPS) 精确读取高楼上的中文文字，获得 "祝全市人民新春快乐"。Turn 4：θ_think 判断证据充分，调用 answer 工具输出最终答案 D。全过程仅观看约 93 帧（vs DVD 的 8,074 帧）。
    - 系统框架：VideoSeek 是一个 model-agnostic 框架，thinking model θ_think 可替换（GPT-5/o4-mini/GPT-4.1），视觉解释也由 θ_think 完成。系统 prompt（Figure 7-8）定义了 Role/Environment/State/Workflow/Toolkit/Operational Rules 六部分，规定了工具调用的约束（每轮仅一个工具，overview 仅用于冷启动，skim→focus 渐进细化）。
    - 编译框架/kernel调度/硬件架构：论文未明确说明（VideoSeek 是纯 prompt-based agent，不涉及编译/算子/硬件层次）。
  - Baseline 缺陷 → 方法设计选择的直接映射：
    - 缺陷 1：baseline 的密集预处理在长视频上计算开销不可接受。→ 方法：用 logic flow 引导的主动 seeking，仅在需要时采样帧，LVBench 仅用 ~92 帧（vs DVD 8,074 帧）。
    - 缺陷 2：baseline 的预建数据库是静态的，无法根据推理中间结果动态调整证据收集策略。→ 方法：全 trajectory τ 作为上下文，每步基于完整历史决定下一步工具和行为。
    - 缺陷 3：baseline 缺乏 coarse-to-fine 的分层探索能力，要么全局过粗要么局部过细。→ 方法：三工具互补设计——overview 提供全局 map，skim 快速缩小搜索空间，focus 精确验证细节。
