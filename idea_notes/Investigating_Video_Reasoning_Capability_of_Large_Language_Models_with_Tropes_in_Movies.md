## Investigating_Video_Reasoning_Capability_of_Large_Language_Models_with_Tropes_in_Movies

- baseline方法是什么？
  Baseline 方法是现有的 LLM-based Video Reasoning 三类范式：(1) **Captioner-Reasoner (LLoVi)** —— VLM (BLIP-2) 将视频帧 tokenize 为文本 caption，LLM 做多轮摘要压缩后做二分类判断；(2) **LMM Instruction Fine-tuning (SeViLA, LLaMA-VID)** —— 通过 projection layer 将视觉特征对齐到 LLM token space，SeViLA 用 localizer 选 16 帧（从 120 帧中），LLaMA-VID 将每帧压缩为 2 tokens 处理长视频；(3) **Visual Programming (ViperGPT)** —— LLM (GPT-4) 生成 Python 代码调用 VLM API 做逐步推理，但原始设计中缺乏角色识别工具（仅有通用 "person" 检测），且将 NExT-QA 式的简单 temporal localization 策略直接用于电影叙事，无法处理复杂 trope 定义。

  Baseline (ViperGPT + BLIP-2 VLM + GPT-4 code generator, 16 frames, TiM Mainset) 全栈执行例子：
  - 算法层：输入电影片段 (16 帧 via SeViLA keyframe selector) + trope query "Is the trope Big Bad present?" + trope definition → GPT-4 生成 Python 程序：for frame in frames: person = frame.find("person") → action = frame.simple_query("What is this person doing?") → 逐帧收集 actions → video_segment.select_answer(info, query) → 输出 {True/False}。整个过程仅对通用 "person" 对象做查询，无法将不同帧中的同一人物关联（无 face_identify），也无法将 trope 的抽象定义（如 "direct cause of all bad happenings"）分解为具体可检验的子问题。TiM Mainset(V+D) F1=20.98 (ViperGPT 16 frames)。
  - 系统框架层：ViperGPT Python 执行引擎 + BLIP-2/GPT-4 API 调用
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：论文未明确说明

  Baseline 的缺陷（在 TiM 数据集暴露的 Abstract Perception 和 Long-range Compositional Reasoning 挑战）：
  1. **Abstract Perception 缺失**：现有方法擅长感知 concrete 元素（动作、物体、属性），但在抽象概念（情感、动机、幽默、判断）上严重不足。ViperGPT 原始设计仅有通用 "person" 检测，无法识别电影中数十个不同角色的身份和交互——而这些角色交互是理解 "Big Bad" 等 trope 的核心。ViperGPT F1=24.39 (120 frames, V+D)，显著低于 FEVoRI 的 32.79。
  2. **Long-range Compositional Reasoning 不足**：电影可长达数小时、数千帧，trope 查询需分解为多个相互依赖的嵌套子查询（如判断 "Big Bad" → 需先识别负面事件 → 再归因到具体角色 → 再验证一致性）。ViperGPT 的简单 NExT-QA 式 prompt（直接 temporal localization + VLM query）无法处理这种多层嵌套推理。在 TiM 上，LLoVi 甚至低于 random baseline (F1=18.97 vs 19.54)，SeViLA/LMM-IF 倾向于盲目猜 "yes"（高 recall 但极低 precision），ViperGPT 虽有较好 precision 但 recall 严重不足。
  3. **Context 与 Query 未解耦**：ViperGPT 将电影上下文和 trope 查询混在一起推理，导致 LLM 在 program generation 时难以同时处理冗长的叙事细节和复杂的 trope 定义。GPT-4 与 GPT-3.5 在 program generation 上仅有 0.17 F1 差异，说明瓶颈不在 code generation 能力，而在 task decomposition 策略。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：(1) **FEVoRI** → 解决 Abstract Perception。在 ViperGPT API 中集成 DeepFace 人脸识别工具 (`face_identify`)，通过 ICL example 引导 LLM 生成包含角色跟踪的 Python 程序——逐帧将检测到的 "person" 映射到具体角色 ID，累积同一角色的行为和其因果效应，将抽象的 trope 概念（如 "Big Bad"）映射为可检验的角色行为链。(2) **ConQueR** → 解决 Long-range Compositional Reasoning。系统性地将 context 和 query 解耦并渐进分解：先将 trope 定义拆分为多个可检验维度 → 逐帧提取人物/动作/事件 → 将事件与 trope 各维度逐一匹配 → 最终汇聚判断。(3) **ABCD** → 量化验证。通过 VP 生成代码的 AST 节点数/边数和 VLM call/Token 数量化数据集的 Abstract Perception 和 Long-range Compositional Reasoning 水平。

  对比 baseline 的全栈执行例子（FEVoRI+ConQueR, ViperGPT + DeepFace + GPT-4 + BLIP-2, 120 frames, TiM Mainset V+D）：
  - 算法层：输入电影 120 帧 + trope query "Is the trope Big Bad present?" + trope definition → GPT-4 生成 Python 程序（含 FEVoRI+ConQueR ICL example 引导的推理模板）：
    1. **Character Identification**（FEVoRI）：逐帧 `frame.find("person")` 检测人物 → `video_segment.face_identify(character)` (DeepFace) 分配唯一角色 ID → 查询角色外观描述 `person.simple_query("Describe appearance in 10 words")` → 记录到 `character_infos[person_id]`
    2. **Action Tracking**（FEVoRI + ConQueR）：对每个角色查询 `person.simple_query("Describe action in the scene")` → 记录到 `character_infos[pid]["actions"]`
    3. **Negative Event Detection**（ConQueR解耦）：`frame.simple_query("Is there any negative event in the scene?", to_yesno=True)` → 如果是，`frame.simple_query("What's happening in the scene")` 提取 event 描述
    4. **Progressive Matching**（ConQueR核心）：对每个负面事件，遍历所有角色信息，逐一匹配：
       - `person_query = f"Is person '{character_description}' a potential cause of '{event}'?"` → 匹配人物
       - `action_query = f"Is action '{prev_action}' a potential cause of '{event}'?"` → 匹配动作
       若任一匹配，将该角色标记为 potential_cause
    5. **Global Aggregation**：`video_segment.select_answer(info, query, possible_answers)` 汇聚所有帧的角色-事件因果链，判断 trope 是否存在 → F1=39.64 (FEVoRI+ConQueR) vs 20.98 (ViperGPT baseline) = +18.66 F1
  - 系统框架层：ViperGPT Python 执行引擎 + BLIP-2/Gemini VLM API + DeepFace 人脸识别 + GPT-4 code generation
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：论文未明确说明

  解决对应关系：
  | Baseline 缺陷 | 论文方法 | 效果 |
  |---|---|---|
  | Abstract Perception 缺失：仅通用"person"检测，无法识别角色身份和交互 | FEVoRI: 集成 DeepFace face_identify + ICL example 引导角色跟踪，将抽象 trope 概念映射为可检验的角色行为链 | FEVoRI +8.5 F1 over ViperGPT (24.39→32.79)，CT 类别 +7.61 F1, RI 类别 +9.93 F1 |
  | Long-range Compositional Reasoning 不足：简单 temporal localization 无法处理多层嵌套推理 | ConQueR: context/query 解耦 + 渐进维度分解 + 逐帧角色-事件匹配，将复杂 trope 定义拆解为可独立验证的子问题 | ConQueR +6.9 F1 over FEVoRI (32.79→39.64)，recall +11.48 |
  | Context 与 Query 未解耦：LLM 难以同时处理冗长叙事和复杂 trope 定义 | ConQueR 渐进推理流水线：先提取角色信息 → 再检测事件 → 再逐维度匹配 → 最后汇聚，将复杂推理拆分为管道式小步推理 | ConQueR AST Nodes +18.6, AST Edges +27.1 vs baseline TiM (Table 4)，ABCD 定量证实推理复杂度提升 |
  | 全局性能仍远低于人类 (65 F1) | 即使 FEVoRI+ConQueR 仅达 40 F1 vs Human 65 F1，但揭示了未来方向：更强的 VLM (Gemini 4.5 F1 gain)、更高帧率 (everyshot +1.5 F1)、更复杂 program generation
  - Buffer: 不用 Buffer → 99.7% (Table 6b: w/o Buffer)，去掉 Buffer+Anchor → 99.7%，证明 Register 可单独支撑，但 Anchor+Buffer 精确保留细节
  - Buffer scheme: Cross(4)/Square(8)/Row(2) 差异微小 (Table 6c: all ~100%)，只要 buffer 覆盖足够即可
  - Attention pattern: Global mean attention 略优于 CLS token (Table 6a)，且 universal（不依赖 CLS token 存在与否）
