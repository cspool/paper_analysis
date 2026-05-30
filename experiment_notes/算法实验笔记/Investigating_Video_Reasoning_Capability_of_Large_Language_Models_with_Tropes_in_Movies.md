## Investigating_Video_Reasoning_Capability_of_Large_Language_Models_with_Tropes_in_Movies

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：(1) **FEVoRI (Face-Enhanced Viper of Role Interactions)** —— 在 ViperGPT 的 Visual Programming API 中集成 DeepFace 人脸识别工具（`face_identify`），通过 ICL example 引导 LLM 生成的代码逐帧识别人物并跟踪角色交互，将抽象概念（如 "Big Bad" 反派角色）的推理映射为具体的人物识别、行为分析和负面事件归因的链式推理。通过角色感知（role awareness）增强对 Abstract Perception 挑战的处理能力。(2) **ConQueR (Context Query Reduction)** —— 系统性地将电影叙事上下文（context）和 trope 查询（query）解耦并渐进分解：先将 trope 定义分解为多个维度（如 "Big Bad" → evil characteristics + negative judgments + causation of terrible events），再对每一帧识别的人物、动作、事件与 trope 各维度逐一匹配对齐，提升 Long-range Compositional Reasoning 能力。(3) **ABCD (AST Based Code Dignosis)** —— 利用 VP 生成代码的 AST 量化数据集的 Abstract Perception（VLM Calls / VLM Tokens）和 Long-range Compositional Reasoning（AST Nodes / AST Edges）水平。

  实验比较：(a) 主实验 —— Fullset(V)和 Mainset(V+D)上 LLoVi(C-R)、SeViLA(LMM-IF)、LLaMA-VID(LMM-IF)、ViperGPT(VP)、FEVoRI、FEVoRI+ConQueR、Gemini 1.5 的 F1/Precision/Recall 及四类别 F1（CT/RI/ST/SL）；(b) FEVoRI 消融 —— modality(V vs V+D)、frames(120 vs everyshot vs 16)、VLM(BLIP-2 vs Gemini)、Coder(GPT-4 vs GPT-3.5)的 F1 变化；(c) ABCD 分析 —— TiM vs NExT-QA/GQA/OKVQA 的 AST 复杂度对比；(d) 与 Human Performance (65 F1, TiMoS) 的对比。

- 硬件平台是什么，配置是什么。
  论文未明确说明硬件平台和 GPU 配置。方法均为 training-free（FEVoRI、ConQueR 为 prompt 工程 + tool API 扩展，不需训练），推理调用 GPT-3.5/GPT-4 API（OpenAI）和 Gemini API。DeepFace 人脸识别在 CPU 上运行。SeViLA 微调使用五折交叉验证，硬件平台论文未明确说明。

- 模型是什么。数据集和bench分别是什么。
  模型：(1) Captioner-Reasoner: LLoVi (BLIP-2 VLM + LLM summarizer)；(2) LMM-IF: SeViLA (zero-shot + fine-tuned localizer)、LLaMA-VID (long-video-tuning, 240 frames)；(3) Visual Programming: ViperGPT (GPT-4 code generator + BLIP-2/Gemini VLM)；(4) Proposed: FEVoRI (ViperGPT + DeepFace face_identify tool)、ConQueR (progressive context-query decomposition)；(5) Upper bound: Gemini 1.5 (trillion-scale)。
  数据集：TiM (Tropes in Movies) —— 684 movies (MovieNet 数据源) × 95 tropes (TVTropes 数据源)，分 Fullset/VDeset/Mainset。Mainset: 50 movies, 平均 1699.6 frames, 1822.2 subtitle lines, 65k characters, 6.08 tropes per movie。Trope 分类: Character Traits(CT)、Role Interaction(RI)、Situation(ST)、Storyline(SL)。任务: 二分类 (trope present or not), metric: Micro F1。
  对比数据集: NExT-QA、GQA、OKVQA（用于 ABCD 分析）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://ander1119.github.io/TiM（数据集+代码）

  算法 pipeline 伪代码（FEVoRI + ConQueR 推理流程）：

  ```
  # === FEVoRI (Face-Enhanced Viper of Role Interactions) ===
  # 基于 ViperGPT 框架，扩展 face_identify API (DeepFace backend)
  # 输入: video (frames), annotation (shot boundaries), trope query, definition
  def execute_command_FEVoRI(video, annotation, possible_answers, query):
      # Trope: Big Bad
      # Definition: The character who is the direct cause of all 
      #             bad happenings in a story.
      video_segment = VideoSegment(video, annotation)
      info = {"character_actions": {}, "negative_impacts": {}}
      
      for i, frame in enumerate(video_segment.frame_iterator()):
          # Step 1: 逐帧识别人物 (face_identify利用DeepFace)
          for character in frame.find("person"):
              character_id = video_segment.face_identify(character)
              if character_id is None: continue
              
              # Step 2: 查询角色行为
              action_query = frame.simple_query(
                  "What is this person doing?")
              
              # Step 3: 判断行为是否有负面影响
              negative_query = f"Does the action '{action_query}' " \
                               "have a negative impact?"
              has_negative_impact = frame.llm_query(
                  negative_query, to_yesno=True)
              
              # Step 4: 累积角色行为与负面事件
              info["character_actions"][character_id].append(
                  action_query)
              if "yes" in has_negative_impact.lower():
                  info["negative_impacts"][character_id] += 1
      
      # Step 5: 汇聚信息后判断 trope 是否存在
      answer, reason = video_segment.select_answer(
          info, query, possible_answers)
      return answer, reason, info

  # === ConQueR (Context Query Reduction) ===
  # 渐进分解电影context和trope query
  def execute_command_ConQueR(video, annotation, 
                               possible_answers, query):
      # Trope: Big Bad
      # Definition: direct cause of all bad happenings
      video_segment = VideoSegment(video, annotation)
      info = {"happened_bad_events": {}, "character_infos": {}}
      
      for i, frame in enumerate(video_segment.frame_iterator()):
          for person in frame.find("person"):
              person_id = video_segment.face_identify(person)
              if person_id is None: continue
              
              # 渐进描述人物外观
              if person_id not in info["character_infos"]:
                  desc_query = "Describe appearance in 10 words"
                  character_desc = person.simple_query(desc_query)
                  info["character_infos"][person_id] = {
                      "description": character_desc,
                      "actions": {}}
              
              # 查询人物动作
              action = person.simple_query(
                  "Describe action in the scene")
              info["character_infos"][person_id]["actions"][
                  f"{i} frame"] = action
          
          # Step: 检查是否有负面事件
          check_neg = "Is there any negative event in the scene?"
          any_neg = frame.simple_query(check_neg, to_yesno=True)
          
          if "yes" in any_neg.lower():
              event = frame.simple_query(
                  "What's happening in the scene")
              info["happened_bad_events"][f"{i} frame"] = {
                  "event": event, "potential_cause": []}
              
              # Step: 逐一匹配角色是否为负面事件的潜在原因
              for pid, cinfos in info["character_infos"].items():
                  desc = cinfos["description"]
                  for prev_i in range(i, max(i-5, 0), -1):
                      prev_action = cinfos["actions"].get(
                          f"{prev_i} frame", None)
                      if prev_action is not None:
                          # 匹配人物描述与事件
                          pq = f"Is person '{desc}' a " \
                               f"potential cause of '{event}'?"
                          is_person = frame.simple_query(
                              pq, to_yesno=True)
                          # 匹配动作与事件
                          aq = f"Is action '{prev_action}' " \
                               f"a potential cause of '{event}'?"
                          is_action = frame.simple_query(
                              aq, to_yesno=True)
                          if "yes" in is_person.lower() or \
                             "yes" in is_action.lower():
                              info["happened_bad_events"][
                                  f"{i} frame"]["potential_cause"
                                  ].append(pid)
                          break
      
      # 汇聚信息后判断 trope
      answer, reason = video_segment.select_answer(
          info, query, possible_answers)
      return answer, reason, info
  ```

  算法 pipeline 全栈执行流程（FEVoRI+ConQueR on ViperGPT, GPT-4 code generator + BLIP-2/Gemini VLM）：
  - 算法层：Trope query + definition → GPT-4 生成 Python 程序（含 ICL example 引导的 FEVoRI/ConQueR 推理模式）→ 程序调用 ViperGPT API（frame.find("person") 检测人物 → frame.simple_query(prompt) 调用 VLM(BLIP-2/Gemini) 提取视觉语义 → video_segment.face_identify() 调用 DeepFace 人脸识别分配角色 ID → frame.llm_query(prompt, to_yesno=True) 调用 LLM 做 Yes/No 判断）→ 逐帧积累角色交互信息 → video_segment.select_answer(info, query) 汇聚全局推理 → 输出 {True/False} + reasoning。
  - 系统框架层：ViperGPT 框架（Python 代码执行引擎 + VLM/LLM API 集成），不涉及 Serving 框架修改。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。
  - 硬件架构层：论文未明确说明。
