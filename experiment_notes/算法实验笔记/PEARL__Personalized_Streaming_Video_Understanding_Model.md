## PEARL__Personalized_Streaming_Video_Understanding_Model

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：PEARL —— 一个**training-free、plug-and-play**的个性化流式视频理解框架，由两大核心组件构成：
  (1) **Dual-grained Memory System（双粒度记忆系统）**：
    - **Streaming Memory（流式记忆）**：对连续视频流使用 PySceneDetect 检测场景边界（基于 HSV 色彩空间像素变化的快速剪辑检测，阈值 27.0，最小片段 1s，最大片段 8s），分段为有序 clip 序列 V = [X1, X2, ...]。对每个新增 clip Xi，使用 Qwen3-VL-Embedding-2B 多模态嵌入模型计算嵌入 e^i = femb(Xi)，存储 (Xi, ei) 对用于后续检索。视频以 1 FPS 采样。
    - **Concept Memory（概念记忆）**：当用户在时间戳 tc 发出 Concept-Definition QA 时，从当前 clip X^tc 提取视觉证据（frame-level 取最后一帧，video-level 取整个 clip），用基础 VLM 生成紧凑的概念描述（frame-level 聚焦永久/稳定特征如性别、面部、发型、体型；video-level 聚焦核心运动学如身体运动、动作序列、涉及的身体部位），将概念名、视觉证据和文本描述三者结构化存储。
  (2) **Concept-aware Retrieval Algorithm（概念感知检索算法）**：
    - 查询重写（Query Rewriting）：识别用户查询 Q 中出现的概念名，从 Concept Memory 检索对应的概念描述，将 Q 中的概念名替换为描述文本得到重写查询 Q̃
    - 嵌入检索：计算 e^Q = femb(Q̃)，与 Streaming Memory 中所有 clip 嵌入 {ei}i≤tq 计算余弦相似度
    - Top-K 选择 + 邻接扩展：选择相似度最高的 K=4 个 clip，每个 clip 扩展其相邻 N=1 个 clip（frame-level, video-level 用 N=0）以捕获局部时序上下文
    - 最终将检索到的概念条目、历史 clip、当前 clip X^tq 和原始查询 Q 送入 VLM 生成响应
  PEARL 无需任何参数更新，无缝适配多种 VLM 架构（LLaVA-OV-7B、Qwen2-VL-7B、Qwen3-VL-8B）。

  实验比较：
  (a) **PEARL-Bench 主实验（Table 3）** —— Frame-level（Real-Time + Past-Time + Avg）和 Video-level（Real-Time）指标。对比 8 个方法：
    - Proprietary Offline: Gemini3-pro-preview (64 frames)
    - Open-source Offline: LLaVA-OV-7B, Qwen2-VL-7B, InternVL3.5-8B, Qwen3-VL-8B (all 64 frames)
    - Open-source Online: ReKV(LLaVA-OV-7B, 0.5fps), StreamForest-7B(1fps), TimeChat-Online-7B(1fps)
    - PEARL variants: LLaVA-OV-7B+PEARL, Qwen2-VL-7B+PEARL, Qwen3-VL-8B+PEARL (all 1fps)
    - Human Score 上界: 97.61% Real-Time, 96.45% Past-Time, 97.49% Video-level Real-Time
    - Text-only 下界 (Qwen3-VL-8B): 11.06% Real-Time, 17.45% Past-Time
    - PEARL 将 Qwen3-VL-8B 的平均帧级准确率从 28.77% 提升至 52.24% (+23.47%)，超过 Gemini3-pro-preview (48.19%)。
  (b) **消融实验（Table 4）** —— 在 Qwen3-VL-8B 上逐组件启用：Text-only (14.26%) → +Current Clip (18.07%) → +Concept Memory (38.42%, Real-Time 猛增 35.57%) → +Streaming Memory (47.96%, Past-Time 猛增 20.26%) → +Query Rewriting = Full PEARL (52.24%)。
  (c) **效率对比（Table 5）** —— 端到端推理延迟：
    - LLaVA-OV-7B: 670ms (64f, 29.48% Avg)
    - LLaVA-OV-7B+PEARL: 775ms (1fps, 38.03% Avg) —— 仅增加 105ms 延迟换取 8.55% 精度提升
    - Qwen3-VL-8B+PEARL: 2111ms (1fps, 52.24% Avg)
    - 延迟分解（Fig.5）：PEARL 核心模块（Concept Retrieval + Query Rewriting + Streaming Memory Retrieval）延迟极低且跨模型恒定，VLM 推理占主导。
  (d) **超参数分析（Fig.4）** —— Past-Time QA 的 Top-K 和邻接扩展 N：K=0 时无法访问历史证据性能极低；K≥3 后趋于饱和。N=1 与 N=2 差距很小。默认 K=4, N=1。
  (e) **模型规模实验（Table 7）** —— Qwen2-VL 系列 (2B, 7B) 和 Qwen3-VL 系列 (4B, 8B) 加/不加 PEARL。PEARL 在所有规模上稳定提升：Qwen3-VL-4B+18.00%, Qwen3-VL-8B+23.47%, Qwen2-VL-7B+9.36%, Qwen2-VL-2B+4.17%。离线模型增大规模无显著收益（范式错配），加 PEARL 后大规模模型优势才得以释放。

- 硬件平台是什么，配置是什么。
  所有实验在 **NVIDIA H200 GPU** 上进行。基础 VLM：LLaVA-OV-7B、Qwen2-VL-7B、Qwen3-VL-8B。嵌入模型：Qwen3-VL-Embedding-2B。场景检测：PySceneDetect（检测阈值 27.0，最小 1s/最大 8s clip）。视频流以 1 FPS 采样。评估策略：循环选项旋转（每个多选题评估 4 次旋转正确选项位置 A/B/C/D，4/4 正确才算正确），消除选项位置偏差。

- 模型是什么。数据集和bench分别是什么。
  模型：PEARL 是 training-free 框架，可适配三种 VLM 架构：
  - LLaVA-OV-7B + PEARL
  - Qwen2-VL-7B + PEARL
  - Qwen3-VL-8B + PEARL
  嵌入模型：Qwen3-VL-Embedding-2B（编码视觉描述和视频 clip 到统一特征空间，用于余弦相似度检索）。
  数据集与 benchmark：**PEARL-Bench**（论文自建，首个 PSVU benchmark）：
  - 总计 132 个视频（Frame-level 112 + Video-level 20），平均时长 1458 秒
  - 2173 条精细标注，均带精确时间戳
  - Frame-level 划分：Concept-Definition QA 418 + Real-Time QA 922 + Past-Time QA 394 = 1734
  - Video-level 划分：Concept-Definition QA 80 + Real-Time QA 359 = 439
  - 视频来源：动漫、电影、真人秀（frame-level）+ Mixamo 数字人合成（video-level，8 角色 × 20 动作 × 20 背景随机组合）
  - 概念名：从 U.S. SSA 数据库随机选取 10k 常用名替换原名，防止先验知识泄露
  - 质量控制：自动过滤（消融法检测 trivial 问题）+ 10 位研究者人工审查
  - Real-Time QA 包含 6 个子任务：Presence, Behavior, Appearance, Location, Relation, Action
  - Past-Time QA 包含 2 个子任务：Event-based, Time-based

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/Yuanhong-Zheng/PEARL，CC-BY 4.0 许可证。
  仓库包含：`clip_memory.py`（片段/记忆管理）、`concept_database.py`（概念存储）、`concept_desc.py`（概念描述生成）、`video_scene_splitter.py`（视频场景分割）、`video_qa_inference.py`（视频QA推理）、`eval.py`（评估指标聚合）、`server/`（VLM和嵌入服务器启动脚本，支持 Qwen3-VL-8B、Qwen3-VL-Embedding-2B、LLaVA-OneVision）、`scripts/`（多GPU评估pipeline shell脚本）、`third_party/`（qwen-vl-utils, Qwen3-VL-Embedding 等）。

  算法 pipeline 伪代码：

```
# === 初始化 ===
StreamingMemory = {}          # {clip_id: (clip_Xi, embedding_ei)}
ConceptMemory = {}            # {concept_name: (visual_evidence, description)}
embedding_model = Qwen3-VL-Embedding-2B
vlm = Qwen3-VL-8B (or LLaVA-OV-7B, Qwen2-VL-7B)

# === 流式视频处理循环 ===
for each arriving video clip X^t at timestamp t:
    # 场景检测与分段（PySceneDetect, HSV delta threshold=27.0）
    if scene_boundary_detected(X^t):
        e^t = embedding_model.encode(X^t)   # 多模态嵌入, 1fps采样
        StreamingMemory.append((X^t, e^t))

    # 解析用户指令
    if instruction is ConceptDefQA(concept_name, concept_type):
        # 概念注册
        if concept_type == "frame-level":
            visual_evidence = X^t.last_frame   # 取最后一帧
        else:  # video-level
            visual_evidence = X^t               # 取整个clip
        description = vlm.generate_concept_desc(visual_evidence, concept_name)
        # 生成描述：frame-level聚焦永久特征(性别/面部/发型/体型)
        #           video-level聚焦核心运动学(身体运动/动作序列/涉及部位)
        ConceptMemory[concept_name] = (visual_evidence, description)

    elif instruction is Query(Q):
        # === Concept-aware Retrieval ===
        # Step 1: Concept Retrieval —— 识别Q中的概念名并检索描述
        mentioned_concepts = extract_concept_names(Q, ConceptMemory.keys())
        C_sub = {c: ConceptMemory[c] for c in mentioned_concepts}
        replacement_rules = {c: desc for c, (_, desc) in C_sub.items()}

        # Step 2: Query Rewriting —— 将概念名替换为视觉描述文本
        Q_tilde = vlm.rewrite_query(Q, replacement_rules)
        # e.g., "What is Adaliz doing?" →
        #       "What is a young female with long black hair doing?"

        # Step 3: Streaming Memory Retrieval —— 余弦相似度匹配
        e_Q = embedding_model.encode(Q_tilde)  # shape: [d_embed]
        similarities = {i: cosine_sim(e_Q, e_i)
                        for i, (_, e_i) in StreamingMemory.items() if i <= t}
        top_K_clips = top_k(similarities, K=4)  # K=4 frame, 4 video

        # Step 4: Adjacent Expansion —— 扩展邻接clips捕获时序上下文
        V_context = top_K_clips ∪ {adjacent_clips(c, N=1) for c in top_K_clips}
        # N=1 for frame-level, N=0 for video-level

        # Step 5: VLM Response —— 组装上下文并推理
        response = vlm.generate(
            concepts=C_sub,
            historical_clips=V_context,
            current_clip=X^t,
            query=Q
        )
        return response
```

  张量计算流程（以 Qwen3-VL-8B+PEARL 为例）：
  - 视频 clip Xi → Qwen3-VL-Embedding-2B → embedding ei ∈ R^d_embed
  - 重写查询 Q̃ → embedding e^Q ∈ R^d_embed
  - cos(e^Q, ei) = e^Q · ei / (||e^Q|| · ||ei||)，排序取 Top-4
  - 检索到的历史 clips Vcontext + 当前 clip X^tq + 概念描述 Csub + 原始查询 Q → VLM tokenizer → [visual_tokens; concept_text_tokens; query_tokens] → VLM decoder → 生成答案 A

  关键设计要点：
  - 概念描述与 clip 嵌入在同一特征空间（Qwen3-VL-Embedding-2B），保证检索一致性
  - Query Rewriting 将个性化名称转换为嵌入模型可理解的描述性语义，是检索质量的关键（消融显示 +4.28% Avg）
  - Streaming Memory 持续的增量归档 + 概念级精确检索，区别于传统在线模型的固定大小状态压缩
