## Personalized Streaming Video Understanding (PSVU)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PSVU（个性化流式视频理解）是 PEARL 论文首次提出并正式定义的新任务。与传统的个性化图像/视频理解不同，PSVU 要求模型：(1) 接收**连续流式视频输入**（而非预录制的完整视频）；(2) 在视频流中的任意时间戳**动态定义个性化概念**（概念在运行时由用户指令创建，非预设词库）；(3) **多轮交互**地回答关于这些已定义概念的实时查询和历史查询。任务定义为：流式视频 V = [X1, X2, ...] 作为连续场景序列，用户在时间戳 tc 通过 Concept-Definition QA 注册新概念，后续在时间戳 tq ≥ tc 发出查询 Q，模型需动态构造上下文 A = M(Csub, Vcontext, Q)，其中 Csub ⊆ C 是查询相关的概念子集，Vcontext 是必要的视觉上下文。PSVU 支持两种概念类型：Frame-level（静态实体，从单帧注册，如特定人物/物体）和 Video-level（动态动作，从连续片段注册，如个性化动作序列/手势）。查询分为三类：Concept-Definition QA（注册新概念，不计入评估）、Real-Time QA（查询概念在当下的状态，需纯粹基于当前场景）、Past-Time QA（查询概念的历史状态，必须检索历史证据片段才能回答）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
PSVU 任务的全栈推理流程：

```
# PSVU 任务主循环
ConceptMemory = {}       # {name: (visual_evidence, description)}
StreamingMemory = []     # [(clip_Xi, embedding_ei)]

for each arriving clip X^t at timestamp t:
    # Step 1: 场景检测与流式归档
    if scene_boundary_detected(X^t):
        e^t = multimodal_embed(X^t)      # Qwen3-VL-Embedding-2B
        StreamingMemory.append((X^t, e^t))

    # Step 2: 处理用户指令
    instruction = user_input_at(t)
    
    if instruction.type == "ConceptDefinition":
        if instruction.concept_type == "frame-level":
            evidence = X^t.last_frame
        else:  # video-level
            evidence = X^t
        description = vlm.describe_concept(evidence, instruction.name)
        # frame-level: "a young female with long black hair and oval face"
        # video-level: "the action of squatting down and then leaping forward"
        ConceptMemory[instruction.name] = (evidence, description)

    elif instruction.type in ("RealTimeQA", "PastTimeQA"):
        Q = instruction.question
        # Step 3: 概念检索
        mentioned = extract_concept_names(Q, ConceptMemory.keys())
        C_sub = {name: ConceptMemory[name] for name in mentioned}
        
        # Step 4: 查询重写 + 流式记忆检索
        Q_tilde = rewrite_query(Q, {n: desc for n, (_, desc) in C_sub.items()})
        e_Q = multimodal_embed(Q_tilde)
        similarities = [cosine_sim(e_Q, e_i) 
                        for (_, e_i) in StreamingMemory if clip_time <= t]
        top_K = top_k(similarities, K=4)
        V_context = top_K + adjacent_clips(top_K, N=1)
        
        # Step 5: VLM 生成回答
        answer = vlm.generate(concepts=C_sub, historical_clips=V_context,
                              current_clip=X^t, query=Q)
        return answer
```

具体数据流：视频以 1 FPS 采样 → PySceneDetect 基于 HSV 色彩空间像素变化（阈值 27.0）检测场景边界 → 分段为 min 1s / max 8s clips → 每个 clip 经 Qwen3-VL-Embedding-2B 编码为固定维度嵌入 → Concept-Definition QA 触发 VLM 生成概念描述 → Real-Time/Past-Time QA 触发检索 → 循环选项旋转评估（每个多选题 4 轮旋转，4/4 正确才算通过）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PSVU 通过 PEARL 框架实现，代码开源（https://github.com/Yuanhong-Zheng/PEARL, CC-BY 4.0）。核心组件：`clip_memory.py`、`concept_database.py`、`concept_desc.py`、`video_scene_splitter.py`、`video_qa_inference.py`、`eval.py`。多 GPU 部署通过 `server/` 目录启动 VLM server 和 embedding server，`scripts/` 协调并行推理。评估使用 PEARL-Bench（132 个视频、2173 条精细标注）。适用场景：定制化健身教练、个性化 AI 助手、实时监控中的个性化事件检测。

涉及论文标题：
- PEARL__Personalized_Streaming_Video_Understanding_Model
