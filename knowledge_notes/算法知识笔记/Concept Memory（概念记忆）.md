## Concept Memory（概念记忆）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Concept Memory 是 PEARL 双粒度记忆系统中专门存储**用户自定义概念**的记忆模块。与 Streaming Memory（存储流式视频 clip 及嵌入）不同，Concept Memory 聚焦于"谁是什么"的个性化知识。当 Concept-Definition QA 触发时，创建包含三个组件的条目：(i) 概念名（如用户定义的 "Adaliz"），(ii) 关联的视觉证据（frame-level 取当前 clip 最后一帧；video-level 取整个 clip），(iii) VLM 生成的紧凑文本描述。frame-level 描述聚焦于永久/稳定特征（性别、面部特征、发型颜色/长度、体型、年龄外观），显式排除临时元素（服装、配饰、表情/姿势、背景）；video-level 描述聚焦于核心运动学（身体运动、动作序列、涉及部位），显式排除执行者身份/外观和背景。生成的描述文本与 clip 嵌入使用相同的 Qwen3-VL-Embedding-2B 特征空间，使查询重写后的检索语义对齐。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Concept Memory 的核心操作：

```
def register_concept(Q_def, X_tc, concept_type):
    concept_name = parse_name(Q_def)
    visual_evidence = X_tc.last_frame if concept_type == "frame-level" else X_tc
    
    # VLM 生成描述（in-context prompting, 无需训练）
    description = vlm.generate(
        prompt=CONCEPT_DESC_TEMPLATE.format(concept_name, Q_def),
        visual_input=visual_evidence
    )
    # Frame-level output: "a young female with long black hair and oval face"
    # Video-level output: "the action of squatting down and then leaping forward"
    
    ConceptMemory[concept_name] = {
        "visual_evidence": visual_evidence,
        "description": description,
        "timestamp": t_c
    }

# 检索：O(1) 按键查找
def retrieve_concepts(Q):
    mentioned = [name for name in ConceptMemory if name in Q]
    return {name: ConceptMemory[name] for name in mentioned}
```

消融证据：添加 Concept Memory 使 Real-Time 准确率从 15.84% 飙升至 51.41%（+35.57%，Table 4），证明显式概念存储对个性化 VLM 理解至关重要——没有概念描述，VLM 无法将用户自定义名称链接到视觉实体。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Concept Memory 通过 VLM 的 in-context prompting 实现，无需参数更新。PEARL 代码库中 `concept_database.py` 管理存储（内存字典），`concept_desc.py` 包含 frame-level 和 video-level 两套 Prompt 模板。Frame-level Prompt 引导 VLM 忽略服装/配饰/表情/背景，聚焦性别/面部特征/发型/体型；Video-level Prompt 引导 VLM 忽略执行者身份/背景，聚焦身体运动/动作序列/涉及部位。描述格式：1 句话约 10-20 词，第三人称，简单描述性英语。适用场景：任何需要在流式视频中动态定义和识别个性化实体/动作的 VLM 应用。

涉及论文标题：
- PEARL__Personalized_Streaming_Video_Understanding_Model
