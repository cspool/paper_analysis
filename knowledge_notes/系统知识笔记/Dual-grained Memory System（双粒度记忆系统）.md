## Dual-grained Memory System（双粒度记忆系统）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dual-grained Memory System 是 PEARL 框架的核心系统架构组件，将个性化流式视频理解中的记忆显式解耦为两个独立模块：(1) **Concept Memory（概念记忆）**——存储用户自定义概念的"概念中心知识"（概念名 + 视觉证据 + VLM 生成的紧凑描述），以概念名作为键进行精确检索；(2) **Streaming Memory（流式记忆）**——增量归档流式视频的"流式中心观察"（视频 clip + 多模态嵌入），以 embedding-based 语义检索访问历史视觉证据。这种解耦设计的动机源于 PSVU 任务的两个正交需求：(a) 通过名称精确知道"谁是什么"（概念级知识），(b) 通过语义检索访问"历史发生了什么"（流式级观察）。将两者合并为单一记忆将导致检索冲突——概念名无法用于 embedding 匹配，而 embedding 匹配无法按名称精确查找。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Dual-grained Memory System 的系统运转流程：

```
┌─────────────────────────────────────────────────────────┐
│              PEARL Dual-grained Memory System            │
├─────────────────────┬───────────────────────────────────┤
│   Concept Memory     │      Streaming Memory              │
│   (key-value store)  │      (vector store)                │
├─────────────────────┼───────────────────────────────────┤
│ Key: "Adaliz"       │ Index: embedding e_i               │
│ Value: {            │ Value: clip X_i                    │
│   evidence: frame,  │                                     │
│   description:      │  [e_0] [e_1] [e_2] ... [e_t]      │
│   "young female     │    ↓     ↓     ↓         ↓        │
│    w/ long black    │   X_0   X_1   X_2       X_t       │
│    hair & oval face"│                                     │
│ }                   │                                     │
├─────────────────────┼───────────────────────────────────┤
│ 操作: 精确键查找     │ 操作: 余弦相似度Top-K检索          │
│ O(1) by name        │ O(N) over all historical clips     │
│ 更新: Concept-Def   │ 更新: 每个新场景clip增量追加       │
│ QA触发注册          │ PySceneDetect触发分割             │
└─────────────────────┴───────────────────────────────────┘

处理流程：
1. 视频流 → PySceneDetect (HSV delta)
   → 分割为clips → 每个clip编码为embedding → 追加至StreamingMemory
2. 用户定义概念 → 提取视觉证据 → VLM生成描述 
   → 以概念名为键存入ConceptMemory
3. 用户查询 → 从ConceptMemory按键检索概念子集 
   → 重写查询(替换概念名为描述) → 编码重写查询
   → 与StreamingMemory中所有embedding计算余弦相似度
   → Top-K检索 → 邻接扩展 → 组装上下文 → VLM推理
```

关键设计决策：(a) 双粒度解耦使概念检索与视觉检索独立优化——Concept Memory 使用哈希表实现 O(1) 精确查找，Streaming Memory 使用向量索引实现 O(N) 语义检索；(b) 两个记忆使用相同的多模态嵌入空间（Qwen3-VL-Embedding-2B），保证概念描述文本嵌入与 clip 嵌入的语义对齐（通过 Query Rewriting 桥接）；(c) 增量追加设计避免全局重编码，每个新 clip 仅需单次嵌入计算即可归档。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PEARL 代码库中：(a) `concept_database.py` 管理 Concept Memory（Python dict/memory-based key-value store）；(b) `clip_memory.py` 管理 Streaming Memory（Python list of (clip, numpy_array_embedding) tuples）；(c) `video_scene_splitter.py` 调用 PySceneDetect (threshold=27.0, min_duration=1s, max_duration=8s) 进行场景检测；(d) 嵌入模型使用 Qwen3-VL-Embedding-2B（MRL 训练，支持 64-2048 维，Apache 2.0 许可证），通过 Sentence Transformers 或 HuggingFace Transformers 调用。对于生产环境，Streaming Memory 可以替换为 FAISS 或 Milvus 等向量数据库以支持更大规模。Concept Memory 可扩展为持久化数据库。适用场景：任何需要同时维护"实体知识"（谁是什么）和"时序视觉观察"（历史发生了什么）的流式视频理解系统。

涉及论文标题：
- PEARL__Personalized_Streaming_Video_Understanding_Model
