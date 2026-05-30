## Streaming Memory with Scene Detection（基于场景检测的流式记忆）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Streaming Memory 是 PEARL Dual-grained Memory System 中负责存储和检索**流式视频历史观察**的模块。其核心功能：(1) **增量归档**——当视频流连续到达时，对每个检测到的场景边界处的新 clip 计算多模态嵌入，以 (clip, embedding) 对的形式追加存储；(2) **语义检索**——基于余弦相似度检索与查询语义最相关的历史 clips。场景检测使用 PySceneDetect 的 ContentDetector，基于 HSV 色彩空间中相邻帧像素变化的加权平均进行快速剪辑（cut）检测（阈值 27.0）。为确保 clips 包含充分时序上下文同时避免语义稀释，强制 min 1s / max 8s 的 clip 时长；超长场景按比例分割为多个子段。嵌入模型为 Qwen3-VL-Embedding-2B，以 1 FPS 采样后编码。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Streaming Memory 的系统运转流程：

```
┌──────────────────────────────────────────────────────────┐
│            Streaming Memory System Pipeline               │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Video Stream (continuous)                               │
│       │                                                  │
│       ▼                                                  │
│  ┌──────────────┐                                       │
│  │ PySceneDetect │  ContentDetector(threshold=27.0)      │
│  │ Scene Split   │  HSV colorspace pixel delta           │
│  └──────┬───────┘                                       │
│         │  scene boundaries detected                     │
│         ▼                                               │
│  ┌──────────────┐                                       │
│  │ Clip Segment │  enforce: min=1.0s, max=8.0s          │
│  │   & Buffer   │  split oversized scenes proportionally │
│  └──────┬───────┘                                       │
│         │  ordered clip sequence: [X1, X2, ..., Xn]    │
│         ▼                                               │
│  ┌──────────────────────┐                               │
│  │ Qwen3-VL-Embedding-2B│  1 FPS sampling               │
│  │   Encode X_i → e_i   │  output: embedding ∈ R^d      │
│  └──────┬───────────────┘                               │
│         │                                               │
│         ▼                                               │
│  ┌──────────────────────┐                               │
│  │  StreamingMemory DB  │  append (X_i, e_i)            │
│  │  [(X1,e1),(X2,e2),..]│  in-memory list storage        │
│  └──────────────────────┘                               │
│                                                          │
│  Query Time:                                             │
│  ┌──────────────────────┐                               │
│  │ cos(e_Q, e_i) ∀ i≤t_q│  cosine similarity scoring    │
│  │ Top-K=4 + expand N=1 │  temporal adjacency expansion  │
│  └──────────────────────┘                               │
└──────────────────────────────────────────────────────────┘
```

场景检测参数的工程含义：(a) threshold=27.0 平衡了过分割（太高则漏检场景变化）和欠分割（太低则产生过多微小片段）；(b) min_duration=1.0s 避免单个闪帧成为独立 clip；(c) max_duration=8.0s 保证每个 clip 语义聚焦——超长静态场景按比例分割使嵌入能捕捉更精细的时间子区间。邻接扩展（N=1）从每个 Top-K clip 向前后各取一个相邻 clip，捕获动作/事件的完整时序上下文（如 cooking 场景可能需要前后各一个 clip 才能完整覆盖）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PEARL 代码库中 `video_scene_splitter.py` 实现场景检测与分割（调用 PySceneDetect 0.6.x API: `scenedetect.detect(video_path, ContentDetector(threshold=27.0))`），`clip_memory.py` 实现 Streaming Memory 存储（Python list of tuples，每个 tuple 包含 clip 路径/帧数据和 numpy embedding）。评估 pipeline：`scripts/` 中的 shell 脚本协调 server（`server/` 启动 Qwen3-VL-8B VLM server + Qwen3-VL-Embedding-2B embedding server）→ 多 GPU 并行推理 → `eval.py` 聚合指标。对于生产部署，可替换为：(a) FAISS IVF/HNSW 索引加速 Top-K 检索（从 O(N) 降至 O(log N)）；(b) Redis/PostgreSQL 做 clip 元数据存储；(c) 流处理框架（如 Kafka + Flink）管理持续的视频流摄入与归档。适用场景：任何需要对长视频/流式视频进行历史语义检索的系统（如监控录像检索、体育赛事回放定位、个性化视频问答）。

涉及论文标题：
- PEARL__Personalized_Streaming_Video_Understanding_Model
