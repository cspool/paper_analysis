## Vgent: Graph-based Retrieval-Reasoning-Augmented Generation For Long Video Understanding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**Vgent**，一个训练无关（training-free）的 graph-based RAG 框架，用于长视频理解。包含四个核心阶段：
  (1) **Offline Video Graph Construction**：将视频以 1.0 FPS 采样，每 K=64 帧分割为一个 clip。对每个 clip 调用 LVLM 提取 JSON 格式的 entities（主体、物体、场景）、actions（交互/动作描述）和 scenes（地点/环境描述）。使用 BAAI/bge-large-en-v1.5 计算 entity 描述的 text embedding 相似度，以阈值 τ=0.7 合并跨 clip 的语义等价 entity。将每个 clip 作为图的 node，通过共享 entity 建立 edge，形成视频知识图谱 G=(V, E)，附带全局 entity 集合 U 及描述 t^u。
  (2) **Graph-based Retrieval**：从 query 中用 LVLM 提取 keywords K。对每个 keyword k 和 entity u，计算 BGE embedding 相似度 sim(k, t^u)，若 >θ=0.5 则将该 entity 关联的所有 nodes 纳入候选集。按 similarity rank 后取 Top-N=20 个 clips。
  (3) **Structured Reasoning**：LVLM 基于 query 和 keywords 生成结构化 subqueries Q（binary yes/no 或数值型），对 Top-N clips 逐一验证。仅保留有 subquery 正向匹配的 clip，最多 r=5 个。然后 LVLM 跨 refined clips 汇总信息。
  (4) **Multimodal Augmented Generation**：将 refined clips（视频帧 + 字幕）和 intermediate reasoning results 作为多模态上下文输入 LVLM 生成最终回答。

  实验比较：(1) **与 LVLM base models 对比**：在 7 种 open-source LVLM（InternVL2.5-2B, Qwen2.5-VL-3B/7B, Qwen2-VL-7B, LongVU-7B, LLaVA-Video-7B）上对比 MLVU、VideoMME、LongVideoBench 三个 benchmark；(2) **与 RAG methods 对比**：NaïveRAG（GoldFish 风格）、Video-RAG（CLIP keyframe + object detection + OCR）、以及 proprietary LLM-based methods（VideoAgent, LLoVi, DrVideo, VideoTree）；(3) **Ablation studies**：NaïveRAG vs GraphRAG vs GraphRAG+Structured Reasoning 的组件消融；confidence-based refinement 对比；retrieval 数量 N 和 r 的消融；retrieval embedding 类型（CLIP/BERT/BGE）、retrieval threshold τ 的影响；(4) **Inference time analysis**：per-minute video 的 offline/online 时间对比。

- 硬件平台是什么，配置是什么。
  **NVIDIA A100 80GB GPU**。所有实验在 A100 80G 上完成。推理时间统计（Table 5）：offline graph construction 20.13 sec/min-video，online retrieval+reasoning+generation 3.93 sec/min-video。VideoAgent 的 proprietary LLM 对比数据来自论文原文引用。

- 模型是什么。数据集和bench分别是什么。
  模型（Base LVLMs）：
  - **InternVL2.5-2B**：InternVL2.5 系列，2B 参数
  - **Qwen2.5-VL-3B**：Qwen2.5-VL，3B 参数
  - **Qwen2.5-VL-7B**：Qwen2.5-VL，7B 参数
  - **Qwen2-VL-7B**：Qwen2-VL，7B 参数
  - **LongVU-7B**：spatiotemporal adaptive compression for long video，7B 参数
  - **LLaVA-Video-7B**：LLaVA 视频版本，7B 参数
  - 另有 Qwen2-VL-2B 在附录 MLVU category-level 结果中评估

  Embedding 模型：**BAAI/bge-large-en-v1.5**（默认，用于 entity 合并和 keyword-entity 相似度计算）。对比实验也测试了 CLIP 和 BERT 作为 retrieval embedding。

  语音转文字：**openai/whisper-large**（用于 MLVU benchmark 无字幕视频的 spoken content 提取）。

  Benchmarks：
  - **MLVU**：多任务长视频理解 benchmark，视频长度 3 min ~ 2 hours，平均 12 min，含 7 类子任务：Count, Ego, Needle, Order, PlotQA, Anomaly, Topic
  - **VideoMME**：含 w/o subtitles 和 w/ subtitles 两个子集，按视频长度分 Short/Medium/Long 三档（11 sec ~ 1 hour）
  - **LongVideoBench (LVB)**：侧重需要跨帧长时间上下文推理的 referential reasoning 任务

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码仓库：https://xiaoqian-shen.github.io/Vgent（论文中给出的项目主页）。

  算法 pipeline 核心伪代码：
  ```
  # === Phase 1: Offline Video Graph Construction ===
  G = Graph()  # G = (V, E)
  U = set()   # global unique entity set
  for each clip V_i in split(video, fps=1.0, K=64):
      entities, actions, scenes = LVLM.extract_entities(clip=V_i, subtitle=C_i)
      # entities: JSON {entity_name: str, description: str}
      for each entity e_j in entities:
          t_j = BGE.encode(e_j.description)  # text embedding
          sim_scores = {u: cosine_sim(t_j, BGE.encode(u.description)) for u in U}
          u_star = argmax(sim_scores)
          if sim_scores[u_star] > 0.7:  # τ = 0.7
              merge(e_j, u_star)
              add_edges(v_i, {v for v in V if u_star in v.entities})
          else:
              U.add(e_j)
      V.add(v_i)  # node for clip i

  # === Phase 2: Graph-based Retrieval ===
  K = LVLM.extract_keywords(query)  # keywords from query
  R = set()
  for each keyword k in K:
      for each entity u in U:
          if cosine_sim(BGE.encode(k), BGE.encode(u.description)) > 0.5:
              R.update(get_nodes_with_entity(u))
  # Re-rank by avg similarity with query keywords across entities, descriptions, subtitles
  R_sorted = rank(R, query=K, fields=[entities, descriptions, subtitles])
  R_top = R_sorted[:20]  # Top-N=20

  # === Phase 3: Structured Reasoning ===
  Q_struct = LVLM.generate_subqueries(query, K)
  # Q_struct: list of {type: "binary"|"numeric", text: str}
  R_prime = []
  for each clip v_i in R_top:
      responses = [LVLM.answer(q, v_i) for q in Q_struct]
      if any(response > 0 for response in responses):
          R_prime.append(v_i)
  R_prime = R_prime[:5]  # max r=5
  reasoning_summary = LVLM.aggregate(R_prime, Q_struct)

  # === Phase 4: Multimodal Augmented Generation ===
  answer = LVLM.generate(query, context={
      "video_clips": R_prime,
      "reasoning": reasoning_summary
  })
  ```

  核心张量计算：entity 合并与检索均基于 BGE text embedding 的 cosine similarity。每个 entity description 经 BGE 编码为 1024-d 向量，entity 合并时对新 entity 与 U 中所有已有 entity 的 embedding 进行 argmax cosine similarity 匹配。检索时对 keyword 和 entity 的 BGE embedding 进行相同的 cosine similarity 计算。Video-RAG baseline 中对比的 CLIP-based 方法则计算 frame 的 CLIP visual feature 和 query text embedding 之间的 cosine similarity。
