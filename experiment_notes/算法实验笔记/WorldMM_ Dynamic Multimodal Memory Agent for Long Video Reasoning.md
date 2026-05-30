## WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：**WorldMM**，一个多模态记忆代理框架，用于对小时到周级别的长视频进行推理。核心包含三个阶段和三类互补记忆：
  (1) **多模态记忆构建 (Multimodal Memory Construction)**：
     - **Episodic Memory（情节记忆）**：将长视频按多时间尺度 T = {t₀, t₁, ..., t_N}（如30s, 3min, 10min, 1h）分段，每段用Video LLM生成caption，再转化为(entity, action, entity)三元组，构建多尺度知识图谱 M_e = {G_{t₀}, ..., G_{t_N}}。
     - **Semantic Memory（语义记忆）**：用更大时间粒度t_s分段，提取语义三元组（关注概念性知识而非具体事件），通过Consolidate(G^k_{t_s}, T^{k+1}_{t_s})过程（embedding相似度匹配→LLM合并/去重/纠错）增量更新跨场景的长期关系、习惯图谱 M_s = G^M_{t_s}。
     - **Visual Memory（视觉记忆）**：两种构建策略——(i) 特征检索：将视频以t_v切片，用多模态编码器（VLM2Vec-V2）编码为特征向量 f_v^k，构成 M_v^f = {f_v^1, ..., f_v^L}；(ii) 时间戳检索：每帧配时间戳 M_v^I = {(t_i, I_i)}。
  (2) **自适应记忆检索 (Adaptive Memory Retrieval)**：
     - Retrieval Agent R 是核心调度模块，每轮 i 输入用户query q和历史检索记录 r_{<i}，输出 (m_i, q_i) 或 STOP：
       R(q, r_{<i}) = { (m^i, q^i) if insufficient and i≤N; STOP otherwise }
     - 记忆检索方式各异：Episodic用Personalized PageRank (PPR) + LLM跨尺度重排序（top-m）；Semantic用PPR边级评分（边得分=两端节点PPR之和）；Visual用余弦相似度检索或时间戳直接访问。
     - 最多N=5轮迭代，自适应停止。
  (3) **响应生成 (Response Generation)**：Retrieval Agent决定STOP后，将全部检索历史送入Response Agent生成最终答案。
  实验比较：
  - 主实验：WorldMM-GPT和WorldMM-8B vs. 四类baseline（Base video LLMs、Long video LLMs、RAG-based、Memory-based），在5个长视频QA benchmark上的accuracy对比。WorldMM-GPT平均69.5%，比最强baseline高8.4%。
  - 消融实验：不同记忆组合(E/V/E+S/E+V/E+S+V)的accuracy对比；单模组变体（固定时间尺度单图、embedding检索代替graph、去consolidation、仅feature/仅timestamp检索）。
  - 效率实验：端到端延迟 vs. accuracy 的 trade-off 对比。
  - 泛化实验：不同backbone（Gemini 3 Flash + Qwen3-VL-Emb / VLM2Vec-V2, GPT-5 + Qwen3-VL-Emb / VLM2Vec-V2）对比。
  - 时序检索实验：tIoU指标对比（WorldMM 10.09% vs. baselines 0.58-4.35% on EgoLifeQA）。
  - 多轮检索消融：最大检索步数1→5的性能提升（EgoLifeQA上最大提升9.3%）。

- 硬件平台是什么，配置是什么。
  论文未明确说明具体GPU型号和实例配置。Backbone推理使用GPT-5 API（闭源商业模型，远程调用）、Gemini 2.5 Pro API、以及本地部署的Qwen3-VL-8B-Instruct（8B参数）。记忆构建阶段使用GPT-5-mini提取captions和三元组。视觉编码使用VLM2Vec-V2。语音转录使用Distil-Whisper large-v3.5。框架本质是API-heavy pipeline，对本地硬件要求主要来自开源模型推理（Qwen3-VL-8B约需16GB+ GPU显存）和embedding编码。

- 模型是什么。数据集和bench分别是什么。
  模型：
  - 骨干Video LLM：GPT-5（闭源）、Qwen3-VL-8B-Instruct（开源，8B）、Gemini 3 Flash（用于backbone泛化实验）
  - 记忆构建LLM：GPT-5-mini
  - 多模态编码器：VLM2Vec-V2、Qwen3-VL-Embedding-2B（用于消融）
  - 语音转录：Distil-Whisper large-v3.5
  数据集和Benchmark：
  - EgoLifeQA：500题，第一人称周级别视频（44.3h），5类：EntityLog, EventRecall, HabitInsight, RelationMap, TaskMaster
  - Ego-R1 Bench：300题，同一周级别视频，侧重多步推理
  - HippoVlog：1,000题，vlog风格（0.45h），4类：Auditory, Visual, Auditory+Visual, Summarization
  - LVBench：1,534题，通用长视频（1.14h），3类粒度：Short(<30s), Medium(30s-5min), Long(>5min)
  - Video-MME (long subset)：900题，通用长视频（0.69h），12类细粒度分类
  - 评估指标：Accuracy（多选题）、tIoU（时序交并比，用于检索质量评估）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源情况：论文提供项目页面 https://worldmm.github.io，但论文明文未确认代码是否已开源。HippoRAG（基于PPR检索）、EgoRAG等baseline有开源实现。论文在Appendix中提供了完整的prompt模板（Fig.9-17）。
  
  算法pipeline伪代码（核心检索循环）：
  ```
  # === 阶段1: 多模态记忆构建 (离线) ===
  输入: 长视频 V, 时间尺度集合 T={t0,...,tN}
  
  # 1a. 情节记忆 (Episodic Memory)
  for ti in T:
      将V按ti等长切分为段 S_i = {s_i^1, s_i^2, ...}
      for s_i^k in S_i:
          采样帧 + 语音转录(Whisper)
          cap_i^k = VideoLLM.generate_caption(frames, transcript)
          trip_i^k = LLM.extract_triplets(cap_i^k)  # → [(e1, action, e2), ...]
      G_ti = construct_KG(trip_i^k from all segments)
  M_e = {G_t0, ..., G_tN}
  
  # 1b. 语义记忆 (Semantic Memory)
  G_s = empty_KG()
  for segment k in coarse_segments(t_s):
      T_s^k = LLM.extract_semantic_triplets(cap)
      matched = embedding_similarity(G_s, T_s^k)  # >0.6阈值
      T_remove, T_update = LLM.consolidate(G_s, T_s^k, matched)
      G_s = (G_s \ T_remove) U T_update  # Consoldiate公式
  M_s = G_s
  
  # 1c. 视觉记忆 (Visual Memory)
  M_v_f = {VLM2Vec.encode(segment): segment for segment in split(V, t_v)}
  M_v_I = {(t, frame): frame for each frame at timestamp t}
  
  # === 阶段2: 自适应记忆检索 (在线) ===
  输入: query q
  history = []
  for i in 1..N:  # N=5
      decision = RetrievalAgent(q, history)  
      if decision == STOP: break
      
      m_type, query_i = decision
      
      if m_type == "episodic":
          candidates = []
          for G_ti in M_e:
              # PPR: Personalized PageRank, seed=query_i中的实体节点
              ppr_scores = PersonalizedPageRank(G_ti, seed=query_i)
              candidates += top_k_by_ppr(ppr_scores, k)
          # LLM跨尺度重排序: 从多尺度候选中选出top-m
          results = LLM.cross_scale_rerank(query, candidates)  # → top-m
      
      elif m_type == "semantic":
          ppr_scores = PersonalizedPageRank(G_s, seed=query_i)
          # 边得分 = 两端节点PPR分数之和
          edge_scores = {e: ppr(u)+ppr(v) for e=(u,v) in G_s}
          results = top_k_triplets(edge_scores, k=10)
      
      elif m_type == "visual":
          if is_timestamp_range(query_i):
              # 格式: "DAY X HH:MM:SS DAY Y HH:MM:SS"
              results = fetch_frames_from_M_v_I(query_i)
          else:
              f_query = VLM2Vec.encode(query_i)
              # 余弦相似度检索
              results = top_k_by_cosine_sim(f_query, M_v_f, k)
      
      history.append((m_type, query_i, results))
  
  # === 阶段3: 响应生成 ===
  answer = ResponseAgent(q, history)
  return answer
  ```
  
  关键张量/数据流与计算：
  - Caption生成: video_frames[N_frames, H, W, 3] + transcript[T_tokens] → VideoLLM → caption[S]
  - 三元组提取: caption[S] → LLM.extract → [(entity:str, action:str, entity:str)]
  - PPR检索（KG迭代传播）: 邻接矩阵 A[N×N], seed向量 s₀[N]（query实体对应位置=1）, 迭代 s = α·A^T·s + (1-α)·s₀, 到收敛后s[i]为节点i的PPR分数
  - 视觉特征编码: video_segment[T, H, W, 3] → VLM2Vec-V2 → f_v[D], D为embedding维度
  - 余弦检索: sim = (f_query·f_v^k) / (||f_query||·||f_v^k||)
  - 语义合并: Consolidate使用embedding cosine similarity >0.6匹配 + LLM决策T_remove/T_update
