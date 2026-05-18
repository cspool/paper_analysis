## Task-Level End-Cloud Collaborative LLM Inference（任务级端云协同LLM推理）

术语是什么？通过联网搜索让回答具体和精准。

Task-Level End-Cloud Collaborative LLM Inference是TailorLLM提出的一种LLM推理服务范式，与token-level collaboration (speculative decoding/HSL)和model partitioning (Petals)形成对比。核心策略：根据任务（而非token）对用户请求进行分类分流——高频简单任务（占>70%请求，如文本翻译、摘要、情感分析）在端侧用经LoRA增强的SLM完成，复杂或稀有任务发送到云端LLM。两个关键观察支撑此设计：(1) 少数任务覆盖大多数用户请求（长尾分布）；(2) 用户请求在task-level比token-level更具可预测性，呈现与时间上下文相关的周期性模式（每50个问题中1-3个类别反复出现）。

相比token-level collaboration（HSL speculative decoding单次QA需15+次端云通信），task-level协作将端云通信频率降到最低（仅任务卸载时需要），显著降低了RTT对延迟的影响（RTT 20→200ms时TPOT仅1%退化 vs HSL 22%退化）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

TailorLLM task-level collaboration的系统流程：
```
// 离线阶段: 训练task-specific LoRA adapters并存入cloud library
1: for each task_i in {summarization, translation, QA, sentiment, ...}:
2:     RFLoRA_finetune(Llama3-1B, task_i_dataset) → B_i, m_i
3: cloud_library.store(task_i: {B_i, m_i, accuracy_i})

// 在线推理: task-level triage
4: user_query = "Please summarize this article..."
5: category = TaskClassifier(user_query)  
6: // Contriever semantic encoding → UMAP dim reduction → HDBSCAN clustering
7: // output: known_category_id OR new_category (-1)

8: if category == -1 or accuracy_table[category] < threshold:
9:     result = Llama3-70B_cloud_inference(user_query)  // 卸载到云
10: elif local_cache.has(category):
11:     load_adapter(local_cache[category].B, pre_stored_A)  // 端侧推理
12:     result = Llama3-1B_lora_inference(user_query)        // 0次云端通信
13: else:
14:     download_adapter_from_cloud(category) → local_cache  // 1次云端通信
15:     result = Llama3-1B_lora_inference(user_query)
```

关键指标：端侧hit rate（AdapterMgr决定）决定cloud cost节省程度。论文实验表明~70%请求端侧处理，cloud computing cost减少69.8%，end-to-end latency减少62%（vs cloud-only）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现需三个核心模块：(1) Task Classifier——无监督动态分类避免新类别误分类问题（TailorLLM用Contriever+HDBSCAN，>95% accuracy @15 categories）；(2) Task Allocator——查表决策：SLM精度是否达标+local cache是否命中→决定端侧/云侧执行；(3) LoRA Cache Manager——管理端侧有限存储空间（如capacity w=5 adapters）的动态替换策略，TailorLLM用AdapterMgr（imitation learning）实现近最优替换。硬件：cloud侧4×RTX 3090 + Llama3-70B，端侧Tesla T4 (10GB) + Llama3-1B + LoRA adapters。无线网络RTT 47ms。论文未明确说明系统代码开源。LoRA adapter切换开销<1ms，task classification 0.45-1.53ms，合计仅占推理延迟的2-7%。

涉及论文标题：
- TailorLLM: Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation
