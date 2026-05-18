## AdapterMgr: Imitation Learning-based LoRA Cache Management（基于模仿学习的LoRA缓存管理）

术语是什么？通过联网搜索让回答具体和精准。

AdapterMgr是TailorLLM中管理端侧LoRA adapter library的动态缓存替换算法。它将端侧LoRA存储管理类比为操作系统内存替换问题，在有限存储空间（capacity w=5个adapter slots）下，根据用户访问模式动态决定evict/load哪些adapter以最大化端侧hit rate。核心设计：(1) 双模态输入——用户历史访问序列（时序）+ 当前端侧LoRA cache状态（快照）；(2) Mamba SSM提取时序特征；(3) 以Belady最优替换策略（evict longest reuse distance的adapter）为训练目标；(4) 通过imitation learning + BCE loss使模型学习近最优替换决策。在实验中AdapterMgr的hit rate最接近Belady理论上界，在用户请求越动态时相对LRU优势越明显。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

AdapterMgr在TailorLLM online inference阶段的运作流程：
```
// 系统状态
端侧cache: L = [slot_0, slot_1, slot_2, slot_3, slot_4]  // w=5 slots
每个slot存储: {task_id, B_matrix, m_vector} 或 empty
滑动窗口: X = [x_{t-H+1}, ..., x_t]  // H=100 历史task访问序列

// 每个time step t的决策:
1: x_t = TaskClassifier(current_user_query)  // 当前请求的任务类别

// 2. Embedding (双模态独立embedding)
2: E(X) = W_x @ one_hot(X) + positional_encoding  // [H, d], d=128
3: E(L) = W_l @ L_encoding                          // [w, d]

// 3. Mamba时序特征提取
4: h_t = Mamba(E(X))  // recurrent: h_t = Ā_t·h_{t-1} + B̄_t·x_t

// 4. 双模态融合
5: F_fused = Concat(W_f @ E(L), h_t)  // projection到同一子空间后拼接
6: F_out = LayerNorm(F_fused)

// 5. 策略生成
7: π̂ = Softmax(MLP(F_out))  // [w] 概率向量，π̂_i = P(替换slot_i)

// 6. 动作执行
8: if x_t 已在cache中:
9:     hit → 不替换任何slot，直接使用
10: else:
11:     max_idx = argmax(π̂)  // 选择最优替换位置
12:     从云端下载 x_t 对应的 B + m (约11.56MB)
13:     替换 cache[max_idx] ← x_t
```

**训练阶段**：以Belady算法在每个时刻的最优决策（evict reuse distance最大的adapter）作为监督标签。训练数据中记录每次替换后的cache状态作为下一时刻的cache state输入。Loss: BCE(π̂, π_Belady)，其中π_Belady是one-hot vector指示Belady最优替换位置。使用连续序列长度b作为warm-up样本。

**Belady最优策略**：当cache满且有新内容miss时，evict reuse distance最大（即最久之后才会被再次访问）的adapter。Van Roy(2007)证明Belady是cache replacement问题的最优解，但需知道未来访问信息，实际中不可实现——AdapterMgr通过imitation learning逼近Belady而无需求解reuse distance。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

AdapterMgr实现为一个轻量级deep neural network（单层Mamba Block + 单层MLP + projection matrices），推理延迟低，可在每个请求到达时实时决策。区别于传统cache替换算法：(1) LRU——evict最近最少使用的，无法预测未来需求；(2) Parrot——使用LSTM+global attention，参数效率低于Mamba。AdapterMgr在MovieLens（162,541用户）和构造的周期性访问数据集（cycle 30/200）上评估，hit rate在所有场景下最接近Belady上界，尤其在用户请求更动态（cycle 200 vs cycle 30）时优势更明显。BCE loss相比传统imitation learning仅学习最优action，还额外学习区分正确/错误策略，增强泛化能力。滑动窗口H=100，embedding dim d=128，cache capacity w=5。论文未明确说明代码开源。

涉及论文标题：
- TailorLLM: Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation
