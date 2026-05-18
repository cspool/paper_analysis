## Bipartite Attention（二分注意力机制）

术语是什么？通过联网搜索让回答具体和精准。
Bipartite Attention是BAT提出的面向生成式推荐(GR)的新型注意力机制。其核心基于关键洞察：推荐prompt中user token和item token的语义是排列不变的(permutation-invariant)——交换user和item的顺序不影响上下文语义。Bipartite Attention自适应选择User-as-prefix或Item-as-prefix两种attention模式，使item KV cache可跨用户共享，打破传统User-as-prefix attention中item cache依赖前置user context而无法跨用户复用的限制。该机制通过修改attention mask和position encoding实现，确保不损失推荐精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Bipartite Attention两种模式的计算流程：

```
// 公共定义
给定: user tokens U (T_u tokens), N items I_1..I_N (T_i tokens total), instr tokens Instr
判别token: 序列最后一个token，投影到vocab logits计算各item relevance score

// ===== Mode 1: User-as-prefix Attention =====
// 输入序列: [U, I1, ..., I_N, Instr]
// KV Cache: user prefix U的K,V预计算并缓存
// 实时计算: 仅item和instruction token的Q,K,V

q_{I,Instr}, k_{I,Instr}, v_{I,Instr} = Proj(x_{I,Instr})  // 新计算
k_U, v_U = LoadFromCache()                                   // 缓存读取
attn = Attention(q_{I,Instr}, k_{I,Instr}∪k_U, v_{I,Instr}∪v_U)
// item间attention被mask屏蔽

position encoding: 
  - user tokens: positions 0..T_u-1
  - item tokens: positions 从T_u开始, 每个item共享相同起始位置
  - 保证item KV cache独立于后续user/instruction token

// ===== Mode 2: Item-as-prefix Attention =====
// 输入序列: [I1, ..., I_N, U, Instr]
// KV Cache: item prefix I的K,V预计算并缓存
// 实时计算: 仅user和instruction token的Q,K,V

q_{U,Instr}, k_{U,Instr}, v_{U,Instr} = Proj(x_{U,Instr})  // 新计算
k_I, v_I = LoadFromCache()                                    // 缓存读取
attn = Attention(q_{U,Instr}, k_{U,Instr}∪k_I, v_{U,Instr}∪v_I)

position encoding:
  - item tokens: positions 重置为0(或可选标记后)，所有item共享相同起始位置
  - user tokens: positions 从T_i开始
  - 保证item KV cache对任意user context独立可用

// Attention Mask (两者共用)
Mask[i][j] = 0 if i和j属于不同item  // 屏蔽跨item attention
Mask[i][j] = 0 if j > i              // causal mask
```

核心设计要点：(1) Item-as-prefix中item的position encoding从0开始，使item cache完全独立于user context；(2) 跨item attention被mask屏蔽，遵循推荐系统中item独立评估的原则；(3) Item-as-prefix不牺牲精度——实验中IP(Item-as-prefix)与UP(User-as-prefix)在Recall@k/MRR@k/NDCG@k上性能相当或更优。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Bipartite Attention在Bat系统中实现为vLLM的自定义attention module，使用FlashInfer高性能backend。模型训练时hardcode新的position encoding（从0重置item位置），无需额外训练开销；推理时根据hotness-aware scheduler决策动态选择使用哪一模式。对instruction-tuned模型（如Llama3-Instruct）可能出现IP精度下降，可选择支持修改position encoding的base model（如Qwen2）或应用Position-Independent Caching算法（如CacheBlend）选择性重新计算关键token以缓解精度损失。Bipartite Attention已在Qwen2-1.5B/7B和Llama3-1B上验证有效，在不同数据集上IP与UP的推荐质量相当。

涉及论文标题：
- BAT: Efficient Generative Recommender Serving with Bipartite Attention

---

