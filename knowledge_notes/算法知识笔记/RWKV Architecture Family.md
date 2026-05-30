## RWKV Architecture Family

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RWKV（Receptance Weighted Key Value）是基于 Linear Attention+RNN 的 LLM 架构家族。核心目标：Transformer 可并行训练 + RNN O(1) 推理。**原始论文"RWKV: Reinventing RNNs for the Transformer Era"（EMNLP 2023）**首次提出：Stacked residual blocks（Time Mixing + Channel Mixing Pre-LayerNorm），向量 state（head size=1），带分母归一化的 WKV 算子，Sigmoid receptance 门控，通道级可学习静态指数衰减 w。训练使用 time-parallel mode（类似 Transformer 并行矩阵乘法），推理使用 time-sequential mode（RNN 递归更新，O(d) 空间 + O(1) 时间）。训练 6 个规模（169M→14B）于 Pile 330B tokens，14B 为当时最大密集 RNN。关键设计：Small Init Embedding（U(±1e-4)+LayerNorm 加速收敛），Custom CUDA kernel 用于 WKV 串行扫描并行化，无位置编码（Token Shift 替代），无 bias 的线性层。演进：RWKV-4（vector state, 分母 WKV, Sigmoid r）→ Eagle/RWKV-5（matrix state, LayerNorm, SiLU gating）→ Finch/RWKV-6（ddlerp, data-dependent w_t, LoRA）→ RWKV-7（in-context learning params）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
RWKV Block: LayerNorm → Time Mixing(WKV: ddlerp→k^T·v decay+accum→receptance query→SiLU gate→output)→residual→LayerNorm→Channel Mixing(ddlerp→ReLU²(k')→σ(r')gate)→residual
State: 每层 2D(token shift history)+D²/h(WKV per head), 总 66DL
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
完全开源 Apache 2.0（GitHub+HuggingFace）。1.12T tokens 多语言训练。支持 NLP/多语言/code/长上下文/多模态（VisualRWKV/Music/Audio）。Finch-C2是GoldFinch论文中提出的Finch(RWKV-6)改进版：移除gate、LayerNorm across heads替代GroupNorm、key×(1-w)保持行归一化、数据依赖的第二Value替代静态bonus项。GoldFinch将Finch-C2作为前2/3层（线性pre-fill），后1/3层使用GOLD Transformer（full attention over compressed key cache）。

涉及论文标题：
- RWKV__Reinventing_RNNs_for_the_Transformer_Era
- Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression
- RWKV-X__A_Linear_Complexity_Hybrid_Language_Model
- VisualRWKV__Exploring_Recurrent_Neural_Networks_for_Visual_Language_Models

---
